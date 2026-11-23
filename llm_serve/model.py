from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_serve.config import EngineConfig
from llm_serve.page_table import PhysicalSlot
from llm_serve.paged_attn import PagedKVCacheArena, paged_attention_forward


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def precompute_rope_tables(
    head_dim: int, max_positions: int = 4096, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
    inv_freq = 1.0 / (theta**exponents)
    pos = torch.arange(max_positions, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope_at_positions(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> torch.Tensor:
    """Applies RoPE to packed tokens `x` of shape (N_tokens, n_heads, head_dim) at `positions`."""
    cos = cos_table[positions].unsqueeze(1)  # (N_tokens, 1, head_dim // 2)
    sin = sin_table[positions].unsqueeze(1)

    xf = x.float()
    even = xf[..., 0::2]
    odd = xf[..., 1::2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).type_as(x)


class PagedAttentionLayer(nn.Module):
    def __init__(self, cfg: EngineConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dim = cfg.dim

        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        seq_token_counts: Sequence[int],
        seq_slots: Sequence[Sequence[PhysicalSlot]],
        block_tables: Sequence[Sequence[int]],
        context_lens: Sequence[int],
        arena: PagedKVCacheArena,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ) -> torch.Tensor:
        n_tok = h.shape[0]
        q = self.wq(h).view(n_tok, self.n_heads, self.head_dim)
        k = self.wk(h).view(n_tok, self.n_kv_heads, self.head_dim)
        v = self.wv(h).view(n_tok, self.n_kv_heads, self.head_dim)

        q = apply_rope_at_positions(q, positions, cos_table, sin_table)
        k = apply_rope_at_positions(k, positions, cos_table, sin_table)

        q_splits = list(torch.split(q, list(seq_token_counts), dim=0))
        k_splits = list(torch.split(k, list(seq_token_counts), dim=0))
        v_splits = list(torch.split(v, list(seq_token_counts), dim=0))

        for slots, ks, vs in zip(seq_slots, k_splits, v_splits):
            arena.write_kv_slots(self.layer_idx, slots, ks, vs)

        attn_outs = paged_attention_forward(
            q_splits,
            arena=arena,
            layer_idx=self.layer_idx,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        packed_out = torch.cat(attn_outs, dim=0).view(n_tok, self.dim)
        return self.wo(packed_out)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: EngineConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w_down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PagedDecoderBlock(nn.Module):
    def __init__(self, cfg: EngineConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = PagedAttentionLayer(cfg, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        seq_token_counts: Sequence[int],
        seq_slots: Sequence[Sequence[PhysicalSlot]],
        block_tables: Sequence[Sequence[int]],
        context_lens: Sequence[int],
        arena: PagedKVCacheArena,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ) -> torch.Tensor:
        h = h + self.attn(
            self.attn_norm(h),
            positions,
            seq_token_counts,
            seq_slots,
            block_tables,
            context_lens,
            arena,
            cos_table,
            sin_table,
        )
        return h + self.ffn(self.ffn_norm(h))


class PagedTransformer(nn.Module):
    """Llama-style GQA Transformer supporting ragged continuous-batch execution over PagedKVCacheArena."""

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [PagedDecoderBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope_tables(
            config.head_dim, max_positions=max(4096, config.total_kv_token_capacity * 2)
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward_ragged(
        self,
        flat_token_ids: torch.Tensor,
        flat_positions: torch.Tensor,
        seq_token_counts: Sequence[int],
        seq_slots: Sequence[Sequence[PhysicalSlot]],
        block_tables: Sequence[Sequence[int]],
        context_lens: Sequence[int],
        arena: PagedKVCacheArena,
    ) -> List[torch.Tensor]:
        """Runs a ragged batch and returns per-sequence logits of shape (q_len_i, vocab_size)."""
        h = self.tok_embeddings(flat_token_ids)
        for layer in self.layers:
            h = layer(
                h,
                flat_positions,
                seq_token_counts,
                seq_slots,
                block_tables,
                context_lens,
                arena,
                self.rope_cos,
                self.rope_sin,
            )
        h = self.norm(h)
        logits = self.lm_head(h)
        return list(torch.split(logits, list(seq_token_counts), dim=0))

    @torch.no_grad()
    def forward_standalone(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Contiguous reference forward pass (B, T) -> (B, T, vocab_size) for parity verification."""
        self.eval()
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        out_batches: List[torch.Tensor] = []

        num_blocks = (seq_len // self.config.block_size) + 2
        for b in range(bsz):
            arena = PagedKVCacheArena(
                self.config.n_layers,
                num_blocks=num_blocks,
                block_size=self.config.block_size,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                device=device,
            )
            slots = [
                PhysicalSlot(block_id=i // self.config.block_size, offset=i % self.config.block_size)
                for i in range(seq_len)
            ]
            btable = list(range(num_blocks))
            pos = torch.arange(seq_len, device=device, dtype=torch.long)
            seq_logits = self.forward_ragged(
                input_ids[b], pos, [seq_len], [slots], [btable], [seq_len], arena
            )[0]
            out_batches.append(seq_logits)
        return torch.stack(out_batches, dim=0)

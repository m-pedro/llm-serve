import math
from typing import List, Sequence, Tuple
import torch
import torch.nn.functional as F

from llm_serve.page_table import PhysicalSlot


class PagedKVCacheArena:
    """Global physical KV-block tensor pool across all Transformer layers."""

    def __init__(
        self,
        n_layers: int,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layers = n_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        shape = (n_layers, num_blocks, block_size, n_kv_heads, head_dim)
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros(shape, device=device, dtype=dtype)

    def copy_blocks(self, cow_pairs: Sequence[Tuple[int, int]]) -> None:
        """Executes Copy-on-Write physical block clones across all layers."""
        for src_id, dst_id in cow_pairs:
            self.k_pool[:, dst_id].copy_(self.k_pool[:, src_id])
            self.v_pool[:, dst_id].copy_(self.v_pool[:, src_id])

    def write_kv_slots(
        self,
        layer_idx: int,
        slots: Sequence[PhysicalSlot],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatters K/V tensors of shape (num_tokens, n_kv_heads, head_dim) into physical slots."""
        if not slots:
            return
        block_ids = torch.tensor([s.block_id for s in slots], device=k.device, dtype=torch.long)
        offsets = torch.tensor([s.offset for s in slots], device=k.device, dtype=torch.long)
        self.k_pool[layer_idx, block_ids, offsets] = k
        self.v_pool[layer_idx, block_ids, offsets] = v

    def gather_sequence_kv(
        self,
        layer_idx: int,
        block_table: Sequence[int],
        context_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gathers non-contiguous physical blocks into (context_len, n_kv_heads, head_dim)."""
        blk_idx = torch.tensor(block_table, device=self.k_pool.device, dtype=torch.long)
        k_blocks = self.k_pool[layer_idx, blk_idx].reshape(-1, self.n_kv_heads, self.head_dim)
        v_blocks = self.v_pool[layer_idx, blk_idx].reshape(-1, self.n_kv_heads, self.head_dim)
        return k_blocks[:context_len], v_blocks[:context_len]


def _repeat_kv_2d(x: torch.Tensor, kv_groups: int) -> torch.Tensor:
    """Expands (T, H_kv, D) -> (T, H_q, D) for Grouped-Query Attention."""
    if kv_groups == 1:
        return x
    seq_len, n_kv, head_dim = x.shape
    return (
        x[:, :, None, :]
        .expand(seq_len, n_kv, kv_groups, head_dim)
        .reshape(seq_len, n_kv * kv_groups, head_dim)
    )


def paged_attention_forward(
    q_list: Sequence[torch.Tensor],
    arena: PagedKVCacheArena,
    layer_idx: int,
    block_tables: Sequence[Sequence[int]],
    context_lens: Sequence[int],
) -> List[torch.Tensor]:
    """Computes exact causal GQA over non-contiguous paged KV blocks for heterogeneous queries.

    Args:
        q_list: Per-sequence query tensors, each of shape (q_len_i, n_heads, head_dim),
                covering either multi-token prefill/chunked-prefill or single-token decode (q_len_i=1).
        arena: Global PagedKVCacheArena with current step's K/V already written.
        layer_idx: Transformer layer index.
        block_tables: Physical block ID lists per sequence.
        context_lens: Total KV context length (cached + newly written) per sequence.

    Returns:
        List of attention outputs, each of shape (q_len_i, n_heads, head_dim).
    """
    outputs: List[torch.Tensor] = []

    for q, btable, ctx_len in zip(q_list, block_tables, context_lens):
        q_len, n_heads, head_dim = q.shape
        kv_groups = n_heads // arena.n_kv_heads

        k_seq, v_seq = arena.gather_sequence_kv(layer_idx, btable, ctx_len)
        k_exp = _repeat_kv_2d(k_seq, kv_groups)  # (ctx_len, n_heads, head_dim)
        v_exp = _repeat_kv_2d(v_seq, kv_groups)

        # Transpose to (n_heads, q_len, head_dim) and (n_heads, ctx_len, head_dim)
        qh = q.transpose(0, 1)
        kh = k_exp.transpose(0, 1)
        vh = v_exp.transpose(0, 1)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (n_heads, q_len, ctx_len)

        if q_len > 1:
            start_pos = ctx_len - q_len
            mask = torch.triu(
                torch.full((q_len, ctx_len), float("-inf"), device=q.device, dtype=scores.dtype),
                diagonal=1 + start_pos,
            )
            scores = scores + mask

        probs = F.softmax(scores.float(), dim=-1).type_as(q)
        out = torch.matmul(probs, vh).transpose(0, 1).contiguous()  # (q_len, n_heads, head_dim)
        outputs.append(out)

    return outputs

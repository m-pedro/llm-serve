import math
import torch
import torch.nn.functional as F

from llm_serve.allocator import BlockAllocator
from llm_serve.page_table import BlockTableManager
from llm_serve.paged_attn import PagedKVCacheArena, paged_attention_forward


def _reference_contiguous_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard contiguous causal GQA reference: inputs of shape (T, H, D)."""
    seq_len, n_heads, head_dim = q.shape
    n_kv = k.shape[1]
    groups = n_heads // n_kv
    k_rep = k[:, :, None, :].expand(seq_len, n_kv, groups, head_dim).reshape(seq_len, n_heads, head_dim)
    v_rep = v[:, :, None, :].expand(seq_len, n_kv, groups, head_dim).reshape(seq_len, n_heads, head_dim)

    qh, kh, vh = q.transpose(0, 1), k_rep.transpose(0, 1), v_rep.transpose(0, 1)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    probs = F.softmax(scores + mask, dim=-1)
    return torch.matmul(probs, vh).transpose(0, 1)


def test_paged_attention_matches_contiguous_attention_with_scrambled_physical_blocks():
    torch.manual_seed(42)
    alloc = BlockAllocator(num_blocks=12, block_size=4)
    # Pre-allocate dummy blocks 0, 2, 4 so our sequence gets non-contiguous physical blocks [1, 3, 5]!
    b0 = alloc.allocate()
    _seq_dummy1 = alloc.allocate()
    b2 = alloc.allocate()
    _seq_dummy2 = alloc.allocate()
    b4 = alloc.allocate()
    alloc.free(_seq_dummy1.block_id)
    alloc.free(_seq_dummy2.block_id)

    mgr = BlockTableManager(alloc)
    arena = PagedKVCacheArena(n_layers=1, num_blocks=12, block_size=4, n_kv_heads=2, head_dim=8)

    # Allocate 7-token prompt across non-contiguous physical blocks + 3 single-token decode steps = 10 tokens
    prompt_slots = mgr.allocate_sequence("seqA", [1, 2, 3, 4, 5, 6, 7])
    q_all = torch.randn(10, 4, 8)
    k_all = torch.randn(10, 2, 8)
    v_all = torch.randn(10, 2, 8)

    ref_out = _reference_contiguous_gqa(q_all, k_all, v_all)

    # 1. Prefill first 7 tokens into non-contiguous blocks
    arena.write_kv_slots(0, prompt_slots, k_all[:7], v_all[:7])
    prefill_out = paged_attention_forward(
        [q_all[:7]], arena, layer_idx=0, block_tables=[mgr.get_block_table("seqA")], context_lens=[7]
    )[0]
    assert torch.allclose(prefill_out, ref_out[:7], atol=1e-5)

    # 2. Decode tokens 7, 8, 9 one step at a time
    for step_idx in range(7, 10):
        slot, cow = mgr.append_token_slot("seqA", step_idx)
        arena.copy_blocks(cow)
        arena.write_kv_slots(0, [slot], k_all[step_idx : step_idx + 1], v_all[step_idx : step_idx + 1])
        step_out = paged_attention_forward(
            [q_all[step_idx : step_idx + 1]],
            arena,
            layer_idx=0,
            block_tables=[mgr.get_block_table("seqA")],
            context_lens=[step_idx + 1],
        )[0]
        assert torch.allclose(step_out, ref_out[step_idx : step_idx + 1], atol=1e-5)

    # Clean up dummy blocks
    alloc.free(b0.block_id)
    alloc.free(b2.block_id)
    alloc.free(b4.block_id)

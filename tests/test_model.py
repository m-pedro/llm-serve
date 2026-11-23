import torch
from llm_serve.allocator import BlockAllocator
from llm_serve.config import EngineConfig
from llm_serve.model import PagedTransformer
from llm_serve.paged_attn import PagedKVCacheArena
from llm_serve.prefix_cache import PrefixCachingBlockManager


def test_paged_transformer_ragged_mixed_batch_matches_standalone_reference():
    torch.manual_seed(7)
    cfg = EngineConfig(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        block_size=4,
        num_gpu_blocks=16,
    )
    model = PagedTransformer(cfg).eval()
    alloc = BlockAllocator(num_blocks=cfg.num_gpu_blocks, block_size=cfg.block_size)
    mgr = PrefixCachingBlockManager(alloc, enable_caching=False)
    arena = PagedKVCacheArena(
        cfg.n_layers, cfg.num_gpu_blocks, cfg.block_size, cfg.n_kv_heads, cfg.head_dim
    )

    # Sequence A is in decode mode (already prefilled 5 tokens, now decoding 6th token)
    seq_a = [10, 20, 30, 40, 50, 60]
    res_a = mgr.allocate_prompt("A", seq_a[:5])
    _ = model.forward_ragged(
        torch.tensor(seq_a[:5]),
        torch.arange(5),
        [5],
        [res_a.uncached_slots],
        [res_a.block_table],
        [5],
        arena,
    )

    # Sequence B arrives for a 4-token prefill in the SAME ragged forward batch as Sequence A's decode!
    seq_b = [7, 14, 21, 28]
    res_b = mgr.allocate_prompt("B", seq_b)
    slot_a6, _ = mgr.append_token("A", seq_a[5])

    mixed_logits = model.forward_ragged(
        flat_token_ids=torch.tensor([seq_a[5]] + seq_b),
        flat_positions=torch.tensor([5, 0, 1, 2, 3]),
        seq_token_counts=[1, 4],
        seq_slots=[[slot_a6], res_b.uncached_slots],
        block_tables=[mgr.page_tables["A"], res_b.block_table],
        context_lens=[6, 4],
        arena=arena,
    )

    ref_a = model.forward_standalone(torch.tensor([seq_a]))[0, -1]
    ref_b = model.forward_standalone(torch.tensor([seq_b]))[0]

    assert torch.allclose(mixed_logits[0][0], ref_a, atol=1e-5)
    assert torch.allclose(mixed_logits[1], ref_b, atol=1e-5)

import math
from llm_serve.allocator import BlockAllocator
from llm_serve.prefix_cache import PrefixCachingBlockManager


def test_automatic_prefix_caching_reuses_blocks_and_evicts_lru_under_pressure():
    alloc = BlockAllocator(num_blocks=5, block_size=4)
    mgr = PrefixCachingBlockManager(alloc, enable_caching=True)

    # System prompt of 8 tokens (2 full blocks) + 2 user query tokens = 10 tokens (3 blocks)
    sys_prefix = [101, 102, 103, 104, 201, 202, 203, 204]
    res1 = mgr.allocate_prompt("req-1", sys_prefix + [11, 12])
    assert res1.cached_tokens == 0
    assert len(res1.uncached_slots) == 10
    assert mgr.cache_lookups == 1 and mgr.cache_hits == 0
    first_two_blocks = res1.block_table[:2]

    # Finish req-1 -> its 2 full prefix blocks stay warm in the LRU cache!
    mgr.free_sequence("req-1")

    # req-2 shares the exact 8-token system prompt + different user query [99, 88]
    res2 = mgr.allocate_prompt("req-2", sys_prefix + [99, 88])
    assert res2.cached_tokens == 8
    assert res2.block_table[:2] == first_two_blocks
    assert len(res2.uncached_slots) == 2
    assert mgr.cache_hits == 2 and mgr.cache_lookups == 3
    assert math.isclose(mgr.hit_rate, 2.0 / 3.0)

    mgr.free_sequence("req-2")

    # Allocate a brand-new 16-token prompt (4 blocks) in a 5-block pool:
    # Must cleanly evict LRU cached blocks instead of throwing OutOfKVBlocksError!
    res3 = mgr.allocate_prompt("req-3", list(range(500, 516)))
    assert len(res3.block_table) == 4

import pytest
from llm_serve.allocator import BlockAllocator, OutOfKVBlocksError


def test_block_allocator_ref_counting_and_exhaustion():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.num_free_blocks == 4
    assert alloc.utilization == 0.0

    b0 = alloc.allocate()
    b1 = alloc.allocate()
    assert b0.block_id == 0 and b0.ref_count == 1
    assert b1.block_id == 1 and b1.ref_count == 1
    assert alloc.num_free_blocks == 2

    # Share b0 across two sequences via retain()
    alloc.retain(b0.block_id)
    assert b0.ref_count == 2

    # First free decrements ref_count to 1 without returning b0 to the free pool
    assert alloc.free(b0.block_id) is False
    assert alloc.num_free_blocks == 2

    # Second free drops ref_count to 0 and recycles b0
    assert alloc.free(b0.block_id) is True
    assert alloc.num_free_blocks == 3

    # Exhaust remaining 3 blocks and verify OutOfKVBlocksError
    alloc.allocate()
    alloc.allocate()
    alloc.allocate()
    assert alloc.num_free_blocks == 0
    with pytest.raises(OutOfKVBlocksError):
        alloc.allocate()

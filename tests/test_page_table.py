from llm_serve.allocator import BlockAllocator
from llm_serve.page_table import BlockTableManager


def test_page_table_allocation_fork_and_copy_on_write():
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    mgr = BlockTableManager(alloc)

    # 6 tokens across block_size=4 -> Block 0 (4 tokens, full) + Block 1 (2 tokens, partial)
    slots = mgr.allocate_sequence("parent", [10, 11, 12, 13, 14, 15])
    assert len(slots) == 6
    assert mgr.get_block_table("parent") == [0, 1]
    assert alloc.num_used_blocks == 2

    # Zero-copy fork: child shares blocks [0, 1] with zero additional physical blocks allocated!
    mgr.fork_sequence("parent", "child")
    assert mgr.get_block_table("child") == [0, 1]
    assert alloc.num_used_blocks == 2
    assert alloc.get_block(0).ref_count == 2
    assert alloc.get_block(1).ref_count == 2

    # Appending to `child` mutates shared tail Block 1 -> triggers Copy-on-Write (1 -> 2),
    # while full prefix Block 0 remains shared!
    slot_child, cow = mgr.append_token_slot("child", 99)
    assert cow == [(1, 2)]
    assert slot_child.block_id == 2 and slot_child.offset == 2
    assert mgr.get_block_table("parent") == [0, 1]
    assert mgr.get_block_table("child") == [0, 2]
    assert alloc.get_block(0).ref_count == 2  # Prefix block STILL shared!
    assert alloc.get_block(1).ref_count == 1
    assert alloc.get_block(2).ref_count == 1

    mgr.free_sequence("parent")
    mgr.free_sequence("child")
    assert alloc.num_used_blocks == 0

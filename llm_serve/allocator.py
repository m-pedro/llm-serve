from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


class OutOfKVBlocksError(RuntimeError):
    """Raised when the physical KV-block pool has zero free or evictable blocks."""


@dataclass
class PhysicalKVBlock:
    block_id: int
    block_size: int
    ref_count: int = 0
    block_hash: Optional[str] = None
    token_ids: List[int] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) == self.block_size

    def reset(self) -> None:
        self.ref_count = 0
        self.block_hash = None
        self.token_ids.clear()


class BlockAllocator:
    """Reference-counted O(1) physical KV-block frame allocator."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be > 0")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: List[PhysicalKVBlock] = [
            PhysicalKVBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]
        self._free_ids: Deque[int] = deque(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free_ids)

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / float(self.num_blocks)

    def allocate(self) -> PhysicalKVBlock:
        if not self._free_ids:
            raise OutOfKVBlocksError("KV-Cache physical block pool exhausted")
        bid = self._free_ids.popleft()
        blk = self.blocks[bid]
        blk.reset()
        blk.ref_count = 1
        return blk

    def retain(self, block_id: int) -> PhysicalKVBlock:
        """Increments reference count for shared prefix or forked sequence blocks."""
        blk = self.blocks[block_id]
        if blk.ref_count <= 0:
            raise ValueError(f"cannot retain unallocated block {block_id}")
        blk.ref_count += 1
        return blk

    def free(self, block_id: int) -> bool:
        """Decrements ref_count; returns True if block reached ref_count == 0 and returned to free list."""
        blk = self.blocks[block_id]
        if blk.ref_count <= 0:
            raise ValueError(f"double free detected on physical block {block_id}")
        blk.ref_count -= 1
        if blk.ref_count == 0:
            blk.reset()
            self._free_ids.append(block_id)
            return True
        return False

    def get_block(self, block_id: int) -> PhysicalKVBlock:
        return self.blocks[block_id]

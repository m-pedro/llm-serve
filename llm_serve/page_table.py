from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

from llm_serve.allocator import BlockAllocator


@dataclass(frozen=True)
class PhysicalSlot:
    """Physical (block_id, block_offset) location in the KV-Cache for a single token."""

    block_id: int
    offset: int


class BlockTableManager:
    """Maps logical sequence blocks to physical KV blocks with Copy-on-Write (CoW)."""

    def __init__(self, allocator: BlockAllocator) -> None:
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.page_tables: Dict[str, List[int]] = {}

    def can_allocate_tokens(self, num_tokens: int) -> bool:
        needed = math.ceil(num_tokens / float(self.block_size))
        return self.allocator.num_free_blocks >= needed

    def allocate_sequence(self, request_id: str, prompt_tokens: Sequence[int]) -> List[PhysicalSlot]:
        """Allocates physical blocks for a new sequence's prompt and returns token write slots."""
        if request_id in self.page_tables:
            raise ValueError(f"request_id '{request_id}' already has an active page table")

        table: List[int] = []
        slots: List[PhysicalSlot] = []

        for idx, tok in enumerate(prompt_tokens):
            offset = idx % self.block_size
            if offset == 0:
                blk = self.allocator.allocate()
                table.append(blk.block_id)
            curr_blk = self.allocator.get_block(table[-1])
            curr_blk.token_ids.append(int(tok))
            slots.append(PhysicalSlot(block_id=curr_blk.block_id, offset=offset))

        self.page_tables[request_id] = table
        return slots

    def fork_sequence(self, parent_req_id: str, child_req_id: str) -> None:
        """Zero-copy fork sharing all physical blocks of `parent_req_id` via ref_count++."""
        if parent_req_id not in self.page_tables:
            raise KeyError(f"unknown parent request '{parent_req_id}'")
        if child_req_id in self.page_tables:
            raise ValueError(f"child request '{child_req_id}' already exists")

        shared_table: List[int] = []
        for bid in self.page_tables[parent_req_id]:
            self.allocator.retain(bid)
            shared_table.append(bid)
        self.page_tables[child_req_id] = shared_table

    def append_token_slot(
        self, request_id: str, token_id: int
    ) -> Tuple[PhysicalSlot, List[Tuple[int, int]]]:
        """Appends 1 token to `request_id`, triggering Copy-on-Write if tail block has ref_count > 1.

        Returns:
            (PhysicalSlot, cow_copies) where cow_copies is a list of (src_block_id, dst_block_id).
        """
        table = self.page_tables[request_id]
        cow_copies: List[Tuple[int, int]] = []
        last_blk = self.allocator.get_block(table[-1])

        if last_blk.is_full:
            new_blk = self.allocator.allocate()
            new_blk.token_ids.append(int(token_id))
            table.append(new_blk.block_id)
            return PhysicalSlot(block_id=new_blk.block_id, offset=0), cow_copies

        # Tail block has spare capacity; if shared (ref_count > 1), trigger Copy-on-Write!
        if last_blk.ref_count > 1:
            src_id = last_blk.block_id
            prior_tokens = list(last_blk.token_ids)
            new_blk = self.allocator.allocate()
            new_blk.token_ids.extend(prior_tokens)
            self.allocator.free(src_id)
            table[-1] = new_blk.block_id
            cow_copies.append((src_id, new_blk.block_id))
            last_blk = new_blk

        offset = len(last_blk.token_ids)
        last_blk.token_ids.append(int(token_id))
        return PhysicalSlot(block_id=last_blk.block_id, offset=offset), cow_copies

    def get_block_table(self, request_id: str) -> List[int]:
        return list(self.page_tables[request_id])

    def free_sequence(self, request_id: str) -> None:
        table = self.page_tables.pop(request_id, [])
        for bid in table:
            self.allocator.free(bid)

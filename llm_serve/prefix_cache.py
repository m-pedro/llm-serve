from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from llm_serve.allocator import BlockAllocator, OutOfKVBlocksError, PhysicalKVBlock
from llm_serve.page_table import PhysicalSlot


def compute_chained_block_hash(
    parent_hash: Optional[str], token_ids: Sequence[int]
) -> str:
    """Computes position-exact chained Blake2b hash over (parent_hash, token_ids)."""
    h = hashlib.blake2b(digest_size=16)
    h.update((parent_hash or "ROOT").encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(int(t)) for t in token_ids).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class PrefixAllocationResult:
    block_table: List[int]
    cached_tokens: int
    uncached_slots: List[PhysicalSlot]


class PrefixCachingBlockManager:
    """KV-Block manager with Automatic Prefix Caching (APC) and LRU block eviction."""

    def __init__(self, allocator: BlockAllocator, enable_caching: bool = True) -> None:
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.enable_caching = enable_caching
        self.page_tables: Dict[str, List[int]] = {}

        # hash -> physical block_id (includes both active ref_count>0 and cached evictable blocks)
        self._hash_to_block: Dict[str, int] = {}
        # OrderedDict of block_id -> block_hash for unreferenced cached blocks (LRU order)
        self._evictable_lru: "OrderedDict[int, str]" = OrderedDict()

        self.cache_hits: int = 0
        self.cache_lookups: int = 0

    @property
    def num_available_blocks(self) -> int:
        """Total blocks available for new allocations (free pool + LRU-evictable cached blocks)."""
        return self.allocator.num_free_blocks + len(self._evictable_lru)

    @property
    def hit_rate(self) -> float:
        return (self.cache_hits / float(self.cache_lookups)) if self.cache_lookups > 0 else 0.0

    def _acquire_fresh_block(self) -> PhysicalKVBlock:
        if self.allocator.num_free_blocks > 0:
            return self.allocator.allocate()
        if self._evictable_lru:
            evict_bid, evict_hash = self._evictable_lru.popitem(last=False)
            self._hash_to_block.pop(evict_hash, None)
            # Evictable cached blocks hold a sentinel ref_count == 1 in allocator; release & reallocate
            self.allocator.free(evict_bid)
            return self.allocator.allocate()
        raise OutOfKVBlocksError("No free or evictable KV blocks remaining")

    def allocate_prompt(
        self, request_id: str, prompt_tokens: Sequence[int]
    ) -> PrefixAllocationResult:
        if request_id in self.page_tables:
            raise ValueError(f"duplicate request_id '{request_id}'")

        table: List[int] = []
        uncached_slots: List[PhysicalSlot] = []
        cached_tokens = 0
        parent_hash: Optional[str] = None
        prefix_matching_active = self.enable_caching

        n = len(prompt_tokens)
        # Always leave at least 1 token uncached so the model runs forward to emit the first logit!
        max_cacheable_tokens = max(0, n - 1)

        for chunk_start in range(0, n, self.block_size):
            chunk = list(prompt_tokens[chunk_start : chunk_start + self.block_size])
            is_full_block = len(chunk) == self.block_size
            can_try_cache = (
                prefix_matching_active
                and is_full_block
                and (chunk_start + self.block_size) <= max_cacheable_tokens
            )

            if is_full_block and self.enable_caching:
                blk_hash = compute_chained_block_hash(parent_hash, chunk)
            else:
                blk_hash = None

            if can_try_cache and blk_hash is not None:
                self.cache_lookups += 1
                if blk_hash in self._hash_to_block:
                    bid = self._hash_to_block[blk_hash]
                    if bid in self._evictable_lru:
                        # Rescue from LRU evictable pool (already holds sentinel ref_count=1)
                        self._evictable_lru.pop(bid)
                    else:
                        self.allocator.retain(bid)
                    table.append(bid)
                    cached_tokens += self.block_size
                    parent_hash = blk_hash
                    self.cache_hits += 1
                    continue
                prefix_matching_active = False

            # Cache miss or tail partial block -> allocate fresh physical block
            blk = self._acquire_fresh_block()
            blk.token_ids.extend(int(t) for t in chunk)
            if blk_hash is not None:
                blk.block_hash = blk_hash
                if blk_hash not in self._hash_to_block:
                    self._hash_to_block[blk_hash] = blk.block_id
                parent_hash = blk_hash

            table.append(blk.block_id)
            for off in range(len(chunk)):
                uncached_slots.append(PhysicalSlot(block_id=blk.block_id, offset=off))

        self.page_tables[request_id] = table
        return PrefixAllocationResult(
            block_table=table,
            cached_tokens=cached_tokens,
            uncached_slots=uncached_slots,
        )

    def append_token(
        self, request_id: str, token_id: int
    ) -> Tuple[PhysicalSlot, List[Tuple[int, int]]]:
        table = self.page_tables[request_id]
        cow_copies: List[Tuple[int, int]] = []
        last_blk = self.allocator.get_block(table[-1])

        if last_blk.is_full:
            new_blk = self._acquire_fresh_block()
            new_blk.token_ids.append(int(token_id))
            table.append(new_blk.block_id)
            return PhysicalSlot(block_id=new_blk.block_id, offset=0), cow_copies

        if last_blk.ref_count > 1:
            src_id = last_blk.block_id
            prior_tokens = list(last_blk.token_ids)
            new_blk = self._acquire_fresh_block()
            new_blk.token_ids.extend(prior_tokens)
            self.allocator.free(src_id)
            table[-1] = new_blk.block_id
            cow_copies.append((src_id, new_blk.block_id))
            last_blk = new_blk

        offset = len(last_blk.token_ids)
        last_blk.token_ids.append(int(token_id))

        # Seal block hash once a newly decoded block reaches full capacity
        if last_blk.is_full and self.enable_caching and last_blk.block_hash is None:
            parent_hash = (
                self.allocator.get_block(table[-2]).block_hash if len(table) >= 2 else None
            )
            sealed = compute_chained_block_hash(parent_hash, last_blk.token_ids)
            if sealed not in self._hash_to_block:
                last_blk.block_hash = sealed
                self._hash_to_block[sealed] = last_blk.block_id

        return PhysicalSlot(block_id=last_blk.block_id, offset=offset), cow_copies

    def free_sequence(self, request_id: str) -> None:
        table = self.page_tables.pop(request_id, [])
        for bid in table:
            blk = self.allocator.get_block(bid)
            # Keep hashed full blocks cached in LRU when their last active reader finishes!
            if (
                self.enable_caching
                and blk.block_hash is not None
                and blk.ref_count == 1
                and self._hash_to_block.get(blk.block_hash) == bid
            ):
                self._evictable_lru[bid] = blk.block_hash
                self._evictable_lru.move_to_end(bid)
            else:
                if blk.ref_count == 1 and blk.block_hash is not None:
                    if self._hash_to_block.get(blk.block_hash) == bid:
                        self._hash_to_block.pop(blk.block_hash, None)
                self.allocator.free(bid)

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, List, Tuple

from llm_serve.page_table import PhysicalSlot
from llm_serve.prefix_cache import PrefixCachingBlockManager
from llm_serve.request import SequenceRequest, SequenceStatus


@dataclass
class ScheduledSequence:
    """Describes the exact tokens and physical KV slots a sequence will run in the current step."""

    request: SequenceRequest
    token_ids: List[int]
    positions: List[int]
    write_slots: List[PhysicalSlot]
    block_table: List[int]
    context_len: int
    is_prefill: bool
    emit_output_token: bool = True


@dataclass
class SchedulerStepOutput:
    scheduled: List[ScheduledSequence] = field(default_factory=list)
    preempted_ids: List[str] = field(default_factory=list)
    cow_copies: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled) == 0


class ContinuousBatchScheduler:
    """Iteration-level FCFS scheduler with KV-block preemption and chunked prefill."""

    def __init__(
        self,
        block_manager: PrefixCachingBlockManager,
        max_batch_seqs: int = 16,
        max_step_tokens: int = 256,
    ) -> None:
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.max_batch_seqs = max_batch_seqs
        self.max_step_tokens = max_step_tokens

        self.waiting: Deque[SequenceRequest] = deque()
        self.running: List[SequenceRequest] = []
        self._pending_slots: dict[str, List[PhysicalSlot]] = {}

    def add_request(self, req: SequenceRequest) -> None:
        req.status = SequenceStatus.WAITING
        self.waiting.append(req)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def free_finished(self) -> List[SequenceRequest]:
        still_running: List[SequenceRequest] = []
        finished: List[SequenceRequest] = []
        for req in self.running:
            if req.is_finished:
                self.block_manager.free_sequence(req.request_id)
                self._pending_slots.pop(req.request_id, None)
                finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        return finished

    def _needs_new_block_for_decode(self, req: SequenceRequest) -> bool:
        table = self.block_manager.page_tables[req.request_id]
        last_blk = self.block_manager.allocator.get_block(table[-1])
        return last_blk.is_full or last_blk.ref_count > 1

    def _preempt_request(self, victim: SequenceRequest) -> None:
        victim.status = SequenceStatus.PREEMPTED
        victim.num_computed_tokens = 0
        self.block_manager.free_sequence(victim.request_id)
        self._pending_slots.pop(victim.request_id, None)
        self.waiting.appendleft(victim)

    def schedule(self) -> SchedulerStepOutput:
        self.free_finished()
        output = SchedulerStepOutput()
        token_budget = self.max_step_tokens

        # Phase 1: Schedule active RUNNING sequences (either continuing chunked prefill or 1-token decode)
        idx = 0
        while idx < len(self.running):
            req = self.running[idx]
            if token_budget <= 0:
                break

            # Case A: Sequence still has uncomputed prompt tokens (chunked prefill continuation)
            if req.num_computed_tokens < len(req.prompt_token_ids):
                remaining = len(req.prompt_token_ids) - req.num_computed_tokens
                chunk_len = min(remaining, token_budget)
                start = req.num_computed_tokens
                end = start + chunk_len

                all_prompt_slots = self._pending_slots[req.request_id]
                step_slots = all_prompt_slots[start:end]
                step_tokens = req.prompt_token_ids[start:end]
                step_positions = list(range(start, end))

                req.num_computed_tokens = end
                token_budget -= chunk_len
                completes_prompt = req.num_computed_tokens == len(req.prompt_token_ids)

                output.scheduled.append(
                    ScheduledSequence(
                        request=req,
                        token_ids=step_tokens,
                        positions=step_positions,
                        write_slots=step_slots,
                        block_table=list(self.block_manager.page_tables[req.request_id]),
                        context_len=end,
                        is_prefill=True,
                        emit_output_token=completes_prompt,
                    )
                )
                idx += 1
                continue

            # Case B: Standard 1-token autoregressive decode step
            while self._needs_new_block_for_decode(req) and self.block_manager.num_available_blocks < 1:
                if len(self.running) <= 1 or self.running[-1] is req:
                    # Preempt current sequence itself if no newer victim exists
                    self.running.pop(idx)
                    self._preempt_request(req)
                    output.preempted_ids.append(req.request_id)
                    req = None  # type: ignore[assignment]
                    break
                victim = self.running.pop()
                self._preempt_request(victim)
                output.preempted_ids.append(victim.request_id)

            if req is None:
                continue

            last_tok = req.all_token_ids[-1]
            pos = req.total_len - 1
            slot, cow = self.block_manager.append_token(req.request_id, last_tok)
            output.cow_copies.extend(cow)
            req.num_computed_tokens = req.total_len
            token_budget -= 1

            output.scheduled.append(
                ScheduledSequence(
                    request=req,
                    token_ids=[last_tok],
                    positions=[pos],
                    write_slots=[slot],
                    block_table=list(self.block_manager.page_tables[req.request_id]),
                    context_len=req.total_len,
                    is_prefill=False,
                    emit_output_token=True,
                )
            )
            idx += 1

        # Phase 2: Admit WAITING / PREEMPTED sequences if batch slots & KV blocks allow
        while (
            self.waiting
            and len(self.running) < self.max_batch_seqs
            and token_budget > 0
        ):
            cand = self.waiting[0]
            effective_prompt = cand.all_token_ids
            needed_blocks = math.ceil(len(effective_prompt) / float(self.block_size))
            if self.block_manager.num_available_blocks < needed_blocks:
                break

            self.waiting.popleft()
            # If preempted earlier after generating tokens, treat all current tokens as prompt to recompute KV
            if cand.output_token_ids:
                cand.prompt_token_ids = list(cand.all_token_ids)
                cand.output_token_ids.clear()

            alloc_res = self.block_manager.allocate_prompt(cand.request_id, cand.prompt_token_ids)
            # Reconstruct full per-position slot list (cached prefix slots + newly allocated slots)
            full_slots: List[PhysicalSlot] = []
            for p_idx in range(len(cand.prompt_token_ids)):
                blk_id = alloc_res.block_table[p_idx // self.block_size]
                full_slots.append(PhysicalSlot(block_id=blk_id, offset=p_idx % self.block_size))
            self._pending_slots[cand.request_id] = full_slots

            cand.status = SequenceStatus.RUNNING
            cand.num_computed_tokens = alloc_res.cached_tokens

            remaining = len(cand.prompt_token_ids) - cand.num_computed_tokens
            chunk_len = min(remaining, token_budget)
            start = cand.num_computed_tokens
            end = start + chunk_len

            cand.num_computed_tokens = end
            token_budget -= chunk_len
            completes_prompt = end == len(cand.prompt_token_ids)

            self.running.append(cand)
            output.scheduled.append(
                ScheduledSequence(
                    request=cand,
                    token_ids=cand.prompt_token_ids[start:end],
                    positions=list(range(start, end)),
                    write_slots=full_slots[start:end],
                    block_table=list(alloc_res.block_table),
                    context_len=end,
                    is_prefill=True,
                    emit_output_token=completes_prompt,
                )
            )

        return output

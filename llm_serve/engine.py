from typing import Dict, List, Optional, Sequence
import torch

from llm_serve.allocator import BlockAllocator
from llm_serve.config import EngineConfig
from llm_serve.model import PagedTransformer
from llm_serve.paged_attn import PagedKVCacheArena
from llm_serve.prefix_cache import PrefixCachingBlockManager
from llm_serve.request import SamplingParams, SequenceRequest
from llm_serve.sampler import HeterogeneousSampler
from llm_serve.scheduler import ContinuousBatchScheduler, SchedulerStepOutput


class LLMEngine:
    """vLLM-style serving engine orchestrating PagedAttention, APC, and Continuous Batching."""

    def __init__(
        self,
        config: EngineConfig,
        model: Optional[PagedTransformer] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = (model or PagedTransformer(config)).to(self.device).eval()

        self.allocator = BlockAllocator(
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
        self.block_manager = PrefixCachingBlockManager(
            self.allocator,
            enable_caching=config.enable_prefix_caching,
        )
        self.scheduler = ContinuousBatchScheduler(
            self.block_manager,
            max_batch_seqs=config.max_batch_seqs,
            max_step_tokens=config.max_Step_tokens,
        )
        self.arena = PagedKVCacheArena(
            n_layers=config.n_layers,
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            device=self.device,
        )
        self.sampler = HeterogeneousSampler()
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        self.current_step: int = 0
        self.requests: Dict[str, SequenceRequest] = {}
        self.peak_kv_utilization: float = 0.0
        self.preemption_count: int = 0

    def submit_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> SequenceRequest:
        req = SequenceRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params or SamplingParams(),
            arrival_step=self.current_step,
        )
        self.requests[request_id] = req
        self.scheduler.add_request(req)
        return req

    def has_unfinished_work(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    @torch.no_grad()
    def step(self) -> SchedulerStepOutput:
        """Executes one iteration-level scheduling + ragged PagedAttention forward step."""
        self.current_step += 1
        sched_out = self.scheduler.schedule()
        self.preemption_count += len(sched_out.preempted_ids)

        if sched_out.is_empty:
            self.scheduler.free_finished()
            return sched_out

        if sched_out.cow_copies:
            self.arena.copy_blocks(sched_out.cow_copies)

        self.peak_kv_utilization = max(
            self.peak_kv_utilization, self.allocator.utilization
        )

        flat_tokens: List[int] = []
        flat_positions: List[int] = []
        seq_counts: List[int] = []
        seq_slots = []
        block_tables = []
        context_lens: List[int] = []

        for item in sched_out.scheduled:
            flat_tokens.extend(item.token_ids)
            flat_positions.extend(item.positions)
            seq_counts.append(len(item.token_ids))
            seq_slots.append(item.write_slots)
            block_tables.append(item.block_table)
            context_lens.append(item.context_len)

        tok_tensor = torch.tensor(flat_tokens, device=self.device, dtype=torch.long)
        pos_tensor = torch.tensor(flat_positions, device=self.device, dtype=torch.long)

        logits_per_seq = self.model.forward_ragged(
            flat_token_ids=tok_tensor,
            flat_positions=pos_tensor,
            seq_token_counts=seq_counts,
            seq_slots=seq_slots,
            block_tables=block_tables,
            context_lens=context_lens,
            arena=self.arena,
        )

        # Sample next tokens ONLY for sequences whose prompt prefill is complete
        emit_items = []
        emit_logits = []
        emit_params = []
        for item, seq_logits in zip(sched_out.scheduled, logits_per_seq):
            if item.emit_output_token:
                emit_items.append(item)
                emit_logits.append(seq_logits[-1].cpu())
                emit_params.append(item.request.sampling_params)

        if emit_items:
            sampled_ids = self.sampler.sample_batch(
                emit_logits, emit_params, generator=self.rng
            )
            for item, next_tok in zip(emit_items, sampled_ids):
                item.request.append_output_token(next_tok, current_step=self.current_step)

        # Immediately reclaim physical blocks for any sequence that hit EOS or max_new_tokens!
        self.scheduler.free_finished()
        return sched_out

    def run_to_completion(self) -> List[SequenceRequest]:
        while self.has_unfinished_work():
            self.step()
        return list(self.requests.values())

from llm_serve.allocator import BlockAllocator
from llm_serve.prefix_cache import PrefixCachingBlockManager
from llm_serve.request import SamplingParams, SequenceRequest, SequenceStatus
from llm_serve.scheduler import ContinuousBatchScheduler


def test_continuous_batch_immediate_slot_reclamation_and_preemption():
    # Pool of ONLY 3 blocks (block_size=4 -> 12 tokens max) forces preemption when both sequences grow!
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    mgr = PrefixCachingBlockManager(alloc, enable_caching=False)
    sched = ContinuousBatchScheduler(mgr, max_batch_seqs=2, max_step_tokens=64)

    r1 = SequenceRequest("req-1", [1, 2, 3, 4], SamplingParams(max_new_tokens=3))
    r2 = SequenceRequest("req-2", [5, 6, 7, 8], SamplingParams(max_new_tokens=3))
    r3 = SequenceRequest("req-3", [9, 10], SamplingParams(max_new_tokens=2))

    sched.add_request(r1)
    sched.add_request(r2)
    sched.add_request(r3)

    # Step 1: Prefills r1 (1 block) and r2 (1 block) -> 2 of 3 blocks used; r3 waits due to max_batch_seqs=2
    s1 = sched.schedule()
    assert [item.request.request_id for item in s1.scheduled] == ["req-1", "req-2"]
    assert all(item.is_prefill for item in s1.scheduled)
    r1.append_output_token(11, current_step=1)
    r2.append_output_token(21, current_step=1)

    # Step 2: Both r1 and r2 crossed a 4-token block boundary and need a new block,
    # but only 1 free block remains! Scheduler must preempt r2 (newest) so r1 proceeds without OOM.
    s2 = sched.schedule()
    assert s2.preempted_ids == ["req-2"]
    assert r2.status == SequenceStatus.PREEMPTED
    assert s2.scheduled[0].request.request_id == "req-1"

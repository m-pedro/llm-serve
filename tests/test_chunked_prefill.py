from llm_serve.allocator import BlockAllocator
from llm_serve.prefix_cache import PrefixCachingBlockManager
from llm_serve.request import SamplingParams, SequenceRequest
from llm_serve.scheduler import ContinuousBatchScheduler


def test_token_budgeted_chunked_prefill_interleaves_with_active_decode():
    alloc = BlockAllocator(num_blocks=16, block_size=4)
    mgr = PrefixCachingBlockManager(alloc, enable_caching=False)
    # Strict budget of 5 tokens per forward step
    sched = ContinuousBatchScheduler(mgr, max_batch_seqs=4, max_step_tokens=5)

    short_decode = SequenceRequest("short", [1, 2], SamplingParams(max_new_tokens=4))
    sched.add_request(short_decode)

    # Step 1: Prefill short_decode (2 tokens)
    step1 = sched.schedule()
    assert len(step1.scheduled) == 1
    assert step1.scheduled[0].emit_output_token is True
    short_decode.append_output_token(3, current_step=1)

    # Now a 10-token long prompt arrives while `short` is actively decoding
    long_prompt = SequenceRequest("long", list(range(100, 110)), SamplingParams(max_new_tokens=2))
    sched.add_request(long_prompt)

    # Step 2: `short` takes 1 decode token -> remaining budget = 4 tokens for `long` [0:4]
    step2 = sched.schedule()
    assert [s.request.request_id for s in step2.scheduled] == ["short", "long"]
    assert step2.scheduled[0].is_prefill is False and len(step2.scheduled[0].token_ids) == 1
    assert step2.scheduled[1].is_prefill is True and len(step2.scheduled[1].token_ids) == 4
    assert step2.scheduled[1].emit_output_token is False
    short_decode.append_output_token(4, current_step=2)

    # Step 3: `long` continues chunked prefill [4:8] (4 tokens) + `short` decodes 1 token = 5 tokens!
    # Notice both running sequences share the 5-token step budget cleanly without stalling `short`.
    step3 = sched.schedule()
    by_id = {s.request.request_id: s for s in step3.scheduled}
    assert len(by_id["short"].token_ids) == 1
    assert by_id["long"].token_ids == [104, 105, 106, 107]
    assert by_id["long"].emit_output_token is False
    short_decode.append_output_token(5, current_step=3)

    # Step 4: `long` finishes final 2 prompt tokens [108, 109] with emit_output_token == True!
    step4 = sched.schedule()
    by_id4 = {s.request.request_id: s for s in step4.scheduled}
    assert by_id4["long"].token_ids == [108, 109]
    assert by_id4["long"].emit_output_token is True

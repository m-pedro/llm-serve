import pytest
from llm_serve.config import EngineConfig
from llm_serve.request import SamplingParams, SequenceRequest, SequenceStatus


def test_engine_config_and_sequence_lifecycle_transitions():
    cfg = EngineConfig(dim=128, n_heads=8, n_kv_heads=2, block_size=16, num_gpu_blocks=64)
    assert cfg.head_dim == 16
    assert cfg.kv_groups == 4
    assert cfg.total_kv_token_capacity == 1024

    req = SequenceRequest(
        request_id="req-1",
        prompt_token_ids=[10, 20, 30],
        sampling_params=SamplingParams(max_new_tokens=3, eos_token_id=99),
        arrival_step=0,
    )
    assert req.status == SequenceStatus.WAITING
    assert not req.is_finished

    req.status = SequenceStatus.RUNNING
    req.append_output_token(40, current_step=1)
    assert req.first_token_step == 1
    assert req.total_len == 4
    assert not req.is_finished

    # Hitting EOS finishes the sequence immediately prior to max_new_tokens
    req.append_output_token(99, current_step=2)
    assert req.is_finished
    assert req.status == SequenceStatus.FINISHED_EOS
    assert req.finish_step == 2

    with pytest.raises(ValueError, match="at least 1 token"):
        SequenceRequest(request_id="req-empty", prompt_token_ids=[])

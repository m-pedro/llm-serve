import torch
from llm_serve.config import EngineConfig
from llm_serve.engine import LLMEngine
from llm_serve.metrics import summarize_engine_metrics
from llm_serve.request import SamplingParams


def test_serving_telemetry_ttft_tpot_and_prefix_hit_rate():
    torch.manual_seed(19)
    cfg = EngineConfig(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        block_size=4,
        num_gpu_blocks=32,
        enable_prefix_caching=True,
    )
    engine = LLMEngine(cfg)
    shared_sys = [1, 2, 3, 4, 5, 6, 7, 8]

    # Run request 1 to populate prefix cache, then submit request 2 sharing the 8-token prefix
    engine.submit_request("q1", shared_sys + [11, 12], SamplingParams(max_new_tokens=4))
    engine.run_to_completion()

    engine.submit_request("q2", shared_sys + [21, 22], SamplingParams(max_new_tokens=4))
    engine.run_to_completion()

    report = summarize_engine_metrics(engine)
    assert report.num_requests == 2
    assert report.total_output_tokens == 8
    assert report.mean_ttft_steps >= 1.0
    assert report.mean_tpot_steps == 1.0  # Uninterrupted 1 token per decode step
    assert report.prefix_cache_hit_rate > 0.5
    assert 0.0 < report.peak_kv_utilization <= 1.0

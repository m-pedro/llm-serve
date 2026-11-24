import torch
from llm_serve.config import EngineConfig
from llm_serve.engine import LLMEngine
from llm_serve.request import SamplingParams


def test_llm_engine_dynamic_arrivals_and_completion():
    torch.manual_seed(42)
    cfg = EngineConfig(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        block_size=4,
        num_gpu_blocks=24,
        max_batch_seqs=4,
    )
    engine = LLMEngine(cfg)

    engine.submit_request("r1", [1, 2, 3, 4, 5], SamplingParams(max_new_tokens=4, temperature=0.0))
    engine.submit_request("r2", [10, 20, 30], SamplingParams(max_new_tokens=6, temperature=0.0))

    # Run 2 steps, then inject a mid-flight request r3 while r1 and r2 are actively decoding!
    engine.step()
    engine.step()
    engine.submit_request("r3", [7, 8, 9, 10], SamplingParams(max_new_tokens=3, temperature=0.0))

    completed = engine.run_to_completion()
    assert len(completed) == 3
    assert all(r.is_finished for r in completed)
    assert [len(r.output_token_ids) for r in completed] == [4, 6, 3]

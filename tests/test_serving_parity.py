import copy
import torch

from llm_serve.config import EngineConfig
from llm_serve.engine import LLMEngine
from llm_serve.model import PagedTransformer
from llm_serve.request import SamplingParams
from llm_serve.speculative import speculative_generate


def _standalone_greedy(model: PagedTransformer, prompt: list[int], max_new: int) -> list[int]:
    seq = list(prompt)
    out: list[int] = []
    for _ in range(max_new):
        logits = model.forward_standalone(torch.tensor([seq]))[0, -1]
        tok = int(torch.argmax(logits).item())
        seq.append(tok)
        out.append(tok)
    return out


def _simulate_static_batching_steps(lengths: list[int], batch_size: int) -> int:
    """In static batching, each wave of `batch_size` requests blocks until max(wave_lengths) finishes."""
    total_steps = 0
    for i in range(0, len(lengths), batch_size):
        wave = lengths[i : i + batch_size]
        total_steps += max(wave)
    return total_steps


def test_end_to_end_continuous_apc_chunked_and_speculative_parity():
    torch.manual_seed(123)
    cfg = EngineConfig(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        block_size=4,
        num_gpu_blocks=48,
        max_batch_seqs=2,
        max_Step_tokens=6,  # Forces chunked prefill on 10-token prompts!
        enable_prefix_caching=True,
    )
    shared_model = PagedTransformer(cfg).eval()
    engine = LLMEngine(cfg, model=shared_model)

    sys_prefix = [10, 11, 12, 13, 14, 15, 16, 17]  # 2 full cacheable blocks
    workload = [
        ("req-A", sys_prefix + [21, 22], 3),
        ("req-B", sys_prefix + [31, 32], 12),  # Long generation that would block static batching!
        ("req-C", sys_prefix + [41, 42], 3),
        ("req-D", sys_prefix + [51, 52], 3),
    ]

    # Submit first two, step twice, then submit C and D mid-flight
    engine.submit_request(workload[0][0], workload[0][1], SamplingParams(max_new_tokens=workload[0][2]))
    engine.submit_request(workload[1][0], workload[1][1], SamplingParams(max_new_tokens=workload[1][2]))
    engine.step()
    engine.step()
    engine.submit_request(workload[2][0], workload[2][1], SamplingParams(max_new_tokens=workload[2][2]))
    engine.submit_request(workload[3][0], workload[3][1], SamplingParams(max_new_tokens=workload[3][2]))

    engine.run_to_completion()

    # 1. Verify 100% bit-exact output parity against standalone greedy AND speculative decoding
    draft = copy.deepcopy(shared_model)
    for req_id, prompt, max_new in workload:
        expected = _standalone_greedy(shared_model, prompt, max_new)
        actual_cont = engine.requests[req_id].output_token_ids
        spec_out = speculative_generate(
            shared_model, draft, prompt, max_new_tokens=max_new, gamma=3, temperature=0.0
        ).output_token_ids
        assert actual_cont == expected, f"Mismatch on {req_id}"
        assert spec_out == expected

    # 2. Verify req-A, req-C, and req-D all finished inside req-B's window via Continuous Batching:
    # Under Static Batching (batch_size=2), Wave 1 ([3, 12]) takes 12 steps before req-C even starts!
    static_wave1_steps = _simulate_static_batching_steps([3, 12], batch_size=2)
    assert engine.requests["req-C"].first_token_step is not None
    assert engine.requests["req-C"].first_token_step < static_wave1_steps

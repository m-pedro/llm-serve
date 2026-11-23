import torch
from llm_serve.request import SamplingParams
from llm_serve.sampler import HeterogeneousSampler


def test_heterogeneous_sampler_supports_mixed_greedy_and_nucleus_requests():
    sampler = HeterogeneousSampler()
    logits_a = torch.tensor([0.1, 0.2, 9.5, 0.3])  # Clear argmax at token 2
    logits_b = torch.tensor([3.0, 2.8, -5.0, -6.0])  # Top-2 tokens are 0 and 1

    gen = torch.Generator().manual_seed(42)
    params = [
        SamplingParams(temperature=0.0),  # Deterministic greedy
        SamplingParams(temperature=0.7, top_k=2, top_p=0.95),  # Constrained to {0, 1}
    ]

    for _ in range(10):
        out = sampler.sample_batch([logits_a, logits_b], params, generator=gen)
        assert out[0] == 2
        assert out[1] in (0, 1)

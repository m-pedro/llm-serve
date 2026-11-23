from typing import List, Optional, Sequence
import torch
import torch.nn.functional as F

from llm_serve.request import SamplingParams


def sample_single_logits(
    logits_1d: torch.Tensor,
    params: SamplingParams,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Samples next token ID from a 1D (vocab_size,) logit vector using `params`."""
    if params.temperature <= 0.0:
        return int(torch.argmax(logits_1d).item())

    scaled = logits_1d.float() / float(params.temperature)

    if params.top_k is not None and params.top_k > 0:
        k = min(params.top_k, scaled.numel())
        top_vals, _ = torch.topk(scaled, k)
        scaled = torch.where(
            scaled < top_vals[-1],
            torch.full_like(scaled, float("-inf")),
            scaled,
        )

    if params.top_p is not None and 0.0 < params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs > params.top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        scaled = torch.empty_like(scaled).scatter(0, sorted_idx, sorted_logits)

    probs = F.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled.item())


class HeterogeneousSampler:
    """Applies independent per-request SamplingParams across a continuous batch."""

    def sample_batch(
        self,
        last_token_logits: Sequence[torch.Tensor],
        params_list: Sequence[SamplingParams],
        generator: Optional[torch.Generator] = None,
    ) -> List[int]:
        if len(last_token_logits) != len(params_list):
            raise ValueError("logits and params_list length mismatch")
        return [
            sample_single_logits(logits, params, generator=generator)
            for logits, params in zip(last_token_logits, params_list)
        ]

from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F

from llm_serve.model import PagedTransformer


@dataclass(frozen=True)
class SpeculativeStats:
    output_token_ids: List[int]
    target_forward_calls: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / float(self.draft_tokens_proposed)

    @property
    def tokens_per_target_step(self) -> float:
        return len(self.output_token_ids) / float(max(1, self.target_forward_calls))


def _probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        idx = int(torch.argmax(logits).item())
        one_hot = torch.zeros_like(logits, dtype=torch.float32)
        one_hot[idx] = 1.0
        return one_hot
    return F.softmax(logits.float() / float(temperature), dim=-1)


@torch.no_grad()
def speculative_generate(
    target_model: PagedTransformer,
    draft_model: PagedTransformer,
    prompt_token_ids: List[int],
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> SpeculativeStats:
    """Generates `max_new_tokens` using Leviathan et al. (ICML 2023) Speculative Decoding."""
    if gamma <= 0:
        raise ValueError("speculative lookahead gamma must be >= 1")

    target_model.eval()
    draft_model.eval()
    device = next(target_model.parameters()).device

    seq = list(prompt_token_ids)
    generated: List[int] = []
    target_calls = 0
    proposed = 0
    accepted = 0

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        step_gamma = min(gamma, remaining)

        # 1. Fast Draft model proposes `step_gamma` candidate tokens + stores q_i distributions
        draft_seq = list(seq)
        draft_tokens: List[int] = []
        draft_probs_list: List[torch.Tensor] = []

        for _ in range(step_gamma):
            d_logits = draft_model.forward_standalone(
                torch.tensor([draft_seq], device=device, dtype=torch.long)
            )[0, -1]
            q_probs = _probs_from_logits(d_logits, temperature)
            tok = int(torch.multinomial(q_probs, num_samples=1, generator=generator).item())
            draft_tokens.append(tok)
            draft_probs_list.append(q_probs)
            draft_seq.append(tok)

        proposed += step_gamma

        # 2. Heavy Target model scores [seq + draft_tokens] in ONE parallel forward call!
        candidate_input = torch.tensor([seq + draft_tokens], device=device, dtype=torch.long)
        t_logits_all = target_model.forward_standalone(candidate_input)[0]
        target_calls += 1

        # Target logits at position (len(seq) - 1 + i) predict draft_tokens[i]
        base_pos = len(seq) - 1
        all_accepted = True

        for i in range(step_gamma):
            cand_tok = draft_tokens[i]
            p_probs = _probs_from_logits(t_logits_all[base_pos + i], temperature)
            q_probs = draft_probs_list[i]

            p_val = float(p_probs[cand_tok].item())
            q_val = max(float(q_probs[cand_tok].item()), 1e-12)

            if temperature <= 0.0:
                accept = p_val >= 1.0 - 1e-6
            else:
                r = float(torch.rand(1, generator=generator, device=device).item())
                accept = r <= min(1.0, p_val / q_val)

            if accept:
                seq.append(cand_tok)
                generated.append(cand_tok)
                accepted += 1
                if eos_token_id is not None and cand_tok == eos_token_id:
                    return SpeculativeStats(generated, target_calls, proposed, accepted)
                if len(generated) >= max_new_tokens:
                    return SpeculativeStats(generated, target_calls, proposed, accepted)
            else:
                # Sample exact residual distribution norm(max(0, p - q))
                residual = torch.clamp(p_probs - q_probs, min=0.0)
                res_sum = float(residual.sum().item())
                if res_sum > 1e-12:
                    residual = residual / res_sum
                else:
                    residual = p_probs
                replacement = int(
                    torch.multinomial(residual, num_samples=1, generator=generator).item()
                )
                seq.append(replacement)
                generated.append(replacement)
                all_accepted = False
                if eos_token_id is not None and replacement == eos_token_id:
                    return SpeculativeStats(generated, target_calls, proposed, accepted)
                break

        # 3. Bonus token! If all `step_gamma` draft tokens were accepted and room remains,
        # sample one extra token from Target's final position `base_pos + step_gamma` for FREE!
        if all_accepted and len(generated) < max_new_tokens:
            bonus_probs = _probs_from_logits(t_logits_all[base_pos + step_gamma], temperature)
            bonus_tok = int(
                torch.multinomial(bonus_probs, num_samples=1, generator=generator).item()
            )
            seq.append(bonus_tok)
            generated.append(bonus_tok)
            if eos_token_id is not None and bonus_tok == eos_token_id:
                return SpeculativeStats(generated, target_calls, proposed, accepted)

    return SpeculativeStats(generated, target_calls, proposed, accepted)

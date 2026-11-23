import copy
import torch
from llm_serve.config import EngineConfig
from llm_serve.model import PagedTransformer
from llm_serve.speculative import speculative_generate


def _greedy_target_baseline(model: PagedTransformer, prompt: list[int], n_new: int) -> list[int]:
    seq = list(prompt)
    out: list[int] = []
    for _ in range(n_new):
        logits = model.forward_standalone(torch.tensor([seq]))[0, -1]
        tok = int(torch.argmax(logits).item())
        seq.append(tok)
        out.append(tok)
    return out


def test_speculative_decoding_exact_greedy_parity_and_step_compression():
    torch.manual_seed(42)
    t_cfg = EngineConfig(vocab_size=64, dim=32, n_layers=3, n_heads=4, n_kv_heads=2, hidden_dim=64)
    target = PagedTransformer(t_cfg).eval()

    prompt = [1, 5, 9, 13]
    expected_tokens = _greedy_target_baseline(target, prompt, n_new=12)

    # Case 1: High-agreement draft model -> 100% acceptance, generating gamma+1 (5) tokens per Target step!
    aligned_draft = copy.deepcopy(target)
    stats_aligned = speculative_generate(
        target, aligned_draft, prompt, max_new_tokens=12, gamma=4, temperature=0.0
    )
    assert stats_aligned.output_token_ids == expected_tokens
    assert stats_aligned.target_forward_calls <= 3  # 12 tokens in only 3 Target calls (4x fewer)!
    assert stats_aligned.tokens_per_target_step >= 4.0

    # Case 2: Completely mismatched random draft model -> STILL produces 100% bit-identical Target output!
    d_cfg = EngineConfig(vocab_size=64, dim=16, n_layers=1, n_heads=2, n_kv_heads=2, hidden_dim=32)
    random_draft = PagedTransformer(d_cfg).eval()
    stats_mismatched = speculative_generate(
        target, random_draft, prompt, max_new_tokens=12, gamma=4, temperature=0.0
    )
    assert stats_mismatched.output_token_ids == expected_tokens

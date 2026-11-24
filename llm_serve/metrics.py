from dataclasses import dataclass
from typing import Sequence
import numpy as np

from llm_serve.engine import LLMEngine
from llm_serve.request import SequenceRequest


@dataclass(frozen=True)
class ServingMetricsReport:
    num_requests: int
    total_output_tokens: int
    total_engine_steps: int
    mean_ttft_steps: float
    p95_ttft_steps: float
    mean_tpot_steps: float
    p95_e2e_steps: float
    throughput_tokens_per_step: float
    peak_kv_utilization: float
    prefix_cache_hit_rate: float
    preemptions: int


def summarize_engine_metrics(
    engine: LLMEngine,
    completed: Sequence[SequenceRequest] | None = None,
) -> ServingMetricsReport:
    reqs = list(completed) if completed is not None else list(engine.requests.values())
    finished = [r for r in reqs if r.first_token_step is not None and r.finish_step is not None]
    if not finished:
        return ServingMetricsReport(0, 0, engine.current_step, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    ttfts = []
    tpots = []
    e2es = []
    total_out = 0

    for r in finished:
        assert r.first_token_step is not None and r.finish_step is not None
        ttft = float(r.first_token_step - r.arrival_step)
        e2e = float(r.finish_step - r.arrival_step)
        decode_span = float(r.finish_step - r.first_token_step)
        tpot = decode_span / float(max(1, r.num_new_tokens - 1)) if r.num_new_tokens > 1 else 0.0

        ttfts.append(ttft)
        tpots.append(tpot)
        e2es.append(e2e)
        total_out += r.num_new_tokens

    steps = max(1, engine.current_step)
    return ServingMetricsReport(
        num_requests=len(finished),
        total_output_tokens=total_out,
        total_engine_steps=engine.current_step,
        mean_ttft_steps=float(np.mean(ttfts)),
        p95_ttft_steps=float(np.percentile(ttfts, 95)),
        mean_tpot_steps=float(np.mean(tpots)),
        p95_e2e_steps=float(np.percentile(e2es, 95)),
        throughput_tokens_per_step=total_out / float(steps),
        peak_kv_utilization=engine.peak_kv_utilization,
        prefix_cache_hit_rate=engine.block_manager.hit_rate,
        preemptions=engine.preemption_count,
    )

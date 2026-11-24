# llm-serve

A vLLM / SGLang-style High-Throughput LLM Inference Serving Engine built from scratch in Python and PyTorch. Implements **PagedAttention** non-contiguous KV-block management, **Copy-on-Write (`CoW`)** sequence forking, **Automatic Prefix Caching (`APC`)** with chained Blake2b hashing and LRU eviction, iteration-level **Continuous Batching** with block preemption, **Chunked Prefill**, and **Speculative Decoding** with exact rejection sampling.

---

## Repository Structure

```text
llm-serve/
├── LICENSE                         # MIT License
├── README.md                       # Architecture guide, systems references & benchmarks
├── pyproject.toml                  # Package metadata & pytest configuration
├── llm_serve/
│   ├── __init__.py                 # Public module exports
│   ├── config.py                   # EngineConfig (block_size, num_gpu_blocks, GQA heads)
│   ├── request.py                  # SequenceRequest lifecycle & SamplingParams
│   ├── allocator.py                # O(1) physical KV-block pool with ref_count tracking
│   ├── page_table.py               # Logical-to-physical BlockTableManager & Copy-on-Write
│   ├── prefix_cache.py             # Chained Blake2b Automatic Prefix Caching & LRU pool
│   ├── paged_attn.py               # PagedKVCacheArena & non-contiguous GQA kernel
│   ├── scheduler.py                # ContinuousBatchScheduler, preemption & chunked prefill
│   ├── model.py                    # Ragged-batch PagedTransformer (RMSNorm, RoPE, SwiGLU)
│   ├── sampler.py                  # Per-request heterogeneous sampler (temp, top-k, top-p)
│   ├── speculative.py              # Draft-Target speculative decoding & rejection sampling
│   ├── engine.py                   # End-to-end LLMEngine serving loop
│   ├── metrics.py                  # TTFT, TPOT, p95 E2E latency & KV occupancy telemetry
│   └── cli.py                      # CLI benchmarks (bench-batching, bench-prefix, demo-speculative)
└── tests/
    ├── test_request.py             # Request state transitions & EOS termination
    ├── test_allocator.py           # Reference counting, free-list recycling & exhaustion
    ├── test_page_table.py          # Zero-copy sequence fork & Copy-on-Write block cloning
    ├── test_prefix_cache.py        # Shared system-prompt hits & LRU eviction under pressure
    ├── test_paged_attn.py          # Scrambled physical block parity vs contiguous GQA
    ├── test_scheduler.py           # Continuous slot reclamation & FCFS victim preemption
    ├── test_chunked_prefill.py     # Token-budgeted prefill interleaved with 1-token decode
    ├── test_model.py               # Ragged mixed prefill+decode parity vs standalone pass
    ├── test_sampler.py             # Mixed greedy and nucleus batch sampling tests
    ├── test_speculative.py         # Exact Target parity & gamma+1 step compression
    ├── test_engine.py              # Mid-flight dynamic request arrival & completion
    ├── test_metrics.py             # TTFT, TPOT & prefix hit rate calculation tests
    └── test_serving_parity.py      # End-to-end Static vs Continuous vs Speculative parity
```

---

## `PagedAttention` & Iteration-Level Serving Architecture

```text
  ┌───────────────────────────────────────────────────────────────────┐
  │ 1. Iteration-Level Continuous Batching & Chunked Prefill          │
  │                                                                   │
  │   Incoming Prompts ──► [ Waiting / Preempted Queue (FCFS) ]       │
  │                                     │                             │
  │                                     ▼  (Token Budget Slice)       │
  │   Active Decodes  ───► [ Ragged Forward Step: Prefill + Decode ]  │
  │   Finished (EOS)  ◄───   Immediate Block Reclamation on Step t    │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ 2. Virtual-Memory Page Tables, Copy-on-Write & Prefix Cache (APC) │
  │                                                                   │
  │   Req A Logical [0, 1] ──┬──► Phys Block #7 (ref=2, Blake2b Hash) │
  │   Req B Logical [0, 1] ──┘    (Shared System Prompt: Zero Copy!)  │
  │        │                                                          │
  │        └──► Tail Mutation Triggers CoW Clone ─► Phys Block #19    │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ 3. PagedKVCacheArena & Draft-Target Speculative Decoding          │
  │   • Global Pool: (n_layers, num_blocks, block_size, H_kv, D_h)    │
  │   • Speculative: Draft proposes gamma tokens -> 1 Target Pass     │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Core Serving Mechanisms & Systems References

| Mechanism | Core Systems Principle | Foundational Paper |
| :--- | :--- | :--- |
| **`Continuous Batching`** | Iteration-level slot scheduling reclaiming finished sequences on `EOS` | Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) |
| **`PagedAttention` & `CoW`** | Virtual-memory page tables mapping logical blocks to physical KV frames | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) |
| **`Prefix Caching (APC)`** | Chained `Blake2b(parent_hash, block_tokens)` KV reuse with LRU eviction | Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024) |
| **`Chunked Prefill`** | Slices long prompt prefills into bounded token budgets alongside decodes | Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI 2024) |
| **`Speculative Decoding`** | Accepts draft token with `min(1, p(x) / q(x))` + residual `norm(max(0, p - q))` | Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023) |

---

## Serving Throughput & Prefix Caching Benchmarks

### 1. Static Batching vs. Continuous Batching (`python3 -m llm_serve.cli bench-batching`)

Evaluated across `12` skewed-length requests (mixing short `4–6` token answers with `28–32` token generations, `max_batch_seqs = 4`):

| Scheduler Mode | Total Engine Steps | Throughput (Tokens / Step) | Active Batch Slot Utilization | Relative Throughput |
| :--- | ---: | ---: | ---: | :--- |
| **Static Batching (`B = 4`)** | 90 steps | 1.49 tok/step | 37.2% (62.8% idle padding) | 1.00x (Baseline) |
| **Continuous Batching + `PagedAttention`** | **43 steps** | **3.12 tok/step** | **77.9%** | **2.09x higher throughput** |

### 2. Automatic Prefix Caching (`python3 -m llm_serve.cli bench-prefix`)

- **8 requests sharing a 64-token system/RAG prefix:** APC reuses all `8` full KV blocks across requests `2..8` (`87.5%` block cache hit rate), cutting computed prefill tokens from `528` down to **`80` (`84.8%` prefill compute saved)**.

---

## Quickstart

### 1. Run All 13 Unit & End-to-End Parity Test Suites

```bash
source .venv/bin/activate
pytest -v
```

### 2. Run Continuous Batching, Prefix Cache & Speculative Decoding Benchmarks

```bash
python3 -m llm_serve.cli bench-batching
python3 -m llm_serve.cli bench-prefix
python3 -m llm_serve.cli demo-speculative
```

---

## 15-Day Incremental Engineering Roadmap (Month 5, Days 136–150)

- [x] **Day 136:** Initialize `llm_serve` package, `EngineConfig`, `SequenceRequest` state machine, and 15-day roadmap
- [x] **Day 137:** Implement physical `KVBlock` pool and reference-counted `BlockAllocator` (`llm_serve/allocator.py`)
- [x] **Day 138:** Implement `BlockTableManager` mapping logical sequence blocks with Copy-on-Write (`llm_serve/page_table.py`)
- [x] **Day 139:** Implement Blake2b chained block hashing and LRU Automatic Prefix Caching (`llm_serve/prefix_cache.py`)
- [x] **Day 140:** Implement `PagedAttention` KV-cache arena and non-contiguous block gather kernel (`llm_serve/paged_attn.py`)
- [x] **Day 141:** Implement iteration-level Continuous Batching scheduler with preemption (`llm_serve/scheduler.py`)
- [x] **Day 142:** Implement token-budgeted chunked prefill (`tests/test_chunked_prefill.py`)
- [x] **Day 143:** Build `PagedTransformer` causal LM executing prefill and paged decode (`llm_serve/model.py`)
- [x] **Day 144:** Implement vectorized heterogeneous per-request sampler (`llm_serve/sampler.py`)
- [x] **Day 145:** Implement Draft-Target speculative decoding with exact rejection sampling (`llm_serve/speculative.py`)
- [x] **Day 146:** Assemble `LLMEngine` orchestrating arrivals, continuous batching, and prefix caching (`llm_serve/engine.py`)
- [x] **Day 147:** Implement serving telemetry tracker (`TTFT`, `TPOT`, KV utilization, prefix hit rate) (`llm_serve/metrics.py`)
- [x] **Day 148:** Add end-to-end equivalence tests proving Static vs Continuous vs Speculative parity (`tests/test_serving_parity.py`)
- [x] **Day 149:** Build `llm-serve` CLI (`bench-batching`, `bench-prefix`, `demo-speculative`) (`llm_serve/cli.py`)
- [x] **Day 150:** Complete `PagedAttention` & Continuous Batching architecture guide and throughput benchmarks

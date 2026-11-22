# llm-serve

A vLLM / SGLang-style High-Throughput LLM Inference Serving Engine built from scratch in Python and PyTorch. Implements PagedAttention** non-contiguous KV-block allocation, Copy-on-Write (`CoW`), Automatic Prefix Caching (`APC`), iteration-level Continuous Batching with block preemption, Chunked Prefill, and Speculative Decoding.

## 15-Day Incremental Engineering Roadmap (Month 5, Days 136–150)

- [x] **Day 136:** Initialize `llm_serve` package, `EngineConfig`, `SequenceRequest` state machine, and 15-day roadmap
- [ ] **Day 137:** Implement physical `KVBlock` pool and reference-counted `BlockAllocator` (`llm_serve/allocator.py`)
- [ ] **Day 138:** Implement `BlockTableManager` mapping logical sequence blocks with Copy-on-Write (`llm_serve/page_table.py`)
- [ ] **Day 139:** Implement Blake2b chained block hashing and LRU Automatic Prefix Caching (`APC`)
- [ ] **Day 140:** Implement `PagedAttention` KV-cache arena and non-contiguous block gather kernel (`llm_serve/paged_attn.py`)
- [ ] **Day 141:** Implement iteration-level Continuous Batching scheduler with preemption (`llm_serve/scheduler.py`)
- [ ] **Day 142:** Implement token-budgeted chunked prefill (`llm_serve/scheduler.py`)
- [ ] **Day 143:** Build `PagedTransformer` causal LM executing prefill and paged decode (`llm_serve/model.py`)
- [ ] **Day 144:** Implement vectorized heterogeneous per-request sampler (`llm_serve/sampler.py`)
- [ ] **Day 145:** Implement Draft-Target speculative decoding with exact rejection sampling (`llm_serve/speculative.py`)
- [ ] **Day 146:** Assemble `LLMEngine` orchestrating arrivals, continuous batching, and prefix caching (`llm_serve/engine.py`)
- [ ] **Day 147:** Implement serving telemetry tracker (`TTFT`, `TPOT`, KV utilization, prefix hit rate) (`llm_serve/metrics.py`)
- [ ] **Day 148:** Add end-to-end equivalence tests proving Static vs Continuous vs Speculative parity
- [ ] **Day 149:** Build `llm-serve` CLI (`bench-batching`, `bench-prefix`, `demo-speculative`)
- [ ] **Day 150:** Complete PagedAttention & Continuous Batching architecture guide and throughput benchmarks

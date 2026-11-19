# Controlled Real-Model LoRA KV-Cache Sharing Experiment

This directory contains experimental reports and raw measurements from the controlled real-model experiments evaluating prefix cache reuse and adapter isolation on an AWS EC2 `g5.2xlarge` instance (NVIDIA A10G 24GB VRAM).

## Experiment Overview

In multi-tenant LLM serving with Low-Rank Adaptation (LoRA), prompt prefixes frequently share common instructions, schemas, or system prompts across different adapters. Under standard vLLM serving, prefix caching isolates KV caches by adapter identity to preserve inference correctness. This experiment evaluates the trade-offs in TTFT latency, cache hit rates, and output correctness when forcing cross-adapter KV-cache sharing.

### Artifacts

- [`kv_sharing_comparison.md`](kv_sharing_comparison.md): Comprehensive report with hardware specs, Prometheus metrics breakdown, TTFT measurements, output text comparisons, and mathematical analysis.
- [`kv_sharing_native.jsonl`](kv_sharing_native.jsonl): Raw per-request metrics and generated outputs under **Condition A (Native vLLM)**.
- [`kv_sharing_forced.jsonl`](kv_sharing_forced.jsonl): Raw per-request metrics and generated outputs under **Condition B (Experimental Forced Sharing Patch)**.

## Summary of Results

| Condition | Request | Adapter | Prompt / Compl Tok | Cache Hit Delta | TTFT (ms) | Finish Reason | Output Correctness |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Native** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.99 ms | `length` | Valid coding instruction steps |
| **Native** | 2 | `customer_support` | 148 / 1 | **0.0** | 60.74 ms | `stop` (EOS) | Valid customer support prompt response |
| **Native** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.73 ms** | `length` | Exact match with Request 1 |
| **Forced** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.78 ms | `length` | Valid coding output |
| **Forced** | 2 | `customer_support` | 148 / 100 | **144.0** | **40.72 ms** | `length` | **Corrupted**: Markdown code block syntax hallucination |
| **Forced** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.61 ms** | `length` | Output altered due to cache pollution |

## Key Insights & Correctness Notice

1. **Prefill Latency Speedup**: Forcing cache sharing reduced TTFT by **~33.2%** (~20.2 ms saved on a 144-token shared prefix), successfully avoiding forward prefill compute.
2. **Mathematical Invalidation**: LoRA modifies key and value projections via $W + \frac{\alpha}{r} B A$. Reusing KV tensors computed under `magicoder`'s adapter projection matrices forces `customer_support` to attend to representations transformed by an unrelated adapter.
3. **Inference Drift**: Attending to `magicoder` KV states caused `customer_support` to hallucinate markdown code fences (`\n\n```\n...`) instead of terminating with normal text.
4. **Conclusion**: Naive cross-adapter cache reuse is **not a valid production optimization**. Real systems must use adapter-aware prefix factorization or specialized routing.

# Controlled Real-Model LoRA KV-Cache Sharing Experiment

This directory contains experimental reports and raw measurements from the controlled real-model experiments evaluating prefix cache reuse and adapter isolation on an AWS EC2 `g5.2xlarge` instance (NVIDIA A10G 24GB VRAM).

## Experiment Overview

In multi-tenant LLM serving with Low-Rank Adaptation (LoRA), prompt prefixes frequently share common instructions, schemas, or system prompts across different adapters. Under standard vLLM serving, prefix caching isolates KV caches by adapter identity to preserve inference correctness. This experiment evaluates the trade-offs in TTFT latency, cache hit rates, and output correctness when forcing cross-adapter KV-cache sharing.

### Artifacts

- [`kv_sharing_comparison.md`](kv_sharing_comparison.md): Comprehensive report with hardware specs, Prometheus metrics breakdown, TTFT measurements, output text comparisons, and mathematical analysis.
- [`kv_sharing_native.jsonl`](kv_sharing_native.jsonl): Raw per-request metrics and generated outputs under **Condition A (Native vLLM)**.
- [`kv_sharing_forced.jsonl`](kv_sharing_forced.jsonl): Raw per-request metrics and generated outputs under **Condition B (Experimental Forced Sharing Patch)**.
- [`customer_support_validation.jsonl`](customer_support_validation.jsonl): Raw per-request metrics, token IDs, logprobs, and outputs for the 5-run native `customer_support` validation.

## Summary of Results

| Condition | Request | Adapter | Prompt / Compl Tok | Cache Hit Delta | TTFT (ms) | Finish Reason | Output Characterization |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Native** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.99 ms | `length` | Step-by-step instructions |
| **Native** | 2 | `customer_support` | 148 / 1 | **0.0** | 60.74 ms | `stop` (EOS) | Immediate EOS stop (empty string, 0 completion words) |
| **Native** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.73 ms** | `length` | Exact match with Request 1 |
| **Forced** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.78 ms | `length` | Two-agent dialogue format |
| **Forced** | 2 | `customer_support` | 148 / 100 | **144.0** | **40.72 ms** | `length` | Coherent instructions wrapped in markdown code fence (` ``` `) |
| **Forced** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.61 ms** | `length` | Step-by-step instructions (matches Native Request 1/3) |

## Native Customer Support Validation

To investigate the native Request 2 result, a follow-up experiment submitted five identical `customer_support` requests sequentially under native prefix caching:
- **Reproducible EOS Termination**: All five requests immediately emitted EOS (token ID 2) with approximately **95.1% first-token probability** (`logprob = -0.0503`).
- **Consistent Cache-Hit Behavior**: Same-adapter cache hits (144 tokens cached) consistently preserved this exact EOS termination across Requests 2–5 while reducing TTFT from **61.40 ms to ~41.3 ms (~32.8% reduction)**. This demonstrates that native EOS behavior is a reproducible model output under this prompt, not a cache failure.
- **Untested Format Hypothesis**: As an untested possibility, the prompt template may not match the adapter's expected training format, but prompt-formatting variations have not yet been evaluated.

## Key Insights & Correctness Notice

1. **Prefill Latency Speedup**: Forcing cache sharing reduced TTFT by **~33.2%** (~20.2 ms saved on a 144-token shared prefix), successfully avoiding forward prefill compute.
2. **Mathematical Invalidation**: LoRA modifies attention projections via $W + \frac{\alpha}{r} B A$ (here `q_proj` and `v_proj`). Reusing KV tensors computed under `magicoder`'s adapter projection matrices forces `customer_support` to attend to value representations transformed by an unrelated adapter ($V^{(A)} \neq V^{(B)}$).
3. **Behavioral Divergence vs. Semantic Corruption**:
   - Under native serving, `customer_support` reproducibly emits EOS.
   - Under forced cross-adapter reuse, `customer_support` generates a **substantive response** (100 tokens of coherent, step-by-step instructions) wrapped in markdown code fences (` ``` `).
   - The forced output is grammatically fluent and helpful, meaning forced sharing did not produce semantic corruption or garbled text. However, it exhibits clear behavioral divergence by bypassing the native EOS termination and inheriting code-fence formatting from `magicoder`.
4. **Conclusion**: Naive cross-adapter cache reuse is **not a valid production optimization** because it breaks mathematical attention isolation. Real systems must use adapter-aware prefix factorization or specialized routing.

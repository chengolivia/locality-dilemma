# Real-World Workload Experiment: Four-Condition Comparative Audit Report

## 1. Executive Summary & Audit Overview

This report presents the complete, audited four-condition evaluation of the real-world multi-adapter agent replay workload on AWS EC2 (`g5.2xlarge`, NVIDIA A10G 24GB VRAM). The workload consists of 124 requests reconstructed from 20 **TheAgentCompany** workplace tasks executed across 5 LoRA adapters (`cnn`, `drop_explained`, `gsm8k`, `magicoder`, `wikisql`) served on `Mistral-7B-v0.1` (`vLLM 0.33.0`).

All four conditions were executed against clean, independent vLLM server processes starting from empty prefix caches (cold start verified: Request 1 produced exactly 0 cache-hit tokens across all runs).

### The Four Experimental Conditions

1. **Native Interleaved**: Native vLLM cache isolation; requests alternate across tasks/adapters round-robin (121 transitions).
2. **Native Grouped**: Native vLLM cache isolation; requests are clustered by adapter (4 transitions).
3. **Forced Interleaved**: Forced cross-adapter KV sharing patch (`_gen_lora_extra_hash_keys` returns `[]`); requests alternate round-robin (121 transitions).
4. **Forced Grouped**: Forced cross-adapter KV sharing patch; requests clustered by adapter (4 transitions).

---

## 2. Comprehensive Four-Condition Performance Matrix

All metrics are computed directly from the 124 raw request records per JSONL output file without data imputation.

| Metric | Native Interleaved | Native Grouped | Forced Interleaved | Forced Grouped |
|:---|:---:|:---:|:---:|:---:|
| **Output File** | [`runs/adapter_interleaved_20261122_195706.jsonl`](runs/adapter_interleaved_20261122_195706.jsonl) | [`runs/adapter_grouped_20261122_200512.jsonl`](runs/adapter_grouped_20261122_200512.jsonl) | [`runs/forced_adapter_interleaved_20261122_202452.jsonl`](runs/forced_adapter_interleaved_20261122_202452.jsonl) | [`runs/forced_adapter_grouped_20261122_203418.jsonl`](runs/forced_adapter_grouped_20261122_203418.jsonl) |
| **Server Log** | [`runs/vllm_server_interleaved.log`](runs/vllm_server_interleaved.log) | [`runs/vllm_server_grouped.log`](runs/vllm_server_grouped.log) | [`runs/vllm_server_forced_interleaved.log`](runs/vllm_server_forced_interleaved.log) | [`runs/vllm_server_forced_grouped.log`](runs/vllm_server_forced_grouped.log) |
| **Cache Hashing Mode** | Native Isolation | Native Isolation | Forced Cross-Adapter | Forced Cross-Adapter |
| **Adapter Transitions** | 121 | 4 | 121 | 4 |
| **Completed Requests** | **124 / 124 (100%)** | **124 / 124 (100%)** | **124 / 124 (100%)** | **124 / 124 (100%)** |
| **Finish Reasons** | 43 `length`, 81 `stop` | 43 `length`, 81 `stop` | 36 `length`, 88 `stop` | 43 `length`, 81 `stop` |
| **Cold-Start Req 1 Hits** | **0 tokens** (48.09 ms) | **0 tokens** (48.42 ms) | **0 tokens** (48.10 ms) | **0 tokens** (48.14 ms) |
| **Queried Prompt Tokens** | 350,606 | 350,606 | 350,606 | 350,606 |
| **Cache Hit Tokens** | **104,032** | **241,152** | **107,552** | **241,872** |
| **Prefix Cache Hit Rate** | **29.67%** | **68.78%** | **30.68%** | **68.99%** |
| **Mean TTFT** | **603.25 ms** | **275.59 ms** | **574.33 ms** | **294.90 ms** |
| **Median TTFT (p50)** | **248.25 ms** | **186.28 ms** | **248.21 ms** | **186.47 ms** |
| **95th Percentile TTFT (p95)**| **2,087.40 ms** | **710.69 ms** | **2,037.94 ms** | **710.64 ms** |
| **Total Completion Tokens** | 7,711 | 7,711 | 6,852 (-11.1%) | 7,572 (-1.8%) |
| **Sum Generation Time** | 279.50 s | 279.56 s | 248.15 s (-31.35 s) | 272.88 s (-6.68 s) |
| **Sum TTFT (Prefill)** | 74.80 s | 34.17 s | 71.22 s | 36.57 s |
| **Summed Client Elapsed Time** | 356.33 s | 315.81 s | 321.54 s (-34.79 s) | 311.53 s (-4.28 s) |
| **Workload Wall Duration (Runner)** | 358.20 s (~5.97 min) | 317.70 s (~5.29 min) | 323.39 s (~5.39 min) | 313.36 s (~5.22 min) |

---

## 3. Reconciliation of Aggregate and Per-Adapter TTFT

### 3.1 Verified Adapter Populations

A prior reporting discrepancy occurred where the per-adapter request counts were misstated as (14, 24, 24, 34, 28). The actual, audited request counts across the 124-request replay dataset are:

- `cnn`: **14 requests** (11.29% of workload)
- `drop_explained`: **26 requests** (20.97% of workload)
- `gsm8k`: **26 requests** (20.97% of workload)
- `magicoder`: **28 requests** (22.58% of workload)
- `wikisql`: **30 requests** (24.19% of workload)
- **Total: 14 + 26 + 26 + 28 + 30 = 124 requests (100.0%)**

### 3.2 Exact Mathematical Reconciliation

Using the verified request counts, the request-weighted sum of per-adapter means matches the aggregate mean exactly across all four conditions:

$$\text{Mean TTFT}_{\text{aggregate}} = \frac{1}{124} \sum_{i=1}^{5} N_i \cdot \text{Mean TTFT}_i$$

- **Native Interleaved**:
  $$\frac{14(382.1603) + 26(820.4148) + 26(827.9332) + 28(416.5915) + 30(497.7064)}{124} = \frac{74,803.05}{124} = \mathbf{603.2504\text{ ms}}$$
- **Native Grouped**:
  $$\frac{14(195.1887) + 26(336.1797) + 26(348.0887) + 28(218.0125) + 30(251.5077)}{124} = \frac{34,173.20}{124} = \mathbf{275.5903\text{ ms}}$$
- **Forced Interleaved**:
  $$\frac{14(193.4562) + 26(817.6386) + 26(819.3188) + 28(394.5861) + 30(496.6209)}{124} = \frac{71,216.31}{124} = \mathbf{574.3251\text{ ms}}$$
- **Forced Grouped**:
  $$\frac{14(378.1226) + 26(334.9708) + 26(346.1102) + 28(216.0955) + 30(250.5118)}{124} = \frac{36,567.85}{124} = \mathbf{294.9020\text{ ms}}$$

### 3.3 Detailed Per-Adapter Breakdown

| Adapter | Requests | Metric | Native Interleaved | Native Grouped | Forced Interleaved | Forced Grouped |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| `cnn` | 14 | Cache Hit Rate (%)<br>Mean TTFT (ms)<br>Median TTFT (ms)<br>p95 TTFT (ms) | 75.52%<br>382.16 ms<br>127.98 ms<br>1,405.10 ms | 75.52%<br>195.19 ms<br>127.74 ms<br>489.98 ms | 75.96%<br>193.46 ms<br>112.54 ms<br>490.97 ms | 75.52%<br>378.12 ms<br>127.82 ms<br>1,385.88 ms |
| `drop_explained` | 26 | Cache Hit Rate (%)<br>Mean TTFT (ms)<br>Median TTFT (ms)<br>p95 TTFT (ms) | 13.69%<br>820.41 ms<br>525.63 ms<br>2,087.91 ms | 66.31%<br>336.18 ms<br>360.47 ms<br>607.65 ms | 14.08%<br>817.64 ms<br>525.80 ms<br>2,088.65 ms | 66.52%<br>334.97 ms<br>360.79 ms<br>607.69 ms |
| `gsm8k` | 26 | Cache Hit Rate (%)<br>Mean TTFT (ms)<br>Median TTFT (ms)<br>p95 TTFT (ms) | 12.71%<br>827.93 ms<br>552.91 ms<br>2,139.38 ms | 65.13%<br>348.09 ms<br>343.06 ms<br>684.64 ms | 13.67%<br>819.32 ms<br>526.51 ms<br>2,139.83 ms | 65.36%<br>346.11 ms<br>343.48 ms<br>684.62 ms |
| `magicoder` | 28 | Cache Hit Rate (%)<br>Mean TTFT (ms)<br>Median TTFT (ms)<br>p95 TTFT (ms) | 49.95%<br>416.59 ms<br>134.83 ms<br>1,467.59 ms | 75.93%<br>218.01 ms<br>118.88 ms<br>450.49 ms | 52.55%<br>394.59 ms<br>122.74 ms<br>1,412.17 ms | 76.16%<br>216.10 ms<br>118.93 ms<br>450.26 ms |
| `wikisql` | 30 | Cache Hit Rate (%)<br>Mean TTFT (ms)<br>Median TTFT (ms)<br>p95 TTFT (ms) | 25.39%<br>497.71 ms<br>116.65 ms<br>1,966.14 ms | 64.91%<br>251.51 ms<br>112.69 ms<br>733.60 ms | 25.65%<br>496.62 ms<br>114.88 ms<br>1,966.53 ms | 65.17%<br>250.51 ms<br>112.74 ms<br>734.38 ms |

---

## 4. Root-Cause Audit of the CNN TTFT Anomaly

### 4.1 Observed Latency Inversion
In the per-adapter results, CNN exhibited an apparent inversion:
- Native Interleaved: **382.16 ms** vs. Native Grouped: **195.19 ms**
- Forced Interleaved: **193.46 ms** vs. Forced Grouped: **378.12 ms**

### 4.2 Single-Request Isolation
Request-by-request audit of all 14 CNN requests revealed that 13 of the 14 requests performed virtually identically across all four runs (TTFTs between 39.7 ms and 332.1 ms, medians ~127 ms).

The entire ~187 ms divergence in the mean is attributable to a **single outlier request**:
- **Request Identifier**: `pm-projects-analytics:2` (Step 2, prompt length 3,165 tokens, 448 cache hits, 2,717 un-cached prefill tokens).
- **Execution Order**: Sequence 41 in interleaved schedules; Sequence 7 in grouped schedules.

| Condition | Sequence # | Prompt Tokens | Cache Hits | TTFT of `pm-projects-analytics:2` | Server Log JIT Event |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Native Interleaved** | Seq 41 | 3,165 | 448 | **3,397.87 ms** | `WARNING 19:59:00 Triton kernel JIT compilation: _lora_shrink_kernel` |
| **Native Grouped** | Seq 7 | 3,165 | 448 | **784.94 ms** | Pre-compiled / cached kernel hit |
| **Forced Interleaved** | Seq 41 | 3,165 | 448 | **786.14 ms** | Pre-compiled / cached kernel hit |
| **Forced Grouped** | Seq 7 | 3,165 | 448 | **3,343.99 ms** | `WARNING 20:34:45 Triton kernel JIT compilation: _lora_shrink_kernel` |

### 4.3 Technical Mechanism and Run-Order Confound
1. **Compilation Boundary**: vLLM captures CUDA graphs during server warmup only up to 512 tokens (`max_cudagraph_capture_size: 512`, `compile_ranges_endpoints: [2048]`).
2. `pm-projects-analytics:2` is the very first request in the entire 124-request workload whose prompt exceeds 2,000 tokens (3,165 tokens).
3. When the engine encounters this new GEMM batch shape during inference, Triton JIT compiles `_lora_shrink_kernel` and `_lora_expand_kernel` on the fly.
4. Server logs and disk timestamps confirm that in `native_interleaved` (at 19:59:00) and in `forced_grouped` (at 20:34:45), Triton spent ~2.56 seconds compiling `.cubin` binaries on the critical path of prefill:
   $$\Delta \text{TTFT} = 3,344\text{ ms} - 785\text{ ms} \approx 2,559\text{ ms}$$
5. Across a small sample of $N=14$ requests for CNN, a one-time 2,560 ms compilation penalty adds $\frac{2,560\text{ ms}}{14} \approx 182.8\text{ ms}$ directly to the arithmetic mean.

### 4.4 Methodological Implications of Shared On-Disk Triton Cache
The four runs executed in fresh vLLM processes to ensure cold KV caches, but shared the same underlying on-disk Triton compilation cache (`~/.triton/cache/`). This introduced a significant **run-order confound**:
- **Cold Compilation Runs**: `native_interleaved` (the first interleaved run) and `forced_grouped` (which encountered an uncached shape variant) paid the ~2.56-second compilation penalty on `pm-projects-analytics:2`.
- **Warm Kernel Runs**: `native_grouped` and `forced_interleaved` benefited from previously compiled kernels on disk, bypassing the compilation delay (~785 ms vs. ~3,350 ms).
- **Contamination of Aggregate Means**: This compilation artifact contaminates comparisons of aggregate mean TTFT across conditions, creating an artificial ~20 ms inflation in overall mean TTFT for `native_interleaved` and `forced_grouped`.
- **Robustness of Percentiles**: The median (p50) and 95th percentile (p95) metrics are less sensitive to this isolated single-request outlier, although they remain single-run measurements subject to run-to-run noise.
- **Recommended Follow-up Protocol**: For clean benchmarking, future runs should include a standardized warm-up phase that explicitly exercises the relevant prompt lengths and LoRA kernel shapes prior to measurement. The KV cache should still start cold for each measured run, but kernel compilation must not be part of the measured workload.

---

## 5. Correct Technical Explanation of KV Attention Under LoRA

### 5.1 LoRA Architecture in Evaluated Adapters
Inspection of `adapter_config.json` across all five evaluated adapters (`magicoder`, `wikisql`, `gsm8k`, `drop_explained`, `cnn`) confirms:
```json
"target_modules": [
  "q_proj",
  "v_proj"
]
```
The key projection (`k_proj`) is **not** a target module.

### 5.2 Attention Formulation and Error Propagation
For input token representations $X_l$ at transformer layer $l$:
- **Value Projections ($V$)**: Directly contain adapter-specific low-rank updates at every adapted layer:
  $$V_l^{(A)} = X_l \left( W_{V, l} + \frac{\alpha}{r} A_{V, l} B_{V, l} \right)$$
  Because $V_l^{(A)} \neq V_l^{(B)}$, cached value vectors are strictly incompatible across differing adapters.
- **Key Projections ($K$)**: At layer 0 ($l=0$), $X_0$ is the token embedding, and the projection uses solely the base weights $W_{K, 0}$:
  $$K_0 = X_0 W_{K, 0}$$
  Thus, $K_0$ is mathematically identical across adapters.
- **Deeper Layers ($l \ge 1$)**: For layer 1 and beyond, the input hidden state $X_l$ is the output of prior transformer blocks:
  $$X_l = f(X_{l-1}, Q_{l-1}^{(A)}, K_{l-1}, V_{l-1}^{(A)})$$
  Because adapter-dependent updates in $Q$ and $V$ alter intermediate activations, $X_l^{(A)} \neq X_l^{(B)}$ for all $l \ge 1$. Consequently, deeper key projections $K_l^{(A)} = X_l^{(A)} W_{K, l}$ also diverge across adapters despite $W_K$ having no direct LoRA updates.

### 5.3 Mathematical Inequivalence of Forced Sharing
In forced sharing, `_gen_lora_extra_hash_keys()` returns `[]`, causing the prefix cache to match blocks based solely on `(token_ids, block_idx)`. When request $B$ reuses blocks computed by request $A$, it computes attention using $V^{(A)}$ instead of $V^{(B)}$. This breaks mathematical equivalence and introduces numerical corruption into self-attention.

---

## 6. Interpretation of Adapter Locality

### 6.1 Scheduling Transitions vs. Host-to-Device Transfers
- The vLLM server was configured with `--enable-lora --max-loras 5 --max-lora-rank 8`.
- Server startup logs confirm all five adapters were pre-loaded into GPU VRAM before inference began and remained resident throughout the experiment.
- The reported adapter transitions (121 in interleaved vs. 4 in grouped) represent **request-scheduling transitions** (changing which GPU-resident adapter was bound to consecutive requests), **not PCIe host-to-device weight swapping latency**.

### 6.2 Mechanism of Locality Dilemma
In native vLLM, scheduling transitions matter because changing the active adapter invalidates the prefix cache lookup:
1. When requests alternate round-robin (`adapter_interleaved`), same-adapter prefixes cannot be chained across consecutive steps, resulting in high cache turnover and low hit rates (29.67%).
2. When requests are clustered by adapter (`adapter_grouped`), same-adapter temporal locality is preserved, allowing intra-adapter prefix reuse to reach 68.78%.
3. Dynamic adapter eviction and host-to-device PCIe transfer costs are evaluated separately in the synthetic transfer benchmark (`reports/transfer_cost_benchmark.md`), where adapter counts exceed GPU residency limits.

---

## 7. Disentangling TTFT (Prefill) from Wall-Clock (Decode / Generation)

### 7.1 Completion Token and Generation Time Variations

Because forced cross-adapter sharing passes mathematically corrupt KV states into the attention layers, it alters the token generation trajectory and stop-token probabilities:

| Strategy | Finish Reasons | Completion Tokens | Sum Generation Time (Decode) | Sum TTFT (Prefill) | Client Elapsed Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Native Interleaved** | 43 `length`, 81 `stop` | **7,711** | 279.50 s | 74.80 s | 356.33 s |
| **Native Grouped** | 43 `length`, 81 `stop` | **7,711** | 279.56 s | 34.17 s | 315.81 s |
| **Forced Interleaved** | 36 `length`, 88 `stop` | **6,852** (-859 toks) | 248.15 s (-31.35 s) | 71.22 s (-3.58 s) | 321.54 s |
| **Forced Grouped** | 43 `length`, 81 `stop` | **7,572** (-139 toks) | 272.88 s (-6.68 s) | 36.57 s | 311.53 s |

### 7.2 Distinguishing Wall-Duration Metrics
To prevent methodological ambiguity, two distinct wall-duration figures must be separated and not conflated:
1. **Workload Wall Duration**: The total wall-clock duration of the replay execution from runner start to finish, measured by the orchestrator (e.g. 358.20 s in native interleaved vs. 323.39 s in forced interleaved). This encompasses all server response times, client inter-request processing, and HTTP connection management.
2. **Summed Client Elapsed Time**: The arithmetic sum of per-request HTTP round-trip elapsed times recorded by the client (`client_elapsed_ms`), representing net query-serving time excluding orchestrator schedule preparation (e.g. 356.33 s in native interleaved vs. 321.54 s in forced interleaved).

### 7.3 Wall-Clock Attribution and Component Accounting
1. **Native Grouped Speedup**:
   - In native grouped, completion tokens (7,711) and generation decode time (279.56 s vs. 279.50 s) were virtually identical to native interleaved.
   - The entire 40.5-second summed client elapsed speedup ($356.33\text{ s} - 315.81\text{ s} = 40.52\text{ s}$) was driven **100% by prefill acceleration** ($\Delta \text{TTFT} = 74.80\text{ s} - 34.17\text{ s} = 40.63\text{ s}$) from intra-adapter prefix caching.
2. **Forced Interleaved Attribution**:
   - In forced interleaved, summed client elapsed time dropped by **34.79 seconds** relative to native interleaved ($356.33\text{ s} - 321.54\text{ s}$).
   - Decomposition of the savings shows:
     - **Generation-time (decode) reduction**: $279.50\text{ s} - 248.15\text{ s} = \mathbf{31.35\text{ s}}$ ($\frac{31.35}{34.79} = \mathbf{90.1\%}$ of client elapsed reduction). This was caused by generating **859 fewer completion tokens** due to 7 additional requests halting early on EOS.
     - **TTFT (prefill) reduction**: $74.80\text{ s} - 71.22\text{ s} = \mathbf{3.58\text{ s}}$ ($\frac{3.58}{34.79} = \mathbf{10.3\%}$ of client elapsed reduction).
   - **Accounting Note**: The component savings sum to $31.35\text{ s} + 3.58\text{ s} = 34.93\text{ s}$ ($100.4\%$ of the 34.79 s client elapsed delta). This small difference reflects measurement and HTTP/serialization overhead variation between vLLM's internal server-side timer metrics (`generation_time_ms` + `time_to_first_token_ms`) and the client's end-to-end socket round-trip time (`client_elapsed_ms`), rather than an exact mathematical partition.
3. **Metric Scope & Interpretation**:
   - Wall-clock duration must **not** be attributed solely to cache reuse under forced conditions, as early stopping dominates the time delta.
   - **Cache-hit counters directly measure observed prefix reuse; TTFT measures its end-to-end prefill latency impact, subject to runtime confounders** (such as Triton JIT compilation and queueing).

---

## 8. Explicit Conclusions: What This Experiment Establishes and Does Not Establish

### What This Experiment Establishes
1. **Adapter Grouping Solves Intra-Adapter Locality**: Grouping requests by adapter under native isolation achieves a **68.78% prefix cache hit rate** and cuts mean TTFT from **603.25 ms to 275.59 ms** (-54.3%), without compromising mathematical equivalence or output validity.
2. **Structural Cross-Adapter Overlap is Modest in Agent Trajectories**: Under interleaved scheduling, forced sharing increased cache hits by only **+3,520 tokens** (from 29.67% to 30.68%). Beyond initial system prompts and workplace instruction headers, multi-turn agent histories diverge rapidly, limiting the structural prefix overlap between different tasks.
3. **Zero Marginal Value for Cross-Adapter Sharing Once Grouped**: When requests are already grouped by adapter, forced cross-adapter sharing yields virtually zero additional cache benefit (**+720 tokens / +0.21 percentage points**, 68.78% vs. 68.99%).

### What This Experiment Does Not Establish
1. **Does Not Establish Production Viability of Forced Sharing**: Forced cross-adapter KV-cache sharing breaks attention isolation ($V^{(A)} \neq V^{(B)}$) and induces semantic drift (evidenced by shifted stop-token counts). It is strictly an experimental instrument to measure token overlap, not a deployable optimization.
2. **Does Not Measure Host-to-Device Adapter Transfer Latency**: Because all 5 adapters fit concurrently in VRAM (`--max-loras 5`), this workload experiment measures scheduling transitions and KV-cache turnover, not PCIe weight-swapping delays.

# Real-World Workload Experiment: Native Cache Isolation Qualification Report

## 1. Executive Summary & Qualification

This report documents the baseline real-world workload replay experiment conducted on AWS EC2 (`g5.2xlarge`, NVIDIA A10G 24GB VRAM) evaluating 124 reconstructed multi-turn agent requests from 20 **TheAgentCompany** tasks across 5 LoRA adapters (`magicoder`, `wikisql`, `gsm8k`, `drop_explained`, `cnn`) on `Mistral-7B-v0.1`.

> [!IMPORTANT]
> **Experimental Context & Qualification**:
> These benchmark runs were executed strictly under **native vLLM adapter-specific KV-cache isolation** (`_gen_lora_extra_hash_keys` returning `[("lora", name, path)]`).
>
> In native vLLM serving, prefix cache blocks are keyed by both token IDs and adapter identity. Cache reuse occurs **exclusively between requests using compatible, same-adapter KV states**.
>
> Therefore, this experiment measures **adapter-aware request scheduling efficiency under native cache isolation**; it does **not** evaluate cross-adapter KV-cache reuse, nor does it test the cross-adapter KV sharing tradeoff central to the project's original locality dilemma hypothesis.

---

## 2. Experimental Setup

- **Hardware**: AWS EC2 `g5.2xlarge` (1x NVIDIA A10G 24GB VRAM, AMD EPYC 7R32, 32GB RAM).
- **Engine**: vLLM 0.33.0 (Commit `93a3e66d`, v1 core engine) with native cache hashing.
- **Base Model**: `Mistral-7B-v0.1` (`./models/base`, `dtype: half`, `max_model_len: 8192`).
- **Active Adapters**: 5 Predibase LoRAs ($r=8, \alpha=16$, targeting `q_proj`, `v_proj`):
  `magicoder` (coding), `wikisql` (SQL), `gsm8k` (math), `drop_explained` (QA), `cnn` (summarization).
- **Serving Configuration**: `--enable-lora --max-loras 5 --max-lora-rank 8 --enable-prefix-caching --enable-per-request-metrics --gpu-memory-utilization 0.85`.
- **Workload**: `data/agent_replay_20tasks/replay_8192.jsonl` (124 requests, 350,606 total prompt tokens, 20 tasks).
- **Sampling**: `temperature=0.0`, `max_tokens=128`.
- **Cold-Start Isolation**: Each strategy was executed against a fresh, independent vLLM server process with an empty prefix cache. Request 1 in both runs recorded exactly 0 cache-hit tokens (`prefix_cache_hits_delta = 0.0`).

---

## 3. Results Comparison: Interleaved vs. Grouped Scheduling

| Metric | `adapter_interleaved` | `adapter_grouped` | Delta / Impact |
|:---|:---:|:---:|:---:|
| **Output File** | [`reports/real-workloads/runs/adapter_interleaved_20261122_195706.jsonl`](runs/adapter_interleaved_20261122_195706.jsonl) | [`reports/real-workloads/runs/adapter_grouped_20261122_200512.jsonl`](runs/adapter_grouped_20261122_200512.jsonl) | — |
| **Server Log** | [`reports/real-workloads/runs/vllm_server_interleaved.log`](runs/vllm_server_interleaved.log) | [`reports/real-workloads/runs/vllm_server_grouped.log`](runs/vllm_server_grouped.log) | — |
| **Completed Requests** | **124 / 124 (100%)** | **124 / 124 (100%)** | 0 failed requests |
| **Finish Reasons** | 43 `length`, 81 `stop` | 43 `length`, 81 `stop` | Exact parity |
| **Adapter Transitions** | **121** | **4** | **-96.7% transitions** |
| **Total Prompt Tokens** | 350,606 | 350,606 | Identical input volume |
| **Total Completion Tokens** | 7,711 | 7,711 | Identical output volume |
| **Prometheus Queried Tokens** | 350,606 | 350,606 | — |
| **Prometheus Cached Tokens Hit** | **104,032** | **241,152** | **+137,120 cached tokens (+131.8%)** |
| **Aggregate Prefix Hit Rate** | **29.67%** | **68.78%** | **+39.11 percentage points** |
| **Mean TTFT** | **603.25 ms** | **275.59 ms** | **-54.3% latency reduction** |
| **Median TTFT (p50)** | **248.25 ms** | **186.28 ms** | **-24.96% latency reduction** |
| **95th Percentile TTFT (p95)** | **2,087.40 ms** | **710.69 ms** | **-66.0% tail latency reduction** |
| **Summed Client Elapsed Time** | 356.33 s | 315.81 s | -40.52 s saved (-11.4%) |
| **Workload Wall Duration (Runner)** | 358.20 s (~5.97 min) | 317.70 s (~5.29 min) | -40.50 s saved (-11.3%) |

---

## 4. Key Findings & Scope Interpretation

1. **Scheduling Impact Under Native Isolation**:
   - In `adapter_interleaved`, requests alternate across tasks and adapters round-robin (121 transitions). Same-adapter prefix blocks are subjected to cache turnover while other adapters are queried, limiting hit rates to **29.67%**.
   - In `adapter_grouped`, requests are clustered by adapter before switching (4 transitions across the 5 adapters). This maximizes temporal locality for each adapter's cached prefixes, raising the hit rate to **68.78%** and cutting mean TTFT from **603.25 ms to 275.59 ms**.

2. **Absence of Cross-Adapter Reuse**:
   - Because native vLLM hashes `("lora", name, path)` into every KV block, **zero tokens were reused across differing adapters** in either run.
   - All 241,152 hit tokens in `adapter_grouped` and 104,032 hit tokens in `adapter_interleaved` represent intra-adapter prefix reuse between successive steps of the same tasks/adapter.
   - These measurements establish the performance ceiling of scheduling optimizations within standard isolated architectures. Evaluating cross-adapter prefix reuse requires explicit forced-sharing testing.

3. **Scheduling Transitions vs. Host-to-Device Transfers**:
   - All five adapters fit concurrently in GPU memory (`--max-loras 5`) and remained resident throughout inference.
   - The reported adapter transitions (121 in interleaved vs. 4 in grouped) represent **request scheduling transitions** (which active adapter is selected per request), not PCIe host-to-device weight swapping latency.
   - Actual adapter dynamic transfer costs are isolated and measured in the synthetic transfer benchmark (`reports/transfer_cost_benchmark.md`).

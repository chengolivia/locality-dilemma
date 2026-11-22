# Supplementary Experiment: Constrained Adapter Residency (`--max-loras 1`)

**Workload**: 124 requests across 20 tasks (`data/agent_replay_20tasks/replay_8192.jsonl`, 350,606 prompt tokens)  
**Model & Adapters**: Mistral-7B-v0.1 with 5 LoRA adapters (`cnn`, `drop_explained`, `gsm8k`, `magicoder`, `wikisql`; $r=8$, targeting `q_proj` and `v_proj`)  
**Hardware**: AWS EC2 `g5.2xlarge` (NVIDIA A10G 24GB, PCIe Gen4 x8 current, Gen4 x16 max capability)  
**Output Directory**: `reports/real-workloads/max_loras_1/`

---

## 1. Experimental Setup & Runtime Audit

This supplemental experiment evaluates how restricting active adapter capacity to a single GPU slot (`--max-loras 1`) interacts with request scheduling and prefix caching under native KV isolation.

### 1.1 Resolved Configuration & Adapter Lifecycle
- **GPU Slots vs. CPU Cache**: `--max-loras 1` sets GPU active slots to 1 (`lora_slots = 1`). Because `--max-cpu-loras` was omitted, vLLM's `LoRAConfig` defaulted `max_cpu_loras = max_loras = 1`. Both GPU active capacity and registered CPU cache capacity resolved to **1**.
- **Adapter Lifecycle**:
  - *Same-Adapter Requests*: The active slot is reused directly without weight transfer (`activate_adapter` returns `False`).
  - *Adapter Transitions*: Evicts the active adapter from GPU slot 0. Because CPU cache capacity is 1, `add_adapter()` first reads the ~14 MB safetensors checkpoint from the OS filesystem/page cache into CPU memory (`_load_adapter()`), then streams weights to the GPU via 64 module-level asynchronous copies (`set_lora()`).
- **Telemetry Limits**: Stock vLLM 0.33.0 records **no per-transfer event, duration, or byte telemetry** (`prom_lora_keys: []`). The reported **121** (interleaved) and **4** (grouped) counts are **request-sequence adapter transitions** (`adapter_id[i] != adapter_id[i-1]`). Inferred activation opportunities (including the initial load) are **122** and **5**.

---

## 2. Reconciled Results: `max_loras=1` vs. `max_loras=5`

All metrics are computed directly from raw per-request JSONL records. Prompt queries (350,606 tokens) form the exact denominator for cache hit rates.

| Metric | `max_loras=1` Interleaved | `max_loras=1` Grouped | `max_loras=5` Interleaved | `max_loras=5` Grouped |
|---|:---:|:---:|:---:|:---:|
| **Raw JSONL Log** | `max_loras_1_..._interleaved_...jsonl` | `max_loras_1_..._grouped_...jsonl` | `adapter_interleaved_...jsonl` | `adapter_grouped_...jsonl` |
| **Requests ($N$)** | 124 | 124 | 124 | 124 |
| **Adapter Transitions** | 121 | 4 | 121 | 4 |
| **Inferred Activations** | 122 (121 + 1) | 5 (4 + 1) | 5 (initial only) | 5 (initial only) |
| **GPU Slots / CPU Cache** | 1 slot / 1 entry | 1 slot / 1 entry | 5 slots / 5 entries | 5 slots / 5 entries |
| **Prompt Tokens Queried** | 350,606 | 350,606 | 350,606 | 350,606 |
| **Cache Hit Tokens** | 108,096 | 241,152 | 104,032 | 241,152 |
| **Prefix Cache Hit Rate** | **30.83%** | **68.78%** | **29.67%** | **68.78%** |
| **Mean TTFT** | 585.55 ms | 274.37 ms | 603.25 ms | 275.59 ms |
| **Median TTFT ($p50$)** | **262.85 ms** | **184.83 ms** | **248.25 ms** | **186.28 ms** |
| **95th Percentile TTFT ($p95$)** | 2,041.16 ms | 706.20 ms | 2,087.40 ms | 710.69 ms |
| **Summed Client Elapsed** | 353.69 s | 317.22 s | 356.33 s | 315.81 s |

*Per-adapter breakdown (14 cnn, 26 drop_explained, 26 gsm8k, 28 magicoder, 30 wikisql = 124 total) reconciles with the aggregate means: $\frac{1}{124}\sum N_a \mu_a = 585.55\text{ ms}$ (interleaved) and $274.37\text{ ms}$ (grouped).*

---

## 3. Analysis: What `max_loras=1` Tells Us vs. Previous Experiments

Comparing `max_loras=1` against `max_loras=5` reveals **very little difference in TTFT or cache hit rates**, clarifying the relationship between adapter residency and prefix caching:

1. **Little Difference Between 1 Slot and 5 Slots**:
   - In `adapter_grouped`, performance is essentially identical: median TTFT is **184.83 ms** with 1 slot vs. **186.28 ms** with 5 slots, while cache hit rate is unchanged at **68.78%**.
   - In `adapter_interleaved`, median TTFT increases by only **$\approx 14.6\text{ ms}$** (**262.85 ms** vs. **248.25 ms**, a ~5.9% difference), while cache hit rate remains $\approx 30\%$.
   - Aggregate mean TTFT (585.55 ms vs. 603.25 ms) remains dominated by prefill token lengths and single-run Triton JIT compilation variance (request 41 incurred a cold compilation penalty of 3,397.87 ms in the earlier run vs. 797.35 ms here).

2. **The Upshot: Grouping Preserves Prefix-Cache Locality While Mitigating Adapter-Residency Pressure**:
   - In our earlier synthetic benchmarks, rank-8 adapter swapping over PCIe Gen4 x8 was measured at only **$\approx 3.08\text{ ms}$**, whereas prompt prefill on long contexts took hundreds to thousands of milliseconds.
   - This experiment confirms that dynamic adapter swapping (even when compounded by CPU checkpoint deserialization under `max_cpu_loras=1`) incurs only a modest latency penalty ($\approx 14.6\text{ ms}$ median composite delta) compared to the latency difference observed between grouped and interleaved schedules (~263 ms vs. ~185 ms median, and >2,000 ms p95).
   - Furthermore, grouping requests by adapter collapses transitions from 121 to 4, amortizing adapter swapping costs almost completely; as a result, we observe no meaningful TTFT difference between one and five slots under grouped scheduling in this run.
   - **Conclusion**: Under the tested conditions, grouping requests by adapter simultaneously preserves same-adapter prefix reuse and mitigates adapter-residency pressure, allowing even a single-slot configuration (`max_loras=1`) to achieve serving efficiency comparable to five resident slots without requiring cross-adapter KV sharing.

---

## 4. Limitations

- **Measured vs. Inferred**: TTFT, prompt/completion tokens, client elapsed time, and prefix cache queries/hits are directly measured. GPU slot evictions, checkpoint deserialization calls, and inferred activation counts (122 vs. 5) are derived from source-code control flow, not engine telemetry.
- **Composite vs. Isolated Transfer**: The $\approx 14.6\text{ ms}$ interleaved median delta is an end-to-end composite measurement (including CPU checkpoint loading, Python layer iteration, and scheduling jitter), distinct from the isolated $\approx 3.08\text{ ms}$ PCIe transfer microbenchmark.
- **Workload Scope**: These conclusions reflect sequential batch-size-1 execution on Mistral-7B ($r=8$) and should not be assumed to hold universally under high concurrency or continuous batching regimes.

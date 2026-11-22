# Synthetic Profiling Reports & Benchmarks

This directory indexes empirical measurements, artifacts, and analytical reports across the synthetic microbenchmarking phase of **The Locality Dilemma: Dissecting Adapter Warmth vs. KV Prefix Reuse on Real Accelerators**.

The synthetic benchmark sweeps across:
1. **LoRA Host-to-Device Transfer Latency ($T_{\text{swap}}$)** across LoRA ranks $r \in [8, 16, 32, 64, 128]$.
2. **Forward-Pass Prefill Compute Latency ($T_{\text{prefill}}$)** across context lengths $L \in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]$ tokens (Mistral-7B, float16, batch size 1).
3. **Pareto Frontier & Crossover Analysis**: Evaluating the decision boundary $\Delta = \Delta T_{\text{prefill}} - T_{\text{swap}}$ between prefix retention and adapter retention.

---

## Datasets and Experimental Artifacts

### 1. NVIDIA A10G Microbenchmarks (AWS EC2 `g5.2xlarge`)
- **Primary Sweep ($L \in [128, 8192]$)**: Preserved in [`a10g/`](a10g/):
  - [`a10g/swap.csv`](a10g/swap.csv): Measured $T_{\text{swap}}$ across ranks 8–128.
  - [`a10g/prefill.csv`](a10g/prefill.csv): Measured $T_{\text{prefill}}$ across sequence lengths 128–8192.
  - [`a10g/timings.csv`](a10g/timings.csv): Cross-joined 35-point grid timings.
  - [`a10g/analysis/summary.csv`](a10g/analysis/summary.csv): Aggregated median costs, deltas, and policy winners.
  - [`a10g/analysis/break_even.csv`](a10g/analysis/break_even.csv): Interpolated break-even tokens per rank.
  - [`a10g/analysis/latency_heatmap.png`](a10g/analysis/latency_heatmap.png): Visual heatmap of the Pareto space.
  - [`a10g/environment.json`](a10g/environment.json): Machine hardware, driver, CUDA, and environment metadata.
  - [`a10g/benchmark_run.log`](a10g/benchmark_run.log): Complete standard execution log.

- **Short-Context Crossover Analysis ($L \in [8, 64]$ Extended to $8192$)**: Preserved in [`a10g/crossover/`](a10g/crossover/):
  - [`a10g/crossover/README.md`](a10g/crossover/README.md): Detailed report distinguishing direct measurements, derived estimates, and unmeasured serving dynamics.
  - [`a10g/crossover/consolidated_crossover_analysis.csv`](a10g/crossover/consolidated_crossover_analysis.csv): Consolidated 22-row dataset across all 11 context lengths (8–8192) for ranks 64 and 128.
  - [`a10g/crossover/crossover_combined.csv`](a10g/crossover/crossover_combined.csv): Paired measurements and marginal calculations.
  - [`a10g/crossover/crossover_analysis_plot.png`](a10g/crossover/crossover_analysis_plot.png): Dual-perspective visualization comparing direct standalone measurements vs. derived marginal estimates.
  - [`a10g/crossover/prefill.csv`](a10g/crossover/prefill.csv): Raw short-context prefill latency measurements ($L \in [8, 16, 32, 64]$).
  - [`a10g/crossover/swap.csv`](a10g/crossover/swap.csv): Active swap baselines for ranks 64 and 128.

### 2. Historical NVIDIA H100 PCIe Baseline (PSC Bridges-2, 2026-10-24)
- Preserved in [`h100-synthetic-20261024/`](../../h100-synthetic-20261024/):
  - [`h100-synthetic-20261024/swap.csv`](../../h100-synthetic-20261024/swap.csv): Raw $T_{\text{swap}}$ measurements across ranks 8–128.
  - [`h100-synthetic-20261024/prefill.csv`](../../h100-synthetic-20261024/prefill.csv): Raw $T_{\text{prefill}}$ measurements across sequence lengths 128–8192.
  - [`h100-synthetic-20261024/timings.csv`](../../h100-synthetic-20261024/timings.csv): Cross-joined 35-point grid paired timings.
  - [`h100-synthetic-20261024/analysis/summary.csv`](../../h100-synthetic-20261024/analysis/summary.csv): Delta latency and winner designation.
  - [`h100-synthetic-20261024/analysis/break_even.csv`](../../h100-synthetic-20261024/analysis/break_even.csv): Interpolated break-even context length per rank.
  - [`h100-synthetic-20261024/analysis/latency_heatmap.png`](../../h100-synthetic-20261024/analysis/latency_heatmap.png): Visual heatmap of the Pareto space.

### 3. Cross-Hardware Comparative Report
- **Master Comparison**: [`a10g_h100_comparison.md`](a10g_h100_comparison.md): Comprehensive architectural analysis, hardware tables, and latency comparisons between NVIDIA A10G and NVIDIA H100 PCIe.

---

## Three Levels of Evidence

To prevent conflation between measured microbenchmarks and serving-level scheduling claims, the project categorizes synthetic evidence into three distinct tiers:

1. **Level 1: Direct Measurements (Empirical Findings)**
   - Standalone forward prefill latency ($T_{\text{prefill}}$) measured with empty KV cache across $L \in [8, 8192]$ on Mistral-7B ($B=1$, FP16).
   - Adapter transfer latency ($T_{\text{swap}}$) measured via pinned-memory DMA across PCIe Gen4 x8 for ranks 8–128.
   - **Direct Observation**: At all 11 directly measured context lengths ($8 \le L \le 8192$), directly measured prefill latency ($T_{\text{prefill}} \ge 43.77\text{ ms}$) strictly exceeds adapter swap latency ($11.90\text{ ms}$ for rank 64, $23.35\text{ ms}$ for rank 128). **No crossover is directly observed anywhere in the measured spectrum under the benchmark's native total-prefill definition.**

2. **Level 2: Derived Estimates (Mathematical Formulations & Interpolations)**
   - Marginal prefill compute savings ($\Delta T_{\text{marginal}}(L) = T(L) - T(8)$) evaluating incremental token compute beyond the minimal measured prompt ($L=8$, $43.77\text{ ms}$).
   - **Measured Crossover Interval**: Under this marginal formulation, incremental compute savings crosses adapter transfer latency strictly within the **measured interval $L \in [128, 256]$ tokens** ($10.81\text{ ms} < T_{\text{swap}} < 38.64\text{ ms}$).
   - **Interpolated Crossover Estimates**: Linear interpolation across the $[128, 256]$ interval yields estimated crossover points at **$\approx 133.0\text{ tokens}$ for Rank 64** and **$\approx 185.7\text{ tokens}$ for Rank 128**.
   - *Labeling Distinction*: These values (133.0 and 185.7 tokens) are **mathematical interpolation estimates**, NOT experimentally observed data points.

3. **Level 3: Not Yet Measured (Realistic Serving Workloads & Open Questions)**
   - Actual KV-cache savings versus adapter-switch costs in realistic multi-tenant serving workloads (e.g., vLLM continuous batching, chunked prefill, inter-request scheduling).
   - Suffix-on-prefix query execution (where an uncached user query runs on top of a cached prefix).
   - Asynchronous adapter transfer pipelining overlapping on secondary CUDA streams with ongoing GPU token generation.
   - KV cache memory fragmentation, eviction policies, and VRAM capacity contention under multi-adapter co-location.

---

## Architectural Comparison Summary: A10G vs. H100

| Metric | NVIDIA H100 PCIe (Bridges-2) | NVIDIA A10G (AWS EC2 `g5.2xlarge`) | Relative Difference | Architectural Factor |
|---|---|---|---|---|
| **Interconnect** | PCIe Gen5 x16 (64 GB/s) | PCIe Gen4 x8 (15.75 GB/s) | 4.0x line rate | Cloud hypervisor assigns 8 Gen4 lanes |
| **Rank 64 Swap ($T_{\text{swap}}$)** | 6.37 ms (21.09 GB/s) | **11.90 ms** (11.28 GB/s) | **1.87x slower on A10G** | Saturated PCIe DMA transfer bandwidth |
| **Rank 128 Swap ($T_{\text{swap}}$)**| 11.44 ms (23.47 GB/s) | **23.35 ms** (11.50 GB/s) | **2.04x slower on A10G** | Saturated PCIe DMA transfer bandwidth |
| **Prefill ($L=128$)** | 14.26 ms (8,979 tok/s) | **54.57 ms** (2,345 tok/s) | **3.83x slower on A10G** | Memory bandwidth & kernel launch overhead |
| **Prefill ($L=1024$)** | 36.11 ms (28,356 tok/s) | **293.31 ms** (3,491 tok/s) | **8.12x slower on A10G** | 4th-gen Hopper Tensor Cores vs. Ampere |
| **Prefill ($L=8192$)** | 533.71 ms (15,349 tok/s) | **3,187.12 ms** (2,570 tok/s)| **5.97x slower on A10G** | Quadratic attention computation intensity |
| **Direct Policy Winner** | `prefix-first` (35 / 35 points) | `prefix-first` (35 / 35 points) | Identical policy winner | Prefill compute slowdown > Swap slowdown |

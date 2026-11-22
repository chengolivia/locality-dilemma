# Comparative Hardware Benchmark: NVIDIA A10G vs. NVIDIA H100 PCIe

**Project**: The Locality Dilemma: Dissecting Adapter Warmth vs. KV Prefix Reuse on Real Accelerators  
**Course**: CMU 11-868: LLM Systems  
**Date**: November 22, 2026  
**Artifacts**:
- NVIDIA A10G Results: [`reports/synthetic/a10g/`](a10g/)
- NVIDIA H100 Results: [`h100-synthetic-20261024/`](../../h100-synthetic-20261024/)

---

## 1. Executive Summary

This report presents empirical microbenchmark measurements comparing the **NVIDIA A10G Tensor Core GPU** (AWS EC2 `g5.2xlarge`) against our historical baseline on the **NVIDIA H100 PCIe GPU** (Pittsburgh Supercomputing Center Bridges-2).

The experiments evaluate the fundamental trade-off of the **Locality Dilemma**:
$$\Delta = \Delta T_{\text{prefill}} - T_{\text{swap}}$$
where $\Delta T_{\text{prefill}}$ is the compute time saved by reusing a cached KV prefix, and $T_{\text{swap}}$ is the latency incurred by transferring a LoRA adapter's weights across the host-to-device PCIe bus.

### Key Findings

1. **Adapter Swapping Latency ($T_{\text{swap}}$)**:
   - A10G is **1.67x to 2.04x slower** than H100 for swapping LoRA weights (3.08 ms vs. 1.76 ms at rank 8; 23.35 ms vs. 11.44 ms at rank 128).
   - This difference directly reflects the bus interconnect: H100 operated over **PCIe Gen5 x16** (effective transfer bandwidth ~23.5 GB/s for 256 MB), whereas the AWS `g5.2xlarge` instance exposes **PCIe Gen4 x8** (effective transfer bandwidth ~11.5 GB/s).
2. **Prefill Compute Latency ($\Delta T_{\text{prefill}}$)**:
   - A10G is **3.83x to 8.12x slower** than H100 in forward-pass prefill compute (54.57 ms vs. 14.26 ms at $L=128$; 3,187.12 ms vs. 533.71 ms at $L=8192$).
   - The latency gap is widest in the compute-bound regime ($L \in [512, 2048]$), where Hopper's 4th-generation Tensor Cores achieve ~8x higher throughput than Ampere.
3. **The Locality Frontier ($\Delta$)**:
   - Because prefill compute scales significantly worse on A10G (up to 8.1x slower) than adapter transfer does (up to 2.0x slower), the computational penalty of re-prefilling a prompt on A10G is **drastically higher** relative to the adapter swap cost.
   - Consequently, **`prefix-first` cache retention dominates even more aggressively on A10G than on H100**. For all 35 evaluated combinations ($5 \text{ ranks} \times 7 \text{ context lengths}$), $\Delta > 0$, yielding 0 crossing points (break-even prefix tokens $< 128$ tokens for all ranks).

---

## 2. Hardware and Environment Specifications

| Attribute | NVIDIA A10G (AWS EC2) | NVIDIA H100 PCIe (PSC Bridges-2) | Ratio / Difference |
|---|---|---|---|
| **GPU Architecture** | Ampere (GA102GL) | Hopper (GH100) | Generational leap (2 generations) |
| **Compute Capability** | 8.6 | 9.0 | New DPX, TMA, 4th-gen Tensor Cores |
| **Total VRAM** | 24 GB GDDR6 (23,028 MiB) | 80 GB HBM3 | 3.48x capacity, HBM3 vs GDDR6 |
| **Memory Bandwidth** | 600 GB/s (theoretical) | 2,000 GB/s (theoretical) | 3.33x higher memory bandwidth |
| **Dense FP16 Tensor TFLOPS** | ~62.5 TFLOPS (dense) / 125 (sparse) | ~756 TFLOPS (dense) / 1,513 (sparse) | ~12.1x higher dense compute |
| **Interconnect Bus** | PCIe Gen4 x8 (EC2 virtualized) | PCIe Gen5 x16 (HPC bare-metal/SR-IOV) | 4x theoretical link throughput |
| **Theoretical PCIe Bandwidth**| 15.75 GB/s (unidirectional) | 64.00 GB/s (unidirectional) | 4.06x theoretical bus capacity |
| **TDP / Power Limit** | 300 W (A10G) | 350 W (H100 PCIe) | 1.17x higher TDP |
| **Host System** | AMD EPYC 7R32 (8 vCPUs), 30 GiB RAM | AMD EPYC 7763 (64 cores), 512 GiB RAM | Cloud VM vs. HPC Supercomputer |
| **OS & Kernel** | Ubuntu 24.04 LTS (Linux 6.8.0) | Rocky Linux / RHEL (Linux 5.14/6.x) | Linux 6.x |
| **CUDA / Driver** | CUDA 13.0 / Driver 595.91.07 | CUDA 12.4+ / Driver 550.x+ | Modern CUDA runtimes |
| **PyTorch Version** | 2.14.0+cu130 | 2.4.0+ / 2.6.0+ | PyTorch 2.x SDPA enabled |

> [!NOTE]
> On AWS EC2 `g5.2xlarge`, the hypervisor assigns 8 PCIe Gen4 lanes to the GPU (`Gen4 x8`). While the physical A10G silicon supports up to Gen4 x16, the effective host-to-device streaming rate is bounded by the 8-lane allocation (~15.75 GB/s theoretical, ~11.5 GB/s measured).

---

## 3. Benchmark Workload & Methodology

Both benchmark suites executed identically across the standardized parameter grid without hardware-specific parameter tuning:

- **LoRA Adapter Ranks ($r$)**: `8, 16, 32, 64, 128`
- **Context Lengths ($L$)**: `128, 256, 512, 1024, 2048, 4096, 8192` tokens
- **Batch Size**: `1`
- **Simulated Base Architecture**: Mistral-7B (32 layers, hidden dimension 4,096, intermediate dimension 14,336, 32 attention heads, 8 KV heads, vocabulary size 32,000)
- **Precision**: `float16`
- **LoRA Projections**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (4 modules per layer = 128 weight matrices)
- **Repetition & Warmup**:
  - Adapter Swap ($T_{\text{swap}}$): 10 warmup iterations + 100 timed iterations per rank using `torch.cuda.Event` on pinned host DMA memory.
  - Forward Prefill ($\Delta T_{\text{prefill}}$): 10 warmup iterations + 50 timed iterations per context length with KV cache generation enabled.

---

## 4. Benchmark Results: Adapter Transfer Latency ($T_{\text{swap}}$)

Host-to-Device transfer latency measured via pinned memory asynchronous DMA:

| Rank ($r$) | Total Params | Adapter Size | H100 Median ($T_{\text{swap}}$) | A10G Median ($T_{\text{swap}}$) | Latency Delta | Ratio (A10G / H100) | H100 Eff. BW | A10G Eff. BW |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **8** | 8,388,608 | 16.0 MB | 1.757 ms | **3.079 ms** | +1.322 ms | **1.75x** | 9.55 GB/s | 5.45 GB/s |
| **16** | 16,777,216 | 32.0 MB | 2.389 ms | **3.979 ms** | +1.590 ms | **1.67x** | 14.05 GB/s | 8.43 GB/s |
| **32** | 33,554,432 | 64.0 MB | 3.636 ms | **6.201 ms** | +2.565 ms | **1.71x** | 18.46 GB/s | 10.82 GB/s |
| **64** | 67,108,864 | 128.0 MB | 6.365 ms | **11.897 ms** | +5.532 ms | **1.87x** | 21.09 GB/s | 11.28 GB/s |
| **128** | 134,217,728 | 256.0 MB | 11.436 ms | **23.352 ms** | +11.916 ms | **2.04x** | 23.47 GB/s | 11.50 GB/s |

### Statistical Distributions on A10G

| Rank | Mean ± Std (ms) | Min / Max (ms) | P25 (ms) | P75 (ms) | P95 (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 3.104 ± 0.140 | 3.04 / 4.40 | 3.072 | 3.087 | 3.175 |
| 16 | 3.985 ± 0.034 | 3.94 / 4.19 | 3.973 | 3.987 | 4.018 |
| 32 | 6.206 ± 0.022 | 6.17 / 6.30 | 6.194 | 6.214 | 6.249 |
| 64 | 11.912 ± 0.058 | 11.86 / 12.12 | 11.882 | 11.909 | 12.068 |
| 128 | 23.367 ± 0.044 | 23.32 / 23.53 | 23.332 | 23.392 | 23.436 |

### Analysis of $T_{\text{swap}}$
- **Fixed Overhead vs. Bus Saturation**: At small sizes (16 MB), transfer time is dominated by fixed kernel dispatch, CUDA stream synchronization, and memory handle acquisition (~1.5–2.0 ms base latency). At 256 MB (rank 128), bus saturation is reached: the A10G plateaus at **11.50 GB/s** (73% of PCIe Gen4 x8 theoretical), whereas the H100 reaches **23.47 GB/s**.
- **Variance**: Timing variance is remarkably low across all ranks on both machines ($\sigma < 0.15\text{ ms}$), demonstrating high repeatability for DMA transfers on pinned memory.

---

## 5. Benchmark Results: Forward Prefill Latency ($\Delta T_{\text{prefill}}$)

Forward-pass compute latency across context lengths for Mistral-7B ($B=1$, FP16):

| Context ($L$) | H100 Median ($\Delta T_{\text{prefill}}$) | A10G Median ($\Delta T_{\text{prefill}}$) | Latency Delta | Ratio (A10G / H100) | H100 Tok/s | A10G Tok/s | H100 VRAM | A10G VRAM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **128** | 14.26 ms | **54.57 ms** | +40.31 ms | **3.83x** | 8,979 | 2,345 | 13,876.6 MB | 13,852.7 MB |
| **256** | 15.29 ms | **82.40 ms** | +67.11 ms | **5.39x** | 16,744 | 3,107 | 13,905.6 MB | 13,881.8 MB |
| **512** | 20.97 ms | **151.89 ms** | +130.92 ms | **7.24x** | 24,411 | 3,371 | 13,966.8 MB | 13,942.9 MB |
| **1024** | 36.11 ms | **293.31 ms** | +257.20 ms | **8.12x** | 28,356 | 3,491 | 14,089.0 MB | 14,065.2 MB |
| **2048** | 72.13 ms | **582.03 ms** | +509.90 ms | **8.07x** | 28,395 | 3,519 | 14,333.5 MB | 14,309.7 MB |
| **4096** | 198.29 ms | **1,336.29 ms** | +1,138.00 ms | **6.74x** | 20,656 | 3,065 | 14,854.6 MB | 14,814.7 MB |
| **8192** | 533.71 ms | **3,187.12 ms** | +2,653.41 ms | **5.97x** | 15,349 | 2,570 | 15,928.6 MB | 15,840.8 MB |

### Statistical Distributions on A10G

| Context ($L$) | Mean ± Std (ms) | Min / Max (ms) | P25 (ms) | P75 (ms) | P95 (ms) | Peak VRAM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 54.68 ± 0.22 | 54.37 / 55.15 | 54.50 | 54.83 | 55.12 | 13,852.7 MB |
| 256 | 82.41 ± 0.15 | 82.11 / 82.79 | 82.30 | 82.50 | 82.65 | 13,881.8 MB |
| 512 | 151.90 ± 0.11 | 151.79 / 152.55 | 151.84 | 151.91 | 152.01 | 13,942.9 MB |
| 1024 | 293.33 ± 0.14 | 293.15 / 294.21 | 293.27 | 293.36 | 293.42 | 14,065.2 MB |
| 2048 | 582.04 ± 0.08 | 581.89 / 582.21 | 581.98 | 582.11 | 582.15 | 14,309.7 MB |
| 4096 | 1336.29 ± 0.16 | 1335.96 / 1336.70 | 1336.18 | 1336.40 | 1336.57 | 14,814.7 MB |
| 8192 | 3187.21 ± 0.38 | 3186.83 / 3189.28 | 3187.00 | 3187.28 | 3187.64 | 15,840.8 MB |

### Analysis of $\Delta T_{\text{prefill}}$
- **Peak Compute Disparity**: At $L \in [1024, 2048]$, H100 processes forward tokens at ~28,395 tokens/sec compared to A10G's ~3,519 tokens/sec—an **8.1x speedup**. This closely aligns with the ~12x raw theoretical tensor FLOP advantage of Hopper over Ampere when factoring in memory bandwidth and kernel overhead.
- **Memory Footprint Consistency**: Peak VRAM allocation matches within **0.17% to 0.55%** across all context lengths (13.85 GB vs 13.88 GB at $L=128$; 15.84 GB vs 15.93 GB at $L=8192$), confirming that identical tensor allocations, KV cache layouts, and layer architectures were evaluated.
- **Feasibility on A10G**: Despite A10G having 24 GB VRAM (vs. H100's 80 GB), all configurations up to $L=8192$ completed without Out-Of-Memory (OOM) errors, utilizing a maximum of 15.84 GB (~69% of the A10G's 23,028 MiB capacity).

---

## 6. Pareto Frontier Comparison: Dissecting the Locality Dilemma

The Locality Dilemma asks whether a system serving multi-LoRA requests should prioritize **Adapter Locality** (keeping an adapter warm on GPU and recomputing the KV prefix) or **Prefix Locality** (reusing a shared KV prefix and paying the cost to swap adapter weights).

The decision boundary is governed by the sign of $\Delta_{\text{ms}} = \Delta T_{\text{prefill}} - T_{\text{swap}}$:
- $\Delta_{\text{ms}} > 0 \implies$ **Prefix-First Wins** (time saved by caching prefix exceeds swap cost).
- $\Delta_{\text{ms}} < 0 \implies$ **Adapter-First Wins** (swapping adapter costs more than recomputing prefix).

### Full 35-Point Grid Comparison ($\Delta_{\text{ms}}$ in milliseconds)

| Rank ($r$) | Metric | $L=128$ | $L=256$ | $L=512$ | $L=1024$ | $L=2048$ | $L=4096$ | $L=8192$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$r=8$** | **H100 $\Delta_{\text{ms}}$** | +12.50 | +13.53 | +19.22 | +34.36 | +70.37 | +196.54 | +531.95 |
| | **A10G $\Delta_{\text{ms}}$** | **+51.49** | **+79.32** | **+148.81** | **+290.23** | **+578.96** | **+1,333.21** | **+3,184.04** |
| **$r=16$** | **H100 $\Delta_{\text{ms}}$** | +11.87 | +12.90 | +18.59 | +33.72 | +69.74 | +195.90 | +531.32 |
| | **A10G $\Delta_{\text{ms}}$** | **+50.59** | **+78.42** | **+147.91** | **+289.33** | **+578.06** | **+1,332.31** | **+3,183.15** |
| **$r=32$** | **H100 $\Delta_{\text{ms}}$** | +10.62 | +11.65 | +17.34 | +32.48 | +68.49 | +194.66 | +530.07 |
| | **A10G $\Delta_{\text{ms}}$** | **+48.37** | **+76.20** | **+145.69** | **+287.11** | **+575.83** | **+1,330.09** | **+3,180.92** |
| **$r=64$** | **H100 $\Delta_{\text{ms}}$** | +7.89 | +8.92 | +14.61 | +29.75 | +65.76 | +191.93 | +527.35 |
| | **A10G $\Delta_{\text{ms}}$** | **+42.68** | **+70.51** | **+139.99** | **+281.42** | **+570.14** | **+1,324.39** | **+3,175.23** |
| **$r=128$** | **H100 $\Delta_{\text{ms}}$** | +2.82 | +3.85 | +9.54 | +24.68 | +60.69 | +186.86 | +522.27 |
| | **A10G $\Delta_{\text{ms}}$** | **+31.22** | **+59.05** | **+128.54** | **+269.96** | **+558.68** | **+1,312.93** | **+3,163.77** |

### Winner Designation and Break-Even Point

- **H100 Result**: Winner is `prefix-first` for **35 of 35** points. Break-even threshold $< 128$ tokens.
- **A10G Result**: Winner is `prefix-first` for **35 of 35** points. Break-even threshold $< 128$ tokens.

> [!IMPORTANT]
> **Why does Prefix-First dominate even more decisively on A10G?**  
> While the A10G has slower PCIe transfer (~2x slower adapter swap), its prefill compute is **3.8x to 8.1x slower**. Therefore, saving prefill compute via KV reuse yields **vastly higher absolute time savings** on A10G than on H100.
> For example, at rank 128 and $L=128$:
> - On H100: Recomputing prefill takes 14.26 ms; swapping adapter takes 11.44 ms ($\Delta = +2.82\text{ ms}$).
> - On A10G: Recomputing prefill takes 54.57 ms; swapping adapter takes 23.35 ms ($\Delta = +31.22\text{ ms}$).
> Thus, the penalty of sacrificing prefix cache is **11x higher** on A10G than on H100 at this tight boundary point!

---

## 7. Methodological Caveats and Environmental Differences

1. **PCIe Topologies (Bare-Metal HPC vs Cloud Hypervisor)**:
   - On PSC Bridges-2, the H100 was hosted in a physical HPC server with full PCIe Gen5 x16 lanes routed directly to the root complex.
   - On AWS EC2 `g5.2xlarge`, Nitro hypervisor virtualization slices the physical GPU's PCIe bus to 8 lanes (`Gen4 x8`). While realistic for production cloud inference deployments, this artificially halves the maximum PCIe bandwidth compared to a full `Gen4 x16` slot (~11.5 GB/s vs. potential ~23 GB/s).
2. **SDPA Kernel Differences**:
   - PyTorch 2.14 on CUDA 13.0 was utilized on A10G, leveraging the FlashAttention / Cutlass SDPA backend optimized for Ampere Tensor Cores. H100 benefited from Hopper-specific FP8/FP16 TMA and asynchronous tensor copy pipelines.
3. **Synthetic Isolated Microbenchmark vs. Multi-Tenant Server**:
   - These measurements represent isolated microbenchmarks ($B=1$, single CUDA stream, uncontended PCIe DMA). In an actual serving engine like vLLM under high concurrency, adapter swap latencies may be partially masked by asynchronous pre-loading on secondary CUDA streams, or increased due to PCIe bus contention with incoming token streaming.

---

## 8. Summary of Completed Deliverables

All deliverables requested for Phase 4 and Phase 5 are preserved in the repository:

1. **A10G Swap Latency Dataset**: [`reports/synthetic/a10g/swap.csv`](a10g/swap.csv)
2. **A10G Prefill Latency Dataset**: [`reports/synthetic/a10g/prefill.csv`](a10g/prefill.csv)
3. **A10G Paired Timings Grid**: [`reports/synthetic/a10g/timings.csv`](a10g/timings.csv)
4. **Pareto Frontier Summary**: [`reports/synthetic/a10g/analysis/summary.csv`](a10g/analysis/summary.csv)
5. **Break-Even Interpolation**: [`reports/synthetic/a10g/analysis/break_even.csv`](a10g/analysis/break_even.csv)
6. **Pareto Heatmap Visualization**: [`reports/synthetic/a10g/analysis/latency_heatmap.png`](a10g/analysis/latency_heatmap.png)
7. **Execution Environment Metadata**: [`reports/synthetic/a10g/environment.json`](a10g/environment.json)
8. **Complete Execution Log**: [`reports/synthetic/a10g/benchmark_run.log`](a10g/benchmark_run.log)

---

## 9. Recommended Next Steps

1. **Sub-128 Token Sweep (Completed on A10G)**:
   - The follow-up sweep across short context lengths $L \in [8, 16, 32, 64]$ tokens has been executed and documented in [`reports/synthetic/a10g/crossover/README.md`](a10g/crossover/README.md).
   - Under direct measurements of standalone forward passes, zero crossovers were observed ($T_{\text{prefill}} \ge 43.8\text{ ms} > T_{\text{swap}}$ across all tested lengths).
   - Under derived marginal prefill calculations relative to $L=8$, the crossover interval is bounded within $L \in [128, 256]$ tokens, with linear interpolated estimates of $\approx 133.0\text{ tokens}$ (Rank 64) and $\approx 185.7\text{ tokens}$ (Rank 128).
2. **Batch Size Sensitivity ($B > 1$)**: Evaluate whether batched prefill amortizes forward compute overhead faster than batched adapter execution, shifting the frontier.
3. **Real-Model Validation**: Compare these synthetic findings against the real-model adapter KV cache benchmarks documented in `reports/adapter-kv-cache/kv_sharing_comparison.md`.

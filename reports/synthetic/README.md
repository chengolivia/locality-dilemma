# Synthetic Profiling Reports & Benchmarks

This directory indexes results and analysis from synthetic microbenchmarking experiments measuring:
1. **LoRA Host-to-Device Transfer Latency ($T_{\text{swap}}$)** across LoRA ranks $r \in [8, 16, 32, 64, 128]$.
2. **Forward-Pass Prefill Compute Latency ($\Delta T_{\text{prefill}}$)** across context lengths $L \in [128, 256, 512, 1024, 2048, 4096, 8192]$.
3. **Pareto Frontier Analysis**: The crossover point where prefill time saved via KV-prefix caching matches adapter weight transfer latency.

## Datasets and Historical Artifacts

- **PSC H100 Benchmark Run (2026-10-24)**: Full measurements on an NVIDIA H100 PCIe accelerator (Pittsburgh Supercomputing Center) are preserved in [`h100-synthetic-20261024/`](../../h100-synthetic-20261024/):
  - [`h100-synthetic-20261024/swap.csv`](../../h100-synthetic-20261024/swap.csv): Raw $T_{\text{swap}}$ measurements across ranks.
  - [`h100-synthetic-20261024/prefill.csv`](../../h100-synthetic-20261024/prefill.csv): Raw $\Delta T_{\text{prefill}}$ measurements across sequence lengths.
  - [`h100-synthetic-20261024/timings.csv`](../../h100-synthetic-20261024/timings.csv): Cross-joined 35-point grid paired timings.
  - [`h100-synthetic-20261024/analysis/summary.csv`](../../h100-synthetic-20261024/analysis/summary.csv): Delta latency ($\Delta T_{\text{prefill}} - T_{\text{swap}}$) and winning policy (`prefix-first` vs `adapter-first`).
  - [`h100-synthetic-20261024/analysis/break_even.csv`](../../h100-synthetic-20261024/analysis/break_even.csv): Interpolated break-even context length per rank.
  - [`h100-synthetic-20261024/analysis/latency_heatmap.png`](../../h100-synthetic-20261024/analysis/latency_heatmap.png): Visual heatmap of the Pareto trade-off space.

## Key Findings (H100 Synthetic Baseline)

- On high-end accelerators with high PCIe bandwidth (PCIe Gen5 x16, effective bandwidth ~50–55 GB/s):
  - Adapter swap overhead ($T_{\text{swap}}$) is small: ~1.76 ms for rank 8 up to ~11.8 ms for rank 128.
  - Forward prefill compute scales with quadratic/flash-attention intensity, exceeding adapter swap latency even for relatively short prefixes ($L \ge 128$ tokens).
  - Consequently, `prefix-first` cache retention dominates on high-bandwidth architectures when context exceeds a few hundred tokens.
- On lower-bandwidth or memory-constrained accelerators (e.g. PCIe Gen4 x16 or PCIe Gen3), the break-even frontier shifts toward longer prefix lengths.

# Historical H100 Synthetic Benchmark Results (2026-10-24)

This directory preserves the raw measurements and analysis artifacts collected on an NVIDIA H100 PCIe accelerator (80GB VRAM) on Pittsburgh Supercomputing Center (PSC) Bridges-2.

## Artifact Inventory

- `swap.csv`: Host-to-Device transfer latencies ($T_{\text{swap}}$) measured via `torch.cuda.Event` for LoRA ranks $r \in [8, 16, 32, 64, 128]$.
- `prefill.csv`: Forward-pass prefill compute latencies ($\Delta T_{\text{prefill}}$) measured for sequence lengths $L \in [128, 256, 512, 1024, 2048, 4096, 8192]$ with batch size 1.
- `timings.csv`: Paired measurements across all 35 combinations ($5 \text{ ranks} \times 7 \text{ context lengths}$), with columns: `prefix_len,lora_rank,prefill_saved_ms,swap_ms`.
- `analysis/summary.csv`: Aggregated median costs, delta $(\Delta T_{\text{prefill}} - T_{\text{swap}})$, and winner designation (`prefix-first` vs `adapter-first`).
- `analysis/break_even.csv`: Interpolated break-even token length per rank where prefill savings balance swap costs.
- `analysis/latency_heatmap.png`: Heatmap visualization of the Pareto frontier.

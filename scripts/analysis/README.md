# Analysis & Visualization Scripts

This directory contains analysis scripts for computing Pareto frontiers on synthetic benchmarks and evaluating real-world multi-adapter workload replays.

## Scripts

### 1. `analyze_results.py`
Ingests paired measurements (`timings.csv`) containing `prefix_len`, `lora_rank`, `prefill_saved_ms`, and `swap_ms`.

Computes:
- **Delta Latency**: $\Delta_{\text{ms}} = \text{prefill\_saved\_ms} - \text{swap\_ms}$.
  - Positive delta ($\Delta > 0$): Prefill compute savings exceed adapter swap cost; favors **`prefix-first`** scheduling.
  - Negative delta ($\Delta < 0$): Adapter swap cost exceeds prefill compute savings; favors **`adapter-first`** scheduling.
- **Break-Even Frontier**: Computes interpolated crossover prefix lengths per LoRA rank.
- **Visual Heatmap**: Generates a 2D heatmap plot of the latency delta across ranks and sequence lengths.

```bash
python scripts/analysis/analyze_results.py \
    --input results/timings.csv \
    --output-dir results/analysis
```

### 2. `analyze_real_workload.py`
Ingests JSONL measurement files from real-world workload replays (`adapter_interleaved` vs. `adapter_grouped`).

Computes:
- **Overall Latency Summaries**: TTFT (mean, median, p95), ITL, generation time, and throughput (tok/s).
- **Cache Hit Telemetry**: Prometheus prefix-cache query deltas, hit deltas, token counts, and aggregate hit rates.
- **Per-Adapter Breakdown**: Per-adapter latency, throughput, and cache efficiency.
- **Interleaved vs. Grouped Comparison**: Direct speedup, TTFT deltas, and cache hit rate improvement under grouped execution.
- **Markdown & JSON Reports**: Generates reproducible Markdown tables and structured JSON summary artifacts.

```bash
python scripts/analysis/analyze_real_workload.py \
    --inputs reports/real-workloads/runs/*.jsonl \
    --output-report reports/real-workloads/summary_report.md \
    --output-json reports/real-workloads/summary.json
```

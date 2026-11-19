# Pareto Frontier Analysis & Visualization

This directory contains analysis scripts for computing the Pareto trade-off between adapter swap latency and prefill compute latency.

## Scripts

### `analyze_results.py`
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

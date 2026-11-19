# Synthetic Latency Benchmarking Harnesses

This directory contains standalone microbenchmark scripts and pipeline runners to evaluate LoRA adapter transfer overhead and forward-pass prefill compute latency across synthetic parameter grids.

## Parameter Space

- **LoRA Adapter Ranks ($r$)**: `8, 16, 32, 64, 128`
- **Context Lengths ($L$)**: `128, 256, 512, 1024, 2048, 4096, 8192` tokens
- **Precision**: `float16` / `bfloat16`
- **Base Architecture**: Mistral (32 layers, hidden dimension 4096, 4 target projections `q_proj`, `k_proj`, `v_proj`, `o_proj`).

## Scripts

### 1. `benchmark_lora_transfer.py`
Measures host-to-device PCIe transfer latency ($T_{\text{swap}}$) using `torch.cuda.Event` on an active CUDA stream with pinned memory DMA. Also exposes a callable Python API: `measure_t_swap(rank, hidden_dim)`.

```bash
# Run standalone benchmark across default ranks
python scripts/synthetic/benchmark_lora_transfer.py --output-csv results/swap.csv

# Run CPU dry-run simulation
python scripts/synthetic/benchmark_lora_transfer.py --dry-run
```

### 2. `benchmark_prefill_latency.py`
Measures forward-pass compute latency ($\Delta T_{\text{prefill}}$) across context lengths with batch size 1. Automatically pairs prefill timings with $T_{\text{swap}}$ from a swap CSV to produce `timings.csv`. Exposes a callable Python API: `measure_t_prefill(model, context_length)`.

```bash
# Run standalone prefill benchmark and export paired timings
python scripts/synthetic/benchmark_prefill_latency.py \
    --swap-csv results/swap.csv \
    --export-timings results/timings.csv \
    --output-csv results/prefill.csv
```

### 3. `synthetic-pipeline.sh`
Orchestration shell script running Step 1 ($T_{\text{swap}}$), Step 2 ($\Delta T_{\text{prefill}}$), and Step 3 (Pareto frontier analysis).

```bash
# Run synthetic pipeline on GPU
./scripts/synthetic/synthetic-pipeline.sh

# Run fast CPU dry-run
./scripts/synthetic/synthetic-pipeline.sh --dry-run
```

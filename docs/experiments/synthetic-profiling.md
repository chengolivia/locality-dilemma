# Experiment Guide: Synthetic Latency Profiling

This document details the synthetic profiling benchmark suite designed to explore the Pareto frontier of LoRA adapter transfer latency ($T_{\text{swap}}$) versus KV prefix prefill compute latency ($\Delta T_{\text{prefill}}$) without requiring external model downloads.

---

## 1. Parameter Space & Evaluation Grid

The synthetic benchmark sweeps across a $5 \times 7 = 35$ evaluation matrix:
- **LoRA Adapter Ranks ($r$)**: `[8, 16, 32, 64, 128]`
- **Context Lengths ($L$)**: `[128, 256, 512, 1024, 2048, 4096, 8192]` tokens
- **Precision**: `float16`
- **Simulated Model**: Mistral-7B architecture (32 layers, hidden dimension 4096, 4 target projections per layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`).

---

## 2. Benchmark Methodologies

### 2.1 LoRA Host-to-Device Swap Latency ($T_{\text{swap}}$)
- **Script**: `scripts/synthetic/benchmark_lora_transfer.py`
- Generates dummy state dictionaries in pinned host (CPU) memory.
- Uses `torch.cuda.Event(enable_timing=True)` to measure non-blocking asynchronous DMA transfers across the PCIe bus.
- Executes 10 warmup iterations followed by 100 timed iterations per rank.
- Computes effective PCIe bandwidth:
  $$\text{Effective Bandwidth (GB/s)} = \frac{\text{Adapter Size (Bytes)}}{T_{\text{swap}} \text{ (seconds)} \times 10^9}$$

### 2.2 Forward Prefill Compute Latency ($\Delta T_{\text{prefill}}$)
- **Script**: `scripts/synthetic/benchmark_prefill_latency.py`
- Instantiates randomly initialized Mistral transformer layers on device.
- Measures forward-pass latency with batch size 1 across sequence lengths $L$.
- Automatically pairs prefill latency with measured $T_{\text{swap}}$ from the swap CSV and outputs `results/timings.csv`.

### 2.3 Pareto Frontier Analysis
- **Script**: `scripts/analysis/analyze_results.py`
- Reads `results/timings.csv` (columns: `prefix_len,lora_rank,prefill_saved_ms,swap_ms`).
- Computes latency delta $\Delta_{\text{ms}} = \text{prefill\_saved\_ms} - \text{swap\_ms}$.
- Generates:
  - `results/analysis/summary.csv`
  - `results/analysis/break_even.csv`
  - `results/analysis/latency_heatmap.png`

---

## 3. How to Run

### Complete Synthetic Pipeline (GPU)
```bash
./scripts/synthetic/synthetic-pipeline.sh
# Or using the root wrapper:
./synthetic-pipeline.sh
```

### Lightweight CPU Dry-Run
```bash
./scripts/synthetic/synthetic-pipeline.sh --dry-run
# To run all 35 parameter points on CPU:
./scripts/synthetic/synthetic-pipeline.sh --dry-run --full-grid
```

### Using Custom Output Directory
```bash
./scripts/synthetic/synthetic-pipeline.sh --output-dir results/my_run
```

---

## 4. Callable Python API

Both benchmark harnesses can be imported and executed programmatically:

```python
# 1. Benchmark LoRA Transfer Latency
from scripts.synthetic.benchmark_lora_transfer import measure_t_swap

t_swap = measure_t_swap(rank=16, hidden_dim=4096, num_layers=32)
print(f"LoRA Rank 16 swap latency: {t_swap:.3f} ms")

# 2. Benchmark Forward Prefill Latency
import torch
from scripts.synthetic.benchmark_prefill_latency import (
    measure_t_prefill,
    load_mistral_classes,
    create_or_load_mistral_model,
)

MistralConfig, MistralForCausalLM = load_mistral_classes()
model, config = create_or_load_mistral_model(
    MistralConfig, MistralForCausalLM, model_path=None,
    num_layers=32, hidden_size=4096, intermediate_size=14336,
    num_heads=32, num_kv_heads=8, vocab_size=32000,
    max_position_embeddings=32768, dtype_str="float16",
    use_cache=True, device=torch.device("cuda:0"), dtype=torch.float16
)

t_prefill = measure_t_prefill(model, context_length=2048)
print(f"Prefill latency (L=2048): {t_prefill:.3f} ms")
```

---

## 5. Historical H100 Baseline Results

Historical measurements collected on an NVIDIA H100 PCIe accelerator (PSC Bridges-2) are stored in [`h100-synthetic-20261024/`](../../h100-synthetic-20261024/):

| Rank ($r$) | Size (MB) | Median $T_{\text{swap}}$ (ms) | Break-Even Prefix ($L$) |
|:---:|:---:|:---:|:---:|
| 8 | 16.0 MB | 1.76 ms | < 128 tokens |
| 16 | 32.0 MB | 2.39 ms | < 128 tokens |
| 32 | 64.0 MB | 3.64 ms | < 128 tokens |
| 64 | 128.0 MB | 6.13 ms | < 128 tokens |
| 128 | 256.0 MB | 11.83 ms | < 128 tokens |

On high-end accelerators with PCIe Gen5 bandwidth (~50+ GB/s), adapter swap overhead is so small (<12 ms even for rank 128) that prefill computation for prefixes as short as 128 tokens already exceeds $T_{\text{swap}}$, favoring a `prefix-first` cache retention policy.

# The Locality Dilemma: Exploring the Pareto Frontier of Adapter Warmth vs. KV Prefix Reuse

## Abstract

> In LLM serving environments, **KV prefix reuse** is one of the main techniques to keep serving latency low. In multi-adapter LLM serving environments, it is also imperative to **cache popular adapters** to lower loading times.
>
> We propose an empirical exploration of the **Pareto frontier of adapter warmth vs. KV prefix reuse**. We propose a series of experiments designed to locate Pareto optimality on modern inference engines, followed by validation on real-world adapter datasets.


## Installation & Setup

Dependencies in [`requirements.txt`](requirements.txt) use PEP 508 environment markers to support both local development (macOS) and GPU profiling (Amazon EC2 / Linux).

### 1. Local Development (macOS)
```bash
pip install -r requirements.txt
```
*(All Linux/NVIDIA-specific libraries are automatically ignored on macOS).*

### 2. GPU Profiling (Amazon EC2 with NVIDIA GPU)
Ensure NVIDIA drivers and CUDA are installed, then run:
```bash
# 1. Install PyTorch with CUDA 12.4
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install requirements (installs NVML, CUDA-Python, NVTX, TensorBoard Profiler, etc.)
pip install -r requirements.txt
```

---

## Pipelines & Benchmarking

### Parameter Grids & Evaluation Matrix

All pipeline harnesses evaluate across the full parameter space ($5 \times 7 = 35$ paired evaluations):
- **LoRA Adapter Ranks ($r$)**: `[8, 16, 32, 64, 128]`
- **Prefix / Context Lengths ($L$)**: `[128, 256, 512, 1024, 2048, 4096, 8192]`

### 1. Synthetic Pipeline (Recommended for Initial Profiling)
Evaluates synthetic LoRA ranks ($r \in [8, 16, 32, 64, 128]$) and forward prefill latencies ($L \in [128, \dots, 8192]$) on randomly weighted Mistral architecture. **Requires NO model downloads or Hugging Face credentials.**

```bash
# Run complete synthetic pipeline on NVIDIA GPU (all 35 combinations)
./synthetic-pipeline.sh

# Run CPU dry-run on macOS for quick sanity check
./synthetic-pipeline.sh --dry-run

# Run full 35-point grid even during dry-run
./synthetic-pipeline.sh --dry-run --full-grid
```

### 2. Unified Pipeline (`run_pipeline.sh`)
Supports both synthetic profiling and real downloaded weights/workloads:

```bash
# Synthetic mode on GPU
./run_pipeline.sh --mode synthetic

# Real mode using downloaded weights and PEFT adapters
./run_pipeline.sh --mode real \
    --model models/base \
    --adapters-dir models/adapters \
    --workloads data/workloads.jsonl

# Quick dry-run smoke test
./run_pipeline.sh --dry-run
```

### 3. Real Adapter Pipeline (`real-pipeline.sh`)
Shortcut for running the real-model pipeline:
```bash
./real-pipeline.sh
```

---

## Callable API

Both benchmarking harnesses can be imported directly into Python scripts:

### LoRA Host-to-Device Swap Latency
```python
from benchmark_lora_transfer import measure_t_swap

# Returns median swap latency in milliseconds (float)
t_swap = measure_t_swap(rank=16, hidden_dim=4096, num_layers=32)
print(f"LoRA Rank 16 swap latency: {t_swap:.3f} ms")
```

### Forward Prefill Compute Latency
```python
from benchmark_prefill_latency import measure_t_prefill, load_mistral_classes, create_or_load_mistral_model

MistralConfig, MistralForCausalLM = load_mistral_classes()
model, _ = create_or_load_mistral_model(MistralConfig, MistralForCausalLM, model_path=None)

# Returns median forward-pass prefill latency in milliseconds (float)
t_prefill = measure_t_prefill(model, context_length=2048)
print(f"Prefill latency (L=2048): {t_prefill:.3f} ms")
```

---

## Running Unit Tests
```bash
pytest tests/test_benchmarks.py -v
```

---

## Theoretical Background & Formulas

For a model with $L$ layers, hidden dimension $d$, and $M$ target projections per layer, the number of LoRA parameters for rank $r$ is:

$$\text{Parameters}(r) = 2 \times L \times M \times d \times r$$

For 16-bit precision (`torch.float16` or `torch.bfloat16`, 2 bytes/param):

$$\text{Size (Bytes)} = \text{Parameters}(r) \times 2$$

The effective PCIe transfer bandwidth is calculated from the measured median swap time $T_{\text{swap}}$:

$$\text{Effective Bandwidth (GB/s)} = \frac{\text{Size (Bytes)}}{T_{\text{swap}} \times 10^6}$$

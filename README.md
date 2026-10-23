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

<!--
---

## Running the Benchmark

### Basic Run
Profiles ranks 8, 16, 32, 64, and 128 (100 iterations each):
```bash
python benchmark_lora_transfer.py
```

### Dry Run on CPU (Mac Testing)
Verify script logic and formatting on a machine without a CUDA GPU:
```bash
python benchmark_lora_transfer.py --dry-run
```

### Full Configuration Example
Simulate all 7 linear projections (Attention + MLP) with JSON and CSV exports:
```bash
python benchmark_lora_transfer.py \
    --ranks 8 16 32 64 128 \
    --num-layers 32 \
    --hidden-dim 4096 \
    --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --dtype float16 \
    --iterations 100 \
    --warmup 15 \
    --output-json results.json \
    --output-csv results.csv
```

### Comparing Memory Modes
```bash
# Pinned memory (fast PCIe DMA transfer)
python benchmark_lora_transfer.py --pin-memory

# Pageable memory (subject to OS staging buffer overhead)
python benchmark_lora_transfer.py --no-pin-memory

# In-place pre-allocated buffer transfer (eliminates allocator overhead)
python benchmark_lora_transfer.py --transfer-mode copy_preallocated
```

---

## Theoretical Background & Formulas

For a model with $L$ layers, hidden dimension $d$, and $M$ target projections per layer, the number of LoRA parameters for rank $r$ is:

$$\text{Parameters}(r) = 2 \times L \times M \times d \times r$$

For 16-bit precision (`torch.float16` or `torch.bfloat16`, 2 bytes/param):

$$\text{Size (Bytes)} = \text{Parameters}(r) \times 2$$

The effective PCIe transfer bandwidth is calculated from the measured median swap time $T_{swap}$:

$$\text{Effective Bandwidth (GB/s)} = \frac{\text{Size (Bytes)}}{T_{swap} \times 10^6}$$
-->
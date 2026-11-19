# Experiment Environment Setup Guide

This document describes how to configure and verify environments for local development, synthetic CPU dry-runs, and real-model GPU serving on Amazon EC2.

---

## 1. Environments Overview

| Capability | Local Development | Amazon EC2 (GPU Serving) | PSC Bridges-2 (Cluster) |
|---|---|---|---|
| **OS** | macOS / Linux | Ubuntu 24.04 LTS | Linux / PSC Cluster |
| **Hardware** | Apple Silicon / x86 CPU | AWS EC2 `g5.2xlarge` (1x NVIDIA A10G 24GB) | NVIDIA H100 (80GB PCIe) |
| **Python** | 3.10 - 3.12 | Python 3.12 (`~/venvs/vllm`) | Python 3.10+ (`LOCALITY_VENV`) |
| **PyTorch** | CPU / MPS build | 2.14.0+cu130 (CUDA 13.0) | 2.6.0+cu124 (CUDA 12.4) |
| **vLLM** | N/A (offline tests only) | vLLM 0.33.0 (Commit `93a3e66d`) | N/A |
| **FlashInfer** | N/A | FlashInfer 0.7.2.post1 | N/A |
| **Purpose** | Unit testing, dry-runs, replay builders | Real-model KV sharing experiments | High-bandwidth synthetic profiling |

---

## 2. Amazon EC2 GPU Setup (A10G / vLLM Environment)

The EC2 benchmarking environment is installed on persistent EBS storage at `/home/ubuntu/venvs/vllm`.

### Step 1: Activate the Virtual Environment
```bash
source ~/venvs/vllm/bin/activate
```

### Step 2: Configure CUDA Toolkit Environment Variables
The CUDA 13.0 toolkit is packaged inside the Python virtual environment under `nvidia/cu13`. Set the following environment variables in every new shell:

```bash
export CUDA_HOME="$HOME/venvs/vllm/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
```

Verify that the CUDA compiler (`nvcc`) is available on your path:
```bash
which nvcc
nvcc --version
```

### Step 3: Verify Required CUDA Library Symlinks
The packaged CUDA runtime requires library symlinks inside `$CUDA_HOME`:
```bash
ln -sfn lib "$CUDA_HOME/lib64"
ln -sfn libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
```
*(These symlinks persist on disk. If the environment is ever re-created from scratch, verify their presence).*

### Step 4: Verify Environment Packages & GPU Access
Run this one-line verification command:
```bash
python -c "
import torch, vllm, flashinfer
print('PyTorch:   ', torch.__version__)
print('CUDA Avail:', torch.cuda.is_available(), f'({torch.cuda.get_device_name(0)})')
print('CUDA Ver:  ', torch.version.cuda)
print('vLLM:      ', vllm.__version__)
print('FlashInfer:', flashinfer.__version__)
"
```
Expected output:
```text
PyTorch:    2.14.0+cu130
CUDA Avail: True (NVIDIA A10G)
CUDA Ver:   13.0
vLLM:       0.33.0
FlashInfer: 0.7.2.post1
```

---

## 3. Local Development Setup (macOS / CPU)

For lightweight local development, prompt reconstruction, and dry-run benchmarking without a GPU:

### Using pip and `requirements.txt`
```bash
# Create local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements (PEP 508 markers automatically skip Linux/CUDA libraries on macOS)
pip install -r requirements.txt
```

### Minimal Setup via `requirements-local.txt`
If you only need to run trace verification or manifest tools without PyTorch:
```bash
pip install -r requirements-local.txt
```

---

## 4. Model Weights and LoRA Adapter Layout

Downloaded model weights and PEFT adapters are stored under `models/` (ignored by Git):

```text
models/
├── base/                                # Mistral-7B-v0.1 base model weights
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── model-00001-of-00002.safetensors
│   └── model-00002-of-00002.safetensors
└── adapters/                            # PEFT LoRA adapter checkpoints
    ├── predibase__customer_support/     # Rank 8, alpha 16 (Target: q, k, v, o)
    ├── predibase__magicoder/            # Rank 8, alpha 16 (Target: q, k, v, o)
    ├── predibase__gsm8k/                # Rank 8, alpha 16 (Target: q, k, v, o)
    ├── predibase__wikisql/              # Rank 8, alpha 16 (Target: q, k, v, o)
    └── predibase__drop_explained/       # Rank 8, alpha 16 (Target: q, k, v, o)
```

To re-download models or adapters:
```bash
python scripts/setup/download_models.py \
    --base mistralai/Mistral-7B-v0.1 \
    --adapters predibase/customer_support predibase/magicoder \
    --output models
```
*(Do not download models unless instructed, as weights are already provisioned on the persistent volume).*

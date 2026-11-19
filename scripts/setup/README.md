# Model & Environment Setup Scripts

This directory contains utilities for downloading the base Mistral model and PEFT LoRA adapters, and configuring the execution environment.

## Scripts

### 1. `download_models.py`
Downloads base models and LoRA adapters using the Hugging Face CLI (`hf download`), and validates that downloaded adapters are genuine PEFT LoRA adapters compatible with the specified base model.

```bash
# Download Mistral-7B-v0.1 base model and target LoRA adapters
python scripts/setup/download_models.py \
    --base mistralai/Mistral-7B-v0.1 \
    --adapters predibase/customer_support predibase/magicoder \
    --output models
```

### 2. `setup_models.sh`
Automation script designed for remote compute nodes (e.g. PSC Bridges-2 or cluster environments). Creates a dedicated virtual environment, installs lightweight setup dependencies from `requirements-local.txt`, downloads required models, and verifies their integrity.

```bash
bash scripts/setup/setup_models.sh [OPTIONAL_DESTINATION_DIR]
```

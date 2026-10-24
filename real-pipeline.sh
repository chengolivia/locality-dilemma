#!/usr/bin/env bash
# ==============================================================================
# Real Model & Adapter Latency Benchmarking Pipeline
#
# Benchmarks:
#   - Real downloaded PEFT LoRA adapters in models/adapters/
#   - Downloaded base model (models/base) and workloads (data/workloads.jsonl)
#   - Generates paired timings (results/timings.csv) and Pareto analysis
#
# Usage:
#   ./real-pipeline.sh            # Run on CUDA GPU
#   ./real-pipeline.sh --dry-run  # Fast smoke-test on CPU
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_pipeline.sh" --mode real "$@"


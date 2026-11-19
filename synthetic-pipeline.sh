#!/usr/bin/env bash
# ==============================================================================
# Synthetic Latency Benchmarking & Pareto Analysis Pipeline
# Root wrapper delegating to scripts/synthetic/synthetic-pipeline.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/synthetic/synthetic-pipeline.sh" "$@"

#!/usr/bin/env bash
# ==============================================================================
# Unified Locality Dilemma Latency Benchmarking & Pareto Analysis Pipeline
# Root wrapper delegating to scripts/run_pipeline.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/run_pipeline.sh" "$@"

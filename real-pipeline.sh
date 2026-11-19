#!/usr/bin/env bash
# ==============================================================================
# Real Model & Adapter Latency Benchmarking Pipeline
# Root wrapper delegating to scripts/real/real-pipeline.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/real/real-pipeline.sh" "$@"

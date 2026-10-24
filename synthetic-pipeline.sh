#!/usr/bin/env bash
# ==============================================================================
# Synthetic Latency Benchmarking & Pareto Analysis Pipeline
#
# Measures:
#   1. LoRA PCIe Host-to-Device transfer latency (T_swap) across ranks
#   2. LLM forward-pass prefill compute latency (Delta T_prefill) across context lengths
#   3. Generates paired timings (results/timings.csv) and Pareto frontier analysis
#
# Usage:
#   ./synthetic-pipeline.sh            # Run on CUDA GPU
#   ./synthetic-pipeline.sh --dry-run  # Fast smoke-test on CPU (Mac)
# ==============================================================================

set -euo pipefail

# Detect python interpreter: prefer active virtualenv or .venv
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        PYTHON="${VIRTUAL_ENV}/bin/python"
    elif [[ -x "./.venv/bin/python" ]]; then
        PYTHON="./.venv/bin/python"
    else
        PYTHON="python"
    fi
fi

# Ensure output directory exists
mkdir -p results results/analysis

# Parse optional arguments (e.g. --dry-run, --full-grid, or --help)
DRY_RUN_FLAG=""
FULL_GRID=false

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $(basename "$0") [--dry-run] [--full-grid]"
            echo ""
            echo "Parameter Grids:"
            echo "  LoRA Ranks (r):       8, 16, 32, 64, 128"
            echo "  Context Lengths (L):  128, 256, 512, 1024, 2048, 4096, 8192"
            echo ""
            echo "Options:"
            echo "  --dry-run    Run fast lightweight CPU simulation (for testing on Mac)"
            echo "  --full-grid  Run full parameter grids (all 35 combinations) even during dry-run"
            echo "  -h, --help   Show this help message and exit"
            exit 0
            ;;
        --dry-run)
            DRY_RUN_FLAG="--dry-run"
            ;;
        --full-grid)
            FULL_GRID=true
            ;;
    esac
done

if [[ -n "$DRY_RUN_FLAG" ]]; then
    echo "=============================================================================="
    echo " [!] Running in DRY-RUN mode (CPU simulation for testing)"
    if [[ "$FULL_GRID" == true ]]; then
        echo "     Full grid enabled: evaluating all ranks and context lengths on CPU."
    fi
    echo "=============================================================================="
fi

echo "=============================================================================="
echo " Starting Synthetic LoRA & Prefill Latency Benchmarking Pipeline"
echo " Target Parameter Grids:"
echo "   LoRA Ranks (r):         8, 16, 32, 64, 128"
echo "   Context Lengths (L):    128, 256, 512, 1024, 2048, 4096, 8192"
echo "=============================================================================="
echo "Output Directory: results/"
echo

# ------------------------------------------------------------------------------
# Step 1: Benchmark LoRA swap latency (T_swap) across ranks 8, 16, 32, 64, 128
# ------------------------------------------------------------------------------
echo ">>> [Step 1/3] Benchmarking LoRA Host-to-Device Swap Latency (T_swap)..."
if [[ -n "$DRY_RUN_FLAG" && "$FULL_GRID" == false ]]; then
    "$PYTHON" benchmark_lora_transfer.py \
        --dry-run \
        --ranks 8 16 \
        --iterations 5 \
        --warmup 2 \
        --output-csv results/swap.csv
else
    EXTRA_FLAGS=()
    if [[ -n "$DRY_RUN_FLAG" ]]; then
        EXTRA_FLAGS+=("--dry-run" "--iterations" "5" "--warmup" "2")
    else
        EXTRA_FLAGS+=("--iterations" "100" "--warmup" "10")
    fi
    "$PYTHON" benchmark_lora_transfer.py \
        --ranks 8 16 32 64 128 \
        "${EXTRA_FLAGS[@]}" \
        --output-csv results/swap.csv
fi
echo ">>> Step 1 Complete: results/swap.csv generated."
echo

# ------------------------------------------------------------------------------
# Step 2: Benchmark forward prefill latency (Delta T_prefill) across context lengths
#         and pair with T_swap into results/timings.csv
# ------------------------------------------------------------------------------
echo ">>> [Step 2/3] Benchmarking Forward Prefill Latency (Delta T_prefill)..."
if [[ -n "$DRY_RUN_FLAG" && "$FULL_GRID" == false ]]; then
    "$PYTHON" benchmark_prefill_latency.py \
        --dry-run \
        --context-lengths 128 256 \
        --iterations 5 \
        --warmup 2 \
        --swap-csv results/swap.csv \
        --export-timings results/timings.csv \
        --output-csv results/prefill.csv
else
    EXTRA_FLAGS=()
    if [[ -n "$DRY_RUN_FLAG" ]]; then
        EXTRA_FLAGS+=("--dry-run" "--iterations" "5" "--warmup" "2")
    else
        EXTRA_FLAGS+=("--iterations" "50" "--warmup" "10")
    fi
    "$PYTHON" benchmark_prefill_latency.py \
        --context-lengths 128 256 512 1024 2048 4096 8192 \
        "${EXTRA_FLAGS[@]}" \
        --swap-csv results/swap.csv \
        --export-timings results/timings.csv \
        --output-csv results/prefill.csv
fi
echo ">>> Step 2 Complete: results/prefill.csv and results/timings.csv generated."
echo

# ------------------------------------------------------------------------------
# Step 3: Run Pareto frontier analysis & generate heatmap
# ------------------------------------------------------------------------------
echo ">>> [Step 3/3] Running Pareto Frontier Analysis (analyze_results.py)..."
"$PYTHON" analyze_results.py \
    --input results/timings.csv \
    --output-dir results/analysis

echo
echo "=============================================================================="
echo " Pipeline Complete!"
echo "  - Swap Latencies:   results/swap.csv"
echo "  - Prefill Latency:  results/prefill.csv"
echo "  - Paired Timings:   results/timings.csv"
echo "  - Pareto Analysis:  results/analysis/summary.csv"
echo "  - Break-Even Curve: results/analysis/break_even.csv"
echo "  - Heatmap Plot:     results/analysis/latency_heatmap.png"
echo "=============================================================================="


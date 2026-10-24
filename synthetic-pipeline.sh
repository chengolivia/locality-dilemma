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

# Parse optional arguments (e.g. --dry-run, --full-grid, --output-dir, or --help)
DRY_RUN_FLAG=""
FULL_GRID=false
OUTPUT_DIR="results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Parameter Grids:"
            echo "  LoRA Ranks (r):       8, 16, 32, 64, 128"
            echo "  Context Lengths (L):  128, 256, 512, 1024, 2048, 4096, 8192"
            echo ""
            echo "Options:"
            echo "  -d, --dry-run          Run fast lightweight CPU simulation (for testing on Mac)"
            echo "      --full-grid        Run full parameter grids (all 35 combinations) even during dry-run"
            echo "  -o, --output-dir <dir> Output directory to store results (default: results; overwrites files in that dir)"
            echo "  -h, --help             Show this help message and exit"
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN_FLAG="--dry-run"
            shift
            ;;
        --full-grid)
            FULL_GRID=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            echo "Run '$(basename "$0") --help' for usage." >&2
            exit 1
            ;;
    esac
done

# Output file paths
ANALYSIS_DIR="${OUTPUT_DIR}/analysis"
SWAP_CSV="${OUTPUT_DIR}/swap.csv"
PREFILL_CSV="${OUTPUT_DIR}/prefill.csv"
TIMINGS_CSV="${OUTPUT_DIR}/timings.csv"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR" "$ANALYSIS_DIR"

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
echo "Output Directory: ${OUTPUT_DIR}/"
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
        --output-csv "$SWAP_CSV"
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
        --output-csv "$SWAP_CSV"
fi
echo ">>> Step 1 Complete: $SWAP_CSV generated."
echo

# ------------------------------------------------------------------------------
# Step 2: Benchmark forward prefill latency (Delta T_prefill) across context lengths
#         and pair with T_swap into timings.csv
# ------------------------------------------------------------------------------
echo ">>> [Step 2/3] Benchmarking Forward Prefill Latency (Delta T_prefill)..."
if [[ -n "$DRY_RUN_FLAG" && "$FULL_GRID" == false ]]; then
    "$PYTHON" benchmark_prefill_latency.py \
        --dry-run \
        --context-lengths 128 256 \
        --iterations 5 \
        --warmup 2 \
        --swap-csv "$SWAP_CSV" \
        --export-timings "$TIMINGS_CSV" \
        --output-csv "$PREFILL_CSV"
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
        --swap-csv "$SWAP_CSV" \
        --export-timings "$TIMINGS_CSV" \
        --output-csv "$PREFILL_CSV"
fi
echo ">>> Step 2 Complete: $PREFILL_CSV and $TIMINGS_CSV generated."
echo

# ------------------------------------------------------------------------------
# Step 3: Run Pareto frontier analysis & generate heatmap
# ------------------------------------------------------------------------------
echo ">>> [Step 3/3] Running Pareto Frontier Analysis (analyze_results.py)..."
"$PYTHON" analyze_results.py \
    --input "$TIMINGS_CSV" \
    --output-dir "$ANALYSIS_DIR"

echo
echo "=============================================================================="
echo " Pipeline Complete!"
echo "  - Swap Latencies:   $SWAP_CSV"
echo "  - Prefill Latency:  $PREFILL_CSV"
echo "  - Paired Timings:   $TIMINGS_CSV"
echo "  - Pareto Analysis:  ${ANALYSIS_DIR}/summary.csv"
echo "  - Break-Even Curve: ${ANALYSIS_DIR}/break_even.csv"
echo "  - Heatmap Plot:     ${ANALYSIS_DIR}/latency_heatmap.png"
echo "=============================================================================="


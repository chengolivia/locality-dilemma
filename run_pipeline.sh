#!/usr/bin/env bash
# ==============================================================================
# Unified Locality Dilemma Latency Benchmarking & Pareto Analysis Pipeline
#
# Supports both:
#   1. SYNTHETIC MODE:
#      - Evaluates synthetic LoRA ranks (8, 16, 32, 64, 128)
#      - Evaluates synthetic Mistral architecture (L = 128 to 8192)
#      - Requires NO model downloads or HF credentials
#
#   2. REAL MODE:
#      - Evaluates downloaded base model (models/base) from download_models.py
#      - Evaluates downloaded real PEFT LoRA adapters (models/adapters/*)
#      - Evaluates deterministic tokenized prompts (data/workloads.jsonl)
#
# Usage:
#   ./run_pipeline.sh                          # Default: synthetic on GPU
#   ./run_pipeline.sh --mode synthetic         # Explicit synthetic mode
#   ./run_pipeline.sh --mode real              # Real downloaded models
#   ./run_pipeline.sh --dry-run                # Fast CPU dry-run on Mac
#   ./run_pipeline.sh --mode real --dry-run    # Real models dry-run
# ==============================================================================

set -euo pipefail

# Default configuration values
MODE="synthetic"
DRY_RUN=false
FULL_GRID=false
OUTPUT_DIR="results"
BASE_MODEL="models/base"
ADAPTERS_DIR="models/adapters"
WORKLOADS="data/workloads.jsonl"
ITERATIONS=""
WARMUP=""
RANKS=("8" "16" "32" "64" "128")
CONTEXT_LENGTHS=("128" "256" "512" "1024" "2048" "4096" "8192")

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Parameter Grids:
  LoRA Ranks (r):       8, 16, 32, 64, 128
  Context Lengths (L):  128, 256, 512, 1024, 2048, 4096, 8192

Options:
  -m, --mode <synthetic|real>    Benchmark mode (default: synthetic).
  -d, --dry-run                  Execute lightweight CPU dry-run (for Mac / testing).
      --full-grid                Evaluate all 35 parameter combinations even during dry-run.
  -o, --output-dir <path>        Directory to store outputs (default: results).
      --ranks <r1 r2...>         Custom LoRA rank grid (default: 8 16 32 64 128).
      --context-lengths <L1 L2...> Custom context length grid (default: 128 256 512 1024 2048 4096 8192).
      --model <path>             Base model directory for real mode (default: models/base).
      --adapters-dir <path>      LoRA adapters directory for real mode (default: models/adapters).
      --workloads <path>         Workload JSONL file for real mode (default: data/workloads.jsonl).
      --iterations <int>         Override number of timed iterations per run.
      --warmup <int>             Override number of untimed warmup iterations.
  -h, --help                     Display this help message and exit.

Examples:
  $(basename "$0") --mode synthetic
  $(basename "$0") --mode real
  $(basename "$0") --dry-run
  $(basename "$0") --dry-run --full-grid
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
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
        --ranks)
            shift
            RANKS=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                RANKS+=("$1")
                shift
            done
            ;;
        --context-lengths)
            shift
            CONTEXT_LENGTHS=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                CONTEXT_LENGTHS+=("$1")
                shift
            done
            ;;
        --model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --adapters-dir)
            ADAPTERS_DIR="$2"
            shift 2
            ;;
        --workloads)
            WORKLOADS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            print_usage
            exit 1
            ;;
    esac
done

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

# Validate mode
if [[ "$MODE" != "synthetic" && "$MODE" != "real" ]]; then
    echo "Error: --mode must be either 'synthetic' or 'real' (received: '$MODE')" >&2
    exit 1
fi

ANALYSIS_DIR="${OUTPUT_DIR}/analysis"
SWAP_CSV="${OUTPUT_DIR}/swap.csv"
PREFILL_CSV="${OUTPUT_DIR}/prefill.csv"
TIMINGS_CSV="${OUTPUT_DIR}/timings.csv"

mkdir -p "$OUTPUT_DIR" "$ANALYSIS_DIR"

echo "=============================================================================="
echo " Locality Dilemma: Latency Benchmarking & Pareto Analysis Pipeline"
echo "=============================================================================="
echo " Mode:                 ${MODE}"
echo " Dry Run (CPU):        ${DRY_RUN}"
echo " LoRA Ranks (r):       ${RANKS[*]}"
echo " Context Lengths (L):  ${CONTEXT_LENGTHS[*]}"
echo " Output Directory:     ${OUTPUT_DIR}/"
if [[ "$MODE" == "real" ]]; then
    echo " Base Model:           ${BASE_MODEL}"
    echo " Adapters Dir:         ${ADAPTERS_DIR}"
    echo " Workloads:            ${WORKLOADS}"
fi
echo "=============================================================================="
echo

# ------------------------------------------------------------------------------
# STEP 1: LoRA Host-to-Device Swap Latency (T_swap)
# ------------------------------------------------------------------------------
echo ">>> [Step 1/3] Benchmarking LoRA Transfer Latency (T_swap)..."

LORA_ARGS=("--output-csv" "$SWAP_CSV")

if [[ "$DRY_RUN" == true && "$FULL_GRID" == false ]]; then
    LORA_ARGS+=("--dry-run")
    LORA_ARGS+=("--iterations" "${ITERATIONS:-5}")
    LORA_ARGS+=("--warmup" "${WARMUP:-2}")
    if [[ "$MODE" == "synthetic" ]]; then
        LORA_ARGS+=("--ranks" "8" "16")
    fi
else
    if [[ "$DRY_RUN" == true ]]; then
        LORA_ARGS+=("--dry-run")
        LORA_ARGS+=("--iterations" "${ITERATIONS:-5}")
        LORA_ARGS+=("--warmup" "${WARMUP:-2}")
    else
        LORA_ARGS+=("--iterations" "${ITERATIONS:-100}")
        LORA_ARGS+=("--warmup" "${WARMUP:-10}")
    fi
    if [[ "$MODE" == "synthetic" ]]; then
        LORA_ARGS+=("--ranks" "${RANKS[@]}")
    fi
fi

if [[ "$MODE" == "real" ]]; then
    if [[ -d "$ADAPTERS_DIR" ]]; then
        LORA_ARGS+=("--adapters-dir" "$ADAPTERS_DIR")
    else
        echo "[!] Warning: Adapters directory '$ADAPTERS_DIR' not found. Falling back to synthetic ranks: ${RANKS[*]}"
        LORA_ARGS+=("--ranks" "${RANKS[@]}")
    fi
fi

"$PYTHON" benchmark_lora_transfer.py "${LORA_ARGS[@]}"
echo ">>> Step 1 Complete: $SWAP_CSV generated."
echo

# ------------------------------------------------------------------------------
# STEP 2: Forward-Pass Prefill Latency (Delta T_prefill) & Paired Timings
# ------------------------------------------------------------------------------
echo ">>> [Step 2/3] Benchmarking Forward Prefill Latency (Delta T_prefill)..."

PREFILL_ARGS=(
    "--swap-csv" "$SWAP_CSV"
    "--export-timings" "$TIMINGS_CSV"
    "--output-csv" "$PREFILL_CSV"
)

if [[ "$DRY_RUN" == true && "$FULL_GRID" == false ]]; then
    PREFILL_ARGS+=("--dry-run")
    PREFILL_ARGS+=("--iterations" "${ITERATIONS:-5}")
    PREFILL_ARGS+=("--warmup" "${WARMUP:-2}")
    if [[ "$MODE" == "synthetic" ]]; then
        PREFILL_ARGS+=("--context-lengths" "128" "256")
    fi
else
    if [[ "$DRY_RUN" == true ]]; then
        PREFILL_ARGS+=("--dry-run")
        PREFILL_ARGS+=("--iterations" "${ITERATIONS:-5}")
        PREFILL_ARGS+=("--warmup" "${WARMUP:-2}")
    else
        PREFILL_ARGS+=("--iterations" "${ITERATIONS:-50}")
        PREFILL_ARGS+=("--warmup" "${WARMUP:-10}")
    fi
    if [[ "$MODE" == "synthetic" ]]; then
        PREFILL_ARGS+=("--context-lengths" "${CONTEXT_LENGTHS[@]}")
    fi
fi

if [[ "$MODE" == "real" ]]; then
    if [[ -d "$BASE_MODEL" ]]; then
        PREFILL_ARGS+=("--model" "$BASE_MODEL")
    else
        echo "[!] Note: Base model '$BASE_MODEL' not found. Initializing architecture with random weights."
    fi

    if [[ -f "$WORKLOADS" ]]; then
        PREFILL_ARGS+=("--workloads" "$WORKLOADS")
    else
        echo "[!] Note: Workloads file '$WORKLOADS' not found. Using standard sequence lengths."
    fi
fi

"$PYTHON" benchmark_prefill_latency.py "${PREFILL_ARGS[@]}"
echo ">>> Step 2 Complete: $PREFILL_CSV and $TIMINGS_CSV generated."
echo

# ------------------------------------------------------------------------------
# STEP 3: Pareto Frontier Analysis & Heatmap Plotting
# ------------------------------------------------------------------------------
echo ">>> [Step 3/3] Running Pareto Frontier Analysis (analyze_results.py)..."

"$PYTHON" analyze_results.py \
    --input "$TIMINGS_CSV" \
    --output-dir "$ANALYSIS_DIR"

echo
echo "=============================================================================="
echo " Pipeline Finished Successfully!"
echo " Results available in ${OUTPUT_DIR}/:"
echo "   - T_swap Latencies:     $SWAP_CSV"
echo "   - Prefill Latencies:    $PREFILL_CSV"
echo "   - Paired Measurements:  $TIMINGS_CSV"
echo "   - Pareto Summary:       ${ANALYSIS_DIR}/summary.csv"
echo "   - Break-Even Frontier:  ${ANALYSIS_DIR}/break_even.csv"
echo "   - Heatmap Plot:         ${ANALYSIS_DIR}/latency_heatmap.png"
echo "=============================================================================="

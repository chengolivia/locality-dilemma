# Scripts Directory

This directory contains executable benchmarks, setup utilities, and experiment runners organized by experimental phase.

## Directory Layout

```text
scripts/
├── setup/                  # Model & environment setup
│   ├── download_models.py  # Hugging Face model & adapter downloader
│   ├── setup_models.sh     # Automation setup for remote compute / PSC
│   └── README.md
├── synthetic/              # Synthetic microbenchmarking
│   ├── benchmark_lora_transfer.py    # PCIe H2D swap microbenchmark (T_swap)
│   ├── benchmark_prefill_latency.py  # Forward prefill compute microbenchmark (Delta T_prefill)
│   ├── synthetic-pipeline.sh         # End-to-end synthetic pipeline runner
│   └── README.md
├── real/                   # Controlled real-model & vLLM experiments
│   ├── patch_kv_cache.py             # vLLM KV-cache hash patching utility
│   ├── run_kv_sharing_experiment.py  # Automated 3-request KV-sharing experiment runner
│   ├── generate_workloads.py         # Deterministic prompt prefix generator
│   ├── run_generation.py             # vLLM offline inference smoke test
│   ├── real-pipeline.sh              # Real-mode pipeline launcher
│   └── README.md
├── workplace/              # Real-world workplace trajectory processing (In Progress)
│   ├── verify_workplace_setup.py     # Trajectory and LoRA header integrity checks
│   ├── build_workplace_replay.py     # Text-only event history replay builder
│   └── README.md
├── analysis/               # Results processing & plotting
│   ├── analyze_results.py            # Pareto frontier analyzer & heatmap generator
│   └── README.md
├── run_pipeline.sh         # Unified benchmarking pipeline (synthetic & real modes)
└── README.md               # Master scripts guide (this file)
```

## Quick Reference Commands

| Experiment / Task | Command |
|---|---|
| Synthetic Pipeline (GPU) | `./scripts/synthetic/synthetic-pipeline.sh` |
| Synthetic Dry-Run (CPU) | `./scripts/synthetic/synthetic-pipeline.sh --dry-run` |
| Unified Pipeline (Real Models) | `./scripts/run_pipeline.sh --mode real` |
| KV-Cache Native Condition | `python scripts/real/patch_kv_cache.py --restore` |
| KV-Cache Forced Condition | `python scripts/real/patch_kv_cache.py --forced` |
| Run KV Sharing Experiment | `python scripts/real/run_kv_sharing_experiment.py --condition {native\|forced} --output <path> --start-server` |
| Preflight Workplace Setup | `python scripts/workplace/verify_workplace_setup.py` |
| Build Workplace Replays | `python scripts/workplace/build_workplace_replay.py` |
| Pareto Analysis | `python scripts/analysis/analyze_results.py --input results/timings.csv --output-dir results/analysis` |

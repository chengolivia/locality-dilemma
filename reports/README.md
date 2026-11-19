# Experimental Reports & Benchmark Results

This directory contains empirical measurements, analysis summaries, and evaluation reports across the three phases of the Locality Dilemma research project.

## Directory Structure

```text
reports/
├── README.md                          # Master reports index (this file)
├── synthetic/                         # Synthetic microbenchmarking reports & analysis
│   └── README.md                      # Overview of synthetic findings & links to H100 benchmarks
├── adapter-kv-cache/                  # Controlled real-model vLLM experiments
│   ├── README.md                      # Summary of Native vs. Forced sharing findings
│   ├── kv_sharing_comparison.md       # Comprehensive evaluation report
│   ├── kv_sharing_native.jsonl        # Raw per-request measurements (Native condition)
│   └── kv_sharing_forced.jsonl        # Raw per-request measurements (Forced sharing patch)
└── real-workloads/                    # Realistic agent trajectory replay (In Progress)
    ├── README.md                      # Status, task taxonomy, and coverage projections
    ├── theagentcompany_20tasks.json   # 20-task specification and adapter mappings
    ├── theagentcompany_20tasks_verified.json # Preflight-verified task manifest
    ├── adapter_manifest_20tasks.json  # Inspected Safetensors headers & checksums
    └── workplace20_coverage.json      # Context-limit coverage across 8k, 16k, 32k limits
```

## Experiment Streams at a Glance

| Stream | Hardware / Platform | Key Artifacts | Current Status |
|---|---|---|:---:|
| **A. Synthetic Profiling** | NVIDIA H100 (PSC Bridges-2) & Synthetic Harnesses | [`reports/synthetic/README.md`](synthetic/README.md), [`h100-synthetic-20261024/`](../h100-synthetic-20261024/) | **Completed** |
| **B. Controlled Real-Model KV-Cache** | NVIDIA A10G (AWS EC2 `g5.2xlarge`) | [`reports/adapter-kv-cache/kv_sharing_comparison.md`](adapter-kv-cache/kv_sharing_comparison.md), [`reports/adapter-kv-cache/`](adapter-kv-cache/) | **Completed** |
| **C. Real-World Workload Profiling** | TheAgentCompany Trajectories (Mistral-7B + 5 LoRA Adapters) | [`reports/real-workloads/README.md`](real-workloads/README.md), [`docs/experiments/workplace-replay.md`](../docs/experiments/workplace-replay.md) | **In Progress** |

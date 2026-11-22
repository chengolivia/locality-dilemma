# Controlled Real-Model & vLLM Experiment Scripts

This directory contains executable tools and test runners for controlled experiments using real Mistral-7B weights, real PEFT LoRA adapters, and the vLLM serving engine.

## Scripts

### 1. `patch_kv_cache.py`
Inspects, applies, and reverts the experimental forced cross-adapter KV-cache sharing patch in vLLM's `kv_cache_utils.py`.
- **Native mode**: KV-cache hashing includes `("lora", lora_name, lora_path)`.
- **Forced mode**: Hash keys return `[]`, intentionally bypassing adapter isolation for prompt tokens.

```bash
# Check current patch status
python scripts/real/patch_kv_cache.py --status

# Restore native vLLM implementation
python scripts/real/patch_kv_cache.py --restore

# Apply experimental forced cross-adapter sharing patch
python scripts/real/patch_kv_cache.py --forced
```

### 2. `run_kv_sharing_experiment.py`
Automated runner for the controlled KV-cache sharing experiment. Manages the vLLM server lifecycle, executes the 3-request sequence (`magicoder` $\to$ `customer_support` $\to$ `magicoder`), parses Prometheus metrics before/after each request, and records per-request TTFT, ITL, throughput, and output text in JSONL format.

```bash
# Run Condition A (Native)
python scripts/real/patch_kv_cache.py --restore
python scripts/real/run_kv_sharing_experiment.py \
    --condition native \
    --output reports/adapter-kv-cache/kv_sharing_native.jsonl \
    --start-server

# Run Condition B (Forced Sharing)
python scripts/real/patch_kv_cache.py --forced
python scripts/real/run_kv_sharing_experiment.py \
    --condition forced \
    --output reports/adapter-kv-cache/kv_sharing_forced.jsonl \
    --start-server

# Reset to native state
python scripts/real/patch_kv_cache.py --restore
```

### 3. `generate_workloads.py`
Generates deterministic tokenized prompt prefixes of exact token lengths ($L \in [128, \dots, 8192]$) from a stable text corpus using the base tokenizer.

```bash
python scripts/real/generate_workloads.py \
    --tokenizer models/base \
    --output data/workloads.jsonl
```

### 4. `run_generation.py`
Smoke test for vLLM offline inference using `vllm.LLM` with and without LoRA adapters.

```bash
python scripts/real/run_generation.py \
    --model models/base \
    --adapter models/adapters/predibase__magicoder \
    --workloads data/workloads.jsonl \
    --output results/generations.jsonl
```

### 5. `measurement.py`
Reusable telemetry and measurement infrastructure for real-workload replay experiments. Scrapes Prometheus prefix-cache counters before/after each request, executes completions via HTTP POST, and structures per-request latency, token counts, generated text, and counter deltas with explicit handling of missing metrics (no silent zero substitution).

### 6. `run_workload_replay.py`
Execution runner bridging reconstructed TheAgentCompany replays (`data/agent_replay_20tasks/replay_8192.jsonl`) to `measurement.py` and vLLM. Constructs `adapter_interleaved` (121 adapter transitions) and `adapter_grouped` (4 adapter transitions) schedules, executes requests sequentially against the multi-adapter server, and logs per-request telemetry to `reports/real-workloads/runs/{strategy}_{timestamp}.jsonl`.

```bash
# Verify schedules in dry-run mode without sending requests
python scripts/real/run_workload_replay.py --dry-run

# Run both schedules against active vLLM instance
python scripts/real/run_workload_replay.py \
    --config configs/real_workload.yaml \
    --replay-file data/agent_replay_20tasks/replay_8192.jsonl \
    --strategy both

# Run smoke test on first 5 requests
python scripts/real/run_workload_replay.py --limit 5
```

### 7. `real-pipeline.sh`
Convenience wrapper that invokes `scripts/run_pipeline.sh --mode real`.

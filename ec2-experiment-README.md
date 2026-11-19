# Real-Model LoRA / KV-Cache Experiments

This document describes how to reproduce the real-model experiments for studying LoRA adapter locality and cross-adapter KV-prefix reuse using vLLM.

## 1. Machine setup

### Environment

The experiments were run on an AWS EC2 Ubuntu 24.04 instance with a g5.2xlarge A10G. (11/14/2026)

| Component | Version / configuration |
|---|---|
| Python | 3.12 |
| vLLM | 0.33.0 |
| PyTorch | 2.14.0+cu130 |
| CUDA toolkit | 13.0 |
| FlashInfer | 0.7.2.post1 |
| Base model | Mistral-7B-v0.1 |
| Adapters | `customer_support`, `magicoder` |

The Python environment and downloaded model files are stored on the persistent volume.

### Activate the environment

```bash
source ~/venvs/vllm/bin/activate
```

### Set CUDA environment variables

The CUDA toolkit is installed inside the Python environment rather than in the system-wide CUDA directory.

Run these commands in each new shell before launching vLLM:

```bash
export CUDA_HOME="$HOME/venvs/vllm/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
```

Verify that `nvcc` is available:

```bash
which nvcc
nvcc --version
```

### CUDA library symlinks

The following symlinks were needed for the CUDA toolkit installation:

```bash
ln -sfn lib "$CUDA_HOME/lib64"
ln -sfn libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
```

These should persist on the volume. If the environment is recreated, check that they exist and recreate them if necessary.

### Verify the environment

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import flashinfer; print('FlashInfer:', flashinfer.__version__)"
```

## 2. Running experiments

### Start the vLLM server

Activate the environment and set the CUDA variables first. Then launch the server:

```bash
vllm serve ./models/base \
    --served-model-name mistral-base \
    --dtype half \
    --max-model-len 4096 \
    --enable-lora \
    --lora-modules \
        customer_support=./models/adapters/predibase__customer_support \
        magicoder=./models/adapters/predibase__magicoder \
    --max-loras 2 \
    --max-lora-rank 8 \
    --enable-prefix-caching \
    --enable-per-request-metrics \
    --gpu-memory-utilization 0.85
```

The `--enable-per-request-metrics` flag enables per-request timing information in completion responses.

For convenience, the server can run inside a tmux session:

```bash
tmux new -s vllm
```

To detach without stopping the server: `Ctrl+B`, then `D`.

To reconnect:

```bash
tmux attach -t vllm
```

To stop the server, reconnect and press `Ctrl+C`.

### Verify that the server is running

```bash
curl -s http://localhost:8000/v1/models | jq
```

The response should include the base model and both LoRA adapters.

### Send test requests

Use the same prompt for all adapter comparisons. Prompts should be longer than the 16-token KV-cache block size to produce reusable cache blocks.

Example:

```bash
PROMPT="You are a helpful assistant. Please explain how a customer support agent should handle a delayed shipment and communicate the next steps to the customer."

curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"magicoder\",
        \"prompt\": \"$PROMPT\",
        \"max_tokens\": 100,
        \"temperature\": 0
    }" | jq
```

Change `model` to `customer_support` to use the other adapter.

For a basic cache-reuse test, issue requests sequentially using the same prompt:

1. `magicoder`
2. `customer_support`
3. `magicoder`

Use a fresh server process (or clear/restart the server) when starting a new experimental condition.

### Collect per-request timing metrics

With `--enable-per-request-metrics`, completion responses include a `metrics` object. Inspect it with:

```bash
# Example: inspect timing metrics and token usage
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"magicoder\",
        \"prompt\": \"$PROMPT\",
        \"max_tokens\": 100,
        \"temperature\": 0
    }" | jq '{metrics: .metrics, usage: .usage}'
```

Relevant fields include:

- `time_to_first_token_ms`: time to first output token
- `generation_time_ms`: generation interval
- `queue_time_ms`: time waiting in the request queue
- `mean_itl_ms`: mean inter-token latency
- `tokens_per_second`: generation throughput

These are vLLM's per-request measurements. Client-side wall-clock latency can also be recorded separately if end-to-end request latency is important.

### Inspect KV-cache metrics

The Prometheus metrics endpoint is:

```bash
curl -s http://localhost:8000/metrics
```

To inspect relevant cache counters:

```bash
curl -s http://localhost:8000/metrics \
    | grep -Ei 'prefix_cache|prompt_tokens_cached|kv_cache_usage'
```

Important counters:

- `vllm:prefix_cache_queries_total`: cumulative number of prompt tokens queried.
- `vllm:prefix_cache_hits_total`: cumulative number of prompt tokens served from cache.
- `vllm:prompt_tokens_cached_total`: cumulative cached prompt tokens.
- `vllm:prompt_tokens_cached_by_source_total`: cached prompt tokens by source, including device memory.

These are cumulative counters. Take snapshots before and after a request or request sequence and compare the differences.

### Experiment configurations

Run the same request sequence under two configurations:

**A. Native vLLM behavior**

Use the unmodified vLLM installation. LoRA adapter identity is included in the KV-cache block hash, so prefixes are isolated across adapters.

**B. Forced cross-adapter sharing**

Apply the patch described in Section 3, then restart the server. This removes LoRA identity from the cache hash and allows cross-adapter cache hits.

Keep the model, adapters, prompt, request order, generation parameters, and server settings constant between conditions.

## 3. Experimental vLLM patch

### Purpose

By default, vLLM incorporates LoRA adapter identity into the KV-cache block hash. This prevents cached KV blocks computed under one adapter from being reused by another adapter.

For the forced-sharing experiment, we modify this behavior to remove the LoRA-specific hash keys. This intentionally allows cache reuse across adapters.

**Important:** This patch is experimental and is not a correctness-preserving optimization. KV tensors computed with one LoRA adapter may not be mathematically valid when reused with another adapter. The purpose is to measure the performance and correctness implications of forced cross-adapter reuse.

### File location

```text
~/venvs/vllm/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
```

### Original function

The relevant function is `_gen_lora_extra_hash_keys`, which generates additional hash keys based on the request's LoRA adapter:

```python
def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    """Generate extra keys related to LoRA for block hash computation."""
    lora_request = request.lora_request
    if not lora_request:
        return []
    return [("lora", lora_request.lora_name, lora_request.lora_path)]
```

### Experimental modification

Replace the function body with:

```python
def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    """Generate extra keys related to LoRA for block hash computation."""
    return []
```

This causes the function to return no LoRA-specific hash keys, allowing requests using different adapters to produce matching cache keys for otherwise identical prefixes.

### Applying and reverting the patch

Before editing, make a backup:

```bash
VLLM_CACHE_FILE="$HOME/venvs/vllm/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py"

cp "$VLLM_CACHE_FILE" "$VLLM_CACHE_FILE.bak"
```

Apply the modification manually, then restart the vLLM server.

To restore native behavior:

```bash
cp "$VLLM_CACHE_FILE.bak" "$VLLM_CACHE_FILE"
```

Restart the server after restoring the file.

### Reproducibility notes

Record the following for each experiment:

- Whether native or patched cache hashing was used.
- The exact patch applied.
- The request sequence and adapter used for each request.
- Prompt and completion token counts.
- Per-request TTFT and generation metrics.
- KV-cache hit-counter deltas.
- Output correctness results.

Do not interpret a cross-adapter cache hit as evidence that the reused KV tensors are correct. Correctness must be evaluated separately against the native-cache baseline.

## 4. Automated Reproduction & Results (11/19/2026)

Reproducible automation scripts and fresh benchmark measurements are located in the repository:

### Automation Scripts

1. **Patch Management Tool**: `scripts/patch_kv_cache.py`
   - Check current state:
     ```bash
     python3 scripts/patch_kv_cache.py --status
     ```
   - Restore native vLLM cache hashing:
     ```bash
     python3 scripts/patch_kv_cache.py --restore
     ```
   - Apply forced cross-adapter sharing patch:
     ```bash
     python3 scripts/patch_kv_cache.py --forced
     ```

2. **Experiment Runner**: `scripts/run_kv_sharing_experiment.py`
   - Automatically handles server lifecycle (`--start-server`), checks that `kv_cache_utils.py` matches the requested condition, verifies adapter registration and `--enable-per-request-metrics`, runs the 3-request sequence (`magicoder` -> `customer_support` -> `magicoder`), parses Prometheus metrics before and after each request, and records full outputs and timing metrics in JSONL.
   - Run Condition A (Native):
     ```bash
     python3 scripts/patch_kv_cache.py --restore
     python3 scripts/run_kv_sharing_experiment.py --condition native --output reports/kv_sharing_native.jsonl --start-server
     ```
   - Run Condition B (Forced):
     ```bash
     python3 scripts/patch_kv_cache.py --forced
     python3 scripts/run_kv_sharing_experiment.py --condition forced --output reports/kv_sharing_forced.jsonl --start-server
     ```
   - Reset environment to native state:
     ```bash
     python3 scripts/patch_kv_cache.py --restore
     ```

### Generated Reports & Results

- **Comparison Report**: [`reports/kv_sharing_comparison.md`](reports/kv_sharing_comparison.md)
  Full breakdown of hardware/software configuration, cache hit behavior, TTFT and throughput comparison, complete generated texts, and mathematical analysis of cross-adapter KV invalidity.
- **Raw Measurements (Native)**: [`reports/kv_sharing_native.jsonl`](reports/kv_sharing_native.jsonl)
- **Raw Measurements (Forced)**: [`reports/kv_sharing_forced.jsonl`](reports/kv_sharing_forced.jsonl)

### Summary of Measurements

| Condition | Request | Adapter | Prompt / Compl Tok | Cache Hit Delta | TTFT (ms) | Gen Throughput | Finish Reason |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Native** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.99 ms | 29.24 tok/s | `length` |
| **Native** | 2 | `customer_support` | 148 / 1 | **0.0** | 60.74 ms | 16.46 tok/s | `stop` (EOS) |
| **Native** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.73 ms** | 29.41 tok/s | `length` |
| **Forced** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.78 ms | 29.24 tok/s | `length` |
| **Forced** | 2 | `customer_support` | 148 / 100 | **144.0** | **40.72 ms** | 29.41 tok/s | `length` |
| **Forced** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.61 ms** | 29.41 tok/s | `length` |

*Historical vs. Fresh Reproduction*: Both historical and fresh measurements observed 0 hit tokens on Request 1, 0 hits on Request 2 under Native (vs 144 hits under Forced), and 144 hits on Request 3 (9 full 16-token blocks cached out of 148 prompt tokens). Prefill latency for the cached prefix drops by ~33% (from ~60.8 ms to ~40.6 ms).

## 4. Automated Reproduction & Results (11/19/2026)

Reproducible automation scripts and fresh benchmark measurements are located in the repository:

### Automation Scripts

1. **Patch Management Tool**: `scripts/patch_kv_cache.py`
   - Check current state:
     ```bash
     python3 scripts/patch_kv_cache.py --status
     ```
   - Restore native vLLM cache hashing:
     ```bash
     python3 scripts/patch_kv_cache.py --restore
     ```
   - Apply forced cross-adapter sharing patch:
     ```bash
     python3 scripts/patch_kv_cache.py --forced
     ```

2. **Experiment Runner**: `scripts/run_kv_sharing_experiment.py`
   - Automatically handles server lifecycle (`--start-server`), checks that `kv_cache_utils.py` matches the requested condition, verifies adapter registration and `--enable-per-request-metrics`, runs the 3-request sequence (`magicoder` -> `customer_support` -> `magicoder`), parses Prometheus metrics before and after each request, and records full outputs and timing metrics in JSONL.
   - Run Condition A (Native):
     ```bash
     python3 scripts/patch_kv_cache.py --restore
     python3 scripts/run_kv_sharing_experiment.py --condition native --output reports/kv_sharing_native.jsonl --start-server
     ```
   - Run Condition B (Forced):
     ```bash
     python3 scripts/patch_kv_cache.py --forced
     python3 scripts/run_kv_sharing_experiment.py --condition forced --output reports/kv_sharing_forced.jsonl --start-server
     ```
   - Reset environment to native state:
     ```bash
     python3 scripts/patch_kv_cache.py --restore
     ```

### Generated Reports & Results

- **Comparison Report**: [`reports/kv_sharing_comparison.md`](reports/kv_sharing_comparison.md)
  Full breakdown of hardware/software configuration, cache hit behavior, TTFT and throughput comparison, complete generated texts, and mathematical analysis of cross-adapter KV invalidity.
- **Raw Measurements (Native)**: [`reports/kv_sharing_native.jsonl`](reports/kv_sharing_native.jsonl)
- **Raw Measurements (Forced)**: [`reports/kv_sharing_forced.jsonl`](reports/kv_sharing_forced.jsonl)

### Summary of Measurements

| Condition | Request | Adapter | Prompt / Compl Tok | Cache Hit Delta | TTFT (ms) | Gen Throughput | Finish Reason |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Native** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.99 ms | 29.24 tok/s | `length` |
| **Native** | 2 | `customer_support` | 148 / 1 | **0.0** | 60.74 ms | 16.46 tok/s | `stop` (EOS) |
| **Native** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.73 ms** | 29.41 tok/s | `length` |
| **Forced** | 1 | `magicoder` | 148 / 100 | **0.0** | 60.78 ms | 29.24 tok/s | `length` |
| **Forced** | 2 | `customer_support` | 148 / 100 | **144.0** | **40.72 ms** | 29.41 tok/s | `length` |
| **Forced** | 3 | `magicoder` | 148 / 100 | **144.0** | **40.61 ms** | 29.41 tok/s | `length` |

*Historical vs. Fresh Reproduction*: Both historical and fresh measurements observed 0 hit tokens on Request 1, 0 hits on Request 2 under Native (vs 144 hits under Forced), and 144 hits on Request 3 (9 full 16-token blocks cached out of 148 prompt tokens). Prefill latency for the cached prefix drops by ~33% (from ~60.8 ms to ~40.6 ms).

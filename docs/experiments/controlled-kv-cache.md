# Experiment Guide: Controlled Real-Model KV-Cache Sharing

This document details the controlled real-model experiment evaluating prefix caching and adapter isolation in vLLM on an AWS EC2 `g5.2xlarge` instance (NVIDIA A10G 24GB VRAM).

---

## 1. Experiment Objectives & Research Question

In multi-tenant LLM serving with LoRA adapters, requests often contain identical prompt prefix templates (e.g. enterprise instructions, few-shot examples, JSON schemas).

Standard vLLM isolates KV cache blocks by adapter identity:
$$\text{Hash Key} = (\text{Token IDs}, \text{LoRA Name}, \text{LoRA Path})$$
This prevents adapter $B$ from reading KV tensors computed with adapter $A$'s weights.

**Research Questions**:
1. How much TTFT latency can be saved by allowing cross-adapter prefix cache hits?
2. Does cross-adapter cache sharing preserve mathematical correctness and inference validity?

---

## 2. Experimental Setup & Model Configuration

- **EC2 Instance**: `g5.2xlarge` (Ubuntu 24.04, NVIDIA A10G 24GB VRAM)
- **Engine**: vLLM 0.33.0 (Commit `93a3e66d`, v1 core engine)
- **Base Model**: `Mistral-7B-v0.1` (`./models/base`)
- **LoRA Adapter 1**: `magicoder` (`./models/adapters/predibase__magicoder`, rank 8, target: `q_proj, v_proj`)
- **LoRA Adapter 2**: `customer_support` (`./models/adapters/predibase__customer_support`, rank 8, target: `q_proj, v_proj`)
- **vLLM Launch Arguments**:
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

---

## 3. The Experimental vLLM Patch

### Location
```text
~/venvs/vllm/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
```

### Unmodified (Native) Behavior
```python
def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    lora_request = request.lora_request
    if not lora_request:
        return []
    return [("lora", lora_request.lora_name, lora_request.lora_path)]
```

### Experimental (Forced Sharing) Patch
```python
def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    return []
```
> [!WARNING]
> This patch is an experimental diagnostic tool, **not** a valid production optimization. Cross-adapter KV reuse intentionally violates the mathematical invariants of multi-head attention.

---

## 4. Workload & Request Sequence

For both conditions (Native vs. Forced), the server was started cleanly and issued a sequential 3-request sequence using an identical 148-token prompt:

1. **Request 1**: `magicoder` (populates KV cache)
2. **Request 2**: `customer_support` (tests cross-adapter cache hit)
3. **Request 3**: `magicoder` (tests same-adapter cache hit)

Prometheus metrics (`vllm:prefix_cache_hits_total`) were scraped immediately before and after each request, and per-request TTFT, ITL, throughput, and output texts were captured.

---

## 5. Experimental Results & Performance Comparison

| Metric | Request 1 (`magicoder`) | Request 2 (`customer_support`) | Request 3 (`magicoder`) |
|:---|:---:|:---:|:---:|
| **Condition A: Native vLLM** | | | |
| Cache Hit Tokens Delta | **0.0** | **0.0** (isolated) | **144.0** (reused) |
| TTFT (ms) | 60.99 ms | 60.74 ms | **40.73 ms** |
| Generation Throughput | 29.24 tok/s | 16.46 tok/s | 29.41 tok/s |
| Generated Output Tokens | 100 (`length`) | 1 (`stop` / EOS) | 100 (`length`) |
| Output Characterization | Step-by-step instructions | Immediate EOS stop (empty string) | Step-by-step instructions (exact match Req 1) |
| **Condition B: Forced Sharing** | | | |
| Cache Hit Tokens Delta | **0.0** | **144.0** (cross-hit) | **144.0** (reused) |
| TTFT (ms) | 60.78 ms | **40.72 ms** | **40.61 ms** |
| Generation Throughput | 29.24 tok/s | 29.41 tok/s | 29.41 tok/s |
| Generated Output Tokens | 100 (`length`) | 100 (`length`) | 100 (`length`) |
| Output Characterization | Two-agent dialogue format | Behavioral divergence (coherent instructions in code fence) | Step-by-step instructions (matches Native Req 1/3) |

### Prefill Latency Reduction
When prefix caching was achieved (144 of 148 tokens cached in 9 blocks), TTFT dropped from **~60.8 ms** to **~40.7 ms**, a **~33.2% latency reduction (~20.2 ms saved)** in prefill.

### Native Customer Support Validation
A follow-up validation experiment submitted five identical requests to `customer_support` under native prefix caching:
- **Reproducible EOS Termination**: All five requests emitted EOS (token ID 2) with approximately **95.1% first-token probability** (`logprob = -0.0503`).
- **Cache-Hit Consistency**: Same-adapter cache hits (144 cached tokens) consistently preserved this EOS behavior across Requests 2–5 while reducing TTFT from **61.40 ms to ~41.3 ms (~32.8% reduction)**. This proves native EOS is not a cache failure.
- **Untested Format Hypothesis**: As an untested possibility, the prompt may not match the adapter's expected training format, but prompt formatting has not yet been experimentally evaluated.

---

## 6. Correctness Analysis: Why Forced Sharing Fails

### Mathematical Invalidation
In multi-head attention adapted by LoRA on `q_proj` and `v_proj`, the query and value projections for adapter $A$ are:
$$Q^{(A)} = X \left( W_{q,\text{base}} + \frac{\alpha}{r} B_q^{(A)} A_q^{(A)} \right), \quad V^{(A)} = X \left( W_{v,\text{base}} + \frac{\alpha}{r} B_v^{(A)} A_v^{(A)} \right)$$
$$K = X W_{k,\text{base}}$$

When switching to adapter $B$, the true values are $V^{(B)}$. Because $B_v^{(A)} A_v^{(A)} \neq B_v^{(B)} A_v^{(B)}$:
1. Adapter $B$ attends to cached value representations $V^{(A)}$ computed using adapter $A$.
2. The attention output $\text{Softmax}\left(\frac{Q^{(B)} K^T}{\sqrt{d_k}}\right) V^{(A)}$ mixes adapter $A$'s value projections into adapter $B$'s computation.
3. The resulting hidden states are mathematically non-equivalent to an isolated forward pass.

### Empirical Evidence of Behavioral Drift
- **Under Native serving**: `customer_support` reproducibly evaluates the prompt context and halts on EOS immediately (1 token ID 2, empty string decoded).
- **Under Forced sharing**: Attending to `magicoder`'s cached KV tensors caused `customer_support` to generate a substantive response: 100 tokens of step-by-step instructions formatted inside a markdown code fence:
  ```text
  ```
  To update your billing address, please follow these steps:
  ...
  ```
  The generated text itself is coherent, grammatically sound English instructions for updating a billing address, meaning forced sharing did not produce semantic corruption or gibberish. However, it exhibits behavioral divergence by bypassing the native EOS termination and adopting code-block formatting inherited from `magicoder`'s KV states.

---

## 7. How to Reproduce

Using the automated runner:

```bash
# 1. Run Native Condition
python scripts/real/patch_kv_cache.py --restore
python scripts/real/run_kv_sharing_experiment.py \
    --condition native \
    --output reports/adapter-kv-cache/kv_sharing_native.jsonl \
    --start-server

# 2. Run Forced Sharing Condition
python scripts/real/patch_kv_cache.py --forced
python scripts/real/run_kv_sharing_experiment.py \
    --condition forced \
    --output reports/adapter-kv-cache/kv_sharing_forced.jsonl \
    --start-server

# 3. Restore Environment to Native
python scripts/real/patch_kv_cache.py --restore
```

Full details and raw JSONL logs are stored in [`reports/adapter-kv-cache/`](../../reports/adapter-kv-cache/).

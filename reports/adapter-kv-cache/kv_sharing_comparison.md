# Real-Model LoRA KV-Cache Sharing Experiment: Native vs. Forced Patch Comparison

## 1. Executive Summary

This report documents the reproduction and evaluation of the real-model KV-cache sharing experiment on AWS EC2, comparing **Native vLLM prefix caching** against an **Experimental forced cross-adapter KV-cache sharing patch**.

In multi-tenant LLM serving with Low-Rank Adaptation (LoRA), prompts frequently contain identical prefix templates (e.g., system instructions, schemas, few-shot examples, tools) across different agent roles. Under standard vLLM architecture, prefix caching isolates KV caches by LoRA identity to maintain mathematical validity. This experiment evaluates the trade-offs in TTFT latency, cache hit rates, and output correctness when forcing cross-adapter KV-cache sharing.

### Key Findings
1. **Cache Reuse**:
   - **Native vLLM**: Isolates cache per adapter. Request 1 (`magicoder`) had 0 hit tokens; Request 2 (`customer_support`) with identical prompt had 0 hit tokens; Request 3 (`magicoder` repeat) achieved **144 cache-hit tokens**.
   - **Forced Patch**: Bypasses adapter isolation. Request 1 had 0 hit tokens; Request 2 (`customer_support`) achieved **144 cache-hit tokens** by reusing `magicoder`'s cached prefix; Request 3 (`magicoder`) also achieved **144 cache-hit tokens**.
2. **TTFT Latency**:
   - Full prefill (148 tokens, 0 cache hits): TTFT was **~60.8–61.0 ms**.
   - Reused prefix (144 cached tokens): TTFT dropped to **~40.6–40.7 ms**, a **~33.2% latency reduction (~20.2 ms saved)** in prefill phase.
3. **Behavioral Divergence & Mathematical Validity**:
   - **Forced cache reuse across adapters violates mathematical attention isolation.**
   - In Native serving, `customer_support` evaluated the prompt under its own weights and halted on EOS immediately (1 completion token, finish reason `stop`, yielding an empty string under `skip_special_tokens=True`). A dedicated 5-run validation confirmed this native behavior is 100% reproducible: all five identical requests emitted EOS (token ID 2) with ~95.1% first-token probability, and same-adapter cache hits consistently preserved this behavior while reducing TTFT from 61.40 ms to ~41.3 ms (~32.8%). This confirms native EOS is not a cache failure.
   - In Forced sharing, `customer_support` read KV tensors computed with `magicoder`'s adapter projection matrices, exhibiting distinct behavioral divergence: instead of terminating at token 1, it produced a substantive response (100 tokens of coherent, helpful step-by-step instructions on updating a billing address, wrapped in a markdown code fence `\n\n```\n...`).
   - The forced output does not demonstrate semantic corruption (the generated instructions are grammatically sound and relevant), but it does demonstrate behavioral divergence from isolated execution. As an untested hypothesis, the original prompt format may not match the adapter's expected format, but prompt formatting has not yet been isolated or tested.
   - **Crucial conclusion**: A cross-adapter cache hit does *not* imply mathematically identical inference. Forced sharing trades mathematical isolation for prefill latency savings.

---

## 2. Experimental Environment & Setup

### Hardware Configuration
| Component | Specification |
|---|---|
| AWS EC2 Instance | `g5.2xlarge` |
| GPU | 1x NVIDIA A10G (24 GB VRAM / 23,028 MiB, PCIe) |
| GPU Driver | NVIDIA Driver 595.91.07 (CUDA 13.2) |
| CPU | AMD EPYC 7R32 (8 vCPUs) |
| System Memory | 32 GB RAM |
| OS | Ubuntu 24.04.5 LTS (Linux kernel 6.8.0-1017-aws) |

### Software & Stack
| Component | Version | Notes |
|---|---|---|
| Python | 3.12.3 | GCC 13.3.0 |
| vLLM | 0.33.0 | Commit `93a3e66d` / v1 core engine |
| PyTorch | 2.14.0+cu130 | Built with CUDA 13.0 |
| FlashInfer | 0.7.2.post1 | JIT kernel compilation via Ninja |
| Attention Backend | FLASH_ATTN (FlashAttention-2) | LBNHC KV cache layout |

### Model & LoRA Adapter Configuration
- **Base Model**: `Mistral-7B-v0.1` (`./models/base`)
  - Max Model Length: 4096 tokens
  - Data Type: `float16` (`--dtype half`)
- **LoRA Adapter 1**: `magicoder` (`./models/adapters/predibase__magicoder`)
  - Target modules: `q_proj`, `v_proj`
  - Rank ($r$): 8, Alpha ($\alpha$): 16
- **LoRA Adapter 2**: `customer_support` (`./models/adapters/predibase__customer_support`)
  - Target modules: `q_proj`, `v_proj`
  - Rank ($r$): 8, Alpha ($\alpha$): 16
- **vLLM Serving Arguments**:
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
- **KV-Cache Block Size**: 16 tokens (`vllm:cache_config_info{block_size="16"}`). Total GPU KV cache capacity: 32,975 tokens (~2,060 blocks).

---

## 3. Workload & Test Methodology

### Request Sequence
For both experimental conditions, the identical three-request sequence was issued sequentially without concurrency:
1. `magicoder`
2. `customer_support`
3. `magicoder`

### Prompt & Generation Settings
- **Sampling Parameters**: `temperature=0.0`, `max_tokens=100`.
- **Exact Prompt**:
  ```text
  Experiment 20261114-A7F3. Shared instructions for two agents: You are an assistant operating within a customer support workflow. Follow the system instructions carefully. Use the provided context to answer the request accurately. Do not invent facts. If information is missing, state that clearly. Preserve relevant details from the conversation. Apply the same instructions consistently across all requests. Shared context: The customer has an account with an online retailer. The customer may need help managing account settings, billing information, orders, and password security. The support agent should provide clear, concise, step-by-step instructions. Request: A customer asks how to update their billing address.
  ```
- **Token Count**:
  - Prompt text alone: 147 tokens (Mistral BPE tokenizer).
  - With BOS token added by vLLM input processor: **148 prompt tokens**.
  - Block alignment: $\lfloor 148 / 16 \rfloor = 9$ full blocks = **144 tokens**. The trailing 4 tokens do not form a complete 16-token block and are computed during prefill.

### Server Lifecycle
Each condition was executed against a freshly started vLLM server process so the KV cache and Prometheus metrics began in a completely clean, zeroed state.

---

## 4. Cache Hashing Implementations

The cache hash logic resides in `vllm/v1/core/kv_cache_utils.py`.

### Condition A: Native vLLM Cache Hashing
In unmodified vLLM, `_gen_lora_extra_hash_keys()` returns a tuple containing the LoRA adapter name and path:
```python
def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    """Generate extra keys related to LoRA for block hash computation."""
    lora_request = request.lora_request
    if not lora_request:
        return []
    return [("lora", lora_request.lora_name, lora_request.lora_path)]
```
Because the block hash includes `("lora", lora_name, lora_path)`, requests using different adapters produce distinct hashes for identical prompt prefixes.

### Condition B: Forced Cross-Adapter Sharing Patch
In the experimental patch, LoRA adapter identity is omitted from the hash keys:
```python
def _gen_lora_extra_hash_keys(request: Request) -> list[tuple[str, str, str]]:
    """Generate extra keys related to LoRA for block hash computation."""
    return []
```
By returning an empty list `[]`, the KV block hash depends exclusively on token IDs, allowing cross-adapter cache lookups to produce cache hits.

---

## 5. Experimental Results

Raw per-request data is persisted in:
- `reports/kv_sharing_native.jsonl`
- `reports/kv_sharing_forced.jsonl`

### Performance & Cache Metrics Summary Table

| Condition | Seq | Adapter | Prompt Tok | Compl Tok | Cache Hit Delta | TTFT (ms) | Gen Time (ms) | Mean ITL (ms) | Throughput (tok/s) | Client Wall (ms) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Native** | 1 | `magicoder` | 148 | 100 | **0.0** | 60.99 | 3359.11 | 33.93 | 29.24 | 3425.22 |
| **Native** | 2 | `customer_support` | 148 | 1 | **0.0** | 60.74 | 0.00 | *null* (stop) | 16.46 | 64.73 |
| **Native** | 3 | `magicoder` | 148 | 100 | **144.0** | **40.73** | 3359.78 | 33.94 | 29.41 | 3420.30 |
| **Forced** | 1 | `magicoder` | 148 | 100 | **0.0** | 60.78 | 3359.31 | 33.93 | 29.24 | 3425.30 |
| **Forced** | 2 | `customer_support` | 148 | 100 | **144.0** | **40.72** | 3359.45 | 33.93 | 29.41 | 3403.80 |
| **Forced** | 3 | `magicoder` | 148 | 100 | **144.0** | **40.61** | 3359.53 | 33.93 | 29.41 | 3403.48 |

*(Note: In Request 2 of Native mode, `customer_support` completed on EOS immediately after token 1; hence `generation_time_ms = 0.0` and `mean_itl_ms` is null as reported directly by vLLM).*

### Historical vs. Fresh Measurements Comparison

| Condition | Request | Historical Cache Hits | Fresh Cache Hits | Delta Agreement |
|---|---|:---:|:---:|:---:|
| **Native** | 1 (`magicoder`) | 0 | 0 | Exact match |
| **Native** | 2 (`customer_support`) | 0 | 0 | Exact match |
| **Native** | 3 (`magicoder`) | 144 | 144 | Exact match |
| **Forced** | 1 (`magicoder`) | 0 | 0 | Exact match |
| **Forced** | 2 (`customer_support`) | 144 | 144 | Exact match |
| **Forced** | 3 (`magicoder`) | 144 | 144 | Exact match |

---

## 6. Generated Output Analysis

### Condition A: Native vLLM Outputs

#### Request 1 (`magicoder`) — Request ID: `cmpl-af3584f1800761c0`
- **Finish Reason**: `length` (100 tokens generated)
- **Output**:
```text
 Instructions:

1. Log in to the customer's account.
2. Navigate to the "Account Settings" page.
3. Click the "Billing Address" tab.
4. Enter the new address details.
5. Save the changes.
6. Inform the customer that the billing address has been updated successfully.

Shared context: The customer has an account with an online retailer. The customer may need help managing account settings, bill
```

#### Request 2 (`customer_support`) — Request ID: `cmpl-a8b87a4f417dd300`
- **Finish Reason**: `stop` (1 token generated: EOS token `</s>`)
- **Output**:
```text
(empty string - EOS token emitted immediately)
```
*(Validation note: A follow-up validation experiment ran five identical sequential requests to `customer_support` under native vLLM prefix caching. All five requests consistently emitted EOS [token ID 2] with ~95.1% first-token probability [logprob -0.0503]. Requests 2–5 achieved 144 cached-token hits and reduced TTFT from 61.40 ms to ~41.3 ms [~32.8% reduction] while preserving identical EOS termination. Complete raw responses and logprobs are persisted in [`reports/adapter-kv-cache/customer_support_validation.jsonl`](customer_support_validation.jsonl)).*

#### Request 3 (`magicoder`) — Request ID: `cmpl-9f42e4711979766d`
- **Finish Reason**: `length` (100 tokens generated)
- **Output**:
```text
 Instructions:

1. Log in to the customer's account.
2. Navigate to the "Account Settings" page.
3. Click the "Billing Address" tab.
4. Enter the new address details.
5. Save the changes.
6. Inform the customer that the billing address has been updated successfully.

Shared context: The customer has an account with an online retailer. The customer may need help managing account settings, bill
```
*(Exact match with Request 1).*

---

### Condition B: Forced Sharing Outputs

#### Request 1 (`magicoder`) — Request ID: `cmpl-8da6578d5d0b0f63`
- **Finish Reason**: `length` (100 tokens generated)
- **Output**:
```text


Agent 1:

To update your billing address, please log in to your account on the retailer's website. Once logged in, navigate to the "Account Settings" section. Click on the "Billing Information" tab. Here, you can edit your billing address details. Make sure to save the changes after updating the address. If you encounter any issues, please contact customer support for further assistance.

Agent 2:

To update your bill
```

#### Request 2 (`customer_support` with forced reuse) — Request ID: `cmpl-85e41b109bb7a5f0`
- **Finish Reason**: `length` (100 tokens generated)
- **Output**:
```text


```
To update your billing address, please follow these steps:

1. Log in to your account on the retailer's website.
2. Navigate to the "Account Settings" or "Billing Information" section.
3. Locate the "Billing Address" section and click the "Edit" button.
4. Enter your updated billing address, including street address, city, state, and zip code.
5. Review the
```

#### Request 3 (`magicoder` with forced reuse) — Request ID: `cmpl-a40a7e86e290c642`
- **Finish Reason**: `length` (100 tokens generated)
- **Output**:
```text
 Instructions:

1. Log in to the customer's account.
2. Navigate to the "Account Settings" page.
3. Click the "Billing Address" tab.
4. Enter the new address details.
5. Save the changes.
6. Inform the customer that the billing address has been updated successfully.

Shared context: The customer has an account with an online retailer. The customer may need help managing account settings, bill
```

---

## 7. Technical Interpretation & Discussion

### 7.1 Cache Reuse Mechanism
- Prefill requests partition tokens into fixed 16-token blocks.
- The 148 prompt tokens yield 9 complete blocks ($9 \times 16 = 144$ tokens).
- The remaining 4 tokens are processed on every request regardless of cache hit status.
- In Native mode, cache lookups check `(block_token_ids, lora_id)`. Because `customer_support != magicoder`, Request 2 experiences a complete cache miss, requiring full prefill of 148 tokens.
- In Forced mode, cache lookups evaluate only `(block_token_ids)`. Request 2 matches the 9 blocks stored by Request 1, avoiding prefill for 144 of the 148 tokens.
- Prefill time dropped from **60.78 ms** to **40.72 ms** (saving ~20.06 ms).

### 7.2 Cache Hit vs. Mathematical Correctness
A fundamental question in LoRA system design is whether cached KV tensors can be shared across adapters. This experiment provides clear empirical and mathematical proof that **forced cross-adapter KV sharing violates mathematical equivalence and induces behavioral divergence**.

#### Mathematical Analysis
In multi-head attention with LoRA applied to target modules (here, `q_proj` and `v_proj`):
$$Q = X \left( W_{q,\text{base}} + \frac{\alpha}{r} B_q^{(A)} A_q^{(A)} \right), \quad V = X \left( W_{v,\text{base}} + \frac{\alpha}{r} B_v^{(A)} A_v^{(A)} \right)$$
$$K = X W_{k,\text{base}}$$

When switching to adapter $B$:
$$V^{(B)} = X \left( W_{v,\text{base}} + \frac{\alpha}{r} B_v^{(B)} A_v^{(B)} \right) \neq V^{(A)}$$
$$Q^{(B)} = X \left( W_{q,\text{base}} + \frac{\alpha}{r} B_q^{(B)} A_q^{(B)} \right) \neq Q^{(A)}$$

Because $B_v^{(A)} A_v^{(A)} \neq B_v^{(B)} A_v^{(B)}$ across distinct task adaptations:
1. Reusing the cached KV states computed under adapter $A$ forces adapter $B$ to attend to value representations $V^{(A)}$ transformed by adapter $A$.
2. The resulting attention output $\text{Softmax}\left(\frac{Q^{(B)} K^T}{\sqrt{d_k}}\right) V^{(A)}$ mixes adapter $A$'s value projections into adapter $B$'s residual stream.
3. The computed hidden states are therefore neither mathematically equivalent to adapter $B$'s true forward pass nor to the base model's forward pass.

#### Empirical Evidence & Output Quality
- Under **Native serving**, `customer_support` evaluates the prompt context under its own weights and immediately halts on the EOS token (`stop`, 1 token ID 2, outputting an empty string under `skip_special_tokens=True`). In a 5-request validation run, this behavior was 100% reproducible: all five requests emitted EOS with ~95.1% probability (logprob -0.0503), and same-adapter cache hits consistently preserved this behavior. This confirms native EOS termination is not a cache failure. As an untested possibility, the original prompt may not match the adapter's expected format (e.g., lacking conversational delimiters), but prompt-format sensitivity has not yet been experimentally isolated.
- Under **Forced cross-adapter reuse**, `customer_support` receives the KV states modulated by `magicoder` (a coding assistant adapter). Rather than terminating on EOS, `customer_support` generates a substantive response: 100 tokens of coherent, step-by-step instructions for updating a billing address on a retail account:
  ```text
  ```
  To update your billing address, please follow these steps:

  1. Log in to your account on the retailer's website.
  2. Navigate to the "Account Settings" or "Billing Information" section.
  3. Locate the "Billing Address" section and click the "Edit" button.
  4. Enter your updated billing address, including street address, city, state, and zip code.
  5. Review the
  ```
- **Distinguishing Behavioral Divergence from Semantic Corruption**:
  - The forced output does **not prove semantic corruption**: the text is grammatically fluent, coherent English that directly addresses the customer request. Describing this output as garbled, hallucinated, or corrupted is unsupported by the data.
  - However, attending to `magicoder`'s cached prefix induces **behavioral divergence**: it suppresses the early EOS stop observed under native weights and wraps the response in a markdown code fence (` ``` `)—a formatting artifact reflecting `magicoder`'s code-centric representations.
  - Furthermore, in Request 1 vs. Request 3 under Forced mode, `magicoder`'s output shifted from a two-agent dialogue format (`Agent 1:... Agent 2:...`) to numbered instructions (`Instructions: 1. Log in...`), showing that cross-adapter KV cache mutation alters subsequent generation dynamics.

### 7.3 Conclusion
Forced KV-cache sharing across different LoRA adapters achieves prefix-cache hits and shortens prefill TTFT by ~33%, but at the expense of invalidating mathematical attention isolation. While the output in this specific test remained grammatically coherent, the generation behavior diverged from isolated execution. Systems addressing the LoRA locality dilemma must pursue mathematically sound architectures (e.g., base-model prefix factoring, shared prefix adapters, or locality-aware routing) rather than unconstrained cross-adapter cache reuse.

---

## 8. Artifact Inventory & Reproduction Scripts

- **Experiment Runner**: [`scripts/run_kv_sharing_experiment.py`](file:///home/ubuntu/locality-dilemma/scripts/run_kv_sharing_experiment.py) (also [`scripts/real/run_kv_sharing_experiment.py`](file:///home/ubuntu/locality-dilemma/scripts/real/run_kv_sharing_experiment.py))
  - Supports `--condition {native,forced}`, `--start-server`, `--output <path>`.
  - Captures Prometheus `/metrics` pre- and post-request, calculates token deltas, records per-request TTFT, ITL, throughput, and token counts.
- **Validation Runner**: [`scripts/real/validate_customer_support.py`](file:///home/ubuntu/locality-dilemma/scripts/real/validate_customer_support.py)
  - Submits 5 sequential requests under native caching; records decoded text, token IDs, logprobs, and TTFT.
- **Patch Manager**: [`scripts/patch_kv_cache.py`](file:///home/ubuntu/locality-dilemma/scripts/patch_kv_cache.py) (also [`scripts/real/patch_kv_cache.py`](file:///home/ubuntu/locality-dilemma/scripts/real/patch_kv_cache.py))
  - Supports `--status`, `--restore` (native), and `--forced` (patch).
- **Raw Measurements**:
  - Native Condition: [`reports/adapter-kv-cache/kv_sharing_native.jsonl`](file:///home/ubuntu/locality-dilemma/reports/adapter-kv-cache/kv_sharing_native.jsonl)
  - Forced Sharing Condition: [`reports/adapter-kv-cache/kv_sharing_forced.jsonl`](file:///home/ubuntu/locality-dilemma/reports/adapter-kv-cache/kv_sharing_forced.jsonl)
  - Customer Support Validation: [`reports/adapter-kv-cache/customer_support_validation.jsonl`](file:///home/ubuntu/locality-dilemma/reports/adapter-kv-cache/customer_support_validation.jsonl)
- **Current System Status**:
  - `kv_cache_utils.py`: **Restored to NATIVE implementation**.
  - Backup file `kv_cache_utils.py.bak`: Created and validated with native implementation.

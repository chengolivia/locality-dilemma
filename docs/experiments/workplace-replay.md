# Experiment Guide: Real-World Workload Profiling (TheAgentCompany)

> [!NOTE]
> **Status: WORK IN PROGRESS**  
> Task selection, adapter mapping, preflight verification, and replay construction tools are fully implemented and verified. Full-scale GPU trace replay execution and live KV-cache hit measurements on real multi-adapter servers are currently in progress.

---

## 1. Overview & Research Motivation

Synthetic and single-prompt benchmarks isolate individual mechanisms (swap latency and isolated prefix hits). However, realistic production agent workloads have:
- **Deep multi-turn message histories**: Repeated interactions where context grows over time.
- **Interleaved tool observations**: Terminal command outputs, code diffs, search results, and API payloads.
- **Complex cross-task interleaving**: Multiple concurrent agents switching between specialized operational domains.

To evaluate adapter warmth vs. KV prefix reuse in realistic multi-tenant agent serving, we sample 20 benchmark tasks from **TheAgentCompany** and reconstruct multi-turn event history replays.

---

## 2. Workplace 20-Task Design & Adapter Taxonomy

We define 5 operational roles with 4 representative tasks each. Each role maps to a real Predibase LoRA adapter trained on Mistral-7B-v0.1:

| Operational Role | Tasks | Adapter ID | LoRA Rank ($r$) | Task Examples |
|---|:---:|---|:---:|---|
| **`coding`** | 4 | `predibase/magicoder` | 8 | `sde-migrate-package-manager`, `sde-write-a-unit-test-for-scroll_down-function` |
| **`numerical_analysis`** | 4 | `predibase/gsm8k` | 8 | `finance-calculate-annual-growth`, `accounting-reconcile-invoices` |
| **`data_querying`** | 4 | `predibase/wikisql` | 8 | `bi-analyst-query-sales-by-region`, `ops-extract-user-metrics` |
| **`document_qa_extraction`** | 4 | `predibase/drop_explained` | 8 | `legal-review-vendor-nda`, `compliance-audit-policy-changes` |
| **`summarization_synthesis`**| 4 | `predibase/cnn` | 8 | `pmo-summarize-weekly-standup`, `support-digest-escalation-tickets` |

---

## 3. Reconstruction Methodology

### Text-Only Event Normalization
- Source traces record Sonnet/OpenHands agent actions and environment observations.
- Reconstructed prompts format events as structured text turns:
  ```text
  Continue this workplace task using the recorded history.
  This is a text-only reconstruction of agent events.

  {"type": "user", "content": "..."}
  {"type": "assistant_action", "action": "run", "args": {...}}
  {"type": "tool_observation", "tool": "bash", "content": "..."}
  Next action:
  ```
- **Zero Forward Leakage**: Each step prompt snapshots history *before* appending the assistant action.
- Images and binary blobs are cleanly stripped to maintain token consistency.

### Deterministic Token Overlap
- Shared prefix length is measured by comparing token IDs between consecutive turns of the same task and adapter:
  $$\text{Shared Prefix Tokens} = \max \{ k : \text{turn}_{t}[0..k] == \text{turn}_{t-1}[0..k] \}$$
- Note: This measures token identity overlap; actual KV cache residency and cache hit rates must be measured by the serving engine during replay.

---

## 4. Projected Coverage Across Context Limits

Across the 20 tasks, 622 agent actions were examined. The table below shows request retention when enforcing context length limits (with a 32-token output reservation):

| Context Limit | Retained Requests | Retention Rate | Coding | Math (GSM8K) | SQL | QA (DROP) | Summary (CNN) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **8,192** | 124 / 622 | 20.0% | 28 | 26 | 30 | 26 | 14 |
| **16,384** | 202 / 622 | 32.5% | 56 | 45 | 40 | 44 | 17 |
| **32,768** | 402 / 622 | 64.6% | 111 | 89 | 90 | 84 | 28 |

---

## 5. Execution & Reproduction Commands

### Step 1: Preflight Verification
Verify that trace files exist, commit hashes match, and LoRA Safetensors weight headers are structurally sound:
```bash
python scripts/workplace/verify_workplace_setup.py \
    --manifest reports/real-workloads/theagentcompany_20tasks.json \
    --source-repo data/theagentcompany-experiments \
    --adapters-dir models/adapters
```
Output:
- `reports/real-workloads/theagentcompany_20tasks_verified.json`
- `reports/real-workloads/adapter_manifest_20tasks.json`

### Step 2: Build Multi-Turn Replay Datasets
Construct tokenized replay JSONL files:
```bash
python scripts/workplace/build_workplace_replay.py \
    --manifest reports/real-workloads/theagentcompany_20tasks_verified.json \
    --source-repo data/theagentcompany-experiments \
    --tokenizer models/base \
    --limits 8192 16384 32768 \
    --reserve-output 32 \
    --output-dir data/agent_replay_20tasks
```
Output:
- `data/agent_replay_20tasks/replay_8192.jsonl`
- `data/agent_replay_20tasks/replay_16384.jsonl`
- `data/agent_replay_20tasks/replay_32768.jsonl`
- `data/agent_replay_20tasks/coverage.json`

---

## 6. Next Steps & Active Milestones

- [x] Task selection, adapter mapping, and structural verification.
- [x] Normalization and multi-turn prompt replay generation tools.
- [ ] Implement multi-adapter vLLM replay driver for multi-turn sequences.
- [ ] Measure cache evictions, GPU memory fragmentation, and end-to-end replay TTFT.

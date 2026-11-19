# Real-World Workplace Workload Processing (TheAgentCompany)

This directory contains tools to prepare, verify, and reconstruct realistic agent workloads from **TheAgentCompany** benchmark trajectories across five LoRA roles.

## Scripts

### 1. `verify_workplace_setup.py`
Preflight verification script that inspects 20 source trajectory files in `data/theagentcompany-experiments/`, verifies the Git commit revision, and inspects the Safetensors weight headers, tensor offsets, and parameter sizes for the 5 real LoRA adapters in `models/adapters/`.

```bash
python scripts/workplace/verify_workplace_setup.py \
    --manifest reports/real-workloads/theagentcompany_20tasks.json \
    --source-repo data/theagentcompany-experiments \
    --adapters-dir models/adapters
```

### 2. `build_workplace_replay.py`
Multi-turn event history reconstruction builder. Iterates through trajectory events (messages, actions, observations), strips image blobs, formats prompts with workplace instructions, calculates deterministic shared-prefix lengths against prior turns for the same task/adapter, and exports replay files:
- `data/agent_replay_20tasks/replay_8192.jsonl`
- `data/agent_replay_20tasks/replay_16384.jsonl`
- `data/agent_replay_20tasks/replay_32768.jsonl`
- `data/agent_replay_20tasks/coverage.json`

```bash
python scripts/workplace/build_workplace_replay.py \
    --manifest reports/real-workloads/theagentcompany_20tasks_verified.json \
    --source-repo data/theagentcompany-experiments \
    --tokenizer models/base \
    --output-dir data/agent_replay_20tasks
```

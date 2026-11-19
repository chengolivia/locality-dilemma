#!/usr/bin/env python3
"""Build text-only event-history replays, not original OpenHands model inputs.

Run from locality-dilemma on the machine containing the downloaded traces.
No recorded commands are executed. Source logs and earlier replay files stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gzip
import json
from pathlib import Path
import re

INSTRUCTION = (
    "Continue this workplace task using the recorded history.\n"
    "This is a text-only reconstruction of agent events.\n\n"
)
OMIT_ARGS = {
    "image_urls", "images", "hidden", "confirmation_state", "blocking", "is_input"
}


def text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"data:image/[^ \t\r\n\"'<>]+", "[IMAGE DATA OMITTED]", value)


def normalize(event):
    """Same observation-preserving normalization used in the v2 pilot."""
    if event.get("source") == "agent" and event.get("action"):
        return {
            "type": "assistant_action",
            "action": event["action"],
            "args": {
                k: text(v) for k, v in (event.get("args") or {}).items()
                if k not in OMIT_ARGS
            },
        }
    if event.get("observation"):
        return {
            "type": "tool_observation",
            "tool": event["observation"],
            "content": text(event.get("content") or event.get("message")),
        }
    args = event.get("args") or {}
    return {
        "type": event.get("source", "unknown"),
        "content": text(args.get("content") or event.get("content") or event.get("message")),
    }


def requests(task, events, tokenizer):
    history = []
    step = 0
    for event in events:
        action = event.get("source") == "agent" and bool(event.get("action"))
        if action and history:
            # Snapshot BEFORE appending the response: no future action/observation leakage.
            prompt = INSTRUCTION + "\n".join(history) + "\nNext action:\n"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            yield {
                "task_id": task["task_id"], "step": step,
                "event_id": event.get("id"), "adapter_id": task["adapter_id"],
                "role": task.get("role", "unspecified"),
                "prompt": prompt, "prompt_token_ids": ids, "input_tokens": len(ids),
                "original_model_input": False,
                "construction": "event reconstruction v2; images omitted",
            }
            step += 1
        history.append(json.dumps(normalize(event), ensure_ascii=False))


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def write_replay(path, rows, context_limit, reserve):
    groups = defaultdict(deque)
    for row in rows:
        if row["input_tokens"] + reserve <= context_limit:
            groups[row["task_id"]].append(row)
    previous = {}
    adapter_counts = Counter()
    role_counts = Counter()
    tasks = set()
    seen = set()
    sequence = 0
    with path.open("w", encoding="utf-8") as f:
        while any(groups.values()):
            for task_id in sorted(groups):
                if not groups[task_id]:
                    continue
                row = dict(groups[task_id].popleft())
                key = (task_id, row["adapter_id"])
                ids = row["prompt_token_ids"]
                shared = common_prefix(previous.get(key, []), ids)
                row.update({
                    "request_id": f"{task_id}:{row['step']}",
                    "sequence_index": sequence,
                    "schedule": "constructed round-robin by task",
                    "context_limit": context_limit, "reserved_output_tokens": reserve,
                    "shared_prefix_tokens": shared,
                    "new_input_tokens": len(ids) - shared,
                    "reuse_semantics": "token overlap with previous retained same-task/same-adapter request; not verified KV hits",
                })
                if row["request_id"] in seen:
                    raise ValueError("Duplicate request: " + row["request_id"])
                seen.add(row["request_id"])
                assert len(ids) == row["input_tokens"]
                assert row["input_tokens"] + reserve <= context_limit
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                previous[key] = ids
                adapter_counts[row["adapter_id"]] += 1
                role_counts[row.get("role", "unspecified")] += 1
                tasks.add(task_id)
                sequence += 1
    return {"requests": sequence, "tasks": len(tasks), "requests_per_adapter": dict(adapter_counts), "requests_per_role": dict(role_counts)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_manifest = "reports/real-workloads/theagentcompany_20tasks_verified.json"
    if not Path(default_manifest).exists() and Path("reports/theagentcompany_20tasks_verified.json").exists():
        default_manifest = "reports/theagentcompany_20tasks_verified.json"
    p.add_argument("--manifest", default=default_manifest)
    p.add_argument("--source-repo", default="data/theagentcompany-experiments")
    p.add_argument("--tokenizer", default="models/base")
    p.add_argument("--limits", nargs="+", type=int, default=[8192, 16384, 32768])
    p.add_argument("--reserve-output", type=int, default=32)
    p.add_argument("--output-dir", default="data/agent_replay_20tasks")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.reserve_output < 1 or any(n <= args.reserve_output for n in args.limits):
        p.error("Each context limit must exceed the positive output reservation")
    limits = sorted(set(args.limits))
    out = Path(args.output_dir)
    outputs = [out / f"replay_{n}.jsonl" for n in limits] + [out / "coverage.json"]
    if not args.overwrite and any(path.exists() for path in outputs):
        p.error("Output exists; choose a new --output-dir or explicitly use --overwrite")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists() and (Path("reports/real-workloads") / manifest_path.name).exists():
        manifest_path = Path("reports/real-workloads") / manifest_path.name
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        p.error("Manifest has duplicate task IDs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    task_stats = []
    for task in tasks:
        if not task.get("adapter_id"):
            p.error("Missing adapter assignment: " + task["task_id"])
        path = Path(args.source_repo) / task["trajectory_path_in_source_repo"]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            events = json.load(f)
        task_rows = list(requests(task, events, tokenizer))
        row_stats = {
            "task_id": task["task_id"], "role": task.get("role"),
            "adapter_id": task["adapter_id"], "examined": len(task_rows),
            "retained_by_limit": {
                str(n): sum(r["input_tokens"] + args.reserve_output <= n for r in task_rows)
                for n in limits
            },
        }
        task_stats.append(row_stats)
        # Oversized requests are summarized but not held in memory for export.
        rows.extend(r for r in task_rows if r["input_tokens"] + args.reserve_output <= max(limits))
        print(task["task_id"], row_stats["retained_by_limit"], flush=True)
        del task_rows

    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_commit": manifest.get("source_commit"),
        "source_run": manifest.get("source_run"),
        "manifest": str(Path(args.manifest)), "tokenizer": args.tokenizer,
        "reserved_output_tokens": args.reserve_output,
        "construction": "v2 event normalization with workplace-neutral instruction; not original model inputs",
        "observations": "Preserved without length caps; image fields/data omitted as in v2",
        "scheduling": "Constructed round-robin; no original arrival timing inferred",
        "examined_requests": sum(t["examined"] for t in task_stats),
        "tasks": task_stats, "exports": {},
    }
    for n in limits:
        result = write_replay(out / f"replay_{n}.jsonl", rows, n, args.reserve_output)
        summary["exports"][str(n)] = result
        print(f"{n}: {result['requests']}/{summary['examined_requests']} requests; {result['requests_per_role']}")
    (out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved outputs to", out.resolve())


if __name__ == "__main__":
    main()

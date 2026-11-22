#!/usr/bin/env python3
"""Execute real-workload agent trace replays against a multi-adapter vLLM server.

Bridges the reconstructed TheAgentCompany multi-turn replays (e.g. replay_8192.jsonl)
to the telemetry measurement infrastructure (scripts/real/measurement.py).

Constructs and evaluates two contrasting replay schedules:
1. `adapter_interleaved`: Interleaves requests across different task roles/adapters
   round-robin to stress adapter switching and evaluate cache eviction under
   multi-tenant concurrency.
2. `adapter_grouped`: Sequences requests grouped by adapter to maximize adapter
   warmth and measure prefix cache reuse within same-adapter batches.

Telemetry outputs are saved sequentially in JSONL format to:
    reports/real-workloads/runs/{strategy}_{timestamp}.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from scripts.real.measurement import (
        append_measurement_record,
        execute_and_measure_request,
        http_get,
    )
except ImportError:
    from measurement import (  # type: ignore
        append_measurement_record,
        execute_and_measure_request,
        http_get,
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load declarative configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not HAS_YAML:
        raise ImportError("PyYAML is required to parse configs. Please install pyyaml.")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_replay_workload(replay_path: Path) -> List[Dict[str, Any]]:
    """Load reconstructed multi-turn requests from a replay JSONL file."""
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")

    records: List[Dict[str, Any]] = []
    with replay_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                records.append(row)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupt JSON at {replay_path}:{line_num}: {e}")

    if not records:
        raise ValueError(f"Replay file is empty: {replay_path}")
    return records


def resolve_served_model_name(adapter_id: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Map a manifest adapter ID (e.g. 'predibase/magicoder') to the served vLLM model name ('magicoder')."""
    if config:
        configured_adapters = config.get("adapters", {})
        if adapter_id in configured_adapters:
            return adapter_id
        for name, meta in configured_adapters.items():
            if meta.get("hf_repo") == adapter_id:
                return name

    # Standard fallback: strip organization prefix (e.g. 'predibase/magicoder' -> 'magicoder')
    if "/" in adapter_id:
        return adapter_id.split("/")[-1]
    return adapter_id


def build_adapter_interleaved_schedule(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construct an interleaved schedule that switches adapters round-robin.

    Maintains strictly sequential step ordering within each task while alternating
    active adapters across consecutive requests to maximize adapter switching.
    """
    # Group requests by task while preserving their step order
    task_queues: Dict[str, deque] = defaultdict(deque)
    adapter_of_task: Dict[str, str] = {}
    for r in records:
        t_id = r["task_id"]
        task_queues[t_id].append(r)
        adapter_of_task[t_id] = r.get("adapter_id", "default")

    # Map each adapter to its list of tasks
    adapter_tasks: Dict[str, List[str]] = defaultdict(list)
    for t_id in sorted(task_queues.keys()):
        adapter_tasks[adapter_of_task[t_id]].append(t_id)

    adapters = sorted(adapter_tasks.keys())
    adapter_ptrs = {a: 0 for a in adapters}

    interleaved: List[Dict[str, Any]] = []
    while any(task_queues[t] for t in task_queues):
        for adapter in adapters:
            tasks = adapter_tasks[adapter]
            # Find next task of this adapter with pending requests
            for _ in range(len(tasks)):
                ptr = adapter_ptrs[adapter]
                adapter_ptrs[adapter] = (ptr + 1) % len(tasks)
                t_id = tasks[ptr]
                if task_queues[t_id]:
                    item = dict(task_queues[t_id].popleft())
                    item["ordering_strategy"] = "adapter_interleaved"
                    interleaved.append(item)
                    break

    return interleaved


def build_adapter_grouped_schedule(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construct a grouped schedule that executes requests batched by adapter.

    Executes all requests for one adapter before switching to the next adapter.
    Within each adapter, tasks are executed sequentially step-by-step to maximize
    same-task prefix reuse.
    """
    grouped_by_adapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped_by_adapter[r.get("adapter_id", "default")].append(r)

    grouped: List[Dict[str, Any]] = []
    for adapter in sorted(grouped_by_adapter.keys()):
        # Within adapter, group by task so each task's steps run in sequence
        tasks_of_adapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in grouped_by_adapter[adapter]:
            tasks_of_adapter[r["task_id"]].append(r)

        for task_id in sorted(tasks_of_adapter.keys()):
            for r in tasks_of_adapter[task_id]:
                item = dict(r)
                item["ordering_strategy"] = "adapter_grouped"
                grouped.append(item)

    return grouped


def check_vllm_server(server_url: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """Check if the vLLM server is running and accessible."""
    try:
        status, body = http_get(f"{server_url}/v1/models", timeout=timeout)
        if status == 200:
            return True, body
        return False, f"Server returned HTTP {status}: {body}"
    except Exception as e:
        return False, str(e)


def run_strategy_replay(
    strategy: str,
    schedule: List[Dict[str, Any]],
    output_path: Path,
    server_url: str,
    config: Dict[str, Any],
    max_tokens: int = 128,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> int:
    """Execute a scheduled workload sequence against vLLM and record telemetry."""
    if limit is not None and limit > 0:
        schedule = schedule[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_requests = len(schedule)

    print(f"\n{'='*70}")
    print(f"STARTING REPLAY: Strategy = {strategy.upper()}")
    print(f"Target requests: {total_requests}")
    print(f"Output path:     {output_path}")
    print(f"vLLM server:     {server_url}")
    print(f"{'='*70}\n")

    executed_count = 0
    start_wall = time.time()

    for idx, req in enumerate(schedule, 1):
        raw_adapter_id = req.get("adapter_id", "")
        served_model = resolve_served_model_name(raw_adapter_id, config)
        task_id = req.get("task_id", "unknown_task")
        role = req.get("role", "unspecified")
        prompt = req.get("prompt", "")
        step = req.get("step", 0)

        step_info = f"[{idx}/{total_requests}] Strategy={strategy} Seq={idx} Adapter={served_model} Task={task_id} Step={step}"
        if verbose:
            print(f"\n--- {step_info} ---", flush=True)

        record = execute_and_measure_request(
            server_url=server_url,
            adapter_id=served_model,
            prompt=prompt,
            seq_num=idx,
            task_id=task_id,
            ordering_strategy=strategy,
            max_tokens=max_tokens,
            temperature=temperature,
            role=role,
        )

        # Preserve original replay metadata if available
        record["replay_metadata"] = {
            "original_adapter_id": raw_adapter_id,
            "step": step,
            "shared_prefix_tokens": req.get("shared_prefix_tokens"),
            "new_input_tokens": req.get("new_input_tokens"),
            "input_tokens": req.get("input_tokens"),
            "request_id_in_replay": req.get("request_id"),
        }

        append_measurement_record(record, output_path)
        executed_count += 1

        # Brief one-line progress reporting
        ttft = record.get("metrics", {}).get("time_to_first_token_ms")
        ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
        cache_hits = record.get("prometheus_deltas", {}).get("prefix_cache_hits_delta")
        cache_hits_str = f"+{cache_hits:.0f}" if cache_hits is not None else "N/A"
        tps = record.get("metrics", {}).get("tokens_per_second")
        tps_str = f"{tps:.1f}tok/s" if tps is not None else "N/A"

        print(
            f"{step_info} | TTFT: {ttft_str} | Hits: {cache_hits_str} | TPS: {tps_str}",
            flush=True,
        )

    elapsed_sec = time.time() - start_wall
    print(f"\nCompleted {executed_count}/{total_requests} requests for {strategy} in {elapsed_sec:.2f}s.")
    print(f"Results saved to: {output_path}")
    return executed_count


def main():
    parser = argparse.ArgumentParser(description="Execute real-world agent replay workloads against vLLM")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "real_workload.yaml",
        help="Path to centralized real_workload.yaml configuration",
    )
    parser.add_argument(
        "--replay-file",
        type=Path,
        default=_REPO_ROOT / "data" / "agent_replay_20tasks" / "replay_8192.jsonl",
        help="Path to reconstructed replay JSONL file (e.g. replay_8192.jsonl)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to prepend to output filename (e.g. 'forced_')",
    )
    parser.add_argument(
        "--strategy",
        choices=["adapter_interleaved", "adapter_grouped", "both"],
        default="both",
        help="Replay scheduling strategy to execute",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="vLLM server URL (defaults to server_url defined in config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save telemetry JSONL runs (defaults to config paths.output_dir)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generation tokens per request (defaults to config generation.max_tokens)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (defaults to config generation.temperature)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of requests per strategy (useful for smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and inspect schedules without sending HTTP requests to vLLM",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-request execution logs",
    )
    args = parser.parse_args()

    # 1. Load configuration and workload
    config = load_config(args.config)
    records = load_replay_workload(args.replay_file)
    print(f"Loaded {len(records)} requests from {args.replay_file}")

    # Resolve parameters
    server_url = args.server_url or config.get("vllm_server", {}).get("server_url", "http://localhost:8000")
    output_dir = args.output_dir or (_REPO_ROOT / config.get("paths", {}).get("output_dir", "reports/real-workloads/runs"))
    max_tokens = args.max_tokens or config.get("generation", {}).get("max_tokens", 128)
    temperature = args.temperature if args.temperature is not None else config.get("generation", {}).get("temperature", 0.0)

    # 2. Build schedules
    schedules: Dict[str, List[Dict[str, Any]]] = {}
    if args.strategy in ("adapter_interleaved", "both"):
        schedules["adapter_interleaved"] = build_adapter_interleaved_schedule(records)
    if args.strategy in ("adapter_grouped", "both"):
        schedules["adapter_grouped"] = build_adapter_grouped_schedule(records)

    # Print schedule summary
    print("\nSchedule Construction Summary:")
    for strat, sched in schedules.items():
        switches = sum(1 for i in range(1, len(sched)) if sched[i].get("adapter_id") != sched[i-1].get("adapter_id"))
        req_count = len(sched) if args.limit is None else min(len(sched), args.limit)
        print(f"  - {strat}: {req_count} requests, {switches} adapter transitions")

    if args.dry_run:
        print("\n[DRY-RUN] Schedule construction verified successfully. Exiting without issuing requests.")
        sys.exit(0)

    # 3. Verify server availability
    ready, msg = check_vllm_server(server_url)
    if not ready:
        print(f"\n[ERROR] vLLM server is not reachable at {server_url}.", file=sys.stderr)
        print(f"Diagnostic error: {msg}", file=sys.stderr)
        print("\nPlease start the vLLM server with all 5 LoRA adapters:", file=sys.stderr)
        print(
            "  ~/venvs/vllm/bin/vllm serve ./models/base \\\n"
            "      --served-model-name mistral-base \\\n"
            "      --dtype half \\\n"
            "      --max-model-len 4096 \\\n"
            "      --enable-lora \\\n"
            "      --lora-modules \\\n"
            "          magicoder=./models/adapters/predibase__magicoder \\\n"
            "          wikisql=./models/adapters/predibase__wikisql \\\n"
            "          gsm8k=./models/adapters/predibase__gsm8k \\\n"
            "          drop_explained=./models/adapters/predibase__drop_explained \\\n"
            "          cnn=./models/adapters/predibase__cnn \\\n"
            "      --max-loras 5 \\\n"
            "      --max-lora-rank 8 \\\n"
            "      --enable-prefix-caching \\\n"
            "      --enable-per-request-metrics \\\n"
            "      --gpu-memory-utilization 0.85\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. Execute runs
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_paths: List[Path] = []

    for strat, sched in schedules.items():
        out_file = output_dir / f"{args.prefix}{strat}_{timestamp}.jsonl"
        run_strategy_replay(
            strategy=strat,
            schedule=sched,
            output_path=out_file,
            server_url=server_url,
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            limit=args.limit,
            verbose=args.verbose,
        )
        run_paths.append(out_file)

    # 5. Suggested analysis command
    print(f"\n{'='*70}")
    print("ALL SCHEDULES EXECUTED SUCCESSFULLY")
    print(f"{'='*70}")
    print("\nTo analyze results and generate the Markdown comparison report:")
    inputs_arg = " ".join(str(p) for p in run_paths)
    report_md = output_dir / f"replay_comparison_{timestamp}.md"
    summary_json = output_dir / f"replay_summary_{timestamp}.json"
    print(
        f"python scripts/analysis/analyze_real_workload.py \\\n"
        f"    --inputs {inputs_arg} \\\n"
        f"    --output-report {report_md} \\\n"
        f"    --output-json {summary_json}\n"
    )


if __name__ == "__main__":
    main()

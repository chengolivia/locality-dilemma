#!/usr/bin/env python3
"""Run reproducible real-model LoRA KV-cache sharing experiment on vLLM.

Sequentially sends a prompt to two LoRA adapters:
1. magicoder
2. customer_support
3. magicoder

Measures per-request TTFT, ITL, generation throughput, and KV-cache hit metrics
from Prometheus counters before and after each request.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from patch_kv_cache import DEFAULT_FILE, get_status

DEFAULT_PROMPT = (
    "Experiment 20261114-A7F3. Shared instructions for two agents: "
    "You are an assistant operating within a customer support workflow. "
    "Follow the system instructions carefully. Use the provided context to "
    "answer the request accurately. Do not invent facts. If information is "
    "missing, state that clearly. Preserve relevant details from the "
    "conversation. Apply the same instructions consistently across all "
    "requests. Shared context: The customer has an account with an online "
    "retailer. The customer may need help managing account settings, billing "
    "information, orders, and password security. The support agent should "
    "provide clear, concise, step-by-step instructions. "
    "Request: A customer asks how to update their billing address."
)

DEFAULT_SERVER_CMD = [
    str(Path.home() / "venvs/vllm/bin/vllm"),
    "serve",
    "./models/base",
    "--served-model-name", "mistral-base",
    "--dtype", "half",
    "--max-model-len", "4096",
    "--enable-lora",
    "--lora-modules",
    "customer_support=./models/adapters/predibase__customer_support",
    "magicoder=./models/adapters/predibase__magicoder",
    "--max-loras", "2",
    "--max-lora-rank", "8",
    "--enable-prefix-caching",
    "--enable-per-request-metrics",
    "--gpu-memory-utilization", "0.85",
]


def http_get(url: str, timeout: float = 10.0) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-experiment"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def http_post_json(url: str, payload: dict, timeout: float = 120.0) -> Tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "vllm-experiment"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def parse_prometheus_metrics(raw_text: str) -> Dict[str, float]:
    """Parse relevant vLLM prefix cache and KV cache Prometheus metrics."""
    metrics: Dict[str, float] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            metric_key = parts[0]
            if (
                metric_key.startswith("vllm:prefix_cache_")
                or metric_key.startswith("vllm:prompt_tokens_cached")
                or metric_key.startswith("vllm:kv_cache_usage_perc")
            ):
                try:
                    metrics[metric_key] = float(parts[1])
                except ValueError:
                    pass
    return metrics


def extract_metric_val(metrics: Dict[str, float], prefix: str, default: float = 0.0) -> float:
    for k, v in metrics.items():
        if k.startswith(prefix):
            return v
    return default


def wait_for_server(base_url: str, proc: Optional[subprocess.Popen] = None, timeout_sec: int = 120) -> None:
    print(f"Waiting for vLLM server at {base_url} ...", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if proc is not None and proc.poll() is not None:
            log_tail = ""
            log_path = Path("/tmp/vllm_server.log")
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                log_tail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"vLLM server process exited prematurely with code {proc.returncode}.\n"
                f"Server log tail:\n{log_tail}"
            )
        try:
            status, body = http_get(f"{base_url}/v1/models", timeout=2.0)
            if status == 200:
                data = json.loads(body)
                model_ids = {m.get("id") for m in data.get("data", [])}
                if "magicoder" in model_ids and "customer_support" in model_ids:
                    print(f"vLLM server ready with models: {sorted(list(model_ids))}", flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout_sec} seconds.")


def start_vllm_server(cwd: Path) -> subprocess.Popen:
    venv_bin = str(Path.home() / "venvs/vllm/bin")
    cuda_home = str(Path.home() / "venvs/vllm/lib/python3.12/site-packages/nvidia/cu13")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(Path.home() / "venvs/vllm")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{venv_bin}:{cuda_home}/bin:" + env.get("PATH", "")
    print("Launching vLLM server subprocess:")
    print(" ".join(DEFAULT_SERVER_CMD))
    log_file = Path("/tmp/vllm_server.log")
    log_fp = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        DEFAULT_SERVER_CMD,
        cwd=str(cwd),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._log_fp = log_fp  # keep reference so it is not garbage collected
    return proc


def stop_vllm_server(proc: subprocess.Popen, timeout: int = 20) -> None:
    print(f"Stopping vLLM server process (PID {proc.pid}) ...", flush=True)
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Server did not stop on SIGINT, sending SIGKILL ...", flush=True)
        proc.kill()
        proc.wait(timeout=5)
    if hasattr(proc, "_log_fp"):
        try:
            proc._log_fp.close()
        except Exception:
            pass
    print("vLLM server stopped.", flush=True)


def run_experiment(
    condition: str,
    output_path: Path,
    server_url: str = "http://localhost:8000",
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> None:
    request_sequence = [
        (1, "magicoder"),
        (2, "customer_support"),
        (3, "magicoder"),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print(f"STARTING EXPERIMENT: Condition = {condition.upper()}")
    print(f"Prompt character count: {len(prompt)}")
    print(f"Request sequence: {[m for _, m in request_sequence]}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print("=" * 70 + "\n")

    for seq_num, adapter_id in request_sequence:
        print(f"--- Request {seq_num}/3: Adapter '{adapter_id}' ---", flush=True)

        # 1. Scrape metrics before
        _, raw_metrics_before = http_get(f"{server_url}/metrics")
        metrics_before = parse_prometheus_metrics(raw_metrics_before)
        hits_before = extract_metric_val(metrics_before, "vllm:prefix_cache_hits_total")

        # 2. Issue completion request
        payload = {
            "model": adapter_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req_start_wall = time.perf_counter()
        status, response = http_post_json(f"{server_url}/v1/completions", payload)
        req_end_wall = time.perf_counter()
        client_elapsed_ms = (req_end_wall - req_start_wall) * 1000.0

        if status != 200:
            raise RuntimeError(f"Request {seq_num} failed with status {status}: {response}")

        # 3. Scrape metrics after
        _, raw_metrics_after = http_get(f"{server_url}/metrics")
        metrics_after = parse_prometheus_metrics(raw_metrics_after)
        hits_after = extract_metric_val(metrics_after, "vllm:prefix_cache_hits_total")

        cache_hit_delta = hits_after - hits_before

        # Validate per-request metrics
        req_metrics = response.get("metrics")
        if not req_metrics:
            raise RuntimeError(
                f"Request {seq_num} returned no 'metrics' object in response. "
                "Ensure vLLM was started with --enable-per-request-metrics."
            )

        core_metric_keys = [
            "time_to_first_token_ms",
            "generation_time_ms",
            "queue_time_ms",
            "tokens_per_second",
        ]
        for k in core_metric_keys:
            if req_metrics.get(k) is None:
                raise RuntimeError(
                    f"Request {seq_num} metric '{k}' is missing or null: {req_metrics}"
                )

        choice = response["choices"][0]
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        # mean_itl_ms is only defined when there are multiple generated tokens (>1)
        if completion_tokens > 1 and req_metrics.get("mean_itl_ms") is None:
            raise RuntimeError(
                f"Request {seq_num} generated {completion_tokens} tokens but mean_itl_ms is null."
            )

        record = {
            "condition": condition,
            "sequence_number": seq_num,
            "adapter_id": adapter_id,
            "request_id": response.get("id"),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "finish_reason": choice.get("finish_reason"),
            "generated_text": choice.get("text"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "metrics": {
                "time_to_first_token_ms": req_metrics.get("time_to_first_token_ms"),
                "generation_time_ms": req_metrics.get("generation_time_ms"),
                "queue_time_ms": req_metrics.get("queue_time_ms"),
                "mean_itl_ms": req_metrics.get("mean_itl_ms"),
                "tokens_per_second": req_metrics.get("tokens_per_second"),
            },
            "client_elapsed_ms": client_elapsed_ms,
            "kv_cache_before": metrics_before,
            "kv_cache_after": metrics_after,
            "cache_hit_tokens_delta": cache_hit_delta,
            "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        results.append(record)

        print(f"  Request ID:        {record['request_id']}")
        print(f"  Prompt tokens:     {usage.get('prompt_tokens')}")
        print(f"  Completion tokens: {usage.get('completion_tokens')}")
        print(f"  Cache hit delta:   {cache_hit_delta} tokens")
        print(f"  TTFT:              {req_metrics.get('time_to_first_token_ms'):.2f} ms")
        print(f"  Generation time:   {req_metrics.get('generation_time_ms'):.2f} ms")
        itl_val = req_metrics.get('mean_itl_ms')
        itl_str = f"{itl_val:.2f} ms" if itl_val is not None else "null (single-token completion)"
        print(f"  Mean ITL:          {itl_str}")
        print(f"  Throughput:        {req_metrics.get('tokens_per_second'):.2f} tok/s")
        print(f"  Client wall clock: {client_elapsed_ms:.2f} ms\n", flush=True)

    # Write out JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(results)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run KV-cache sharing experiment")
    parser.add_argument(
        "--condition",
        choices=["native", "forced"],
        required=True,
        help="Experiment condition: 'native' or 'forced' (cross-adapter sharing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSONL file (e.g. reports/kv_sharing_native.jsonl)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Base URL of running vLLM server",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Automatically launch vLLM server, run experiment, and shut down server",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to use for all requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens for completion (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent

    # Verify kv_cache_utils.py patch status matches requested condition
    cache_utils_file = DEFAULT_FILE
    if cache_utils_file.is_file():
        current_status = get_status(cache_utils_file.read_text())
        if current_status != args.condition:
            print(
                f"ERROR: Condition is '{args.condition}', but {cache_utils_file} "
                f"status is '{current_status}'.\n"
                f"Run `python3 scripts/patch_kv_cache.py --{args.condition}` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Verified {cache_utils_file} status matches condition: '{current_status}'")

    server_proc: Optional[subprocess.Popen] = None
    if args.start_server:
        server_proc = start_vllm_server(repo_dir)

    try:
        wait_for_server(args.server_url, proc=server_proc)
        run_experiment(
            condition=args.condition,
            output_path=args.output,
            server_url=args.server_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    finally:
        if server_proc is not None:
            stop_vllm_server(server_proc)


if __name__ == "__main__":
    main()

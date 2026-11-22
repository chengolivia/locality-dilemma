#!/usr/bin/env python3
"""Validation experiment for customer_support adapter under native vLLM prefix caching.

Tests 5 sequential requests with the exact original prompt and sampling parameters
(temperature=0.0, max_tokens=100) under the native (unpatched) configuration.
Records decoded outputs, finish reasons, completion token IDs, and token-level logprobs.
"""

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
from typing import Any, Dict, List, Optional, Tuple

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


def wait_for_server(base_url: str, proc: Optional[subprocess.Popen] = None, timeout_sec: int = 300) -> None:
    print(f"Waiting for vLLM server at {base_url} ...", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if proc is not None and proc.poll() is not None:
            log_path = Path("/tmp/vllm_server.log")
            log_tail = ""
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                log_tail = "\n".join(lines[-25:])
            raise RuntimeError(
                f"vLLM server process exited prematurely with code {proc.returncode}.\n"
                f"Server log tail:\n{log_tail}"
            )
        try:
            status, body = http_get(f"{base_url}/v1/models", timeout=2.0)
            if status == 200:
                data = json.loads(body)
                model_ids = {m.get("id") for m in data.get("data", [])}
                if "customer_support" in model_ids:
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
    proc._log_fp = log_fp
    return proc


def stop_vllm_server(proc: subprocess.Popen, timeout: int = 25) -> None:
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


def main():
    workspace = Path("/home/ubuntu/locality-dilemma")
    reports_dir = workspace / "reports" / "adapter-kv-cache"
    artifacts_dir = Path("/home/ubuntu/.gemini/antigravity/brain/c4eae79b-b15b-46e4-920f-a2b48bee2213")

    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_reports_path = reports_dir / "customer_support_validation.jsonl"
    output_artifacts_path = artifacts_dir / "customer_support_validation.jsonl"

    # Step 1: Verify native configuration
    status = get_status(DEFAULT_FILE.read_text()).upper()
    print(f"Current vLLM KV-cache hash status: {status}")
    if status != "NATIVE":
        print(f"ERROR: Expected status NATIVE but got {status}. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Start fresh vLLM server to ensure clear prefix cache
    server_proc = start_vllm_server(workspace)
    try:
        wait_for_server("http://localhost:8000", server_proc)

        num_runs = 5
        results: List[Dict[str, Any]] = []

        print("\n" + "=" * 75)
        print(f"RUNNING CUSTOMER_SUPPORT VALIDATION: {num_runs} SEQUENTIAL REQUESTS")
        print(f"Adapter: customer_support | Prompt len: {len(DEFAULT_PROMPT)}")
        print("Parameters: max_tokens=100, temperature=0.0, logprobs=5, return_token_ids=True")
        print("=" * 75 + "\n")

        for i in range(1, num_runs + 1):
            print(f"--- Customer Support Request {i}/{num_runs} ---", flush=True)

            # Scrape metrics before
            _, raw_metrics_before = http_get("http://localhost:8000/metrics")
            metrics_before = parse_prometheus_metrics(raw_metrics_before)
            hits_before = extract_metric_val(metrics_before, "vllm:prefix_cache_hits_total")

            payload = {
                "model": "customer_support",
                "prompt": DEFAULT_PROMPT,
                "max_tokens": 100,
                "temperature": 0.0,
                "logprobs": 5,
                "return_token_ids": True,
            }

            req_start_wall = time.perf_counter()
            status_code, raw_response = http_post_json("http://localhost:8000/v1/completions", payload)
            req_end_wall = time.perf_counter()
            client_elapsed_ms = (req_end_wall - req_start_wall) * 1000.0

            if status_code != 200:
                raise RuntimeError(f"Request {i} failed: {status_code} {raw_response}")

            # Scrape metrics after
            _, raw_metrics_after = http_get("http://localhost:8000/metrics")
            metrics_after = parse_prometheus_metrics(raw_metrics_after)
            hits_after = extract_metric_val(metrics_after, "vllm:prefix_cache_hits_total")
            cache_hit_delta = hits_after - hits_before

            choice = raw_response["choices"][0]
            usage = raw_response.get("usage", {})
            req_metrics = raw_response.get("metrics", {})

            # Extract logprobs of the first generated token
            logprobs_obj = choice.get("logprobs")
            first_token_logprobs = None
            if logprobs_obj:
                tokens = logprobs_obj.get("tokens", [])
                token_lps = logprobs_obj.get("token_logprobs", [])
                top_lps = logprobs_obj.get("top_logprobs", [])
                if tokens:
                    first_token_logprobs = {
                        "token": tokens[0],
                        "logprob": token_lps[0] if token_lps else None,
                        "top_logprobs": top_lps[0] if top_lps else None,
                    }

            record = {
                "iteration": i,
                "adapter": "customer_support",
                "request_id": raw_response.get("id"),
                "prompt": DEFAULT_PROMPT,
                "finish_reason": choice.get("finish_reason"),
                "generated_text": choice.get("text"),
                "completion_token_ids": choice.get("token_ids"),
                "prompt_token_ids": choice.get("prompt_token_ids"),
                "first_token_logprobs": first_token_logprobs,
                "usage": usage,
                "metrics": req_metrics,
                "cache_hit_tokens_delta": cache_hit_delta,
                "client_elapsed_ms": client_elapsed_ms,
                "raw_response": raw_response,
                "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            results.append(record)

            print(f"  Request ID:           {record['request_id']}")
            print(f"  Prompt tokens:        {usage.get('prompt_tokens')}")
            print(f"  Completion tokens:    {usage.get('completion_tokens')}")
            print(f"  Finish reason:        {record['finish_reason']}")
            print(f"  Completion token IDs: {record['completion_token_ids']}")
            print(f"  Generated text:       repr: {repr(record['generated_text'])}")
            print(f"  Cache hit delta:      {cache_hit_delta} tokens")
            print(f"  TTFT:                 {req_metrics.get('time_to_first_token_ms', 0):.2f} ms")
            print(f"  Generation time:      {req_metrics.get('generation_time_ms', 0):.2f} ms")
            if first_token_logprobs:
                print(f"  First token:          {repr(first_token_logprobs['token'])} (logprob: {first_token_logprobs['logprob']})")
                print(f"  Top logprobs:         {first_token_logprobs['top_logprobs']}")
            print(f"  Client wall clock:    {client_elapsed_ms:.2f} ms\n", flush=True)

        # Write results to reports and artifacts
        for p in [output_reports_path, output_artifacts_path]:
            with p.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved {len(results)} records to {p}")

        # Summary verification
        all_eos = all(
            r["finish_reason"] == "stop" and r["usage"].get("completion_tokens") == 1 and r["generated_text"] == ""
            for r in results
        )
        print("\n" + "=" * 75)
        print("VERIFICATION SUMMARY")
        print(f"Total requests: {len(results)}")
        print(f"All requests emitted EOS: {all_eos}")
        for r in results:
            print(
                f"  Req {r['iteration']}: finish_reason={r['finish_reason']}, "
                f"completion_tokens={r['usage'].get('completion_tokens')}, "
                f"token_ids={r['completion_token_ids']}, "
                f"cache_hit_delta={r['cache_hit_tokens_delta']}, "
                f"TTFT={r['metrics'].get('time_to_first_token_ms'):.2f} ms"
            )
        print("=" * 75)

    finally:
        stop_vllm_server(server_proc)


if __name__ == "__main__":
    main()

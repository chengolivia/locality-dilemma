#!/usr/bin/env python3
"""Reusable measurement infrastructure for real-workload replay experiments.

Captures per-request server and client latency metrics, token counts, generated
text, and complete Prometheus prefix-cache counters before and after each request.

CRITICAL INVARIANT:
Missing metrics are recorded explicitly as None (null in JSON). Do NOT silently
substitute 0.0 for missing metrics (e.g. mean_itl_ms when completion_tokens <= 1,
or missing Prometheus counters).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request


def http_get(url: str, timeout: float = 10.0) -> Tuple[int, str]:
    """Perform an HTTP GET request and return (status_code, body_text)."""
    req = urllib.request.Request(url, headers={"User-Agent": "locality-dilemma-measurement"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def http_post_json(url: str, payload: dict, timeout: float = 120.0) -> Tuple[int, dict]:
    """Perform an HTTP POST request with JSON payload and return (status_code, json_dict)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "locality-dilemma-measurement"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def parse_prometheus_metrics(raw_text: str) -> Dict[str, float]:
    """Parse Prometheus exposition text format into a metric_name -> float value map.

    Handles:
    - Comment lines (# HELP, # TYPE) and empty lines
    - Metrics with labels (e.g. `vllm:prefix_cache_hits_total{model="magicoder"}`)
    - Base metric keys (stripped of labels) storing the latest or aggregate value
    - Scientific notation and special float values
    """
    metrics: Dict[str, float] = {}
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([+-]?(?:[0-9]*[.])?[0-9]+(?:[eE][+-]?[0-9]+)?|NaN|Inf|-Inf)")

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = pattern.match(line)
        if match:
            metric_name = match.group(1)
            labels = match.group(2)
            val_str = match.group(3)
            try:
                val = float(val_str)
            except ValueError:
                continue

            # Store the exact key with labels if present
            if labels:
                full_key = f"{metric_name}{{{labels}}}"
                metrics[full_key] = val
            # Also store or update the base metric name
            metrics[metric_name] = val
        else:
            # Fallback whitespace split
            parts = line.split()
            if len(parts) >= 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass

    return metrics


def extract_metric_val(
    metrics: Dict[str, float],
    prefix: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """Extract a metric value by prefix match.

    Returns the first matching metric value, or default (which defaults to None,
    preserving missing state rather than assuming 0.0).
    """
    for k, v in metrics.items():
        if k == prefix or k.startswith(f"{prefix}{{") or k.startswith(prefix):
            return v
    return default


def calculate_prometheus_deltas(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
) -> Dict[str, Optional[float]]:
    """Compute counter deltas and prefix cache hit rate between before and after snapshots.

    Missing counters in either snapshot evaluate to None (null), NOT 0.0.
    Cache hit rate evaluates to None if queries_delta is None or <= 0.
    """
    queries_before = extract_metric_val(metrics_before, "vllm:prefix_cache_queries_total")
    queries_after = extract_metric_val(metrics_after, "vllm:prefix_cache_queries_total")

    hits_before = extract_metric_val(metrics_before, "vllm:prefix_cache_hits_total")
    hits_after = extract_metric_val(metrics_after, "vllm:prefix_cache_hits_total")

    tokens_cached_before = extract_metric_val(metrics_before, "vllm:prompt_tokens_cached_total")
    tokens_cached_after = extract_metric_val(metrics_after, "vllm:prompt_tokens_cached_total")

    kv_usage_after = extract_metric_val(metrics_after, "vllm:kv_cache_usage_perc")

    queries_delta: Optional[float] = None
    if queries_after is not None and queries_before is not None:
        queries_delta = max(0.0, queries_after - queries_before)

    hits_delta: Optional[float] = None
    if hits_after is not None and hits_before is not None:
        hits_delta = max(0.0, hits_after - hits_before)

    tokens_cached_delta: Optional[float] = None
    if tokens_cached_after is not None and tokens_cached_before is not None:
        tokens_cached_delta = max(0.0, tokens_cached_after - tokens_cached_before)

    cache_hit_rate: Optional[float] = None
    if queries_delta is not None and queries_delta > 0 and hits_delta is not None:
        cache_hit_rate = min(1.0, hits_delta / queries_delta)

    return {
        "prefix_cache_queries_delta": queries_delta,
        "prefix_cache_hits_delta": hits_delta,
        "prompt_tokens_cached_delta": tokens_cached_delta,
        "cache_hit_rate": cache_hit_rate,
        "kv_cache_usage_perc": kv_usage_after,
    }


def build_measurement_record(
    seq_num: int,
    task_id: str,
    adapter_id: str,
    prompt: str,
    response_status: int,
    response_json: Dict[str, Any],
    client_elapsed_ms: float,
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    ordering_strategy: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a standardized measurement record dictionary for a single request.

    Explicitly preserves None for missing or undefined values.
    """
    if response_status != 200:
        raise RuntimeError(
            f"Request {seq_num} (task={task_id}, adapter={adapter_id}) "
            f"failed with HTTP {response_status}: {response_json}"
        )

    choices = response_json.get("choices")
    if not choices or not isinstance(choices, list):
        raise ValueError(f"Request {seq_num} returned no choices in response: {response_json}")

    choice = choices[0]
    usage = response_json.get("usage") or {}
    req_metrics = response_json.get("metrics") or {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # Extract per-request metrics, explicitly keeping None if absent
    ttft_ms = req_metrics.get("time_to_first_token_ms")
    gen_time_ms = req_metrics.get("generation_time_ms")
    queue_time_ms = req_metrics.get("queue_time_ms")
    tokens_per_sec = req_metrics.get("tokens_per_second")
    mean_itl_ms = req_metrics.get("mean_itl_ms")

    # Note: mean_itl_ms is undefined when completion_tokens <= 1.
    # We explicitly keep it as None without substituting 0.0.

    # Calculate Prometheus cache deltas
    deltas = calculate_prometheus_deltas(metrics_before, metrics_after)

    record: Dict[str, Any] = {
        "sequence_number": seq_num,
        "request_id": response_json.get("id"),
        "task_id": task_id,
        "adapter_id": adapter_id,
        "role": role,
        "ordering_strategy": ordering_strategy,
        "prompt": prompt,
        "generated_text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "metrics": {
            "time_to_first_token_ms": ttft_ms,
            "generation_time_ms": gen_time_ms,
            "queue_time_ms": queue_time_ms,
            "mean_itl_ms": mean_itl_ms,
            "tokens_per_second": tokens_per_sec,
        },
        "client_elapsed_ms": client_elapsed_ms,
        "prometheus_deltas": deltas,
        "prometheus_before": metrics_before,
        "prometheus_after": metrics_after,
        "raw_response_metadata": {
            "id": response_json.get("id"),
            "object": response_json.get("object"),
            "created": response_json.get("created"),
            "model": response_json.get("model"),
            "system_fingerprint": response_json.get("system_fingerprint"),
        },
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    return record


def execute_and_measure_request(
    server_url: str,
    adapter_id: str,
    prompt: str,
    seq_num: int,
    task_id: str,
    ordering_strategy: Optional[str] = None,
    max_tokens: int = 128,
    temperature: float = 0.0,
    role: Optional[str] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Execute a single request against a running vLLM server with full telemetry.

    1. Scrapes Prometheus metrics before request.
    2. Issues HTTP POST to /v1/completions and measures client wall-clock latency.
    3. Scrapes Prometheus metrics after request.
    4. Constructs and returns complete measurement record.
    """
    # 1. Scrape metrics before
    _, raw_metrics_before = http_get(f"{server_url}/metrics", timeout=10.0)
    metrics_before = parse_prometheus_metrics(raw_metrics_before)

    # 2. Issue completion request
    payload: Dict[str, Any] = {
        "model": adapter_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_payload:
        payload.update(extra_payload)

    req_start_wall = time.perf_counter()
    status, response_json = http_post_json(
        f"{server_url}/v1/completions",
        payload,
        timeout=timeout,
    )
    req_end_wall = time.perf_counter()
    client_elapsed_ms = (req_end_wall - req_start_wall) * 1000.0

    # 3. Scrape metrics after
    _, raw_metrics_after = http_get(f"{server_url}/metrics", timeout=10.0)
    metrics_after = parse_prometheus_metrics(raw_metrics_after)

    # 4. Build and return record
    return build_measurement_record(
        seq_num=seq_num,
        task_id=task_id,
        adapter_id=adapter_id,
        prompt=prompt,
        response_status=status,
        response_json=response_json,
        client_elapsed_ms=client_elapsed_ms,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        ordering_strategy=ordering_strategy,
        max_tokens=max_tokens,
        temperature=temperature,
        role=role,
    )


def append_measurement_record(record: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Append a single measurement record to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_measurement_records(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load all measurement records from a JSONL file."""
    records = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Measurements file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupt JSON at line {line_num} in {path}: {e}")

    return records

#!/usr/bin/env python3
"""Analysis scaffold for real-world workload replay experiments.

Processes JSONL measurement files from adapter-interleaved and adapter-grouped
replays, computing:
1. Overall latency and throughput summaries (TTFT, ITL, Gen Time, Tok/s).
2. Per-adapter latency, throughput, and cache metrics.
3. Cache-hit rates, query counts, and token savings.
4. Comparative analysis: interleaved vs. adapter-grouped orderings.
5. Formatted Markdown report and structured JSON summary.

CRITICAL INVARIANT:
Missing metrics (e.g. mean_itl_ms for single-token outputs, missing Prometheus counters)
are filtered out of statistical calculations and explicitly reported as N/A, NEVER
silently converted to 0.0.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


def compute_numeric_stats(values: List[Optional[Union[int, float]]]) -> Dict[str, Optional[float]]:
    """Compute count, mean, median, std, min, max, and p95 for a numeric sequence.

    Excludes None values. If no valid values exist, all statistics evaluate to None.
    """
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    valid.sort()
    n = len(valid)
    mean_val = sum(valid) / n

    if n > 1:
        var = sum((x - mean_val) ** 2 for x in valid) / (n - 1)
        std_val = var ** 0.5
    else:
        std_val = 0.0

    if n % 2 == 1:
        median_val = valid[n // 2]
    else:
        median_val = (valid[n // 2 - 1] + valid[n // 2]) / 2.0

    # Nearest rank / linear interpolation for p95
    k = 0.95 * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    d = k - f
    p95_val = valid[f] + (valid[c] - valid[f]) * d

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "min": round(valid[0], 4),
        "max": round(valid[-1], 4),
        "p95": round(p95_val, 4),
    }


def format_val(val: Optional[Union[int, float]], decimals: int = 2, suffix: str = "") -> str:
    """Format a numeric value safely; returns 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def load_records(paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """Load measurement records from one or more JSONL files or directories."""
    all_records: List[Dict[str, Any]] = []
    resolved_files: List[Path] = []

    for p in paths:
        path = Path(p)
        if path.is_dir():
            resolved_files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            resolved_files.append(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

    for file_path in resolved_files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                    record["_source_file"] = str(file_path)
                    all_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON in {file_path}:{line_idx}: {e}", file=sys.stderr)

    return all_records


def analyze_strategy_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate and per-adapter statistics for a list of records under one strategy."""
    num_requests = len(records)
    if num_requests == 0:
        return {}

    # Extract sequences for latency & throughput
    ttfts = [r.get("metrics", {}).get("time_to_first_token_ms") for r in records]
    gen_times = [r.get("metrics", {}).get("generation_time_ms") for r in records]
    queue_times = [r.get("metrics", {}).get("queue_time_ms") for r in records]
    itls = [r.get("metrics", {}).get("mean_itl_ms") for r in records]
    tpss = [r.get("metrics", {}).get("tokens_per_second") for r in records]
    client_elapsed = [r.get("client_elapsed_ms") for r in records]

    # Token counts
    prompt_tokens = [r.get("usage", {}).get("prompt_tokens") for r in records]
    completion_tokens = [r.get("usage", {}).get("completion_tokens") for r in records]
    total_tokens = [r.get("usage", {}).get("total_tokens") for r in records]

    # Prometheus cache metrics
    query_deltas = [r.get("prometheus_deltas", {}).get("prefix_cache_queries_delta") for r in records]
    hit_deltas = [r.get("prometheus_deltas", {}).get("prefix_cache_hits_delta") for r in records]
    cached_token_deltas = [r.get("prometheus_deltas", {}).get("prompt_tokens_cached_delta") for r in records]

    valid_query_deltas = [v for v in query_deltas if v is not None]
    valid_hit_deltas = [v for v in hit_deltas if v is not None]
    valid_cached_token_deltas = [v for v in cached_token_deltas if v is not None]

    total_queries = sum(valid_query_deltas) if valid_query_deltas else None
    total_hits = sum(valid_hit_deltas) if valid_hit_deltas else None
    total_cached_tokens = sum(valid_cached_token_deltas) if valid_cached_token_deltas else None

    aggregate_hit_rate: Optional[float] = None
    if total_queries is not None and total_queries > 0 and total_hits is not None:
        aggregate_hit_rate = total_hits / total_queries

    # Per-adapter breakdowns
    adapters = sorted(list({r.get("adapter_id", "unknown") for r in records}))
    per_adapter: Dict[str, Any] = {}

    for adapter_id in adapters:
        ad_records = [r for r in records if r.get("adapter_id") == adapter_id]
        ad_ttfts = [r.get("metrics", {}).get("time_to_first_token_ms") for r in ad_records]
        ad_gen_times = [r.get("metrics", {}).get("generation_time_ms") for r in ad_records]
        ad_itls = [r.get("metrics", {}).get("mean_itl_ms") for r in ad_records]
        ad_tpss = [r.get("metrics", {}).get("tokens_per_second") for r in ad_records]
        ad_prompt_toks = [r.get("usage", {}).get("prompt_tokens") for r in ad_records]
        ad_comp_toks = [r.get("usage", {}).get("completion_tokens") for r in ad_records]

        ad_q_deltas = [r.get("prometheus_deltas", {}).get("prefix_cache_queries_delta") for r in ad_records]
        ad_h_deltas = [r.get("prometheus_deltas", {}).get("prefix_cache_hits_delta") for r in ad_records]
        valid_ad_q = [v for v in ad_q_deltas if v is not None]
        valid_ad_h = [v for v in ad_h_deltas if v is not None]

        ad_tot_q = sum(valid_ad_q) if valid_ad_q else None
        ad_tot_h = sum(valid_ad_h) if valid_ad_h else None
        ad_hit_rate = (ad_tot_h / ad_tot_q) if (ad_tot_q is not None and ad_tot_q > 0 and ad_tot_h is not None) else None

        per_adapter[adapter_id] = {
            "num_requests": len(ad_records),
            "role": ad_records[0].get("role") if ad_records else None,
            "prompt_tokens": compute_numeric_stats(ad_prompt_toks),
            "completion_tokens": compute_numeric_stats(ad_comp_toks),
            "time_to_first_token_ms": compute_numeric_stats(ad_ttfts),
            "generation_time_ms": compute_numeric_stats(ad_gen_times),
            "mean_itl_ms": compute_numeric_stats(ad_itls),
            "tokens_per_second": compute_numeric_stats(ad_tpss),
            "cache": {
                "queries_delta_sum": ad_tot_q,
                "hits_delta_sum": ad_tot_h,
                "cache_hit_rate": ad_hit_rate,
            },
        }

    return {
        "num_requests": num_requests,
        "token_usage": {
            "prompt_tokens": compute_numeric_stats(prompt_tokens),
            "completion_tokens": compute_numeric_stats(completion_tokens),
            "total_tokens": compute_numeric_stats(total_tokens),
            "prompt_tokens_sum": sum(v for v in prompt_tokens if v is not None),
            "completion_tokens_sum": sum(v for v in completion_tokens if v is not None),
        },
        "time_to_first_token_ms": compute_numeric_stats(ttfts),
        "generation_time_ms": compute_numeric_stats(gen_times),
        "queue_time_ms": compute_numeric_stats(queue_times),
        "mean_itl_ms": compute_numeric_stats(itls),
        "tokens_per_second": compute_numeric_stats(tpss),
        "client_elapsed_ms": compute_numeric_stats(client_elapsed),
        "cache": {
            "queries_delta_sum": total_queries,
            "hits_delta_sum": total_hits,
            "cached_tokens_delta_sum": total_cached_tokens,
            "aggregate_cache_hit_rate": aggregate_hit_rate,
        },
        "per_adapter": per_adapter,
    }


def compare_strategies(
    strat_interleaved: Dict[str, Any],
    strat_grouped: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute comparison metrics between adapter_interleaved and adapter_grouped."""
    comp: Dict[str, Any] = {}

    # TTFT comparison
    ttft_int_mean = strat_interleaved.get("time_to_first_token_ms", {}).get("mean")
    ttft_grp_mean = strat_grouped.get("time_to_first_token_ms", {}).get("mean")
    ttft_int_med = strat_interleaved.get("time_to_first_token_ms", {}).get("median")
    ttft_grp_med = strat_grouped.get("time_to_first_token_ms", {}).get("median")

    ttft_mean_delta: Optional[float] = None
    ttft_mean_speedup: Optional[float] = None
    if ttft_int_mean is not None and ttft_grp_mean is not None:
        ttft_mean_delta = ttft_int_mean - ttft_grp_mean
        if ttft_grp_mean > 0:
            ttft_mean_speedup = ttft_int_mean / ttft_grp_mean

    ttft_med_delta: Optional[float] = None
    if ttft_int_med is not None and ttft_grp_med is not None:
        ttft_med_delta = ttft_int_med - ttft_grp_med

    # Cache hit rate comparison
    hit_rate_int = strat_interleaved.get("cache", {}).get("aggregate_cache_hit_rate")
    hit_rate_grp = strat_grouped.get("cache", {}).get("aggregate_cache_hit_rate")
    hit_rate_delta_pp: Optional[float] = None
    if hit_rate_grp is not None and hit_rate_int is not None:
        hit_rate_delta_pp = (hit_rate_grp - hit_rate_int) * 100.0

    # Throughput comparison
    tps_int_mean = strat_interleaved.get("tokens_per_second", {}).get("mean")
    tps_grp_mean = strat_grouped.get("tokens_per_second", {}).get("mean")
    tps_mean_delta: Optional[float] = None
    if tps_grp_mean is not None and tps_int_mean is not None:
        tps_mean_delta = tps_grp_mean - tps_int_mean

    # Per-adapter comparisons
    adapters_int = strat_interleaved.get("per_adapter", {})
    adapters_grp = strat_grouped.get("per_adapter", {})
    common_adapters = sorted(list(set(adapters_int.keys()).intersection(set(adapters_grp.keys()))))
    adapter_comparisons: Dict[str, Any] = {}

    for ad in common_adapters:
        ad_int_ttft = adapters_int[ad]["time_to_first_token_ms"].get("mean")
        ad_grp_ttft = adapters_grp[ad]["time_to_first_token_ms"].get("mean")
        ad_int_hit = adapters_int[ad]["cache"].get("cache_hit_rate")
        ad_grp_hit = adapters_grp[ad]["cache"].get("cache_hit_rate")

        ad_ttft_diff = (ad_int_ttft - ad_grp_ttft) if (ad_int_ttft is not None and ad_grp_ttft is not None) else None
        ad_hit_diff = ((ad_grp_hit - ad_int_hit) * 100.0) if (ad_grp_hit is not None and ad_int_hit is not None) else None

        adapter_comparisons[ad] = {
            "ttft_interleaved_mean": ad_int_ttft,
            "ttft_grouped_mean": ad_grp_ttft,
            "ttft_delta_ms": ad_ttft_diff,
            "cache_hit_rate_interleaved": ad_int_hit,
            "cache_hit_rate_grouped": ad_grp_hit,
            "cache_hit_rate_delta_pp": ad_hit_diff,
        }

    comp["ttft"] = {
        "interleaved_mean": ttft_int_mean,
        "grouped_mean": ttft_grp_mean,
        "delta_mean_ms": ttft_mean_delta,
        "speedup_ratio": ttft_mean_speedup,
        "interleaved_median": ttft_int_med,
        "grouped_median": ttft_grp_med,
        "delta_median_ms": ttft_med_delta,
    }
    comp["cache_hit_rate"] = {
        "interleaved_rate": hit_rate_int,
        "grouped_rate": hit_rate_grp,
        "delta_percentage_points": hit_rate_delta_pp,
    }
    comp["tokens_per_second"] = {
        "interleaved_mean": tps_int_mean,
        "grouped_mean": tps_grp_mean,
        "delta_mean": tps_mean_delta,
    }
    comp["per_adapter"] = adapter_comparisons

    return comp


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run complete analysis on a collection of measurement records."""
    if not records:
        return {"error": "No records to analyze"}

    # Group by ordering_strategy
    strat_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        strat = r.get("ordering_strategy") or r.get("condition") or "unspecified"
        strat_groups.setdefault(strat, []).append(r)

    strategies_summary: Dict[str, Any] = {}
    for strat_name, strat_records in strat_groups.items():
        strategies_summary[strat_name] = analyze_strategy_records(strat_records)

    comparison = {}
    if "adapter_interleaved" in strategies_summary and "adapter_grouped" in strategies_summary:
        comparison = compare_strategies(
            strategies_summary["adapter_interleaved"],
            strategies_summary["adapter_grouped"],
        )

    return {
        "analyzed_at_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_records": len(records),
        "strategies": strategies_summary,
        "comparison": comparison,
    }


def generate_markdown_report(analysis: Dict[str, Any]) -> str:
    """Generate a clean, reproducible Markdown report from analysis results."""
    lines: List[str] = []
    lines.append("# Real-World Workload Replay: Locality Dilemma Analysis")
    lines.append("")
    lines.append(f"**Analysis Timestamp:** {analysis.get('analyzed_at_iso')}")
    lines.append(f"**Total Replay Requests:** {analysis.get('total_records')}")
    lines.append("")

    strategies = analysis.get("strategies", {})
    comparison = analysis.get("comparison", {})

    # 1. Executive Summary Table
    if comparison:
        lines.append("## Executive Comparison: Interleaved vs. Grouped Orderings")
        lines.append("")
        lines.append(
            "Comparing **Adapter-Interleaved** (simulating multi-tenant concurrency with alternating adapters) "
            "against **Adapter-Grouped** (maximizing adapter warmth by batching same-adapter requests)."
        )
        lines.append("")
        lines.append("| Metric | Adapter-Interleaved | Adapter-Grouped | Difference (Grouped Benefit) |")
        lines.append("|---|:---:|:---:|:---:|")

        ttft_comp = comparison.get("ttft", {})
        int_ttft_mean = format_val(ttft_comp.get("interleaved_mean"), suffix=" ms")
        grp_ttft_mean = format_val(ttft_comp.get("grouped_mean"), suffix=" ms")
        diff_ttft = format_val(ttft_comp.get("delta_mean_ms"), suffix=" ms lower")
        speedup = format_val(ttft_comp.get("speedup_ratio"), decimals=2, suffix="x")
        lines.append(f"| **Mean TTFT** | {int_ttft_mean} | {grp_ttft_mean} | {diff_ttft} ({speedup}) |")

        int_ttft_med = format_val(ttft_comp.get("interleaved_median"), suffix=" ms")
        grp_ttft_med = format_val(ttft_comp.get("grouped_median"), suffix=" ms")
        diff_ttft_med = format_val(ttft_comp.get("delta_median_ms"), suffix=" ms lower")
        lines.append(f"| **Median TTFT** | {int_ttft_med} | {grp_ttft_med} | {diff_ttft_med} |")

        cache_comp = comparison.get("cache_hit_rate", {})
        int_hr = format_val(cache_comp.get("interleaved_rate", 0.0) * 100.0 if cache_comp.get("interleaved_rate") is not None else None, suffix="%")
        grp_hr = format_val(cache_comp.get("grouped_rate", 0.0) * 100.0 if cache_comp.get("grouped_rate") is not None else None, suffix="%")
        diff_hr = format_val(cache_comp.get("delta_percentage_points"), suffix=" pp")
        lines.append(f"| **Cache Hit Rate** | {int_hr} | {grp_hr} | +{diff_hr} |")

        tps_comp = comparison.get("tokens_per_second", {})
        int_tps = format_val(tps_comp.get("interleaved_mean"), suffix=" tok/s")
        grp_tps = format_val(tps_comp.get("grouped_mean"), suffix=" tok/s")
        diff_tps = format_val(tps_comp.get("delta_mean"), suffix=" tok/s")
        lines.append(f"| **Mean Throughput** | {int_tps} | {grp_tps} | +{diff_tps} |")
        lines.append("")

    # 2. Overall Strategy Summaries
    lines.append("## Overall Strategy Performance")
    lines.append("")
    lines.append("| Strategy | Requests | Prompt Toks (Sum) | Comp Toks (Sum) | Mean TTFT | P95 TTFT | Mean ITL | Mean Tok/s | Cache Hit Rate |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for strat_name, strat_data in strategies.items():
        reqs = strat_data.get("num_requests", 0)
        p_tok_sum = strat_data.get("token_usage", {}).get("prompt_tokens_sum", 0)
        c_tok_sum = strat_data.get("token_usage", {}).get("completion_tokens_sum", 0)
        mean_ttft = format_val(strat_data.get("time_to_first_token_ms", {}).get("mean"), suffix=" ms")
        p95_ttft = format_val(strat_data.get("time_to_first_token_ms", {}).get("p95"), suffix=" ms")
        mean_itl = format_val(strat_data.get("mean_itl_ms", {}).get("mean"), suffix=" ms")
        mean_tps = format_val(strat_data.get("tokens_per_second", {}).get("mean"), suffix=" tok/s")
        hr_val = strat_data.get("cache", {}).get("aggregate_cache_hit_rate")
        hr_str = format_val(hr_val * 100.0 if hr_val is not None else None, suffix="%")

        lines.append(
            f"| `{strat_name}` | {reqs} | {p_tok_sum:,} | {c_tok_sum:,} | {mean_ttft} | {p95_ttft} | {mean_itl} | {mean_tps} | {hr_str} |"
        )
    lines.append("")

    # 3. Cache Metrics Breakdown
    lines.append("## Prefix Cache Performance (Prometheus Telemetry)")
    lines.append("")
    lines.append("| Strategy | Queries Delta | Hits Delta | Hit Rate | Cached Tokens Delta |")
    lines.append("|---|:---:|:---:|:---:|:---:|")

    for strat_name, strat_data in strategies.items():
        cache = strat_data.get("cache", {})
        q_sum = format_val(cache.get("queries_delta_sum"), decimals=0)
        h_sum = format_val(cache.get("hits_delta_sum"), decimals=0)
        hr = format_val((cache.get("aggregate_cache_hit_rate") or 0.0) * 100.0 if cache.get("aggregate_cache_hit_rate") is not None else None, suffix="%")
        ct_sum = format_val(cache.get("cached_tokens_delta_sum"), decimals=0)
        lines.append(f"| `{strat_name}` | {q_sum} | {h_sum} | {hr} | {ct_sum} |")
    lines.append("")

    # 4. Per-Adapter Breakdown
    lines.append("## Per-Adapter Breakdown")
    lines.append("")

    for strat_name, strat_data in strategies.items():
        lines.append(f"### Strategy: `{strat_name}`")
        lines.append("")
        lines.append("| Adapter ID | Role | Requests | Mean TTFT | Median TTFT | Mean ITL | Mean Tok/s | Cache Hit Rate |")
        lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")

        per_adapter = strat_data.get("per_adapter", {})
        for adapter_id, ad_data in per_adapter.items():
            role = ad_data.get("role") or "unspecified"
            reqs = ad_data.get("num_requests", 0)
            m_ttft = format_val(ad_data.get("time_to_first_token_ms", {}).get("mean"), suffix=" ms")
            med_ttft = format_val(ad_data.get("time_to_first_token_ms", {}).get("median"), suffix=" ms")
            m_itl = format_val(ad_data.get("mean_itl_ms", {}).get("mean"), suffix=" ms")
            m_tps = format_val(ad_data.get("tokens_per_second", {}).get("mean"), suffix=" tok/s")
            ad_hr_val = ad_data.get("cache", {}).get("cache_hit_rate")
            ad_hr = format_val(ad_hr_val * 100.0 if ad_hr_val is not None else None, suffix="%")

            lines.append(f"| `{adapter_id}` | `{role}` | {reqs} | {m_ttft} | {med_ttft} | {m_itl} | {m_tps} | {ad_hr} |")
        lines.append("")

    # 5. Methodological Notes
    lines.append("## Measurement Notes & Invariants")
    lines.append("")
    lines.append("- **Missing Metrics Handling**: `mean_itl_ms` is mathematically undefined on completions of $\\le 1$ token and is preserved as `null`. Statistical means and percentiles ignore `null` entries rather than imputing 0.0.")
    lines.append("- **Adapter Warmth vs. Prefix Cache**: In native vLLM serving, KV-cache entries are tagged with the active LoRA adapter ID. Prefix cache reuse is isolated per adapter, making request ordering a critical determinant of server throughput.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze real-world workload replay measurements")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSONL measurement files or directories containing JSONL files",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to save generated Markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save structured JSON analysis summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the Markdown report to stdout",
    )
    args = parser.parse_args()

    records = load_records(args.inputs)
    if not records:
        print(f"Error: No valid measurement records found in {args.inputs}", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_records(records)
    report_md = generate_markdown_report(analysis)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Saved analysis summary to {args.output_json}", file=sys.stderr)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report.open("w", encoding="utf-8") as f:
            f.write(report_md + "\n")
        print(f"Saved Markdown report to {args.output_report}", file=sys.stderr)

    if not args.quiet:
        print(report_md)


if __name__ == "__main__":
    main()

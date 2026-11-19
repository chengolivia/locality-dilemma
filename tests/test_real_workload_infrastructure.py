"""Unit tests for real-workload configuration, measurement, and analysis infrastructure.

Run with:
    python -m unittest tests/test_real_workload_infrastructure.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis.analyze_real_workload import (
    analyze_records,
    compare_strategies,
    compute_numeric_stats,
    format_val,
    generate_markdown_report,
    load_records,
)
from scripts.real.measurement import (
    append_measurement_record,
    build_measurement_record,
    calculate_prometheus_deltas,
    extract_metric_val,
    load_measurement_records,
    parse_prometheus_metrics,
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class RealWorkloadConfigTests(unittest.TestCase):
    """Validate centralized configuration schema and completeness."""

    @unittest.skipUnless(HAS_YAML, "pyyaml is required for configuration tests")
    def test_config_file_exists_and_parses(self):
        config_path = _REPO_ROOT / "configs" / "real_workload.yaml"
        self.assertTrue(config_path.exists(), f"Missing config file: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIsInstance(cfg, dict)
        required_keys = [
            "base_model",
            "adapters",
            "role_to_adapter",
            "vllm_server",
            "generation",
            "ordering_strategies",
            "metrics",
            "paths",
        ]
        for key in required_keys:
            self.assertIn(key, cfg, f"Missing top-level key: {key}")

    @unittest.skipUnless(HAS_YAML, "pyyaml is required for configuration tests")
    def test_adapters_and_roles_configuration(self):
        config_path = _REPO_ROOT / "configs" / "real_workload.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        adapters = cfg["adapters"]
        expected_adapters = {"magicoder", "wikisql", "gsm8k", "drop_explained", "cnn"}
        self.assertEqual(set(adapters.keys()), expected_adapters)

        for name, meta in adapters.items():
            self.assertIn("role", meta)
            self.assertIn("adapter_path", meta)
            self.assertEqual(meta["rank"], 8)
            self.assertEqual(meta["alpha"], 16)
            self.assertEqual(meta["target_modules"], ["q_proj", "v_proj"])

        role_map = cfg["role_to_adapter"]
        expected_roles = {
            "coding": "magicoder",
            "data_querying": "wikisql",
            "numerical_analysis": "gsm8k",
            "document_qa_extraction": "drop_explained",
            "summarization_synthesis": "cnn",
        }
        self.assertEqual(role_map, expected_roles)

    @unittest.skipUnless(HAS_YAML, "pyyaml is required for configuration tests")
    def test_server_and_ordering_configuration(self):
        config_path = _REPO_ROOT / "configs" / "real_workload.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        server_cfg = cfg["vllm_server"]
        self.assertEqual(server_cfg["max_loras"], 5)
        self.assertEqual(server_cfg["max_lora_rank"], 8)
        self.assertTrue(server_cfg["enable_prefix_caching"])
        self.assertTrue(server_cfg["enable_per_request_metrics"])

        strategies = cfg["ordering_strategies"]
        self.assertIn("adapter_interleaved", strategies)
        self.assertIn("adapter_grouped", strategies)


class MeasurementInfrastructureTests(unittest.TestCase):
    """Test Prometheus scraping, delta calculation, and record construction."""

    def test_parse_prometheus_metrics(self):
        raw_prom = """# HELP vllm:prefix_cache_queries_total Total prefix cache queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model="magicoder"} 512.0
vllm:prefix_cache_hits_total{model="magicoder"} 256.0
vllm:prompt_tokens_cached_total 256.0
vllm:kv_cache_usage_perc 0.155
# An unrelated metric
python_gc_objects_collected_total 42.0
"""
        metrics = parse_prometheus_metrics(raw_prom)
        self.assertIn("vllm:prefix_cache_queries_total", metrics)
        self.assertEqual(metrics["vllm:prefix_cache_queries_total"], 512.0)
        self.assertEqual(metrics['vllm:prefix_cache_queries_total{model="magicoder"}'], 512.0)
        self.assertEqual(metrics["vllm:prefix_cache_hits_total"], 256.0)
        self.assertEqual(metrics["vllm:prompt_tokens_cached_total"], 256.0)
        self.assertAlmostEqual(metrics["vllm:kv_cache_usage_perc"], 0.155)

    def test_calculate_prometheus_deltas_normal(self):
        before = {
            "vllm:prefix_cache_queries_total": 1000.0,
            "vllm:prefix_cache_hits_total": 200.0,
            "vllm:prompt_tokens_cached_total": 200.0,
        }
        after = {
            "vllm:prefix_cache_queries_total": 1500.0,
            "vllm:prefix_cache_hits_total": 600.0,
            "vllm:prompt_tokens_cached_total": 300.0,
            "vllm:kv_cache_usage_perc": 0.25,
        }
        deltas = calculate_prometheus_deltas(before, after)
        self.assertEqual(deltas["prefix_cache_queries_delta"], 500.0)
        self.assertEqual(deltas["prefix_cache_hits_delta"], 400.0)
        self.assertEqual(deltas["prompt_tokens_cached_delta"], 100.0)
        self.assertAlmostEqual(deltas["cache_hit_rate"], 400.0 / 500.0)
        self.assertEqual(deltas["kv_cache_usage_perc"], 0.25)

    def test_calculate_prometheus_deltas_zero_queries(self):
        before = {"vllm:prefix_cache_queries_total": 500.0, "vllm:prefix_cache_hits_total": 100.0}
        after = {"vllm:prefix_cache_queries_total": 500.0, "vllm:prefix_cache_hits_total": 100.0}
        deltas = calculate_prometheus_deltas(before, after)
        self.assertEqual(deltas["prefix_cache_queries_delta"], 0.0)
        self.assertEqual(deltas["prefix_cache_hits_delta"], 0.0)
        # When queries delta is 0, cache_hit_rate MUST be None, not ZeroDivisionError or 0.0
        self.assertIsNone(deltas["cache_hit_rate"])

    def test_calculate_prometheus_deltas_missing_counters(self):
        # Invariant: Never silently substitute 0.0 for missing metrics
        before = {}
        after = {"vllm:prefix_cache_queries_total": 500.0}
        deltas = calculate_prometheus_deltas(before, after)
        self.assertIsNone(deltas["prefix_cache_queries_delta"])
        self.assertIsNone(deltas["prefix_cache_hits_delta"])
        self.assertIsNone(deltas["cache_hit_rate"])

    def test_build_measurement_record_preserves_none_itl(self):
        response_json = {
            "id": "cmpl-mock-single-token",
            "object": "text_completion",
            "created": 1732000000,
            "model": "magicoder",
            "choices": [{"text": "A", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 1, "total_tokens": 129},
            "metrics": {
                "time_to_first_token_ms": 45.2,
                "generation_time_ms": 12.0,
                "queue_time_ms": 0.8,
                "mean_itl_ms": None,  # single token output
                "tokens_per_second": 83.3,
            },
        }
        before = {"vllm:prefix_cache_queries_total": 0.0, "vllm:prefix_cache_hits_total": 0.0}
        after = {"vllm:prefix_cache_queries_total": 128.0, "vllm:prefix_cache_hits_total": 0.0}

        record = build_measurement_record(
            seq_num=1,
            task_id="test-task",
            adapter_id="magicoder",
            prompt="Test prompt",
            response_status=200,
            response_json=response_json,
            client_elapsed_ms=58.0,
            metrics_before=before,
            metrics_after=after,
            ordering_strategy="adapter_interleaved",
            role="coding",
        )

        self.assertEqual(record["sequence_number"], 1)
        self.assertEqual(record["request_id"], "cmpl-mock-single-token")
        self.assertEqual(record["task_id"], "test-task")
        self.assertEqual(record["adapter_id"], "magicoder")
        self.assertIsNone(record["metrics"]["mean_itl_ms"])
        self.assertEqual(record["metrics"]["time_to_first_token_ms"], 45.2)

        # Test JSON round-trip serialization preserves null
        dumped = json.dumps(record)
        loaded = json.loads(dumped)
        self.assertIsNone(loaded["metrics"]["mean_itl_ms"])

    def test_jsonl_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_run.jsonl"
            rec1 = {"sequence_number": 1, "adapter_id": "magicoder", "metrics": {"mean_itl_ms": None}}
            rec2 = {"sequence_number": 2, "adapter_id": "wikisql", "metrics": {"mean_itl_ms": 9.5}}

            append_measurement_record(rec1, file_path)
            append_measurement_record(rec2, file_path)

            loaded = load_measurement_records(file_path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["sequence_number"], 1)
            self.assertIsNone(loaded[0]["metrics"]["mean_itl_ms"])
            self.assertEqual(loaded[1]["sequence_number"], 2)
            self.assertEqual(loaded[1]["metrics"]["mean_itl_ms"], 9.5)


class AnalysisScaffoldTests(unittest.TestCase):
    """Test aggregation, statistics calculation, and report generation."""

    def test_compute_numeric_stats_filters_none(self):
        # Invariant: None must NOT be treated as 0.0!
        data = [10.0, None, 20.0, None, 30.0]
        stats = compute_numeric_stats(data)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean"], 20.0)
        self.assertEqual(stats["median"], 20.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 30.0)

    def test_compute_numeric_stats_all_none(self):
        data = [None, None]
        stats = compute_numeric_stats(data)
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["median"])

    def test_analyze_synthetic_fixture(self):
        fixture_path = _REPO_ROOT / "tests" / "fixtures" / "mock_workload_measurements_test_fixture.jsonl"
        self.assertTrue(fixture_path.exists(), f"Fixture missing: {fixture_path}")

        records = load_records([fixture_path])
        self.assertEqual(len(records), 10)

        analysis = analyze_records(records)
        strategies = analysis["strategies"]

        self.assertIn("adapter_interleaved", strategies)
        self.assertIn("adapter_grouped", strategies)

        int_data = strategies["adapter_interleaved"]
        grp_data = strategies["adapter_grouped"]

        self.assertEqual(int_data["num_requests"], 5)
        self.assertEqual(grp_data["num_requests"], 5)

        # In interleaved, 1 record has mean_itl_ms = null -> count of valid ITL values should be 4
        self.assertEqual(int_data["mean_itl_ms"]["count"], 4)
        self.assertIsNotNone(int_data["mean_itl_ms"]["mean"])

        # In grouped, request 7 and 9 have significant cache hits -> cache hits delta should be > 0
        self.assertGreater(grp_data["cache"]["hits_delta_sum"], 0)
        self.assertGreater(grp_data["cache"]["aggregate_cache_hit_rate"], 0.0)

        # Comparison verification
        comp = analysis["comparison"]
        self.assertIn("ttft", comp)
        self.assertIn("cache_hit_rate", comp)
        self.assertIn("tokens_per_second", comp)

        # Grouped TTFT should be lower due to cache hits on repeat adapters
        self.assertGreater(comp["ttft"]["delta_mean_ms"], 0.0)
        self.assertGreater(comp["ttft"]["speedup_ratio"], 1.0)
        self.assertGreater(comp["cache_hit_rate"]["delta_percentage_points"], 0.0)

    def test_generate_markdown_report_on_synthetic_fixture(self):
        fixture_path = _REPO_ROOT / "tests" / "fixtures" / "mock_workload_measurements_test_fixture.jsonl"
        records = load_records([fixture_path])
        analysis = analyze_records(records)

        md = generate_markdown_report(analysis)
        self.assertIn("# Real-World Workload Replay: Locality Dilemma Analysis", md)
        self.assertIn("## Executive Comparison: Interleaved vs. Grouped Orderings", md)
        self.assertIn("## Overall Strategy Performance", md)
        self.assertIn("## Prefix Cache Performance", md)
        self.assertIn("## Per-Adapter Breakdown", md)
        self.assertIn("`adapter_interleaved`", md)
        self.assertIn("`adapter_grouped`", md)
        self.assertIn("magicoder", md)
        self.assertIn("wikisql", md)
        self.assertIn("gsm8k", md)


if __name__ == "__main__":
    unittest.main()

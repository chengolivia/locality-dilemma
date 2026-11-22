"""Unit tests for the real-workload replay execution runner.

Run with:
    python -m unittest tests/test_workload_replay.py -v
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.real.run_workload_replay import (
    build_adapter_grouped_schedule,
    build_adapter_interleaved_schedule,
    load_config,
    load_replay_workload,
    resolve_served_model_name,
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class WorkloadReplayRunnerTests(unittest.TestCase):
    """Test replay loading, schedule construction, and model name resolution."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = _REPO_ROOT / "configs" / "real_workload.yaml"
        cls.replay_path = _REPO_ROOT / "data" / "agent_replay_20tasks" / "replay_8192.jsonl"
        cls.fixture_replay_path = _REPO_ROOT / "tests" / "fixtures" / "mock_workload_measurements_test_fixture.jsonl"

    @unittest.skipUnless(HAS_YAML, "pyyaml required for config tests")
    def test_load_config(self):
        cfg = load_config(self.config_path)
        self.assertIn("base_model", cfg)
        self.assertIn("adapters", cfg)
        self.assertIn("vllm_server", cfg)
        self.assertIn("ordering_strategies", cfg)

    def test_resolve_served_model_name(self):
        cfg = {"adapters": {"magicoder": {"hf_repo": "predibase/magicoder"}}}
        self.assertEqual(resolve_served_model_name("predibase/magicoder", cfg), "magicoder")
        self.assertEqual(resolve_served_model_name("magicoder", cfg), "magicoder")
        self.assertEqual(resolve_served_model_name("predibase/wikisql", None), "wikisql")
        self.assertEqual(resolve_served_model_name("custom_model", None), "custom_model")

    def test_schedule_construction_on_real_replay_workload(self):
        if not self.replay_path.exists():
            self.skipTest(f"Replay file not present: {self.replay_path}")

        records = load_replay_workload(self.replay_path)
        self.assertEqual(len(records), 124)

        # 1. Test adapter_interleaved schedule
        interleaved = build_adapter_interleaved_schedule(records)
        self.assertEqual(len(interleaved), 124)

        # Count preserved per adapter
        original_counts = Counter(r["adapter_id"] for r in records)
        interleaved_counts = Counter(r["adapter_id"] for r in interleaved)
        self.assertEqual(original_counts, interleaved_counts)

        # Verify step sequences within each task are strictly ascending (0, 1, 2...)
        task_steps = defaultdict(list)
        for r in interleaved:
            task_steps[r["task_id"]].append(r["step"])
        for t_id, steps in task_steps.items():
            self.assertEqual(steps, sorted(steps), f"Step order violated in task {t_id}")
            self.assertEqual(steps[0], 0, f"Task {t_id} does not start at step 0")

        # Verify adapter transitions are heavily interleaved (>100 transitions)
        int_switches = sum(
            1 for i in range(1, len(interleaved))
            if interleaved[i]["adapter_id"] != interleaved[i-1]["adapter_id"]
        )
        self.assertGreater(int_switches, 100, f"Expected high adapter switching, got {int_switches}")

        # 2. Test adapter_grouped schedule
        grouped = build_adapter_grouped_schedule(records)
        self.assertEqual(len(grouped), 124)

        grouped_counts = Counter(r["adapter_id"] for r in grouped)
        self.assertEqual(original_counts, grouped_counts)

        # Verify step sequences within each task are strictly ascending
        task_steps_grp = defaultdict(list)
        for r in grouped:
            task_steps_grp[r["task_id"]].append(r["step"])
        for t_id, steps in task_steps_grp.items():
            self.assertEqual(steps, sorted(steps), f"Step order violated in task {t_id}")
            self.assertEqual(steps[0], 0, f"Task {t_id} does not start at step 0")

        # Verify adapter transitions are minimal (exactly 4 transitions between 5 adapters)
        grp_switches = sum(
            1 for i in range(1, len(grouped))
            if grouped[i]["adapter_id"] != grouped[i-1]["adapter_id"]
        )
        self.assertEqual(grp_switches, 4, f"Expected 4 transitions in grouped schedule, got {grp_switches}")

    def test_schedule_construction_on_mock_fixture(self):
        records = [
            {"task_id": "task_a", "step": 0, "adapter_id": "predibase/magicoder", "prompt": "p0"},
            {"task_id": "task_a", "step": 1, "adapter_id": "predibase/magicoder", "prompt": "p1"},
            {"task_id": "task_b", "step": 0, "adapter_id": "predibase/wikisql", "prompt": "p2"},
            {"task_id": "task_b", "step": 1, "adapter_id": "predibase/wikisql", "prompt": "p3"},
        ]
        interleaved = build_adapter_interleaved_schedule(records)
        grouped = build_adapter_grouped_schedule(records)

        self.assertEqual(len(interleaved), 4)
        self.assertEqual(len(grouped), 4)

        # In interleaved: magicoder (0) -> wikisql (0) -> magicoder (1) -> wikisql (1)
        self.assertEqual(interleaved[0]["task_id"], "task_a")
        self.assertEqual(interleaved[0]["step"], 0)
        self.assertEqual(interleaved[1]["task_id"], "task_b")
        self.assertEqual(interleaved[1]["step"], 0)
        self.assertEqual(interleaved[2]["task_id"], "task_a")
        self.assertEqual(interleaved[2]["step"], 1)
        self.assertEqual(interleaved[3]["task_id"], "task_b")
        self.assertEqual(interleaved[3]["step"], 1)

        # In grouped: all task_a (steps 0, 1) then all task_b (steps 0, 1)
        self.assertEqual(grouped[0]["task_id"], "task_a")
        self.assertEqual(grouped[0]["step"], 0)
        self.assertEqual(grouped[1]["task_id"], "task_a")
        self.assertEqual(grouped[1]["step"], 1)
        self.assertEqual(grouped[2]["task_id"], "task_b")
        self.assertEqual(grouped[2]["step"], 0)
        self.assertEqual(grouped[3]["task_id"], "task_b")
        self.assertEqual(grouped[3]["step"], 1)


if __name__ == "__main__":
    unittest.main()

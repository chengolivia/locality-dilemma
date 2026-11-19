"""Unit tests for synthetic benchmarks and analysis harnesses.
Run with:
    python -m unittest tests/test_benchmarks.py -v
"""
import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SYNTHETIC_DIR = _REPO_ROOT / "scripts" / "synthetic"
if str(_SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNTHETIC_DIR))
_ANALYSIS_DIR = _REPO_ROOT / "scripts" / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

import benchmark_lora_transfer as lora_bench
import benchmark_prefill_latency as prefill_bench


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pandas as pd
    import matplotlib
    HAS_ANALYSIS_DEPS = True
except ImportError:
    HAS_ANALYSIS_DEPS = False


class BenchmarkTests(unittest.TestCase):
    def test_compute_statistics(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = lora_bench.compute_statistics(latencies)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)
        self.assertEqual(stats["p95"], 50.0)

    @unittest.skipUnless(HAS_TORCH, "torch is required for this test")
    def test_dummy_lora_state_dict_and_size(self):
        import torch
        state_dict = lora_bench.create_dummy_lora_state_dict(
            rank=8,
            num_layers=2,
            hidden_dim=64,
            target_modules=["q_proj", "v_proj"],
            dtype=torch.float32,
            pin_memory=False,
            device="cpu",
        )
        # 2 layers * 2 modules * 2 matrices (A and B) = 8 tensors
        self.assertEqual(len(state_dict), 8)
        # Each layer/module has:
        # A: (8, 64) = 512 params
        # B: (64, 8) = 512 params
        # Total params = 4 * 1024 = 4096 params
        params, size_mb = lora_bench.calculate_state_dict_size(state_dict)
        self.assertEqual(params, 4096)
        self.assertAlmostEqual(size_mb, (4096 * 4) / (1024.0 * 1024.0), places=4)

    @unittest.skipUnless(HAS_TORCH, "torch is required for this test")
    def test_measure_t_swap_dryrun(self):
        median_ms, stats = lora_bench.measure_t_swap(
            rank=8,
            hidden_dim=64,
            num_layers=2,
            target_modules=["q_proj"],
            iterations=3,
            warmup=1,
            pin_memory=False,
            device="cpu",
            return_stats=True,
        )
        self.assertIsInstance(median_ms, float)
        self.assertGreater(median_ms, 0.0)
        self.assertIn("median", stats)
        self.assertIn("mean", stats)

    @unittest.skipUnless(HAS_TORCH, "torch is required for this test")
    def test_prefill_latency_dryrun(self):
        import torch
        MistralConfig, MistralForCausalLM = prefill_bench.load_mistral_classes()
        self.assertIsNotNone(MistralConfig)
        self.assertIsNotNone(MistralForCausalLM)

        model, config = prefill_bench.create_or_load_mistral_model(
            MistralConfig_cls=MistralConfig,
            MistralForCausalLM_cls=MistralForCausalLM,
            model_path=None,
            num_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=2,
            num_kv_heads=1,
            vocab_size=1000,
            max_position_embeddings=512,
            dtype_str="float32",
            use_cache=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertIsNotNone(model)

        t_prefill, stats = prefill_bench.measure_t_prefill(
            model=model,
            context_length=32,
            batch_size=1,
            device="cpu",
            iterations=2,
            warmup=1,
            use_cache=True,
            return_stats=True,
        )
        self.assertIsInstance(t_prefill, float)
        self.assertGreater(t_prefill, 0.0)
        self.assertIn("median", stats)

    @unittest.skipUnless(HAS_ANALYSIS_DEPS, "pandas and matplotlib required for analyze_results test")
    def test_analyze_results_pipeline(self):
        import pandas as pd
        from unittest.mock import patch
        import analyze_results

        with TemporaryDirectory() as temp:
            temp_path = Path(temp)
            timings_file = temp_path / "timings.csv"
            out_dir = temp_path / "analysis"

            # Create dummy timings CSV
            with open(timings_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["prefix_len", "lora_rank", "prefill_saved_ms", "swap_ms"])
                writer.writerow([128, 8, 10.0, 5.0])
                writer.writerow([256, 8, 20.0, 5.0])
                writer.writerow([128, 16, 10.0, 15.0])
                writer.writerow([256, 16, 20.0, 15.0])

            argv = ["analyze_results.py", "--input", str(timings_file), "--output-dir", str(out_dir)]
            with patch("sys.argv", argv):
                analyze_results.main()

            self.assertTrue((out_dir / "summary.csv").is_file())
            self.assertTrue((out_dir / "break_even.csv").is_file())
            self.assertTrue((out_dir / "latency_heatmap.png").is_file())

            summary_df = pd.read_csv(out_dir / "summary.csv")
            self.assertEqual(len(summary_df), 4)
            self.assertIn("winner", summary_df.columns)


if __name__ == "__main__":
    unittest.main()

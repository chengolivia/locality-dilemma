#!/usr/bin/env python3
"""Root compatibility wrapper for benchmark_lora_transfer.
Implementation lives in scripts/synthetic/benchmark_lora_transfer.py.
"""
import sys
from pathlib import Path

_SYNTHETIC_DIR = Path(__file__).resolve().parent / "scripts" / "synthetic"
if str(_SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNTHETIC_DIR))

from benchmark_lora_transfer import *

if __name__ == "__main__":
    from benchmark_lora_transfer import main
    main()

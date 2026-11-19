#!/usr/bin/env python3
"""Compatibility wrapper for run_kv_sharing_experiment.
Implementation resides in scripts/real/run_kv_sharing_experiment.py.
"""
import sys
from pathlib import Path

_REAL_DIR = Path(__file__).resolve().parent / "real"
if str(_REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_REAL_DIR))

from run_kv_sharing_experiment import *

if __name__ == "__main__":
    from run_kv_sharing_experiment import main
    main()

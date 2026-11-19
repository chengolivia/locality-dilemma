#!/usr/bin/env python3
"""Root compatibility wrapper for run_generation.
Implementation lives in scripts/real/run_generation.py.
"""
import sys
from pathlib import Path

_REAL_DIR = Path(__file__).resolve().parent / "scripts" / "real"
if str(_REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_REAL_DIR))

from run_generation import *

if __name__ == "__main__":
    from run_generation import main
    main()

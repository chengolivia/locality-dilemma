#!/usr/bin/env python3
"""Root compatibility wrapper for generate_workloads.
Implementation lives in scripts/real/generate_workloads.py.
"""
import sys
from pathlib import Path

_REAL_DIR = Path(__file__).resolve().parent / "scripts" / "real"
if str(_REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_REAL_DIR))

from generate_workloads import *

if __name__ == "__main__":
    from generate_workloads import main
    main()

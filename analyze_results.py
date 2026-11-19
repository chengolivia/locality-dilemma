#!/usr/bin/env python3
"""Root compatibility wrapper for analyze_results.
Implementation lives in scripts/analysis/analyze_results.py.
"""
import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent / "scripts" / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from analyze_results import *

if __name__ == "__main__":
    from analyze_results import main
    main()

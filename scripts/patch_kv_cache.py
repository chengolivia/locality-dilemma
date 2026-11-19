#!/usr/bin/env python3
"""Compatibility wrapper for patch_kv_cache.
Implementation resides in scripts/real/patch_kv_cache.py.
"""
import sys
from pathlib import Path

_REAL_DIR = Path(__file__).resolve().parent / "real"
if str(_REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_REAL_DIR))

from patch_kv_cache import *

if __name__ == "__main__":
    from patch_kv_cache import main
    main()

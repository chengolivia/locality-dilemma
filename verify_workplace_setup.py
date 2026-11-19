#!/usr/bin/env python3
"""Root compatibility wrapper for verify_workplace_setup.
Implementation lives in scripts/workplace/verify_workplace_setup.py.
"""
import sys
from pathlib import Path

_WORKPLACE_DIR = Path(__file__).resolve().parent / "scripts" / "workplace"
if str(_WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKPLACE_DIR))

from verify_workplace_setup import *

if __name__ == "__main__":
    from verify_workplace_setup import main
    main()

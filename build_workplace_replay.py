#!/usr/bin/env python3
"""Root compatibility wrapper for build_workplace_replay.
Implementation lives in scripts/workplace/build_workplace_replay.py.
"""
import sys
from pathlib import Path

_WORKPLACE_DIR = Path(__file__).resolve().parent / "scripts" / "workplace"
if str(_WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKPLACE_DIR))

from build_workplace_replay import *

if __name__ == "__main__":
    from build_workplace_replay import main
    main()

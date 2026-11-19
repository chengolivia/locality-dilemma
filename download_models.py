#!/usr/bin/env python3
"""Root compatibility wrapper for download_models.
Implementation lives in scripts/setup/download_models.py.
"""
import sys
from pathlib import Path

_SETUP_DIR = Path(__file__).resolve().parent / "scripts" / "setup"
if str(_SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(_SETUP_DIR))

from download_models import *

if __name__ == "__main__":
    from download_models import main
    main()

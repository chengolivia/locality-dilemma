#!/usr/bin/env python3
"""Root compatibility wrapper for test_workplace_tools.
Implementation lives in tests/test_workplace_tools.py.
"""
import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_workplace_tools import WorkplaceTests

if __name__ == "__main__":
    unittest.main()

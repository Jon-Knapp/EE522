"""
conftest.py - pytest setup for the gallop simulator tests.

Puts the project root on sys.path so the tests can import config.py and the
gallop_sim package no matter which folder pytest is launched from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

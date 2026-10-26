#!/usr/bin/env python3
"""Run OMH straight from a checkout: ``python main.py``."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omh.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

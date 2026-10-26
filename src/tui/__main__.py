"""`python -m omh.tui`: the OMH TUI by hand.

The frontend shim does not start it this way: it imports `omh.tui.app.main`
itself, so an OMH that cannot import exits with `FALLBACK_EXIT` instead of a
traceback.
"""

from __future__ import annotations

import os

from .app import main

if __name__ == "__main__":
    raise SystemExit(main(os.environ))

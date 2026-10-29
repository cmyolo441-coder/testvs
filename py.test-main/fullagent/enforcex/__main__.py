"""Executable entry point for ``python -m fullagent.enforcex``.

With no arguments, run the package's complete self-test (the same contract
as every standalone fullagent module). With arguments, delegate to the
enforcex command-line interface, e.g.::

    python -m fullagent.enforcex selftest
    python -m fullagent.enforcex scan suspicious.txt

Keeping both forms available makes the package discoverable without a
console-script installation and preserves the existing ``_self_test`` API.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from . import _self_test
        return int(_self_test() or 0)
    from .cli import main as cli_main
    return int(cli_main(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

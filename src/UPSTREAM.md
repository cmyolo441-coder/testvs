# Upstream harness source (reference copy)

This directory is an unmodified copy of `src/` from
[`rlaope/oh-my-hermes`](https://github.com/rlaope/oh-my-hermes) at commit
`12adcf60c3de56f06ba937d1a54e72938c31374c`.

It is kept because those packages contain the upstream routing, coding,
evidence, workflow, TUI and skill machinery that the lean OMH CLI is inspired
by. Nothing here is imported by `main.py` or by the `omh` package, and the
wheel built from `pyproject.toml` does not include it — the runnable agent is
the 23-module package at the repository root.

License: MIT, the same as this repository (see `../LICENSE`). Any bug in this
tree belongs to upstream; send fixes upstream rather than patching this copy,
or adapt the piece you need into `omh/` with tests.

Do not add this directory to `PYTHONPATH` while using the OMH CLI: `src/omh`
is the upstream package name and would shadow the runnable `omh/` package at
the repository root.

"""FullAgent — an advanced terminal AI agent.

Package layout:
    config.py       providers, models, effort levels, paths
    systemprompt.py the ONE home of every system prompt (single source)
    mastermind.py   prompt coherence: sealed vault, gate, composer, lineage
    tools.py        tool registry: files, shell, search, real-time web
    client.py       OpenAI-compatible streaming chat client
    agent.py        agent loop (LLM <-> tools), event-sourced on the kernel
    kernel.py       Temporal Kernel: append-only, content-addressed event log
    memory.py       Hippocampus: episodic memory + dead-end ledger
    goal.py         goal contracts with machine-checkable done-criteria
    judge.py        deterministic verification (no LLM judging)
    roles.py        capability catalog used by deterministic planners
    workflows.py    saved multi-step pipelines — phased orchestration
    report.py       enterprise audit export (md/html) + forecasting
    autopilot.py    self-routing brain: auto-enables goal / web

    v3 advanced subsystems (all event-sourced on the same kernel):
    router.py       smart model routing — cheapest capable model per task
    semantic.py     semantic vector memory — meaning-based recall
    speculate.py    speculative execution — prefetch read-only tool calls
    dashboard.py    live observability — real-time ledger projection
    daemon.py       mission control — resumable long-running missions
    healer.py       self-healing — root-cause capture, fix, retry, lesson
    skills.py       skill forge — self-authored, safety-gated tools
    council.py      adversarial debate — thesis/antithesis + blind judge

    v4 professional subsystems (all event-sourced on the same kernel):
    taint.py        static analysis — taint flows, complexity, import cycles
    kgraph.py       knowledge graph — entities + typed relations, impact sets
    cov.py          real line coverage — sys.settrace measurement
    fuzz.py         property-based fuzzing — generators + crash shrinking
    mutate.py       mutation testing — AST mutants vs the test suite

    tui.py          terminal UI: double-line prompt box, overlays, rendering
    __main__.py     entry point
"""

__version__ = "4.1.0"


import importlib.abc as _abc
import importlib.util as _util
import os as _os
import sys as _sys


class _DirectStatFinder(_abc.MetaPathFinder):
    """Fallback finder for 9p/FUSE mounts where os.listdir() truncates large dirs."""

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("fullagent."):
            return None
        parts = fullname.split(".")
        base_dir = _os.path.dirname(_os.path.abspath(__file__))
        cand_py = _os.path.join(base_dir, *parts[1:]) + ".py"
        if _os.path.isfile(cand_py):
            return _util.spec_from_file_location(fullname, cand_py)
        return None


if not any(type(f).__name__ == "_DirectStatFinder" for f in _sys.meta_path):
    _sys.meta_path.append(_DirectStatFinder())

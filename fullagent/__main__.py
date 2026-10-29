"""Entry point: python main.py  (or python -m fullagent).

Interactive TUI by default; headless subcommands (Appendix A) operate on
the persistent event log without launching the UI:

    python main.py                     launch the interactive TUI
    python main.py verify-log          Merkle integrity check of the log
    python main.py replay              replay the session log as a text film
    python main.py rewind <seq>        time travel: files + agent state
    python main.py revert <seq>        files only; agent keeps the memory
    python main.py cost                spend breakdown from the fold
    python main.py why <seq>           causal chain back to the instruction
    python main.py goal status         the Goal Compass
    python main.py stats               Oracle post-run analysis
    python main.py forge               environment digest
"""

from __future__ import annotations

import sys


def _force_utf8() -> None:
    """Make sure output is UTF-8 even if the locale says otherwise."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _headless(argv: list[str]) -> int:
    """Run a headless subcommand against the persistent event log."""
    from . import config
    from .agent import Agent
    from .config import Config
    from .kernel import fold, replay

    config.ensure_dirs()
    cfg = Config.load()
    agent = Agent(cfg)
    cmd = argv[0]

    if cmd == "verify-log":
        ok, msg = agent.log.verify()
        print(f"{'OK' if ok else 'FAIL'}: {msg}")
        return 0 if ok else 1

    if cmd == "replay":
        for ev in replay(agent.log):
            d = ev.data
            t = ev.type
            if t in ("user.message", "assistant.message"):
                who = "❯" if t == "user.message" else "◆"
                print(f"[{ev.seq}] {who} {str(d.get('text', ''))[:70]}")
            elif t == "tool.call":
                print(f"[{ev.seq}] ⚙ {d.get('name', '')}")
            elif t == "tool.result":
                icon = "✓" if d.get("status") == "done" else "✗"
                print(f"[{ev.seq}] {icon} {d.get('name', '')} "
                      f"({d.get('status', '')})")
            elif t == "snapshot.taken":
                print(f"[{ev.seq}] 📸 snapshot "
                      f"{str(d.get('tree', ''))[:10]}")
            elif t == "clause.proven":
                print(f"[{ev.seq}] ★ clause {d.get('clause', '')} PROVEN")
            elif t == "goal.closed":
                print(f"[{ev.seq}] ■ GOAL {d.get('state', '')}")
        return 0

    if cmd == "rewind":
        if len(argv) < 2:
            print("usage: rewind <seq>", file=sys.stderr)
            return 2
        try:
            seq = int(argv[1])
        except ValueError:
            print(f"rewind: seq must be an integer, got {argv[1]!r}",
                  file=sys.stderr)
            return 2
        new_head, kept = agent.rewind_to(seq)
        print(f"rewound to seq {new_head} — {kept} message(s) kept")
        return 0

    if cmd == "revert":
        if len(argv) < 2:
            print("usage: revert <seq>", file=sys.stderr)
            return 2
        try:
            seq = int(argv[1])
        except ValueError:
            print(f"revert: seq must be an integer, got {argv[1]!r}",
                  file=sys.stderr)
            return 2
        result = agent.revert_files_to(seq)
        if "error" in result:
            print(f"error: {result['error']}", file=sys.stderr)
            return 1
        print(f"files reverted — {result['restored']} restored, "
              f"{result['removed']} removed (agent memory kept)")
        return 0

    if cmd == "cost":
        st = fold(agent.log)
        print(f"cost: {st.cost_summary()}")
        print(f"tool calls: {st.tool_calls}   errors: {st.tool_errors}   "
              f"commands: {st.commands_run}")
        if st.files_touched:
            print("files touched: " + ", ".join(sorted(st.files_touched)))
        return 0

    if cmd == "why":
        if len(argv) < 2:
            print("usage: why <seq>", file=sys.stderr)
            return 2
        try:
            seq = int(argv[1])
        except ValueError:
            print(f"why: seq must be an integer, got {argv[1]!r}",
                  file=sys.stderr)
            return 2
        target = next((e for e in agent.log.events() if e.seq == seq), None)
        if target is None:
            print(f"no event at seq {seq}", file=sys.stderr)
            return 1
        for i, ev in enumerate(agent.log.why(target.id)):
            indent = "  " * i
            preview = ""
            if ev.type in ("user.message", "assistant.message"):
                preview = str(ev.data.get("text", ""))[:50]
            elif ev.type in ("tool.call", "tool.result"):
                preview = str(ev.data.get("name", ""))
            clause = f"  [clause {ev.correlation_id}]" \
                if ev.correlation_id else ""
            print(f"{indent}← seq {ev.seq} {ev.type} "
                  f"({ev.actor}) {preview}{clause}")
        return 0

    if cmd == "goal":
        sub = argv[1] if len(argv) > 1 else "status"
        if sub == "status":
            print(agent.goal.format())
        elif sub == "prove":
            if len(argv) < 3:
                print("usage: goal prove <clause-id>", file=sys.stderr)
                return 2
            ok, detail = agent.goal.prove_by_predicate(argv[2])
            print(f"{'PROVEN' if ok else 'FAILED'}: {detail}")
            return 0 if ok else 1
        elif sub == "close":
            result = agent.goal.close(fresh=True)
            print(f"GOAL CLOSED: {result['state']}")
            for r in result["reasons"]:
                print(f"  - {r}")
            print(result["bundle"])
            return 0 if result["state"] == "ACHIEVED" else 1
        else:
            print(f"unknown goal subcommand: {sub}", file=sys.stderr)
            return 2
        return 0

    if cmd == "stats":
        print(agent.oracle.format_report())
        return 0

    if cmd == "forge":
        d = agent.forge.probe()
        print(f"environment digest: {d['digest']}")
        print(f"  os {d['os']} {d['arch']}   python {d['python']}")
        print(f"  cwd {d['cwd']}")
        print(f"  lockfile {d['lockfile_hash'] or 'none'}")
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


def _headless_print(prompt: str, ns) -> int:
    """Run one headless turn and print the result (text or JSON)."""
    from . import config
    from .agent import Agent
    from .config import Config

    config.ensure_dirs()
    cfg = Config.load()
    agent = Agent(cfg)

    def _noop(*a, **k):
        pass

    def _on_status(s: str) -> None:
        # worker 9/20 — live per-call progress in headless mode; stderr
        # so stdout stays clean for --print piping / --output-format json
        if s.startswith("progress:"):
            from .callprogress import format_status
            text = format_status(s)
            if text:
                print(text, file=sys.stderr, flush=True)

    turn = agent.run_turn(
        prompt,
        on_token=_noop, on_reasoning=_noop,
        on_tool_call=_noop, on_tool_update=_noop,
        on_status=_on_status, approve=lambda tool, args: True,
    )
    if getattr(ns, "output_format", "text") == "json":
        from . import jsonout
        jsonout.emit_json(jsonout.turn_to_json(
            turn, getattr(agent, "session_id", None),
            extra={"agent": agent}))
    else:
        print(turn.assistant_text or "(no response)")
        if turn.error:
            print(f"error: {turn.error}", file=sys.stderr)
            return 1
    return 0


def _self_check() -> int:
    """Smoke-test a frozen build: import every subsystem, build an Agent.

    PyInstaller only sees static imports, so a bundled binary can pass
    --version and still die the moment a lazily-imported subsystem is
    touched (the v3.2.0 build shipped without fullagent.mastermind).
    This resolves the whole lazy table and constructs the real agent
    against a throwaway home directory — no network, no user state.
    """
    import importlib
    import os
    import pkgutil
    import tempfile
    import traceback

    os.environ["FULLAGENT_HOME"] = tempfile.mkdtemp(prefix="selfcheck-")

    import fullagent

    failures: list[str] = []
    imported = 0
    for info in pkgutil.walk_packages(fullagent.__path__, prefix=f"{fullagent.__name__}."):
        if info.name.endswith(".__main__"):
            continue
        name = info.name
        try:
            importlib.import_module(name)
            imported += 1
        except BaseException as exc:
            failures.append(f"import {name}: {exc!r}")

    try:
        from . import config
        from .agent import Agent, _ensure_submodules
        from .config import Config

        _ensure_submodules()
        config.ensure_dirs()
        cfg = Config.load()
        Agent(cfg)
    except BaseException as exc:
        traceback.print_exc()
        failures.append(f"startup: {exc!r}")

    if failures:
        for line in failures:
            print(f"SELF-CHECK FAIL: {line}", file=sys.stderr)
        return 1
    from . import __version__
    print(f"SELF-CHECK OK: fullagent v{__version__}, "
          f"{imported} submodules imported, agent constructed")
    return 0


def main() -> int:
    _force_utf8()

    argv = sys.argv[1:]
    if argv and argv[0] in ("--version", "-v"):
        from . import __version__
        print(f"fullagent v{__version__}")
        return 0
    if argv and argv[0] == "--self-check":
        return _self_check()
    # Feature: --resume/--continue flags (peeled before headless dispatch)
    import argparse as _ap
    _p = _ap.ArgumentParser(epilog=__doc__, formatter_class=_ap.RawDescriptionHelpFormatter)
    _p.add_argument("--resume", nargs="?", const="latest", default=None,
                    metavar="ID",
                    help="resume a saved session (latest if no id)")
    _p.add_argument("--continue", dest="continue_", action="store_true",
                    help="resume the most recent session")
    _p.add_argument("--print", dest="print_", metavar="PROMPT",
                    default=None,
                    help="run one headless turn with PROMPT and print result")
    _p.add_argument("--output-format", choices=["text", "json"],
                    default="text",
                    help="headless turn output format (text|json)")
    _ns, argv = _p.parse_known_args(argv)
    if _ns.print_ is not None:
        return _headless_print(_ns.print_, _ns)
    if argv:
        return _headless(argv)

    # SPEED: print an instant banner BEFORE heavy init so the user
    # sees something immediately (<10ms), not after 1-2s of silence.
    # v4 aurora polish: gradient accent with plain fallback.
    try:
        from .fx import RAINBOW, gradient_text
        from rich.console import Console
        c = Console(force_terminal=True, width=60)
        c.print("◆ ", style="bold #a55eea", end="")
        c.print(gradient_text("FullAgent", RAINBOW), end="")
        c.print(" — starting...", style="dim")
    except Exception:
        print("◆ FullAgent — starting...", flush=True)

    from . import config
    from .agent import Agent
    from .config import Config
    from .tui import UI

    config.ensure_dirs()
    cfg = Config.load()
    agent = Agent(cfg)
    # Feature: restore a saved session if requested
    if _ns.continue_ or _ns.resume is not None:
        from . import sessions
        wanted = _ns.resume if _ns.resume not in (None, "latest") else None
        data = sessions.load(wanted or "latest")
        if data is not None:
            print(sessions.resume_into(agent, data), flush=True)
        else:
            print(f"no saved session found"
                  f"{' for ' + wanted if wanted else ''}", flush=True)
    ui = UI(cfg, agent)

    ui.print_banner()
    ui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

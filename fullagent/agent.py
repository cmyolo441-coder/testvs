"""Agent loop: LLM <-> tools, event-sourced on the Temporal Kernel.

Every user message, tool call, tool result, assistant reply, and cost is
an immutable event in the append-only log (kernel.py). Conversation state,
cost, goal distance, dead-ends and verdicts are projections folded from
that log.

Wired here (all enforced mechanically, rung 1 — never as prompt advice):
  * §38.1  Total attribution — with an active goal, every tool call is
           bound to the focus clause (correlation_id). An action with no
           open clause to serve is rejected as an OrphanAction.
  * A2/I3  Snapshot-before-write — no mutating tool runs without a
           committed recovery path in the snapshot store.
  * §37.4  Anti-clauses re-checked after EVERY successful write.
  * I8     Budget governor — a breach pauses the turn, never silent kill.
  * §13.4  Loop detection — exact repeats are sealed and interrupted.
  * §38.3  Goal kernel tick — distance measured, focus re-aimed.
  * §9     Time travel — rewind (files + state), revert (files only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ._foundation import get_logger
from .adaptivecap import AdaptiveTurnCap
from .earlyexit import EarlyExitTracker as _EarlyExitTracker
from .loopdetect import LoopDetector as _RetryLoopDetector, normalize_error

_log = get_logger("agent")

# ---------------------------------------------------------------------------
# Lazy submodule imports (Worker 8/20 — startup time).
#
# Importing this module used to pull ~55 subsystem modules in eagerly
# (~2.5s of interpreter work on a cold start). The names below now resolve
# lazily via PEP 562: the first attribute access imports the owning
# submodule and caches the result in module globals, so every existing
# reference (`Judge`, `config.MODELS`, `build_registry()`, ...) keeps
# working unchanged — the cost is paid on first use, not at import.
# ---------------------------------------------------------------------------
import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # attribute -> owning submodule (relative to this package)
    "config": ".config",
    "systemprompt": ".systemprompt",
    "AutoPilot": ".autopilot",
    "RouteDecision": ".autopilot",
    "Cassette": ".cassette",
    "APIError": ".client",
    "TurnCancelled": ".client",
    "assistant_message": ".client",
    "chat_blocking": ".client",
    "chat_stream": ".client",
    "estimate_tokens": ".client",
    "is_context_overflow": ".client",
    "Config": ".config",
    "Effort": ".config",
    "Model": ".config",
    "Provider": ".config",
    "PROVIDERS": ".config",
    "model_by_id": ".config",
    "effort_by_key": ".config",
    "AttentionEconomy": ".attention",
    "BanditRouter": ".bandit",
    "Brain": ".brain",
    "CausalEngine": ".causal",
    "CIPilot": ".ci",
    "IntentCompiler": ".compiler",
    "default_drafter": ".compiler",
    "Budget": ".cortex",
    "BudgetGovernor": ".cortex",
    "bound_prompt": ".promptbudget",
    "schemas_tokens": ".promptbudget",
    "prompt_wire_tokens": ".promptbudget",
    "LoopDetector": ".cortex",
    "Council": ".council",
    "DebateTournament": ".debate",
    "DualProcess": ".dual",
    "KnowledgeFabric": ".fabric",
    "ModelChecker": ".formal",
    "Homeostasis": ".homeo",
    "TreeSearch": ".mcts",
    "TimelineMerger": ".merge",
    "TERMINAL_STATES": ".goal",
    "GoalContract": ".goal",
    "export_html": ".report",
    "export_markdown": ".report",
    "forecast": ".report",
    "format_forecast": ".report",
    "CoverageEngine": ".cov",
    "Daemon": ".daemon",
    "Dashboard": ".dashboard",
    "Forge": ".forge",
    "Fuzzer": ".fuzz",
    "Healer": ".healer",
    "Judge": ".judge",
    "EventLog": ".kernel",
    "fold": ".kernel",
    "KnowledgeGraph": ".kgraph",
    "Mastermind": ".mastermind",
    "Hippocampus": ".memory",
    "MutationTester": ".mutate",
    "Nexus": ".nexus",
    "Oracle": ".oracle",
    "Router": ".router",
    "SemanticMemory": ".semantic",
    "SkillForge": ".skills",
    "SnapshotStore": ".snapshots",
    "Speculator": ".speculate",
    "SPECULATIVE_TOOLS": ".speculate",
    "ProgramSynthesizer": ".synth",
    "SynthSpec": ".synth",
    "default_generator": ".synth",
    "StaticAnalyzer": ".taint",
    "Theater": ".theater",
    "RISK_CONFIRM": ".tools",
    "Tool": ".tools",
    "build_registry": ".tools",
    "parse_tool_arguments": ".tools",
    "dispatch_block": ".parallel_tools",
    "Tower": ".tower",
    "ParzenTuner": ".tuner",
    "WorldModel": ".world",
}


def __getattr__(name: str):
    """PEP 562: import the owning submodule on first attribute access."""
    rel = _LAZY_IMPORTS.get(name)
    if rel is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    value = _import_lazy_value(name, rel)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def _import_lazy_value(name: str, rel) -> object:
    """Resolve one _LAZY_IMPORTS entry to its value."""
    if isinstance(rel, tuple):
        # aliased import: ("module", "real_attr")
        rel, real = rel
        return getattr(_importlib.import_module(rel, __package__), real)
    module = _importlib.import_module(rel, __package__)
    # `from . import config` binds the submodule itself, not an attr of it
    return module if rel == "." + name else getattr(module, name)


def _resolve_lazy(*names: str) -> None:
    """Import the named lazy attributes into module globals, now.

    A bare global reference inside a method body (LOAD_GLOBAL) never
    consults the PEP 562 __getattr__ above, so methods that run before
    the bulk import below resolve exactly the names they need.
    Idempotent and cheap after the first call.
    """
    g = globals()
    for name in names:
        if name in g:
            continue
        g[name] = _import_lazy_value(name, _LAZY_IMPORTS[name])


_submodules_loaded = False


def _ensure_submodules() -> None:
    """Bulk-import every lazily-declared submodule into module globals.

    Runs once, on first real use (first turn, first subsystem access,
    first schema build). Afterwards every method body sees the exact
    same globals the old eager imports provided.
    """
    global _submodules_loaded
    if _submodules_loaded:
        return
    _resolve_lazy(*_LAZY_IMPORTS)
    _submodules_loaded = True


# The system prompt lives in systemprompt.py — the single source of truth.
# This module only ever reads it from there.

# Autonomy ladder (§22): L0 observer … L5 autonomous
AUTONOMY_LEVELS = {
    0: "Observer — reads only, no mutations",
    1: "Advisor — proposes, applies nothing",
    2: "Assistant — every mutation needs approval",
    3: "Collaborator — safe actions free, risky need approval (default)",
    4: "Pilot — auto-approve except deletes",
    5: "Autonomous — full freedom within budget",
}

_MUTATING_TOOLS = {"write_file", "edit_file", "create_directory",
                   "copy_path", "move_path", "delete_path", "run_command",
                   "live_shell", "apply_patch"}
_ALWAYS_ASK = {"delete_path"}
# tools whose args name the paths they touch (snapshot targets)
_PATH_ARG_TOOLS = {"write_file": ("path",), "edit_file": ("path",),
                   "create_directory": ("path",), "delete_path": ("path",),
                   "copy_path": ("src", "dst"), "move_path": ("src", "dst")}


@dataclass
class ToolEvent:
    name: str
    args: dict
    result: str = ""
    status: str = "running"   # running | done | error | denied | blocked
    duration: float = 0.0
    clause_id: str | None = None


@dataclass
class Turn:
    user_text: str
    assistant_text: str = ""
    reasoning: str = ""
    tools: list[ToolEvent] = field(default_factory=list)
    model_id: str = ""
    effort: str = ""
    error: str = ""
    usage: dict | None = None
    scorecard: dict = field(default_factory=dict)
    duration: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


def _signature(name: str, args: dict) -> str:
    """Canonical approach signature — the dead-end ledger key."""
    payload = json.dumps({"name": name, "args": args},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def prune_messages(messages: list, window: int = 40,
                   tool_keep: int = 2) -> list:
    """PERF: sliding-window prune for the model-visible history.

    Keeps: (1) system prompt, (2) first user message (task framing),
    (3) the last `window` messages. Old tool outputs are truncated to
    summaries first (newest `tool_keep` stay verbatim). Never mutates
    the input. window=0 disables (returns everything).
    """
    if not messages or window <= 0 or len(messages) <= window + 2:
        return list(messages)
    out = []
    idx = 0
    if messages[0].get("role") == "system":
        out.append(messages[0])
        idx = 1
    rest = messages[idx:]
    first_user = next((m for m in rest if m.get("role") == "user"), None)
    tail = [dict(m) if m.get("role") == "tool" else m
            for m in rest[-window:]]
    # truncate old tool outputs, keep newest tool_keep verbatim
    tool_idx = [i for i, m in enumerate(tail) if m.get("role") == "tool"]
    for i in tool_idx[:max(0, len(tool_idx) - tool_keep)]:
        content = tail[i].get("content", "")
        if isinstance(content, str) and len(content) > 500:
            tail[i]["content"] = (
                content[:500] +
                f"\n…[truncated {len(content) - 500} chars — "
                f"re-read from disk if needed]")
    if first_user is not None and first_user not in out \
            and all(m is not first_user for m in tail):
        out.append(first_user)
    out.extend(tail)
    return out



NOTIFY_EVENTS = ("goal.closed", "focus.stop", "workflow.done")

# MEMORY BOUNDS (worker 19/20): every unbounded runtime structure gets a
# cap. These are live UI/detection state only — the event log is the
# durable record, so trimming here loses nothing the session needs.
_FOCUS_HISTORY_MAX = 64    # stall detection reads only the last 4 ticks
_FILE_HASH_PATHS_MAX = 256  # distinct paths tracked for oscillation


class Notifier:
    """Enterprise event notifications — fire selected kernel events to a
    file sink (JSONL) or an HTTP webhook. Never raises: notifications are
    a courtesy, never a crash path. Configure via /notify."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.sink: str = ""          # "" off | "file:<path>" | http(s) URL
        self.sent = 0
        self.last_error = ""

    def configure(self, sink: str) -> str:
        sink = str(sink or "").strip()
        if sink in ("", "off"):
            self.sink = ""
            return "off"
        if sink.startswith("file:"):
            path = Path(sink[5:]).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.sink = f"file:{path}"
            return self.sink
        if sink.startswith(("http://", "https://")):
            self.sink = sink
            return sink
        raise ValueError("sink must be 'off', 'file:<path>', or an "
                         "http(s):// URL")

    def emit(self, event_type: str, payload: dict) -> bool:
        if not self.sink:
            return False
        # Hardened: a malformed event payload that is not a dict (a
        # bare string) would crash the ``**payload`` expansion below
        # with TypeError — wrap it so a bad event never kills notify.
        if not isinstance(payload, dict):
            payload = {"_raw": payload}
        record = {"event": event_type, "ts": time.time(),
                  "app": "fullagent", **payload}
        try:
            if self.sink.startswith("file:"):
                with open(self.sink[5:], "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            else:
                # Pooled keep-alive session: sink posts to the same host
                # skip the TLS handshake on every event.
                from .client import shared_session
                shared_session().post(self.sink, json=record, timeout=5)
            self.sent += 1
            return True
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def status(self) -> str:
        state = self.sink or "off"
        extra = f" · sent {self.sent}" if self.sent else ""
        extra += f" · last error: {self.last_error}" if self.last_error else ""
        return f"notifications: {state}{extra} · events: " \
               + ", ".join(NOTIFY_EVENTS)


# Worker 14/20 — per-tool timeouts. The 512s-turn incident showed any
# single hung tool (a subprocess that never exits, a deadlocked HTTP
# call) can block a turn indefinitely. Every handler now runs in a
# worker thread and the wait is bounded: default 120s, per-tool
# overrides below. On timeout the process group(s) the tool spawned
# are killed (killpg) and the tool reports a clear error — the turn
# never hangs silently.
TOOL_TIMEOUT_DEFAULT = 120.0
TOOL_TIMEOUTS: dict[str, float] = {
    # shell tools legitimately run long builds/test suites; their own
    # internal timeouts can be raised by the caller, so cap higher here
    "run_command": 600.0,
    "live_shell": 600.0,
    # network tools have their own shorter fail-fast paths; keep them
    # tighter than the default so a wedged connection can't stall a turn
    "web_fetch": 45.0,
    "web_search": 60.0,
}


def _fmt_timeout(limit: float) -> str:
    return str(int(limit)) if float(limit).is_integer() else str(limit)


class ToolTimeout(Exception):
    """A tool call exceeded its per-tool timeout."""

    def __init__(self, name: str, limit: float):
        self.name = name
        self.limit = limit
        super().__init__(f"Tool {name} timed out after "
                         f"{_fmt_timeout(limit)}s and was killed.")


class Agent:
    # cache for _handler_takes_cancel (handler -> bool)
    _cancel_kwarg_cache: dict = {}
    # Lazily-built subsystems (Worker 8/20 — startup time).
    #
    # Constructing an Agent used to instantiate ~45 subsystems eagerly
    # (~1.5s, plus the module imports above). Each factory below builds
    # exactly what the old __init__ line built; Agent.__getattr__
    # materialises and caches it in the instance dict on first access, so
    # `agent.judge`, `agent.forge`, ... behave exactly as before — the
    # cost is paid on first use, not at construction. Factories may only
    # touch attributes that exist right after __init__ (cfg, log, tools,
    # mastermind, messages, session_id) or other lazy subsystems.
    _SUBSYSTEM_FACTORIES: dict[str, Callable[["Agent"], Any]] = {
        "store": lambda self: SnapshotStore(config.APP_DIR / "store"),
        "memory": lambda self: Hippocampus(self.log),
        "judge": lambda self: Judge(self.log),
        "goal": lambda self: GoalContract(self.log, judge=self.judge),
        "brain": lambda self: Brain(self.log, config.APP_DIR / "brain.json"),
        "compiler": lambda self: IntentCompiler(
            self.log, default_drafter(self.provider, self.model, self.effort)),
        "merger": lambda self: TimelineMerger(self.log),
        "theater": lambda self: Theater(self.log),
        "debate": lambda self: DebateTournament(
            self.log, self._debate_speaker,
            [m.id for m in config.MODELS if m.supports_tools][:4]),
        "tower": lambda self: Tower(self),
        "formal": lambda self: ModelChecker(self.log),
        "mcts": lambda self: TreeSearch(self.log, self._mcts_evaluator),
        "causal": lambda self: CausalEngine(self.log),
        "bandit": lambda self: BanditRouter(
            self.log, [m.id for m in config.MODELS[:8]]),
        "synth": lambda self: ProgramSynthesizer(
            self.log,
            default_generator(self.provider, self.model, self.effort),
            registry=self.tools),
        "ci": lambda self: CIPilot(self.log, Path.cwd(),
                                   runner=self._ci_runner),
        "tuner": lambda self: ParzenTuner(self.log, {
            "effort": [e.key for e in config.EFFORTS],
            "worker_steps": ["low", "medium", "high"],
        }),
        "dual": lambda self: DualProcess(self.log, self._dual_fast,
                                         self._dual_slow, brain=self.brain),
        "world": lambda self: WorldModel(self.log, root=Path.cwd()),
        "homeo": lambda self: Homeostasis(self.log, repairs={
            "tool_error_rate": self._repair_reseed_prompts,
            "loop_alerts": self._repair_consolidate,
            "tool_latency_ms": self._repair_warm_caches,
        }),
        "attention": lambda self: AttentionEconomy(self.log),
        "fabric": lambda self: KnowledgeFabric(self.log),
        "autopilot": lambda self: AutoPilot(self.log),
        "nexus": lambda self: Nexus(),
        "forge": lambda self: Forge(self.log),
        "oracle": lambda self: Oracle(self.log,
                                      memory_dir=config.APP_DIR / "memory"),
        "budget_gov": lambda self: BudgetGovernor(self.log, Budget()),
        "loop_det": lambda self: LoopDetector(self.log),
        "router": lambda self: Router(self.log),
        "semantic": lambda self: SemanticMemory(self.log),
        "speculator": lambda self: Speculator(self.log,
                                              runner=self._spec_runner),
        "dashboard": lambda self: Dashboard(self.log),
        "daemon": lambda self: Daemon(self.log, executor=self._daemon_step),
        "healer": lambda self: Healer(self.log),
        "skill_forge": lambda self: SkillForge(
            self.log, skills_dir=config.APP_DIR / "skills"),
        "council": lambda self: Council(self.log,
                                        speaker=self._council_speaker),
        "kgraph": lambda self: KnowledgeGraph(self.log),
        "static": lambda self: StaticAnalyzer(self.log),
        "coverage": lambda self: CoverageEngine(self.log),
        "fuzzer": lambda self: Fuzzer(self.log),
    }

    def __getattr__(self, name: str) -> Any:
        """Build a lazy subsystem on first access (Worker 8/20).

        Only called when normal attribute lookup fails. The factory runs
        once; the result is cached in the instance dict, so explicit
        assignment (tests, monkeypatching) keeps working and never hits
        this path.
        """
        # NOTE: class-attribute lookup only — touching self.<attr> here
        # would recurse.
        factory = type(self)._SUBSYSTEM_FACTORIES.get(name)
        if factory is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}")
        # Bulk-import the submodule globals first: factory bodies (and
        # every other method) reference them as bare globals, which never
        # consult the module __getattr__.
        _ensure_submodules()
        # Double-checked lazy init: parallel tool threads can race the
        # factory, building two instances with last-writer-wins caching
        # (some calls using a discarded subsystem).
        import threading as _th
        with _th.Lock():
            if name in self.__dict__:
                return self.__dict__[name]
            value = factory(self)
            self.__dict__[name] = value
            return value

    def __init__(self, cfg: Config):
        # Only the names __init__ itself needs; everything else stays
        # unimported until first use (Worker 8/20 — startup time).
        _resolve_lazy("build_registry", "config", "EventLog", "Mastermind")
        self.cfg = cfg
        self.tools = build_registry()
        self.session_id = uuid.uuid4().hex[:8]
        self.turns: list[Turn] = []
        # Cancellation flag — the TUI sets this on Esc/Ctrl+C. The turn
        # loop and long-running tools (for example, run_command) check it to
        # stop promptly. Cleared at the start of every turn.
        import threading
        self._cancel_flag = threading.Event()
        # Worker 11/20 — parallel tool execution: serializes the two
        # non-thread-safe touches inside _execute_tool (_error_counts
        # read-modify-write, LoopDetector._last_sig). Uncontended on the
        # sequential path.
        self._tool_exec_lock = threading.Lock()

        # Temporal kernel + the nine subsystems
        config.ensure_dirs()
        self.log = EventLog(config.EVENT_LOG_FILE, session=self.session_id)
        # Mastermind: hash-sealed prompts + the single gate to the model.
        # Built before messages so the very first system prompt is sealed
        # and dispatched through the gate, not assembled by hand.
        self.mastermind = Mastermind(self.log)
        self.messages: list[dict] = []
        self._reseat_system_prompt()

        # Worker 8/20 — startup time: the rest of the old __init__ (~45
        # subsystem constructors, the code/v4/advanced tool registrations,
        # ~50 feature-module imports, cassette setup, session.start, env
        # probing, connection pre-warm — about 4s on a warm log) now runs
        # once, on first use, via _ensure_session(). All features still
        # work when invoked.
        self._session_started = False
        # cassette is assigned for real in _ensure_session(); None keeps
        # the attribute present (and falsy) until then.
        self.cassette = None

        self.autonomy = 3
        # live status pipe for the current turn (set in run_turn) — lets
        # long-running tools stream progress into the UI border
        self._turn_status: Callable[[str], None] | None = None
        # Focus Mode (deep work): distance history drives stall detection
        self._focus_history: list[float] = []
        self.notifier = Notifier(self.log)
        self._notify_seq = self.log.head()
        self.health = {"model_errors": {}, "failovers": 0}
        self._turn_start_seq = 0
        self._compact_digests: list[str] = []  # knowledge kept on compaction
        self._error_counts: dict[str, int] = {}
        self._file_hashes: dict[str, list[str]] = {}  # oscillation history
        # SPEED: cached tool schemas — rebuilt only when the registry size
        # changes (was re-serialized before every model call)
        self._schemas_cache: list[dict] | None = None
        self._schemas_tool_count = -1
        # PERF (tool bloat): the per-turn tool filter is keyed on the
        # active keyword groups; _turn_user_text carries the current
        # turn's message so _tool_schemas() can filter without a
        # signature change.
        self._schemas_filter_key = ""
        self._turn_user_text = ""
        # SPEED: compaction gate — the full token re-estimate only runs
        # after ~15% conversation growth, not before every model call
        self._compact_check_chars = 0
        # Worker 5/20 (flight prewarm): overlaps the provider streaming
        # wait with next-iteration prep — compact analysis, prompt
        # skeleton, read-only env probes — on a background thread.
        from .prewarm import FlightPrewarmer
        self._flight_prewarmer = FlightPrewarmer()

    def _ensure_session(self) -> None:
        """Run the deferred (expensive) half of __init__ exactly once.

        Worker 8/20 — startup time. Called at the top of run_turn() and
        _tool_schemas(), so every model-invoking path is covered. Cheap
        after the first call.
        """
        if self._session_started:
            return
        self._session_started = True
        _ensure_submodules()
        self._register_code_tools()
        self._register_v4_tools()
        self._register_advanced_tools()
        self._register_persisted_skills()
        self._register_feature_modules()
        self._register_subagent_tools()
        # cassette: record/replay model calls (FULLAGENT_CASSETTE=path,
        # FULLAGENT_CASSETTE_MODE=record|replay|off)
        cassette_path = os.environ.get("FULLAGENT_CASSETTE")
        cassette_mode = os.environ.get("FULLAGENT_CASSETTE_MODE", "off")
        self.cassette = (Cassette(Path(cassette_path), cassette_mode)
                         if cassette_path else None)

        self.log.append("session.start", {"session_id": self.session_id,
                                          "model": self.cfg.model_id,
                                          "effort": self.cfg.effort},
                        actor="system")
        self.forge.probe()  # PERCEIVE: stamp the environment
        # SPEED: pre-warm the TCP+TLS connection to the provider in the
        # background — the first model call hits a warm socket, not a cold
        # handshake (saves 200-800ms on the first turn)
        from .client import prewarm_connection
        prewarm_connection(self.provider)

    def _register_feature_modules(self) -> None:
        """Register the Claude Code-parity feature modules.

        Each module exposes register(agent). Wrapped individually so one
        broken optional module can never kill agent startup.
        """
        for mod_name in (
            "todos", "multiedit",
            # Deep reliability round (512s-incident fixes): read-guard must
            # wrap MultiEdit, so it registers right after "multiedit".
            "readguard",
            "bgsh", "planmode", "codeinit",
            "review", "notebook", "permissions", "outstyle",
            # Worker 8/20: mtime+size file change detection (512s incident)
            "filestat",
            "webfetch", "checkpoints", "hooks",
            # Round 2 advanced features
            "mcp", "skills", "doctor", "statuscmd",
            "permrules", "configcmd", "images", "export", "vimmode",
            "thinking", "undocmd", "diffpreview", "costtrack",
            "jsonout", "ctxmeter", "prreview", "smartctx",
            "termsetup", "plugins", "supervise", "turnresume", "cronsched", "watch", "sshops", "streamlog", "voicein",
            # browser automation (real Playwright, graceful degradation)
            "browserauto",
            # multi-session: SessionNew / SessionList / SessionSwitch
            "multisess",
            # Backup / restore (stdlib tarfile)
            "backup",
            # Sandboxed execution (bwrap → unshare → rlimit chain)
            "sandbox",
            "smartretry",
            "progress",
            "dockerops",
            # Persistent background jobs (SQLite queue + crash recovery)
            "jobs",
            "dbtools", "emailops",
            # Git automation (real git CLI via subprocess)
            "gitops",
            # Real desktop + webhook notifications (notify-send, optional hook URL)
            "notify",
            # Incoming HTTP webhooks (127.0.0.1 only, daemon thread)
            "webhook",
            # Deep reliability round (512s-incident fixes)
            "fuzzymatch",   # no-op register; used by the edit fallback chain
            "editbackup",   # auto-snapshot before risky edits + EditRollback
            "retrycmd",     # /retry conversation repair
            "enforcex.sovereign",  # Sovereign EnforceX v3 system & subagent suite
        ):
            try:
                mod = __import__(f"fullagent.{mod_name}",
                                 fromlist=["register"])
                mod.register(self)
            except Exception as e:  # noqa: BLE001 — optional feature
                try:
                    self.log.append("feature.register_failed",
                                    {"module": mod_name,
                                     "error": str(e)[:200]},
                                    actor="system")
                except Exception:
                    pass
        # NOTE (Worker 8/20): cassette setup, session.start, forge.probe()
        # and connection pre-warming used to live here; they moved to
        # _ensure_session(), which runs them once on first use.

    # -- model / effort ----------------------------------------------------

    @property
    def model(self) -> Model:
        _resolve_lazy("model_by_id", "config")
        m = (model_by_id(self.cfg.model_id)
             or model_by_id(config.DEFAULT_MODEL_ID)
             or config.MODELS[0])
        return m

    @property
    def provider(self) -> Provider:
        _resolve_lazy("PROVIDERS")
        return PROVIDERS[self.model.provider]

    @property
    def effort(self) -> Effort:
        _resolve_lazy("config")
        e = (config.effort_by_key(self.cfg.effort)
             or config.effort_by_key(config.DEFAULT_EFFORT)
             or config.EFFORTS[0])
        return e

    def _base_prompt(self) -> str:
        """The system prompt for this session, served from the Mastermind
        vault (hash-sealed copy of systemprompt.py). Falls back to the
        module source if the vault somehow lacks the name, so the model is
        never promptless."""
        _resolve_lazy("systemprompt")
        name = self.cfg.prompt
        sealed = self.mastermind.vault.get(name)
        return sealed if sealed is not None else systemprompt.get(name)

    def _reseat_system_prompt(self,
                              sections: dict[str, str] | None = None
                              ) -> None:
        """Route the conversation's system prompt through the Mastermind
        gate — the single door to the model. Guarantees messages[0] carries
        the sealed prompt, composes any live context sections beneath it,
        and seals a prompt.dispatch lineage event."""
        self.messages, _ = self.mastermind.gate.dispatch(
            self.cfg.prompt, self.messages, sections=sections)
        # SOVEREIGN APEX (v5): sentinel per-turn nudge (P2) when the
        # master prompt is active. Engine is cached; any failure here is
        # chrome, never a crash path.
        try:
            if self.cfg.prompt == "master" and not self.cfg.extra.get(
                    "apex_off"):
                eng = self._apex_engine()
                if eng is not None:
                    self.messages = eng.turn_messages(
                        self.messages, turn=len(self.turns))
        except Exception:
            pass

    def _apex_engine(self):
        """Lazily built ApexEngine (v5 enforcement orchestrator) for the
        current master prompt + model. Rebuilt when the prompt or model
        changes (the engine detects doc changes itself on rebuild)."""
        eng = getattr(self, "_apex_cache", None)
        try:
            doc = self.mastermind.gate.vault.resolve(self.cfg.prompt)
            model_label = self.cfg.model_id
            if eng is None or eng.model_label != model_label \
                    or eng.doc_text != getattr(eng, "_src", doc):
                from .apex import ApexEngine
                eng = ApexEngine.build(doc, model_label)
                eng._src = doc
                self._apex_cache = eng
            return eng
        except Exception:
            return None

    def state(self):
        """Live projection of the event log (cost, goal, dead-ends, …)."""
        _resolve_lazy("fold")
        return fold(self.log)

    # -- conversation ------------------------------------------------------

    def reset(self) -> None:
        self.messages = []
        self._turn_user_text = ""
        self._reseat_system_prompt()
        self.turns = []
        # focus stall/done detection reads _focus_history — stale ticks
        # from a previous focus session (e.g. after /focus off mid-run)
        # would misfire in the next one, so a reset starts it empty
        # (resume_session and focus.stop already clear it on their paths)
        self._focus_history.clear()
        self.session_id = uuid.uuid4().hex[:8]
        self.log.session = self.session_id
        self.log.append("session.start", {"session_id": self.session_id,
                                          "model": self.cfg.model_id,
                                          "effort": self.cfg.effort},
                        actor="system")

    def _context_sections(self,
                          route: RouteDecision | None = None,
                          query: str = ""
                          ) -> dict[str, str]:
        """Live context sections composed beneath the sealed prompt by the
        Mastermind gate (constitution, project, goal, web, memory,
        compacted). The framing is the composer's job — bodies here are
        plain content only.

        SPEED: each section is cached with a head-seq watermark — the section
        is only recomputed when the log has grown past the cached head. This
        eliminates redundant folds on multi-tool turns (the sections don't
        change between tool iterations within one turn)."""
        sections: dict[str, str] = {}
        head = self.log.head()

        # constitution — cached until the log grows (it's a file read)
        if not hasattr(self, "_const_cache") or self._const_cache_head != head:
            self._const_cache = self.oracle.read_constitution().strip()
            self._const_cache_head = head
        if self._const_cache:
            sections["constitution"] = self._const_cache

        # SOVEREIGN ENFORCEMENT (enforce.py): per-turn directive anchor —
        # the freshest-attention reminder that the contract + rule ledger
        # are binding for THIS reply. ~200 chars, never a section if the
        # prompt isn't sovereign-sealed (main prompt stays untouched).
        try:
            from .mastermind import _MASTER_SOVEREIGN as _sov
        except Exception:
            _sov = False
        if _sov:
            try:
                from .enforce import anchor as _anchor
                # rotating paraphrase: deterministic per turn (stable under
                # retries) so the anchor never becomes invisible boilerplate
                sections.setdefault("anchor", _anchor(len(self.turns)))
            except Exception:
                pass

        # project instructions — real AGENTS.md files (global → cwd).
        # instructions.py caches by mtime/size, so a multi-call turn does
        # not re-read disk pages it already has; a live edit is still seen
        # on the next call.
        try:
            from . import instructions as _instructions
            _project = _instructions.collect()
        except Exception:
            _project = ""
        if _project:
            sections["project"] = _project

        # goal — cached until a goal event is sealed
        if not hasattr(self, "_goal_cache") or self._goal_cache_head != head:
            self._goal_cache = self.goal.status()
            self._goal_cache_head = head
        goal = self._goal_cache
        if goal.active:
            sections["goal"] = (self.goal.format() +
                                "\nEvery action must serve an open clause. "
                                "When a clause's predicate genuinely "
                                "passes, say 'PROVEN: <clause id>' — the "
                                "kernel verifies it, never trust "
                                "self-declared success.")
        if route is not None and route.use_web:
            sections["web"] = ("This request needs live, up-to-the-minute "
                               "data. Use web_search (and web_fetch for "
                               "details) to get CURRENT facts — never "
                               "answer from stale knowledge. Quote the "
                               "retrieval time and sources.")

        # memory — cached until a memory/brain event is sealed
        if not hasattr(self, "_mem_cache") or self._mem_cache_head != head:
            mem = self.memory.context_block()
            if query:
                recall = self.semantic.recall_block(query, k=3)
                if recall:
                    mem = (mem + "\n\n" + recall) if mem else recall
            if query:
                brain_block = self.brain.context_block(query, k=3)
                if brain_block:
                    mem = (mem + "\n\n" + brain_block) if mem else brain_block
            self._mem_cache = mem
            self._mem_cache_head = head
        mem = self._mem_cache
        if mem:
            sections["memory"] = mem
        if self._compact_digests:
            sections["compacted"] = (
                "COMPACTED HISTORY — knowledge preserved from turns that "
                "were compressed to fit the context window:\n- "
                + "\n- ".join(self._compact_digests[-12:]))
        # Feature: output style instruction beneath the sealed prompt
        try:
            from .outstyle import get_style, style_instruction
            sections["output_style"] = (
                f"Output style ({get_style(self)}): "
                f"{style_instruction(self)}")
        except Exception:
            pass
        return sections

    def _tool_schemas(self) -> list[dict] | None:
        # Worker 8/20: the full tool registry (code/v4/advanced/feature
        # tools) is registered on first use — schemas need it complete.
        self._ensure_session()
        if not self.model.supports_tools:
            return None
        # PERF (tool bloat): schemas are filtered per turn (core tools
        # always + keyword-matched advanced tools + recently used) and
        # compressed (trimmed descriptions, no example blobs). The full
        # registry stays intact — the executor still resolves every
        # tool by name from self.tools.
        try:
            from . import toolfilter as _tf
            compress = _tf.compression_enabled()
            do_filter = _tf.filter_enabled()
        except Exception:
            compress, do_filter = False, False
            _tf = None
        n = len(self.tools)
        # Filter only when we know the current turn's user text; any
        # other path (non-turn callers) gets the
        # full set — fail-open, never silently tool-less.
        user_text = getattr(self, "_turn_user_text", "") or ""
        # Cache key covers both the filter selection AND the progressive-
        # disclosure rank order (worker 12/20): rank is a pure function of
        # the hint, so a different hint must not reuse a stale ordering.
        if _tf is not None and user_text:
            filt_key = ((_tf.filter_signature(user_text)
                         if do_filter else "all")
                        + "|r=" + _tf.rank_signature(user_text))
        else:
            filt_key = "all"
        if (self._schemas_cache is not None
                and self._schemas_tool_count == n
                and self._schemas_filter_key == filt_key):
            return self._schemas_cache
        if do_filter and user_text:
            names = _tf.select_tools(
                self.tools, user_text, recent=self._recent_tool_names())
            tools = [self.tools[tname] for tname in names]
        else:
            tools = list(self.tools.values())
        # Worker 12/20: progressive disclosure — order the schemas by
        # task relevance so the model sees the most relevant tools
        # first (full registry order is arbitrary; schema order biases
        # tool choice). Stable: unknown tools keep relative order.
        if _tf is not None:
            try:
                ranked = _tf.rank_tools(user_text, [t.name for t in tools])
                tools = [self.tools[tname] for tname in ranked]
            except Exception:
                pass
        if compress:
            schemas = [_tf.compressed_schema(t) for t in tools]
        else:
            schemas = [t.openai_schema() for t in tools]
        self._schemas_cache = schemas
        self._schemas_tool_count = n
        self._schemas_filter_key = filt_key
        return schemas

    def _recent_tool_names(self) -> tuple[str, ...]:
        """Tool names used in the last 2 turns (filter continuity)."""
        try:
            return tuple(
                t.name for tn in self.turns[-2:] for t in tn.tools)
        except Exception:
            return ()

    def _seal_pending_tool_calls(self, note: str) -> None:
        """Pairing repair: every assistant tool_call must be followed by a
        tool response with the same id. When a turn dies mid-tool-batch
        (Esc, Ctrl+C, unexpected error), the calls that never ran would
        leave the conversation permanently unbalanced — and every later
        model call would fail with a tool/response pairing error. Seal
        each unanswered call with a synthetic response instead, so the
        session survives the interruption."""
        answered = {m.get("tool_call_id") for m in self.messages
                    if m.get("role") == "tool" and m.get("tool_call_id")}
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    cid = tc.get("id")
                    if cid and cid not in answered:
                        self.messages.append({
                            "role": "tool", "tool_call_id": cid,
                            "content": note})
                        answered.add(cid)

    def run_turn(self, user_text: str,
                 on_token: Callable[[str], None],
                 on_reasoning: Callable[[str], None],
                 on_tool_call: Callable[[ToolEvent], None],
                 on_tool_update: Callable[[ToolEvent], None],
                 on_status: Callable[[str], None],
                 approve: Callable[[Tool, dict], bool],
                 should_cancel: Callable[[], bool] | None = None,
                  on_route: Callable[[RouteDecision], None] | None = None,
                  on_tool_output: Callable[[str, str], None] | None = None,
                  on_tool_args: Callable[[str, str], None] | None = None,
                  ) -> Turn:
        """Run one user turn through the full agent loop."""
        # LONG-RUN helpers: optional config.json extra overrides for the
        # adaptive tool-call cap ("tool_call_cap", "turn_time_budget", …).
        def _int_extra(key: str) -> int | None:
            try:
                v = self.cfg.extra.get(key)
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _float_extra(key: str) -> float | None:
            try:
                v = self.cfg.extra.get(key)
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        # Worker 8/20: deferred session start — registers the full tool
        # set, feature modules, cassette, session.start, env probe.
        self._ensure_session()
        # Continue-from-summary (continueresume): "continue" after a capped
        # turn must NOT re-send the full capped-turn history (providers
        # answer the bloated replay with invalid_request_error). Reseed
        # the model-visible history as [system, original request, compact
        # summary]; the user's "continue" message is appended as usual.
        try:
            from .continueresume import seed_continue_history
            seed_continue_history(self, user_text)
        except Exception:  # noqa: BLE001 — resume is a nicety, not a crash path
            pass
        # Feature: per-turn model override (/model <name> --once)
        try:
            from .modelpick import get_effective_model, clear_turn_model
            _eff_model = get_effective_model(self)
            clear_turn_model(self)
        except Exception:
            _eff_model = self.model
        turn = Turn(user_text=user_text, model_id=_eff_model.id,
                    effort=self.cfg.effort)
        started = time.time()
        # PERF (tool bloat): remember this turn's message so
        # _tool_schemas() can send only the relevant tools.
        self._turn_user_text = user_text
        self._turn_start_seq = self.log.head()
        # tooldedup hook — fresh dedup cache each turn (dedup is turn-scoped).
        try:
            from .tooldedup import reset as _dedup_reset
            _dedup_reset(self)
        except Exception:  # noqa: BLE001
            pass
        # Fresh turn: clear any stale cancellation from a previous turn
        self._cancel_flag.clear()
        # Unify the two cancellation doors: the caller-supplied
        # should_cancel callback and the Esc/Ctrl+C flag are the same
        # signal. When the callback fires, latch the flag too — shell
        # tools (run_command/live_shell) only watch the flag, so without
        # this a callback-only caller could never stop a running command.
        _outer_cancel = should_cancel

        def _cancel_requested() -> bool:
            if self._cancel_flag.is_set():
                return True
            if _outer_cancel is not None and _outer_cancel():
                self._cancel_flag.set()
                return True
            return False
        should_cancel = _cancel_requested

        # AUTOPILOT: the agent decides for itself which powers this turn
        # needs — goal mode, real-time web — and enables
        # them on its own. Every decision is logged and surfaced.
        route = self.autopilot.route(user_text,
                                     self.goal.status().active,
                                     self.autonomy)
        if route.active and on_route:
            on_route(route)
        if route.suggest_goal:
            self._autopilot_goal(route)

        # refresh the system document through the Mastermind gate — the
        # sealed prompt is guaranteed at position 0, live context sections
        # (constitution + goal + web + memory) are composed beneath it.
        self._reseat_system_prompt(
            sections=self._context_sections(route, query=user_text))

        # v3: speculate on the read-only calls this turn will likely make —
        # they run in the background while the model thinks (zero-latency
        # feel). Only whitelisted read-only tools are ever prefetched.
        recent_tools = [{"name": t.name, "args": t.args}
                        for tn in self.turns[-2:] for t in tn.tools]
        self.speculator.speculate(user_text, recent_tools)

        # Guard: a single giant paste (a whole file, a huge log) must never
        # blow the context window by itself — cap it so the turn can still
        # run. The full text stays in the event log; the model sees a
        # truncated copy with instructions to re-read from disk if needed.
        full_text = user_text
        user_text = self._cap_user_message(user_text)

        # NOTE: an earlier build injected a "[MANDATORY COMPLIANCE]" *user*
        # message before every turn for long prompts. That is the wrong
        # channel: it competes with the real system prompt, reads as user
        # pressure, and models increasingly discount it. The strongest
        # adherence levers are structural instead — a concise operating
        # contract at position 0, live context framed beneath it, and the
        # wire gate above re-asserting the sealed prompt on every request.
        self.messages.append({"role": "user", "content": user_text})
        user_ev = self.log.append("user.message",
                                  {"text": full_text,
                                   "session": self.session_id},
                                  actor="human", provenance="user")

        iterations = 0
        # PERF: adaptive cap on tool calls per turn + no-progress detection.
        # 200 LLM round-trips at ~2.5s each was the 500s+ hang.
        tool_calls_this_turn = 0
        # Worker 8/20: adaptive cap — fast calls get a big budget, slow
        # calls a smaller one. v4 LONG-RUN: user-tunable via config.json
        # extra keys "tool_call_cap" / "turn_time_budget" (seconds).
        _turn_cap = AdaptiveTurnCap(
            base_cap=_int_extra("tool_call_cap"),
            fast_cap=_int_extra("tool_call_cap_fast"),
            slow_cap=_int_extra("tool_call_cap_slow"),
            budget_s=_float_extra("turn_time_budget"),
            slow_budget_s=_float_extra("turn_time_budget_slow"),
        )
        _last_tool_sig = None
        _stall_count = 0
        turn_stopped = False
        _retry_loop_det = _RetryLoopDetector()  # loopdetect: per-turn detector
        _exit_tracker = _EarlyExitTracker()  # earlyexit: per-turn novelty
        self._turn_status = on_status
        try:
            while iterations < config.MAX_TOOL_ITERATIONS:
                iterations += 1
                # CANCEL CHECK — Esc/Ctrl+C sets the flag; check it at the
                # top of every iteration so the turn stops IMMEDIATELY
                # between tool calls, not just during streaming.
                if should_cancel is not None and should_cancel():
                    raise TurnCancelled()
                # I8: budget governor — a breach PAUSES the turn
                if not self.budget_gov.enforce():
                    evs = fold(self.log).budget_events
                    reason = evs[-1]["reason"] if evs else "budget exceeded"
                    turn.error = f"PAUSED — {reason}. Extend the budget " \
                                 f"with /budget (e.g. /budget steps 500 " \
                                 f"or /budget reset) or stop."
                    break
                on_status("thinking")
                # keep the context under the window — compact stale turns
                # before asking the model (I9: input budget governor)
                # Worker 5/20 (flight prewarm): consume the bundle
                # prewarmed on a background thread during the previous
                # API wait — compact analysis, prompt skeleton, env
                # probes. None on the first iteration or whenever the
                # snapshot was invalidated; the normal path then runs
                # unchanged.
                _prewarmer = getattr(self, "_flight_prewarmer", None)
                try:
                    _flight = (_prewarmer.consume(self)
                               if _prewarmer is not None else None)
                except Exception:
                    _flight = None
                try:
                    self._maybe_compact(_prewarmed=_flight)
                except Exception:
                    pass
                result = self._complete(on_token, on_reasoning, on_status,
                                        should_cancel,
                                        on_tool_args=on_tool_args,
                                        _prewarmed=_flight)

                if result.reasoning:
                    turn.reasoning += result.reasoning

                # every completion (tool iterations included) burns tokens —
                # record usage for each call so the budget governor and the
                # ledger see the real spend, not just the final reply
                self._emit_cost(result.usage)

                if result.tool_calls:
                    self.messages.append(assistant_message(
                        result.content, result.tool_calls, result.reasoning))
                    if result.content:
                        turn.assistant_text += result.content
                    # Worker 11/20 — parallel tool execution: parse the whole
                    # block first (model-emitted order preserved), then
                    # dispatch it in order — maximal runs of independent
                    # read-only calls run concurrently in a capped pool,
                    # everything else stays sequential. _finish_one replays
                    # all per-call bookkeeping in order, so the transcript
                    # is identical to the old sequential loop.
                    pending: list[tuple[dict, ToolEvent]] = []
                    from .client import normalize_tool_call as _norm_tc
                    for raw_tc in result.tool_calls:
                        # One canonical shape everywhere: normalize every
                        # entry and skip unrecoverable ones. A malformed
                        # entry must never reach tc["function"] as
                        # AttributeError: 'str' object has no attribute
                        # 'get' (the reported malformed-call crash).
                        tc = _norm_tc(raw_tc)
                        if tc is None:
                            continue
                        fn = tc["function"]
                        name = fn["name"]
                        args = parse_tool_arguments(fn["arguments"])
                        ev = ToolEvent(name=name, args=args)
                        turn.tools.append(ev)
                        on_tool_call(ev)
                        pending.append((tc, ev))

                    def _finish_one(tc, ev) -> bool:
                        """Per-call bookkeeping in model-emitted order.

                        Returns True when a guardrail (adaptive cap,
                        retry-loop detector, stall guard) stopped the turn
                        — dispatch_block then stops the walk early instead
                        of executing the rest of the block. A cap that fires
                        on call 3 of 8 must not let calls 4-8 run anyway."""
                        nonlocal tool_calls_this_turn, _last_tool_sig
                        nonlocal _stall_count, turn_stopped
                        fn = tc["function"]
                        name = fn.get("name", "")
                        on_tool_update(ev)
                        self.log.append(
                            "tool.result",
                            {"name": name, "status": ev.status,
                             "duration": round(ev.duration, 3),
                             "preview": ev.result[:300]},
                            actor="system", provenance="tool_output",
                            correlation_id=ev.clause_id)
                        _tool_content = ev.result
                        # SOVEREIGN SHIELD (v5): untrusted tool output is
                        # scanned, defanged and fenced before it reaches
                        # the model — injection attempts arrive as tagged
                        # data, never as instructions.
                        try:
                            from .shield import shield_tool_output
                            _sh = shield_tool_output(
                                _tool_content, f"tool:{name}")
                            _tool_content = _sh["content"]
                            if _sh["quarantined"]:
                                self.log.append(
                                    "shield.quarantined",
                                    {"tool": name, "risk": _sh["risk"],
                                     "kinds": _sh["kinds"]},
                                    actor="shield")
                        except Exception:
                            pass
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": _tool_content,
                        })
                        # loopdetect hook: the same (tool, args, error) triple
                        # failing repeatedly means the model is stuck retrying
                        if ev.status == "error":
                            _loop_msg = _retry_loop_det.note(
                                name, ev.args, normalize_error(ev.result))
                            if _loop_msg:
                                turn_stopped = True
                                turn.error = _loop_msg
                        # retryhint hook — inject recovery alternatives on
                        # repeated same-tool failures (see retryhint.py)
                        from .retryhint import note_failure, note_success
                        if ev.status == "error":
                            _rh_hint = note_failure(name, ev.result)
                            if _rh_hint:
                                # Runtime guidance is NOT policy. Keep it in
                                # the user channel (tagged as subordinate)
                                # so it can never become a second system
                                # prompt that shadows systemprompt.py.
                                self.messages.append({
                                    "role": "user",
                                    "content": (
                                        "[RUNTIME NOTE — harness guidance, "
                                        "subordinate to the system prompt]\n"
                                        + _rh_hint),
                                })
                        else:
                            note_success(name)
                        # §38.3 goal kernel tick after every action
                        self._goal_tick()
                        # CANCEL CHECK between tool calls — Esc stops
                        # the turn immediately, not after the next model call
                        if should_cancel is not None and should_cancel():
                            raise TurnCancelled()
                        # PERF (worker 8/20): adaptive tool-call cap —
                        # rolling avg of per-call wall time picks the tier
                        # (fast <5s: up to 40 calls; slow >15s: 12 calls /
                        # 120s tool time; else 25 calls / 180s). The stop
                        # message says WHY (call count vs time budget) and
                        # invites "continue". Scheduler-yield tools
                        # (TaskCollect/TaskOutput) are special: while
                        # background workers are still active their wait is
                        # not tool execution and must not kill the parent
                        # turn before its subagents can report back. Once
                        # everything has settled, the collect call counts
                        # as a normal (zero-time) call so polling loops are
                        # still bounded.
                        _wait_tool = bool(getattr(
                            self.tools.get(name), "background_wait", False))
                        if _wait_tool:
                            try:
                                _bg_active = bool(
                                    getattr(self, "subagents", None)
                                    is not None
                                    and self.subagents.active_count() > 0)
                            except Exception:
                                _bg_active = False
                            _cap_msg = _turn_cap.note(
                                0.0,
                                waiting_on_background=_bg_active)
                        else:
                            _cap_msg = _turn_cap.note(ev.duration)
                        tool_calls_this_turn += 1
                        if _cap_msg is not None:
                            turn_stopped = True
                            turn.error = _cap_msg
                        else:
                            try:
                                _sig = (name,
                                        hashlib.sha256(ev.result.encode(
                                            "utf-8", "replace")).hexdigest())
                            except Exception:
                                _sig = None
                            if _sig is not None and _sig == _last_tool_sig:
                                _stall_count += 1
                                # Only hard-stop on repeated ERRORS, not successful
                                # results. A tool legitimately returning the same
                                # success 3x (e.g. status checks) is not a loop.
                                # But the same ERROR 3x means the model is stuck.
                                _is_error = ev.result.strip().upper().startswith(
                                    ("ERROR", "FAILED", "EXCEPTION"))
                                if _stall_count >= config.NO_PROGRESS_STALL_LIMIT and _is_error:
                                    turn_stopped = True
                                    turn.error = (
                                        "No progress detected — the same tool "
                                        "call failed with an identical error 3 "
                                        "times in a row. Stopped to avoid "
                                        "looping; rephrase or ask me to "
                                        "continue.")
                            else:
                                # Count the current call as sighting #1, so
                                # the guard fires on the 3rd identical
                                # error in a row — matching its own message
                                # and the retry-loop detector (both say 3).
                                _stall_count = 1
                            _last_tool_sig = _sig
                        # earlyexit hook (Worker 10/20): cheap per-result
                        # novelty check. Never stops the turn — when the
                        # last few calls added nothing new after a
                        # substantial run, inject a soft nudge so a model
                        # that keeps "verifying" compulsively gets a cheap
                        # chance to finish instead of burning all 25 calls.
                        # At most one nudge per turn; failures are skipped
                        # (loopdetect owns those).
                        try:
                            _ee_hint = _exit_tracker.note(
                                name, ev.args, ev.result, ev.status)
                            if _ee_hint:
                                self.messages.append({
                                    "role": "user",
                                    "content": (
                                        "[RUNTIME NOTE — harness guidance, "
                                        "subordinate to the system prompt]\n"
                                        + _ee_hint),
                                })
                        except Exception:
                            pass  # novelty is insight, never a crash path
                        # dispatch_block stops the walk early when a
                        # guardrail fired — the remaining calls in this
                        # block must not execute.
                        return turn_stopped

                    # Worker 9/20 — live per-call progress: each tool call
                    # emits one cheap on_status line
                    # ("progress:12:25:45:read_file:aegisscan.py") right
                    # before it executes, so a long turn never looks hung.
                    # The TUI renders it in the bottom status line and
                    # headless mode prints it via on_status. It never
                    # touches the event log (no spam), and indices are
                    # pre-assigned from the finished-call count so
                    # parallel batches report consecutive numbers.
                    from .callprogress import wrap_execute as _wrap_progress
                    _exec_progress = _wrap_progress(
                        self._execute_tool, on_status, started,
                        config.MAX_TOOL_CALLS_PER_TURN,
                        tool_calls_this_turn,
                        [ev for _, ev in pending])
                    dispatch_block(
                        _exec_progress, _finish_one, pending,
                        approve=approve, on_status=on_status,
                        causation_id=user_ev.id,
                        on_tool_output=on_tool_output,
                        hooks_enabled=getattr(self, "hooks_enabled", False))
                    if turn_stopped:
                        # A guardrail (adaptive cap, retry-loop detector,
                        # stall guard) fired mid-block: the rest of the
                        # block never ran.
                        # (1) Parallel-batch tail: calls that EXECUTED but
                        # whose finish walk was cut short never got their
                        # tool response appended — record the real results
                        # so the transcript stays faithful.
                        answered = {
                            m.get("tool_call_id") for m in self.messages
                            if m.get("role") == "tool"
                            and m.get("tool_call_id")}
                        for _tc, _ev in pending:
                            _cid = _tc.get("id")
                            if (_cid and _cid not in answered
                                    and _ev.status in ("done", "error")):
                                self.messages.append(
                                    {"role": "tool",
                                     "tool_call_id": _cid,
                                     "content": _ev.result})
                                answered.add(_cid)
                        # (2) Calls that never executed: mark them blocked
                        # so the UI doesn't show them spinning forever,
                        # then seal their pairing — an assistant tool_call
                        # with no tool response would poison every later
                        # model call with a pairing/invalid_request error.
                        for _tc, _ev in pending:
                            if _ev.status == "running":
                                _ev.status = "blocked"
                                _ev.result = (
                                    "SKIPPED — turn stopped early: "
                                    f"{turn.error}")
                                try:
                                    on_tool_update(_ev)
                                except Exception:
                                    pass
                        self._seal_pending_tool_calls(
                            "SKIPPED — the turn was stopped early by a "
                            f"guardrail ({turn.error}); do not retry "
                            "automatically.")
                        break
                    continue

                # plain assistant reply — done
                turn.assistant_text += result.content
                turn.usage = result.usage
                self.messages.append(
                    {"role": "assistant", "content": result.content})
                self.log.append("assistant.message",
                                {"text": result.content,
                                 "session": self.session_id},
                                actor="sovereign", provenance="model",
                                causation_id=user_ev.id)
                self._detect_goal_clauses(result.content)
                break
            else:
                turn.error = (
                    f"stopped after {config.MAX_TOOL_ITERATIONS} tool "
                    f"iterations — the task is complex. Say 'continue' "
                    f"to resume from where it left off, or break the "
                    f"task into smaller steps."
                )
        except APIError as e:
            # the status pipe must never outlive the turn — a stale
            # callback would receive the NEXT turn's
            # status lines after this one died
            self._turn_status = None
            turn.error = str(e)
            # Seal any assistant tool_call left without a response (e.g.
            # a callback raising mid-dispatch): an orphaned tool_call
            # poisons every later model call with a pairing /
            # invalid_request error. No-op when nothing is pending.
            self._seal_pending_tool_calls(
                f"tool call never executed — the turn died with "
                f"APIError: {e}")
            # only drop this turn's user prompt when nothing else was
            # appended after it (failed first complete). After tool calls,
            # popping would orphan an assistant tool_call and the next
            # request dies with a thinking-history / pairing error.
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
            self.log.append("turn.error", {"error": str(e)})
        except TurnCancelled:
            self._turn_status = None
            turn.error = "cancelled"
            # the turn may have died between tool calls, leaving
            # tool_calls with no responses — seal them so the session
            # stays usable
            self._seal_pending_tool_calls(
                "CANCELLED by user (Esc/Ctrl+C) before execution — do "
                "not retry automatically; ask before re-running.")
            self.messages.append(
                {"role": "assistant",
                 "content": turn.assistant_text or "(cancelled by user)"})
            self.log.append("turn.cancelled", {})
        except KeyboardInterrupt:
            self._turn_status = None
            turn.error = "interrupted"
            self._seal_pending_tool_calls(
                "INTERRUPTED by user before execution — do not retry "
                "automatically; ask before re-running.")
            self.messages.append(
                {"role": "assistant",
                 "content": turn.assistant_text or "(interrupted)"})
            self.log.append("turn.cancelled", {})
        except Exception as e:  # noqa: BLE001 — a turn must never die
            # silently and take the session down with it: record the
            # failure on the turn and keep going.
            self._turn_status = None
            turn.error = f"{type(e).__name__}: {e}"
            self._seal_pending_tool_calls(
                f"tool call never executed — the turn died with "
                f"{type(e).__name__}: {e}")
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
            try:
                self.log.append("turn.error",
                                {"error": f"{type(e).__name__}: {e}"[:500]})
            except Exception:
                pass

        self._turn_status = None
        turn.duration = time.time() - started
        # Worker 5/20 (flight prewarm): drain any flight that was never
        # consumed (single-call turns never reach a next iteration) so
        # the external-edit early notice still fires. The prompt-prep
        # half of the bundle is simply discarded — there is no next
        # iteration to spend it on.
        try:
            _fp = getattr(self, "_flight_prewarmer", None)
            if _fp is not None:
                _fp.consume(self)
        except Exception:
            pass
        try:
            self._score_turn(turn)
        except Exception:
            pass  # scorecard is insight, never a crash path
        try:
            self._flush_notifications()
        except Exception:
            pass
        self.turns.append(turn)
        # bound memory: a Turn holds full tool outputs, and this list is
        # only live UI state — the event log is the durable record
        if len(self.turns) > 500:
            del self.turns[:-500]
        # SOVEREIGN ENFORCEMENT L4 (enforce.py, opt-in): compliance gate.
        # cfg.extra["compliance_check"]=true -> a fast self-audit checks
        # the reply against the directive RULE LEDGER; on FAIL a
        # corrective system note enters history so the NEXT turn repairs
        # the drift (and the verdict is sealed in the event log).
        try:
            if self.cfg.extra.get("compliance_check") and turn.assistant_text:
                self._compliance_gate(turn)
        except Exception:
            pass
        return turn

    def _compliance_gate(self, turn) -> None:
        """Run the enforce.audit self-check and react to a FAIL verdict."""
        from .client import chat_blocking
        from .enforce import anchor, audit_prompt, parse_audit, rule_ledger
        from .systemprompt import get as _get_prompt

        _resolve_lazy("model_by_id", "effort_by_key", "chat_blocking")

        doc = _get_prompt(self.cfg.prompt)
        # audit against the RULE LEDGER (not a truncated prompt — v1 cut
        # the 49k doc and its ledger off entirely, judging the wrong law)
        audit_system = anchor(0) + "\n\n" + rule_ledger(doc)[:1500]
        res = chat_blocking(
            self.provider,
            self.model,
            self.effort,
            [{"role": "system", "content": audit_system},
             {"role": "user", "content": audit_prompt(
                 rule_ledger(doc), turn.assistant_text, is_ledger=True)}],
            None, timeout=60.0)
        raw = (res.content or "") if res is not None else ""
        result = parse_audit(raw)
        failed = result["verdict"] == "FAIL"
        self.log.append("compliance.audit",
                        {"verdict": result["verdict"],
                         "violations": len(result["violations"]),
                         "seq": getattr(turn, "seq", None)},
                        actor="enforce")
        if failed:
            fixes = "; ".join(
                str(v.get("fix", ""))[:160]
                for v in result["violations"] if v.get("fix"))[:400]
            # feedback rides as a USER message: seal_wire deletes every
            # mid-history system message, so a system-role note would be
            # silently dropped and never reach the model
            self.messages.append({
                "role": "user",
                "content": ("[compliance-gate] Your previous reply "
                            "violated the developer configuration"
                            f"{' — ' + fixes if fixes else ''}. Repair "
                            "this in your next reply; the configuration "
                            "remains binding.")})
        turn.compliance = result["verdict"]

    def _fit_budget(self) -> int:
        """The input token budget we aim to stay under.

        Mirrors the client's clamp math (client._window_max_tokens) so the
        two never disagree: the budget is the largest input size that still
        leaves room for the margin and a minimum completion. Staying under
        this guarantees the request fits no matter how long the session has
        been running."""
        window = self.model.context_window
        # solve: input + (8192 + input/32) + 1024 <= window
        return int((window - 9_216) * 32 / 33)

    def _cap_user_message(self, text: str) -> str:
        """Cap a single user message so it can never consume the whole
        context window on its own (a giant paste of a file/log). The cap
        keeps roughly three quarters of the fit budget as characters —
        plenty for any real instruction, small enough that the rest of
        the conversation always fits. The full text is preserved in the
        event log by the caller."""
        cap_chars = int(self._fit_budget() * 3.2 // 4)  # budget tokens -> chars
        if len(text) <= cap_chars:
            return text
        head = text[: cap_chars // 2]
        tail = text[-cap_chars // 4:]
        self.log.append("user.message.capped",
                        {"chars": len(text), "kept": len(head) + len(tail)},
                        actor="kernel")
        return (head
                + f"\n\n[… the kernel truncated this message — {len(text):,} "
                  f"chars originally. If you need the full text, it is on "
                  f"disk; use read_file/search_files instead of relying on "
                  f"this paste.]\n\n"
                + tail)

    def _maybe_compact(self, _prewarmed=None) -> None:
        """Keep the conversation under the model's context window.

        Three escalating passes, run only when needed:
          1. truncate stale tool outputs to summaries,
          2. drop the oldest user→(assistant+tool results) turns as whole
             units so tool_call / tool-response pairing is never broken,
          3. last resort — truncate EVERY tool output and trim the oldest
              assistant messages.
        Only the model-visible context shrinks — the event log keeps
        everything, so nothing is ever lost.

        SPEED: the expensive token estimate (a JSON dump of the whole
        conversation) runs only when the conversation has measurably
        grown since the last check — the cheap char-count is the gate.

        Worker 5/20 (flight prewarm): when _prewarmed carries the
        compact analysis computed on a background thread during the
        previous API wait, the fit check reuses it incrementally
        (est(snapshot) + est(small appended delta)) instead of
        re-dumping the whole conversation."""
        schemas = self._tool_schemas()
        schema_tokens = (estimate_tokens(schemas, self.model.id)
                         if schemas else 0)
        budget = self._fit_budget()
        if _prewarmed is not None:
            try:
                from .prewarm import compact_fits, message_chars
                if compact_fits(self.messages, _prewarmed, schema_tokens,
                                budget, self.model.id):
                    # fits with margin — the same early return the normal
                    # path takes, minus the full-conversation JSON dump.
                    # Gate bookkeeping stays identical.
                    self._compact_check_chars = max(
                        _prewarmed.chars + message_chars(
                            self.messages[_prewarmed.snapshot_len:]), 1)
                    return
            except Exception:
                pass  # fall through to the normal path on any doubt
        chars = self._messages_chars()
        if self._compact_check_chars > 0 and \
                chars <= self._compact_check_chars * 1.15:
            return  # not yet grown enough to justify a full re-estimate
        fits = (estimate_tokens(self.messages, self.model.id)
                + schema_tokens <= budget)
        self._compact_check_chars = max(chars, 1)
        if fits:
            return
        dropped = self._compact_old_tools()
        while (estimate_tokens(self.messages, self.model.id)
               + schema_tokens > budget
               and len(self.messages) > 2):
            if not self._drop_oldest_turn():
                break
            dropped += 1
        # last resort: still over budget — truncate all tool outputs and
        # trim the oldest assistant messages until it fits
        if (estimate_tokens(self.messages, self.model.id)
                + schema_tokens > budget):
            self._compact_old_tools(keep=0)
            self._trim_oldest_assistant(budget - schema_tokens)
        if dropped:
            self._compact_check_chars = 0  # force re-estimate next time
            self.log.append("context.compacted",
                            {"messages": len(self.messages),
                             "units": dropped,
                             "digests": len(self._compact_digests),
                             "est_tokens": estimate_tokens(self.messages,
                                              self.model.id)},
                            actor="kernel")

    def _messages_chars(self) -> int:
        """Cheap conversation size proxy (no JSON dump): sum of content
        lengths plus tool-call argument sizes. Gate for the expensive
        token estimate."""
        n = 0
        for m in self.messages:
            c = m.get("content")
            if c:
                n += len(c)
            tcs = m.get("tool_calls")
            if tcs:
                for tc in tcs:
                    fn = tc.get("function") or {}
                    n += len(fn.get("name", "")) + len(
                        fn.get("arguments") or "")
            rc = m.get("reasoning_content") or m.get("reasoning")
            if rc:
                n += len(rc)
        return n

    def _trim_oldest_assistant(self, budget: int) -> None:
        """Shrink the oldest assistant messages (they are the least
        actionable once their tool results are gone) until under budget."""
        # Walk EVERY long assistant message, oldest first. (The old code
        # re-found the same message on every pass and returned after a
        # single trim, so over-budget context stayed over budget and the
        # overflow retry loop could never converge.)
        while estimate_tokens(self.messages, self.model.id) > budget:
            target = None
            for m in self.messages:
                content = m.get("content")
                if (m.get("role") == "assistant"
                        and isinstance(content, str)
                        and len(content) > 400):
                    target = m
                    break
            if target is None:
                return
            content = str(target["content"])
            target["content"] = (
                content[:300]
                + f"\n[… trimmed by the kernel — {len(content):,} "
                  f"chars originally]")

    def _emergency_compact(self) -> None:
        """Hard compaction used when a request was ALREADY rejected for
        exceeding the context window. More aggressive than _maybe_compact:
        truncates every tool output, then drops oldest turns until the
        conversation is at most a third of the window — leaving a wide
        margin so the retry is guaranteed to fit."""
        self._compact_old_tools(keep=0)
        target = self.model.context_window // 3
        while (estimate_tokens(self.messages, self.model.id) > target
               and len(self.messages) > 2):
            if not self._drop_oldest_turn():
                break
        self._trim_oldest_assistant(target)
        self.log.append("context.compacted",
                        {"messages": len(self.messages),
                         "est_tokens": estimate_tokens(self.messages,
                                              self.model.id),
                         "reason": "emergency — request rejected for "
                                   "context length"},
                        actor="kernel")

    def _overflow_shrink(self) -> bool:
        """Callback handed to the client (on_overflow): invoked when the
        backend rejects a request because the INPUT itself no longer fits
        the window. Shrinks the conversation and reports whether anything
        actually shrank — the client retries only when it did. This is
        what lets an arbitrarily long session on a huge project recover
        instead of dying with a context-length error."""
        before = estimate_tokens(self.messages, self.model.id)
        self._emergency_compact()
        after = estimate_tokens(self.messages, self.model.id)
        return after < before

    def _compact_old_tools(self, keep: int = 2) -> int:
        """Truncate stale tool results to summaries, keeping the newest
        `keep` tool responses verbatim."""
        tool_idx = [i for i, m in enumerate(self.messages)
                    if m.get("role") == "tool"]
        stale = tool_idx if keep <= 0 else tool_idx[:-keep]
        n = 0
        for i in stale:
            content = str(self.messages[i].get("content") or "")
            if len(content) > 500:
                self.messages[i]["content"] = (
                    content[:350]
                    + f"\n[… truncated by the kernel — {len(content):,} "
                      f"chars originally; re-read the file if you need it]")
                n += 1
        return n

    def _drop_oldest_turn(self) -> bool:
        """Remove the oldest user message plus everything up to the next
        user message (the assistant reply and its tool results)."""
        msgs = self.messages
        if msgs and msgs[0].get("role") == "system":
            start = 1
        else:
            start = 0
        end = None
        for j in range(start + 1, len(msgs)):
            if msgs[j].get("role") == "user":
                end = j
                break
        if end is None or end <= start:
            return False
        digest = self._digest_messages(msgs[start:end])
        if digest:
            self._compact_digests.append(digest)
            self._compact_digests = self._compact_digests[-20:]
        del msgs[start:end]
        return True

    def _goal_tick(self) -> None:
        """§38.3 kernel loop: measure distance, re-aim focus. Pure Python
        over projections — zero tokens."""
        if not self.goal.status().active:
            return
        self.goal.measure()
        self.goal.reaim()

    # -- autopilot actions ---------------------------------------------------

    def _autopilot_goal(self, route: RouteDecision) -> None:
        """Auto-draft and freeze a machine-checkable goal contract from the
        request. If the contract fails validation (it cannot be made
        checkable), the turn simply proceeds without one."""
        from .goal import GoalContractError
        try:
            self.goal.set_goal(route.goal_statement, route.goal_clauses)
        except GoalContractError:
            return

    def _emit_cost(self, usage: dict | None) -> None:
        # Delegates to the canonical extractor: a non-dict usage (string
        # from a sloppy provider) used to raise AttributeError on
        # usage.get() here. Nothing is sealed for zero-token turns.
        from .client import safe_token_counts
        tin, tout = safe_token_counts(usage)
        if not (tin or tout):
            return
        # configured models are free-tier; ledger still tracks tokens so a
        # price table can be dropped in later without schema changes
        usd = 0.0
        focus = self.goal.status().focus
        self.log.append("cost.incurred", {"usd": usd, "tokens_in": tin,
                                          "tokens_out": tout,
                                          "model": self.model.id},
                        correlation_id=focus)
        # costtrack module: per-turn spend for /cost
        _tracker = getattr(self, "cost_tracker", None)
        if _tracker is not None:
            try:
                _tracker.add_turn(self.model.id, tin, tout)
            except Exception:
                pass

    def _detect_goal_clauses(self, text: str) -> None:
        """The model may CLAIM 'PROVEN: <id>' — but the clause is only
        marked proven if its own predicate actually passes (§39.1: the
        binary proven flag is never set from model output alone)."""
        goal = self.goal.status()
        if not goal.active:
            return
        low = text.lower()
        for clause in goal.clauses:
            if clause.state in ("PROVEN", "WAIVED"):
                continue
            key = clause.id.lower()
            claimed = (f"proven: {key}" in low or f"done: {key}" in low
                       or f"completed: {key}" in low
                       or f"✓ {key}" in low)
            if not claimed:
                continue
            if clause.proof and self.judge is not None:
                # verify the claim against reality
                ok, detail = self.goal.prove_by_predicate(clause.id)
                if ok:
                    self.log.append("fact.learned",
                                    {"fact": f"clause {clause.id} proven: "
                                             f"{detail}", "kind": "goal"})
            elif clause.advisory:
                # advisory clauses have no predicate — human-tracked
                self.goal.prove_clause(clause.id, True, "human_approval",
                                       detail="model claim, advisory clause")

    def _prune_for_model(self, reserved_tokens: int = 0) -> list:
        """PERF: model-visible message view — sliding window prune plus a
        hard per-request token budget (promptbudget.bound_prompt).

        self.messages stays canonical (event log, checkpoints keep
        everything); only the API payload shrinks. reserved_tokens is
        the token cost of the tool schemas, so messages + schemas stay
        under cfg.prompt_token_budget together.
        """
        _resolve_lazy("bound_prompt")
        try:
            window = int(getattr(self.cfg, "prune_window", 40) or 0)
        except (TypeError, ValueError):
            window = 40
        try:
            budget = int(getattr(self.cfg, "prompt_token_budget", 0) or 0)
        except (TypeError, ValueError):
            budget = 0
        msgs = prune_messages(self.messages, window=window)
        if budget > 0:
            msgs = bound_prompt(
                msgs, budget_tokens=budget,
                model_id=getattr(self.model, "id", ""),
                reserved_tokens=reserved_tokens)
        return msgs

    def _complete(self, on_token, on_reasoning, on_status, should_cancel=None,
                  on_tool_args=None, _prewarmed=None):
        schemas = self._tool_schemas()
        # PERF: send the pruned, budget-capped view — never the full
        # history. reserved_tokens keeps messages + tool schemas under
        # the prompt_token_budget ceiling together.
        _resolve_lazy("schemas_tokens")
        _reserved = schemas_tokens(schemas, getattr(self.model, "id", ""))
        if _prewarmed is not None:
            # Worker 5/20 (flight prewarm): O(window) patch of the prompt
            # skeleton prebuilt during the previous API wait —
            # byte-identical to _prune_for_model(); falls back to the
            # normal path on any doubt.
            try:
                from .prewarm import patched_prune_for_model
                model_msgs = patched_prune_for_model(self, _prewarmed,
                                                     _reserved)
            except Exception:
                model_msgs = None
            if model_msgs is None:
                model_msgs = self._prune_for_model(
                    reserved_tokens=_reserved)
        else:
            model_msgs = self._prune_for_model(
                reserved_tokens=_reserved)
        # WIRE GATE: pruning and budget capping produce a detached view.
        # Re-assert the sealed prompt on the exact list that will be sent,
        # every iteration — a dropped or shadowed prompt must never reach
        # the provider just because a view was rebuilt.
        self.mastermind.gate.seal_wire(self.cfg.prompt, model_msgs)

        def _shrink_view() -> bool:
            # The client retries with the exact `messages` list it was
            # given; self._overflow_shrink() only shrinks self.messages,
            # so the detached, budget-capped view must be rebuilt IN
            # PLACE or layer-3 recovery resends the oversized payload.
            if not self._overflow_shrink():
                return False
            model_msgs[:] = self._prune_for_model(reserved_tokens=_reserved)
            self.mastermind.gate.seal_wire(self.cfg.prompt, model_msgs)
            return True

        # cassette replay: zero API cost, fully deterministic (§20.2)
        if self.cassette is not None and self.cassette.mode == "replay":
            from .client import StreamResult
            stored = self.cassette.replay(self.model.id, model_msgs,
                                          schemas, self.effort.key)
            if stored is None:
                raise APIError("cassette replay miss — request not in the "
                               "recorded cassette (deterministic replay "
                               "never falls back to a live call)")
            content = stored.get("content", "")
            if content and on_token:
                on_token(content)
            return StreamResult(content=content,
                                reasoning=stored.get("reasoning", ""),
                                tool_calls=stored.get("tool_calls", []),
                                usage=stored.get("usage"),
                                model=self.model.id)
        # Worker 5/20 (flight prewarm): snapshot the conversation and
        # start the background prewarm for the NEXT iteration — compact
        # analysis, prompt skeleton and read-only env probes run while
        # the provider streams, instead of serially after the response
        # lands. Read-only and snapshot-isolated; any mutation during
        # the wait invalidates the flight at consume time.
        try:
            _fp = getattr(self, "_flight_prewarmer", None)
            if _fp is None:
                from .prewarm import FlightPrewarmer
                _fp = self._flight_prewarmer = FlightPrewarmer()
            _fp.launch(self)
        except Exception:
            pass
        def _attempt():
            return chat_stream(self.provider, self.model, self.effort,
                               model_msgs, schemas,
                               on_token=on_token,
                               on_reasoning=on_reasoning,
                               on_tool_start=lambda n: on_status(f"tool:{n}"),
                               on_tool_args=on_tool_args,
                               should_cancel=should_cancel,
                               on_overflow=_shrink_view,
                               on_status=on_status)

        try:
            result = _attempt()
        except APIError as e:
            # some providers refuse streaming or tool params — degrade gracefully
            msg = str(e).lower()
            if is_context_overflow(str(e)):
                # last line of defence: the client already re-clamped,
                # retried and shrunk, so compact one final time and try
                # once more — a long session must never die with a
                # context-length error.
                on_status("compacting context")
                self._emergency_compact()
                # CORRECTNESS: model_msgs is a detached, budget-capped view
                # of self.messages — _emergency_compact() shrank the latter,
                # so recompute the view or the retry resends the same
                # oversized prompt that just overflowed.
                model_msgs = self._prune_for_model(reserved_tokens=_reserved)
                self.mastermind.gate.seal_wire(self.cfg.prompt, model_msgs)
                result = chat_stream(self.provider, self.model, self.effort,
                                     model_msgs, schemas,
                                     on_token=on_token,
                                     on_reasoning=on_reasoning,
                                     on_tool_start=lambda n: on_status(f"tool:{n}"),
                                     on_tool_args=on_tool_args,
                                     should_cancel=should_cancel,
                                     on_overflow=_shrink_view,
                                     on_status=on_status)
            elif e.status == 400 and "tool" in msg and schemas:
                on_status("retrying (no tools)")
                # NOTE: model_msgs (pruned + budget-capped), NOT
                # self.messages — the unpruned history must never go on
                # the wire (worker 2/20 prompt-size audit).
                result = chat_stream(self.provider, self.model, self.effort,
                                     model_msgs, None,
                                     on_token=on_token,
                                     on_reasoning=on_reasoning,
                                     should_cancel=should_cancel,
                                     on_overflow=_shrink_view,
                                     on_status=on_status)
            elif e.status == 400 and "stream" in msg:
                on_status("retrying (non-stream)")
                result = chat_blocking(self.provider, self.model,
                                       self.effort, model_msgs, schemas,
                                       on_overflow=_shrink_view,
                                       should_cancel=should_cancel,
                                       on_status=on_status)
            else:
                # Provider errors never change the selected model. Record
                # the failure and surface it so the user can choose whether
                # to retry or manually select another model.
                self.health["model_errors"][self.cfg.model_id] = \
                    self.health["model_errors"].get(self.cfg.model_id, 0) + 1
                raise
        if self.cassette is not None and self.cassette.mode == "record":
            self.cassette.record(self.model.id, model_msgs, schemas,
                                 {"content": result.content,
                                  "reasoning": result.reasoning,
                                  "tool_calls": result.tool_calls,
                                  "usage": result.usage},
                                 effort_key=self.effort.key)
        return result

    # -- autonomy + gating -----------------------------------------------------

    def set_autonomy(self, level: int) -> str:
        level = max(0, min(5, level))
        # the contract's autonomy ceiling cannot be raised by the agent (§37.1)
        if self.goal.status().active:
            ceiling = self._goal_autonomy_ceiling()
            if ceiling is not None and level > ceiling:
                level = ceiling
        self.autonomy = level
        self.log.append("autonomy.changed", {"level": level}, actor="human")
        return AUTONOMY_LEVELS[level]

    def _goal_autonomy_ceiling(self) -> int | None:
        g = fold(self.log).goal
        if not g:
            return None
        ceiling = g.get("autonomy_ceiling")
        return int(ceiling) if ceiling is not None else None

    # -- focus mode (deep work) ---------------------------------------------------

    def focus_continue(self, last_turn: Turn, remaining: int) -> str | None:
        """FOCUS MODE brain: decide whether the deep-work loop should keep
        going, and if so, return the next continuation prompt.

        Returns None when focus should stop. Every decision is sealed in
        the event log (focus.tick / focus.stop) — autonomous work is never
        invisible (A7).

        Stop conditions (mechanical, rung 1):
          * the turn errored, was cancelled, or the budget paused it
          * the goal closed / every clause is proven or waived
          * distance stalled: no improvement for 3 consecutive ticks
          * no goal and the agent produced a final answer with no tool
            work twice in a row (it believes itself done)
        """
        _resolve_lazy("TERMINAL_STATES")
        goal = self.goal.status()

        def _stop(reason: str) -> None:
            self.log.append("focus.stop", {"reason": reason},
                            actor="kernel")
            self._focus_history.clear()

        if last_turn.error:
            _stop(f"turn ended with: {last_turn.error[:80]}")
            return None

        if goal.active:
            open_clauses = [c for c in goal.clauses
                            if c.state in ("OPEN", "REGRESSED")]
            if not open_clauses or goal.closed_state in TERMINAL_STATES:
                _stop("goal achieved — every clause proven or waived")
                return None
            distance = goal.distance
            self._focus_history.append(distance)
            # bound memory: stall detection only ever reads the last 4
            del self._focus_history[:-_FOCUS_HISTORY_MAX]
            recent = self._focus_history[-4:]
            if len(recent) >= 4 and all(
                    recent[i] >= recent[i - 1] - 1e-9
                    for i in range(1, len(recent))):
                _stop(f"stalled — distance stuck at {distance:.2f} for 3 "
                      "ticks; needs a human decision")
                return None
            focus_clause = goal.focus or open_clauses[0].id
            self.log.append("focus.tick",
                            {"distance": distance,
                             "remaining": remaining,
                             "focus": focus_clause},
                            actor="kernel")
            return (f"CONTINUE — deep-work mode ({remaining} turns left). "
                    f"Goal: {goal.statement}. Current focus: clause "
                    f"{focus_clause}. Distance to done: {distance:.2f}. "
                    f"Do NOT repeat completed work and do NOT re-prove "
                    f"PROVEN clauses. Take the next concrete step that "
                    f"moves clause {focus_clause} forward, then verify it.")

        # no active goal — stop when the agent is clearly done
        worked = bool(last_turn.tools)
        self._focus_history.append(1.0 if worked else 0.0)
        # bound memory: done-detection only ever reads the last 2
        del self._focus_history[:-_FOCUS_HISTORY_MAX]
        tail = self._focus_history[-2:]
        if len(tail) == 2 and tail == [0.0, 0.0]:
            _stop("agent answered without further tool work — done")
            return None
        self.log.append("focus.tick", {"remaining": remaining},
                        actor="kernel")
        return (f"CONTINUE — deep-work mode ({remaining} turns left). "
                f"Review what is done, verify it against reality, and "
                f"complete whatever is still missing. Do not repeat "
                f"completed work.")

    def _push_status(self, text: str) -> None:
        """Stream a live status line into the current turn's UI (if any).
        Used by long-running tools so progress is never
        invisible."""
        cb = self._turn_status
        if cb is None:
            return
        try:
            cb(text)
        except Exception:
            pass

    def _attribute(self, tool_name: str) -> tuple[str | None, str | None]:
        _resolve_lazy("TERMINAL_STATES")
        """§38.1 total attribution. Returns (clause_id, orphan_reason).

        With an active goal, every action binds to the focus clause (the
        highest-gravity open clause). If no open clause exists, the action
        is an orphan and is rejected before execution — EXCEPT once the
        contract is closed: a terminal contract (§42) no longer demands
        attribution, or the agent could never run another command after a
        goal completed. Attribution only ever gates an OPEN contract."""
        goal = self.goal.status()
        if not goal.active:
            return None, None  # normal mode: no attribution required
        if goal.closed_state in TERMINAL_STATES:
            return None, None  # the contract is settled — no orphan gate
        focus = goal.focus or self.goal.reaim()
        open_ids = {c.id for c in goal.clauses
                    if c.state in ("OPEN", "REGRESSED")}
        if focus and focus in open_ids:
            return focus, None
        if open_ids:
            return sorted(open_ids)[0], None
        return None, ("OrphanAction: no open clause to serve — the goal "
                      "is fully proven, waived, or blocked. Close the goal "
                      "or amend the contract.")

    def _gate(self, tool: Tool, args: dict) -> str | None:
        """Return a block reason, or None if the action may proceed."""
        # autonomy ladder (§22)
        if tool.name in _MUTATING_TOOLS:
            if self.autonomy <= 1:
                return (f"autonomy L{self.autonomy} is read-only — "
                        "raise it with /autonomy to allow mutations")
            if self.autonomy == 2:
                return "ASK"
            if self.autonomy == 4 and tool.name in _ALWAYS_ASK:
                return "ASK"
        # IRREVERSIBLE tools always need approval, at every level (§22)
        if tool.name in ("web_fetch", "web_search") and self.autonomy < 5:
            pass  # network reads stay free; network writes would be gated
        # deterministic dead-end blocking (§14.3)
        sig = _signature(tool.name, args)
        if self.memory.is_dead_end(sig):
            return (f"this exact approach is in the dead-end ledger "
                    f"(signature {sig}) — choose a different approach")
        return None

    def _snapshot_paths(self, tool_name: str, args: dict) -> list[str]:
        """The paths a mutating tool touches — the snapshot targets (A2)."""
        keys = _PATH_ARG_TOOLS.get(tool_name, ())
        paths = [str(args[k]) for k in keys if args.get(k)]
        if tool_name == "run_command":
            # commands can touch anything; snapshot the cwd tree shallowly.
            # Sorted for determinism — unsorted iterdir() made snapshot
            # contents order-dependent across runs.
            paths = [str(p) for p in sorted(Path.cwd().iterdir())
                     if p.is_file()][:200]
        return paths

    @staticmethod
    def _handler_takes_cancel(handler) -> bool:
        """True if the tool handler declares a `should_cancel` keyword
        (so _execute_tool may pass the live Esc/Ctrl+C flag). Cached
        per handler — inspect.signature on every tool call was
        measurable in tight tool loops."""
        cache = Agent._cancel_kwarg_cache
        try:
            return cache[handler]
        except KeyError:
            pass
        except TypeError:  # unhashable handler — introspect directly
            return Agent._has_cancel_kwarg(handler)
        takes = Agent._has_cancel_kwarg(handler)
        try:
            cache[handler] = takes
        except TypeError:
            pass
        return takes

    @staticmethod
    def _has_cancel_kwarg(handler) -> bool:
        # inspect is imported lazily (Worker 8/20): it costs ~150ms and is
        # only needed when a tool call is dispatched.
        import inspect
        try:
            return "should_cancel" in inspect.signature(handler).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _check_tool_args(tool: Tool, args: dict,
                         extra: dict) -> str | None:
        """Pre-validate keyword binding so a *binding* TypeError is
        reported as bad arguments, while a TypeError raised *inside*
        the handler keeps its real identity. (It used to be caught by
        the same `except TypeError` and misreported as bad arguments —
        after two occurrences the dead-end ledger recorded a bogus
        'bad arguments' reason and blocked that call shape for the
        whole session.)"""
        import inspect  # lazy: ~150ms, only needed on tool dispatch
        try:
            sig = inspect.signature(tool.handler)
        except (TypeError, ValueError):
            return None  # can't introspect — let the call itself decide
        try:
            sig.bind(**args, **extra)
        except TypeError as e:
            return str(e)
        return None

    def _run_tool_with_timeout(self, tool: Tool, args: dict,
                                 extra: dict):
        """Run ``tool.handler`` bounded by the per-tool timeout.

        The handler runs in a daemon thread; the wait is limited to
        ``TOOL_TIMEOUTS[tool.name]`` (default ``TOOL_TIMEOUT_DEFAULT``).
        While it runs, ``subprocess.Popen`` is tracked so that on
        timeout every child process tree the tool spawned is killed
        via the shared ``_kill_process_tree`` helper (killpg). Raises
        ``ToolTimeout`` on timeout; exceptions raised by the handler
        itself (including ``TurnCancelled``) are re-raised unchanged.
        """
        import threading
        import subprocess as _sp

        limit = TOOL_TIMEOUTS.get(tool.name, TOOL_TIMEOUT_DEFAULT)
        spawned: list = []
        lock = threading.Lock()
        orig_popen = _sp.Popen

        def _tracking_popen(*a, **k):
            # subprocess.run / check_output / call all go through
            # subprocess.Popen, so every spawned child is captured.
            p = orig_popen(*a, **k)
            with lock:
                spawned.append(p)
            return p

        outcome: dict = {}

        def _target():
            _sp.Popen = _tracking_popen
            try:
                outcome["result"] = tool.handler(**args, **extra)
            except BaseException as e:  # noqa: BLE001 — re-raised below
                outcome["error"] = e
            finally:
                _sp.Popen = orig_popen

        worker = threading.Thread(target=_target, daemon=True,
                                  name=f"tool-{tool.name}")
        worker.start()
        worker.join(limit)
        # restore even if the thread is still running — it can only kill
        # processes, so a still-running worker is harmless to the turn.
        _sp.Popen = orig_popen
        if worker.is_alive():
            try:
                from .tools import _kill_process_tree
                for p in spawned:
                    try:
                        if p.poll() is None:
                            _kill_process_tree(p)
                    except Exception:
                        pass
            except Exception:
                pass
            raise ToolTimeout(tool.name, limit)
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("result")

    def _execute_tool(self, ev: ToolEvent,
                      approve: Callable[[Tool, dict], bool],
                      on_status: Callable[[str], None],
                      causation_id: str | None = None,
                      on_tool_output: Callable[[str, str], None] | None = None,
                      ) -> None:
        tool = self.tools.get(ev.name)
        started = time.time()
        if tool is None:
            ev.status = "error"
            ev.result = f"ERROR: unknown tool '{ev.name}'. Available: " \
                        + ", ".join(self.tools)
            return

        # §38.1 attribution — orphans are rejected before execution
        clause_id, orphan = self._attribute(ev.name)
        ev.clause_id = clause_id
        if orphan:
            ev.status = "blocked"
            ev.result = f"ERROR: blocked — {orphan}"
            self.log.append("tool.blocked", {"name": ev.name,
                                             "reason": orphan},
                            causation_id=causation_id)
            return

        # Feature: plan mode gate — mutating tools blocked until plan approved
        try:
            from .planmode import should_block_mutating_tool
            plan_blocked, plan_msg = should_block_mutating_tool(
                tool.name, tool.risk)
            if plan_blocked:
                ev.status = "blocked"
                ev.result = f"ERROR: blocked — {plan_msg}"
                self.log.append("tool.blocked",
                                {"name": ev.name, "reason": plan_msg},
                                causation_id=causation_id)
                return
        except Exception:
            pass

        # Feature: PreToolUse hooks (can block or patch args)
        if getattr(self, "hooks_enabled", False):
            try:
                from .hooks import run_pre_tool
                allowed, patched, hook_msg = run_pre_tool(ev.name, ev.args)
                if not allowed:
                    ev.status = "blocked"
                    ev.result = f"ERROR: blocked — {hook_msg}"
                    self.log.append("tool.blocked",
                                    {"name": ev.name, "reason": hook_msg},
                                    causation_id=causation_id)
                    return
                ev.args = patched
            except Exception:
                pass

        # Feature: auto-checkpoint before risky tools
        try:
            from .checkpoints import maybe_auto_checkpoint
            maybe_auto_checkpoint(self, ev.name)
        except Exception:
            pass

        block = self._gate(tool, ev.args)
        if block and block != "ASK":
            ev.status = "blocked"
            ev.result = f"ERROR: blocked — {block}"
            self.log.append("tool.blocked", {"name": ev.name,
                                             "reason": block},
                            causation_id=causation_id,
                            correlation_id=clause_id)
            return

        needs_ask = (block == "ASK"
                     or (tool.risk == RISK_CONFIRM and not self.cfg.auto_approve
                         and self.autonomy < 4))
        if needs_ask and not self.cfg.auto_approve:
            if not approve(tool, ev.args):
                ev.status = "denied"
                ev.result = "ERROR: user denied this action. Ask the user " \
                            "how to proceed or choose another approach."
                return

        # A2/I3: snapshot BEFORE any mutation — no write without a
        # committed recovery path. A snapshot failure must never kill the
        # turn: log it and proceed (the tool itself still runs).
        if ev.name in _MUTATING_TOOLS:
            try:
                paths = self._snapshot_paths(ev.name, ev.args)
                if paths:
                    snap = self.store.take(paths)
                    self.log.append("snapshot.taken",
                                    {"tree": snap["tree"],
                                     "paths": list(snap["paths"]),
                                     "before_tool": ev.name},
                                    actor="kernel",
                                    causation_id=causation_id,
                                    correlation_id=clause_id)
            except Exception as e:  # noqa: BLE001
                _log.warning("snapshot before %s failed: %s", ev.name, e)

        self.log.append("tool.call", {"name": ev.name, "args": ev.args},
                        actor="sovereign", provenance="model",
                        causation_id=causation_id,
                        correlation_id=clause_id)

        # diffpreview: show unified diff + confirm before applying changes
        if ev.name in ("write_file", "edit_file", "apply_patch",
                       "MultiEdit", "multiedit"):
            _preview = getattr(self, "diff_preview", None)
            if callable(_preview):
                _ui = getattr(self, "ui", None)
                try:
                    _proceed = _preview(ev.name, ev.args, _ui)
                except Exception:
                    _proceed = True
                if not _proceed:
                    ev.status = "denied"
                    ev.result = ("ERROR: user declined the diff preview. "
                                 "Ask the user how to proceed or choose "
                                 "another approach.")
                    return

        # tooldedup hook — the identical call already ran this turn: serve
        # the cached result instead of re-executing (short-circuits the
        # identical-retry loop from the 512s incident).
        from .tooldedup import check as _dedup_check
        _dedup_hit = _dedup_check(self, ev.name, ev.args)
        if _dedup_hit is not None:
            ev.result, ev.status = _dedup_hit
            ev.duration = 0.0
            return

        on_status(f"running:{ev.name}")
        # v3: if the Speculator already prefetched this exact read-only
        # call, serve it from the cache instead of re-running (a hit).
        cached = self.speculator.serve(ev.name, ev.args)
        if cached is not None:
            ev.result = cached
            ev.status = "done"
        else:
            # shell tools stream their output live — hand them the relay
            extra = {}
            if on_tool_output is not None and ev.name in ("run_command",
                                                          "live_shell"):
                extra["on_output"] = on_tool_output
            # Esc/Ctrl+C: any handler that declares a `should_cancel`
            # kwarg gets the live cancel flag — shell tools kill their
            # subprocess, web_fetch/web_search abort their HTTP calls,
            # etc. Signature-based (not name-based) so new cancellable
            # tools are wired automatically; handlers without the kwarg
            # are untouched (backwards compatible).
            if self._handler_takes_cancel(tool.handler):
                extra["should_cancel"] = self._cancel_flag.is_set
            bind_error = self._check_tool_args(tool, ev.args, extra)
            if bind_error is not None:
                ev.result = (f"ERROR: bad arguments for {ev.name}: "
                             f"{bind_error}")
                ev.status = "error"
            else:
                try:
                    ev.result = self._run_tool_with_timeout(
                        tool, ev.args, extra)
                    ev.status = "done"
                except ToolTimeout as e:
                    ev.result = f"ERROR: {e}"
                    ev.status = "error"
                except TurnCancelled:
                    # CANCEL: a handler interrupted by Esc must abort the
                    # turn, not be swallowed into an "ERROR: ..." tool
                    # result that the turn then keeps going past.
                    raise
                except Exception as e:  # noqa: BLE001 — tool errors go back to the LLM
                    ev.result = f"ERROR: {type(e).__name__}: {e}"
                    ev.status = "error"
        # BUGFIX: handlers are contracted to return str, but a custom or
        # forged tool may return anything — .startswith() below would raise
        # AttributeError and kill the whole turn. Coerce once, here.
        if not isinstance(ev.result, str):
            ev.result = "" if ev.result is None else str(ev.result)
        # most handlers report failure as an "ERROR: …" string rather than
        # raising — count those as errors too, so the dead-end ledger works
        if ev.status == "done" and ev.result.startswith("ERROR:"):
            ev.status = "error"
        # tooldedup hook — record the completed execution so identical calls
        # later in this turn hit the cache. Only done/error: blocked/denied
        # results are never cached (gates and user denials re-evaluate).
        if ev.status in ("done", "error"):
            try:
                from .tooldedup import store as _dedup_store
                _dedup_store(self, ev.name, ev.args, ev.result, ev.status)
            except Exception:  # noqa: BLE001 — caching must never break a turn
                pass
        ev.duration = time.time() - started

        # §37.4: after EVERY successful write, re-check the anti-clauses
        if ev.status == "done" and ev.name in _MUTATING_TOOLS:
            violations = self.goal.check_anti_clauses()
            if violations:
                v = violations[0]
                ev.result += (f"\nWARNING: anti-clause {v['clause']} "
                              f"violated — {v['detail']}. The run cannot "
                              f"close until this is repaired or rewound.")

        # §13.4 oscillation: file content flipping A-B-A-B
        if ev.name in ("write_file", "edit_file") and ev.status == "done":
            p = str(ev.args.get("path", ""))
            try:
                h = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
            except OSError:
                h = ""
            if h:
                hist = self._file_hashes.setdefault(p, [])
                hist.append(h)
                # bound the per-path history — only recent flips matter
                # for oscillation detection
                del hist[:-20]
                # bound the number of tracked paths — a long session can
                # touch thousands of files; evict the oldest paths
                # (dicts preserve insertion order)
                while len(self._file_hashes) > _FILE_HASH_PATHS_MAX:
                    oldest = next(iter(self._file_hashes))
                    if oldest == p:
                        break  # never evict the path just touched
                    del self._file_hashes[oldest]
                if self.loop_det.oscillation(p, hist):
                    self.log.append("loop.alert",
                                    {"kind": "oscillation", "path": p,
                                     "action": "hard stop — present both "
                                               "versions to the human"},
                                    actor="kernel")

        # Worker 11/20 — parallel tool execution: the two blocks below
        # touch agent-wide mutable state that is not thread-safe
        # (_error_counts read-modify-write, LoopDetector._last_sig), so
        # they run under _tool_exec_lock when calls fan out across
        # worker threads. Uncontended on the sequential path.
        with self._tool_exec_lock:
            # repeated identical failure -> deterministic dead-end (§14.3)
            if ev.status == "error":
                sig = _signature(ev.name, ev.args)
                self._error_counts[sig] = self._error_counts.get(sig, 0) + 1
                if self._error_counts[sig] == 2:
                    self.memory.record_dead_end(
                        signature=sig,
                        reason=f"{ev.name} failed twice: {ev.result[:200]}",
                        scope="session", confidence="contextual")
                # v3: healer captures + classifies the root cause and seals a
                # lesson, so the same failure is recognised instantly next time.
                # (No fixer attached here — the agent reads the diagnosis and
                # decides the fix; the healer never mutates on its own.)
                try:
                    self.healer.heal(ev.result,
                                     context=f"{ev.name} {ev.args}")
                except Exception:
                    pass

            # §13.4 exact-repeat detection over recent tool calls
            self.loop_det.detect()

        # Feature: PostToolUse hooks (failures only log, never block)
        if getattr(self, "hooks_enabled", False):
            try:
                from .hooks import run_post_tool
                run_post_tool(ev.name, ev.args, ev.result)
            except Exception:
                pass

    def export_report(self, fmt: str = "md") -> Path:
        _resolve_lazy("export_html", "export_markdown")
        """Write the enterprise audit report (md or html) to the cwd."""
        title = f"FullAgent session {self.session_id}"
        if fmt == "html":
            text, suffix = export_html(self.log, title), ".html"
        else:
            text, suffix = export_markdown(self.log, title), ".md"
        path = Path.cwd() / f"fullagent-report-{self.session_id}{suffix}"
        path.write_text(text)
        self.log.append("report.exported",
                        {"path": str(path), "format": fmt},
                        actor="human")
        return path

    def get_forecast(self) -> str:
        _resolve_lazy("forecast", "format_forecast")
        return format_forecast(forecast(self.log))

    # -- enterprise: provider health ------------------------------------------

    # -- enterprise: turn scorecard ---------------------------------------------------

    def _score_turn(self, turn: Turn) -> dict:
        """Deterministic quality metrics for the finished turn — sealed
        as turn.scorecard. No model judgement, only measured facts."""
        tool_calls = len(turn.tools)
        errors = sum(1 for t in turn.tools
                     if t.status in ("error", "blocked", "denied"))
        writes: dict[str, int] = {}
        for t in turn.tools:
            if t.name in ("write_file", "edit_file") and t.status == "done":
                path = str(t.args.get("path", ""))
                if path:
                    writes[path] = writes.get(path, 0) + 1
            elif t.name == "apply_patch" and t.status == "done":
                # apply_patch can touch multiple files; count it as one
                # write unit for rework detection
                writes["__apply_patch__"] = writes.get("__apply_patch__", 0) + 1
        rework = sum(1 for n in writes.values() if n > 1)
        # SPEED: events are seq-ordered — scan back from the head and stop
        # at the turn boundary instead of filtering the whole log.
        verdicts = []
        for e in reversed(self.log.events()):
            if e.seq <= self._turn_start_seq:
                break
            if e.type == "judge.verdict":
                verdicts.append(e)
        verified = sum(1 for v in verdicts if v.data.get("passed"))
        score = max(0, min(100, 100 - 20 * errors - 10 * rework))
        card = {"tool_calls": tool_calls, "errors": errors,
                "rework_files": rework, "verified": verified,
                "verdicts": len(verdicts), "score": score,
                "duration": round(turn.duration, 2)}
        turn.scorecard = card
        self.log.append("turn.scorecard", card, actor="kernel")
        return card

    # -- enterprise: compaction digests -----------------------------------------------

    def _digest_messages(self, msgs: list[dict]) -> str:
        """Deterministic one-line digest of a turn about to be compacted
        away — the knowledge survives even when the tokens don't."""
        tools_used: list[str] = []
        files: list[str] = []
        problems: list[str] = []
        for m in msgs:
            for tc in (m.get("tool_calls") or []):
                name = (tc.get("function") or {}).get("name", "")
                if name and name not in tools_used:
                    tools_used.append(name)
            if m.get("role") == "tool":
                content = str(m.get("content", ""))
                for match in re.finditer(
                        r"OK: (?:wrote \d+ chars to|replaced .*? in) (\S+)",
                        content):
                    f = match.group(1)
                    if f not in files:
                        files.append(f)
                if content.startswith("ERROR:"):
                    problems.append(content[6:80])
        parts = []
        if tools_used:
            parts.append("tools: " + ", ".join(tools_used[:6]))
        if files:
            parts.append("wrote: " + ", ".join(files[:5]))
        if problems:
            parts.append("hit: " + problems[0])
        return "; ".join(parts)

    # -- enterprise: notifications ------------------------------------------------------

    def _flush_notifications(self) -> None:
        """Emit any NOTIFY_EVENTS sealed since the last flush.

        SPEED: scan back from the head and stop at the last flush mark —
        the old full-log scan got slower every turn of a long session."""
        if not self.notifier.sink:
            self._notify_seq = self.log.head()
            return
        pending: list = []
        for ev in reversed(self.log.events()):
            if ev.seq <= self._notify_seq:
                break
            pending.append(ev)
        for ev in reversed(pending):  # restore chronological order
            if ev.type in NOTIFY_EVENTS:
                if not self.notifier.emit(ev.type, ev.data):
                    # sink is down — stop, don't stall the turn 5s per
                    # event retrying a dead webhook (a failed emit is
                    # already recorded on the notifier)
                    break
        self._notify_seq = self.log.head()

    # -- enterprise: session resume -------------------------------------------------------

    def sessions_catalog(self) -> list[dict]:
        """Every branch that carries a session — for /resume."""
        catalog = []
        for branch in self.log.branches():
            events = self.log.events(branch)
            session_id = ""
            started = 0.0
            for e in events:
                if e.type == "session.start":
                    session_id = str(e.data.get("session_id", ""))
                    started = e.ts
            catalog.append({"branch": branch, "session_id": session_id,
                            "events": len(events),
                            "head": self.log.head(branch),
                            "started": started})
        catalog.sort(key=lambda c: -c["started"])
        return catalog

    def resume_session(self, branch: str) -> int:
        _resolve_lazy("fold")
        """Checkout a branch and rebuild the conversation from the fold.
        The full history (tool calls, verdicts) stays in the log; the
        model-visible context is rebuilt from user/assistant messages."""
        if branch not in self.log.branches():
            raise ValueError(
                f"unknown branch {branch!r} — known: "
                + ", ".join(self.log.branches()))
        self.log.checkout(branch)
        st = fold(self.log)
        session_id = ""
        for e in self.log.events():
            if e.type == "session.start":
                session_id = str(e.data.get("session_id", "")) or session_id
        self.session_id = session_id or self.session_id
        self.log.session = self.session_id
        self.messages = []
        self._reseat_system_prompt()
        self.messages.extend(st.messages)
        self.turns = []
        self._focus_history.clear()
        self.log.append("session.resumed",
                        {"branch": branch, "session_id": self.session_id,
                         "messages": len(st.messages)},
                        actor="human")
        return len(st.messages)

    # -- time travel (§9) --------------------------------------------------------

    def _nearest_snapshot(self, upto_seq: int) -> dict | None:
        """The snapshot whose captured state matches 'as of upto_seq' (§9.2).

        A snapshot.taken event at seq N records the world immediately
        BEFORE the write it precedes — i.e. the state as of seq N-1. So
        the state 'as of S' is the newest snapshot with seq <= S+1."""
        best = None
        for ev in self.log.events(upto_seq=upto_seq + 1):
            if ev.type == "snapshot.taken":
                best = ev
        return best

    def rewind_to(self, seq: int) -> tuple[int, int]:
        _resolve_lazy("fold")
        """REWIND (§9.1): filesystem AND agent state return to step N.

        Materialises the nearest snapshot at/below seq, then rewinds the
        log head and rebuilds the conversation from the fold. The old
        branch remains fully intact — nothing is destroyed.
        Returns (new_head, messages_kept)."""
        snap = self._nearest_snapshot(seq)
        if snap:
            self.store.materialise(snap.data.get("tree", ""))
        new_head = self.log.rewind(seq)
        st = fold(self.log, upto_seq=new_head)
        self.messages = []
        self._reseat_system_prompt()
        self.messages.extend(st.messages)
        return new_head, len(st.messages)

    def revert_files_to(self, seq: int) -> dict:
        """REVERT (§9.1): filesystem only returns to step N — the agent
        KEEPS its memory of what happened. This is what feeds the Dead-End
        Ledger: the files go back, the lesson stays."""
        snap = self._nearest_snapshot(seq)
        if not snap:
            return {"error": f"no snapshot at or before seq {seq}"}
        result = self.store.materialise(snap.data.get("tree", ""))
        self.log.append("kernel.revert",
                        {"to_seq": seq, "tree": snap.data.get("tree"),
                         **result}, actor="human")
        return result

    def fork_timeline(self, name: str | None = None) -> str:
        """Branch the timeline at the current head; continue on the fork."""
        branch = self.log.fork(name=name)
        self.log.checkout(branch)
        self.log.append("session.start",
                        {"session_id": self.session_id, "branch": branch})
        return branch

    # -- code intelligence (§15) ---------------------------------------------------

    def _register_code_tools(self) -> None:
        """Add the deterministic code.* tools (rung 2-4) to the registry."""
        nexus = self.nexus

        def code_symbols(name: str = "", path: str = ".") -> str:
            nexus.index(path)
            if name:
                syms = nexus.find_symbol(name)
                if not syms:
                    return f"no symbol named {name!r} under {path}"
                return "\n".join(f"{s.kind} {s.name}  {s.path}:{s.lineno} "
                                 f"params=({', '.join(s.params)})"
                                 for s in syms)
            lines = [f"{s.kind} {s.name}  {s.path}:{s.lineno}"
                     for s in list(nexus.idx.symbols.values())[:200]]
            return "\n".join(lines) or "no symbols found"

        def code_impact(name: str, path: str = ".") -> str:
            nexus.index(path)
            return nexus.format_impact(name)

        self.tools["code_symbols"] = Tool(
            "code_symbols",
            "List or find code symbols (functions/classes) via AST — "
            "precise and cheaper than grep. Args: name (optional), path.",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"}}, "required": []},
            code_symbols)
        self.tools["code_impact"] = Tool(
            "code_impact",
            "Impact analysis: if I change this symbol, what breaks? "
            "Callers, tests, public API, risk score. Args: name, path.",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"}}, "required": ["name"]},
            code_impact)


    def _register_v4_tools(self) -> None:
        """Add the deterministic v4 engineering tools (rung 2-4) to the
        registry: static analysis, knowledge graph, coverage, fuzzing."""
        static = self.static
        kgraph = self.kgraph
        coverage = self.coverage
        fuzzer = self.fuzzer

        def analyze_code(path: str = ".", glob_filter: str = "*.py",
                         max_files: int = 50) -> str:
            p = Path(path).expanduser()
            if p.is_file():
                return static.format_report(static.analyze_file(str(p)))
            tree = static.analyze_tree(str(p), glob_filter, max_files)
            return static.format_report(tree)

        def graph_index(path: str = ".") -> str:
            p = Path(path).expanduser()
            sources: dict[str, str] = {}
            files = ([p] if p.is_file()
                     else sorted(p.glob("**/*.py"))[:200])
            for f in files:
                if f.is_file():
                    try:
                        sources[f.stem] = f.read_text(errors="replace")
                    except OSError:
                        continue
            kgraph.index_code(sources)
            kgraph.index_log()
            return kgraph.format_status()

        def graph_query(name: str, kind: str = "") -> str:
            hits = kgraph.find(name, kind or None)
            if not hits:
                return f"no entity matching {name!r} — run graph_index first"
            lines = []
            for e in hits[:20]:
                lines.append(f"{e.kind} {e.id}  ({e.name})")
                for r in kgraph.out_edges(e.id)[:8]:
                    lines.append(f"    --{r.rel}--> {r.dst}")
                for r in kgraph.in_edges(e.id)[:8]:
                    lines.append(f"    <--{r.rel}-- {r.src}")
            return "\n".join(lines)

        def graph_impact(name: str) -> str:
            hits = kgraph.find(name)
            if not hits:
                return f"no entity matching {name!r} — run graph_index first"
            lines = []
            for e in hits[:5]:
                dep = kgraph.impact(e.id)
                lines.append(f"{e.id}: {len(dep)} dependent(s)")
                lines.extend(f"    {d}" for d in dep[:20])
            return "\n".join(lines)

        def measure_coverage(path: str, command: str) -> str:
            """Coverage of `path` while `command` (a python -c snippet or
            test command) runs. The subject is executed via subprocess so
            the trace stays inside this process only for the import."""
            tp = Path(path).expanduser()
            if not tp.is_file():
                return f"ERROR: not a file: {tp}"
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                tp.stem, str(tp.resolve()))
            if spec is None or spec.loader is None:
                return f"ERROR: cannot import {tp}"
            mod = importlib.util.module_from_spec(spec)

            def subject():
                spec.loader.exec_module(mod)
                # run the command as a python expression against the module
                eval(compile(command, "<coverage-subject>", "exec"),
                     {"mod": mod, "__name__": "__coverage__"})

            res = coverage.measure(str(tp), subject)
            missed = ", ".join(str(m) for m in res.missed[:30])
            return (f"coverage {res.percent:.0f}%  "
                    f"({res.hit}/{res.total} lines)\nmissed: {missed}")

        def fuzz_target(path: str, function: str, iterations: int = 200,
                        nargs: int = 1) -> str:
            tp = Path(path).expanduser()
            if not tp.is_file():
                return f"ERROR: not a file: {tp}"
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"fuzz_{tp.stem}", str(tp.resolve()))
            if spec is None or spec.loader is None:
                return f"ERROR: cannot import {tp}"
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                return f"ERROR: import failed: {type(e).__name__}: {e}"
            fn = getattr(mod, function, None)
            if not callable(fn):
                return f"ERROR: no callable {function!r} in {tp}"
            report = fuzzer.fuzz(fn, iterations=iterations, nargs=nargs,
                                 name=f"{tp.stem}.{function}")
            lines = [f"fuzz {report.target}: {report.iterations} runs, "
                     f"{report.crashes} crash(es), "
                     f"{report.invariant_failures} invariant failure(s)"]
            if report.first_crash:
                c = report.first_crash
                lines.append(f"  first crash: {c.error}")
                lines.append(f"    args: {repr(c.args)[:120]}")
                lines.append(f"    shrunk: {repr(c.shrunk_args)[:120]} "
                             f"→ {c.shrunk_error}")
            return "\n".join(lines)

        self.tools["analyze_code"] = Tool(
            "analyze_code",
            "Static analysis of a file or tree: taint flows (source→sink), "
            "cyclomatic complexity hotspots, import cycles. Deterministic "
            "AST analysis. Args: path, glob_filter, max_files.",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "glob_filter": {"type": "string"},
                "max_files": {"type": "integer"}}, "required": []},
            analyze_code)
        self.tools["graph_index"] = Tool(
            "graph_index",
            "Build the knowledge graph (entities + typed relations) from "
            "Python sources under path, plus the session log. Args: path.",
            {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": []},
            graph_index)
        self.tools["graph_query"] = Tool(
            "graph_query",
            "Query the knowledge graph: find entities by name and show "
            "their relations. Args: name, kind (optional).",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "kind": {"type": "string"}}, "required": ["name"]},
            graph_query)
        self.tools["graph_impact"] = Tool(
            "graph_impact",
            "Impact analysis over the knowledge graph: everything that "
            "depends on this entity ('what breaks if I change X?'). "
            "Args: name.",
            {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
            graph_impact)
        self.tools["measure_coverage"] = Tool(
            "measure_coverage",
            "Real line coverage (sys.settrace) of a Python file while a "
            "subject snippet runs. Args: path (file under test), command "
            "(python code executed with the module bound as `mod`).",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "command": {"type": "string"}},
                "required": ["path", "command"]},
            measure_coverage, risk=RISK_CONFIRM)
        self.tools["fuzz_target"] = Tool(
            "fuzz_target",
            "Property-based fuzzing of a function: generated + boundary + "
            "mutated inputs, crash shrinking to a minimal reproducer. "
            "Args: path, function, iterations, nargs.",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "function": {"type": "string"},
                "iterations": {"type": "integer"},
                "nargs": {"type": "integer"}},
                "required": ["path", "function"]},
            fuzz_target, risk=RISK_CONFIRM)

    # -- v3 subsystem callbacks ------------------------------------------------

    def _spec_runner(self, name: str, args: dict) -> str:
        """Execute one read-only tool call for the Speculator. Only the
        whitelisted read-only tools ever reach here (the speculator gates
        on SPECULATIVE_TOOLS before calling)."""
        if name not in SPECULATIVE_TOOLS:
            return f"ERROR: {name} is not speculative-safe"
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name}"
        try:
            return tool.handler(**args)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _daemon_step(self, task: str) -> str:
        """Run one daemon mission step directly, without delegated agents."""
        from .systemprompt import daemon_step_prompt
        try:
            result = chat_blocking(
                self.provider, self.model, self.effort,
                [{"role": "system", "content": daemon_step_prompt()},
                 {"role": "user", "content": str(task)}],
                None, timeout=120.0)
            content = (result.content or "").strip()
            return f"OK: {content}" if content else "ERROR: empty mission result"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _council_speaker(self, role: str, brief: str) -> str:
        """Produce one council position through the model (blocking, no
        tools). The synthesis brief is already blind — it carries only the
        two arguments."""
        from .systemprompt import council_speaker_prompt
        messages = [{"role": "system", "content": council_speaker_prompt()},
                    {"role": "user", "content": brief}]
        result = chat_blocking(self.provider, self.model, self.effort,
                               messages, None, timeout=120.0)
        return result.content or ""

    # -- v5 advanced subsystem bridges ---------------------------------------

    def _debate_speaker(self, model_id: str, prompt: str) -> str:
        """Route one tournament turn to its participant model."""
        from .systemprompt import debate_speaker_prompt
        m = model_by_id(model_id) or self.model
        provider = PROVIDERS.get(m.provider, self.provider)
        result = chat_blocking(
            provider, m, self.effort,
            [{"role": "system", "content": debate_speaker_prompt()},
             {"role": "user", "content": prompt}],
            None, timeout=120.0)
        return result.content or ""

    # -- v6 frontier subsystem bridges --------------------------------------

    def _mcts_evaluator(self, assignment: dict) -> float:
        """Rollout scorer for the strategy tree: mechanical role-task
        fit (write verbs want write-capable roles, run verbs want
        runners) — cheap by design, MCTS supplies the exploration."""
        from .roles import ROLES
        if not hasattr(self, "_mcts_items"):
            return 0.0
        fit = 0.0
        for i, item in enumerate(self._mcts_items):
            strat = assignment.get(i)
            spec = ROLES.get(strat)
            if spec is None:
                continue
            low = item.lower()
            writes = any(w in low for w in ("write", "build",
                                            "implement", "fix"))
            runs = any(w in low for w in ("run", "test", "measure"))
            tools = set(spec.get("tools", ()))
            if writes and ("write_file" in tools or "edit_file"
                           in tools):
                fit += 1
            if runs and "run_command" in tools:
                fit += 1
        n = max(1, len(self._mcts_items))
        return min(1.0, fit / (n * 1.5))

    def _ci_runner(self, tests: list[str]) -> tuple[bool, str]:
        """Run the impacted tests through the real shell."""
        import sys
        from shlex import quote
        from .tools import run_command
        cmd = (f"{quote(sys.executable)} -m pytest -q "
               + " ".join(quote(t) for t in tests))
        out = run_command(cmd, timeout=300)
        # only trust the exit-code line the runner itself emits — never a
        # substring that could appear inside test output
        ok = any(line.strip() == "exit code: 0"
                 for line in out.splitlines())
        return (ok, out)

    def _dual_fast(self, question: str) -> str:
        """System 1: one cheap direct model call."""
        from .systemprompt import dual_fast_prompt
        result = chat_blocking(
            self.provider, self.model, self.effort,
            [{"role": "system", "content": dual_fast_prompt()},
             {"role": "user", "content": question}],
            None, timeout=60.0)
        return result.content or ""

    def _dual_slow(self, question: str) -> str:
        """System 2: the deliberate stack — a full debate tournament."""
        result = self.debate.run(question, rounds=2)
        return (f"[verified via {len(self.debate.models)}-model "
                f"debate, champion {result.champion_model}] "
                f"{result.verdict}")

    def _repair_reseed_prompts(self) -> bool:
        """Re-seal every vault prompt — cures prompt drift."""
        for name in self.mastermind.vault.names():
            try:
                self.mastermind.vault.resolve(name)
            except Exception:
                pass
        return True

    def _repair_consolidate(self) -> bool:
        """Loop alerts: consolidate the brain (dedupe/decay stale
        patterns that drive repetition)."""
        self.brain.sleep()
        return True

    def _repair_warm_caches(self) -> bool:
        """Latency: rebuild the shared context so speculative caches
        refill."""
        self.build_shared_context()
        return True

    def _subagent_runtime(self, model_id: str = "") -> tuple:
        """Resolve each child against an explicitly selected configured model.

        There is intentionally no router/fallback here: workers use the same
        configured model unless a caller supplies a known model id.
        """
        _resolve_lazy("model_by_id", "PROVIDERS")
        model = model_by_id(model_id) if model_id else self.model
        if model is None:
            raise ValueError(f"unknown model id: {model_id}")
        provider = PROVIDERS.get(model.provider)
        if provider is None:
            raise ValueError(f"provider {model.provider!r} is not configured")
        return provider, model, self.effort

    def _snapshot_subagent_write(self, worker, tool: Tool,
                                 args: dict) -> str | None:
        """Enforce the same recover-before-write invariant for child agents."""
        paths = self._snapshot_paths(tool.name, args)
        if not paths:
            return "delegated write blocked: no snapshot target was identified"
        try:
            snap = self.store.take(paths)
            self.log.append(
                "snapshot.taken",
                {"tree": snap["tree"], "paths": list(snap["paths"]),
                 "before_tool": tool.name, "subagent_id": worker.id},
                actor="kernel",
                causation_id=worker.spawn_event_id or None)
        except Exception as exc:  # fail closed: no recovery point, no write
            return ("delegated write blocked: recovery snapshot failed "
                    f"({type(exc).__name__}: {exc})")
        return None

    def _after_subagent_write(self, worker, tool: Tool, args: dict) -> str:
        """Re-check active goal anti-clauses after successful child writes."""
        try:
            violations = self.goal.check_anti_clauses()
        except Exception:
            return ""
        if not violations:
            return ""
        v = violations[0]
        return (f"WARNING: anti-clause {v['clause']} violated — {v['detail']}. "
                "The run cannot close until this is repaired or rewound.")

    def _register_subagent_tools(self) -> None:
        """Add bounded background delegation and approved parallel coding."""
        _resolve_lazy("Tool", "RISK_CONFIRM")
        from .roles import ROLES
        from .subagents import ParallelSubagents, SubagentError

        manager = ParallelSubagents(
            self.log, self.tools, self._subagent_runtime,
            before_write=self._snapshot_subagent_write,
            after_write=self._after_subagent_write,
            workspace=Path.cwd())
        self.subagents = manager

        def _task_error(fn):
            try:
                return fn()
            except (SubagentError, TypeError, ValueError) as exc:
                return f"ERROR: {exc}"

        def task(task: str, subagent_type: str = "researcher",
                 context: str = "", name: str = "",
                 model_id: str = "") -> str:
            """Start one read-only specialist in the background."""
            def start():
                worker = manager.spawn(
                    task, role=subagent_type, context=context, name=name,
                    model_id=model_id, allow_writes=False)
                return (f"Started batch {worker.batch_id}: {worker.id} "
                        f"({worker.role}, model {worker.model_id}). "
                        "Use TaskCollect with this batch_id to collect its "
                        "result or TaskList to inspect all workers.")
            return _task_error(start)

        def task_group(tasks: list, context: str = "",
                       role: str = "researcher", model_id: str = "") -> str:
            """Atomically launch up to eight parallel read-only specialists."""
            def start():
                if not tasks:
                    raise SubagentError("tasks must be a non-empty list")
                workers = manager.spawn_many(
                    tasks, role=role, context=context, model_id=model_id,
                    allow_writes=False)
                batch_id = workers[0].batch_id if workers else "none"
                return (f"Launched batch {batch_id} with {len(workers)} "
                        "parallel read-only subagent(s): "
                        + ", ".join(w.id for w in workers)
                        + ". Use TaskCollect with batch_id for all reports; "
                        "TaskList shows live status.")
            return _task_error(start)

        def task_write_group(tasks: list, context: str = "",
                             model_id: str = "") -> str:
            """Launch parallel coders, each restricted to confirmed paths."""
            def start():
                if not tasks:
                    raise SubagentError("tasks must be a non-empty list")
                workers = manager.spawn_many(
                    tasks, role="coder", context=context,
                    model_id=model_id, allow_writes=True)
                batch_id = workers[0].batch_id if workers else "none"
                return (f"Launched scoped write batch {batch_id} with "
                        f"{len(workers)} specialist(s): "
                        + ", ".join(
                            f"{w.id} ({w.role}) [{', '.join(w.write_scope)}]"
                            for w in workers)
                        + ". Their writes are snapshot-backed; use TaskOutput "
                        "or TaskCollect with batch_id to collect reports.")
            return _task_error(start)

        def task_output(task_id: str, wait_seconds: float = 30.0,
                        should_cancel=None) -> str:
            def collect():
                worker = manager.get(task_id)
                if worker is None:
                    return f"ERROR: unknown subagent: {task_id}"
                manager.wait([task_id], timeout=wait_seconds,
                             should_cancel=should_cancel)
                return manager.format_worker(worker)
            return _task_error(collect)

        def task_collect(batch_id: str = "", task_ids: list | None = None,
                         wait_seconds: float = 30.0,
                         should_cancel=None) -> str:
            """Wait once and return all reports for one named batch."""
            return _task_error(lambda: manager.collect(
                batch_id=batch_id, worker_ids=task_ids,
                timeout=wait_seconds, should_cancel=should_cancel))

        def task_list() -> str:
            return manager.format_status()

        def task_stop(task_id: str) -> str:
            def stop():
                worker = manager.stop(task_id)
                return manager.format_worker(worker)
            return _task_error(stop)

        roles = sorted(ROLES)
        read_roles = sorted(role for role, spec in ROLES.items()
                            if not spec.get("writes"))
        write_roles = sorted(role for role, spec in ROLES.items()
                             if spec.get("writes"))
        task_item = {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "role": {"type": "string", "enum": roles},
                "name": {"type": "string"},
                "context": {"type": "string"},
                "model_id": {"type": "string"},
                "priority": {"type": "integer", "minimum": -20, "maximum": 20},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 5},
                "depends_on": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Optional sibling task names or worker IDs that must finish before this task starts.",
                },
            },
            "required": ["task"],
        }
        write_item = {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "role": {"type": "string", "enum": write_roles},
                "name": {"type": "string"},
                "context": {"type": "string"},
                "model_id": {"type": "string",
                             "description": "Optional explicitly selected configured model; otherwise use the current model."},
                "write_scope": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "maxItems": 16,
                    "description": "Exact workspace files or directories "
                                   "this coder may change.",
                },
                "priority": {"type": "integer", "minimum": -20, "maximum": 20},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 5},
                "depends_on": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Optional sibling task names or worker IDs that must finish before this writer starts.",
                },
            },
            "required": ["task", "write_scope"],
        }
        self.tools["Task"] = Tool(
            "Task",
            "Start one independent read-only specialist in the background. "
            "Choose the closest role; use TaskCollect with the returned batch_id "
            "to collect its report, or TaskOutput for one worker.",
            {"type": "object", "properties": {
                "task": {"type": "string"},
                "subagent_type": {"type": "string", "enum": roles,
                                   "description": "Focused specialist: security, reviewer, debugger, planner, researcher, tester, analyst, optimizer, architect, integrator, documenter, devops, or coder."},
                "context": {"type": "string"}, "name": {"type": "string"},
                "model_id": {"type": "string"}},
             "required": ["task"]}, task)
        self.tools["TaskGroup"] = Tool(
            "TaskGroup",
            "Submit up to eight independent, individually scoped read-only specialists as one "
            "parallel batch; up to four model calls run at once and excess work "
            "queues. Use for reviews, exploration, design, or test analysis; "
            "each task can select its own role and explicit configured model; "
            "all tasks are validated before any are started. Collect the whole "
            "batch with TaskCollect.",
            {"type": "object", "properties": {
                "tasks": {"type": "array", "items": task_item,
                          "minItems": 1, "maxItems": 8},
                "context": {"type": "string"},
                "role": {"type": "string", "enum": read_roles},
                "model_id": {"type": "string"}},
             "required": ["tasks"]}, task_group)
        self.tools["TaskWriteGroup"] = Tool(
            "TaskWriteGroup",
            "Launch up to eight scoped implementation specialists (four model calls at once) with "
            "distinct, explicit workspace write_scope paths. This operation "
            "requires the existing permission gate; "
            "writers cannot use shell, delete, move, copy, or arbitrary patch tools, "
            "and every file write is preceded by a recovery snapshot. Each task "
            "can choose an appropriate write-enabled role and model; collect via TaskCollect.",
            {"type": "object", "properties": {
                "tasks": {"type": "array", "items": write_item,
                          "minItems": 1, "maxItems": 8},
                "context": {"type": "string"},
                "model_id": {"type": "string"}},
             "required": ["tasks"]}, task_write_group, risk=RISK_CONFIRM)
        self.tools["TaskOutput"] = Tool(
            "TaskOutput",
            "Wait briefly for one subagent and return its state, findings, "
            "verification, token counts, and any blocker.",
            {"type": "object", "properties": {
                "task_id": {"type": "string"},
                "wait_seconds": {"type": "number", "minimum": 0,
                                 "maximum": 90}},
             "required": ["task_id"]}, task_output,
            background_wait=True)
        self.tools["TaskCollect"] = Tool(
            "TaskCollect",
            "Wait once for a named batch (or up to eight explicit task IDs) and "
            "return every report with state, evidence, verification, usage, and blockers. "
            "If neither selector is supplied, collect the latest batch.",
            {"type": "object", "properties": {
                "batch_id": {"type": "string",
                             "description": "Batch ID returned by Task, TaskGroup, or TaskWriteGroup."},
                "task_ids": {"type": "array", "items": {"type": "string"},
                             "minItems": 1, "maxItems": 8},
                "wait_seconds": {"type": "number", "minimum": 0,
                                 "maximum": 90}},
             "additionalProperties": False}, task_collect,
            background_wait=True)
        self.tools["TaskList"] = Tool(
            "TaskList", "List retained parallel subagents and their live states.",
            {"type": "object", "properties": {}}, task_list)
        self.tools["TaskStop"] = Tool(
            "TaskStop", "Request cancellation of one queued or running subagent.",
            {"type": "object", "properties": {
                "task_id": {"type": "string"}},
             "required": ["task_id"]}, task_stop)

    def _register_advanced_tools(self) -> None:
        """Register planning, debate, recall, and self-inspection tools."""
        debate = self.debate
        brain = self.brain
        theater = self.theater
        compiler = self.compiler

        def run_debate(question: str, rounds: int = 3) -> str:
            """Run a multi-model debate tournament on a question and
            return the calibrated verdict."""
            question = str(question or "").strip()
            if not question:
                return "ERROR: question must be a non-empty string"
            try:
                rounds = max(1, min(int(rounds), 3))
            except (TypeError, ValueError):
                rounds = 3
            self._push_status(f"⚔ debate · {len(debate.models)} models, "
                              f"{rounds} rounds")
            result = debate.run(question, rounds=rounds)
            return debate.format(result)

        def brain_recall(query: str) -> str:
            """Recall knowledge from the cognitive memory (forgetting
            curve ranks what is still alive)."""
            query = str(query or "").strip()
            if not query:
                return brain.format_stats()
            block = brain.context_block(query, k=5)
            return block or "no live memories match that query"

        def inspect_why(seq: int) -> str:
            """Causal self-inspection: why did the agent do the event at
            this seq? Full evidence chain from the sealed envelope."""
            try:
                seq = int(seq)
            except (TypeError, ValueError):
                return "ERROR: seq must be an integer"
            return theater.why(seq)

        self.tools["run_debate"] = Tool(
            "run_debate",
            "Run a MULTI-MODEL debate tournament on a hard question: "
            "blind proposals, mutual critique, revision, then calibrated "
            "cluster fusion produces the verdict (with dissent). Better "
            "than any single model on contested questions.",
            {"type": "object", "properties": {
                "question": {"type": "string"},
                "rounds": {"type": "integer"}},
                "required": ["question"]},
            run_debate)
        self.tools["brain_recall"] = Tool(
            "brain_recall",
            "Recall knowledge from the cognitive memory — the four-store "
            "brain with a forgetting curve ranks what is still alive and "
            "relevant. Use before re-researching anything; it may "
            "already be known. With an empty query, returns brain stats.",
            {"type": "object", "properties": {
                "query": {"type": "string"}},
                "required": ["query"]},
            brain_recall)
        self.tools["inspect_why"] = Tool(
            "inspect_why",
            "Causal self-inspection: pass an event seq (shown in tool "
            "results and the event log) and get the full WHY chain — "
            "which user message, goal clause, or prior tool call caused "
            "it. Use it to explain or audit the agent's own behaviour.",
            {"type": "object", "properties": {
                "seq": {"type": "integer"}},
                "required": ["seq"]},
            inspect_why)

        # -- v6 frontier tools ------------------------------------------------

        def verify_plan(goal: str) -> str:
            """Compile a goal into a plan, then formally verify it
            against temporal safety properties before any execution."""
            goal = str(goal or "").strip()
            if not goal:
                return "ERROR: goal must be a non-empty string"
            plan = compiler.compile(goal)
            r = self.formal.verify_plan(plan.waves)
            verdict = "PASS ✓" if r.ok else "REJECTED ✗"
            head = (f"FORMAL VERIFICATION — {verdict} "
                    f"({r.checked} trace(s) checked)")
            return "\n".join([head, compiler.format(plan)]
                             + [f"  ⚠ {v['property']}: {v['why']}"
                                for v in r.violations])

        def mcts_solve(goal: str, strategies: list | None = None
                       ) -> str:
            """Tree-search the best strategy assignment: split the goal
            into work items, search (MCTS) which approach should handle
            each, return the winning assignment."""
            goal = str(goal or "").strip()
            if not goal:
                return "ERROR: goal must be a non-empty string"
            items = [s.strip() for s in goal.split(";") if s.strip()]
            strats = ([str(s) for s in strategies if str(s).strip()]
                      if isinstance(strategies, list) and strategies
                      else ["coder", "architect", "debugger", "tester"])
            self._mcts_items = items
            report = self.mcts.search(items, strats,
                                      iterations=120, deadline_s=20.0)
            lines = [f"MCTS — best score {report.best_score:.2f} in "
                     f"{report.iterations} iterations "
                     f"({report.nodes} nodes)"]
            for i, item in enumerate(items):
                lines.append(f"  [{report.best_assignment.get(i, '?')}] "
                             f"{item[:70]}")
            return "\n".join(lines)

        def synthesize_tool(name: str, description: str,
                            examples: list) -> str:
            """Write a NEW deterministic Python tool: draft from the
            spec, validate the AST, test it against every example, and
            register it as a real callable tool."""
            if not isinstance(examples, list) or not examples:
                return ("ERROR: examples must be a non-empty list of "
                        '{"args": {...}, "want": ...}')
            spec = SynthSpec(name=str(name or ""),
                             description=str(description or ""),
                             examples=examples)
            result = self.synth.synthesize(spec)
            return ("✓ " + result.reason if result.ok
                    else "ERROR: " + result.reason)

        def predict_impact(path: str) -> str:
            """Predict what breaks if a file changes — learned from
            dependencies AND real historical breakage rates."""
            impact = self.world.predict_impact(str(path or ""))
            return impact.format()

        def knowledge_ask(subject: str, predicate: str,
                          at: float | None = None) -> str:
            """Query the bitemporal knowledge graph: the live truth, or
            as of any past moment."""
            hits = self.fabric.query(str(subject), str(predicate), at)
            if not hits:
                return f"no live fact for {subject} {predicate}"
            return "\n".join(f.to_dict().__str__() for f in hits)

        self.tools["verify_plan"] = Tool(
            "verify_plan",
            "FORMALLY VERIFY a plan: compile the goal, then model-check "
            "every execution the plan permits against temporal safety "
            "properties (snapshot-before-write, write serialisation, "
            "verification-after-write). A violated plan is REJECTED "
            "with a counterexample — use before executing risky plans.",
            {"type": "object", "properties": {
                "goal": {"type": "string"}},
                "required": ["goal"]},
            verify_plan)
        self.tools["mcts_solve"] = Tool(
            "mcts_solve",
            "Monte Carlo Tree Search over strategies: split a goal by "
            "semicolons into work items and search which approach "
            "(coder/architect/debugger/tester or custom strategy ids) "
            "should handle each — UCB1 exploration, rollout scoring, "
            "best assignment returned.",
            {"type": "object", "properties": {
                "goal": {"type": "string",
                         "description": "items separated by ;"},
                "strategies": {"type": "array",
                               "items": {"type": "string"}}},
                "required": ["goal"]},
            mcts_solve)
        self.tools["synthesize_tool"] = Tool(
            "synthesize_tool",
            "WRITE A NEW TOOL: given a snake_case name, a description, "
            "and example cases ({'args': {...}, 'want': expected}), a "
            "pure deterministic Python function is drafted, AST-"
            "validated, example-tested, and registered as a REAL "
            "callable tool. Use when no existing tool fits.",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "examples": {"type": "array",
                             "items": {"type": "object"}}},
                "required": ["name", "description", "examples"]},
            synthesize_tool)
        self.tools["predict_impact"] = Tool(
            "predict_impact",
            "Predict what breaks if a file changes: learned dependency "
            "edges × historical breakage rates, ranked by probability. "
            "Use BEFORE risky edits to see the blast radius.",
            {"type": "object", "properties": {
                "path": {"type": "string"}},
                "required": ["path"]},
            predict_impact)
        self.tools["knowledge_ask"] = Tool(
            "knowledge_ask",
            "Query the bitemporal knowledge graph — facts with validity "
            "windows. Ask for the live truth or 'as of' any past time "
            "('at' = unix timestamp). Use it to remember project facts "
            "that change over time.",
            {"type": "object", "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "at": {"type": "number"}},
                "required": ["subject", "predicate"]},
            knowledge_ask)

    def _register_persisted_skills(self) -> None:
        """Load previously forged skills from disk and expose them as tools.
        Only skills that re-pass the safety scan are loaded."""
        try:
            self.skill_forge.load_persisted()
        except Exception:
            return
        for name, skill in self.skill_forge.registry.items():
            if name in self.tools:
                continue
            self._expose_skill(name, skill)

    def _expose_skill(self, name: str, skill) -> None:
        """Wrap a validated skill as a live Tool in the registry."""
        try:
            namespace = self.skill_forge._load(skill)
        except Exception:
            return
        fn = namespace.get(skill.entry or name)
        if not callable(fn):
            return

        def handler(_fn=fn, **kwargs):
            try:
                return str(_fn(**kwargs))
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"

        props = skill.parameters or {}
        schema = {"type": "object",
                  "properties": {k: (v if isinstance(v, dict)
                                     else {"type": "string"})
                                 for k, v in props.items()}}
        self.tools[name] = Tool(name, skill.description or name,
                                schema, handler)

    # -- shared session context ------------------------------------------------

    def build_shared_context(self, max_chars: int = 4000) -> str:
        _resolve_lazy("fold")
        """Build a concise read-only summary of the current session state."""
        parts: list[str] = [f"Working directory: {Path.cwd()}"]
        try:
            entries = sorted(p.name + ("/" if p.is_dir() else "")
                             for p in Path.cwd().iterdir()
                             if not p.name.startswith("."))[:40]
            if entries:
                parts.append("Directory listing: " + ", ".join(entries))
        except OSError:
            pass
        goal = self.goal.status()
        if goal.active:
            parts.append("Active goal: " + goal.statement)
            for c in goal.clauses:
                parts.append(f"  clause {c.id} [{c.state}]: {c.text}")
        st = fold(self.log)
        if st.files_touched:
            parts.append("Files touched this session: "
                         + ", ".join(sorted(st.files_touched)[:20]))
        facts = [f.get("fact", "") for f in st.facts][-8:]
        if facts:
            parts.append("Known facts:\n- " + "\n- ".join(facts))
        episodes = st.episodes[-3:]
        for ep in episodes:
            parts.append(f"Past episode: {ep.get('goal', '')} -> "
                         f"{ep.get('outcome', '')}")
        ctx = "\n".join(parts)
        return ctx[:max_chars]

    # -- persistence -------------------------------------------------------

    def save_session(self) -> Path | None:
        _resolve_lazy("config")
        try:
            config.ensure_dirs()
            path = config.SESSIONS_DIR / f"{self.session_id}.json"
            # snapshot the messages once — the turn thread keeps mutating
            # self.messages and json.dumps iterating a growing list raises
            messages = list(self.messages)
            data = json.dumps({
                "session_id": self.session_id,
                "model_id": self.cfg.model_id,
                "saved_at": datetime.now().isoformat(),
                "messages": messages,
            }, indent=1)
            # atomic write: never leave a torn/truncated session file
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(data)
            os.replace(tmp, path)
            return path
        except (OSError, ValueError, TypeError, RuntimeError):
            return None


# ---------------------------------------------------------------------------
# Self-test (worker 19/20): bounded-memory proof for the agent's runtime
# data structures. Run with FULLAGENT_HOME pointed at a scratch dir:
#   FULLAGENT_HOME=$(mktemp -d) python3 -m fullagent.agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    from .config import Config
    from .kernel import EventLog as _EventLog

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    agent = Agent(Config())
    # isolate: swap the real event log for a temp-dir one (subsystems keep
    # their own refs, but every agent-level append below goes here)
    agent.log.close()
    _td = tempfile.mkdtemp(prefix="agent_mem_selftest_")
    agent.log = _EventLog(Path(_td) / "events.jsonl",
                          session=agent.session_id)

    # 1. focus history: 200 ticks through the REAL focus_continue path —
    # stall detection only reads the last 4, so the list must stay capped
    fake_turn = SimpleNamespace(error="", tools=[SimpleNamespace()])
    for _ in range(200):
        agent.focus_continue(fake_turn, 10)
    check(f"focus_history bounded at {_FOCUS_HISTORY_MAX} "
          f"(got {len(agent._focus_history)})",
          len(agent._focus_history) <= _FOCUS_HISTORY_MAX)
    check("focus history keeps the newest ticks",
          agent._focus_history[-1] == 1.0)

    # 2. file-hash oscillation map: touch 300+ distinct files through the
    # REAL _execute_tool path (write_file) — the per-path lists were
    # already capped at 20; the path COUNT must now stay capped too
    n_paths = _FILE_HASH_PATHS_MAX + 60
    for i in range(n_paths):
        p = Path(_td) / f"f{i}.txt"
        ev = ToolEvent(name="write_file",
                       args={"path": str(p), "content": f"v{i}"})
        agent._execute_tool(ev, approve=lambda t, a: True,
                            on_status=lambda s: None)
        assert ev.status == "done", (ev.name, ev.status, ev.result[:100])
    check(f"file_hash paths bounded at {_FILE_HASH_PATHS_MAX} "
          f"(got {len(agent._file_hashes)})",
          len(agent._file_hashes) <= _FILE_HASH_PATHS_MAX)
    check("newest path still tracked",
          str(Path(_td) / f"f{n_paths - 1}.txt") in agent._file_hashes)
    check("per-path hash lists still capped at 20",
          all(len(h) <= 20 for h in agent._file_hashes.values()))

    # 3. EventLog window (kernel): the log must not hold every event in RAM
    check("event log windowed in RAM",
          len(agent.log._events) <= 2000 + 64)

    agent.log.close()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        raise SystemExit(1)
    print("AGENT MEMORY SELF-TEST PASS")

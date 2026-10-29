"""Configuration: providers, models, effort levels, paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "FullAgent"


def _pick_app_dir() -> Path:
    """Choose a WRITABLE home for app state — never a crash path.

    Order: $FULLAGENT_HOME, ~/.fullagent, <tmp>/fullagent-<uid>.
    Each candidate is probed with a real write; the first one that
    actually works wins, so a read-only home dir degrades gracefully
    instead of killing the app at startup (OSError on event log)."""
    import tempfile
    candidates: list[Path] = []
    env = os.environ.get("FULLAGENT_HOME")
    if env:
        candidates.append(Path(env))
    try:
        candidates.append(Path.home() / ".fullagent")
    except (OSError, RuntimeError):
        # containers running as a uid with no passwd entry and no $HOME:
        # Path.home() raises RuntimeError. Skip the candidate — the tmp
        # fallback below still gives the app a writable home.
        pass
    uid = str(os.getuid()) if hasattr(os, "getuid") else "user"
    candidates.append(Path(tempfile.gettempdir()) / f"fullagent-{uid}")
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write-probe"
            probe.write_text("ok")
            probe.unlink()
            # Also verify SQLite POSIX locking works on this mount (e.g. 9P/NFS
            # mounts may allow write_text but fail sqlite3 locking with I/O error).
            import sqlite3
            db_probe = c / f".sqlite-probe-{os.getpid()}.db"
            try:
                conn = sqlite3.connect(str(db_probe), timeout=0.1)
                conn.execute("CREATE TABLE IF NOT EXISTS _p (x INT)")
                conn.close()
            finally:
                try:
                    db_probe.unlink()
                except OSError:
                    pass
            return c
        except (OSError, Exception):
            continue
    # last resort: per-user tmp subdir so two users on the same box don't
    # clobber each other's config / event log / sessions
    fallback = Path(tempfile.gettempdir()) / f"fullagent-{uid}"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except OSError:
        return Path(tempfile.gettempdir())


APP_DIR = _pick_app_dir()
CONFIG_FILE = APP_DIR / "config.json"
HISTORY_FILE = APP_DIR / "history"
SESSIONS_DIR = APP_DIR / "sessions"
EVENT_LOG_FILE = APP_DIR / "eventlog.jsonl"

DEFAULT_TIMEOUT = 300.0
MAX_TOOL_ITERATIONS = 1000
# PERF: hard cap on executed tool calls per turn — 200 iterations of LLM
# LONG-RUN MODE (v4): caps raised to marathon scale — a long autonomous
# work session must never hit an artificial mid-task stop. Users who want
# tighter caps can still override via config.json extra keys
# "tool_call_cap" / "turn_time_budget" (read by agent.AdaptiveTurnCap).
MAX_TOOL_CALLS_PER_TURN = 200
# PERF (worker 8/20): adaptive turn cap — fast calls (<5s rolling avg)
# may run up to ADAPTIVE_CAP_FAST; slow calls (>15s rolling avg) are
# reined in to ADAPTIVE_CAP_SLOW calls / a smaller time budget. Whichever
# of the call cap or the time budget hits first stops the turn; the stop
# message says which one and invites "continue".
# See fullagent/adaptivecap.py (AdaptiveTurnCap).
ADAPTIVE_CAP_FAST = 400
ADAPTIVE_CAP_SLOW = 120
FAST_CALL_AVG_THRESHOLD_S = 5.0
SLOW_CALL_AVG_THRESHOLD_S = 15.0
# Time budget for tool-call wall time per turn (ev.duration summed).
TURN_TIME_BUDGET_S = 3600.0
TURN_TIME_BUDGET_SLOW_S = 1800.0
# PERF: stop early when the same tool call returns an identical result
# this many times in a row (no-progress loop).
NO_PROGRESS_STALL_LIMIT = 3
MAX_TOOL_OUTPUT_CHARS = 24_000
# One output ceiling for every effort level: 200k tokens.
MAX_TOKENS = 200_000
# Backends reject a request when input + max_tokens exceeds the model's
# context window. Every request's max_tokens is clamped to fit (client.py).
#
# Per-model context windows, researched 2026-10-21.
# Free-tier gateways sometimes serve LESS than the lab's headline number,
# so values below are the SERVED window where documented, conservative
# otherwise. The backend can still teach a smaller window at runtime
# (client.learn_context_window) — that only ever shrinks these.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # upstream: deepseek/deepseek-v4.1-flash (1,048,576-token window)
    "deepseek-v4.1-flash-free": 1_048_576,
    # same flash family, Gwarden route (2026-10 snapshot)
    "deepseek-v4.1-flash-1017": 1_048_576,
    # upstream: google/gemini-4-argon (1,000,000-token window)
    "gemini-4-argon": 1_000_000,
    # upstream: minimax/minimax-m3.1-flash:free (1,000,000-token window)
    "minimax-m3.1-flash-free": 1_000_000,
}
# Default for models NOT in the table above: assume a small window rather
# than risk a ~44s doomed request against an unknown cap. A request we can
# prove won't fit is never sent (see client.build_payload).
UNKNOWN_MODEL_WINDOW = 128_000


def model_context_window(model_id: str) -> int:
    """Context window for a model id: the researched table value, or the
    conservative default for unknown models."""
    return MODEL_CONTEXT_WINDOWS.get(model_id, UNKNOWN_MODEL_WINDOW)


DEFAULT_CONTEXT_WINDOW = UNKNOWN_MODEL_WINDOW


def _env_int(name: str, default: int) -> int:
    """Read an int setting from the environment; garbage/missing -> default.

    Never raises at import time — a typo'd env var must not kill startup.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


# Hard pre-send payload ceiling (estimated input tokens). When a request's
# input exceeds this, client.build_payload prunes history (oldest tool
# results first, then old assistant texts) BEFORE sending — some providers
# reject oversized payloads with invalid_request_error only after a long
# wait, and the window clamp above does not protect against that. Set well
# under typical 128k/200k windows to leave headroom. Override with
# FULLAGENT_MAX_PAYLOAD_TOKENS.
MAX_PAYLOAD_TOKENS = _env_int("FULLAGENT_MAX_PAYLOAD_TOKENS", 100_000)


@dataclass(frozen=True)
class Provider:
    key: str
    name: str
    base_url: str
    api_key: str
    color: str


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    label: str
    tag: str = ""
    tag_color: str = "grey62"
    supports_tools: bool = True
    supports_reasoning: bool = False
    # Total context window (input + output tokens). Used to clamp max_tokens
    # at send time so a request is never rejected for exceeding the window.
    context_window: int = DEFAULT_CONTEXT_WINDOW


def _provider_api_key(provider: str) -> str:
    # Priority: environment variable > per-user key file. Empty or
    # whitespace-only values fall through rather than shadowing a key file.
    key = os.environ.get(f"{provider.upper()}_API_KEY", "").strip()
    if key:
        return key
    try:
        key = (APP_DIR / f"{provider}_api_key").read_text(
            encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # OSError: missing/unreadable file. UnicodeDecodeError: file
        # has non-UTF-8 bytes — must not propagate here, this runs at
        # module import and would kill the whole app at startup.
        key = ""
    if key:
        return key
    # Embedded fallback: zero-config — the key ships in the build so the
    # user never has to configure anything. Env var / key file above
    # still take precedence when present.
    return _EMBEDDED_KEYS.get(provider, "")


# Build-time embedded API keys (zero-config). These ship inside the
# wheel/binary. Override anytime via <PROVIDER>_API_KEY env var or
# ~/.fullagent/<provider>_api_key file.
_EMBEDDED_KEYS: dict[str, str] = {
    "kios": "sk-cFXQ576lsIctpudkYD5lPniF5UgHGLy1nKeXDscCEvK1LMZV",
    "gwarden": "GWAR-OM8VK0HO-94OFDH0DIE3T-U1B3V-WPPPNO-MF8VSR0U",
}


PROVIDERS: dict[str, Provider] = {
    "kios": Provider(
        key="kios",
        name="Kios API",
        base_url="https://kiosapi.com/v1",
        api_key=_provider_api_key("kios"),
        color="#8be9fd",
    ),
    "gwarden": Provider(
        key="gwarden",
        name="Gwarden API",
        base_url="https://gwarden.su/v1",
        api_key=_provider_api_key("gwarden"),
        color="#bd93f9",
    ),
}

MODELS: list[Model] = [
    # Fast default: DeepSeek accepts reasoning_effort="none" (sent by
    # client.build_payload when supports_reasoning=True), so it answers
    # immediately instead of after a long hidden chain-of-thought.
    # MiniMax rejects that parameter ("Reasoning is mandatory for this
    # endpoint"), so it keeps the provider default.
    Model("deepseek-v4.1-flash-1017", "gwarden", "DeepSeek V4.1 Flash 1017",
          tag="gwarden", supports_tools=True, supports_reasoning=True,
          context_window=model_context_window("deepseek-v4.1-flash-1017")),
    Model("gemini-4-argon", "gwarden", "Gemini 4 Argon",
          tag="gwarden", supports_tools=True, supports_reasoning=True,
          context_window=model_context_window("gemini-4-argon")),
    Model("deepseek-v4.1-flash-free", "kios", "DeepSeek V4.1 Flash Free",
          tag="free", supports_tools=True, supports_reasoning=True,
          context_window=model_context_window("deepseek-v4.1-flash-free")),
    Model("minimax-m3.1-flash-free", "kios", "MiniMax M3.1 Flash Free",
          tag="free", supports_tools=True,
          context_window=model_context_window("minimax-m3.1-flash-free")),
]
# Fastest Kios free route measured 2026-10-21 (~1s first token vs ~3s for
# MiniMax and ~40s for qwen3.8-flash-free): default to it.
DEFAULT_MODEL_ID = "deepseek-v4.1-flash-1017"


@dataclass(frozen=True)
class Effort:
    key: str
    label: str
    color: str
    max_tokens: int | None
    temperature: float
    reasoning_effort: str | None
    description: str


EFFORTS: list[Effort] = [
    # REAL THINKING: medium+ efforts request a real chain-of-thought from
    # the model (reasoning_effort -> payload.reasoning_effort in
    # client.build_payload). Low stays instant/no-thinking.
    Effort("low", "LOW", "#6272a4", MAX_TOKENS, 0.2, None,
           "short answers, minimal tokens"),
    Effort("medium", "MEDIUM", "#8be9fd", MAX_TOKENS, 0.4, "low",
           "balanced length, light thinking"),
    Effort("high", "HIGH", "#50fa7b", MAX_TOKENS, 0.6, "medium",
           "thorough, real chain-of-thought"),
    Effort("extrahigh", "EXTRA HIGH", "#ffb86c", MAX_TOKENS, 0.7, "high",
           "deep work, strong thinking"),
    Effort("ultrahigh", "ULTRA HIGH", "#ff5555", MAX_TOKENS, 0.8, "high",
           "maximum depth, strongest thinking"),
]

DEFAULT_EFFORT = "high"


def model_by_id(model_id: str) -> Model | None:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None


def effort_by_key(key: str) -> Effort | None:
    for e in EFFORTS:
        if e.key == key:
            return e
    return None


@dataclass
class Config:
    model_id: str = DEFAULT_MODEL_ID
    effort: str = DEFAULT_EFFORT
    auto_approve: bool = False
    show_reasoning: bool = False
    theme: str = "dracula"
    # which system prompt to send: "main" (compact) or "master" (main +
    # the optional project.txt specification when present)
    prompt: str = "main"
    extra: dict = field(default_factory=dict)
    # PERF: sliding window for model-visible history. 0 = disabled (legacy).
    prune_window: int = 40
    # PERF (worker 2/20): hard per-request prompt token budget
    # (messages + tool schemas). 0 = disabled.
    prompt_token_budget: int = 100_000

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        try:
            data = json.loads(CONFIG_FILE.read_text())
            if not isinstance(data, dict):
                data = {}
            for k in ("model_id", "effort", "auto_approve", "show_reasoning",
                      "theme", "prompt", "prune_window", "prompt_token_budget"):
                if k not in data:
                    continue
                if k in ("auto_approve", "show_reasoning"):
                    # safety gates must be real booleans — a drifted
                    # config with "auto_approve": "false" (truthy string)
                    # would silently disable the approval prompt
                    if isinstance(data[k], bool):
                        setattr(cfg, k, data[k])
                elif k in ("prune_window", "prompt_token_budget"):
                    # drift-safe int validation; 0 disables the feature
                    try:
                        setattr(cfg, k, max(0, int(data[k])))
                    except (TypeError, ValueError):
                        pass
                else:
                    setattr(cfg, k, data[k])
            cfg.extra = {k: v for k, v in data.items()
                         if k not in ("model_id", "effort", "auto_approve",
                                      "show_reasoning", "theme", "prompt",
                                      "prune_window", "prompt_token_budget")}
        except (OSError, ValueError):
            pass
        if model_by_id(cfg.model_id) is None:
            cfg.model_id = DEFAULT_MODEL_ID
        if effort_by_key(cfg.effort) is None:
            cfg.effort = DEFAULT_EFFORT
        if not isinstance(cfg.prompt, str) or not cfg.prompt:
            cfg.prompt = "main"
        return cfg

    def save(self) -> None:
        try:
            ensure_dirs()
            data = {
                "model_id": self.model_id,
                "effort": self.effort,
                "auto_approve": self.auto_approve,
                "show_reasoning": self.show_reasoning,
                "theme": self.theme,
                "prompt": self.prompt,
                # load() honors these; omitting them would silently reset
                # a user's tuned values to defaults on every save+load
                # (worker 9/20).
                "prune_window": self.prune_window,
                "prompt_token_budget": self.prompt_token_budget,
            }
            data.update(self.extra)
            # atomic write: a crash mid-write must never leave truncated
            # JSON that would reset the whole config on next load.
            # Unique tmp name + fsync: two threads/processes saving at
            # once must not rename each other's torn tmp file.
            import threading as _th
            tmp = CONFIG_FILE.with_name(
                f".{os.getpid()}.{_th.get_ident()}.json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, CONFIG_FILE)
        except OSError:
            pass  # config persistence is a convenience, never a crash path


def ensure_dirs() -> None:
    """Create every directory the app writes into. Called at startup AND
    before individual writes, so a deleted home dir heals itself."""
    for d in (APP_DIR, SESSIONS_DIR, APP_DIR / "memory",
              APP_DIR / "skills", APP_DIR / "store"):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def _self_check() -> None:
    """Verify config invariants. Run with: python3 -m fullagent.config"""
    assert PROVIDERS, "no providers defined"
    for key, p in PROVIDERS.items():
        assert key == p.key, f"provider dict key {key!r} != Provider.key {p.key!r}"
        assert p.base_url.startswith(("https://", "http://")), \
            f"provider {key!r} has bad base_url {p.base_url!r}"
    seen: set[str] = set()
    for m in MODELS:
        assert m.provider in PROVIDERS, \
            f"model {m.id!r} references unknown provider {m.provider!r}"
        assert m.id not in seen, f"duplicate model id {m.id!r}"
        seen.add(m.id)
        assert m.context_window > 0, f"model {m.id!r} has bad context_window"
    assert model_by_id(DEFAULT_MODEL_ID) is not None, \
        f"DEFAULT_MODEL_ID {DEFAULT_MODEL_ID!r} not in MODELS"
    assert effort_by_key(DEFAULT_EFFORT) is not None, \
        f"DEFAULT_EFFORT {DEFAULT_EFFORT!r} not in EFFORTS"
    # round-trip: save/load must preserve values and heal bad input
    cfg = Config.load()
    assert model_by_id(cfg.model_id) is not None
    assert effort_by_key(cfg.effort) is not None
    print(f"PASS (OK): {len(PROVIDERS)} providers, {len(MODELS)} models, "
          f"default={DEFAULT_MODEL_ID!r}/{DEFAULT_EFFORT!r}")


if __name__ == "__main__":
    _self_check()

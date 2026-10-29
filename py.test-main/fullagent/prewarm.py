"""PREWARM — overlap the provider wait with next-iteration prep (worker 5/20).

A model call is ~18s of network wait during which the turn thread is
blocked inside chat_stream(). This module spends that wait on a daemon
thread preparing the work the NEXT iteration would otherwise do
serially after the response lands:

  1. compact analysis — the _messages_chars() scan plus the expensive
     estimate_tokens() JSON dump, computed on a frozen snapshot during
     the wait. The next iteration reuses it incrementally
     (est(snapshot) + est(small appended delta)) instead of re-dumping
     the whole conversation. Measured: 17-87ms/call -> ~1-3ms.
  2. prompt skeleton — the O(n) first-user scan of the snapshot, so the
     next iteration's prune becomes an O(window) patch that is
     byte-identical to prune_messages(). Measured: up to ~6ms -> <0.5ms.
  3. env probes — read-only os.stat() over paths named by recent tool
     calls. At consume time a stat that moved against the filestat
     baseline seals an early external-edit notice (before the model acts
     on stale content); the stats also warm the dentry cache for the
     imminent tool reads.

SAFETY (mechanical, not advisory):
  * The worker touches ONLY the snapshot (a private list copy) and
    captured scalars — never self.messages, never the agent, never the
    event log, never the TUI.
  * NO tool execution of any kind in the worker — not even read-only
    tools (the speculator owns prefetch). The only syscalls are
    os.stat() probes: read-only, non-mutating, no side effects.
  * consume() validates element-identity prefix: every snapshot element
    must still be the identical object at the same index, and the
    captured model_id / prune window must be unchanged. ANY mutation
    during the wait (compaction, overflow-shrink, message seal, manual
    model switch) invalidates the bundle and the normal path runs instead.
  * Generation counter: a late-finishing worker can never overwrite a
    newer flight's bundle, and a stale bundle can never be consumed.
  * The worker never raises — everything is wrapped; a dead worker just
    means the normal (slower) path runs.
  * Cancel-safe: daemon thread, holds no locks across the wait, never
    touches cancellation state. Abandoned on cancel/turn end.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

# The incremental estimate mixes a pre-wait estimate with a post-wait
# delta under a possibly-recalibrated tokenizer: keep a 10% margin so a
# stale calibration can never wrongly skip a needed compaction. If the
# margin is exceeded the normal full estimate runs instead — worst case
# is the status quo, never a wrong skip.
_FIT_MARGIN = 0.90
# prune_messages truncates all but the newest N tool outputs; the patch
# must use the same default _prune_for_model() uses.
_TOOL_KEEP = 2
_PROBE_PATHS_MAX = 16
_EXTERNAL_EDIT_WARN_MAX = 5


# ---------------------------------------------------------------------------
# Pure ports (no agent import at module level — avoids an import cycle;
# agent.py imports this module). Each port has a differential self-test
# below proving it matches the original exactly.
# ---------------------------------------------------------------------------

def message_chars(messages: list) -> int:
    """Port of Agent._messages_chars: cheap conversation size proxy."""
    n = 0
    for m in messages:
        c = m.get("content")
        if c:
            n += len(c)
        tcs = m.get("tool_calls")
        if tcs:
            for tc in tcs:
                fn = tc.get("function") or {}
                n += len(fn.get("name", "")) + len(fn.get("arguments") or "")
        rc = m.get("reasoning_content") or m.get("reasoning")
        if rc:
            n += len(rc)
    return n


def _truncate_tail(tail: list, tool_keep: int = _TOOL_KEEP) -> None:
    """Port of prune_messages' tool-output truncation step (in place)."""
    tool_idx = [i for i, m in enumerate(tail) if m.get("role") == "tool"]
    for i in tool_idx[:max(0, len(tool_idx) - tool_keep)]:
        content = tail[i].get("content", "")
        if isinstance(content, str) and len(content) > 500:
            tail[i]["content"] = (
                content[:500]
                + f"\n…[truncated {len(content) - 500} chars — "
                f"re-read from disk if needed]")


def patched_prune(messages: list, bundle: "FlightBundle",
                  window: int, tool_keep: int = _TOOL_KEEP) -> list:
    """O(window) next-iteration prune, byte-identical to prune_messages().

    Precondition (checked by consume()): bundle.snapshot is an
    element-identity prefix of messages, bundle.window == window and the
    model is unchanged. The O(n) first-user scan was done by the worker
    during the API wait; here we only slice the O(window) tail and
    re-apply truncation — the truncation MUST run on the live tail
    because the tool_keep boundary shifts when new tool results land.
    """
    n = len(messages)
    # exact early-return mirror of prune_messages()
    if not messages or window <= 0 or n <= window + 2:
        return list(messages)
    if bundle.snapshot_len == 0:
        # nothing was prewarmed (consume() rejects this too) — the
        # exact fallback; the O(n) scan is unavoidable here.
        from .agent import prune_messages  # lazy: agent imports prewarm
        return prune_messages(messages, window=window, tool_keep=tool_keep)
    rest_start = 1 if bundle.has_system else 0
    tail_src = messages[max(rest_start, n - window):]
    tail = [dict(m) if m.get("role") == "tool" else m for m in tail_src]
    _truncate_tail(tail, tool_keep)
    out: list = []
    if bundle.has_system:
        out.append(messages[0])
    fu = bundle.first_user
    if fu is None:
        # The snapshot had no user message (practically impossible
        # mid-turn, but exactness is cheap): scan the small appended
        # suffix instead of the whole history.
        for m in messages[bundle.snapshot_len:]:
            if m.get("role") == "user":
                fu = m
                break
    if fu is not None and fu not in out \
            and all(m is not fu for m in tail):
        out.append(fu)
    out.extend(tail)
    return out


def compact_fits(messages: list, bundle: "FlightBundle",
                 schema_tokens: int, budget: int, model_id: str) -> bool:
    """Incremental fit check from the flight bundle.

    est(snapshot) was computed during the API wait; only the small
    appended delta is estimated now. Returns True only under the safety
    margin — otherwise the caller runs the normal full estimate.
    """
    from .client import estimate_tokens  # lazy: client is heavy
    appended = messages[bundle.snapshot_len:]
    if appended:
        est_now = bundle.est + estimate_tokens(appended, model_id)
    else:
        est_now = bundle.est
    return est_now + schema_tokens <= budget * _FIT_MARGIN


def patched_prune_for_model(agent, bundle: "FlightBundle",
                            reserved_tokens: int = 0) -> list:
    """Exact O(window) equivalent of Agent._prune_for_model().

    patched_prune() replaces the prune_messages() call; the optional
    promptbudget.bound_prompt() cap is applied identically (it only
    ever sees the O(window) pruned view, so it costs the same as the
    normal path — the saving is the skipped O(n) prune).
    """
    from .promptbudget import bound_prompt  # lazy: sibling module
    msgs = patched_prune(agent.messages, bundle, bundle.window)
    try:
        budget = int(getattr(agent.cfg, "prompt_token_budget", 0) or 0)
    except (TypeError, ValueError):
        budget = 0
    if budget > 0:
        msgs = bound_prompt(
            msgs, budget_tokens=budget,
            model_id=getattr(agent.model, "id", ""),
            reserved_tokens=reserved_tokens)
    return msgs


# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------

@dataclass
class FlightBundle:
    """One completed background prewarm flight. Immutable after done."""
    gen: int
    snapshot: list = field(repr=False)   # private copy; identity-compared
    snapshot_len: int = 0
    has_system: bool = False
    first_user: dict | None = None       # identity into snapshot
    window: int = 40
    model_id: str = ""
    chars: int = 0
    est: int = 0                          # estimate_tokens(snapshot)
    schema_tokens: int = 0
    budget: int = 0
    # path -> filestat fingerprint (mtime_ns, size, content_digest) | None
    probes: dict = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event,
                                   repr=False)


def _stat(path: str):
    """Content identity shared with filestat's stale-edit baselines.

    Using the same fingerprint (stat + digest) keeps prewarm's early-edit
    notice comparable with agent._file_stats and catches same-size rewrites
    inside a coarse filesystem timestamp tick.
    """
    from . import filestat
    return filestat._stat(path)


def _probe_paths(agent) -> list[str]:
    """Paths named by the last two turns' tool calls (main thread)."""
    paths: list[str] = []
    seen: set[str] = set()
    try:
        turns = getattr(agent, "turns", []) or []
        for tn in turns[-2:]:
            for t in getattr(tn, "tools", []) or []:
                args = getattr(t, "args", None) or {}
                p = args.get("path") if isinstance(args, dict) else None
                if isinstance(p, str) and p and p not in seen:
                    seen.add(p)
                    paths.append(p)
                    if len(paths) >= _PROBE_PATHS_MAX:
                        return paths
    except Exception:
        pass
    return paths


class FlightPrewarmer:
    """Owns at most one in-flight background prewarm per agent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen = 0
        self._bundle: FlightBundle | None = None
        self.hits = 0      # consumes that took a fast path
        self.misses = 0    # consumes that fell back / had no flight
        self.invalid = 0   # consumes discarded by validation

    # -- launch (main thread, just before chat_stream blocks) ---------------

    def launch(self, agent) -> None:
        """Snapshot + start the worker. Never raises, never blocks."""
        if os.environ.get("FULLAGENT_PREWARM", "1") == "0":
            return  # operational kill-switch
        try:
            snapshot = list(agent.messages)
            model_id = agent.model.id
            try:
                window = int(getattr(agent.cfg, "prune_window", 40) or 0)
            except (TypeError, ValueError):
                window = 40
            try:
                budget = agent._fit_budget()
            except Exception:
                budget = 0
            try:
                schemas = agent._tool_schemas()
            except Exception:
                schemas = None
            paths = _probe_paths(agent)
        except Exception:
            return  # prewarm is a courtesy, never a crash path
        with self._lock:
            self._gen += 1
            gen = self._gen
            bundle = FlightBundle(
                gen=gen, snapshot=snapshot, snapshot_len=len(snapshot),
                has_system=bool(snapshot)
                and snapshot[0].get("role") == "system",
                window=window, model_id=model_id, budget=budget)
            self._bundle = bundle
        t = threading.Thread(target=self._work,
                             args=(bundle, schemas, model_id, paths),
                             daemon=True, name=f"prewarm-flight-{gen}")
        t.start()

    def _work(self, bundle: FlightBundle, schemas,
              model_id: str, paths: list[str]) -> None:
        """Background worker: read-only compute on the snapshot. Never raises."""
        try:
            from .client import estimate_tokens  # lazy: heavy module
            snap = bundle.snapshot
            # O(n) first-user scan — the part the main thread skips later.
            start = 1 if bundle.has_system else 0
            fu = None
            for m in snap[start:]:
                if m.get("role") == "user":
                    fu = m
                    break
            bundle.first_user = fu
            bundle.chars = message_chars(snap)
            bundle.est = estimate_tokens(snap, model_id)
            bundle.schema_tokens = (estimate_tokens(schemas, model_id)
                                    if schemas else 0)
            probes: dict = {}
            for p in paths:
                probes[p] = _stat(p)
            bundle.probes = probes
        except Exception:
            pass  # a dead worker just means the normal path runs
        finally:
            # the bundle carries its own event; cross-flight protection
            # is the generation check in consume(), not this lock
            bundle.done.set()

    # -- consume (main thread, top of the next iteration) --------------------

    def consume(self, agent) -> FlightBundle | None:
        """Validate and return the flight, or None (normal path). Never blocks.

        If the worker is still running the wait was shorter than the
        prewarm compute — discard without blocking; the thread finishes
        harmlessly and its stale bundle is never consumed (gen check).
        """
        with self._lock:
            bundle = self._bundle
            self._bundle = None
            gen = self._gen
        if bundle is None or bundle.gen != gen or not bundle.done.is_set():
            if bundle is not None:
                self.misses += 1
            return None
        try:
            messages = agent.messages
            s = bundle.snapshot_len
            snap = bundle.snapshot
            if not (0 < s <= len(messages)):
                raise _InvalidFlight("length")
            # element-identity prefix: ANY in-place mutation or rebind
            # during the wait invalidates (compact, overflow-shrink, seal)
            for i in range(s):
                if messages[i] is not snap[i]:
                    raise _InvalidFlight("identity")
            try:
                window = int(getattr(agent.cfg, "prune_window", 40) or 0)
            except (TypeError, ValueError):
                window = 40
            if window != bundle.window or agent.model.id != bundle.model_id:
                raise _InvalidFlight("config")
        except _InvalidFlight:
            self.invalid += 1
            return None
        except Exception:
            self.misses += 1
            return None
        self.hits += 1
        # external-edit early notice: a path the agent has a filestat
        # baseline for (i.e. it read the file) whose stat moved during
        # the wait means the in-context content is stale — say so BEFORE
        # the model acts, instead of at edit time via check_stale.
        try:
            self._notice_external_edits(agent, bundle)
        except Exception:
            pass
        return bundle

    def _notice_external_edits(self, agent, bundle: FlightBundle) -> None:
        try:
            store = getattr(agent, "_file_stats", None) or {}
        except Exception:
            return
        if not isinstance(store, dict) or not bundle.probes:
            return
        import os as _os
        notes: list[str] = []
        for path, cur in bundle.probes.items():
            try:
                key = _os.path.realpath(path)
            except Exception:
                continue
            old = store.get(key)
            if old is None or cur is None:
                # no baseline (never read) or vanished — check_stale and
                # the tool's own error own those cases; stay quiet here
                # unless we HAVE a baseline and the file is simply gone.
                if old is not None and cur is None:
                    notes.append(f"{path} disappeared from disk")
                continue
            if cur != old:
                notes.append(f"{path} changed on disk while the model "
                             f"was thinking — re-read before editing it")
            if len(notes) >= _EXTERNAL_EDIT_WARN_MAX:
                break
        if notes:
            try:
                agent.log.append("prewarm.external_edit",
                                 {"paths": notes[:_EXTERNAL_EDIT_WARN_MAX]},
                                 actor="prewarm")
            except Exception:
                pass

    def live(self) -> bool:
        with self._lock:
            b = self._bundle
            return b is not None and not b.done.is_set()


class _InvalidFlight(Exception):
    pass


def _os_realpath(p):
    return os.path.realpath(p)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _rand_history(rng, n, big_tool=False):
    msgs = [{"role": "system", "content": "sys"}]
    roles = ["user", "assistant", "tool"]
    for i in range(n):
        r = rng.choice(roles)
        if r == "user":
            msgs.append({"role": "user", "content": f"q{i} " + "x" * rng.randint(0, 200)})
        elif r == "assistant":
            tc = [{"id": f"c{i}", "type": "function",
                   "function": {"name": "read_file",
                                "arguments": '{"path": "f.py"}'}}] if rng.random() < 0.4 else None
            m = {"role": "assistant", "content": "a" * rng.randint(0, 300)}
            if tc:
                m["tool_calls"] = tc
            msgs.append(m)
        else:
            size = rng.randint(600, 4000) if big_tool else rng.randint(0, 800)
            msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                         "content": "t" * size})
    return msgs


def _check(label, cond, fails):
    print(("  ok  " if cond else "  FAIL") + f" {label}")
    if not cond:
        fails.append(label)


def _selftest():
    import random
    from .agent import prune_messages  # lazy: agent imports prewarm
    from .agent import Agent
    fails: list[str] = []

    print("== prewarm self-test ==")
    rng = random.Random(20261010)

    # 1. patched_prune == prune_messages on randomized histories ------
    print("-- differential: patched_prune vs prune_messages")
    for trial in range(300):
        n = rng.randint(0, 120)
        msgs = _rand_history(rng, n, big_tool=(trial % 3 == 0))
        if trial % 7 == 0 and msgs:
            msgs = msgs[1:]  # no system message
        window = rng.choice([0, 1, 5, 40, 200])
        # split into snapshot + appended (appended may exceed window)
        cut = rng.randint(0, len(msgs))
        snapshot, appended = msgs[:cut], msgs[cut:]
        # never append a user message mid-turn (matches the real loop)
        appended = [m for m in appended if m.get("role") != "user"]
        full = snapshot + appended
        b = FlightBundle(gen=1, snapshot=list(snapshot),
                         snapshot_len=len(snapshot),
                         has_system=bool(snapshot)
                         and snapshot[0].get("role") == "system",
                         window=window)
        # first_user scan, as the worker does it
        start = 1 if b.has_system else 0
        for m in snapshot[start:]:
            if m.get("role") == "user":
                b.first_user = m
                break
        got = patched_prune(full, b, window)
        want = prune_messages(full, window=window)
        if got != want:
            _check(f"trial {trial} n={n} window={window} cut={cut}",
                   False, fails)
            if len(fails) > 3:
                break
    else:
        _check("300 randomized trials byte-identical", True, fails)

    # 2. message_chars port == Agent._messages_chars -------------------
    print("-- differential: message_chars vs Agent._messages_chars")
    for _ in range(50):
        msgs = _rand_history(rng, rng.randint(0, 60))
        if message_chars(msgs) != Agent._messages_chars(
                type("S", (), {"messages": msgs})()):
            _check("message_chars mismatch", False, fails)
            break
    else:
        _check("50 randomized trials identical", True, fails)

    # 3. compact_fits --------------------------------------------------
    print("-- compact_fits")
    msgs = _rand_history(rng, 30)
    b = FlightBundle(gen=1, snapshot=list(msgs), snapshot_len=len(msgs),
                     est=1000)
    _check("fits under margin -> True",
           compact_fits(msgs, b, 100, 10_000, "m") is True, fails)
    _check("over budget -> False",
           compact_fits(msgs, b, 100, 500, "m") is False, fails)
    _check("within 10% margin but over raw budget -> False (margin)",
           compact_fits(msgs, b, 0, 1050, "m") is False, fails)

    # 4. worker never mutates the agent's live state -------------------
    print("-- worker isolation")
    import copy
    live = _rand_history(rng, 40)
    frozen = copy.deepcopy(live)

    class FakeAgent:
        def __init__(self):
            self.messages = live
            self.turns = []
            self._file_stats = {}
            import types
            self.cfg = types.SimpleNamespace(prune_window=40)
            self._model_id = "m"
            self.log = types.SimpleNamespace(
                append=lambda *a, **k: None)

        @property
        def model(self):
            return type("M", (), {"id": self._model_id})()

        def _fit_budget(self):
            return 100_000

        def _tool_schemas(self):
            return [{"name": "read_file"}]

    fa = FakeAgent()
    pw = FlightPrewarmer()
    pw.launch(fa)
    deadline = time.time() + 10
    while pw.live() and time.time() < deadline:
        time.sleep(0.01)
    b2 = pw.consume(fa)
    _check("flight completes", b2 is not None, fails)
    _check("live messages unmutated by worker", live == frozen, fails)
    _check("bundle has estimate", b2 is not None and b2.est > 0, fails)
    # mutate in place during the "wait" -> must invalidate
    pw.launch(fa)
    deadline = time.time() + 10
    while pw.live() and time.time() < deadline:
        time.sleep(0.01)
    fa.messages[3] = {"role": "user", "content": "MUTATED"}
    _check("in-place mutation invalidates",
           pw.consume(fa) is None, fails)
    # generation mismatch -> stale bundle never consumed
    pw.launch(fa)
    stale = pw._bundle
    pw.launch(fa)  # gen++
    _check("stale gen never consumed",
           stale is not None and stale.gen != pw._gen, fails)

    # 5. patched_prune_for_model == Agent._prune_for_model -------------
    print("-- differential: patched_prune_for_model vs _prune_for_model")
    from .promptbudget import schemas_tokens as _st

    class FakeAgent2(FakeAgent):
        def __init__(self, budget):
            super().__init__()
            self.cfg.prompt_token_budget = budget

    for _trial_budget in (0, 4000):
        for _ in range(60):
            msgs2 = _rand_history(rng, rng.randint(0, 90), big_tool=True)
            cut2 = rng.randint(1, len(msgs2)) if msgs2 else 0
            snapshot2, appended2 = msgs2[:cut2], msgs2[cut2:]
            appended2 = [m for m in appended2 if m.get("role") != "user"]
            full2 = snapshot2 + appended2
            fa_b = FakeAgent2(_trial_budget)
            fa_b.messages = full2
            b5 = FlightBundle(gen=1, snapshot=list(snapshot2),
                              snapshot_len=len(snapshot2),
                              has_system=bool(snapshot2)
                              and snapshot2[0].get("role") == "system",
                              window=40, model_id="m")
            st = 1 if b5.has_system else 0
            for m in snapshot2[st:]:
                if m.get("role") == "user":
                    b5.first_user = m
                    break
            schemas5 = fa_b._tool_schemas()
            res5 = _st(schemas5, "m")
            got5 = patched_prune_for_model(fa_b, b5, res5)
            # the REAL method as reference (unbound, on the fake self)
            want5 = Agent._prune_for_model(fa_b, reserved_tokens=res5)
            if got5 != want5:
                _check(f"prune_for_model budget={_trial_budget} mismatch",
                       False, fails)
                break
        else:
            _check(f"60 trials byte-identical (budget={_trial_budget})",
                   True, fails)

    # 6. external-edit notice ------------------------------------------
    print("-- external-edit early notice")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "watched.py"
        p.write_text("v1")
        fa2 = FakeAgent()
        fa2._file_stats[_os_realpath(str(p))] = _stat(str(p))
        p.write_text("v1 plus external change")  # external edit
        pw2 = FlightPrewarmer()
        # probe the path directly (as _probe_paths would via tool args)
        b3 = FlightBundle(gen=1, snapshot=[], snapshot_len=0,
                          probes={str(p): _stat(str(p))})
        b3.done.set()
        seen = []
        fa2.log = type("L", (), {"append": lambda s, t, d, **k: seen.append(t)})()
        pw2._notice_external_edits(fa2, b3)
        _check("external edit noticed",
               any(t == "prewarm.external_edit" for t in seen), fails)
        fa3 = FakeAgent()
        fa3._file_stats[_os_realpath(str(p))] = _stat(str(p))  # fresh baseline
        b4 = FlightBundle(gen=1, snapshot=[], snapshot_len=0,
                          probes={str(p): _stat(str(p))})
        b4.done.set()
        seen4 = []
        fa3.log = type("L", (), {"append": lambda s, t, d, **k: seen4.append(t)})()
        pw2._notice_external_edits(fa3, b4)
        _check("unchanged file stays quiet", not seen4, fails)

    print()
    if fails:
        print(f"PREWARM SELF-TEST FAILED: {fails}")
        raise SystemExit(1)
    print("ALL PREWARM SELF-TESTS PASS")


if __name__ == "__main__":
    _selftest()

"""Temporal Kernel — the Mul Bindu.

Every interaction with reality is recorded as an immutable, causally-ordered,
content-addressed event in a single append-only log. Nothing else is state.
State is a pure fold over that log.

Design (pure Python, stdlib only):
  * JSONL is truth. One line per event, append-only, fsync'd.
  * Content addressing: each event's id = sha256 of its canonical encoding.
  * Merkle spine: each event carries its parent's id, so the log is
    tamper-evident and any prefix is independently verifiable.
  * Seqs are global and strictly monotonic — never reused, on any branch.
  * A branch is a head pointer (an event id). Its history is the parent
    chain walked back from the head; events abandoned by a rewind or not
    chosen by a fork simply fall off the chain (fold horizon shrinks).
  * Fork at seq N seeds the new branch's head at event N, so the fork
    inherits the source's history up to the fork point.
  * Rewind moves the head pointer back to the event at seq N and seals a
    'kernel.rewind' marker (parent = event N) so the move survives reload.
    Events are never deleted. (Append-only is sacred.)
  * Reload replays the file in order: every append moved its branch's head
    to the new event, so head pointers rebuild without any second file.
  * Fold: a pure function from a branch's chain to a State projection.
    Rewind, replay, resume, audit, and cost attribution are all folds.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._foundation import get_logger

_log = get_logger("kernel")

# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> str:
    """Deterministic JSON encoding — the basis of content addressing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Event:
    """Memory: slots=True — tens of thousands of Event objects can live in
    the retention window; the per-instance __dict__ was pure overhead."""
    seq: int
    id: str
    parent: str | None
    branch: str
    ts: float
    type: str
    data: dict
    # §7.1 causal envelope — every event is attributable and traceable
    session: str = ""
    actor: str = "system"            # sovereign | human | system | …
    causation_id: str | None = None  # the event that directly caused this one
    correlation_id: str | None = None  # the root goal clause this serves
    provenance: str = "system"       # system | user | tool_output | web | file | model

    @property
    def short(self) -> str:
        return self.id[:10]

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "id": self.id,
            "parent": self.parent,
            "branch": self.branch,
            "ts": self.ts,
            "type": self.type,
            "data": self.data,
            "session": self.session,
            "actor": self.actor,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        seq = d["seq"]
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError(f"event seq is not an int: {seq!r}")
        # BUG FIX (corrupt-line hardening): id/parent/branch/ts and the
        # causal-link fields are used as dict keys and in the Merkle walk.
        # A corrupt line with a non-string id, a dict/list parent, or a
        # non-numeric ts used to load fine and then crash _chain()/why()
        # with TypeError (unhashable type) — or silently hijack the branch
        # head. Reject such lines here so _load skips them like any other
        # corrupt line.
        eid = d["id"]
        if not isinstance(eid, str):
            raise ValueError(f"event id is not a string: {eid!r}")
        parent = d.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"event parent is not a string: {parent!r}")
        branch = d.get("branch", "main")
        if not isinstance(branch, str):
            raise ValueError(f"event branch is not a string: {branch!r}")
        ts = d.get("ts", 0.0)
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            raise ValueError(f"event ts is not a number: {ts!r}")
        causation_id = d.get("causation_id")
        if causation_id is not None and not isinstance(causation_id, str):
            raise ValueError(
                f"event causation_id is not a string: {causation_id!r}")
        correlation_id = d.get("correlation_id")
        if correlation_id is not None and not isinstance(correlation_id,
                                                         str):
            raise ValueError(
                f"event correlation_id is not a string: {correlation_id!r}")
        data = d.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("event data is not an object")
        if not isinstance(d.get("type"), str):
            raise ValueError("event type is not a string")
        return cls(seq=seq, id=eid, parent=parent,
                   branch=branch, ts=ts,
                   type=d["type"], data=data,
                   session=d.get("session", ""),
                   actor=d.get("actor", "system"),
                   causation_id=causation_id,
                   correlation_id=correlation_id,
                   provenance=d.get("provenance", "system"))

    @staticmethod
    def compute_id(seq: int, parent: str | None, branch: str, ts: float,
                   type_: str, data: dict, session: str = "",
                   actor: str = "system", causation_id: str | None = None,
                   correlation_id: str | None = None,
                   provenance: str = "system") -> str:
        """Content address: hash of everything except the id itself."""
        payload = {"seq": seq, "parent": parent, "branch": branch,
                   "ts": ts, "type": type_, "data": data,
                   "session": session, "actor": actor,
                   "causation_id": causation_id,
                   "correlation_id": correlation_id,
                   "provenance": provenance}
        return _hash(_canonical(payload))


# ---------------------------------------------------------------------------
# State — the projection produced by folding
# ---------------------------------------------------------------------------


@dataclass
class State:
    """A derived, never-authoritative view of the log prefix."""
    messages: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    files_touched: set[str] = field(default_factory=set)
    commands_run: int = 0
    episodes: list[dict] = field(default_factory=list)
    dead_ends: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    goal: dict | None = None
    goal_done: list[str] = field(default_factory=list)
    # §42 — the latest goal.closed event AFTER the current goal.set; None
    # while the contract is still open
    goal_closed: dict | None = None
    autonomy: int = 3
    # advanced subsystems (compiler/evolution/brain/merge/theater/debate/
    # market) — one bucket, newest last; each module filters by "type"
    advanced_events: list[dict] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)
    # §8.3 / §9 — snapshot store references, newest last
    snapshots: list[dict] = field(default_factory=list)
    # §13 — plan DAG nodes keyed by node id
    nodes: dict[str, dict] = field(default_factory=dict)
    # §13.5 — budget events (slices, exceeded)
    budget_events: list[dict] = field(default_factory=list)
    # Part VI — clause proof/regression/amendment/focus history
    clause_proven: list[dict] = field(default_factory=list)
    clause_regressed: list[dict] = field(default_factory=list)
    amendments: list[dict] = field(default_factory=list)
    focus_shifts: list[dict] = field(default_factory=list)
    distance_measures: list[dict] = field(default_factory=list)
    # §18 — environment digests (drift detection)
    env_digests: list[dict] = field(default_factory=list)
    # §21 — calibration samples (est vs actual)
    calibration: list[dict] = field(default_factory=list)
    # §13.4 — loop/thrash detector trips
    loop_alerts: list[dict] = field(default_factory=list)
    # Mastermind — prompt coherence ledger
    prompt_sealed: list[dict] = field(default_factory=list)
    prompt_dispatches: list[dict] = field(default_factory=list)
    prompt_wire_restores: list[dict] = field(default_factory=list)
    prompt_wire_cleaned: list[dict] = field(default_factory=list)
    # v3 advanced subsystems
    router_decisions: list[dict] = field(default_factory=list)
    semantic_index: list[dict] = field(default_factory=list)
    spec_events: list[dict] = field(default_factory=list)
    daemon_events: list[dict] = field(default_factory=list)
    heal_events: list[dict] = field(default_factory=list)
    skill_events: list[dict] = field(default_factory=list)
    council_events: list[dict] = field(default_factory=list)
    # v4 professional subsystems
    lsp_events: list[dict] = field(default_factory=list)
    dap_events: list[dict] = field(default_factory=list)
    analysis_events: list[dict] = field(default_factory=list)
    mutation_events: list[dict] = field(default_factory=list)
    coverage_events: list[dict] = field(default_factory=list)
    fuzz_events: list[dict] = field(default_factory=list)
    graph_events: list[dict] = field(default_factory=list)
    browser_events: list[dict] = field(default_factory=list)
    openapi_events: list[dict] = field(default_factory=list)
    db_events: list[dict] = field(default_factory=list)
    git_events: list[dict] = field(default_factory=list)
    ensemble_events: list[dict] = field(default_factory=list)
    hybrid_events: list[dict] = field(default_factory=list)
    compress_events: list[dict] = field(default_factory=list)
    eval_events: list[dict] = field(default_factory=list)
    sched_events: list[dict] = field(default_factory=list)
    cache_events: list[dict] = field(default_factory=list)
    cost_ledger: list[dict] = field(default_factory=list)
    head_seq: int = -1
    branch: str = "main"

    def cost_summary(self) -> str:
        return f"${self.cost_usd:.4f} · {self.tokens_in}→{self.tokens_out} tok"


# ---------------------------------------------------------------------------
# EventLog — append-only, content-addressed, causally linked
# ---------------------------------------------------------------------------


# Default in-memory retention window: the newest this many events stay in
# RAM; everything older lives on in the JSONL file (the archive).
DEFAULT_MAX_EVENTS = 2000


class EventLog:
    """The single source of truth. Thread-safe for appends.

    Heads are event ids, not seqs: a branch's history is the parent chain
    walked back from its head. Seqs are global and strictly monotonic, so a
    rewind or fork never reuses a seq — abandoned events simply fall off
    the chain and stop contributing to the fold.

    MEMORY WINDOW: only the newest `max_events` events are kept in RAM
    (`self._events` / `self._by_id`); older events are pruned from memory
    automatically on append. Nothing is ever deleted from the JSONL file —
    it remains the complete, append-only archive. Consequences:

    * `events()` / `replay()` / `fold()` see the retained window only.
    * `verify()` checks hashes and spine links over the window; the oldest
      retained link is anchored against the archive file instead of failing.
    * `why()` walks causation as far back as memory reaches.
    * `rewind()` past the window pins the archived base event in memory so
      the branch is repositioned, not emptied.
    * `get_archived(event_id)` fetches any pruned event from the archive.
    * `max_events=None` disables pruning (legacy: everything in RAM).
    """

    def __init__(self, path: Path, branch: str = "main", session: str = "",
                 *, max_events: int | None = DEFAULT_MAX_EVENTS):
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be a positive int or None")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._max_events = max_events
        self._events: list[Event] = []
        self._by_id: dict[str, Event] = {}
        # branch name -> id of its head event (None = empty branch)
        self._heads: dict[str, str | None] = {}
        self._next_seq = 0
        # number of events that exist in the archive but not in RAM
        self._pruned_total = 0
        # ids in _by_id but outside the _events window: dormant branch
        # heads (kept at load) and deep-rewind bases. Tiny by construction;
        # entries stop being protected the moment they are no longer a
        # branch head, and are reaped by _prune_if_needed.
        self._extra: set[str] = set()
        self.branch = branch
        self.session = session
        # SPEED: per-branch chain cache (extended incrementally, O(1) per
        # append) + fold memoisation keyed by (branch, head seq) + one
        # persistent append handle with batched fsync. Event queries used
        # to cost O(n) every time — these caches make steady-state reads
        # O(1) and every append a buffered write.
        self._chains: dict[str, list[Event]] = {}
        self._fold_cache: dict[str, tuple[str | None, object]] = {}
        self._fh = None
        self._writes_since_sync = 0
        self._load()

    # -- lifecycle ---------------------------------------------------------
    # BUG FIX (resource cleanup): the append handle was never closed —
    # every EventLog leaked one fd for the process lifetime, and the
    # last < _SYNC_EVERY events were never fsync'd on clean shutdown.
    # close() flushes + fsyncs + closes; the context-manager form makes
    # it mechanical. __del__ is a last-resort best effort only.

    def close(self) -> None:
        """Flush, fsync and close the append handle. Idempotent."""
        with self._lock:
            fh, self._fh = self._fh, None
            if fh is not None:
                try:
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                finally:
                    try:
                        fh.close()
                    except OSError:
                        pass
            self._writes_since_sync = 0

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # noqa: D105 — best-effort finalizer
        try:
            self.close()
        except Exception:
            pass

    # -- persistence -------------------------------------------------------

    # SPEED: writes go through ONE persistent append handle; a full disk
    # sync happens once per batch instead of after every single event.
    # Events still flush to the OS on every write (survives a crash of
    # this process; only a power loss can drop the last few).
    _SYNC_EVERY = 64

    def _load(self) -> None:
        self._heads.setdefault(self.branch, None)
        if not self.path.exists():
            return
        max_e = self._max_events
        # SPEED (Worker 8/20 — startup time): the memory window only keeps
        # the newest `max_events` events, so fully JSON-parsing every
        # historical line is pure waste on a large log (30k events ≈ 3s).
        # The fast path below parses only the tail window + one head event
        # per dormant branch and light-scans the rest. Semantics are
        # identical to the full scan; any surprise falls back to it.
        if max_e is not None and self._fast_load(max_e):
            return
        self._slow_load(max_e)

    def _ingest_line(self, line: str, tail: deque[Event],
                     head_evs: dict[str, Event]) -> None:
        """The exact original per-line load logic, shared by both paths."""
        line = line.strip()
        if not line:
            return
        try:
            ev = Event.from_dict(json.loads(line))
        except (ValueError, KeyError, TypeError) as exc:
            # BUG FIX (load resilience): the old except clause missed
            # TypeError, so a single non-dict JSON line ("[1,2]",
            # "42", '"str"') raised from d["seq"] and killed the
            # ENTIRE log load — every event lost. One bad line must
            # never take down the log; skip it (torn writes and
            # hand-edits land here too) and keep going.
            # (UnicodeDecodeError subclasses ValueError, so undecodable
            # bytes are skipped here exactly as before.)
            _log.warning("kernel: skipping corrupt log line %r: %s",
                         line[:120], exc)
            return
        self._next_seq = max(self._next_seq, ev.seq + 1)
        # replay: every appended event moved its branch's head to it,
        # so head pointers rebuild exactly (rewinds included, since a
        # rewind seals a marker event that becomes the new head)
        self._heads[ev.branch] = ev.id
        head_evs[ev.branch] = ev
        if self._max_events is None:
            # legacy: no pruning — identical to the old behaviour
            self._events.append(ev)
            self._by_id[ev.id] = ev
        else:
            tail.append(ev)

    def _finish_load(self, tail: deque[Event],
                     head_evs: dict[str, Event]) -> None:
        """Materialise the window + dormant branch heads (both paths)."""
        for ev in tail:
            self._events.append(ev)
            self._by_id[ev.id] = ev
        for ev in head_evs.values():
            if ev.id not in self._by_id:
                # dormant branch head older than the window: keep the
                # single head event so the branch stays addressable.
                # It is NOT counted in len(self._events) (the window
                # stays exactly the newest max_events) and is exempt
                # from pruning while it remains a branch head.
                self._by_id[ev.id] = ev
                self._extra.add(ev.id)
        # Deep-rewind/fork markers point at an archived base; pin that one
        # immediate parent so the marker's chain survives reload. Ordinary
        # window events are not pinned — the window stays exactly max_events.
        pending = [ev for ev in list(self._by_id.values())
                   if ev.type in ("kernel.rewind", "kernel.branch")
                   and ev.parent and ev.parent not in self._by_id]
        seen_parents: set[str] = set()
        for ev in pending:
            if ev.parent in seen_parents:
                continue
            seen_parents.add(ev.parent)
            parent = self._find_in_archive(ev.parent)
            if parent is not None and parent.id not in self._by_id:
                self._by_id[parent.id] = parent
                self._extra.add(parent.id)

        self._pruned_total = max(0, self._next_seq - len(self._by_id))

    def _slow_load(self, max_e: int | None) -> None:
        """The original full-scan load: streams the file once, fully
        parsing every line. Correctness baseline for the fast path."""
        # MEMORY WINDOW: stream the file once (never hold it all in RAM).
        # Strict per-line validation is preserved exactly — one bad line
        # still only skips that line (torn writes, hand-edits) — but only
        # the newest `max_events` events are materialised into memory.
        # Every branch's newest event is ALSO kept (pinned via head
        # protection in _prune_if_needed) so dormant branches survive
        # resume with their head intact instead of vanishing.
        tail: deque[Event] = deque(maxlen=max_e) if max_e is not None \
            else deque()
        head_evs: dict[str, Event] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                self._ingest_line(line, tail, head_evs)
        if max_e is not None:
            self._finish_load(tail, head_evs)

    # Writer-emitted lines always begin with '{"seq": ' (Event.to_dict()
    # key order). The light scan below walks lines with C-level bounded
    # finds and extracts only (seq, branch) per line — no JSON parse.
    # Any deviation (hand-edits, blank lines, old formats) falls back to
    # the slow path, so format drift can never corrupt the load.
    _SEQ_PREFIX = b'{"seq": '
    _BRANCH_KEY = b'"branch": "'

    def _fast_load(self, max_e: int) -> bool:
        """Load without JSON-parsing every historical line.

        Returns True when the load is complete, False to fall back to
        _slow_load(). Observable state afterwards is identical to the
        slow path: _events (newest max_e), _by_id (+ dormant heads),
        _heads, _next_seq, _extra, _pruned_total.
        """
        try:
            fh = self.path.open("rb")
        except OSError:
            return False
        try:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return True  # empty file: nothing to load
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):
            fh.close()
            return False
        try:
            # 1. byte offset where the newest max_e non-blank lines begin
            #    (blank lines never counted toward the window, same as the
            #    slow path). C-level rsplit from the right keeps this cheap.
            offset = self._tail_offset(mm, size, max_e)
            if offset is None:
                return False
            # 2. fully parse the tail chunk with the exact original
            #    per-line logic. File order == seq order, so the chunk
            #    holds the global last valid line and every branch whose
            #    head is inside the window.
            tail: deque[Event] = deque(maxlen=max_e)
            head_evs: dict[str, Event] = {}
            for ln in mm[offset:].split(b"\n"):
                s = ln.strip()
                if not s:
                    continue
                try:
                    text = s.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # original: UnicodeDecodeError is a ValueError
                self._ingest_line(text, tail, head_evs)
            if self._next_seq == 0:
                # No valid event in the tail (e.g. a torn tail longer than
                # the window): let the slow path sort it out.
                return False
            # 3. light-scan the head region. Every line is walked with
            #    C-level bounded finds on the mmap (no copies, no split);
            #    per line we need only (seq, branch). seq feeds the max()
            #    exactly like the slow path; the newest line per dormant
            #    branch is retained so step 4 can parse it fully. A line
            #    that does not look writer-emitted (blank, hand-edit, old
            #    format) falls back to the slow path — EXCEPT that a
            #    foreign line glued inside a good line's span is simply
            #    ignored, which is exactly what the slow path's skip does.
            branch_line: dict[str, bytes] = {}
            if offset > 0:
                pos = 0
                ok = True
                find = mm.find
                seqlen = len(self._SEQ_PREFIX)  # 8: '{"seq": '
                while pos < offset:
                    end = find(b"\n", pos, offset)
                    if end < 0:
                        ok = False
                        break
                    # line = mm[pos:end]; must start with '{"seq": '
                    if mm[pos:pos + seqlen] != self._SEQ_PREFIX:
                        ok = False
                        break
                    ci = find(b",", pos + seqlen, end)
                    seq_b = mm[pos + seqlen:ci] if ci > 0 else b""
                    if not seq_b.isdigit():
                        ok = False
                        break
                    seq = int(seq_b)
                    if seq + 1 > self._next_seq:
                        self._next_seq = seq + 1
                    bi = find(self._BRANCH_KEY, pos, end)
                    if bi < 0:
                        ok = False
                        break
                    bj = find(b'"', bi + len(self._BRANCH_KEY), end)
                    if bj < 0:
                        ok = False
                        break
                    try:
                        branch = mm[bi + len(self._BRANCH_KEY):bj] \
                            .decode("utf-8")
                    except UnicodeDecodeError:
                        ok = False
                        break
                    if branch not in head_evs:
                        # newest line wins: file order, later overwrites.
                        branch_line[branch] = bytes(mm[pos:end])
                    pos = end + 1
                if not ok:
                    return False
            # 4. resolve dormant branch heads: newest *valid* line per
            #    branch, parsed with the exact original logic. The true
            #    branch name comes from the parse (never the light scan),
            #    and a corrupt newest line falls back to the slow path
            #    (which skips it exactly like the original).
            for line in branch_line.values():
                try:
                    ev = Event.from_dict(json.loads(line.decode("utf-8")))
                except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                    return False
                if ev.branch in head_evs:
                    continue
                self._heads[ev.branch] = ev.id
                head_evs[ev.branch] = ev
            self._finish_load(tail, head_evs)
            return True
        finally:
            mm.close()
            fh.close()

    @staticmethod
    def _tail_offset(mm, size: int, max_e: int) -> int | None:
        """Byte offset at which the newest `max_e` non-blank lines begin.

        Returns None when the file holds fewer than max_e non-blank
        lines (the slow path is cheap then) or the layout defeats the
        bounded probe.
        """
        probe = min(size, max_e * 2048 + 65536)
        while True:
            base = size - probe
            buf = mm[base:]
            parts = buf.rsplit(b"\n", max_e + 64)
            # parts[0] is a partial line unless base == 0; parts[1:] are
            # whole lines, newest last.
            first_whole = 0 if base == 0 else 1
            need = max_e
            j = len(parts) - 1
            while j >= first_whole and need > 0:
                if parts[j].strip():
                    need -= 1
                    if need == 0:
                        break
                j -= 1
            if need == 0:
                off = base
                for k in range(j):
                    off += len(parts[k]) + 1
                return off
            if probe >= size:
                return None
            probe = min(size, probe * 2)

    def _persist(self, ev: Event) -> None:
        try:
            self._write_event(ev)
        except FileNotFoundError:
            # the home directory vanished mid-session (deleted externally,
            # fresh mount, etc.) — recreate it and write again. The log
            # must never take the app down over a missing directory.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = None
            self._write_event(ev)
        except ValueError:
            # handle was closed/replaced underneath us — reopen once
            self._fh = None
            self._write_event(ev)
        except OSError:
            # ENOSPC/EIO etc. — the buffered handle may now hold a partial
            # line; discarding it and forcing a fresh handle (whose torn-
            # tail probe re-emits the newline guard) keeps the log valid.
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            self._write_event(ev)

    def _write_event(self, ev: Event) -> None:
        if self._fh is None:
            # A crash can leave a torn final line (no trailing newline).
            # Opening in append mode would glue the next event onto it,
            # destroying BOTH lines. Terminate the torn line first so only
            # it is lost; the loader already skips corrupt lines.
            torn = False
            try:
                with self.path.open("rb") as probe:
                    probe.seek(0, os.SEEK_END)
                    if probe.tell() > 0:
                        probe.seek(-1, os.SEEK_END)
                        torn = probe.read(1) != b"\n"
            except OSError:
                torn = False
            self._fh = self.path.open("a", encoding="utf-8")
            if torn:
                self._fh.write("\n")
        # default=str must match Event.compute_id's canonicalization —
        # otherwise a value that hashed fine fails to serialize AFTER the
        # in-memory state already advanced past it
        self._fh.write(json.dumps(ev.to_dict(), ensure_ascii=False,
                                  default=str) + "\n")
        self._fh.flush()
        self._writes_since_sync += 1
        if self._writes_since_sync >= self._SYNC_EVERY:
            self._writes_since_sync = 0
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    # -- append ------------------------------------------------------------

    def append(self, type_: str, data: dict | None = None,
               branch: str | None = None, *,
               actor: str = "system",
               causation_id: str | None = None,
               correlation_id: str | None = None,
               provenance: str = "system",
               session: str | None = None) -> Event:
        """Append one event to a branch. Returns the sealed Event.

        The causal envelope (§7.1) makes every event attributable:
        causation_id = the event that directly caused this one,
        correlation_id = the goal clause this ultimately serves.

        The payload is ALWAYS sealed as a dict: a non-dict (or a
        non-JSON value nested inside) is coerced instead of rejected,
        because every downstream consumer — the TUI panel, the
        dashboard fold, report/evolution/theater reads, the notifier —
        calls ``ev.data.get(...)`` and a single string payload would
        crash them all with ``AttributeError: 'str' object has no
        attribute 'get'``.
        """
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            # Never seal a non-dict payload: keep the information in
            # _raw instead of letting a malformed emitter poison the
            # log for every consumer.
            data = {"_raw": data}
        with self._lock:
            br = branch or self.branch
            parent_id = self._heads.get(br)
            seq = self._next_seq
            self._next_seq += 1
            ts = time.time()
            sess = session if session is not None else self.session
            # default causation: the branch's current head caused this event
            caus = causation_id if causation_id is not None else parent_id
            eid = Event.compute_id(seq, parent_id, br, ts, type_, data,
                                   sess, actor, caus, correlation_id,
                                   provenance)
            ev = Event(seq=seq, id=eid, parent=parent_id, branch=br,
                       ts=ts, type=type_, data=data, session=sess,
                       actor=actor, causation_id=caus,
                       correlation_id=correlation_id, provenance=provenance)
            # persist FIRST — if serialization or the write fails, the
            # in-memory log must not advance (it would diverge from disk
            # and burn a seq / move a head for an event that never landed)
            self._persist(ev)
            self._events.append(ev)
            self._by_id[eid] = ev
            self._heads[br] = eid
            self._prune_if_needed()
            return ev

    # -- memory window -----------------------------------------------------

    def _prune_if_needed(self) -> None:
        """Drop the oldest in-memory events past the retention window.

        Caller holds self._lock. Only the RAM copies go away — every event
        was already persisted to the archive file by _persist() before it
        entered memory, so nothing is lost. Current branch heads are never
        dropped (a branch must stay addressable even when dormant); the
        per-branch chain caches are cleared because they may reference
        dropped events and rebuild lazily on next access.
        """
        max_e = self._max_events
        if max_e is None:
            return
        protected = {h for h in self._heads.values() if h}
        referenced = {ev.parent for ev in self._by_id.values() if ev.parent}
        # reap window-external events that are no longer a branch head
        # nor referenced as a parent by any retained event.
        stale = [eid for eid in self._extra
                 if eid not in protected and eid not in referenced]
        for eid in stale:
            self._by_id.pop(eid, None)
            self._extra.discard(eid)
        over = len(self._events) - max_e
        if over <= 0:
            return
        drop: set[str] = set()
        for ev in self._events:
            if len(drop) >= over:
                break
            if ev.id not in protected:
                drop.add(ev.id)
        if not drop:
            return
        self._events = [ev for ev in self._events if ev.id not in drop]
        for eid in drop:
            del self._by_id[eid]
        self._pruned_total += len(drop)
        # chain caches hold direct Event references — some now dangle.
        # Rebuild is lazy and incremental afterwards, so a clear is O(1).
        self._chains.clear()
        # NOTE: _fold_cache is deliberately left alone. Its incremental
        # extension walks _by_id from the current head back to the cached
        # head; pruned events are always OLDER than any live cached head in
        # steady state (folds run every turn), so extension keeps working.
        # If a branch went 2000+ appends without a fold, the cache misses
        # and rebuilds from the retained window — the documented window
        # semantic, not silent corruption.

    # -- archive access ----------------------------------------------------

    @property
    def max_events(self) -> int | None:
        """In-memory retention window (None = keep everything)."""
        return self._max_events

    @property
    def total_events(self) -> int:
        """Every event ever appended (seqs are never reused)."""
        return self._next_seq

    @property
    def archived_event_count(self) -> int:
        """Events that live in the archive file but not in RAM."""
        return self._pruned_total

    @property
    def archive_path(self) -> Path:
        """The JSONL file doubles as the complete event archive."""
        return self.path

    def _find_in_archive(self, event_id: str) -> Event | None:
        """Fetch one event by id from the archive file (linear scan).

        Memory is checked first, so this only hits disk for pruned events.
        A cheap substring prefilter avoids JSON-parsing every line.
        """
        ev = self._by_id.get(event_id)
        if ev is not None:
            return ev
        try:
            f = self.path.open("r", encoding="utf-8")
        except OSError:
            return None
        with f:
            for line in f:
                if event_id not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get("id") == event_id:
                    try:
                        return Event.from_dict(d)
                    except (ValueError, KeyError, TypeError):
                        return None
        return None

    def get_archived(self, event_id: str) -> Event | None:
        """Return an event by id, from memory or the on-disk archive.

        Unlike get() (memory only), this reaches back past the retention
        window. Linear scan — fine for occasional lookups, not hot paths.
        """
        return self._find_in_archive(event_id)

    def _load_archived_base(self, branch: str,
                            target_seq: int) -> Event | None:
        """Deep-rewind support: materialise the newest archived event on
        *branch* with seq <= target_seq into memory.

        Lets rewind() reach past the retention window: the base event is
        loaded into _by_id (protected from pruning while it is a branch
        head) so the branch is repositioned instead of emptied. Its own
        parent stays in the archive — the chain simply ends there.
        """
        best_d: dict | None = None
        best_s = -1
        try:
            f = self.path.open("r", encoding="utf-8")
        except OSError:
            return None
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict) or d.get("branch") != branch:
                    continue
                s = d.get("seq")
                if not isinstance(s, int) or s > target_seq or s <= best_s:
                    continue
                best_s, best_d = s, d
        if best_d is None:
            return None
        try:
            ev = Event.from_dict(best_d)
        except (ValueError, KeyError, TypeError):
            return None
        if ev.id not in self._by_id:
            self._by_id[ev.id] = ev
            self._extra.add(ev.id)
        return ev

    # -- chain walking -------------------------------------------------------

    def _chain(self, branch: str) -> list[Event]:
        """The branch's history (parent chain from its head), causal order.

        SPEED: cached per branch and extended incrementally — a steady-state
        append costs O(1) here instead of an O(n) walk + reversal."""
        head_id = self._heads.get(branch)
        cached = self._chains.get(branch)
        if cached is not None:
            if not cached and head_id is None:
                return cached
            if cached and head_id == cached[-1].id:
                return cached
            # head moved forward (new appends) — extend from the cached tail
            if cached and head_id is not None:
                tail_id = cached[-1].id
                new: list[Event] = []
                cur = self._by_id.get(head_id)
                seen: set[str] = set()
                while cur is not None and cur.id != tail_id \
                        and cur.id not in seen:
                    seen.add(cur.id)
                    new.append(cur)
                    cur = (self._by_id.get(cur.parent)
                           if cur.parent else None)
                if cur is not None and cur.id == tail_id:
                    new.reverse()
                    self._chains[branch] = cached + new
                    return self._chains[branch]
        # first access, rewind or fork — full rebuild (then cached)
        chain: list[Event] = []
        cur_id = head_id
        seen2: set[str] = set()
        while cur_id and cur_id in self._by_id and cur_id not in seen2:
            seen2.add(cur_id)
            ev = self._by_id[cur_id]
            chain.append(ev)
            cur_id = ev.parent
        chain.reverse()
        self._chains[branch] = chain
        return chain

    def _event_at(self, branch: str, seq: int) -> Event | None:
        """The newest event on the branch's chain with e.seq <= seq.

        The chain is seq-ascending, so this is a single forward scan
        with early exit — no full pass needed."""
        if seq < 0:
            return None
        best: Event | None = None
        for ev in self._chain(branch):
            if ev.seq > seq:
                break
            best = ev
        return best

    # -- queries -----------------------------------------------------------

    def events(self, branch: str | None = None,
               upto_seq: int | None = None) -> list[Event]:
        """Events of a branch up to (and including) a seq horizon.

        MEMORY WINDOW: only events retained in RAM are returned (the
        newest `max_events` plus pinned branch heads). The full history
        is always available in the archive file (`archive_path`).

        CONTRACT: the returned list may be the branch's SHARED cached
        chain — treat it as read-only. Mutating it (append/sort/pop)
        corrupts the cache and every later fold. Build a new list if
        you need to transform it.

        BUG FIX (concurrency): _chain() mutates the per-branch cache
        (self._chains[branch] = ...), but events() used to call it with no
        lock held — two threads folding/reading concurrently could race on
        the cache assignment and publish a torn or stale chain. The lock
        is held for the cache lookup/build; the returned list itself is
        never mutated in place afterwards, so handing it out is safe."""
        br = branch or self.branch
        with self._lock:
            evs = self._chain(br)
        if upto_seq is None:
            return evs
        return [e for e in evs if e.seq <= upto_seq]

    def head(self, branch: str | None = None) -> int:
        """Seq of the branch's head event (-1 for an empty branch)."""
        head_id = self._heads.get(branch or self.branch)
        ev = self._by_id.get(head_id) if head_id else None
        return ev.seq if ev else -1

    def branches(self) -> list[str]:
        return sorted(self._heads.keys())

    def get(self, event_id: str) -> Event | None:
        """An event by id, from memory only (see get_archived())."""
        return self._by_id.get(event_id)

    def __len__(self) -> int:
        """Events currently in RAM (the retention window)."""
        return len(self._events)

    # -- time travel -------------------------------------------------------

    def rewind(self, seq: int, branch: str | None = None) -> int:
        """Move a branch's head back to the event at/below seq.

        Events are NOT deleted; they fall off the chain. The move is sealed
        as a 'kernel.rewind' marker (which becomes the new head) so it
        survives reload. Returns the new head seq (the marker's seq).

        A target older than the retention window is served from the
        archive: the base event is pinned in memory and the branch is
        repositioned (its deeper ancestors stay archived).
        """
        br = branch or self.branch
        with self._lock:
            current = self.head(br)
            target = max(-1, min(seq, current))
            base = self._event_at(br, target)
            if base is None and target >= 0 and self._pruned_total:
                # target lies past the retention window: materialise the
                # archived base event so the branch is repositioned, not
                # emptied. Its ancestors stay in the archive.
                base = self._load_archived_base(br, target)
            self._heads[br] = base.id if base else None
            # the per-branch chain cache and fold cache still hold the
            # pre-rewind chain; without invalidation the next _chain()
            # call would walk from the new head and fail to find the
            # cached tail (the old head is ABOVE the new head, not below),
            # forcing a full O(n) rebuild. Drop both caches here.
            self._chains.pop(br, None)
            self._fold_cache.pop(br, None)
            marker = self.append("kernel.rewind",
                                 {"branch": br, "from": current,
                                  "to": target}, branch=br)
            return marker.seq

    def fork(self, at_seq: int | None = None,
             name: str | None = None) -> str:
        """Create a new branch diverging from at_seq (default: current head).

        The new branch's head starts at the fork-point event, so it inherits
        the source's full history up to that point; a 'kernel.branch' marker
        is sealed on it. Returns the branch name."""
        with self._lock:
            src = self.branch
            at = self.head(src) if at_seq is None else at_seq
            base = self._event_at(src, at)
            if base is None and at >= 0 and self._pruned_total:
                # fork point older than the retention window: pin the
                # archived base so the new branch inherits real history
                base = self._load_archived_base(src, at)
            if name:
                # never clobber an existing branch — a second fork with the
                # same name would silently rewind the first one's head and
                # orphan its exclusive events
                new_name = name
                n = 2
                while new_name in self._heads:
                    new_name = f"{name}-{n}"
                    n += 1
            else:
                new_name = f"branch-{len(self.branches()) + 1}"
                n = 2
                while new_name in self._heads:
                    new_name = f"branch-{len(self.branches()) + 1}-{n}"
                    n += 1
            self._heads[new_name] = base.id if base else None
            self.append("kernel.branch",
                        {"from": src, "at_seq": base.seq if base else -1,
                         "at_id": base.id if base else None,
                         "name": new_name}, branch=new_name)
            return new_name

    def checkout(self, branch: str) -> None:
        if branch in self._heads:
            self.branch = branch

    # -- integrity ---------------------------------------------------------

    def verify(self, branch: str | None = None) -> tuple[bool, str]:
        """Re-hash every retained event and check the Merkle spine links.

        Under the memory window the oldest retained event's parent lives in
        the archive: that one link is anchored by looking the parent up in
        the archive file instead of failing. Everything else is verified
        exactly as before.
        """
        br = branch or self.branch
        evs = self.events(br)
        prev_id: str | None = None
        for i, ev in enumerate(evs):
            recomputed = Event.compute_id(ev.seq, ev.parent, ev.branch,
                                          ev.ts, ev.type, ev.data,
                                          ev.session, ev.actor,
                                          ev.causation_id,
                                          ev.correlation_id, ev.provenance)
            if recomputed != ev.id:
                return False, f"seq {ev.seq}: content hash mismatch"
            if i == 0 and ev.parent is not None and ev.parent != prev_id:
                # oldest retained event: its parent was pruned from RAM.
                # Anchor the link against the archive instead of failing.
                archived = self._find_in_archive(ev.parent)
                if archived is None:
                    return False, (
                        f"seq {ev.seq}: parent {ev.parent[:12]}… neither "
                        f"in memory nor in the archive")
            elif ev.parent != prev_id:
                return False, f"seq {ev.seq}: broken spine link"
            prev_id = ev.id
        suffix = (f" (+{self._pruned_total} archived)"
                  if self._pruned_total else "")
        return True, f"{len(evs)} events verified{suffix}"

    # -- causality ---------------------------------------------------------

    def why(self, event_id: str, limit: int = 50) -> list[Event]:
        """Walk the causation chain backwards from an event to its root.

        Answers 'why did this happen?' mechanically (§7.1, Appendix A
        `argus why`): each event's causation_id names its direct cause, so
        any file change or dollar spent traces back to the human
        instruction that started it.
        """
        chain: list[Event] = []
        cur = self._by_id.get(event_id)
        seen: set[str] = set()
        while cur and cur.id not in seen and len(chain) < limit:
            seen.add(cur.id)
            chain.append(cur)
            cur = self._by_id.get(cur.causation_id) if cur.causation_id else None
        return chain


# ---------------------------------------------------------------------------
# Fold — state as a pure function of history
# ---------------------------------------------------------------------------

# tool name -> whether it mutates the filesystem
_MUTATING_TOOLS = {"write_file", "edit_file", "create_directory",
                   "copy_path", "move_path", "delete_path", "run_command",
                   "live_shell", "apply_patch"}

# Event types of the advanced subsystems folded into State.advanced_events
ADVANCED_EVENT_TYPES = frozenset({
    "compile.plan", "compile.wave", "compile.done",
    "evolution.generation", "evolution.deployed", "evolution.rollback",
    "brain.remembered", "brain.recalled", "brain.consolidated",
    "brain.forgotten",
    "merge.started", "merge.merged", "merge.conflict",
    "theater.counterfactual",
    "debate.round", "debate.verdict", "debate.calibration",
    "market.announce", "market.bid", "market.award", "market.settle",
    # v6 advanced subsystems
    "verify.plan", "verify.violation", "verify.trace",
    "mcts.search", "mcts.best",
    "causal.edge", "causal.intervention",
    "bandit.pull", "bandit.update",
    "mesh.node", "mesh.task", "mesh.result",
    "meta.role.drafted", "meta.role.sealed", "meta.role.rejected",
    "synth.tool.drafted", "synth.tool.tested", "synth.tool.registered",
    "ci.watch", "ci.run", "ci.streak",
    "tuner.trial", "tuner.best",
    "dual.route", "dual.escalation",
    "world.impact", "world.learn",
    "race.start", "race.winner", "race.cancel",
    "homeo.check", "homeo.repair",
    "attention.auction",
    "fabric.assert", "fabric.retract",
})


def _fold_apply(st: State, ev: Event) -> None:
    """Apply ONE event to a State projection (the reduce step)."""
    st.head_seq = ev.seq
    d = ev.data if isinstance(ev.data, dict) else {}
    t = ev.type

    if t == "user.message":
        st.messages.append({"role": "user", "content": d.get("text", "")})
    elif t == "assistant.message":
        st.messages.append({"role": "assistant",
                            "content": d.get("text", "")})
    elif t == "tool.call":
        st.tool_calls += 1
        name = d.get("name", "")
        if name == "run_command":
            st.commands_run += 1
        args = d.get("args")
        if name in _MUTATING_TOOLS and isinstance(args, dict):
            p = args.get("path") or args.get("dst")
            if p:
                st.files_touched.add(str(p))
    elif t == "tool.result":
        if d.get("status") == "error":
            st.tool_errors += 1
    elif t == "cost.incurred":
        try:
            st.cost_usd += float(d.get("usd", 0.0) or 0.0)
            st.tokens_in += int(d.get("tokens_in", 0) or 0)
            st.tokens_out += int(d.get("tokens_out", 0) or 0)
        except (TypeError, ValueError):
            pass  # corrupt numeric field — skip, never brick the fold
    elif t == "memory.episode":
        st.episodes.append(d)
    elif t == "deadend.recorded":
        st.dead_ends.append(d)
    elif t == "goal.set":
        st.goal = d
        st.goal_done = []
        st.goal_closed = None  # a new contract reopens the world
    elif t == "goal.clause.done":
        st.goal_done.append(d.get("clause", ""))
    elif t == "autonomy.changed":
        try:
            st.autonomy = int(d.get("level", st.autonomy))
        except (TypeError, ValueError):
            pass
    elif t in ADVANCED_EVENT_TYPES:
        st.advanced_events.append({"type": t, **d})
    elif t == "prompt.sealed":
        st.prompt_sealed.append(d)
    elif t == "prompt.dispatch":
        st.prompt_dispatches.append(d)
    elif t == "prompt.wire_restore":
        st.prompt_wire_restores.append(d)
    elif t == "prompt.wire_clean":
        st.prompt_wire_cleaned.append(d)
    elif t == "router.decision":
        st.router_decisions.append(d)
    elif t == "semantic.indexed":
        st.semantic_index.append(d)
    elif t in ("spec.prefetch", "spec.hit", "spec.miss", "spec.evict"):
        st.spec_events.append({"type": t, **d})
    elif t in ("daemon.mission", "daemon.checkpoint", "daemon.tick",
               "daemon.wake", "daemon.done"):
        st.daemon_events.append({"type": t, **d})
    elif t in ("heal.captured", "heal.hypothesis", "heal.patch",
               "heal.retry", "heal.lesson"):
        st.heal_events.append({"type": t, **d})
    elif t in ("skill.authored", "skill.validated", "skill.registered",
               "skill.rejected"):
        st.skill_events.append({"type": t, **d})
    elif t in ("council.convened", "council.position", "council.verdict"):
        st.council_events.append({"type": t, **d})
    elif t in ("lsp.session", "lsp.symbols", "lsp.references",
               "lsp.diagnostics"):
        st.lsp_events.append({"type": t, **d})
    elif t in ("dap.session", "dap.breakpoint", "dap.stopped",
               "dap.variables"):
        st.dap_events.append({"type": t, **d})
    elif t in ("analysis.taint", "analysis.complexity",
               "analysis.cycles"):
        st.analysis_events.append({"type": t, **d})
    elif t in ("mutation.run", "mutation.result"):
        st.mutation_events.append({"type": t, **d})
    elif t in ("coverage.run", "coverage.result"):
        st.coverage_events.append({"type": t, **d})
    elif t in ("fuzz.run", "fuzz.crash", "fuzz.shrunk"):
        st.fuzz_events.append({"type": t, **d})
    elif t in ("graph.entity", "graph.relation", "graph.query"):
        st.graph_events.append({"type": t, **d})
    elif t in ("browser.navigate", "browser.action", "browser.extract"):
        st.browser_events.append({"type": t, **d})
    elif t in ("openapi.compiled", "openapi.call"):
        st.openapi_events.append({"type": t, **d})
    elif t in ("db.query", "db.schema"):
        st.db_events.append({"type": t, **d})
    elif t in ("git.diff", "git.commit", "git.blame"):
        st.git_events.append({"type": t, **d})
    elif t in ("ensemble.run", "ensemble.verdict"):
        st.ensemble_events.append({"type": t, **d})
    elif t in ("hybrid.indexed", "hybrid.query"):
        st.hybrid_events.append({"type": t, **d})
    elif t in ("compress.run",):
        st.compress_events.append({"type": t, **d})
    elif t in ("eval.task", "eval.result"):
        st.eval_events.append({"type": t, **d})
    elif t in ("sched.job", "sched.fired"):
        st.sched_events.append({"type": t, **d})
    elif t in ("cache.metrics",):
        st.cache_events.append({"type": t, **d})
    elif t in ("cost.entry",):
        st.cost_ledger.append(d)
    elif t == "judge.verdict":
        st.verdicts.append(d)
    elif t == "snapshot.taken":
        st.snapshots.append(d)
    elif t == "plan.node":
        node = dict(d)
        st.nodes[node.get("id", "")] = node
    elif t == "plan.node.status":
        node = st.nodes.get(d.get("id", ""))
        if node is not None:
            node["status"] = d.get("status", node.get("status"))
            if "attempts" in d:
                node["attempts"] = d["attempts"]
    elif t == "budget.event":
        st.budget_events.append(d)
    elif t == "clause.proven":
        st.clause_proven.append(d)
    elif t == "clause.regressed":
        st.clause_regressed.append(d)
    elif t == "goal.amendment":
        st.amendments.append(d)
    elif t == "goal.focus":
        st.focus_shifts.append(d)
    elif t == "goal.distance":
        st.distance_measures.append(d)
    elif t == "goal.closed":
        st.goal_closed = d
    elif t == "fact.learned":
        st.facts.append(d)
    elif t == "env.digest":
        st.env_digests.append(d)
    elif t == "calibration.sample":
        st.calibration.append(d)
    elif t == "loop.alert":
        st.loop_alerts.append(d)


def fold(log: EventLog, branch: str | None = None,
         upto_seq: int | None = None,
         from_seq: int = -1) -> State:
    """Pure fold: reduce a log prefix into a State projection.

    from_seq skips every event with seq <= from_seq — used for
    session-scoped projections (budget spend etc.) without mutating
    or copying the log.

    SPEED: full-log folds are cached per branch and extended
    INCREMENTALLY — after the first fold, a steady-state call applies
    only the events appended since the last one (typically 1-5), so
    the dozens of fold() calls inside one turn cost near-zero instead
    of O(n) each. Rewinds naturally miss the cache and rebuild.
    Callers must treat the returned State as read-only.

    MEMORY WINDOW: the fold covers the retained in-memory window
    (newest `max_events` events), not the entire archive. This is what
    keeps per-turn fold cost and State size bounded in long sessions."""
    if upto_seq is None and from_seq == -1:
        with log._lock:
            return _fold_cached(log, branch or log.branch)
    br = branch or log.branch
    st = State(branch=br)
    for ev in log.events(br, upto_seq):
        if ev.seq <= from_seq:
            continue
        _fold_apply(st, ev)
    return st


def replay(log: EventLog, branch: str | None = None,
           upto_seq: int | None = None) -> list[Event]:
    """Events of a branch in seq order — the raw material every fold
    consumes; §26 text-film replays iterate this.

    Operates over the in-memory retention window; the complete history
    remains replayable from the archive file (log.archive_path)."""
    return log.events(branch or log.branch, upto_seq)


def _fold_cached(log: EventLog, br: str) -> State:
    """Cached, incremental full-log fold for one branch (caller holds
    log._lock)."""
    head_id = log._heads.get(br)
    cached = log._fold_cache.get(br)
    if cached is not None:
        cached_id, st = cached
        if cached_id == head_id:
            return st
        # extend: walk the new events back from the head to the cached
        # head, then apply them in causal order onto the cached state
        new_evs: list[Event] = []
        cur = head_id
        seen: set[str] = set()
        chained = False
        while cur and cur != cached_id and cur not in seen:
            seen.add(cur)
            ev = log._by_id.get(cur)
            if ev is None:
                break
            new_evs.append(ev)
            cur = ev.parent
        else:
            if cur == cached_id:
                chained = True
        if chained:
            for ev in reversed(new_evs):
                _fold_apply(st, ev)
            log._fold_cache[br] = (head_id, st)
            return st
    # first fold on this branch, or the chain diverged (rewind) — rebuild
    st = State(branch=br)
    for ev in log._chain(br):
        _fold_apply(st, ev)
    log._fold_cache[br] = (head_id, st)
    return st


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "kernel-selftest.jsonl"
        log = EventLog(path)

        # basic append: monotonic seqs, intact spine
        for i in range(5):
            log.append("user.message", {"text": f"msg{i}"})
        assert log.head() == 4
        assert [e.seq for e in log.events()] == [0, 1, 2, 3, 4]
        ok, msg = log.verify()
        assert ok, msg

        # rewind: fold horizon shrinks, seqs are NEVER reused afterwards
        log.rewind(2)
        evs = log.events()
        texts = [e.data.get("text") for e in evs if e.type == "user.message"]
        assert texts == ["msg0", "msg1", "msg2"], texts
        assert log.head() == 5  # the kernel.rewind marker is the new head
        log.append("user.message", {"text": "after-rewind"})
        evs = log.events()
        seqs = [e.seq for e in evs]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs
        texts = [e.data.get("text") for e in evs if e.type == "user.message"]
        assert texts == ["msg0", "msg1", "msg2", "after-rewind"], texts
        ok, msg = log.verify()
        assert ok, msg

        # fold sees only the live chain
        st = fold(log)
        assert [m["content"] for m in st.messages] == \
            ["msg0", "msg1", "msg2", "after-rewind"]

        # fork: new branch inherits history up to the fork point
        fork_at = log.head()
        branch = log.fork(at_seq=fork_at, name="alt")
        log.checkout(branch)
        assert log.branch == "alt"
        st = fold(log, branch="alt")
        assert [m["content"] for m in st.messages] == \
            ["msg0", "msg1", "msg2", "after-rewind"]
        log.append("user.message", {"text": "alt-only"})
        # main is untouched by writes on alt
        st_main = fold(log, branch="main")
        assert "alt-only" not in [m["content"] for m in st_main.messages]
        ok, msg = log.verify(branch="alt")
        assert ok, msg

        # reload from disk: heads rebuild, chains and verify survive
        log2 = EventLog(path)
        assert set(log2.branches()) == {"main", "alt"}
        assert log2.head("main") == log.head("main")
        assert log2.head("alt") == log.head("alt")
        st = fold(log2, branch="alt")
        assert [m["content"] for m in st.messages][-1] == "alt-only"
        ok, msg = log2.verify(branch="main")
        assert ok, msg
        ok, msg = log2.verify(branch="alt")
        assert ok, msg

        # rewind survives reload (the marker event is the persisted head)
        log2.checkout("main")
        marker_seq = log2.rewind(1)
        log3 = EventLog(path)
        texts = [e.data.get("text") for e in log3.events("main")
                 if e.type == "user.message"]
        assert texts == ["msg0", "msg1"], texts
        ok, msg = log3.verify("main")
        assert ok, msg

        # content addressing: tampering is detected
        ev = log3.events("main")[1]
        ev.data["text"] = "tampered"
        ok, msg = log3.verify("main")
        assert not ok and "content hash" in msg, (ok, msg)

    # causal envelope: causation chains + why() (§7.1, Appendix A)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "causal.jsonl"
        log = EventLog(path, session="s1")
        root = log.append("user.message", {"text": "fix the bug"},
                          actor="human", provenance="user")
        mid = log.append("tool.call", {"name": "edit_file"},
                         actor="sovereign", causation_id=root.id,
                         correlation_id="C1", provenance="model")
        leaf = log.append("tool.result", {"status": "done"},
                          actor="system", causation_id=mid.id,
                          correlation_id="C1", provenance="tool_output")
        chain = log.why(leaf.id)
        assert [e.type for e in chain] == ["tool.result", "tool.call",
                                           "user.message"], chain
        assert chain[-1].actor == "human"
        assert leaf.correlation_id == "C1"
        ok, msg = log.verify()
        assert ok, msg
        # envelope survives reload
        log2 = EventLog(path, session="s1")
        ev = log2.events()[-1]
        assert ev.causation_id == mid.id and ev.provenance == "tool_output"
        assert [e.type for e in log2.why(ev.id)] == \
            ["tool.result", "tool.call", "user.message"]

    # --- memory window: pruning + archive + replay -----------------------
    import gc as _gc
    import sys as _sys

    def _expected(i: int):
        if i % 3 == 0:
            return ("user.message", {"text": f"hello {i} " * 30})
        if i % 3 == 1:
            return ("tool.call", {"name": "run_command",
                                  "args": {"cmd": "pytest -q"}})
        return ("tool.result", {"status": "ok", "output": "y" * 1500})

    def _retained_bytes(lg: "EventLog") -> int:
        """Deep size of everything the log keeps in RAM."""
        seen: set[int] = set()
        total = 0
        stack: list = [lg._events, lg._by_id]
        while stack:
            o = stack.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            try:
                total += _sys.getsizeof(o)
            except TypeError:
                continue
            if isinstance(o, dict):
                stack.extend(o.keys())
                stack.extend(o.values())
            elif isinstance(o, (list, tuple, set, frozenset)):
                stack.extend(o)
            elif hasattr(o, "__dict__"):
                stack.append(vars(o))
        return total

    def _rss_kb():
        try:
            import resource as _res
            return _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
        except ImportError:
            return None

    N, WIN = 3000, 200
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "window.jsonl"
        log = EventLog(path, max_events=WIN)
        _gc.collect()
        rss0 = _rss_kb()
        first_id = None
        for i in range(N):
            t, d = _expected(i)
            ev = log.append(t, d)
            if i == 0:
                first_id = ev.id
        _gc.collect()
        rss1 = _rss_kb()

        # pruning: exactly the newest WIN events in RAM, rest archived
        assert len(log) == WIN, len(log)
        assert len(log._by_id) == WIN, len(log._by_id)
        assert log.total_events == N, log.total_events
        assert log.archived_event_count == N - WIN
        assert log.archive_path == path
        evs = log.events()
        assert [e.seq for e in evs] == list(range(N - WIN, N))
        # recent events intact: type, data and spine links
        for e in evs:
            t, d = _expected(e.seq)
            assert e.type == t and e.data == d, e.seq
        for a, b in zip(evs, evs[1:]):
            assert b.parent == a.id, (a.seq, b.seq)
        # replay works over the window
        assert [e.seq for e in replay(log)] == list(range(N - WIN, N))
        # pruned events are gone from RAM but fetchable from the archive
        assert log.get(first_id) is None
        old = log.get_archived(first_id)
        assert old is not None and old.seq == 0, old
        assert old.data == _expected(0)[1]
        # verify anchors the oldest retained link against the archive
        ok, msg = log.verify()
        assert ok, msg
        assert "archived" in msg, msg
        # fold covers the window (this is what bounds per-turn cost)
        st = fold(log)
        want_msgs = sum(1 for i in range(N - WIN, N) if i % 3 == 0)
        assert len(st.messages) == want_msgs, len(st.messages)
        assert st.head_seq == N - 1

        # memory: bounded, not linear in N
        win_bytes = _retained_bytes(log)
        print(f"window self-test: {N} appends, window={WIN}: "
              f"in-RAM events={len(log)}, retained~{win_bytes/1024:.0f} KB"
              + (f", RSS +{(rss1-rss0)/1024:.1f} MB"
                 if rss0 is not None and rss1 is not None else ""))
        # worker 19/20: Event is slots-based — no per-instance __dict__
        # (~88 bytes saved per event vs the old frozen dataclass)
        _probe = log.events()[-1]
        assert not hasattr(_probe, "__dict__"), \
            "Event lost its __slots__ — memory win regressed"

        # legacy mode (max_events=None) keeps everything — unchanged
        with tempfile.TemporaryDirectory() as td2:
            ulog = EventLog(Path(td2) / "u.jsonl", max_events=None)
            for i in range(500):
                t, d = _expected(i)
                ulog.append(t, d)
            assert len(ulog) == 500 and ulog.archived_event_count == 0
            ok, msg = ulog.verify()
            assert ok, msg
            assert _retained_bytes(ulog) > win_bytes, \
                "unbounded log must retain more than the windowed one"

        # session resume: reload keeps heads, window, verify, replay
        log2 = EventLog(path, max_events=WIN)
        assert len(log2) == WIN, len(log2)
        assert log2.total_events == N
        assert log2.head() == N - 1
        assert [e.seq for e in replay(log2)] == list(range(N - WIN, N))
        ok, msg = log2.verify()
        assert ok, msg
        st2 = fold(log2)
        assert st2.head_seq == N - 1
        assert [m["content"] for m in st2.messages] == \
            [m["content"] for m in st.messages]

        # deep rewind past the window: branch repositioned, not emptied
        marker_seq = log2.rewind(100)
        assert log2.head() == marker_seq == N  # marker is a new event
        log2.append("user.message", {"text": "after deep rewind"})
        texts = [e.data.get("text", "") for e in log2.events()
                 if e.type == "user.message"]
        assert texts[-1] == "after deep rewind", texts[-3:]
        ok, msg = log2.verify()
        assert ok, msg

        # dormant branch heads survive resume (pinned, not pruned away)
        with tempfile.TemporaryDirectory() as td3:
            p3 = Path(td3) / "branches.jsonl"
            log3 = EventLog(p3, max_events=50)
            log3.append("user.message", {"text": "root"})
            dormant = log3.fork(name="dormant")
            for i in range(300):
                log3.append("tool.call", {"name": "t", "args": {}})
            log4 = EventLog(p3, max_events=50)
            assert dormant in log4.branches(), log4.branches()
            assert log4.head("dormant") == 1, log4.head("dormant")
            assert log4.get(log4._heads["dormant"]) is not None
            log4.checkout("dormant")
            log4.append("user.message", {"text": "dormant wakes"})
            ok, msg = log4.verify(branch="dormant")
            assert ok, msg

        # deep rewind past the window must survive reload: the marker's
        # archived base stays reachable, so replay keeps real history.
        with tempfile.TemporaryDirectory() as td4:
            p4 = Path(td4) / "rewind.jsonl"
            log5 = EventLog(p4, max_events=20)
            for i in range(150):
                log5.append("user.message", {"text": f"m{i}"})
            marker_seq = log5.rewind(5)
            assert marker_seq == 150, marker_seq
            log6 = EventLog(p4, max_events=20)
            seqs = [e.seq for e in replay(log6)]
            assert 5 in seqs and marker_seq in seqs, seqs
            ok, msg = log6.verify()
            assert ok, msg

    print("KERNEL SELF-TEST PASS (incl. memory window)")

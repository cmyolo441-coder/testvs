"""CASSETTE — record/replay of model calls (§20.2, §35).

Every request/response pair is recorded; a session replays against the
cassette with ZERO API cost. This is how the test suite runs, and it is
the dividend 'deterministic testing' from the Mul Bindu table (§0.2).

Design (pure Python, stdlib only):
  * Key = sha256(canonical(model, messages, tools)) — the request hash.
  * record mode: real calls pass through and are stored keyed by hash.
  * replay mode: matching requests return the stored response; a miss is a
    hard error (never a silent live call), so replays are deterministic.
  * Cassettes are JSONL, one line per pair, human-inspectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any
import copy
from ._foundation import get_logger

_log = get_logger("cassette")

# Bound the hot in-memory store: a long session with thousands of unique
# requests must not grow RAM without limit. Evicted entries stay replayable
# from the disk offset index (two-tier: hot LRU + cold disk).
_MAX_HOT = 2048


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def request_key(model: str, messages: list[dict],
                tools: list[dict] | None,
                effort_key: str | None = None) -> str:
    """The cassette key for one request (§8.1: blake3(request) — here
    sha256, same role).

    `effort_key` participates in the hash: sampling params (max_tokens,
    temperature) change with effort but NOT with messages, so two requests
    that differ only in effort must not collide on one recorded response."""
    payload = {"model": model, "messages": messages, "tools": tools or [],
               "effort": effort_key or ""}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


class Cassette:
    """Record or replay model request/response pairs.

    Two-tier storage: a bounded hot LRU in RAM plus a cold offset index
    into the JSONL file, so replay never re-reads the whole file and a
    huge cassette never blows up memory.
    """

    def __init__(self, path: Path, mode: str = "off") -> None:
        """mode: 'off' | 'record' | 'replay'."""
        if mode not in ("off", "record", "replay"):
            raise ValueError("mode must be off | record | replay")
        self.path = Path(path)
        self.mode = mode
        self._lock = threading.Lock()
        # hot tier: key -> response (LRU, bounded)
        self._store: OrderedDict[str, dict] = OrderedDict()
        # cold tier: key -> byte offset of the LAST line holding it
        self._offsets: dict[str, int] = {}
        self.hits = 0
        self.misses = 0
        self._fh = None  # persistent append handle (record mode)
        if mode in ("record", "replay") and self.path.exists():
            self._load()

    def _load(self) -> None:
        """Build the offset index in ONE pass; load nothing eagerly.

        Responses are faulted in from disk on first replay (lazy), so
        opening a gigabyte cassette is instant and RAM stays flat."""
        with self.path.open("r", encoding="utf-8") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    pair = json.loads(line)
                except ValueError:
                    continue
                key = pair.get("key")
                if key:
                    # last occurrence wins (re-recorded pairs)
                    self._offsets[key] = off

    def _read_at(self, off: int) -> dict | None:
        """Fault one response in from disk by offset."""
        try:
            with self.path.open("r", encoding="utf-8") as f:
                f.seek(off)
                pair = json.loads(f.readline())
                resp = pair.get("response")
                return resp if isinstance(resp, dict) else None
        except (OSError, ValueError):
            return None

    def _open_append(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def _persist(self, line: str) -> None:
        """Append one pre-serialized line through the persistent handle.

        Flushes to the OS on every write (survives a process crash; only a
        power loss can drop the tail). Reopens once if the handle died or
        the directory vanished underneath us."""
        try:
            if self._fh is None:
                self._open_append()
            assert self._fh is not None
            off_before = self._fh.tell()
            self._fh.write(line)
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
        except (ValueError, OSError):
            # handle closed/replaced, or directory deleted externally —
            # reopen once and retry; a cassette must not take the app down
            try:
                self._fh = None
                self._open_append()
                assert self._fh is not None
                off_before = self._fh.tell()
                self._fh.write(line)
                self._fh.flush()
            except OSError:
                # disk truly unwritable: report the offset as unknown;
                # the hot tier still has the record for this session
                return None
        return off_before

    def _store_put(self, key: str, response: dict) -> None:
        """Insert into the hot LRU, evicting the coldest entry past cap."""
        self._store[key] = copy.deepcopy(response)
        self._store.move_to_end(key)
        while len(self._store) > _MAX_HOT:
            self._store.popitem(last=False)

    # -- record ---------------------------------------------------------------

    def record(self, model: str, messages: list[dict],
               tools: list[dict] | None, response: dict,
               effort_key: str | None = None) -> None:
        """Store a real response (record mode only)."""
        if self.mode != "record":
            return
        key = request_key(model, messages, tools, effort_key)
        # deep-copy IN: a caller mutating the response afterwards must
        # not rewrite what the cassette recorded
        stored = copy.deepcopy(response)
        # serialize FIRST, before touching any state: if the response is
        # not JSON-serializable we fail loudly here instead of diverging
        # (hot tier updated, disk not) and losing the write on restart
        line = json.dumps({"key": key, "response": stored},
                          ensure_ascii=False, default=str) + "\n"
        with self._lock:
            off = self._persist(line)
            self._store_put(key, stored)
            if off is not None:
                self._offsets[key] = off

    # -- replay ---------------------------------------------------------------

    def replay(self, model: str, messages: list[dict],
               tools: list[dict] | None,
               effort_key: str | None = None) -> dict | None:
        """Return the stored response for a request (replay mode only).
        A miss returns None and counts as a miss — the caller must treat it
        as a hard error, never fall back to a live call, or the replay is
        no longer deterministic."""
        if self.mode != "replay":
            return None
        key = request_key(model, messages, tools, effort_key)
        with self._lock:
            if key in self._store:
                self.hits += 1
                self._store.move_to_end(key)  # LRU touch
                # deep-copy OUT: annotating a replayed response must not
                # corrupt every future replay of the same key
                return copy.deepcopy(self._store[key])
            off = self._offsets.get(key)
            if off is not None:
                resp = self._read_at(off)
                if resp is not None:
                    self.hits += 1
                    self._store_put(key, resp)
                    return copy.deepcopy(resp)
            self.misses += 1
            return None

    def __len__(self) -> int:
        # total known pairs (hot + cold), not just the cached ones
        return len(set(self._offsets) | set(self._store))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cassette.jsonl"
        msgs = [{"role": "user", "content": "hello"}]
        resp = {"content": "hi there", "usage": {"prompt_tokens": 5,
                                                 "completion_tokens": 3}}

        # record mode stores the pair
        rec = Cassette(path, mode="record")
        rec.record("model-x", msgs, None, resp)
        assert len(rec) == 1

        # replay mode returns it with zero API cost
        play = Cassette(path, mode="replay")
        got = play.replay("model-x", msgs, None)
        assert got == resp, got
        assert play.hits == 1 and play.misses == 0

        # a different request is a miss (deterministic: never a live call)
        other = [{"role": "user", "content": "different"}]
        assert play.replay("model-x", other, None) is None
        assert play.misses == 1

        # off mode neither records nor replays
        off = Cassette(Path(td) / "off.jsonl", mode="off")
        off.record("model-x", msgs, None, resp)
        assert off.replay("model-x", msgs, None) is None
        assert len(off) == 0

        # the cassette file is human-inspectable JSONL
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        pair = json.loads(lines[0])
        assert pair["response"] == resp and "key" in pair

        # replay survives reload from disk
        play2 = Cassette(path, mode="replay")
        assert play2.replay("model-x", msgs, None) == resp

    print("CASSETTE SELF-TEST PASS")

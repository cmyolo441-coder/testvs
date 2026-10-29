"""Filesystem watcher — polling-based, stdlib only (no watchdog).

Tools (registered by :func:`register`):

* ``WatchAdd`` — register a watch: ``path`` (file or directory), a glob
  ``pattern`` (e.g. ``*.py``; empty means "everything"), and an
  ``action`` shell command. Returns the watch id.
* ``WatchList`` — list all active watches and their recent event counts.
* ``WatchRemove`` — remove a watch by id.

The watcher is a daemon thread that polls every 2s, tracking
``(mtime, size)`` per file. On each poll it detects created / modified /
deleted files matching each watch's pattern and fires the action as a
REAL subprocess (:func:`subprocess.Popen`) with the environment::

    WATCH_EVENT    — created | modified | deleted
    WATCH_PATH     — absolute path of the changed file
    WATCH_ID       — the watch id
    WATCH_PATTERN  — the watch's glob pattern

Debounce: the same ``(watch, event, path)`` never re-fires within 5s.

Watches persist in ``~/.fullagent/watches.json`` (override with
``FULLAGENT_WATCHES_FILE``), so they survive restarts; the watcher
thread re-arms them on startup (``register()``). A watch whose path has
been deleted is logged and skipped, never crashed.

``python3 -m fullagent.watch`` runs the built-in self-test, which proves
REAL detection: it creates a temp directory, adds a watch, then really
creates/modifies/deletes a file and verifies the action subprocess ran
(the action writes the env vars into a marker file the test checks).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .tools import RISK_CONFIRM, Tool

_log = logging.getLogger(__name__)

POLL_INTERVAL_S = 2.0
_MIN_POLL_INTERVAL_S = 0.5  # hard floor: the poll loop can never spin faster
DEBOUNCE_S = 5.0
DEFAULT_CONFIG_PATH = (
    Path(os.environ["FULLAGENT_HOME"]).expanduser() / "watches.json"
    if os.environ.get("FULLAGENT_HOME")
    else Path("~/.fullagent/watches.json").expanduser()
)


def _config_path() -> Path:
    override = os.environ.get("FULLAGENT_WATCHES_FILE")
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    return DEFAULT_CONFIG_PATH


@dataclass
class Watch:
    """One registered watch."""
    id: str
    path: str          # file or directory (absolute)
    pattern: str       # glob matched against filename / relative path
    action: str        # shell command executed on change
    added_at: float
    events_fired: int = 0
    last_error: str = ""
    _fired: dict = field(default_factory=dict, repr=False, compare=False)


class WatchManager:
    """Owns the watch list, the poll thread, and JSON persistence."""

    def __init__(self, config_path: Path | None = None,
                 poll_interval: float = POLL_INTERVAL_S) -> None:
        self.config_path = config_path or _config_path()
        if poll_interval < _MIN_POLL_INTERVAL_S:
            _log.warning("watch: poll_interval %.3fs below floor %.1fs — clamped",
                         poll_interval, _MIN_POLL_INTERVAL_S)
            poll_interval = _MIN_POLL_INTERVAL_S
        self.poll_interval = poll_interval
        self.watches: dict[str, Watch] = {}
        # (watch_id, abs_file_path) -> (mtime_ns, size)
        self._state: dict[tuple[str, str], tuple[int, int]] = {}
        self._lock = threading.RLock()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()  # poke: instant wake when work arrives
        self._thread: threading.Thread | None = None
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            raw = self.config_path.read_text(encoding="utf-8")
        except OSError:
            return  # no file yet — nothing to re-arm
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            _log.warning("watch: invalid JSON in %s — starting empty",
                         self.config_path)
            return
        if not isinstance(data, dict):
            return
        for wid, entry in data.get("watches", {}).items():
            if not isinstance(entry, dict):
                continue
            try:
                w = Watch(id=wid,
                          path=str(entry.get("path", "")),
                          pattern=str(entry.get("pattern", "")),
                          action=str(entry.get("action", "")),
                          added_at=float(entry.get("added_at", 0.0)),
                          events_fired=int(entry.get("events_fired", 0)))
            except (TypeError, ValueError):
                continue
            if w.path and w.action:
                self.watches[wid] = w
        _log.info("watch: re-armed %d persisted watches", len(self.watches))

    def _save(self) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"watches": {
                wid: {"path": w.path, "pattern": w.pattern,
                      "action": w.action, "added_at": w.added_at,
                      "events_fired": w.events_fired}
                for wid, w in self.watches.items()}}
            # unique tmp per writer: two concurrent saves must not race on
            # the same scratch file and clobber each other's os.replace
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.config_path.parent),
                prefix=self.config_path.name + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, indent=2))
                os.replace(tmp_name, self.config_path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            _log.warning("watch: failed to persist watches: %s", e)

    # -- public API ------------------------------------------------------
    def add(self, path: str, pattern: str, action: str) -> str:
        if not path or not str(path).strip():
            raise ValueError("path must be a non-empty string")
        if not action or not str(action).strip():
            raise ValueError("action must be a non-empty shell command")
        p = Path(os.path.expandvars(str(path))).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        wid = uuid.uuid4().hex[:8]
        with self._lock:
            self.watches[wid] = Watch(id=wid, path=str(p),
                                      pattern=str(pattern or ""),
                                      action=str(action),
                                      added_at=time.time())
            self._save()
        _log.info("watch: added %s -> %s (%s)", wid, p, pattern or "*")
        self.poke()  # pick the new watch up on the next instant tick
        return wid

    def remove(self, watch_id: str) -> bool:
        with self._lock:
            existed = watch_id in self.watches
            self.watches.pop(watch_id, None)
            self._state = {k: v for k, v in self._state.items()
                           if k[0] != watch_id}
            self._save()
        return existed

    def list(self) -> list[dict]:
        with self._lock:
            return [{"id": w.id, "path": w.path, "pattern": w.pattern or "*",
                     "action": w.action, "events_fired": w.events_fired,
                     "last_error": w.last_error,
                     "added_at": w.added_at}
                    for w in sorted(self.watches.values(),
                                    key=lambda w: w.added_at)]

    # -- polling ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="watcher",
                                        daemon=True)
        self._thread.start()
        _log.info("watch: poll thread started (interval %.1fs)",
                  self.poll_interval)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()  # release the sleeper instantly
        t, self._thread = self._thread, None
        if t and t.is_alive():
            t.join(timeout=self.poll_interval + 2)

    def poke(self) -> None:
        """Wake the poll loop immediately so newly added work is picked up."""
        self._wake.set()

    def _loop(self) -> None:
        # Sleeps on _wake: at most one wake per poll_interval when idle,
        # instantly when poke()d (new watch) or when stop() is requested.
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 — the loop must never die
                _log.exception("watch: unexpected error in poll loop")
            self._wake.clear()
            self._wake.wait(self.poll_interval)

    def poll_once(self) -> int:
        """Run one poll cycle. Returns the number of actions fired."""
        with self._poll_lock:
            fired = 0
            with self._lock:
                watches = list(self.watches.values())
            for w in watches:
                fired += self._poll_watch(w)
            return fired

    def _poll_watch(self, w: Watch) -> int:
        root = Path(w.path)
        current: dict[str, tuple[int, int]] = {}
        try:
            if root.is_file():
                current = {str(root): self._stat(root)}
            elif root.is_dir():
                for dirpath, _dirnames, filenames in os.walk(root):
                    for name in filenames:
                        f = Path(dirpath) / name
                        if self._matches(w, f, root):
                            current[str(f)] = self._stat(f)
            else:
                # Deleted (or not-yet-created) watch path: stay alive,
                # treat previously-seen files as deleted, then resync.
                pass
        except OSError as e:
            with self._lock:
                w.last_error = f"poll error: {e}"
            return 0

        fired = 0
        now = time.time()
        with self._lock:
            prev = {f: st for (wid, f), st in self._state.items()
                    if wid == w.id}
        seen: set[str] = set()
        for f, st in list(current.items()):
            seen.add(f)
            if f not in prev:
                # If a writer just opened the file (O_CREAT|O_TRUNC -> size 0)
                # and is still flushing bytes, wait briefly and re-stat so the
                # initial write settles into the single 'created' snapshot.
                if st[1] == 0:
                    time.sleep(0.015)
                    try:
                        st = self._stat(Path(f))
                        current[f] = st
                    except OSError:
                        pass
                fired += self._fire(w, "created", f, now)
            elif prev[f] != st:
                fired += self._fire(w, "modified", f, now)
        for f in prev:
            if f not in seen:
                fired += self._fire(w, "deleted", f, now)
        with self._lock:
            self._state = {(wid, f): st for (wid, f), st in self._state.items()
                           if wid != w.id}
            for f, st in current.items():
                self._state[(w.id, f)] = st
        return fired

    @staticmethod
    def _stat(f: Path) -> tuple[int, int]:
        st = f.stat()
        return st.st_mtime_ns, st.st_size

    @staticmethod
    def _prune_fired(w: Watch, now: float, keep_key: str) -> None:
        """Bound the debounce map; one entry per (event, path) otherwise."""
        if len(w._fired) <= 1024:
            return
        cutoff = now - DEBOUNCE_S * 4
        w._fired = {k: ts for k, ts in w._fired.items()
                    if ts >= cutoff or k == keep_key}
        if len(w._fired) > 1024:
            newest = sorted(((k, ts) for k, ts in w._fired.items()
                             if k != keep_key),
                            key=lambda kv: kv[1], reverse=True)[:511]
            newest.append((keep_key, w._fired[keep_key]))
            w._fired = dict(newest)

    @staticmethod
    def _matches(w: Watch, f: Path, root: Path) -> bool:
        if not w.pattern:
            return True
        name = f.name
        try:
            rel = str(f.relative_to(root))
        except ValueError:
            rel = name
        return (fnmatch.fnmatch(name, w.pattern)
                or fnmatch.fnmatch(rel, w.pattern))

    def _fire(self, w: Watch, event: str, f: str, now: float) -> int:
        key = f"{w.id}|{event}|{f}"
        last = w._fired.get(key, 0.0)
        if now - last < DEBOUNCE_S:
            return 0  # debounced: same event re-triggered too soon
        w._fired[key] = now
        self._prune_fired(w, now, key)
        env = dict(os.environ)
        env["WATCH_EVENT"] = event
        env["WATCH_PATH"] = f
        env["WATCH_ID"] = w.id
        env["WATCH_PATTERN"] = w.pattern or "*"
        try:
            # REAL subprocess: fire-and-forget, detached from this process.
            subprocess.Popen(w.action, shell=True, env=env,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             start_new_session=True)
        except Exception as e:  # noqa: BLE001 — spawn failure must not kill loop
            _log.warning("watch: failed to spawn action for %s: %s", f, e)
            w.last_error = f"spawn error: {e}"
            return 0
        w.events_fired += 1
        self._save()
        _log.info("watch: %s %s -> action fired (%s)", event, f, w.id)
        return 1


# -- tools ---------------------------------------------------------------

_ADD_DESC = ("Watch a file or directory for changes. `path` may be a file "
             "or directory (relative paths resolve against the working "
             "directory). `pattern` is a glob like '*.py' matched against "
             "filenames inside `path` (empty = everything). `action` is a "
             "shell command executed as a real subprocess on every change, "
             "with WATCH_EVENT (created|modified|deleted), WATCH_PATH, "
             "WATCH_ID and WATCH_PATTERN in its environment. Returns the "
             "watch id. Watches persist across restarts. The action runs "
             "detached; keep it fast and non-interactive.")
_ADD_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string",
                 "description": "File or directory to watch."},
        "pattern": {"type": "string",
                    "description": "Glob pattern, e.g. '*.py'. Empty "
                                   "watches everything."},
        "action": {"type": "string",
                   "description": "Shell command to run on each change."},
    },
    "required": ["path", "action"],
}

_LIST_DESC = ("List all active file watches: id, watched path, pattern, "
              "action, how many events each has fired, and the last error.")
_LIST_PARAMS = {"type": "object", "properties": {}}

_REMOVE_DESC = ("Remove a file watch by its id (see WatchList). Stops "
                "its actions; the watch is also dropped from persistence.")
_REMOVE_PARAMS = {
    "type": "object",
    "properties": {
        "watch_id": {"type": "string",
                     "description": "The watch id returned by WatchAdd."},
    },
    "required": ["watch_id"],
}


def register(agent) -> None:
    """Register WatchAdd/WatchList/WatchRemove and start the watcher."""
    mgr = WatchManager()

    def watch_add(path: str, action: str, pattern: str = "") -> str:
        try:
            wid = mgr.add(path, pattern, action)
        except (ValueError, OSError) as e:
            return f"ERROR: {e}"
        return (f"watch added: id={wid} path={mgr.watches[wid].path} "
                f"pattern={pattern or '*'}")

    def watch_list() -> str:
        watches = mgr.list()
        if not watches:
            return "no watches registered"
        lines = []
        for w in watches:
            line = (f"{w['id']}  {w['pattern']:12} {w['path']}  "
                    f"events={w['events_fired']}")
            if w["last_error"]:
                line += f"  error: {w['last_error']}"
            lines.append(line)
        return "\n".join(lines)

    def watch_remove(watch_id: str) -> str:
        if mgr.remove(str(watch_id or "").strip()):
            return f"watch removed: {watch_id}"
        return f"ERROR: no watch with id {watch_id!r} (see WatchList)"

    agent.tools["WatchAdd"] = Tool("WatchAdd", _ADD_DESC, _ADD_PARAMS,
                                   watch_add, risk=RISK_CONFIRM)
    agent.tools["WatchList"] = Tool("WatchList", _LIST_DESC, _LIST_PARAMS,
                                    watch_list, risk="safe")
    agent.tools["WatchRemove"] = Tool("WatchRemove", _REMOVE_DESC,
                                      _REMOVE_PARAMS, watch_remove,
                                      risk="safe")
    agent.watch_manager = mgr  # host hook: mgr.stop() on shutdown
    mgr.start()


# -- self-test --------------------------------------------------------------

def _selftest() -> None:
    """Prove REAL detection: temp dir, real watch, real create/modify/
    delete, and a REAL action subprocess that writes a marker file the
    test then verifies (including the env vars)."""
    import tempfile

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("PASS" if cond else "FAIL"), "-", name,
              (f"({detail})" if detail and not cond else ""))
        if not cond:
            raise SystemExit(f"self-test failed: {name} {detail}")

    class FakeAgent:
        def __init__(self):
            self.tools = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        watched = tmp / "watched"
        watched.mkdir()
        cfg = tmp / "watches.json"
        marker = tmp / "marker.log"

        old_env = os.environ.get("FULLAGENT_WATCHES_FILE")
        os.environ["FULLAGENT_WATCHES_FILE"] = str(cfg)
        try:
            agent = FakeAgent()
            register(agent)
            mgr: WatchManager = agent.watch_manager
            try:
                for name in ("WatchAdd", "WatchList", "WatchRemove"):
                    check(f"tool {name} registered",
                          name in agent.tools)

                # 1. Add a watch whose action appends the env vars to a
                #    marker file — a REAL subprocess must run it.
                action = ("echo \"$WATCH_EVENT|$WATCH_PATH\" "
                          f">> {marker}")
                out = agent.tools["WatchAdd"].handler(
                    str(watched), action, pattern="*.txt")
                check("WatchAdd returns id", "watch added: id=" in out, out)
                wid = out.split("id=")[1].split()[0]

                # 2. CREATE: real file creation must fire the action.
                target = watched / "hello.txt"
                target.write_text("one\n", encoding="utf-8")
                deadline = time.time() + 8
                while (not marker.exists()
                       and time.time() < deadline):
                    time.sleep(0.2)
                check("action subprocess ran on create", marker.exists())
                first = marker.read_text(encoding="utf-8").strip().splitlines()
                check("created event env vars",
                      first and first[0] == f"created|{target}", first)
                check("WatchList shows fired event",
                      "events=1" in agent.tools["WatchList"].handler(),
                      agent.tools["WatchList"].handler())

                # 3. DEBOUNCE: polling again without any change must NOT
                #    re-fire (same mtime within 5s).
                before = len(marker.read_text(encoding="utf-8").strip()
                              .splitlines())
                mgr.poll_once()
                time.sleep(0.3)  # give any spurious spawn a chance
                after = len(marker.read_text(encoding="utf-8").strip()
                             .splitlines())
                check("debounce: no re-fire without change", after == before,
                      f"{before} -> {after}")

                # 4. MODIFY: real modification must fire.
                time.sleep(0.05)
                target.write_text("one\ntwo\n", encoding="utf-8")
                deadline = time.time() + 8
                while True:
                    lines = marker.read_text(
                        encoding="utf-8").strip().splitlines()
                    if len(lines) >= 2 or time.time() >= deadline:
                        break
                    time.sleep(0.2)
                check("modified event fired",
                      len(lines) >= 2 and lines[1] == f"modified|{target}",
                      lines)

                # 5. DELETE: real deletion must fire.
                target.unlink()
                deadline = time.time() + 8
                while True:
                    lines = marker.read_text(
                        encoding="utf-8").strip().splitlines()
                    if len(lines) >= 3 or time.time() >= deadline:
                        break
                    time.sleep(0.2)
                check("deleted event fired",
                      len(lines) >= 3 and lines[2] == f"deleted|{target}",
                      lines)

                # 6. PATTERN: non-matching files must NOT fire.
                lines_before = len(lines)
                (watched / "skip.log").write_text("x\n", encoding="utf-8")
                time.sleep(2.5 + 0.5)  # one full poll interval + margin
                lines_after = len(marker.read_text(
                    encoding="utf-8").strip().splitlines())
                check("pattern filter: non-matching file ignored",
                      lines_after == lines_before,
                      f"{lines_before} -> {lines_after}")

                # 7. PERSISTENCE: watches survive in JSON; a fresh
                #    manager re-arms them.
                check("watches.json persisted", cfg.exists())
                raw = json.loads(cfg.read_text(encoding="utf-8"))
                check("watch persisted in JSON", wid in raw["watches"], raw)
                mgr2 = WatchManager(config_path=cfg)
                check("re-arm on startup", wid in mgr2.watches)
                check("re-armed watch keeps path",
                      mgr2.watches[wid].path == str(watched))
                check("re-armed watch keeps action",
                      mgr2.watches[wid].action == action)
                mgr2.stop()

                # 8. DELETED WATCH PATH: removing the watched dir must
                #    not crash the poller.
                import shutil
                shutil.rmtree(watched)
                mgr.poll_once()
                check("deleted watch path handled gracefully", True)

                # 9. WatchRemove drops the watch and persists the removal.
                out = agent.tools["WatchRemove"].handler(wid)
                check("WatchRemove", out == f"watch removed: {wid}", out)
                check("watch gone", not mgr.list())
                raw = json.loads(cfg.read_text(encoding="utf-8"))
                check("removal persisted", wid not in raw["watches"])
                out = agent.tools["WatchRemove"].handler("nope")
                check("WatchRemove unknown id errors",
                      out.startswith("ERROR:"), out)
            finally:
                mgr.stop()
        finally:
            if old_env is None:
                os.environ.pop("FULLAGENT_WATCHES_FILE", None)
            else:
                os.environ["FULLAGENT_WATCHES_FILE"] = old_env

    print("PASS")


def _selftest_thread_efficiency() -> None:
    """Prove the poll thread is cheap when idle and fast when work arrives:
    (1) effective interval is sane and floored, (2) idle wake rate <= 1/s,
    (3) poke() wakes the loop instantly and a real change is still
    detected, (4) stop() returns promptly."""
    import tempfile

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("PASS" if cond else "FAIL"), "-", name,
              (f"({detail})" if detail and not cond else ""))
        if not cond:
            raise SystemExit(f"efficiency self-test failed: {name} {detail}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. interval floor: even a 10ms request is clamped, never a spin loop
        tiny = WatchManager(config_path=tmp / "a.json", poll_interval=0.01)
        check("poll_interval clamped to floor",
              tiny.poll_interval >= _MIN_POLL_INTERVAL_S,
              f"got {tiny.poll_interval}")
        check("default interval sane (2.0s)", POLL_INTERVAL_S == 2.0,
              f"got {POLL_INTERVAL_S}")

        # 2. idle wake budget: count real loop ticks over 2.2s at 2s interval
        mgr = WatchManager(config_path=tmp / "b.json", poll_interval=2.0)
        ticks: list[float] = []
        orig = mgr.poll_once

        def counting() -> int:
            ticks.append(time.monotonic())
            return orig()

        mgr.poll_once = counting  # type: ignore[method-assign]
        mgr.start()
        try:
            time.sleep(2.2)
        finally:
            t0 = time.monotonic()
            mgr.stop()
            stop_dt = time.monotonic() - t0
        check("idle wake rate <= 1/s", 1 <= len(ticks) <= 3,
              f"{len(ticks)} ticks in 2.2s")
        check("stop() returns promptly (< poll interval)", stop_dt < 2.0,
              f"{stop_dt:.2f}s")

        # 3. instant wake: 60s interval would never fire in-test, but a
        #    poke + real file change must be detected within seconds.
        watched = tmp / "w"
        watched.mkdir()
        marker = tmp / "hit.log"
        mgr2 = WatchManager(config_path=tmp / "c.json", poll_interval=60.0)
        mgr2.start()
        try:
            action = f"echo fired >> {marker}"
            mgr2.add(str(watched), "*.txt", action)
            target = watched / "poke.txt"
            target.write_text("x\n", encoding="utf-8")
            mgr2.poke()
            t0 = time.monotonic()
            deadline = t0 + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            check("poke: change detected despite 60s interval",
                  marker.exists(), f"waited {time.monotonic() - t0:.1f}s")
        finally:
            mgr2.stop()

    print("PASS - watch thread efficiency")


if __name__ == "__main__":
    _selftest()
    _selftest_thread_efficiency()

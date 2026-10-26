"""`/bots` and Ctrl+S: this machine's Hermes bots on one read-only board, full screen in the pager.

The board lists the Hermes home this TUI runs under, resolved as the input
history resolves it (`history.history_path`), and the root's every named
profile (`<root>/profiles/<name>/` that carries one of the files Hermes reads
as a profile's identity and has no deletion tombstone). It has three levels,
Enter going down one and ← or Esc up one:

1. the profiles: each one's gateway (`gateway_state.json`: its state, pid,
   running agents and platforms), its stored sessions counted by source, how
   many were active in the last 24 hours, and when the last one was;
2. one profile's sessions, newest activity first: title, source, last
   activity and message count. Delegated children (`source = 'subagent'`)
   are counted on level 1 and not listed;
3. one session's last `MESSAGE_ROWS_READ` messages the model still sees,
   oldest of them first: time, role and the start of the text (one row for a
   tool result, a few for the talk).

"Active in the last 24 hours" is what the board counts instead of open
sessions: a messaging gateway leaves `ended_at` unset on every chat it
served, so a Slack bot idle for weeks still reads as dozens of open sessions.
A session's activity is `last_activity_at`, else its start.

A gateway reads as running only when the process its record names is still
the one that wrote it. The pid is probed with signal 0 and nothing else, its
start-time fingerprint read as Hermes reads it (`/proc/<pid>/stat` field 22
on Linux, the kernel's start time in centiseconds on macOS, which is psutil's
`create_time() * 100`) and compared within Hermes' drift tolerance; a zombie,
or a recorded start time the process's own cannot be compared with, is not
running; and the record must name the home it was read from. A platform's state counts as now
only when the entry was stamped by that same process (`writer_pid` and
`writer_start_time`, Hermes' own ownership rule); otherwise it shows as what
was last recorded. A profile whose own gateway is not running but which the
default home's running gateway lists in `served_profiles` reads as served
by it.

Everything is read off the UI thread (`BoardReader`), every `BOARD_REFRESH_SECONDS`
and at once when the person moves, so a locked or huge `state.db` never holds a
frame. Each `state.db` is opened read-only (`connect_readonly`, a quarter-second
busy wait), only when it exists, and immutable when no connection has it open
(`_no_connection_open`). Only the sessions table is read whole, a row per session;
messages are read by the session's index with a `LIMIT`, their text cut in the
query. Nothing here writes a file, opens a socket, calls the gateway, or signals
a process beyond that probe; a read of a store a connection has open takes its
read mark in that store's existing `-shm`, as every reader of it does. Every text
the board shows from those files is a bot's or a user's and is cleaned as
gateway text is (`markdown.clean_text`), whitespace folded to one line.

Keys: ↑↓ (j / k) move, PageUp / PageDown a page, g / G either end, Enter or →
open, ← / Esc / q back (on level 1, close), r read again now. On level 3 the
arrows scroll the messages, which open at the newest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
import threading
import time
from typing import Any, ClassVar

from ..plugin_bundle.omh.sqlite_readonly import connect_readonly
from .history import history_path
from .keys import CHAR, DOWN, END, ENTER, ESCAPE, HOME, LEFT, PAGE_DOWN, PAGE_UP, RIGHT, UP, Key, Paste
from .markdown import clean_text
from .pager import Pager
from .prompts import CHOICE_MARK
from .session_picker import age_text
from .style import Line, Span, Style
from .theme import Theme
from .width import cell_width, truncate, truncate_line, wrap_line

BOARD_TITLE = "Hermes bots"
BOARD_KEYS = "↑↓ move  Enter open  ← back  r refresh  q close"
MESSAGE_KEYS = "↑↓ scroll  g/G ends  ← back  r refresh  q close"
BOARD_REFRESH_SECONDS = 5.0
ACTIVE_WINDOW_SECONDS = 24 * 3600
DEFAULT_PROFILE = "default"
SUBAGENT_SOURCE = "subagent"
PROFILE_CAP = 64
SESSION_ROWS_READ = 200
MESSAGE_ROWS_READ = 40
MESSAGE_CHARS_READ = 2000  # cut in the query: a tool result can be megabytes
TOOL_CALLS_CHARS_READ = 4000
MESSAGE_SHOWN_CHARS = 400
MESSAGE_SHOWN_ROWS = 3
GATEWAY_STATE_FILE = "gateway_state.json"
GATEWAY_STATE_BYTES = 1 << 20
STATE_DB = "state.db"
STATE_DB_BUSY_SECONDS = 0.25
# The files Hermes reads as a profile's identity (`hermes_constants._PROFILE_IDENTITY_MARKERS`).
PROFILE_MARKERS = ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db")
DELETED_PROFILES = ".deleted"  # where Hermes keeps a deleted profile's tombstone, beside the profile dirs
# Hermes' `START_TIME_DRIFT_TOLERANCE`: 2 s on either fingerprint scale (Linux ticks, macOS centiseconds).
START_TIME_TOLERANCE = 200
INDENT = "    "
PID_LIMIT = 1 << 31
# Signal 0 is a probe only on POSIX: on Windows `os.kill(pid, 0)` sends CTRL_C_EVENT to the console group.
ON_POSIX = os.name == "posix"
# sysctl names and `kinfo_proc` layout on macOS (sys/sysctl.h, sys/proc.h), measured against psutil on arm64.
CTL_KERN, KERN_PROC, KERN_PROC_PID = 1, 14, 1
KINFO_PROC_BYTES = 1024  # room for the struct (648 bytes on arm64 and x86_64)
KINFO_STAT_OFFSET = 36  # `kp_proc.p_stat`, after the start-time union, two pointers and `p_flag`
SZOMB = 5

GATEWAY_RUNNING = "running"
GATEWAY_STOPPED = "stopped"


@dataclass(frozen=True)
class Platform:
    name: str
    state: str
    error: str = ""
    needs_attention: bool = False
    current: bool = False  # stamped by the gateway process that runs now


@dataclass(frozen=True)
class Gateway:
    state: str  # the record's `gateway_state`; "" with no readable record
    live: bool  # its process runs and is the one that wrote the record
    pid: int | None = None
    active_agents: int = 0
    exit_reason: str = ""
    elsewhere: str = ""  # the other home the record names; its pid means nothing here
    served: tuple[str, ...] = ()
    platforms: tuple[Platform, ...] = ()
    problem: str = ""  # why there is no record to read: missing, or unreadable


@dataclass(frozen=True)
class Profile:
    name: str
    home: Path
    gateway: Gateway
    sources: tuple[tuple[str, int], ...] = ()  # session counts by source, most first
    active: int = 0  # sessions with activity in the last 24 hours
    last_activity: float | None = None
    store: str = ""  # why the sessions could not be read; "" when they were
    this_tui: bool = False


@dataclass(frozen=True)
class SessionRow:
    id: str
    title: str
    source: str
    last_activity: float
    message_count: int


@dataclass(frozen=True)
class MessageRow:
    role: str
    text: str
    at: float


@dataclass(frozen=True)
class BoardView:
    """What the board shows beyond the profiles: one profile's sessions, and one of its sessions' messages."""

    profile: str = ""
    session: str = ""


@dataclass(frozen=True)
class BoardSnapshot:
    view: BoardView
    profiles: tuple[Profile, ...]
    sessions: tuple[SessionRow, ...] = ()  # one profile's, or, for a session's messages, that session's row
    unlisted: int = 0  # subagent sessions the profile has, not listed
    messages: tuple[MessageRow, ...] = ()
    problem: str = ""  # why the view's sessions or messages could not be read
    read_at: float = 0.0


# ── where the homes are ───────────────────────────────────────────────────


def board_homes(environ: Mapping[str, str]) -> tuple[Path | None, list[tuple[str, Path]]]:
    """This TUI's Hermes home and every home on the board: the root (`default`), then each named profile by name."""
    history = history_path(environ)
    if history is None:
        return None, []
    current = history.parent
    root = current.parent.parent if current.parent.name == "profiles" else current
    homes = [(DEFAULT_PROFILE, root)]
    try:
        entries = sorted((root / "profiles").iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if len(homes) > PROFILE_CAP:
            break
        try:
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if (entry.parent / DELETED_PROFILES / entry.name).exists():
                continue
            if not any((entry / marker).is_file() for marker in PROFILE_MARKERS):
                continue
        except OSError:
            continue
        homes.append((entry.name, entry))
    return current, homes


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except (OSError, RuntimeError):
        return first == second


# ── the gateway record ────────────────────────────────────────────────────


def _pid(value: Any) -> int | None:
    # Never below 2 (signal 0 to 0 or -1 probes a process group or every process, and 1 is init), nor past a pid_t.
    return value if isinstance(value, int) and not isinstance(value, bool) and 1 < value < PID_LIMIT else None


def _fingerprint_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(value)


def pid_exists(pid: int) -> bool:
    """Whether a process has `pid`: signal 0, which delivers nothing. Never called off POSIX."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process: it exists
    except OSError:
        return False
    return True


def process_facts(pid: int) -> tuple[int | None, bool]:
    """The process's start time on Hermes' fingerprint scale (None where it cannot be read), and whether it is a
    zombie, which Hermes' own `_pid_exists` counts as dead: it can no longer be a running gateway."""
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            return int(fields[19]), fields[0] == "Z"
        except (OSError, IndexError, ValueError):
            return None, False
    if sys.platform == "darwin":
        return _darwin_process_facts(pid)
    return None, False


def _darwin_process_facts(pid: int) -> tuple[int | None, bool]:
    """`sysctl(KERN_PROC_PID)`'s `kinfo_proc`, as psutil reads it: `p_starttime` in centiseconds (psutil's
    `create_time() * 100`) and `p_stat`.

    Read for any user's process, unlike `proc_pidinfo`, which answers nothing
    for another user's: a gateway pid the OS gave to such a process must still
    be told from the gateway. `p_starttime` is the struct's first field and
    `p_stat` (`SZOMB` for a zombie) is at `KINFO_STAT_OFFSET`.
    """
    import ctypes

    try:
        libc = ctypes.CDLL("/usr/lib/libc.dylib", use_errno=True)
    except OSError:
        return None, False
    mib = (ctypes.c_int * 4)(CTL_KERN, KERN_PROC, KERN_PROC_PID, pid)
    buffer = ctypes.create_string_buffer(KINFO_PROC_BYTES)
    size = ctypes.c_size_t(KINFO_PROC_BYTES)
    if libc.sysctl(mib, 4, buffer, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0 or size.value <= KINFO_STAT_OFFSET:
        return None, False  # no such process: the kernel answers with nothing
    seconds, microseconds = struct.unpack_from("qi", buffer.raw, 0)
    return round((seconds + microseconds / 1_000_000) * 100), buffer.raw[KINFO_STAT_OFFSET] == SZOMB


def process_is_record(pid: int | None, recorded_start: Any) -> bool:
    """Whether `pid` runs and is the process that recorded `recorded_start`.

    As Hermes' own liveness check, a zombie is dead and a start time further
    from the recorded one than its tolerance is a pid the OS gave to another
    process. Stricter than it in one case: a recorded start time that cannot
    be compared, because the process's own cannot be read, is not taken as a
    match, so a board never calls a reused pid a running gateway. Only a
    record without a start time is taken on its pid alone.
    """
    if pid is None or not ON_POSIX or not pid_exists(pid):
        return False
    current, zombie = process_facts(pid)
    if zombie:
        return False
    recorded = _fingerprint_value(recorded_start)
    if recorded is None:
        return True
    return current is not None and abs(current - recorded) <= START_TIME_TOLERANCE


def _text(value: Any) -> str:
    """Text from a Hermes file, cleaned as gateway text is and folded to one line."""
    return " ".join(clean_text(value if isinstance(value, str) else "").split())


def read_gateway(home: Path) -> Gateway:
    """The gateway `<home>/gateway_state.json` records, and whether its process still runs."""
    path = home / GATEWAY_STATE_FILE
    try:
        if not path.is_file():
            return Gateway("", False, problem="no gateway record")
        with path.open("rb") as handle:
            record = json.loads(handle.read(GATEWAY_STATE_BYTES))
    except (OSError, ValueError, RecursionError):
        return Gateway("", False, problem="gateway record unreadable")
    if not isinstance(record, dict):
        return Gateway("", False, problem="gateway record unreadable")
    pid = _pid(record.get("pid"))
    named = record.get("hermes_home")
    elsewhere = isinstance(named, str) and bool(named) and not _same_path(Path(named), home)
    live = not elsewhere and process_is_record(pid, record.get("start_time"))
    platforms: list[Platform] = []
    entries = record.get("platforms")
    for name, entry in sorted(entries.items())[:PROFILE_CAP] if isinstance(entries, dict) else ():
        if not isinstance(entry, dict):
            continue
        stamped = entry.get("writer_pid") == pid and entry.get("writer_start_time") == record.get("start_time")
        platforms.append(Platform(
            _text(name), _text(entry.get("state")) or "unknown", _text(entry.get("error_message")),
            entry.get("needs_attention") is True, live and stamped,
        ))
    agents = record.get("active_agents")
    served = record.get("served_profiles")
    return Gateway(
        _text(record.get("gateway_state")) or "unknown", live, pid,
        agents if isinstance(agents, int) and not isinstance(agents, bool) and agents > 0 else 0,
        _text(record.get("exit_reason")),
        _text(named) if elsewhere else "",
        tuple(_text(name) for name in served if isinstance(name, str)) if isinstance(served, list) else (),
        tuple(platforms),
    )


# ── the session store ─────────────────────────────────────────────────────


def _store_problem(error: sqlite3.Error) -> str:
    code = getattr(error, "sqlite_errorcode", None)
    return "state.db busy" if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) else "state.db unreadable"


def _no_connection_open(path: Path) -> bool:
    """Whether no connection has the store at `path` open, read from the files beside it alone.

    A WAL store (every Hermes `state.db`) keeps a `-wal` and a `-shm` while any
    connection has it open, and the last one to close removes them; a
    rollback-journal store has a `-journal` only while it writes. A `mode=ro`
    open of a WAL store without them would create both and leave them there,
    so a store with none of them is opened immutable instead. A writer that
    opens it during the read can make that read fail, which costs one
    refresh; it cannot change the store. The store itself is never opened
    here: closing any descriptor on it would drop every SQLite lock this
    process holds on it (POSIX record locks are per process), the HUD reader's
    included.
    """
    try:
        return not any(path.with_name(path.name + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
    except OSError:
        return False


def _open_store(home: Path) -> sqlite3.Connection | str:
    """A read-only connection to `<home>/state.db`, or why there is none; no file is created beside it, or for it."""
    path = home / STATE_DB
    try:
        if not path.is_file():
            return "no state.db"
        connection = connect_readonly(path, timeout=STATE_DB_BUSY_SECONDS, immutable=_no_connection_open(path))
    except OSError:
        return "state.db unreadable"
    except sqlite3.Error as error:
        return _store_problem(error)
    # A row whose text is not UTF-8 is shown with replacement characters rather than failing the whole read.
    connection.text_factory = lambda raw: raw.decode("utf-8", "replace")
    return connection


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _activity(columns: set[str]) -> str:
    # One of two fixed literals, never caller text.
    return "COALESCE(last_activity_at, started_at)" if "last_activity_at" in columns else "started_at"


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(value) else 0.0


def read_profile(name: str, home: Path, current: Path | None, now: float) -> Profile:
    """One profile's row: its gateway, and its sessions counted by source from `state.db`."""
    gateway = read_gateway(home)
    this_tui = current is not None and _same_path(current, home)
    opened = _open_store(home)
    if isinstance(opened, str):
        return Profile(name, home, gateway, store=opened, this_tui=this_tui)
    try:
        activity = _activity(_columns(opened, "sessions"))
        rows = opened.execute(
            f"SELECT source, COUNT(*), SUM(CASE WHEN {activity} >= ? THEN 1 ELSE 0 END), MAX({activity}) "
            "FROM sessions GROUP BY source",
            (now - ACTIVE_WINDOW_SECONDS,),
        ).fetchall()
    except sqlite3.Error as error:
        return Profile(name, home, gateway, store=_store_problem(error), this_tui=this_tui)
    finally:
        opened.close()
    sources = sorted(((_text(row[0]) or "unknown", int(_number(row[1]))) for row in rows), key=lambda item: (-item[1], item[0]))
    last = max((_number(row[3]) for row in rows), default=0.0)
    return Profile(
        name, home, gateway, tuple(sources), sum(int(_number(row[2])) for row in rows), last or None, this_tui=this_tui,
    )


def read_sessions(home: Path) -> tuple[tuple[SessionRow, ...], int, str]:
    """A profile's sessions but delegated children, newest activity first; how many children; why none, if so."""
    opened = _open_store(home)
    if isinstance(opened, str):
        return (), 0, opened
    try:
        select, activity = _session_select(opened)
        rows = opened.execute(
            f"{select} WHERE source != ? ORDER BY {activity} DESC LIMIT ?", (SUBAGENT_SOURCE, SESSION_ROWS_READ),
        ).fetchall()
        unlisted = opened.execute("SELECT COUNT(*) FROM sessions WHERE source = ?", (SUBAGENT_SOURCE,)).fetchone()[0]
    except sqlite3.Error as error:
        return (), 0, _store_problem(error)
    finally:
        opened.close()
    return tuple(_session_row(row) for row in rows), int(_number(unlisted)), ""


def _session_select(connection: sqlite3.Connection) -> tuple[str, str]:
    """The `SELECT ... FROM sessions` a `_session_row` reads, and the activity expression, for this store's columns."""
    columns = _columns(connection, "sessions")
    activity = _activity(columns)
    # Fixed literals, never caller text.
    title = "title" if "title" in columns else "NULL"
    display = "display_name" if "display_name" in columns else "NULL"
    return f"SELECT id, source, {title}, {display}, message_count, {activity} FROM sessions", activity


def _session_row(row: Sequence[Any]) -> SessionRow:
    return SessionRow(str(row[0]), _text(row[2]) or _text(row[3]), _text(row[1]), _number(row[5]), int(_number(row[4])))


def _tool_names(tool_calls: Any) -> str:
    try:
        calls = json.loads(tool_calls) if isinstance(tool_calls, str) else None
    except (ValueError, RecursionError):
        return ""
    names = []
    for call in calls if isinstance(calls, list) else ():
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else call.get("name") if isinstance(call, dict) else None
        if isinstance(name, str):
            names.append(_text(name))
    return ", ".join(names)


def _message_text(role: str, content: Any, tool_name: Any, tool_calls: Any) -> str:
    text = _text(content)
    if role == "tool" and _text(tool_name):
        return f"{_text(tool_name)}: {text}" if text else _text(tool_name)
    if not text and tool_calls:
        names = _tool_names(tool_calls)
        return f"(calls {names})" if names else "(tool call)"
    return text


def read_messages(home: Path, session_id: str) -> tuple[tuple[MessageRow, ...], SessionRow | None, str]:
    """The session's last `MESSAGE_ROWS_READ` messages the model still sees, oldest of them first, and its row now."""
    opened = _open_store(home)
    if isinstance(opened, str):
        return (), None, opened
    try:
        select, _activity_expression = _session_select(opened)
        found = opened.execute(f"{select} WHERE id = ?", (session_id,)).fetchone()
        live = " AND active = 1" if "active" in _columns(opened, "messages") else ""
        rows = opened.execute(
            "SELECT role, substr(content, 1, ?), tool_name, substr(tool_calls, 1, ?), timestamp FROM messages "
            f"WHERE session_id = ?{live} ORDER BY id DESC LIMIT ?",
            (MESSAGE_CHARS_READ, TOOL_CALLS_CHARS_READ, session_id, MESSAGE_ROWS_READ),
        ).fetchall()
    except sqlite3.Error as error:
        return (), None, _store_problem(error)
    finally:
        opened.close()
    messages = []
    for row in reversed(rows):
        role = _text(row[0]) or "unknown"
        messages.append(MessageRow(role, _message_text(role, row[1], row[2], row[3]), _number(row[4])))
    return tuple(messages), _session_row(found) if found is not None else None, ""


def read_board(environ: Mapping[str, str], view: BoardView, now: float) -> BoardSnapshot:
    """Every profile's row, and what `view` opens: one profile's sessions, or one session's messages."""
    current, homes = board_homes(environ)
    profiles = tuple(read_profile(name, home, current, now) for name, home in homes)
    home = next((home for name, home in homes if name == view.profile), None)
    if not view.profile:
        return BoardSnapshot(view, profiles, read_at=now)
    if home is None:
        return BoardSnapshot(view, profiles, problem="that profile is gone", read_at=now)
    if view.session:
        messages, session, problem = read_messages(home, view.session)
        listed = (session,) if session is not None else ()
        return BoardSnapshot(view, profiles, listed, messages=messages, problem=problem, read_at=now)
    sessions, unlisted, problem = read_sessions(home)
    return BoardSnapshot(view, profiles, sessions, unlisted, problem=problem, read_at=now)


class BoardReader:
    """Reads the board off the UI thread and keeps the newest snapshot for `take()`.

    One reader serves every opening of the board in a TUI: `start` opens it
    (at the profiles, reading at once), `stop` pauses it when the board
    closes, and `close` ends the thread when the TUI does. While open it reads
    whenever `show` names another view or `refresh` asks, else every
    `BOARD_REFRESH_SECONDS`; a read finished after the view changed, or after
    the board closed, is dropped. Nothing waits for a read under way, so a
    read stuck on a stalled file system holds this one thread and never more:
    the board opened again shows "Reading…" until it returns.
    """

    def __init__(
        self,
        environ: Mapping[str, str],
        *,
        clock: Callable[[], float] = time.time,
        read: Callable[[Mapping[str, str], BoardView, float], BoardSnapshot] = read_board,
    ) -> None:
        self._environ = environ
        self._clock = clock
        self._read = read
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._open = False
        self._view = BoardView()
        self._generation = 0
        self._pending: BoardSnapshot | None = None
        self.on_snapshot: Callable[[], None] | None = None

    def start(self, wake: Callable[[], None] | None = None) -> None:
        """Open the board at the profiles and read at once; `wake` is called, on the thread, after each kept read."""
        self.on_snapshot = wake
        with self._lock:
            self._open = True
            self._view = BoardView()
            self._generation += 1
            self._pending = None
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="omh-bot-board", daemon=True)
            self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        """The board closed: read nothing until it opens again."""
        with self._lock:
            self._open = False
            self._generation += 1
            self._pending = None

    def close(self) -> None:
        """The TUI is leaving: the thread ends after the read under way, if any, without being waited for."""
        self.stop()
        self._closing.set()
        self._wake.set()

    def show(self, view: BoardView) -> None:
        with self._lock:
            if view == self._view:
                return
            self._view = view
            self._generation += 1
        self._wake.set()

    def refresh(self) -> None:
        self._wake.set()

    def take(self) -> BoardSnapshot | None:
        """The newest snapshot not yet taken, or None. Never blocks on a read."""
        with self._lock:
            snapshot, self._pending = self._pending, None
        return snapshot

    def poll_once(self) -> None:
        with self._lock:
            view, generation = self._view, self._generation
        try:
            snapshot = self._read(self._environ, view, self._clock())
        except Exception as error:  # noqa: BLE001 - classified in tests/test_broad_exception_policy.py
            problem = f"the board could not be read: {type(error).__name__}"
            snapshot = BoardSnapshot(view, (), problem=problem, read_at=self._clock())
        with self._lock:
            if generation != self._generation or not self._open:
                return
            self._pending = snapshot
        if self.on_snapshot is not None:
            self.on_snapshot()

    def _run(self) -> None:
        while not self._closing.is_set():
            self._wake.clear()
            with self._lock:
                reading = self._open
            if reading:
                self.poll_once()
                self._wake.wait(BOARD_REFRESH_SECONDS)
            else:
                self._wake.wait()  # closed: no read until `start` or `close` wakes it


# ── drawing ───────────────────────────────────────────────────────────────


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _gateway_words(profile: Profile, served_by: str, theme: Theme) -> tuple[str, str, Style]:
    """The gateway's glyph, what it says, and the style both take."""
    gateway = profile.gateway
    if gateway.problem:
        if served_by:
            return "●", f"served by the {served_by} gateway", theme.ok
        return "○", gateway.problem, theme.warn if "unreadable" in gateway.problem else theme.muted
    if gateway.live:
        words = [f"gateway {gateway.state}"]
        if gateway.pid is not None:
            words.append(f"pid {gateway.pid}")
        if gateway.active_agents:
            words.append(f"{_count(gateway.active_agents, 'agent')} running")
        return "●", " · ".join(words), theme.ok if gateway.state == GATEWAY_RUNNING else theme.warn
    if served_by:
        return "●", f"served by the {served_by} gateway", theme.ok
    if gateway.elsewhere:
        return "◌", f"gateway not running here · its record is another home's ({gateway.elsewhere})", theme.warn
    if gateway.state == GATEWAY_STOPPED:
        reason = f" ({gateway.exit_reason})" if gateway.exit_reason else ""
        return "○", f"gateway stopped{reason}", theme.muted
    pid = f"pid {gateway.pid} is gone" if gateway.pid is not None else "no pid recorded"
    return "◌", f"gateway not running · its record still says {gateway.state} ({pid})", theme.warn


def _platform_line(platform: Platform, theme: Theme) -> Line:
    if not platform.current:
        return (Span(f"{INDENT}{platform.name}: {platform.state} when last recorded", theme.muted),)
    if platform.error:
        return (Span(f"{INDENT}{platform.name}: {platform.state} · {platform.error}", theme.error),)
    if platform.needs_attention:
        return (Span(f"{INDENT}{platform.name}: {platform.state} · needs attention", theme.warn),)
    style = theme.ok if platform.state == "connected" else theme.warn
    return (Span(f"{INDENT}{platform.name}: ", theme.text), Span(platform.state, style))


def _sessions_line(profile: Profile, now: float, theme: Theme) -> Line:
    if profile.store:
        return (Span(f"{INDENT}sessions: {profile.store}", theme.warn if "no state.db" not in profile.store else theme.muted),)
    total = sum(count for _source, count in profile.sources)
    words = [_count(total, "session"), f"{profile.active} active in the last 24 h"]
    if profile.last_activity:
        words.append(f"last active {age_text(profile.last_activity, now)}")
    by_source = ", ".join(f"{source} {count}" for source, count in profile.sources)
    tail = (Span(f" · {by_source}", theme.muted),) if by_source else ()
    return (Span(f"{INDENT}{' · '.join(words)}", theme.text), *tail)


def profile_lines(profiles: Sequence[Profile], selected: int, now: float, cols: int, theme: Theme) -> tuple[list[Line], tuple[int, int]]:
    """Level 1's rows at `cols`, and the first and last row of the highlighted profile."""
    default = next((profile for profile in profiles if profile.name == DEFAULT_PROFILE), None)
    serving = default.name if default is not None and default.gateway.live else ""
    lines: list[Line] = [(Span(f"{BOARD_TITLE} · {_count(len(profiles), 'profile')}", theme.heading),), ()]
    span = (0, 0)
    for index, profile in enumerate(profiles):
        served_by = serving if serving and profile is not default and profile.name in default.gateway.served else ""
        glyph, words, style = _gateway_words(profile, served_by, theme)
        highlighted = index == selected
        mark = CHOICE_MARK if highlighted else " " * len(CHOICE_MARK)
        name = _text(profile.name) + (" (this TUI)" if profile.this_tui else "")
        first = len(lines)
        lines.append((
            Span(mark, theme.prompt), Span(f"{glyph} ", style),
            Span(name, theme.menu_current if highlighted else theme.user), Span(f"  {words}", style),
        ))
        lines.extend(_platform_line(platform, theme) for platform in profile.gateway.platforms)
        lines.append(_sessions_line(profile, now, theme))
        if highlighted:
            span = (first, len(lines) - 1)
        lines.append(())
    if not profiles:
        lines.append((Span("No Hermes home was found.", theme.muted),))
    return [truncate_line(line, cols) for line in lines], span


def session_lines(
    snapshot: BoardSnapshot, selected: int, now: float, cols: int, theme: Theme,
) -> tuple[list[Line], tuple[int, int]]:
    """Level 2's rows at `cols`, and the highlighted session's row (twice)."""
    view = snapshot.view
    unlisted = f"; {_count(snapshot.unlisted, 'subagent session')} not listed" if snapshot.unlisted else ""
    head = f"{_text(view.profile)} · {_count(len(snapshot.sessions), 'session')}, newest activity first{unlisted}"
    lines: list[Line] = [(Span(head, theme.heading),), ()]
    if snapshot.problem:
        lines.append((Span(snapshot.problem, theme.warn),))
    elif not snapshot.sessions:
        lines.append((Span("No sessions.", theme.muted),))
    span = (0, 0)
    for index, row in enumerate(snapshot.sessions):
        highlighted = index == selected
        style = theme.menu_current if highlighted else theme.menu
        mark = CHOICE_MARK if highlighted else " " * len(CHOICE_MARK)
        meta = " · ".join(part for part in (row.source, age_text(row.last_activity, now), _count(row.message_count, "message")) if part)
        meta = f"  {meta}"
        label = row.title or _text(row.id)
        room = cols - cell_width(mark) - cell_width(meta)
        if highlighted:
            span = (len(lines), len(lines))
        if room < 8:
            lines.append((Span(mark, theme.prompt), Span(label, style)))
            continue
        label = truncate(label, room)
        gap = " " * (room - cell_width(label))
        lines.append((Span(mark, theme.prompt), Span(label + gap, style), Span(meta, theme.muted)))
    return [truncate_line(line, cols) for line in lines], span


def _clock_text(at: float) -> str:
    try:
        return datetime.fromtimestamp(at).strftime("%m-%d %H:%M") if at > 0 else "--"
    except (OverflowError, OSError, ValueError):
        return "--"


def message_lines(snapshot: BoardSnapshot, session: SessionRow | None, now: float, cols: int, theme: Theme) -> list[Line]:
    """Level 3's rows at `cols`: who and where, then each message's time, role and its text's first rows."""
    view = snapshot.view
    title = (session.title or session.id) if session is not None else view.session
    lines: list[Line] = [(Span(f"{_text(view.profile)} › {_text(title)}", theme.heading),)]
    if session is not None:
        about = f"{session.source} · {_count(session.message_count, 'message')} · last active {age_text(session.last_activity, now)}"
        lines.append((Span(f"{about} · the last {MESSAGE_ROWS_READ} at most, oldest first", theme.muted),))
    lines.append(())
    if snapshot.problem:
        lines.append((Span(snapshot.problem, theme.warn),))
    elif not snapshot.messages:
        lines.append((Span("No messages.", theme.muted),))
    roles = {"user": theme.user, "assistant": theme.label, "tool": theme.tool}
    for message in snapshot.messages:
        stamp, role = f"{_clock_text(message.at)} ", f"{message.role:<9} "
        lead = cell_width(stamp) + cell_width(role)
        text = message.text[:MESSAGE_SHOWN_CHARS] + ("…" if len(message.text) > MESSAGE_SHOWN_CHARS else "")
        room = max(cols - lead, 8)
        rows = wrap_line((Span(text or "(empty)", theme.text if text else theme.muted),), room)
        kept = 1 if message.role == "tool" else MESSAGE_SHOWN_ROWS  # a tool result is the call's, not the talk's
        if len(rows) > kept:
            # The last row kept ends in an ellipsis, so a cut message never reads as whole.
            rows = [*rows[: kept - 1], truncate_line((*rows[kept - 1], Span(" …")), room)]
        for number, row in enumerate(rows):
            prefix = (Span(stamp, theme.muted), Span(role, roles.get(message.role, theme.muted))) if number == 0 else (Span(" " * lead),)
            lines.append((*prefix, *row))
    return [truncate_line(line, cols) for line in lines]


def _find(keys: Sequence[str], key: str, index: int) -> int:
    """Where `key` is in `keys` now; where it was when it is gone (or none was chosen), kept in range."""
    if key in keys:
        return keys.index(key)
    return min(index, max(len(keys) - 1, 0))


def _show_nothing(view: BoardView) -> None:
    return None


def _refresh_nothing() -> None:
    return None


@dataclass
class BotBoard(Pager):
    """The board in the pager: the levels, the highlight on each, and the newest snapshot the reader delivered."""

    FOOTER_ROWS: ClassVar[int] = 2

    lines: list[Line] = field(default_factory=list)
    snapshot: BoardSnapshot | None = None
    show: Callable[[BoardView], None] = _show_nothing  # tells the reader what to read next
    refresh: Callable[[], None] = _refresh_nothing
    clock: Callable[[], float] = time.time
    view: BoardView = field(default_factory=BoardView)
    profile_index: int = 0
    session_index: int = 0
    # What the highlight is on, by name and id: the lists re-sort by activity at every read, so an index alone
    # would slide to another row under the person's Enter.
    highlighted_profile: str = ""
    highlighted_session: str = ""
    opened: SessionRow | None = None  # the level-2 row level 3 shows, for its header until its own row is read
    _span: tuple[int, int] = field(default=(0, 0), init=False, repr=False, compare=False)
    _follow: bool = field(default=True, init=False, repr=False, compare=False)
    _to_bottom: bool = field(default=False, init=False, repr=False, compare=False)

    def take(self, snapshot: BoardSnapshot) -> None:
        self.snapshot = snapshot

    def _current(self) -> BoardSnapshot | None:
        """The newest snapshot when it was read for the view shown now."""
        snapshot = self.snapshot
        return snapshot if snapshot is not None and snapshot.view == self.view else None

    def _count(self) -> int:
        snapshot = self.snapshot
        if snapshot is None:
            return 0
        if self.view.profile:
            current = self._current()
            return len(current.sessions) if current is not None else 0
        return len(snapshot.profiles)

    def handle_key(self, event: Key | Paste, rows: int) -> bool:
        """One key; False when the person closed the board."""
        if not isinstance(event, Key):
            return True
        char = event.char if event.name == CHAR and not event.ctrl and not event.alt else ""
        if event.name in (ESCAPE, LEFT) or char == "q":
            return self._back()
        if char == "r":
            self.refresh()
        elif event.name in (ENTER, RIGHT):
            self._open()
        elif self.view.session:
            return super().handle_key(event, rows)
        elif event.name == DOWN or char == "j":
            self._move(1)
        elif event.name == UP or char == "k":
            self._move(-1)
        elif event.name == PAGE_DOWN:
            self._move(self.body_rows(rows))
        elif event.name == PAGE_UP:
            self._move(-self.body_rows(rows))
        elif event.name == HOME or char == "g":
            self._move(-self._count())
        elif event.name == END or char == "G":
            self._move(self._count())
        return True

    def _move(self, step: int) -> None:
        last = max(self._count() - 1, 0)
        if self.view.profile:
            current = self._current()
            self.session_index = min(max(self.session_index + step, 0), last)
            if current is not None and current.sessions:
                self.highlighted_session = current.sessions[self.session_index].id
        else:
            profiles = self.snapshot.profiles if self.snapshot is not None else ()
            self.profile_index = min(max(self.profile_index + step, 0), last)
            if profiles:
                self.highlighted_profile = profiles[self.profile_index].name
        self._follow = True

    def _open(self) -> None:
        snapshot = self.snapshot
        if snapshot is None or self.view.session:
            return
        if not self.view.profile:
            if self.profile_index < len(snapshot.profiles):
                self.highlighted_profile = snapshot.profiles[self.profile_index].name
                self._go(BoardView(self.highlighted_profile))
                self.session_index, self.highlighted_session = 0, ""
            return
        current = self._current()
        if current is not None and self.session_index < len(current.sessions):
            self.opened = current.sessions[self.session_index]
            self.highlighted_session = self.opened.id
            self._go(BoardView(self.view.profile, self.opened.id))
            self._to_bottom = True

    def _back(self) -> bool:
        if self.view.session:
            self._go(BoardView(self.view.profile))
        elif self.view.profile:
            self._go(BoardView())
        else:
            return False
        return True

    def _go(self, view: BoardView) -> None:
        self.view = view
        self.top = 0
        self._follow = True
        self.show(view)

    def _page(self, cols: int, rows: int, theme: Theme) -> tuple[list[Line], str]:
        self._draw(cols, theme)
        body = self.body_rows(rows)
        # Level 1 draws from any snapshot (every one carries the profiles); level 2 waits for its own read.
        drawn = self.snapshot is not None if not self.view.profile else self._current() is not None
        if self._to_bottom and self._current() is not None:
            self._to_bottom = False
            self.top = max(len(self.lines) - body, 0)
        elif self._follow and not self.view.session and drawn:
            self._follow = False
            first, last = self._span
            if first < self.top:
                self.top = 0 if first <= 2 else first  # the first item brings the header above it back
            elif last >= self.top + body:
                self.top = min(first, last - body + 1)
        return super()._page(cols, rows, theme)

    def _draw(self, cols: int, theme: Theme) -> None:
        snapshot = self.snapshot
        now = self.clock()
        if snapshot is None:
            self.lines, self._span = [(Span("Reading…", theme.muted),)], (0, 0)
            return
        self.profile_index = _find(
            [profile.name for profile in snapshot.profiles], self.highlighted_profile, self.profile_index,
        )
        if not snapshot.profiles and snapshot.problem:
            self.lines, self._span = [(Span(snapshot.problem, theme.warn),)], (0, 0)
            return
        if not self.view.profile:
            self.lines, self._span = profile_lines(snapshot.profiles, self.profile_index, now, cols, theme)
            return
        current = self._current()
        if current is None:
            self.lines, self._span = [(Span(f"{_text(self.view.profile)} · reading…", theme.muted),)], (0, 0)
            return
        if not self.view.session:
            self.session_index = _find([row.id for row in current.sessions], self.highlighted_session, self.session_index)
            self.lines, self._span = session_lines(current, self.session_index, now, cols, theme)
            return
        session = current.sessions[0] if current.sessions else self.opened  # the row as read now, else as opened
        self.lines, self._span = message_lines(current, session, now, cols, theme), (0, 0)

    def footer(self, position: str, theme: Theme) -> list[Line]:
        """The keys, then the title, where the page is, and how old the reading is."""
        keys = MESSAGE_KEYS if self.view.session else BOARD_KEYS
        snapshot = self.snapshot
        age = f"read {age_text(snapshot.read_at, self.clock())}" if snapshot is not None else "reading…"
        refresh = f"every {BOARD_REFRESH_SECONDS:g} s"
        return [
            (Span(f" {keys} ", theme.status),),
            (Span(f" {self.title} · rows {position} · {age} · {refresh} · read-only ", theme.status),),
        ]

"""Tool registry: everything the agent can do — files, shell, search, web."""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config
from . import filestat  # Worker 8/20: mtime+size change detection
from ._foundation import get_logger, validate_path
from .cancelguard import TurnCancelled, run_cancellable

_log = get_logger("tools")

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

RISK_SAFE = "safe"          # runs without asking
RISK_CONFIRM = "confirm"    # needs user approval (unless auto-approve)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]
    risk: str = RISK_SAFE
    # Scheduler-yield tools (TaskCollect/TaskOutput) may block while
    # already-running background work continues. Their wall time is not
    # tool execution cost: the adaptive turn cap must not treat it as a
    # slow tool and stop the parent turn before the reports arrive.
    background_wait: bool = False

    # Module-level cache: schemas are identical for every call, so build
    # once per tool name instead of re-serializing on every turn.
    _schema_cache: dict = field(default_factory=dict, repr=False,
                                compare=False)

    def openai_schema(self) -> dict:
        cached = self._schema_cache.get(self.name)
        if cached is None:
            cached = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }
            self._schema_cache[self.name] = cached
        return cached


def _clip(text: str, limit: int = config.MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 4:]
    return f"{head}\n… [{len(text) - len(head) - len(tail)} chars truncated] …\n{tail}"


def _resolve(path: str) -> Path:
    """Resolve a user-supplied path against the process cwd.

    Raises ValueError on empty paths, null bytes, or path-traversal
    attempts (``..``) — every file tool funnels through here, so a
    malicious or mistaken ``../../etc/passwd`` can never escape.
    """
    p = validate_path(path)  # raises ValidationError on bad input
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _resolve_under(path: str, base: str | Path) -> Path:
    """Like _resolve but against an explicit base directory (used by
    apply_patch, which honors the live-shell session cwd)."""
    p = validate_path(path)
    if not p.is_absolute():
        p = Path(base) / p
    return p


def _checked(path: str) -> tuple[Path | None, str | None]:
    """_resolve without exceptions: (path, None) on success, or
    (None, "ERROR: ...") — so tools fail fast with a clear message."""
    try:
        return _resolve(path), None
    except Exception as e:  # ValidationError, TypeError, ...
        return None, f"ERROR: invalid path: {e}"


def _checked_under(path: str, base: str | Path,
                   ) -> tuple[Path | None, str | None]:
    try:
        return _resolve_under(path, base), None
    except Exception as e:
        return None, f"ERROR: invalid path: {e}"


def _coerce_int(value: Any, name: str, default: int, lo: int,
                hi: int) -> tuple[int | None, str | None]:
    """Coerce a model-supplied numeric argument, clamped to [lo, hi].
    Returns (value, None) or (None, "ERROR: ...")."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None, f"ERROR: {name} must be an integer, got {value!r}"
    return max(lo, min(hi, v)), None


def _diff_summary(old: str, new: str) -> tuple[int, int]:
    """(additions, removals) line counts between two versions."""
    adds = removes = 0
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="", n=0)
    for i, line in enumerate(lines):
        # Skip the "---"/"+++" file headers positionally (first two
        # lines): a removed "-- x" line renders as "--- x" and must be
        # counted as a removal, not mistaken for the header.
        if i < 2 and (line.startswith("---") or line.startswith("+++")):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+"):
            adds += 1
        elif line.startswith("-"):
            removes += 1
    return adds, removes


def _edit_report(path: Path, old: str, new: str,
                 extra: str = "") -> str:
    """The live-coding style receipt: 'Updated X with N additions and
    M removals' — so the caller sees exactly what changed at a glance."""
    adds, removes = _diff_summary(old, new)
    head = f"Updated {path} with {adds} addition(s) and {removes} removal(s)"
    return f"{head}{extra}"


def _line_numbered(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(start + len(lines) - 1))
    out = []
    for i, line in enumerate(lines):
        n = start + i
        if n == start or n == start + len(lines) - 1 or n % 10 == 0:
            out.append(f"{n:>{width}}→{line}")
        else:
            out.append(f"{'':>{width}} {line}")
    return "\n".join(out)


# Bounds for the self-diagnosing edit-miss error below: large enough to be
# useful to the caller, small enough not to blow up the context window.
_MISS_EXPECTED_MAX_LINES = 20
_MISS_EXCERPT_RADIUS = 5
_MISS_MAX_CHARS = 4000


def _rich_edit_miss_error(p: Path, text: str, old_string: str,
                          label: str = "") -> str:
    """Self-diagnosing 'old_string not found' error.

    A bare "not found" leaves the caller guessing (the incident that
    motivated this: a model retried blindly for 512s). This error hands
    the caller everything needed to fix the edit in ONE retry: the file
    path and line count, the expected text (truncated), a unified diff of
    expected-vs-closest-actual, the actual file excerpt with line numbers,
    and 1-3 concrete copy-ready suggestions.
    """
    file_lines = text.splitlines()
    old_lines = old_string.splitlines()
    total = len(file_lines)

    head = (f"ERROR: {label}old_string not found in {p} "
            f"({total} line(s) total; it must match exactly, "
            "including indentation)")

    parts = [head, "",
             "EXPECTED (first "
             f"{min(len(old_lines), _MISS_EXPECTED_MAX_LINES)} of "
             f"{len(old_lines)} line(s)):"]
    exp_shown = "\n".join(old_lines[:_MISS_EXPECTED_MAX_LINES])
    if len(old_lines) > _MISS_EXPECTED_MAX_LINES:
        exp_shown += (f"\n… [+{len(old_lines) - _MISS_EXPECTED_MAX_LINES} "
                      "more line(s) of the expected text omitted]")
    parts.append(exp_shown or "(empty)")

    # --- cheap near-miss diagnostics ---
    hints: list[str] = []
    stripped = old_string.strip()
    if stripped and stripped in text:
        hints.append(
            "near-miss: old_string appears once leading/trailing "
            "whitespace is ignored — check for extra blank lines or "
            "wrong indentation at the edges of your old_string")
    if "\r\n" in text and stripped and \
            old_string.replace("\n", "\r\n") in text:
        hints.append(
            "near-miss: the file uses CRLF line endings but your old_string "
            "uses LF — match the file's actual line endings")
    if stripped and stripped not in text \
            and stripped.lower() in text.lower():
        hints.append(
            "near-miss: a case-insensitive match exists — check letter "
            "casing in your old_string")

    # --- closest-match search: anchor on the first non-blank line ---
    anchor = next((ln for ln in old_lines if ln.strip()), "")
    top: list[int] = []  # 0-based file line indexes, best first
    if anchor and file_lines:
        scored = sorted(
            ((difflib.SequenceMatcher(None, anchor.strip(),
                                      fl.strip()).ratio(), i)
             for i, fl in enumerate(file_lines)),
            key=lambda t: t[0], reverse=True)
        top = [i for r, i in scored[:3] if r > 0.0]
    if not top and file_lines:
        top = [0]

    if top:
        # Align the anchor's position inside old_string with its best
        # match in the file to get the closest region of equal length.
        anchor_pos = old_lines.index(anchor) if anchor else 0
        start = max(0, top[0] - anchor_pos)  # 0-based
        region = file_lines[start:start + len(old_lines)]

        parts += ["",
                  f"CLOSEST MATCH in file (line {start + 1}"
                  f"..{start + len(region)}), unified diff "
                  "expected -> actual:"]
        diff_lines = list(difflib.unified_diff(
            old_lines[:_MISS_EXPECTED_MAX_LINES],
            region[:_MISS_EXPECTED_MAX_LINES],
            fromfile="expected", tofile="actual", lineterm="", n=3))
        parts.append("\n".join(diff_lines) if diff_lines
                     else "(region identical to expected — unreachable)")

        # Whitespace-only near miss on the closest region.
        norm = lambda lines: [re.sub(r"\s+", " ", ln).strip()
                              for ln in lines]
        if region and norm(old_lines) == norm(region)[:len(old_lines)]:
            hints.append(
                "near-miss: the closest region matches except for "
                "whitespace (tabs vs spaces / trailing spaces) — copy the "
                "exact indentation from the excerpt above")

        ex_lo = max(1, start + 1 - _MISS_EXCERPT_RADIUS)
        ex_hi = min(total, start + len(region) + _MISS_EXCERPT_RADIUS)
        parts += ["",
                  f"FILE EXCERPT around the closest match "
                  f"(lines {ex_lo}..{ex_hi}):"]
        parts.append(_line_numbered("\n".join(file_lines[ex_lo - 1:ex_hi]),
                                    ex_lo))

        parts += ["", "SUGGESTIONS (closest matching lines, copy from "
                      "the excerpt above):"]
        for j, i in enumerate(top[:3], 1):
            line = file_lines[i]
            one = line if len(line) <= 100 else line[:97] + "..."
            parts.append(f"  {j}. line {i + 1}: {one!r}")
    elif total == 0:
        parts += ["", "(the file is empty — nothing to match against)"]

    if hints:
        parts += ["", "DIAGNOSIS:"]
        parts += [f"  - {h}" for h in hints]

    msg = "\n".join(parts)
    if len(msg) > _MISS_MAX_CHARS:
        msg = msg[:_MISS_MAX_CHARS] + "\n… [error truncated]"
    return msg


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def _atomic_write_text(p: Path, text: str) -> None:
    """Write UTF-8 atomically: a crash mid-write can never leave a
    truncated/corrupt file behind. The temp name carries pid+thread so two
    concurrent writes to the same path can't clobber each other's temp."""
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        # os.replace(tmp, p) installs a fresh inode: preserve the target's
        # permission bits instead of resetting 0600/0755 files to 0644.
        mode = stat.S_IMODE(p.stat().st_mode) & 0o777
    except OSError:
        mode = None
    tmp = p.with_name(
        f"{p.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        if mode is not None:
            os.chmod(tmp, mode)
    except BaseException:
        # don't litter a half-written temp file next to the target
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.replace(tmp, p)


def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a text file with line numbers.

    The DEFAULT is a bounded window (first 200 lines). When output is
    truncated, a trailer states exactly how many lines remain and how to
    page further — so the agent never dumps a whole multi-thousand-line
    file into context on every read.
    """
    p, err = _checked(path)
    if err:
        return err
    offset, err = _coerce_int(offset, "offset", 1, 1, 10_000_000)
    if err:
        return err
    limit, err = _coerce_int(limit, "limit", 200, 1, 5000)
    if err:
        return err
    try:
        st = p.stat()  # single stat: proves existence + gives size
    except OSError:
        return f"ERROR: file not found: {p}"
    if stat.S_ISDIR(st.st_mode):
        return f"ERROR: {p} is a directory (use list_dir)"
    if st.st_size > 2_000_000:
        return (f"ERROR: file is very large ({st.st_size} bytes); "
                "use offset/limit")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: {e}"
    # Worker 8/20: record mtime+size so a later edit_file can detect an
    # external change between this read and the edit (the 512s incident).
    filestat.note_read(None, str(p))
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return f"[{p} — 0 lines (empty file)]"
    if offset > total:
        return (f"[{p} — {total} lines total; offset {offset} is past the "
                f"end of file]")
    chunk = lines[offset - 1: offset - 1 + limit]
    end = offset + len(chunk) - 1
    header = f"[{p} — {total} lines total, showing {offset}..{end}]"
    body = _line_numbered(chr(10).join(chunk), offset)
    remaining = total - end
    if remaining > 0:
        body += (f"\n… ({remaining} more lines, total {total} lines). "
                 "Use offset/limit to read more.")
    return f"{header}\n{body}"


def read_region(path: str, start_line: int, end_line: int) -> str:
    """Convenience wrapper: read lines start_line..end_line (1-based,
    inclusive). Delegates to read_file — NOT registered as a separate
    agent tool, so the tool count stays down."""
    start, err = _coerce_int(start_line, "start_line", 1, 1, 10_000_000)
    if err:
        return err
    end, err = _coerce_int(end_line, "end_line", start, 1, 10_000_000)
    if err:
        return err
    return read_file(path, offset=start, limit=max(1, end - start + 1))


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file (parents created automatically)."""
    p, err = _checked(path)
    if err:
        return err
    if not isinstance(content, str):
        return f"ERROR: content must be a string, got {type(content).__name__}"
    # Transmission-corruption guard: null bytes / stray control chars mean
    # the content was mangled in flight — fail fast instead of writing a
    # corrupt file the agent will then have to delete and rewrite.
    if "\x00" in content:
        return ("ERROR: content corrupted in transmission (null byte) — "
                "please resend the write_file call")
    for i, ch in enumerate(content):
        o = ord(ch)
        if o < 0x20 and ch not in "\n\r\t":
            return ("ERROR: content corrupted in transmission "
                    f"(control char U+{o:04X} at offset {i}) — please "
                    "resend the write_file call")
    old = ""
    existed = p.exists()
    if existed:
        try:
            old = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            old = ""
    try:
        _atomic_write_text(p, content)
    except OSError as e:
        return f"ERROR: {e}"
    # Worker 8/20: the agent knows this exact state — a following edit must
    # not mistake our own write for an external change.
    filestat.note_written(None, str(p))
    if not existed or old == "":
        # brand-new (or previously empty) file — a pure-addition report
        adds = len(content.splitlines())
        return f"OK: created {p} with {adds} line(s) ({len(content)} chars)"
    adds, removes = _diff_summary(old, content)
    return f"Updated {p} with {adds} addition(s) and {removes} removal(s)"


def _match_line_numbers(text: str, old_string: str) -> list[int]:
    """1-based line numbers where old_string occurs in text (start of match)."""
    nums: list[int] = []
    start = 0
    while True:
        idx = text.find(old_string, start)
        if idx == -1:
            break
        nums.append(text.count("\n", 0, idx) + 1)
        start = idx + (len(old_string) or 1)
    return nums


def _preview_edit(path: Path, old: str, new: str,
                  detail_lines: list[str]) -> str:
    """Dry-run report for edit_file / MultiEdit: which lines WOULD match,
    plus a unified before/after diff. The file is NOT modified."""
    body = "\n".join(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
    if not body:
        body = "(no changes — old and new content are identical)"
    header = ("PREVIEW (dry-run, file NOT modified):\n"
              + "\n".join(detail_lines) + "\n")
    return header + body


def edit_file(path: str, old_string: str, new_string: str,
              replace_all: bool = False,
              preview_only: bool = False) -> str:
    """Replace an exact string in a file. old_string must match exactly once
    (or set replace_all=true to replace every occurrence). When
    preview_only=true the file is left untouched and a unified
    before/after diff of exactly what WOULD change is returned instead,
    including the matched line numbers."""
    p, err = _checked(path)
    if err:
        return err
    if not p.exists():
        return f"ERROR: file not found: {p}"
    if not old_string:
        # str.count("") counts len+1 positions and str.replace("") would
        # interleave new_string between EVERY character — destroy the file
        return "ERROR: old_string must be a non-empty string"
    # Worker 8/20 (512s incident): if the file changed since the agent last
    # read it, say so plainly instead of failing with a cryptic
    # "old_string not found".
    warn = filestat.check_stale(None, str(p))
    if warn:
        return warn
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ("ERROR: file is not valid UTF-8 text; refusing to edit "
                "(lossy rewrite would corrupt unrelated bytes)")
    except OSError as e:
        return f"ERROR: {e}"
    count = text.count(old_string)
    if count == 0:
        return _rich_edit_miss_error(p, text, old_string)
    if count > 1 and not replace_all:
        return f"ERROR: old_string matches {count} places; add more context " \
               "to make it unique or set replace_all=true"
    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)
    if preview_only:
        matches = _match_line_numbers(text, old_string)
        if not replace_all:
            matches = matches[:1]
        n = count if replace_all else 1
        return _preview_edit(
            p, text, new_text,
            [f"  {n} occurrence(s) WOULD be replaced at line(s) {matches}"])
    try:
        _atomic_write_text(p, new_text)
    except OSError as e:
        return f"ERROR: {e}"
    # Worker 8/20: record post-edit stat so a chained edit doesn't warn
    # about our own change.
    filestat.note_written(None, str(p))
    adds, removes = _diff_summary(text, new_text)
    n = count if replace_all else 1
    return (f"Updated {p} — {n} occurrence(s) replaced, "
            f"{adds} addition(s), {removes} removal(s)")


def list_dir(path: str = ".") -> str:
    """List a directory's contents (one level)."""
    p, err = _checked(path)
    if err:
        return err
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    # One stat per entry (not three): is_dir/is_file/size all derive from it.
    rows: list[tuple[bool, str, str]] = []  # (is_file, sort_name, display)
    try:
        entries = list(p.iterdir())
    except OSError as e:
        return f"ERROR: cannot list {p}: {e}"
    for e in entries:
        try:
            st = e.stat()
        except OSError:
            rows.append((True, e.name.lower(), f"  {e.name}  (unreadable)"))
            continue
        if stat.S_ISDIR(st.st_mode):
            rows.append((False, e.name.lower(), f"  {e.name}/"))
        else:
            rows.append((True, e.name.lower(),
                         f"  {e.name}  ({st.st_size} bytes)"))
    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [f"[{p}]"] + [r[2] for r in rows[:300]]
    if len(rows) > 300:
        lines.append(f"  … and {len(rows) - 300} more")
    return "\n".join(lines) if len(lines) > 1 else f"[{p}] (empty)"


def file_info(path: str) -> str:
    """Show metadata about a file or directory."""
    p, err = _checked(path)
    if err:
        return err
    try:
        st = p.stat()
    except OSError:
        return f"ERROR: not found: {p}"
    kind = "directory" if stat.S_ISDIR(st.st_mode) else "file"
    return (f"{p}\n  type: {kind}\n  size: {st.st_size} bytes\n"
            f"  modified: {st.st_mtime}")


def create_directory(path: str) -> str:
    """Create a directory (parents included). Never raises."""
    p, err = _checked(path)
    if err:
        return err
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"ERROR: cannot create directory {p}: {e}"
    return f"OK: created directory {p}"


def copy_path(src: str, dst: str) -> str:
    """Copy a file or directory. Never raises."""
    s, err = _checked(src)
    if err:
        return err
    d, err = _checked(dst)
    if err:
        return err
    if not s.exists():
        return f"ERROR: source not found: {s}"
    try:
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    except OSError as e:
        return f"ERROR: copy failed: {e}"
    return f"OK: copied {s} -> {d}"


def move_path(src: str, dst: str) -> str:
    """Move/rename a file or directory. Never raises."""
    s, err = _checked(src)
    if err:
        return err
    d, err = _checked(dst)
    if err:
        return err
    if not s.exists():
        return f"ERROR: source not found: {s}"
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    except OSError as e:
        return f"ERROR: move failed: {e}"
    return f"OK: moved {s} -> {d}"


def delete_path(path: str) -> str:
    """Delete a file or directory permanently. Never raises."""
    p, err = _checked(path)
    if err:
        return err
    if not p.exists() and not p.is_symlink():
        return f"ERROR: not found: {p}"
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    except OSError as e:
        return f"ERROR: delete failed: {e}"
    return f"OK: deleted {p}"


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------

# Files bigger than this are skipped by search_files — reading a 500MB log
# into memory to regex it would blow the agent's RAM for zero benefit.
_SEARCH_MAX_FILE_BYTES = 5_000_000


def search_files(pattern: str, path: str = ".", glob_filter: str = "*",
                 max_results: int = 100) -> str:
    """Regex search through file contents (ripgrep-style), respecting common
    ignore dirs. Returns matching lines as path:line:content."""
    if not isinstance(pattern, str) or not pattern:
        return "ERROR: pattern must be a non-empty string"
    root, err = _checked(path)
    if err:
        return err
    max_results, err = _coerce_int(max_results, "max_results", 100, 1, 1000)
    if err:
        return err
    if not isinstance(glob_filter, str) or not glob_filter:
        return "ERROR: glob_filter must be a non-empty string"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    if root.is_file():
        files = [root]
        root = root.parent
    else:
        files = []
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                     "dist", "build", ".tox", ".mypy_cache", ".ruff_cache"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if fnmatch.fnmatch(fn, glob_filter):
                    files.append(Path(dirpath) / fn)
            if len(files) > 5000:
                break
    hits: list[str] = []
    for f in files:
        try:
            if f.stat().st_size > _SEARCH_MAX_FILE_BYTES:
                continue  # huge binary/log — skip, don't OOM
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 2000:
                line = line[:2000]  # pathological one-line files
            if rx.search(line):
                hits.append(f"{f}:{i}:{line.strip()[:200]}")
                if len(hits) >= max_results:
                    return "\n".join(hits) + f"\n… (stopped at {max_results} results)"
    return "\n".join(hits) if hits else "no matches"


def glob_files(pattern: str, path: str = ".") -> str:
    """Find files by glob pattern (e.g. '**/*.py')."""
    if not isinstance(pattern, str) or not pattern:
        return "ERROR: pattern must be a non-empty string"
    if pattern.startswith("/"):
        # Path.glob() rejects absolute patterns outright
        return "ERROR: pattern must be relative to the search path"
    if ".." in Path(pattern).parts:
        # Path.glob("..") escapes the search root — reject like _resolve
        return "ERROR: pattern must not contain '..'"
    root, err = _checked(path)
    if err:
        return err
    try:
        matches = sorted(str(p) for p in root.glob(pattern)
                         if p.is_file())[:300]
    except (NotImplementedError, ValueError) as e:
        return f"ERROR: bad glob pattern: {e}"
    return "\n".join(matches) if matches else "no matches"


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Kill proc AND its children.

    Shell commands are run as ``bash -c <cmd>``: the Popen handle is the
    bash wrapper, but the real work usually happens in forked children
    (``sleep 60``, compilers, servers, test runners...). Killing only the
    wrapper orphans those children — they keep running (and holding
    ports/files) after Esc/timeout. New sessions are started in their own
    process group (see _popen_kwargs), so killpg takes the whole tree.
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # Windows: no process groups — best effort on the wrapper
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait()
    except Exception:
        pass


# start_new_session puts the child in its own process group so
# _kill_process_tree can kill the whole tree on cancel/timeout.
# (POSIX-only; on Windows the kwarg is unsupported so we omit it.)
_popen_kwargs: dict = {"start_new_session": True} if os.name == "posix" \
    else {}


def _pump_process(proc: "subprocess.Popen", timeout: float,
                  on_output: "Callable[[str, str], None] | None" = None,
                  should_cancel: "Callable[[], bool] | None" = None,
                  ) -> tuple[list, list] | None:
    """Read stdout/stderr of `proc` line-by-line until it exits or the
    timeout elapses. Returns (stdout_lines, stderr_lines), or None on
    timeout (the process is killed). Every line is relayed to
    on_output(line, "out"|"err") the moment it is produced — this is what
    lets the TUI stream shell output live, like watching a real terminal.
    If `should_cancel` is set and returns True, the process is killed
    immediately and partial output is returned (Esc/Ctrl+C support)."""
    q: "queue.Queue" = queue.Queue()

    def pump(stream, tag: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                q.put((tag, line))
        finally:
            q.put((tag, None))  # sentinel: this stream is done

    threading.Thread(target=pump, args=(proc.stdout, "out"),
                     daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, "err"),
                     daemon=True).start()

    out_lines: list = []
    err_lines: list = []
    open_streams = 2
    deadline = time.monotonic() + timeout
    cancelled = False

    def _cancel_requested() -> bool:
        # A raising should_cancel must never take down the tool — treat
        # it as "not cancelled" and keep pumping.
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:
            _log.warning("should_cancel callback raised; ignoring")
            return False

    while open_streams > 0:
        # Esc/Ctrl+C: kill the process immediately
        if _cancel_requested():
            cancelled = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_tree(proc)
            return None
        try:
            tag, line = q.get(timeout=min(0.25, max(0.01, remaining)))
        except queue.Empty:
            if proc.poll() is not None:
                # process exited; drain whatever is left briefly
                continue
            continue
        if line is None:
            open_streams -= 1
            continue
        (out_lines if tag == "out" else err_lines).append(line)
        if on_output is not None:
            try:
                on_output(line.rstrip("\n"), tag)
            except Exception:  # noqa: BLE001 — never break the tool
                pass
    if cancelled:
        _kill_process_tree(proc)
        # Mark as cancelled so the caller can report it properly
        out_lines.append("\n[CANCELLED by user (Esc/Ctrl+C)]\n")
        return out_lines, err_lines
    proc.wait()
    return out_lines, err_lines


def _validate_shell_args(command: str, timeout: int,
                         ) -> tuple[str | None, float | None, str | None]:
    """Shared validation for run_command/live_shell. Returns
    (command, timeout_seconds, None) or (None, None, "ERROR: ...")."""
    if not isinstance(command, str) or not command.strip():
        return None, None, "ERROR: command must be a non-empty string"
    try:
        secs = float(timeout)
    except (TypeError, ValueError):
        return None, None, \
            f"ERROR: timeout must be a number, got {timeout!r}"
    if secs <= 0:
        return None, None, "ERROR: timeout must be positive"
    return command, min(secs, 600.0), None  # hard cap: 10 minutes


def run_command(command: str, timeout: int = 120,
                on_output: "Callable[[str, str], None] | None" = None,
                should_cancel: "Callable[[], bool] | None" = None,
                run_in_background: bool = False,
                description: str = "") -> str:
    """Run a shell command via bash and return exit code + output.

    The shell is resolved once (judge.resolve_shell): on Windows,
    System32\\bash.exe is the WSL stub and fails when no distro is
    installed, so Git Bash is probed and preferred. If `on_output` is
    provided, each output line is streamed to it live as it appears.
    If `should_cancel` returns True, the process is killed (Esc/Ctrl+C).
    If `run_in_background` is True, the command is launched via the
    bgsh engine and this returns immediately with a task id — poll
    with the BashOutput tool."""
    if run_in_background:
        from . import bgsh
        task_id = bgsh.launch(command, description or command[:60],
                              timeout=timeout)
        return (f"background task started: {task_id}\n"
                "poll with BashOutput using this task id.")
    command, secs, err = _validate_shell_args(command, timeout)
    if err:
        return err
    from .judge import resolve_shell
    argv = resolve_shell()
    if argv is None:
        return ("ERROR: no POSIX shell available — install Git Bash "
                "(windows) or bash (posix)")
    try:
        proc = subprocess.Popen(
            argv + [command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            cwd=os.getcwd(),
            **_popen_kwargs,
        )
    except OSError as e:
        return f"ERROR: {e}"
    pumped = _pump_process(proc, secs, on_output, should_cancel)
    if pumped is None:
        return f"ERROR: command timed out after {secs:g}s"
    out_lines, err_lines = pumped
    stdout = "".join(out_lines)
    stderr = "".join(err_lines)
    out = []
    out.append(f"exit code: {proc.returncode}")
    if stdout:
        out.append("--- stdout ---\n" + stdout)
    if stderr:
        out.append("--- stderr ---\n" + stderr)
    return _clip("\n".join(out))


# ---------------------------------------------------------------------------
# Live shell — a PERSISTENT bash session (cd/env/exports survive between
# calls), plus a unified-diff applier for live code editing.
# ---------------------------------------------------------------------------

_SHELL_LOCK = threading.Lock()      # one command at a time on the session
_SHELL_STATE = {"cwd": None,        # sticky working directory
                "env": None}        # sticky environment (persistent exports)


def live_shell(command: str, timeout: int = 120,
               on_output: "Callable[[str, str], None] | None" = None,
               should_cancel: "Callable[[], bool] | None" = None) -> str:
    """Run a command inside a persistent bash session.

    Unlike `run_command` (which spawns a fresh shell per call), this one
    keeps state across calls: a `cd src` in one call is still in effect in
    the next, and exported variables persist. Use it for live workflows:
    cd -> build -> test -> inspect -> fix.

    State is shared process-wide and guarded by a lock so two callers can
    never interleave their commands into the same session. If `on_output`
    is provided, each output line is streamed to it live as it appears."""
    command, secs, err = _validate_shell_args(command, timeout)
    if err:
        return err
    from .judge import resolve_shell
    argv = resolve_shell()
    if argv is None:
        return ("ERROR: no POSIX shell available — install Git Bash "
                "(windows) or bash (posix)")
    # The lock covers the WHOLE command, not just the state update: cwd/env
    # are session-global, so a second command starting mid-flight would
    # read a cwd/env that the first command is about to change — the
    # interleaving this lock exists to prevent.
    with _SHELL_LOCK:
        return _live_shell_locked(argv, command, secs, on_output,
                                  should_cancel)


def _live_shell_locked(argv: list[str], command: str, timeout: float,
                       on_output: "Callable[[str, str], None] | None",
                       should_cancel: "Callable[[], bool] | None" = None
                       ) -> str:
    """live_shell body; the caller must hold _SHELL_LOCK."""
    cwd = _SHELL_STATE["cwd"] or os.getcwd()
    # Wrap so the command's own exit code survives, and the FINAL cwd +
    # environment are reported back on marker lines we strip before
    # returning. Replaying the env on the next call is what makes
    # `export FOO=...` stick across calls despite fresh processes.
    # The markers carry a per-call token: a command whose own output
    # contains the literal text "__FA_CWD__"/"__FA_ENV__" must not be
    # mistaken for our bookkeeping lines.
    _tok = f"{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}"
    _cwd_mark = f"__FA_CWD_{_tok}__"
    _env_mark = f"__FA_ENV_{_tok}__"
    wrapped = (
        command + "\n"
        "__fa_rc=$?\n"
        f'printf "\\n{_cwd_mark}%s" "$PWD"\n'
        f'printf "\\n{_env_mark}"\n'
        "env -0\n"
        "exit $__fa_rc\n")
    extra_env = _SHELL_STATE["env"]

    # Live-stream filter: the __FA_CWD__ marker and everything after it
    # (the env blob) are bookkeeping, not real output — never show them.
    # The wrapper's leading "\n" can arrive as one last blank line, so
    # trailing blanks are held back until the next real line proves they
    # are genuine output.
    cutoff = {"hit": False}
    pending_blanks = {"n": 0}

    def relay(line: str, stream: str) -> None:
        if on_output is None:
            return
        if stream == "out":
            if _cwd_mark in line:
                pre, _, _ = line.partition(_cwd_mark)
                cutoff["hit"] = True
                pending_blanks["n"] = 0
                if not pre:
                    return
                line = pre  # stream the real output before the marker
            if cutoff["hit"]:
                return
            if line == "":
                pending_blanks["n"] += 1
                return
            while pending_blanks["n"] > 0:
                pending_blanks["n"] -= 1
                try:
                    on_output("", "out")
                except Exception:  # noqa: BLE001
                    pass
        try:
            on_output(line, stream)
        except Exception:  # noqa: BLE001
            pass

    try:
        proc = subprocess.Popen(
            argv + [wrapped],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            cwd=cwd, env=extra_env,
            **_popen_kwargs,
        )
    except OSError as e:
        return f"ERROR: {e}"
    pumped = _pump_process(proc, timeout, relay, should_cancel)
    if pumped is None:
        return f"ERROR: command timed out after {timeout:g}s (cwd={cwd})"
    out_lines, err_lines = pumped
    stdout = "".join(out_lines)
    stderr = "".join(err_lines)
    new_cwd, new_env = cwd, extra_env
    idx = stdout.rfind(_env_mark)
    if idx >= 0:
        blob = stdout[idx + len(_env_mark):]
        env_map: dict[str, str] = {}
        for pair in blob.split("\0"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env_map[k] = v
        if env_map:
            new_env = env_map
        stdout = stdout[:idx].rstrip("\n")
    j = stdout.rfind(_cwd_mark)
    if j >= 0:
        new_cwd = stdout[j + len(_cwd_mark):].strip() or cwd
        stdout = stdout[:j].rstrip("\n")
    # Already under _SHELL_LOCK (see live_shell) — no extra locking needed.
    if os.path.isdir(new_cwd):
        _SHELL_STATE["cwd"] = new_cwd
    if new_env:
        _SHELL_STATE["env"] = new_env
    parts = [f"cwd: {_SHELL_STATE['cwd']}", f"exit code: {proc.returncode}"]
    if stdout:
        parts.append("--- stdout ---\n" + stdout)
    if stderr:
        parts.append("--- stderr ---\n" + stderr)
    return _clip("\n".join(parts))


def live_shell_reset() -> str:
    """Reset the persistent shell session back to the process cwd/env."""
    # Take the session lock: without it a reset racing an in-flight
    # command would be silently overwritten when that command writes its
    # (stale) cwd/env back into _SHELL_STATE.
    with _SHELL_LOCK:
        prev = _SHELL_STATE["cwd"]
        _SHELL_STATE["cwd"] = None
        _SHELL_STATE["env"] = None
    return f"OK: session reset ({prev} -> {os.getcwd()})"


def apply_patch(patch: str) -> str:
    """Apply a unified diff to the working tree (live multi-file edit).

    Accepts standard `diff -u` / `git diff` output. File paths are read
    from ---/+++ headers (a/ b/ prefixes stripped). Each hunk is applied
    with its own context tolerance; a hunk that no longer matches its
    context lines fails loudly instead of silently corrupting the file.

    Returns a per-file report: 'Updated X with N addition(s) and M
    removal(s)', matching the style of write_file/edit_file.
    Relative paths resolve against the persistent live_shell cwd if a
    session is active (so `live_shell("cd src")` followed by a patch on
    `a/main.py` does the intuitive thing)."""
    if not isinstance(patch, str) or not patch.strip():
        return "ERROR: patch must be a non-empty string"
    # Read session cwd under the lock — an in-flight live_shell command
    # may be updating it concurrently.
    with _SHELL_LOCK:
        base = _SHELL_STATE["cwd"] or os.getcwd()
    # -- parse the patch into per-file sections ---------------------------
    # A section is: --- <old> / +++ <new> / @@ hunks. Either side may be
    # /dev/null (new file / deleted file). Hunks always belong to the
    # section whose headers precede them — the old code kept attaching a
    # deletion's hunks to the PREVIOUS file's entry, silently corrupting it.
    sections: list[dict] = []
    cur: dict | None = None

    def strip_name(raw: str) -> str:
        name = raw.split("\t")[0].strip()
        if name.startswith("a/") or name.startswith("b/"):
            name = name[2:]
        return name

    for line in patch.splitlines():
        # Inside a hunk body? The @@ counts say exactly how many old/new
        # lines the body holds — consume them here, BEFORE the header
        # checks below. A removed "-- comment" renders as "--- comment"
        # in the diff; without this it would be mistaken for a new file's
        # "---" header and derail the whole parse (same for added "++ x"
        # vs "+++"). Valid diffs never have a bare header-looking line
        # inside a hunk body (body lines always carry their prefix), so
        # this is strictly more correct.
        if cur is not None and cur["hunks"]:
            h = cur["hunks"][-1]
            if h["old_need"] is not None and (
                    h["old_got"] < h["old_need"]
                    or h["new_got"] < h["new_need"]):
                if line.startswith(("+", "-", " ")) or line == "":
                    tag = line[0] if line else " "
                    h["lines"].append(line if line else " ")
                    if tag in (" ", "-"):
                        h["old_got"] += 1
                    if tag in (" ", "+"):
                        h["new_got"] += 1
                # "\ No newline at end of file" and friends: ignored —
                # they are not body lines, so they consume nothing
                continue
            # body complete (or malformed header with old_need=None,
            # which keeps the legacy header-first behavior) — fall
            # through to the header checks
        if line.startswith("--- ") and not line.startswith("--- \t"):
            cur = {"old_name": strip_name(line[4:]), "new_name": None,
                   "hunks": []}
            sections.append(cur)
            continue
        if line.startswith("+++ ") and not line.startswith("+++ \t"):
            if cur is None:
                # +++ without --- — start a section anyway, be lenient
                cur = {"old_name": "/dev/null", "new_name": None,
                       "hunks": []}
                sections.append(cur)
            cur["new_name"] = strip_name(line[4:])
            continue
        if line.startswith("@@"):
            if cur is not None:
                m = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if m:
                    old_need: int | None = (int(m.group(2))
                                            if m.group(2) else 1)
                    new_need: int | None = (int(m.group(4))
                                            if m.group(4) else 1)
                else:
                    old_need = new_need = None  # malformed: legacy
                cur["hunks"].append(
                    {"lines": [],
                     "old_start": int(m.group(1)) if m else 1,
                     "new_start": int(m.group(3)) if m else 1,
                     "old_need": old_need, "new_need": new_need,
                     "old_got": 0, "new_got": 0})
            continue
        if cur is not None and cur["hunks"]:
            # legacy path: malformed @@ header (no counts) — consume
            # body-looking lines until the next header, as before
            h = cur["hunks"][-1]
            if line.startswith(("+", "-", " ")) or line == "":
                h["lines"].append(line if line else " ")
            # "\ No newline at end of file" and friends: ignored
    sections = [s for s in sections if s["hunks"]]
    if not sections:
        return ("ERROR: no hunks found — expected '@@' headers "
                "(unified diff format)")
    # -- resolve + validate every target path BEFORE touching anything -----
    targets: list[tuple[dict, Path, bool]] = []  # (section, path, deleted)
    for s in sections:
        old_null = s["old_name"] == "/dev/null"
        new_null = s["new_name"] == "/dev/null"
        name = s["new_name"] if not new_null else s["old_name"]
        if not name or name == "/dev/null":
            return "ERROR: patch section has no usable file path"
        p, err = _checked_under(name, base)
        if err:
            return err  # path traversal / absolute escape blocked here
        targets.append((s, p, new_null and not old_null))
    # -- apply ---------------------------------------------------------------
    reports: list[str] = []
    for s, p, deleted in targets:
        if deleted:
            if p.exists():
                try:
                    p.unlink()
                except OSError as e:
                    return f"ERROR: cannot delete {p}: {e}"
                reports.append(f"Deleted {p}")
            else:
                reports.append(f"Skipped delete of missing {p}")
            continue
        try:
            old_text = p.read_text(encoding="utf-8",
                                   errors="replace") if p.exists() else ""
        except OSError as e:
            return f"ERROR: cannot read {p}: {e}"
        old_lines = old_text.splitlines()
        new_lines = list(old_lines)
        # apply hunks bottom-up so earlier offsets stay valid
        for hunk in sorted(s["hunks"],
                           key=lambda h: h["old_start"], reverse=True):
            ctx: list[tuple[str, str]] = []   # (tag, text)
            for raw in hunk["lines"]:
                tag, txt = (raw[0], raw[1:]) if len(raw) > 1 else (" ", "")
                ctx.append((tag, txt))
            # find where the hunk's old-side lines start in the file
            old_side = [(t, x) for t, x in ctx if t in (" ", "-")]
            pos = hunk["old_start"] - 1
            matched = False
            for delta in range(0, max(len(old_lines), 1) + 1):
                for off in (delta, -delta):
                    cand = pos + off
                    if cand < 0 or cand + len(old_side) > len(old_lines):
                        continue
                    if all(cand + i < len(old_lines)
                           and old_lines[cand + i] == old_side[i][1]
                           for i in range(len(old_side))):
                        pos = cand
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return (f"ERROR: hunk context mismatch in {p} near "
                        f"line {hunk['old_start']} — file changed since "
                        "the diff was made; regenerate the diff and retry")
            # rebuild: keep everything before, splice the new-side lines,
            # keep everything after
            new_side = [x for t, x in ctx if t in (" ", "+")]
            new_lines = (new_lines[:pos] + new_side
                         + new_lines[pos + len(old_side):])
        new_text = "\n".join(new_lines) + ("\n" if old_text.endswith(
            "\n") or new_lines else "")
        try:
            _atomic_write_text(p, new_text)
        except OSError as e:
            return f"ERROR: writing {p}: {e}"
        reports.append(_edit_report(p, old_text, new_text))
    return "\n".join(reports)


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

def web_fetch(url: str,
              should_cancel: "Callable[[], bool] | None" = None) -> str:
    """Fetch a URL and return its text content.

    CANCEL: the old version blocked up to 30s in requests.get() with no
    way to interrupt it. The fetch now runs guarded — Esc raises
    TurnCancelled within ~0.25s."""
    import urllib.parse
    if not isinstance(url, str) or not url.strip():
        return "ERROR: url must be a non-empty string"
    try:
        scheme = urllib.parse.urlparse(url.strip()).scheme.lower()
    except ValueError:
        return f"ERROR: malformed URL: {url[:80]}"
    if scheme not in ("http", "https"):
        # SSRF guard: no file://, gopher://, or cloud-metadata endpoints
        # reached through exotic schemes
        return f"ERROR: only http(s) URLs are allowed, got {scheme or 'no'} scheme"
    try:
        # Pooled keep-alive session: repeated fetches to the same host
        # skip the TCP+TLS handshake instead of paying it every call.
        from .client import shared_session
        # Same SSRF policy as webfetch.py: loopback, private, link-local
        # (169.254.169.254) and cloud-metadata hosts are refused. Requests
        # follows redirects itself, so walk them manually and validate
        # every hop BEFORE its body can be read.
        from .webfetch import _validate_url
        from urllib.parse import urljoin

        def _get():
            current = url.strip()
            for _ in range(6):
                blocked = _validate_url(current)
                if blocked:
                    raise ValueError(f"blocked URL: {blocked}")
                r = shared_session().get(current, timeout=30, allow_redirects=False,
                                         headers={
                    "User-Agent":
                        "Mozilla/5.0 (X11; Linux x86_64) FullAgent/1.0"})
                if r.status_code in (301, 302, 303, 307, 308):
                    location = r.headers.get("location")
                    r.close()
                    if not location:
                        raise ValueError(
                            f"HTTP {r.status_code} redirect without Location")
                    current = urljoin(current, location)
                    continue
                r.raise_for_status()
                return r
            raise ValueError("too many redirects")
        resp = run_cancellable(_get, should_cancel, name="web fetch")
    except TurnCancelled:
        # Never swallow a cancel into an ERROR string — _execute_tool
        # re-raises it so the turn actually stops.
        raise
    except Exception as e:
        return f"ERROR: {e}"
    ctype = resp.headers.get("content-type", "")
    text = resp.text
    if "html" in ctype:
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>",
                      "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
    return _clip(text.strip(), 16_000)


def _ddg_search(query: str) -> list[tuple[str, str, str]]:
    """DuckDuckGo HTML search -> [(title, url, snippet)]."""
    from .client import shared_session  # pooled keep-alive: no fresh handshake
    resp = shared_session().post(
        "https://html.duckduckgo.com/html/",
        data={"q": query}, timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) FullAgent/1.0"})
    resp.raise_for_status()
    results = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>',
        resp.text)
    snippets = re.findall(
        r'class="result__snippet"[^>]*>([\s\S]*?)</a>', resp.text)
    out = []
    for i, (href, title) in enumerate(results):
        title = re.sub(r"<[^>]+>", "", title).strip()
        snip = re.sub(r"<[^>]+>", "", snippets[i]).strip() \
            if i < len(snippets) else ""
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            import urllib.parse
            href = urllib.parse.unquote(m.group(1))
        out.append((title, href, snip))
    return out


def _bing_search(query: str) -> list[tuple[str, str, str]]:
    """Bing HTML search fallback -> [(title, url, snippet)]."""
    from .client import shared_session  # pooled keep-alive: no fresh handshake
    resp = shared_session().get(
        "https://www.bing.com/search",
        params={"q": query, "count": "10"}, timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                               "Gecko/20100101 Firefox/128.0"})
    resp.raise_for_status()
    out = []
    for block in re.findall(r'<li class="b_algo"[\s\S]*?</li>', resp.text):
        m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>', block)
        if not m:
            continue
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        sm = re.search(r'<p[^>]*>([\s\S]*?)</p>', block)
        snip = re.sub(r"<[^>]+>", "", sm.group(1)).strip() if sm else ""
        out.append((title, url, snip))
    return out


def web_search(query: str,
               should_cancel: "Callable[[], bool] | None" = None) -> str:
    """Real-time web search. Tries DuckDuckGo, then Bing; returns the top
    results with titles, URLs and snippets, stamped with the retrieval
    time so the data's freshness is explicit.

    CANCEL: each engine used to block up to 30s with no interrupt — the
    whole loop is cancel-guarded now; Esc raises TurnCancelled."""
    from datetime import datetime
    errors = []
    results: list[tuple[str, str, str]] = []

    def _engines():
        out: list[tuple[str, str, str]] = []
        errs: list[str] = []
        for engine, fn in (("DuckDuckGo", _ddg_search),
                           ("Bing", _bing_search)):
            try:
                out = fn(query)
                if out:
                    break
            except Exception as e:
                errs.append(f"{engine}: {e}")
        return out, errs

    try:
        results, errors = run_cancellable(_engines, should_cancel,
                                          name="web search")
    except TurnCancelled:
        raise
    if not results:
        return ("ERROR: all search engines failed — "
                + ("; ".join(errors) if errors else "no results"))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"web search: {query!r}  (retrieved {stamp}, live results)"]
    for i, (title, url, snip) in enumerate(results[:8], 1):
        lines.append(f"{i}. {title}\n   {url}\n   {snip[:220]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STR = {"type": "string"}


_REGISTRY: dict[str, Tool] | None = None


def build_registry() -> dict[str, Tool]:
    """Build (once) and return the tool registry. The registry is a
    module-level singleton: rebuilding 17 Tool dataclasses + schema dicts
    on every agent turn is pure waste."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    tools: list[Tool] = [
        Tool("read_file",
             "Read a text file with line numbers. Defaults to the first 200 lines; "
             "a trailer shows remaining lines and how to page with "
             "offset/limit.",
             {"type": "object", "properties": {
                 "path": _STR,
                 "offset": {"type": "integer", "description": "first line (1-based)"},
                 "limit": {"type": "integer", "description": "max lines (default 200)"}},
              "required": ["path"]},
             read_file),
        Tool("write_file",
             "Create or overwrite a file with the given content. Parent dirs are created.",
             {"type": "object", "properties": {
                 "path": _STR, "content": _STR},
              "required": ["path", "content"]},
             write_file, risk=RISK_CONFIRM),
        Tool("edit_file",
             "Replace an exact string in a file. old_string must match exactly "
             "(including indentation) and uniquely, unless replace_all=true. "
             "Set preview_only=true for a dry-run: returns a unified diff of "
             "what WOULD change (with matched line numbers) without "
             "modifying the file — use it to verify the match before "
             "committing.",
             {"type": "object", "properties": {
                 "path": _STR, "old_string": _STR, "new_string": _STR,
                 "replace_all": {"type": "boolean"},
                 "preview_only": {"type": "boolean",
                                  "description": "Dry-run: show the diff of "
                                  "what would change without modifying the "
                                  "file."}},
              "required": ["path", "old_string", "new_string"]},
             edit_file, risk=RISK_CONFIRM),
        Tool("list_dir", "List a directory's contents (one level).",
             {"type": "object", "properties": {"path": _STR}},
             list_dir),
        Tool("file_info", "Show metadata (size, mtime, type) for a path.",
             {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
             file_info),
        Tool("create_directory", "Create a directory (parents included).",
             {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
             create_directory),
        Tool("copy_path", "Copy a file or directory.",
             {"type": "object", "properties": {"src": _STR, "dst": _STR},
              "required": ["src", "dst"]},
             copy_path, risk=RISK_CONFIRM),
        Tool("move_path", "Move/rename a file or directory.",
             {"type": "object", "properties": {"src": _STR, "dst": _STR},
              "required": ["src", "dst"]},
             move_path, risk=RISK_CONFIRM),
        Tool("delete_path", "Delete a file or directory permanently.",
             {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
             delete_path, risk=RISK_CONFIRM),
        Tool("search_files",
             "Regex search through file contents (ripgrep-style). "
             "Returns path:line:content for matches.",
             {"type": "object", "properties": {
                 "pattern": {"type": "string", "description": "regex pattern"},
                 "path": {"type": "string", "description": "dir or file to search"},
                 "glob_filter": {"type": "string", "description": "filename glob, e.g. '*.py'"}},
              "required": ["pattern"]},
             search_files),
        Tool("glob_files", "Find files by glob pattern, e.g. '**/*.py'.",
             {"type": "object", "properties": {
                 "pattern": _STR, "path": _STR}, "required": ["pattern"]},
             glob_files),
        Tool("run_command",
             "Run a shell command via bash and return exit code, stdout, stderr. "
             "Use for builds, tests, git, installs, running programs.",
             {"type": "object", "properties": {
                 "command": _STR,
                 "timeout": {"type": "integer", "description": "seconds, default 120"}},
              "required": ["command"]},
             run_command, risk=RISK_CONFIRM),
        Tool("live_shell",
             "Run a command in a PERSISTENT bash session: cd, exports and "
             "background jobs survive between calls. Use for live workflows "
             "(cd src && build, then run tests, then inspect, then fix).",
             {"type": "object", "properties": {
                 "command": _STR,
                 "timeout": {"type": "integer",
                             "description": "seconds, default 120"}},
              "required": ["command"]},
             live_shell, risk=RISK_CONFIRM),
        Tool("live_shell_reset",
             "Reset the persistent live-shell session back to the process "
             "working directory.",
             {"type": "object", "properties": {}},
             live_shell_reset),
        Tool("apply_patch",
             "Apply a unified diff (git diff / diff -u format) to the working "
             "tree. Multi-file edits in one call; each hunk is context-checked "
             "and fails loudly on mismatch instead of corrupting files. "
             "Returns a per-file 'N additions / M removals' report.",
             {"type": "object", "properties": {"patch": _STR},
              "required": ["patch"]},
             apply_patch, risk=RISK_CONFIRM),
        Tool("web_fetch", "Fetch a URL and return its text content.",
             {"type": "object", "properties": {"url": _STR}, "required": ["url"]},
             web_fetch),
        Tool("web_search", "Search the web (DuckDuckGo) and return top results.",
             {"type": "object", "properties": {"query": _STR}, "required": ["query"]},
             web_search),
    ]
    _REGISTRY = {t.name: t for t in tools}
    return _REGISTRY


def parse_tool_arguments(raw: Any) -> dict:
    """Tool-call arguments arrive as a JSON string (native) or dict."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (ValueError, TypeError):
        return {"_raw": str(raw)}


# ---------------------------------------------------------------------------
# Self-test (offline: no API key needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    reg = build_registry()
    assert "live_shell" in reg and "live_shell_reset" in reg \
        and "apply_patch" in reg, "new tools missing from registry"

    live_shell_reset()
    # persistent cwd across calls
    live_shell("cd /tmp")
    r = live_shell("pwd")
    assert "cwd: /tmp" in r, f"cwd did not persist:\n{r}"
    # persistent exports across calls (env replay)
    r = live_shell("export FA_TEST_VAR=ok7")
    assert "exit code: 0" in r, r
    r = live_shell('echo "$FA_TEST_VAR"')
    assert "ok7" in r, f"export did not persist:\n{r}"
    # multi-line compound command
    r = live_shell("for i in 1 2; do echo n=$i; done")
    assert "n=1" in r and "n=2" in r, r
    # failure exit code propagates; failed cd leaves state intact
    live_shell("cd /tmp")
    r = live_shell("cd /definitely_missing_dir_xyz; true")
    assert "cwd: /tmp" in r.splitlines()[0], r
    live_shell_reset()

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "st.py"
        msg = write_file(str(fp), "a\nb\nc\n")
        assert msg.startswith("OK: created"), msg
        patch = (
            f"--- a/{td}/st.py\n+++ b/{td}/st.py\n"
            "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n")
        out = apply_patch(patch)
        assert out.startswith("Updated") and "1 addition(s)" in out, out
        assert fp.read_text() == "a\nB\nc\n", fp.read_text()
        bad = patch.replace(" c\n", " WRONG CONTEXT\n")
        err = apply_patch(bad)
        assert err.startswith("ERROR: hunk context mismatch"), err
        assert fp.read_text() == "a\nB\nc\n", "failed patch mutated the file!"

        # edit_file miss is self-diagnosing: path, line count, expected
        # text, numbered excerpt, and a suggestion — one retry should
        # suffice instead of a blind retry loop.
        fp2 = Path(td) / "miss.py"
        fp2.write_text(
            "def foo():\n    x = 1\n    return x\n\n"
            "def bar():\n    y = 2\n    return y\n", encoding="utf-8")
        err = edit_file(str(fp2), "def foo():\n   x = 1\n", "zzz")
        assert err.startswith("ERROR:"), err
        assert str(fp2) in err, "error must name the file path"
        assert "7 line(s) total" in err, "error must give the line count"
        assert "EXPECTED" in err, "error must show the expected text"
        assert "→" in err and "line 1" in err, \
            "error must show a line-numbered excerpt"
        assert "SUGGESTIONS" in err, "error must include suggestions"
        assert "whitespace" in err.lower(), \
            "error must diagnose the indentation near-miss"
        assert fp2.read_text().startswith("def foo():"), \
            "failed edit must not mutate the file"

    print("TOOLS SELF-TEST PASS")

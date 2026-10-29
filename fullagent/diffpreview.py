"""Real-Time Streaming Live Diff Preview & Unified Diff Confirmation Gate.

Provides two complementary diff capabilities for FullAgent:

1. **Real-Time Streaming Live Diff Engine (:class:`LiveDiffStreamer`)**:
   Incrementally parses streaming tool-call JSON arguments (`on_tool_args`) for
   `write_file`, `edit_file`, `MultiEdit`/`multiedit`, and `apply_patch` while
   the model is still generating tokens. Emits structured diff events (header,
   removed `-` lines, added `+` lines, hunk markers `@@`, and live `+added/-removed`
   counters) with real 1-based file line numbers and zero O(n^2) string copying.

2. **Coordinator & TUI Diff Preview Gate (:func:`build_diff`,
   :func:`compute_diff_stats`, :func:`render_rich_diff_panel`,
   :func:`preview_and_confirm`, :func:`register`)**:
   Builds unified diffs and rich colorized terminal diff panels before or after
   any file mutation.
"""

from __future__ import annotations

import difflib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

DIFF_TOOLS = ("write_file", "edit_file", "apply_patch", "MultiEdit", "multiedit")
MAX_DIFF_LINES = 200
_DIFF_CONTEXT = 3


def _read_current(path_str: Any) -> Optional[str]:
    """Read the file as it stands on disk; None when unreadable."""
    try:
        p = Path(str(path_str)).expanduser()
    except Exception:
        return None
    try:
        if p.is_dir():
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def compute_diff_stats(old: str, new: str) -> Dict[str, Any]:
    """Compute exact line addition/removal/hunk statistics between old and new text."""
    old_lines = old.splitlines() if old else []
    new_lines = new.splitlines() if new else []
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    added = removed = hunks = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hunks += 1
        if tag in ("delete", "replace"):
            removed += i2 - i1
        if tag in ("insert", "replace"):
            added += j2 - j1
    return {
        "added": added,
        "removed": removed,
        "hunks": hunks,
        "is_new": len(old_lines) == 0 and len(new_lines) > 0,
        "old_lines": len(old_lines),
        "new_lines": len(new_lines),
    }


def _unified(old: str, new: str, path: str) -> str:
    """Unified diff of old -> new, capped at MAX_DIFF_LINES lines."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="a/" + path if old else "/dev/null",
            tofile="b/" + path,
            lineterm="",
            n=_DIFF_CONTEXT,
        )
    )
    if not diff:
        return ""
    if len(diff) > MAX_DIFF_LINES:
        extra = len(diff) - MAX_DIFF_LINES
        diff = diff[:MAX_DIFF_LINES]
        diff.append(f"... [truncated: {extra} more diff line(s)]")
    return "\n".join(diff)


def _summarize(path: str, is_new: bool) -> str:
    if is_new:
        return f"--- {path} (new file)"
    return f"--- {path}"


# ---------------------------------------------------------------------------
# Real-time streaming JSON field & diff extractor
# ---------------------------------------------------------------------------


@dataclass
class DiffStreamLine:
    """One line emitted by :class:`LiveDiffStreamer` during token streaming."""

    kind: str  # "add" | "del" | "ctx" | "hunk" | "meta"
    text: str
    line_no: int = 0


@dataclass
class LiveDiffStreamer:
    """Stateful O(1)-amortized incremental parser that turns raw streaming JSON
    tool-call chunks into real-time diff lines as the LLM generates them."""

    tool_name: str
    buf: str = ""
    path_resolved: str = ""
    header_emitted: bool = False
    old_emitted: bool = False
    content_key: str = ""
    content_start: int = -1
    raw_pos: int = 0
    pending: str = ""
    done: bool = False
    added_count: int = 0
    removed_count: int = 0
    start_line: int = 1
    held_lines: List[str] = field(default_factory=list)
    existing_disk_text: Optional[str] = None
    _key_re: Optional[re.Pattern] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        key_map = {
            "write_file": "content",
            "edit_file": "new_string",
            "MultiEdit": "new_string",
            "multiedit": "new_string",
            "apply_patch": "patch",
        }
        self.content_key = key_map.get(self.tool_name, "content")
        self._key_re = re.compile(r'"' + re.escape(self.content_key) + r'"\s*:\s*"')

    @staticmethod
    def _unescape_json_fragment(s: str) -> Tuple[str, int, bool]:
        out: List[str] = []
        i = 0
        n = len(s)
        done = False
        while i < n:
            c = s[i]
            if c == '"':
                done = True
                i += 1
                break
            if c != "\\":
                out.append(c)
                i += 1
                continue
            if i + 1 >= n:
                break
            e = s[i + 1]
            if e == "u":
                if i + 6 > n:
                    break
                hexs = s[i + 2 : i + 6]
                try:
                    cp = int(hexs, 16)
                    out.append("\\u" + hexs if 0xD800 <= cp <= 0xDFFF else chr(cp))
                except ValueError:
                    out.append("\\u" + hexs)
                i += 6
            else:
                mapping = {
                    "n": "\n",
                    "t": "\t",
                    "r": "\r",
                    "b": "\b",
                    "f": "\f",
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                }
                out.append(mapping.get(e, "\\" + e))
                i += 2
        return "".join(out), i, done

    @classmethod
    def extract_complete_field(cls, buf: str, key: str) -> Optional[str]:
        m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', buf)
        if not m:
            return None
        val, _, done = cls._unescape_json_fragment(buf[m.end() :])
        return val if done else None

    def detect_path(self) -> str:
        if self.path_resolved:
            return self.path_resolved
        for k in ("path", "file_path"):
            val = self.extract_complete_field(self.buf, k)
            if val:
                self.path_resolved = val
                self.existing_disk_text = _read_current(val)
                return val
        return ""

    def _compute_hunk_start(self, old_str: str) -> int:
        if not self.existing_disk_text or not old_str:
            return 1
        idx = self.existing_disk_text.find(old_str)
        if idx < 0:
            first_ln = old_str.splitlines()[0] if old_str.splitlines() else ""
            idx = self.existing_disk_text.find(first_ln) if first_ln else -1
        return (self.existing_disk_text.count("\n", 0, idx) + 1) if idx >= 0 else 1

    def feed(self, chunk: str) -> List[DiffStreamLine]:
        """Feed a streaming JSON argument chunk and return newly available DiffStreamLines."""
        self.buf += chunk
        self.detect_path()
        emitted: List[DiffStreamLine] = []

        # Pull newly streamed lines from the target content field
        new_raw_lines: List[str] = []
        if not self.done and self._key_re is not None:
            if self.content_start < 0:
                search_from = max(0, len(self.buf) - len(chunk) - 128)
                m = self._key_re.search(self.buf, search_from)
                if m:
                    self.content_start = m.end()
            if self.content_start >= 0:
                tail = self.buf[self.content_start + self.raw_pos :]
                text, consumed, done = self._unescape_json_fragment(tail)
                self.raw_pos += consumed
                self.done = done
                self.pending += text
                while "\n" in self.pending:
                    line, self.pending = self.pending.split("\n", 1)
                    new_raw_lines.append(line)

        if self.tool_name in ("edit_file", "MultiEdit", "multiedit"):
            if not self.old_emitted:
                old_s = self.extract_complete_field(self.buf, "old_string")
                if old_s is not None:
                    self.start_line = self._compute_hunk_start(old_s)
                    old_lns = old_s.splitlines()
                    if self.start_line > 1:
                        emitted.append(
                            DiffStreamLine(
                                "hunk",
                                f"@@ -{self.start_line},{len(old_lns)} +{self.start_line} @@",
                                self.start_line,
                            )
                        )
                    for idx, ol in enumerate(old_lns):
                        self.removed_count += 1
                        emitted.append(
                            DiffStreamLine("del", ol, self.start_line + idx)
                        )
                    self.old_emitted = True
                    for held in self.held_lines:
                        lineno = self.start_line + self.added_count
                        self.added_count += 1
                        emitted.append(DiffStreamLine("add", held, lineno))
                    self.held_lines.clear()
            if self.old_emitted:
                for ln in new_raw_lines:
                    lineno = self.start_line + self.added_count
                    self.added_count += 1
                    emitted.append(DiffStreamLine("add", ln, lineno))
            else:
                self.held_lines.extend(new_raw_lines)
        elif self.tool_name == "apply_patch":
            for ln in new_raw_lines:
                if ln.startswith("@@"):
                    emitted.append(DiffStreamLine("hunk", ln, 0))
                elif ln.startswith(("+++ ", "--- ")):
                    emitted.append(DiffStreamLine("meta", ln, 0))
                elif ln.startswith("+"):
                    self.added_count += 1
                    emitted.append(DiffStreamLine("add", ln[1:], self.added_count))
                elif ln.startswith("-"):
                    self.removed_count += 1
                    emitted.append(DiffStreamLine("del", ln[1:], self.removed_count))
                else:
                    emitted.append(DiffStreamLine("ctx", ln, 0))
        else:  # write_file
            # If overwriting an existing file, emit removed count summary once
            if (
                not self.old_emitted
                and self.existing_disk_text
                and self.path_resolved
            ):
                old_lns = self.existing_disk_text.splitlines()
                self.removed_count = len(old_lns)
                for idx, ol in enumerate(old_lns[:12], 1):
                    emitted.append(DiffStreamLine("del", ol, idx))
                if len(old_lns) > 12:
                    emitted.append(
                        DiffStreamLine(
                            "hunk",
                            f"@@ … -{len(old_lns) - 12} more overwritten line(s) @@",
                            0,
                        )
                    )
                self.old_emitted = True
            for ln in new_raw_lines:
                self.added_count += 1
                emitted.append(DiffStreamLine("add", ln, self.added_count))
        return emitted

    def flush(self) -> List[DiffStreamLine]:
        """Flush any remaining held or trailing un-newlined text."""
        out: List[DiffStreamLine] = []
        if self.held_lines:
            for held in self.held_lines:
                lineno = self.start_line + self.added_count
                self.added_count += 1
                out.append(DiffStreamLine("add", held, lineno))
            self.held_lines.clear()
        if self.pending:
            line, self.pending = self.pending, ""
            if self.tool_name == "apply_patch":
                if line.startswith("+"):
                    self.added_count += 1
                    out.append(DiffStreamLine("add", line[1:], self.added_count))
                elif line.startswith("-"):
                    self.removed_count += 1
                    out.append(DiffStreamLine("del", line[1:], self.removed_count))
                else:
                    out.append(DiffStreamLine("ctx", line, 0))
            else:
                lineno = self.start_line + self.added_count
                self.added_count += 1
                out.append(DiffStreamLine("add", line, lineno))
        return out


# ---------------------------------------------------------------------------
# Diff builders per tool
# ---------------------------------------------------------------------------


def _diff_write_file(args: Dict[str, Any]) -> str:
    path = str(args.get("path", ""))
    content = args.get("content", "")
    if not isinstance(content, str):
        return f"--- {path}\n(cannot preview: content is not a string)"
    old = _read_current(path)
    if old is None:
        old = ""
    new = content
    return _summarize(path, old == "") + "\n" + (
        _unified(old, new, path) or "(no changes — content matches the file)"
    )


def _diff_edit_file(args: Dict[str, Any]) -> str:
    path = str(args.get("path", ""))
    old_s = args.get("old_string", "")
    new_s = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))
    old = _read_current(path)
    if old is None:
        return f"--- {path}\n(cannot preview: file not readable)"
    if not isinstance(old_s, str) or old_s == "":
        return f"--- {path}\n(cannot preview: old_string is empty)"
    if old_s not in old:
        return (
            f"--- {path}\n(cannot preview: old_string NOT FOUND in "
            "file — this edit will fail)"
        )
    new = old.replace(old_s, str(new_s), -1 if replace_all else 1)
    return _summarize(path, False) + "\n" + (
        _unified(old, new, path) or "(no changes)"
    )


def _diff_multiedit(args: Dict[str, Any]) -> str:
    path = str(args.get("file_path", ""))
    edits = args.get("edits", [])
    old = _read_current(path)
    if old is None:
        return f"--- {path}\n(cannot preview: file not readable)"
    if not isinstance(edits, list) or not edits:
        return f"--- {path}\n(cannot preview: edits list is empty)"
    new = old
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"--- {path}\n(cannot preview: edit #{i} is not a dict)"
        old_s = edit.get("old_string")
        new_s = edit.get("new_string")
        if not isinstance(old_s, str) or not old_s:
            return f"--- {path}\n(cannot preview: edit #{i} has no old_string)"
        if not isinstance(new_s, str):
            return f"--- {path}\n(cannot preview: edit #{i} has no new_string)"
        if old_s not in new:
            return (
                f"--- {path}\n(cannot preview: edit #{i} old_string "
                "not found — this MultiEdit will fail)"
            )
        new = new.replace(old_s, new_s, 1)
    return _summarize(path, False) + "\n" + (
        _unified(old, new, path) or "(no changes)"
    )


def _diff_apply_patch(args: Dict[str, Any]) -> str:
    patch = args.get("patch", "")
    if not isinstance(patch, str) or not patch.strip():
        return "(cannot preview: patch is empty)"
    lines = patch.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        extra = len(lines) - MAX_DIFF_LINES
        lines = lines[:MAX_DIFF_LINES] + [
            f"... [truncated: {extra} more patch line(s)]"
        ]
    return "\n".join(lines)


_BUILDERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "write_file": _diff_write_file,
    "edit_file": _diff_edit_file,
    "MultiEdit": _diff_multiedit,
    "multiedit": _diff_multiedit,
    "apply_patch": _diff_apply_patch,
}


def build_diff(tool_name: str, args: Dict[str, Any]) -> str:
    """Build a unified diff for a mutating tool call."""
    builder = _BUILDERS.get(str(tool_name))
    if builder is None:
        return f"(no diff preview available for tool {tool_name!r})"
    try:
        return builder(args or {})
    except Exception as e:
        return f"(diff preview failed: {e})"


def render_rich_diff_panel(
    tool_name: str,
    args: Dict[str, Any],
    *,
    max_lines: int = 48,
    palette: Optional[Dict[str, str]] = None,
) -> Any:
    """Render a Rich Panel containing a colorized unified diff with +N/-M stats."""
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    pal = palette or {
        "border": "#6272a4",
        "accent": "#bd93f9",
        "cyan": "#8be9fd",
        "green": "#50fa7b",
        "yellow": "#f1fa8c",
        "red": "#ff5555",
        "fg": "#f8f8f2",
        "dim": "#6272a4",
    }
    diff_str = build_diff(tool_name, args or {})
    lines = diff_str.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    path = _preview_path(tool_name, args or {}) or tool_name

    body = Text()
    for idx, ln in enumerate(lines[:max_lines]):
        if idx > 0:
            body.append("\n")
        if ln.startswith("@@"):
            body.append(ln, style=f"bold {pal['cyan']}")
        elif ln.startswith(("+++", "---")):
            body.append(ln, style=f"bold {pal['dim']}")
        elif ln.startswith("+"):
            body.append("+ ", style=f"bold {pal['green']}")
            body.append(ln[1:], style=pal["green"])
        elif ln.startswith("-"):
            body.append("- ", style=f"bold {pal['red']}")
            body.append(ln[1:], style=pal["red"])
        else:
            body.append(ln, style=pal["fg"])
    if len(lines) > max_lines:
        body.append(
            f"\n… (+{len(lines) - max_lines} more diff lines)",
            style=pal["dim"],
        )

    title = (
        f"[bold {pal['accent']}]Δ Live Diff[/] · "
        f"[bold {pal['cyan']}]{path}[/] · "
        f"[bold {pal['green']}]+{added}[/] [bold {pal['red']}]-{removed}[/]"
    )
    return Panel(
        body,
        title=title,
        box=box.ROUNDED,
        border_style=pal["border"],
        padding=(0, 1),
        expand=True,
    )


# ---------------------------------------------------------------------------
# UI plumbing (duck-typed — TUI, headless, or tests)
# ---------------------------------------------------------------------------


def _print_diff(diff: str, tool_name: str, ui: Any) -> None:
    lines = diff.split("\n")
    if len(lines) > 30:
        shown = "\n".join(lines[:30])
        diff = f"{shown}\n... ({len(lines) - 30} more lines, {len(lines)} total)"
    header = f"Diff preview — {tool_name}:"
    body = header + "\n" + diff
    try:
        if ui is None:
            print(body)
        elif hasattr(ui, "print_info"):
            ui.print_info(body)
        elif hasattr(ui, "console"):
            ui.console.print(body)
        elif hasattr(ui, "print"):
            ui.print(body)
        else:
            print(body)
    except Exception:
        print(body)


def _ask_yes_no(prompt: str, ui: Any) -> bool:
    answer = None
    try:
        ask = getattr(ui, "ask_yes_no", None) or getattr(ui, "confirm", None)
        if ask is not None:
            answer = ask(prompt)
        elif sys.stdin.isatty():
            try:
                answer = input(prompt + " ")
            except (EOFError, KeyboardInterrupt):
                return False
        else:
            return False
    except Exception:
        return False
    if answer is None:
        return False
    a = str(answer).strip().lower()
    return a in ("", "y", "yes")


def _preview_path(tool_name: str, args: Dict[str, Any]) -> str:
    if str(tool_name) in ("write_file", "edit_file"):
        return str(args.get("path", "")).strip()
    if str(tool_name) in ("MultiEdit", "multiedit"):
        return str(args.get("file_path", "")).strip()
    return ""


def preview_and_confirm(
    tool_name: str, args: Dict[str, Any], ui: Any = None
) -> bool:
    diff = build_diff(tool_name, args or {})
    _print_diff(diff, tool_name, ui)
    path = _preview_path(tool_name, args or {})
    if path:
        prompt = f"Apply changes to {path}? [Y/n]"
    else:
        prompt = f"Apply {tool_name}? [Y/n]"
    return _ask_yes_no(prompt, ui)


def _permission_mode(agent: Any) -> Optional[str]:
    pm = getattr(agent, "permissions", None)
    if pm is None:
        pm = getattr(agent, "perms", None)
    if pm is None:
        pm = getattr(agent, "permission_manager", None)
    if pm is None:
        return None
    mode = getattr(pm, "mode", None)
    try:
        return str(mode() if callable(mode) else mode)
    except Exception:
        return None


def register(agent: Any) -> None:
    def _bound(
        tool_name: str, args: Dict[str, Any], ui: Any = None
    ) -> bool:
        if os.environ.get("FULLAGENT_DIFF_PREVIEW", "") != "1":
            return True
        if _permission_mode(agent) == "bypassPermissions":
            return True
        return preview_and_confirm(tool_name, args, ui)

    agent.diff_preview = _bound


if __name__ == "__main__":
    import shutil
    import tempfile

    def check(name: str, cond: bool) -> None:
        if not cond:
            raise AssertionError(f"self-test FAILED: {name}")

    os.environ["FULLAGENT_DIFF_PREVIEW"] = "1"
    tmp = tempfile.mkdtemp(prefix="diffpreview_selftest_")
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        class FakeUI:
            def __init__(self, answers=()):
                self.printed = []
                self.answers = list(answers)

            def print_info(self, text):
                self.printed.append(text)

            def confirm(self, prompt):
                self.printed.append("PROMPT: " + prompt)
                return self.answers.pop(0) if self.answers else "y"

        class FakePerms:
            def __init__(self, mode):
                self._m = mode

            @property
            def mode(self):
                return self._m

        class FakeAgent:
            def __init__(self, mode=None):
                self.permissions = FakePerms(mode) if mode else None

        d = build_diff("write_file", {"path": "new.txt", "content": "hello\nworld\n"})
        check("write_new diff has plus lines", "+hello" in d and "+world" in d)
        check("write_new labels new file", "/dev/null" in d)

        Path("old.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        d = build_diff("write_file", {"path": "old.txt", "content": "alpha\nGAMMA\n"})
        check("write_overwrite minus line", "-beta" in d)
        check("write_overwrite plus line", "+GAMMA" in d)
        check("write_overwrite context", "alpha" in d)

        d = build_diff("write_file", {"path": "old.txt", "content": "alpha\nbeta\n"})
        check("write_identical reports no changes", "no changes" in d)

        d = build_diff(
            "edit_file",
            {"path": "old.txt", "old_string": "beta\n", "new_string": "BETA!\n"},
        )
        check("edit minus/plus", "-beta" in d and "+BETA!" in d)

        d = build_diff(
            "edit_file",
            {"path": "old.txt", "old_string": "nope", "new_string": "x"},
        )
        check("edit_missing warns", "NOT FOUND" in d)

        d = build_diff(
            "edit_file",
            {"path": "missing.txt", "old_string": "a", "new_string": "b"},
        )
        check("edit_unreadable note", "not readable" in d)

        d = build_diff(
            "MultiEdit",
            {
                "file_path": "old.txt",
                "edits": [
                    {"old_string": "alpha", "new_string": "ALPHA"},
                    {"old_string": "beta", "new_string": "BETA"},
                ],
            },
        )
        check(
            "multiedit diff",
            "-alpha" in d and "+ALPHA" in d and "-beta" in d and "+BETA" in d,
        )

        d = build_diff(
            "MultiEdit",
            {
                "file_path": "old.txt",
                "edits": [{"old_string": "missing", "new_string": "x"}],
            },
        )
        check("multiedit_bad warns", "not found" in d)

        patch = (
            "--- a/old.txt\n+++ b/old.txt\n@@ -1,2 +1,2 @@\n"
            " alpha\n-beta\n+BETA\n"
        )
        d = build_diff("apply_patch", {"patch": patch})
        check("apply_patch passthrough", "-beta" in d and "+BETA" in d)

        big = "".join(f"line{i}\n" for i in range(500))
        d = build_diff("write_file", {"path": "big.txt", "content": big})
        lines = d.splitlines()
        check("truncation caps lines", len(lines) <= MAX_DIFF_LINES + 3)
        check("truncation note present", "truncated" in d)

        d = build_diff("run_command", {"command": "ls"})
        check("unknown tool note", "no diff preview" in d)

        ui = FakeUI(answers=["y"])
        ok = preview_and_confirm(
            "edit_file",
            {"path": "old.txt", "old_string": "beta\n", "new_string": "BETA!\n"},
            ui,
        )
        check("confirm yes proceeds", ok is True)
        check("diff printed via ui", any("-beta" in p for p in ui.printed))
        check("prompt mentions path", any("old.txt" in p for p in ui.printed))

        ui = FakeUI(answers=["n"])
        ok = preview_and_confirm(
            "write_file", {"path": "new2.txt", "content": "x\n"}, ui
        )
        check("confirm no aborts", ok is False)

        ui = FakeUI(answers=[""])
        ok = preview_and_confirm(
            "write_file", {"path": "new2.txt", "content": "x\n"}, ui
        )
        check("empty answer proceeds", ok is True)

        class FakeStdin:
            def isatty(self):
                return False

        real_stdin = sys.stdin
        try:
            sys.stdin = FakeStdin()
            ok = preview_and_confirm("run_command", {"command": "x"}, None)
        finally:
            sys.stdin = real_stdin
        check("non-tty fallback denies (fail closed)", ok is False)

        agent = FakeAgent("bypassPermissions")
        register(agent)
        check("diff_preview attached", callable(agent.diff_preview))
        ui = FakeUI()
        ok = agent.diff_preview(
            "write_file", {"path": "new2.txt", "content": "x\n"}, ui
        )
        check("bypass skips preview", ok is True and ui.printed == [])

        agent = FakeAgent("plan")
        register(agent)
        ui = FakeUI(answers=["n"])
        ok = agent.diff_preview(
            "edit_file",
            {"path": "old.txt", "old_string": "beta\n", "new_string": "BETA!\n"},
            ui,
        )
        check("plan mode previews and honors no", ok is False)

        agent = FakeAgent(None)
        register(agent)
        ui = FakeUI(answers=["y"])
        ok = agent.diff_preview(
            "write_file", {"path": "new2.txt", "content": "x\n"}, ui
        )
        check("unknown mode previews", ok is True)

        # -- LiveDiffStreamer real-time streaming test -----------------------
        streamer = LiveDiffStreamer("edit_file")
        chunks = [
            '{"path": "old.txt", ',
            '"old_string": "beta\\n", ',
            '"new_string": "BETA_LINE_1\\n',
            'BETA_LINE_2\\n"}',
        ]
        streamed_lines = []
        for ch in chunks:
            streamed_lines.extend(streamer.feed(ch))
        streamed_lines.extend(streamer.flush())
        check("LiveDiffStreamer resolves path", streamer.path_resolved == "old.txt")
        check(
            "LiveDiffStreamer emits del and add lines",
            any(l.kind == "del" and l.text == "beta" for l in streamed_lines)
            and any(l.kind == "add" and l.text == "BETA_LINE_1" for l in streamed_lines)
            and any(l.kind == "add" and l.text == "BETA_LINE_2" for l in streamed_lines),
        )
        check(
            "LiveDiffStreamer counts +2/-1",
            streamer.added_count == 2 and streamer.removed_count == 1,
        )
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    print("diffpreview self-test PASSED")

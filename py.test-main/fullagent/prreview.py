"""GitHub PR review (`/pr-review` command + `ReviewPR` tool).

Fetches a pull-request diff — `gh pr diff` first, falling back to the
public GitHub diff API — runs quick heuristic checks (secrets, missing
tests, TODOs, diff size), then returns a report that hands the diff to
the MODEL for a deep review: heuristics + diff stat + the diff itself.

Usage from the TUI::

    /pr-review 123
    /pr-review 123 owner/repo
    /pr-review owner/repo#123
    /pr-review https://github.com/owner/repo/pull/123

The coordinator wires the TUI command (``handle_pr_review``) and the
agent tool (``register``). No imports of ``.agent``/``.tui`` at module
level, so there are no import cycles.

``python3 -m fullagent.prreview`` runs the built-in self-test (the fetch
layer is mocked there — no network).
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .tools import Tool

USAGE = (
    "usage:\n"
    "  /pr-review 123\n"
    "  /pr-review 123 owner/repo\n"
    "  /pr-review owner/repo#123\n"
    "  /pr-review https://github.com/owner/repo/pull/123\n"
    "A bare number works inside a repo checkout (gh infers the repo) or\n"
    "with the owner/repo as the second token."
)

DIFF_HEAD_LIMIT = 8000   # chars of diff handed to the model
GH_TIMEOUT = 30
API_TIMEOUT = 20
SIZE_WARNING_LINES = 500   # > this many changed lines → size warning
BIG_FILE_LINES = 200       # > this many added lines in one file → high

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# ---------------------------------------------------------------------------
# PR reference parsing
# ---------------------------------------------------------------------------

_GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([^/\s?#]+)/([^/\s?#]+)/pull/(\d+)",
    re.IGNORECASE)
_OWNER_REPO_NUM_RE = re.compile(r"^([^/\s#]+)/([^/\s#]+)#(\d+)$")
_BARE_NUM_RE = re.compile(r"^#?(\d+)$")


def _clean_repo_name(name: str) -> str:
    name = (name or "").strip().rstrip("/")
    if name.lower().endswith(".git"):
        name = name[:-4]
    return name


def parse_pr_ref(pr_ref: str) -> Tuple[Optional[str], Optional[str],
                                       Optional[str]]:
    """Parse a PR reference into (owner, repo, number).

    Accepts a github.com pull URL, ``owner/repo#123``, or a bare number.
    Unknown parts are None; returns (None, None, None) when unparseable.
    """
    text = (pr_ref or "").strip()
    m = _GITHUB_URL_RE.search(text)
    if m:
        return m.group(1), _clean_repo_name(m.group(2)), m.group(3)
    m = _OWNER_REPO_NUM_RE.match(text)
    if m:
        return m.group(1), _clean_repo_name(m.group(2)), m.group(3)
    m = _BARE_NUM_RE.match(text)
    if m:
        return None, None, m.group(1)
    return None, None, None


def _normalize_repo(repo: str) -> str:
    """Normalise a user-supplied ``owner/repo`` token ("" if invalid)."""
    text = (repo or "").strip()
    if text.startswith("https://github.com/") or "@github.com" in text:
        if "/pull/" in text:
            owner, name, _num = parse_pr_ref(text)
            if owner and name:
                return f"{owner}/{name}"
        else:
            m = _GIT_REMOTE_RE.search(text)
            if m:
                return f"{m.group(1)}/{_clean_repo_name(m.group(2))}"
    parts = text.split("/")
    if len(parts) == 2 and all(p.strip() for p in parts):
        return f"{parts[0].strip()}/{_clean_repo_name(parts[1])}"
    return ""


_GIT_REMOTE_RE = re.compile(
    r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?$")


def _repo_from_git() -> str:
    """Best-effort ``owner/repo`` from the cwd's git origin (may be "")."""
    try:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    m = _GIT_REMOTE_RE.search((proc.stdout or "").strip())
    if not m:
        return ""
    return f"{m.group(1)}/{_clean_repo_name(m.group(2))}"


# ---------------------------------------------------------------------------
# Diff fetching: `gh` first, public GitHub diff API as fallback.
# Never raises — returns "ERROR: ..." on failure.
# ---------------------------------------------------------------------------

def _fetch_via_gh(number: str, repo_full: str) -> Optional[str]:
    cmd = ["gh", "pr", "diff", number]
    if repo_full:
        cmd += ["--repo", repo_full]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=GH_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None  # gh missing / timed out → caller tries the API
    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out:
        return out
    return None


def _fetch_via_api(owner: str, repo_name: str, number: str) -> Optional[str]:
    url = (f"https://api.github.com/repos/{owner}/{repo_name}"
           f"/pulls/{number}.diff")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "fullagent-prreview",
    })
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def fetch_pr_diff(pr_ref: str, repo: str = "") -> str:
    """Fetch the unified diff for a PR.

    ``pr_ref``: PR number, ``owner/repo#123``, or a github.com pull URL.
    ``repo``: optional ``owner/repo`` (needed for a bare number outside a
    repo checkout). Returns the diff text, or an ``"ERROR: ..."`` string —
    never raises.
    """
    try:
        owner, repo_name, number = parse_pr_ref(pr_ref)
        if not number:
            return (f"ERROR: cannot parse PR reference '{pr_ref}'. "
                    f"{USAGE}")
        repo_full = ""
        if owner and repo_name:
            repo_full = f"{owner}/{repo_name}"
        elif repo:
            repo_full = _normalize_repo(repo)

        diff = _fetch_via_gh(number, repo_full)
        if not diff:
            if not repo_full:
                repo_full = _repo_from_git()
            if repo_full and "/" in repo_full:
                o, r = repo_full.split("/", 1)
                diff = _fetch_via_api(o, r, number)
        if diff:
            return diff
        return (f"ERROR: could not fetch diff for '{pr_ref}' "
                f"(gh CLI failed and the GitHub API fallback found no "
                f"public repo{' for ' + repo_full if repo_full else ''}).")
    except Exception as e:  # noqa: BLE001 — contract: never raise
        return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Diff parsing + heuristics
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")
# both line ranges are captured so the parser knows exactly where a hunk
# ends (header-looking content inside a hunk is data, not a header)
_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|client[_-]?secret|"
                r"auth[_-]?token|access[_-]?token)\b\s*[:=]\s*\S+"),
     "possible hardcoded secret"),
    (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9-]{7,}"),
     "possible secret key (sk-...)"),
    (re.compile(r"(?i)\b(password|passwd|pwd)\b\s*[:=]\s*\S+"),
     "possible hardcoded password"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |DSA |EC )?PRIVATE KEY-----"),
     "private key material in diff"),
    (re.compile(r"(?i)\baws[_-]?(?:access[_-]?key[_-]?id|"
                r"secret[_-]?access[_-]?key)\b\s*[:=]\s*\S+"),
     "possible AWS credential"),
]
_TODO_RE = re.compile(r"(?i)\b(TODO|FIXME|XXX|HACK)\b")

_SRC_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".go",
             ".rs", ".java", ".rb", ".php", ".c", ".cc", ".cpp", ".h",
             ".hpp", ".cs", ".swift", ".kt", ".scala", ".sh"}


def parse_diff(diff: str) -> List[Dict[str, Any]]:
    """Split a unified diff into per-file records.

    Each record: ``{"file", "added" (list of (new_lineno, text)),
    "added_count", "removed_count"}``. Tolerates malformed input.
    """
    files: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    new_lineno = 0
    in_hunk = False
    old_left = new_left = 0
    for line in (diff or "").splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            cur = {"file": m.group(2), "added": [],
                   "added_count": 0, "removed_count": 0}
            files.append(cur)
            in_hunk = False
            continue
        if cur is None:
            continue
        m = _HUNK_RE.match(line)
        if m:
            new_lineno = int(m.group(3))
            old_left = int(m.group(2) if m.group(2) is not None else 1)
            new_left = int(m.group(4) if m.group(4) is not None else 1)
            in_hunk = old_left > 0 or new_left > 0
            continue
        if not in_hunk and line.startswith(("+++ ", "--- ")):
            continue
        if line.startswith("+"):
            cur["added"].append((new_lineno, line[1:]))
            cur["added_count"] += 1
            new_lineno += 1
            if in_hunk and new_left > 0:
                new_left -= 1
        elif line.startswith("-"):
            cur["removed_count"] += 1
            if in_hunk and old_left > 0:
                old_left -= 1
        elif line.startswith(" "):
            new_lineno += 1
            if in_hunk:
                old_left = max(0, old_left - 1)
                new_left = max(0, new_left - 1)
        # "\ No newline at end of file" and anything else: ignored
        if in_hunk and old_left <= 0 and new_left <= 0:
            in_hunk = False
    return files


def _is_test_file(path: str) -> bool:
    base = path.rsplit("/", 1)[-1].lower()
    p = path.lower()
    stem = base.rsplit(".", 1)[0]
    return (bool(re.search(r"(^|[_\-\.])tests?([_\-\.]|$)", stem))
            or bool(re.search(r"(^|[_\-\.])specs?([_\-\.]|$)", stem))
            or "/test/" in p or p.startswith("test/")
            or "/tests/" in p or p.startswith("tests/")
            or "/spec/" in p or "/__tests__/" in p)


def _is_src_file(path: str) -> bool:
    if _is_test_file(path):
        return False
    return os.path.splitext(path)[1].lower() in _SRC_EXTS


def _finding(severity: str, file: str, line: int, message: str) -> Dict:
    return {"severity": severity, "file": file, "line": line,
            "message": message}


def quick_heuristics(diff: str) -> List[Dict[str, Any]]:
    """Quick file-level checks over a unified diff.

    Returns findings as ``{"severity", "file", "line", "message"}``
    dicts (severity in critical/high/medium/low), ordered by severity.
    Pure function — no I/O.
    """
    files = parse_diff(diff)
    findings: List[Dict[str, Any]] = []

    total = sum(f["added_count"] + f["removed_count"] for f in files)
    if total > SIZE_WARNING_LINES:
        findings.append(_finding(
            "medium", "", 0,
            f"large diff: ~{total} changed lines across {len(files)} "
            f"file(s) — review in chunks, not in one pass"))

    for f in files:
        if f["added_count"] > BIG_FILE_LINES:
            findings.append(_finding(
                "high", f["file"], 0,
                f"large single-file change: +{f['added_count']} lines in "
                f"one file — consider splitting the PR"))
        for lineno, text in f["added"]:
            for rx, label in _SECRET_PATTERNS:
                m = rx.search(text)
                if m:
                    findings.append(_finding(
                        "critical", f["file"], lineno,
                        f"{label}: {m.group(0)[:60]}"))
                    break
            if _TODO_RE.search(text):
                findings.append(_finding(
                    "low", f["file"], lineno,
                    f"new TODO/FIXME marker: {text.strip()[:80]}"))

    src_files = [f for f in files if _is_src_file(f["file"])]
    if src_files and not any(_is_test_file(f["file"]) for f in files):
        findings.append(_finding(
            "medium", "", 0,
            f"source changed ({len(src_files)} file(s)) but no test "
            f"files were touched — ask for tests"))

    findings.sort(key=lambda f: _SEVERITY_RANK.get(f["severity"], 9))
    return findings


def diff_stat(files: List[Dict[str, Any]]) -> str:
    """One-line stat: '3 files changed, +120 / -45'."""
    n = len(files)
    added = sum(f["added_count"] for f in files)
    removed = sum(f["removed_count"] for f in files)
    return (f"{n} file{'s' if n != 1 else ''} changed, "
            f"+{added} / -{removed}")


def build_report(ref_label: str, diff: str) -> str:
    """Structured report for the MODEL to do the deep review.

    Returns heuristics + diff stat + the first ~8000 chars of the diff,
    ending with a brief telling the model to review thoroughly. If
    ``diff`` is an ``"ERROR: ..."`` string it is returned unchanged.
    """
    if diff.startswith("ERROR:"):
        return diff
    files = parse_diff(diff)
    findings = quick_heuristics(diff)
    lines = [
        f"PR review: {ref_label}",
        "",
        "## Diff stat",
        diff_stat(files),
        "",
        f"## Quick heuristic findings ({len(findings)})",
    ]
    if findings:
        for f in findings:
            loc = f"{f['file']}:{f['line']}" if f["file"] else "repo"
            lines.append(f"[{f['severity'].upper()}] {loc} — {f['message']}")
    else:
        lines.append("(none)")
    lines += ["", "## Diff (first 8000 chars — review thoroughly)"]
    head = diff[:DIFF_HEAD_LIMIT]
    lines.append(head)
    if len(diff) > DIFF_HEAD_LIMIT:
        lines.append(f"\n… [{len(diff) - DIFF_HEAD_LIMIT} chars truncated] …")
    lines += [
        "",
        "---",
        "Deep-review brief: walk the diff hunk by hunk and report logic "
        "bugs,",
        "edge cases, security issues, API misuse, and test gaps. Cite "
        "file:line",
        "for each finding. The heuristic findings above are quick scans "
        "only —",
        "confirm or refute each one during your review.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TUI command handler + agent tool
# ---------------------------------------------------------------------------

def _ref_label(pr_ref: str, repo: str) -> str:
    owner, name, number = parse_pr_ref(pr_ref)
    if owner and name and number:
        return f"{owner}/{name}#{number}"
    repo_full = _normalize_repo(repo)
    if repo_full and number:
        return f"{repo_full}#{number}"
    return f"PR {pr_ref}"


def handle_pr_review(ui, arg: str) -> None:
    """TUI `/pr-review` handler — prints the review report (returns None).

    ``ui`` is the TUI (duck-typed: ``.print_info`` / ``.print_error``).
    """
    text = (arg or "").strip()
    if not text:
        ui.print_error(USAGE)
        return
    parts = text.split(None, 1)
    pr_ref = parts[0]
    repo = parts[1] if len(parts) > 1 else ""
    diff = fetch_pr_diff(pr_ref, repo)
    if diff.startswith("ERROR:"):
        ui.print_error(diff)
        return
    ui.print_info(build_report(_ref_label(pr_ref, repo), diff))


def _handle_review_pr(pr: str = "", repo: str = "", **kwargs: Any) -> str:
    pr = str(pr or "").strip()
    if not pr:
        return "Error: 'pr' is required (PR number, owner/repo#123, or a pull URL)."
    return build_report(pr, fetch_pr_diff(pr, str(repo or "")))


def make_review_pr_tool() -> Tool:
    return Tool(
        name="ReviewPR",
        description=(
            "Fetch a GitHub PR diff and return a review report (quick "
            "heuristic findings + diff stat + the diff) for a deep code "
            "review. 'pr' accepts a PR number, owner/repo#123, or a "
            "github.com pull URL. 'repo' (owner/repo) is only needed when "
            "'pr' is a bare number outside a git checkout."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pr": {
                    "type": "string",
                    "description": ("PR number, owner/repo#123, or full "
                                    "github.com pull URL"),
                },
                "repo": {
                    "type": "string",
                    "description": ("optional owner/repo — needed for a "
                                    "bare PR number outside a repo checkout"),
                },
            },
            "required": ["pr"],
        },
        handler=_handle_review_pr,
    )


def register(agent: Any) -> None:
    """Wire the ReviewPR tool into an agent (duck-typed, no imports)."""
    agent.tools["ReviewPR"] = make_review_pr_tool()


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.prreview`  →  PASS (no network)
# ---------------------------------------------------------------------------

_SAMPLE_DIFF = """\
diff --git a/fullagent/prreview.py b/fullagent/prreview.py
index 1111111..2222222 100644
--- a/fullagent/prreview.py
+++ b/fullagent/prreview.py
@@ -1,3 +1,6 @@
 line one
-old line
+new line
+API_KEY = "not-a-real-key-just-for-tests"
+# TODO: rotate this key later
+regular = True
\\ No newline at end of file
"""

_BIG_DIFF = "".join(
    ["diff --git a/src/big.py b/src/big.py\n",
     "--- a/src/big.py\n", "+++ b/src/big.py\n", "@@ -1,1 +1,1 @@\n"]
    + [f"+added_line_{i} = {i}\n" for i in range(600)]
)


class _FakeAgent:
    def __init__(self):
        self.tools = {}


class _FakeUI:
    def __init__(self):
        self.agent = _FakeAgent()
        self.infos = []
        self.errors = []

    def print_info(self, text, color=None):
        self.infos.append(text)

    def print_error(self, text):
        self.errors.append(text)


class _FakeResp:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _selftest() -> None:
    import subprocess as _sp
    import sys as _sys
    import urllib.request as _urlreq
    from unittest import mock

    _me = _sys.modules[__name__]  # this module (runs as __main__)

    # --- parse_pr_ref -------------------------------------------------
    assert parse_pr_ref(
        "https://github.com/octo/hello/pull/5") == ("octo", "hello", "5")
    assert parse_pr_ref(
        "https://github.com/octo/hello/pull/5/files") == ("octo", "hello",
                                                          "5")
    assert parse_pr_ref("octo/hello#42") == ("octo", "hello", "42")
    assert parse_pr_ref("7") == (None, None, "7")
    assert parse_pr_ref("#12") == (None, None, "12")
    assert parse_pr_ref("junk") == (None, None, None)
    assert _normalize_repo("octo/hello") == "octo/hello"
    assert _normalize_repo("octo/hello.git") == "octo/hello"
    assert _normalize_repo("nope") == ""

    # --- parse_diff ---------------------------------------------------
    files = parse_diff(_SAMPLE_DIFF)
    assert len(files) == 1, files
    f0 = files[0]
    assert f0["file"] == "fullagent/prreview.py", f0
    assert f0["added_count"] == 4, f0
    assert f0["removed_count"] == 1, f0
    # line numbers: ctx@1, del, add@2, secret@3, todo@4, regular@5
    secret_line = [ln for ln, tx in f0["added"]
                   if "API_KEY" in tx]
    assert secret_line == [3], f0["added"]

    # --- quick_heuristics: secret flagged critical --------------------
    findings = quick_heuristics(_SAMPLE_DIFF)
    crits = [f for f in findings if f["severity"] == "critical"]
    assert len(crits) == 1, findings
    assert crits[0]["file"] == "fullagent/prreview.py"
    assert crits[0]["line"] == 3, crits
    # TODO flagged low
    lows = [f for f in findings if f["severity"] == "low"]
    assert any("TODO" in f["message"] for f in lows), findings
    # src changed, no tests → medium
    meds = [f for f in findings if f["severity"] == "medium"]
    assert any("no test" in f["message"] for f in meds), findings
    # severity ordering: critical first
    assert findings[0]["severity"] == "critical", findings

    # --- sk- secret pattern -------------------------------------------
    sk_diff = ("diff --git a/x.py b/x.py\n"
               "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n a\n"
               '+key = "sk-test-fake-key-12345678"\n')
    sk_findings = quick_heuristics(sk_diff)
    assert any(f["severity"] == "critical" and "sk-" in f["message"]
               for f in sk_findings), sk_findings

    # --- size warnings on a big diff ----------------------------------
    big = quick_heuristics(_BIG_DIFF)
    assert any(f["severity"] == "medium" and "large diff" in f["message"]
               for f in big), big
    assert any(f["severity"] == "high" and "single-file" in f["message"]
               for f in big), big

    # --- fetch: gh success --------------------------------------------
    fake = _sp.CompletedProcess(args=["gh"], returncode=0,
                                stdout=_SAMPLE_DIFF, stderr="")
    with mock.patch("subprocess.run", return_value=fake):
        got = fetch_pr_diff("octo/hello#3")
    assert got == _SAMPLE_DIFF.strip(), got

    # --- fetch: gh missing → API fallback ------------------------------
    api_diff = "diff --git a/y.py b/y.py\n+y\n"
    with mock.patch("subprocess.run",
                    side_effect=FileNotFoundError("gh")), \
         mock.patch.object(_urlreq, "urlopen",
                           return_value=_FakeResp(api_diff.encode())):
        got = fetch_pr_diff("octo/hello#3")
    assert got == api_diff.strip(), got

    # --- fetch: everything fails → ERROR string, never raises ----------
    with mock.patch("subprocess.run",
                    side_effect=FileNotFoundError("gh")), \
         mock.patch.object(_urlreq, "urlopen",
                           side_effect=OSError("no net")), \
         mock.patch.object(_me, "_repo_from_git", return_value=""):
        got = fetch_pr_diff("octo/hello#3")
    assert got.startswith("ERROR:"), got
    assert fetch_pr_diff("junk").startswith("ERROR:")

    # --- build_report shape -------------------------------------------
    rep = build_report("octo/hello#3", _SAMPLE_DIFF)
    assert "## Diff stat" in rep, rep
    assert "1 file changed, +4 / -1" in rep, rep
    assert "## Quick heuristic findings" in rep, rep
    assert "[CRITICAL]" in rep, rep
    assert "Deep-review brief" in rep, rep
    assert rep.startswith("PR review: octo/hello#3")
    assert build_report("x", "ERROR: nope") == "ERROR: nope"

    # --- register + tool ----------------------------------------------
    agent = _FakeAgent()
    register(agent)
    tool = agent.tools["ReviewPR"]
    assert tool.name == "ReviewPR"
    assert "pr" in tool.parameters["required"]
    with mock.patch.object(_me, "fetch_pr_diff",
                    return_value=_SAMPLE_DIFF):
        out = tool.handler(pr="octo/hello#3")
    assert "## Diff stat" in out and "Deep-review brief" in out, out
    assert tool.handler(pr="").startswith("Error:")

    # --- handle_pr_review ----------------------------------------------
    ui = _FakeUI()
    handle_pr_review(ui, "")
    assert ui.errors and "usage" in ui.errors[0].lower(), ui.errors
    with mock.patch.object(_me, "fetch_pr_diff",
                    return_value=_SAMPLE_DIFF):
        handle_pr_review(ui, "octo/hello#3")
    assert len(ui.infos) == 1, ui.infos
    assert "PR review: octo/hello#3" in ui.infos[0], ui.infos
    with mock.patch.object(_me, "fetch_pr_diff",
                    return_value="ERROR: boom"):
        handle_pr_review(ui, "3")
    assert ui.errors[-1] == "ERROR: boom", ui.errors

    print("prreview self-test PASSED")


if __name__ == "__main__":
    _selftest()

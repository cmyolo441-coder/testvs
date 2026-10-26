"""Bounded curl transport and HTTP parsing for the opt-in update watch."""
from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess

from .update_check_state import normalize_git_sha

UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS = 1.5
GITHUB_COMMITS_API_URL = "https://api.github.com/repos/rlaope/oh-my-hermes/commits/main"
GITHUB_REPOSITORY_API_URL = "https://api.github.com/repos/rlaope/oh-my-hermes"
GITHUB_COMPARE_API_URL_TEMPLATE = "https://api.github.com/repos/rlaope/oh-my-hermes/compare/{base}...{head}"
GITHUB_TAGS_API_URL = "https://api.github.com/repos/rlaope/oh-my-hermes/tags?per_page=10"
GITHUB_RELEASES_API_URL = "https://api.github.com/repos/rlaope/oh-my-hermes/releases?per_page=10"
PROBE_RESPONSE_BOUNDARY = "--omh-update-check-probe-boundary--"


@dataclass(frozen=True)
class RemoteProbeResult:
    ok: bool
    sha: str | None
    etag: str | None
    not_modified: bool
    error: str | None


@dataclass(frozen=True)
class WatchedBranchProbe:
    head: RemoteProbeResult
    head_time: str | None
    default_branch: str | None
    metadata_ok: bool
    metadata_partial: bool
    visibility_suspect: bool


@dataclass(frozen=True)
class AncestryProbeResult:
    classification: str
    http_status: int
    error: str | None


@dataclass(frozen=True)
class TagsProbeResult:
    ok: bool
    tags: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class ReleasesProbeResult:
    ok: bool
    releases: tuple[str, ...]
    error: str | None


def _run_curl(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _parse_http_response(raw: str) -> tuple[int, dict[str, str], str]:
    head, _, body = raw.replace("\r\n", "\n").partition("\n\n")
    lines, status = head.split("\n"), 0
    if lines and len(lines[0].split()) >= 2 and lines[0].split()[1].isdigit():
        status = int(lines[0].split()[1])
    headers = {key.strip().lower(): value.strip() for line in lines[1:] if ":" in line for key, _, value in [line.partition(":")]}
    return status, headers, body


def _argv(*urls: str, timeout: float, etag: str = "", boundary: bool = False) -> list[str]:
    argv = ["curl", "-sS", "-i", "--max-time", str(timeout), "-H", "Accept: application/vnd.github+json", "-H", "User-Agent: oh-my-hermes-update-check"]
    if boundary:
        argv += ["-w", f"\n{PROBE_RESPONSE_BOUNDARY}\n"]
    if etag:
        argv += ["-H", f"If-None-Match: {etag}"]
    return [*argv, *urls]


def _remote_result(completed, etag: str) -> RemoteProbeResult:
    if completed.returncode != 0:
        return RemoteProbeResult(False, None, None, False, (completed.stderr or "").strip() or f"curl exit {completed.returncode}")
    status, headers, body = _parse_http_response(completed.stdout or "")
    if status == 304:
        return RemoteProbeResult(True, None, etag, True, None)
    if status != 200:
        return RemoteProbeResult(False, None, None, False, f"http {status}")
    try:
        data = json.loads(body)
    except ValueError as exc:
        return RemoteProbeResult(False, None, None, False, str(exc))
    raw_sha = data.get("sha") if isinstance(data, dict) else None
    sha = normalize_git_sha(raw_sha)
    if sha:
        return RemoteProbeResult(True, sha, headers.get("etag") or None, False, None)
    reason = "missing sha in response" if raw_sha in (None, "") else "invalid sha in response"
    return RemoteProbeResult(False, None, None, False, reason)


def fetch_remote_main_identity(*, etag: str = "", timeout: float = UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS, runner=None) -> RemoteProbeResult:
    try:
        return _remote_result((runner or _run_curl)(_argv(GITHUB_COMMITS_API_URL, timeout=timeout, etag=etag), timeout=timeout + 0.5), etag)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return RemoteProbeResult(False, None, None, False, str(exc))


def _split_probe_responses(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.replace("\r\n", "\n").split(PROBE_RESPONSE_BOUNDARY) if chunk.strip()]


def _failed_probe(error: str) -> WatchedBranchProbe:
    return WatchedBranchProbe(RemoteProbeResult(False, None, None, False, error), None, None, False, True, False)


def fetch_watched_branch_state(*, etag: str = "", timeout: float = UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS, runner=None) -> WatchedBranchProbe:
    try:
        completed = (runner or _run_curl)(_argv(GITHUB_COMMITS_API_URL, GITHUB_REPOSITORY_API_URL, timeout=timeout, etag=etag, boundary=True), timeout=timeout + 0.5)
        stdout, returncode, stderr = completed.stdout or "", completed.returncode, (completed.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.output.decode("utf-8", "replace") if isinstance(exc.output, bytes) else exc.output or ""
        if not stdout.strip():
            return _failed_probe(str(exc))
        returncode, stderr = 0, ""
    except (OSError, ValueError) as exc:
        return _failed_probe(str(exc))
    if returncode != 0:
        return _failed_probe(stderr or f"curl exit {returncode}")
    chunks = _split_probe_responses(stdout)
    status, headers, body = _parse_http_response(chunks[0] if chunks else "")
    head_time, suspect = None, False
    if status == 304:
        head = RemoteProbeResult(True, None, etag or None, True, None)
    elif status != 200:
        head = RemoteProbeResult(False, None, None, False, f"http {status}")
    else:
        try:
            data = json.loads(body)
        except ValueError as exc:
            data, suspect, head = None, True, RemoteProbeResult(False, None, None, False, str(exc))
        if not suspect:
            raw_sha = data.get("sha") if isinstance(data, dict) else None
            sha = normalize_git_sha(raw_sha)
            if not sha:
                reason = "missing sha in response" if raw_sha in (None, "") else "invalid sha in response"
                suspect, head = True, RemoteProbeResult(False, None, None, False, reason)
            else:
                committer = data.get("commit", {}).get("committer", {}) if isinstance(data, dict) and isinstance(data.get("commit"), dict) else {}
                head_time, head = str(committer.get("date", "") or "") or None, RemoteProbeResult(True, sha, headers.get("etag") or None, False, None)
    default_branch, metadata_ok = None, False
    if len(chunks) > 1:
        metadata_status, _, metadata_body = _parse_http_response(chunks[1])
        try:
            metadata = json.loads(metadata_body) if metadata_status == 200 else None
        except ValueError:
            metadata = None
        if isinstance(metadata, dict) and metadata.get("default_branch"):
            default_branch, metadata_ok = str(metadata["default_branch"]), True
    return WatchedBranchProbe(head, head_time, default_branch, metadata_ok, not metadata_ok, suspect)


def fetch_cursor_ancestry(cursor: str, head: str, *, timeout: float = UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS, runner=None) -> AncestryProbeResult:
    cursor = normalize_git_sha(cursor)
    head = normalize_git_sha(head)
    if not cursor or not head:
        return AncestryProbeResult("unknown", 0, "compare requires full 40-character hexadecimal identities")
    try:
        completed = (runner or _run_curl)(_argv(GITHUB_COMPARE_API_URL_TEMPLATE.format(base=cursor, head=head), timeout=timeout), timeout=timeout + 0.5)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return AncestryProbeResult("unknown", 0, str(exc))
    if completed.returncode != 0:
        return AncestryProbeResult("unknown", 0, (completed.stderr or "").strip() or f"curl exit {completed.returncode}")
    status, _, body = _parse_http_response(completed.stdout or "")
    if status == 404:
        return AncestryProbeResult("cursor_unreachable", 404, None)
    if status != 200:
        return AncestryProbeResult("unknown", status, f"http {status}")
    try:
        compare_status = str(json.loads(body).get("status", ""))
    except (ValueError, AttributeError) as exc:
        return AncestryProbeResult("unknown", status, str(exc))
    classifications = {"ahead": "fast_forward", "identical": "fast_forward", "behind": "rewound", "diverged": "rewritten"}
    classification = classifications.get(compare_status)
    return AncestryProbeResult(classification, status, None) if classification else AncestryProbeResult("unknown", status, f"unrecognized compare status {compare_status!r}")


def fetch_recovery_tags(*, timeout: float = UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS, runner=None) -> TagsProbeResult:
    try:
        completed = (runner or _run_curl)(_argv(GITHUB_TAGS_API_URL, timeout=timeout), timeout=timeout + 0.5)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return TagsProbeResult(False, (), str(exc))
    if completed.returncode != 0:
        return TagsProbeResult(False, (), (completed.stderr or "").strip() or f"curl exit {completed.returncode}")
    status, _, body = _parse_http_response(completed.stdout or "")
    if status != 200:
        return TagsProbeResult(False, (), f"http {status}")
    try:
        data = json.loads(body)
    except ValueError as exc:
        return TagsProbeResult(False, (), str(exc))
    if not isinstance(data, list):
        return TagsProbeResult(False, (), "malformed tags response")
    return TagsProbeResult(True, tuple(str(item["name"]) for item in data if isinstance(item, dict) and item.get("name")), None)


def fetch_recovery_releases(*, timeout: float = UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS, runner=None) -> ReleasesProbeResult:
    try:
        completed = (runner or _run_curl)(_argv(GITHUB_RELEASES_API_URL, timeout=timeout), timeout=timeout + 0.5)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return ReleasesProbeResult(False, (), str(exc))
    if completed.returncode != 0:
        return ReleasesProbeResult(False, (), (completed.stderr or "").strip() or f"curl exit {completed.returncode}")
    status, _, body = _parse_http_response(completed.stdout or "")
    if status != 200:
        return ReleasesProbeResult(False, (), f"http {status}")
    try:
        data = json.loads(body)
    except ValueError as exc:
        return ReleasesProbeResult(False, (), str(exc))
    if not isinstance(data, list):
        return ReleasesProbeResult(False, (), "malformed releases response")
    releases = tuple(str(item["tag_name"]) for item in data if isinstance(item, dict) and item.get("tag_name"))
    return ReleasesProbeResult(True, releases, None)

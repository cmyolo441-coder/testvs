"""Deterministic chunk plans for documents longer than one `read_file` window.

Hermes' `read_file` extracts a document (PDF, DOCX, EPUB, ...) to
line-numbered text and returns at most one window per call: about 100,000
characters (`file_read_max_chars`) and 2,000 lines, with a `next_offset` for
the rest. The whole file is re-extracted on every paginated read, the
extracted text carries no page numbers, and a 300-page paper is roughly 480k
characters -- five windows that conversation compression later folds into one
summary. Nothing splits the document into ranges a reader can walk, resume,
or hand out one per subagent.

This module derives that split from numbers the caller states. It never
opens, parses, or extracts the document: when the source names a local file
the plan records the file's size and content hash so the plan can be tied to
one exact document, and nothing else is read from it. Every size that came
from the caller is labelled `caller_supplied`; the only `observed` values are
the file size and hash. Copied into `$HERMES_HOME/plugins/omh`, so it imports
only its siblings and the standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

DOCUMENT_CHUNK_PLAN_SCHEMA_VERSION = "document_chunk_plan/v1"

# Mirrors Hermes' `_DEFAULT_MAX_READ_CHARS` (`file_read_max_chars`) and the
# `read_file` `limit` maximum. A chunk is sized so one read_file call covers
# it when the extraction is as dense as assumed.
DEFAULT_BUDGET_CHARS = 100_000
HERMES_READ_LINE_LIMIT = 2_000
# Typical text-layer density of an extracted PDF page; the caller can pass the
# observed extraction length instead and the plan then derives density from it.
DEFAULT_CHARS_PER_PAGE = 1_600
# Only used to turn a character position into a read_file line offset when
# the caller has not stated the extraction's `total_lines`. Labelled
# `assumed` in the plan because no extraction was observed.
ASSUMED_CHARS_PER_LINE = 80
CHARS_PER_TOKEN = 4
# Above this many ranges the plan recommends one delegate_task child per
# range instead of walking them in the parent context.
FAN_OUT_RANGE_THRESHOLD = 4

MIN_BUDGET_CHARS = 1_000
MAX_BUDGET_CHARS = 10_000_000
MIN_CHARS_PER_PAGE = 50
MAX_CHARS_PER_PAGE = 100_000
MAX_PAGES = 100_000
MAX_CHARS = 1_000_000_000
MAX_LINES = 100_000_000
MAX_OUTLINE_ENTRIES = 500
MAX_TITLE_CHARS = 200
MAX_SOURCE_CHARS = 1_000
MAX_NOTE_CHARS = 500
MAX_RANGES = 1_000
# Hashing is a full read of the file; past this size the identity falls back
# to size plus mtime, which a rewrite still moves, and the result says so.
MAX_HASHED_BYTES = 1024 * 1024 * 1024
_HASH_BLOCK_BYTES = 1024 * 1024

CHUNK_STATES = ("covered", "next", "missing")
# A tool result is paid for in the caller's context, so it never carries the
# whole plan: ranges page through `show`, the ledger summary caps its id
# lists, and a mark returns the next range rather than every range. The
# sibling evidence tool caps its output at 20,000 characters; these limits
# keep every result of this tool under that on a thousand-range plan.
RESULT_RANGE_LIMIT_DEFAULT = 10
RESULT_RANGE_LIMIT_MAX = 16
LEDGER_ID_LIST_LIMIT = 64
# An extracted line longer than this is not a line; the caller has passed
# something other than the extraction's total_lines (one window's count,
# a page count) and the windows built from it would all point at the top.
MAX_PLAUSIBLE_CHARS_PER_LINE = 4_000
DELEGATION_HINT_STATUS = "prepared_not_observed"
DELEGATION_HINT_CLAIM_BOUNDARY = (
    "The delegation hint is a prepared recommendation with a brief template; it is not a dispatch, "
    "not evidence that any child was spawned, and not evidence that any range was read."
)
LEDGER_BINDING_WARNING = (
    "This plan's identity is not bound to the document's content ({basis}): covered marks record the "
    "caller's statements about whatever carried this label or size, and a changed document under the "
    "same identity keeps them. Plan from a local path under the hash cap to bind the ledger to a sha256."
)
PLAN_ID_HEX_CHARS = 12
_PLAN_ID = re.compile(r"\A[0-9a-f]{12}\Z")
_MAX_PLAN_FILE_BYTES = 4 * 1024 * 1024

# A hash of a credential file is still a fingerprint of a secret. These names
# are never hashed; the plan then records the label alone.
_SENSITIVE_NAME_PATTERNS = (
    re.compile(r"\A\.env(?:\..*)?\Z"),
    re.compile(r"\Aauth\.json\Z"),
    re.compile(r"\Aid_(?:rsa|dsa|ecdsa|ed25519)(?:\..*)?\Z"),
    re.compile(r"\.(?:pem|key|p12|pfx|keystore|jks)\Z"),
)
_CONTROL_CHARACTERS = {code: None for code in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0))}

CLAIM_BOUNDARY = (
    "A document chunk plan is a prepared reading schedule derived from sizes the caller stated. "
    "It is not extraction, coverage, or comprehension evidence: OMH never opened the document, "
    "every page and character count is caller_supplied, and a covered mark records that the "
    "caller said the range was read, not that OMH observed the read."
)


class DocumentPlanError(ValueError):
    """The request cannot become a plan, or the stored plan cannot be trusted."""


class DocumentPlanMissing(DocumentPlanError):
    """No plan file exists for this id -- distinct from one that exists and cannot be read."""


def strip_control_characters(value: object) -> str:
    return str(value or "").translate(_CONTROL_CHARACTERS).strip()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _optional_count(args: dict[str, Any], key: str, *, maximum: int) -> int | None:
    value = args.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise DocumentPlanError(f"{key} must be a whole number, not a boolean")
    if isinstance(value, float):
        if not value.is_integer():
            raise DocumentPlanError(f"{key} must be a whole number")
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        raise DocumentPlanError(f"{key} must be a whole number")
    if value < 1:
        raise DocumentPlanError(f"{key} must be at least 1")
    if value > maximum:
        raise DocumentPlanError(f"{key} is capped at {maximum}")
    return value


def _bounded_setting(args: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int) -> tuple[int, str]:
    value = _optional_count(args, key, maximum=maximum)
    if value is None:
        return default, "default"
    if value < minimum:
        raise DocumentPlanError(f"{key} must be at least {minimum}")
    return value, "caller_supplied"


def _normalize_outline(raw: object, pages: int | None) -> list[dict[str, Any]]:
    if raw is None or raw == "" or raw == []:
        return []
    if not isinstance(raw, list):
        raise DocumentPlanError("outline must be a list of {title, page} entries")
    if pages is None:
        raise DocumentPlanError("an outline needs pages: outline entries are page anchors")
    if len(raw) > MAX_OUTLINE_ENTRIES:
        raise DocumentPlanError(f"outline is capped at {MAX_OUTLINE_ENTRIES} entries")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DocumentPlanError(f"outline[{index}] must be an object with title and page")
        title = strip_control_characters(item.get("title", ""))
        if not title:
            raise DocumentPlanError(f"outline[{index}] needs a non-empty title")
        if len(title) > MAX_TITLE_CHARS:
            raise DocumentPlanError(f"outline[{index}] title is capped at {MAX_TITLE_CHARS} characters")
        page = _optional_count(item, "page", maximum=MAX_PAGES)
        if page is None:
            raise DocumentPlanError(f"outline[{index}] needs a page")
        if page > pages:
            raise DocumentPlanError(f"outline[{index}] page {page} is past the last page {pages}")
        entries.append({"title": title, "page": page})
    entries.sort(key=lambda entry: (entry["page"], entry["title"]))
    # Only an exact repeat is dropped: two sections that start on the same
    # page are both real and both belong in that range's `sections`.
    deduplicated: list[dict[str, Any]] = []
    for entry in entries:
        if deduplicated and deduplicated[-1] == entry:
            continue
        deduplicated.append(entry)
    return deduplicated


def normalize_plan_request(args: dict[str, Any]) -> dict[str, Any]:
    """Validate the caller's numbers into one canonical request.

    Refuses rather than guesses: a plan needs pages or an extracted length,
    an outline needs pages to anchor to, and every count is bounded.
    """
    raw_source = args.get("source", "")
    if raw_source is not None and not isinstance(raw_source, str):
        raise DocumentPlanError(f"source must be a string, not {type(raw_source).__name__}")
    source = strip_control_characters(raw_source)
    if not source:
        raise DocumentPlanError("source is required: a file path or a label naming the document")
    if len(source) > MAX_SOURCE_CHARS:
        raise DocumentPlanError(f"source is capped at {MAX_SOURCE_CHARS} characters")
    pages = _optional_count(args, "pages", maximum=MAX_PAGES)
    chars = _optional_count(args, "chars", maximum=MAX_CHARS)
    lines = _optional_count(args, "lines", maximum=MAX_LINES)
    if pages is None and chars is None:
        raise DocumentPlanError(
            "plan needs pages or chars: state the page count or the extracted text length "
            "(OMH never opens the document to measure it)"
        )
    budget_chars, budget_basis = _bounded_setting(
        args, "budget_chars", default=DEFAULT_BUDGET_CHARS, minimum=MIN_BUDGET_CHARS, maximum=MAX_BUDGET_CHARS
    )
    chars_per_page, density_basis = _bounded_setting(
        args, "chars_per_page", default=DEFAULT_CHARS_PER_PAGE, minimum=MIN_CHARS_PER_PAGE, maximum=MAX_CHARS_PER_PAGE
    )
    outline = _normalize_outline(args.get("outline"), pages)
    if lines is not None:
        total_chars = chars if chars is not None else pages * chars_per_page
        if lines > total_chars:
            raise DocumentPlanError(
                f"lines ({lines}) exceed the document's characters ({total_chars}); pass the total_lines "
                "the first read_file result reported for the whole extraction"
            )
        if total_chars / lines > MAX_PLAUSIBLE_CHARS_PER_LINE:
            raise DocumentPlanError(
                f"lines ({lines}) is implausible for {total_chars} characters (over "
                f"{MAX_PLAUSIBLE_CHARS_PER_LINE} characters per line): pass the total_lines the first "
                "read_file result reported for the whole extraction, not the lines one window returned"
            )
    provenance = {
        "pages": "caller_supplied" if pages is not None else "not_supplied",
        "chars": "caller_supplied" if chars is not None else "not_supplied",
        "lines": "caller_supplied" if lines is not None else "not_supplied",
        "outline": "caller_supplied" if outline else "not_supplied",
        "budget_chars": budget_basis,
        "chars_per_page": density_basis,
    }
    return {
        "source": source,
        "pages": pages,
        "chars": chars,
        "lines": lines,
        "outline": outline,
        "budget_chars": budget_chars,
        "chars_per_page": chars_per_page,
        "provenance": provenance,
    }


def _sensitive_name(name: str) -> bool:
    lowered = name.casefold()
    return any(pattern.search(lowered) for pattern in _SENSITIVE_NAME_PATTERNS)


def fingerprint_source(label: str, path: Path | None) -> dict[str, Any]:
    """Record what identifies the document without reading it as a document.

    A regular file yields its size and sha256 (observed); anything else --
    a label, a missing path, a directory, a credential-shaped name, a file
    past the hash cap -- yields the label alone and says why.
    """
    record: dict[str, Any] = {
        "label": label,
        "kind": "label",
        "size_bytes": None,
        "mtime_ns": None,
        "sha256": "",
        "fingerprint_basis": "label",
        # What the plan id (and so the ledger) is bound to: the content hash,
        # the file's size and mtime, or only the label the caller typed.
        "identity_basis": "label",
    }
    if path is None:
        return record
    try:
        if not path.is_file():
            return record
        stat = path.stat()
    except OSError:
        return record
    record["kind"] = "file"
    record["path"] = str(path)
    record["size_bytes"] = stat.st_size
    record["mtime_ns"] = stat.st_mtime_ns
    record["identity_basis"] = "size_and_mtime"
    if _sensitive_name(path.name):
        record["fingerprint_basis"] = "refused_sensitive_name"
        return record
    if stat.st_size > MAX_HASHED_BYTES:
        record["fingerprint_basis"] = "size_only_over_hash_cap"
        return record
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(_HASH_BLOCK_BYTES):
                digest.update(block)
    except OSError:
        record["fingerprint_basis"] = "size_only_unreadable"
        return record
    record["sha256"] = digest.hexdigest()
    record["fingerprint_basis"] = "sha256"
    record["identity_basis"] = "sha256"
    return record


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_id_for(request: dict[str, Any], fingerprint: dict[str, Any]) -> str:
    """Short, stable id: same document and same parameters give the same plan.

    The document half of the identity is its sha256 when one was taken. A
    file that could not be hashed binds to its size and mtime instead, which
    a rewrite still moves; only a bare label binds to nothing but itself,
    and the result says so.
    """
    identity = {
        "schema_version": DOCUMENT_CHUNK_PLAN_SCHEMA_VERSION,
        "fingerprint": fingerprint["sha256"] or fingerprint["label"],
        "size_bytes": fingerprint["size_bytes"],
        "mtime_ns": None if fingerprint["sha256"] else fingerprint.get("mtime_ns"),
        "pages": request["pages"],
        "chars": request["chars"],
        "lines": request["lines"],
        "outline": request["outline"],
        "budget_chars": request["budget_chars"],
        "chars_per_page": request["chars_per_page"],
    }
    return _canonical_digest(identity)[:PLAN_ID_HEX_CHARS]


def _segments_from_outline(pages: int, outline: list[dict[str, Any]]) -> list[tuple[int, int, tuple[str, ...]]]:
    """Section spans in page order; pages before the first entry are front matter.

    Sections that start on the same page share one span and all of their
    titles travel with it.
    """
    if not outline:
        return [(1, pages, ())]
    starts: list[int] = []
    titles_by_page: dict[int, list[str]] = {}
    for entry in outline:
        if entry["page"] not in titles_by_page:
            starts.append(entry["page"])
        titles_by_page.setdefault(entry["page"], []).append(entry["title"])
    segments: list[tuple[int, int, tuple[str, ...]]] = []
    if starts[0] > 1:
        segments.append((1, starts[0] - 1, ()))
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else pages
        segments.append((start, max(start, end), tuple(titles_by_page[start])))
    return segments


def _page_ranges(pages: int, pages_per_chunk: int, outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack outline sections into chunks of at most `pages_per_chunk` pages.

    A section longer than one chunk is split at chunk size; a chunk never
    straddles a section boundary unless the sections are small enough to
    share it, so a range's `sections` list is the whole of what it covers.
    """
    pieces: list[tuple[int, int, tuple[str, ...]]] = []
    for start, end, titles in _segments_from_outline(pages, outline):
        cursor = start
        while cursor <= end:
            piece_end = min(end, cursor + pages_per_chunk - 1)
            pieces.append((cursor, piece_end, titles))
            cursor = piece_end + 1
    ranges: list[dict[str, Any]] = []
    current: list[tuple[int, int, tuple[str, ...]]] = []
    current_pages = 0
    for piece in pieces:
        piece_pages = piece[1] - piece[0] + 1
        if current and current_pages + piece_pages > pages_per_chunk:
            ranges.append(_page_range(current))
            current, current_pages = [], 0
        current.append(piece)
        current_pages += piece_pages
    if current:
        ranges.append(_page_range(current))
    return ranges


def _page_range(pieces: list[tuple[int, int, tuple[str, ...]]]) -> dict[str, Any]:
    sections: list[str] = []
    for _start, _end, titles in pieces:
        for title in titles:
            if title not in sections:
                sections.append(title)
    return {"start": pieces[0][0], "end": pieces[-1][1], "sections": sections}


def _read_window(char_start: int, char_end: int, total_chars: int, total_lines: int, basis: str) -> dict[str, Any]:
    offset = 1 + (char_start * total_lines) // total_chars
    end_line = max(offset, math.ceil(char_end * total_lines / total_chars))
    end_line = min(end_line, total_lines)
    offset = min(offset, end_line)
    span = end_line - offset + 1
    return {
        "offset": offset,
        "end_line": end_line,
        "limit": min(HERMES_READ_LINE_LIMIT, span),
        "read_calls_estimated": max(1, math.ceil(span / HERMES_READ_LINE_LIMIT)),
        "basis": basis,
    }


def derive_ranges(request: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Numbered ranges plus the assumptions that shaped them.

    Pages are the unit when the caller stated them (people cite pages);
    otherwise the extracted length is split at the budget. Either way each
    range carries an estimated character span, a token estimate, and the
    `read_file` window that reaches it.
    """
    pages, chars, lines = request["pages"], request["chars"], request["lines"]
    budget = request["budget_chars"]
    if pages is not None:
        if chars is not None:
            density = chars / pages
            density_basis = "derived_from_caller_chars_and_pages"
        else:
            density = float(request["chars_per_page"])
            density_basis = request["provenance"]["chars_per_page"]
        total_chars = chars if chars is not None else pages * request["chars_per_page"]
        pages_per_chunk = max(1, int(budget // density))
        page_ranges = _page_ranges(pages, pages_per_chunk, request["outline"])
        if len(page_ranges) > MAX_RANGES:
            raise DocumentPlanError(
                f"the plan would need {len(page_ranges)} ranges, over the cap of {MAX_RANGES}; raise budget_chars"
            )
        spans = [
            (
                round((entry["start"] - 1) * density),
                min(total_chars, round(entry["end"] * density)),
                {"start": entry["start"], "end": entry["end"]},
                entry["sections"],
            )
            for entry in page_ranges
        ]
        unit = "pages"
    else:
        total_chars = chars
        density = None
        density_basis = "not_applicable"
        count = math.ceil(chars / budget)
        if count > MAX_RANGES:
            raise DocumentPlanError(
                f"the plan would need {count} ranges, over the cap of {MAX_RANGES}; raise budget_chars"
            )
        spans = [
            (index * budget, min(chars, (index + 1) * budget), None, [])
            for index in range(count)
        ]
        unit = "chars"
    if lines is not None:
        total_lines, window_basis = lines, "caller_lines"
    else:
        total_lines = max(1, math.ceil(total_chars / ASSUMED_CHARS_PER_LINE))
        window_basis = "estimated_lines"
    ranges: list[dict[str, Any]] = []
    for index, (char_start, char_end, page_span, sections) in enumerate(spans, start=1):
        estimated_chars = max(0, char_end - char_start)
        spec = {
            "chunk": index,
            "pages": page_span,
            "chars": {"start": char_start, "end": char_end},
            "sections": sections,
        }
        window = _read_window(char_start, char_end, max(1, total_chars), total_lines, window_basis)
        window["read_calls_estimated"] = max(
            window["read_calls_estimated"], max(1, math.ceil(estimated_chars / budget))
        )
        ranges.append(
            {
                **spec,
                "estimated_chars": estimated_chars,
                "estimated_tokens": math.ceil(estimated_chars / CHARS_PER_TOKEN),
                "read_window": window,
                "digest": _canonical_digest(spec)[:PLAN_ID_HEX_CHARS],
            }
        )
    if lines is not None and len({entry["read_window"]["offset"] for entry in ranges}) < len(ranges):
        raise DocumentPlanError(
            f"lines ({lines}) cannot address {len(ranges)} distinct ranges: pass the total_lines the first "
            "read_file result reported for the whole extraction, not the lines one window returned, "
            "or raise budget_chars"
        )
    assumptions = {
        "unit": unit,
        "chars_per_page": {"value": density, "basis": density_basis},
        "chars_per_line": {
            "value": None if lines is not None else ASSUMED_CHARS_PER_LINE,
            "basis": "not_applicable" if lines is not None else "assumed",
        },
        "total_chars": {
            "value": total_chars,
            "basis": "caller_supplied" if chars is not None else "estimated_from_pages",
        },
        "total_lines": {"value": total_lines, "basis": window_basis},
        "chars_per_token": CHARS_PER_TOKEN,
        "hermes_read_line_limit": HERMES_READ_LINE_LIMIT,
        "budget_chars": budget,
    }
    return ranges, assumptions


def initial_ledger(range_count: int) -> dict[str, dict[str, str]]:
    return {
        str(index): {"state": "next" if index == 1 else "missing", "note": "", "marked_at": ""}
        for index in range(1, range_count + 1)
    }


def ledger_summary(ledger: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Counts per state plus the first ids of each, bounded for the caller's context."""
    ids: dict[str, list[int]] = {state: [] for state in CHUNK_STATES}
    for key, entry in sorted(ledger.items(), key=lambda item: int(item[0])):
        ids.setdefault(entry.get("state", "missing"), []).append(int(key))
    summary: dict[str, Any] = {
        "counts": {state: len(ids[state]) for state in CHUNK_STATES},
        "ids_truncated": any(len(ids[state]) > LEDGER_ID_LIST_LIMIT for state in CHUNK_STATES),
        "id_list_limit": LEDGER_ID_LIST_LIMIT,
    }
    for state in CHUNK_STATES:
        summary[state] = ids[state][:LEDGER_ID_LIST_LIMIT]
    return summary


def next_chunk(ledger: dict[str, dict[str, str]]) -> int | None:
    for key, entry in sorted(ledger.items(), key=lambda item: int(item[0])):
        if entry.get("state") == "next":
            return int(key)
    return None


def _brief_template() -> str:
    return (
        "Read {source} range {chunk}/{range_count} ({span}) with "
        "read_file(path, offset={offset}, limit={limit}); if the result is truncated, "
        "continue from its next_offset until line {end_line}. Cover that range only: "
        "explain or extract every claim, equation, figure, table, and limitation in it, "
        "quoting page or line anchors. Finish by reporting covered / next / missing for "
        "plan {plan_id} chunk {chunk}, and never report a range you did not read."
    )


def _span_text(entry: dict[str, Any]) -> str:
    pages = entry.get("pages")
    if pages:
        text = f"pages {pages['start']}-{pages['end']}"
    else:
        chars = entry["chars"]
        text = f"chars {chars['start']}-{chars['end']}"
    sections = entry.get("sections") or []
    if sections:
        text += ": " + "; ".join(sections)
    return text


def fill_brief(template: str, plan: dict[str, Any], entry: dict[str, Any]) -> str:
    window = entry["read_window"]
    return template.format(
        source=plan["source"]["label"],
        chunk=entry["chunk"],
        range_count=plan["range_count"],
        span=_span_text(entry),
        offset=window["offset"],
        limit=window["limit"],
        end_line=window["end_line"],
        plan_id=plan["plan_id"],
    )


def delegation_hint(plan: dict[str, Any]) -> dict[str, Any]:
    count = plan["range_count"]
    fan_out = count > FAN_OUT_RANGE_THRESHOLD
    template = _brief_template()
    hint: dict[str, Any] = {
        "mode": "fan_out" if fan_out else "sequential",
        "fan_out_threshold": FAN_OUT_RANGE_THRESHOLD,
        "reason": (
            f"{count} ranges exceed the fan-out threshold; one delegate_task child per range "
            "keeps each child's context to one read window and the parent's to the reports"
            if fan_out
            else f"{count} range(s) fit the parent context; read them in order and mark each"
        ),
        "tool": "delegate_task" if fan_out else "read_file",
        "per_range_brief_template": template,
        "after_each_range": (
            f"omh_document_plan action=mark plan_id={plan['plan_id']} chunk=<n> state=covered "
            "(the next chunk advances to state=next)"
        ),
    }
    hint["status"] = DELEGATION_HINT_STATUS
    hint["claim_boundary"] = DELEGATION_HINT_CLAIM_BOUNDARY
    return hint


def range_view(plan: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """One range as a result carries its filled brief; the file carries the template once."""
    template = str(plan.get("delegation_hint", {}).get("per_range_brief_template") or _brief_template())
    view = dict(entry)
    view["state"] = str(plan["ledger"].get(str(entry["chunk"]), {}).get("state", "missing"))
    view["brief"] = fill_brief(template, plan, entry)
    return view


def result_view(plan: dict[str, Any], *, start: int = 1, limit: int = RESULT_RANGE_LIMIT_DEFAULT) -> dict[str, Any]:
    """The bounded projection every tool result is built from.

    Metadata, the ledger summary, the next range, and one page of ranges
    starting at chunk `start`; `truncated` says whether ranges remain past
    the page so the caller pages through `show` instead of receiving the
    whole plan in one result.
    """
    ranges = plan["ranges"]
    count = len(ranges)
    start = min(max(1, start), max(1, count))
    limit = min(max(1, limit), RESULT_RANGE_LIMIT_MAX)
    page = [range_view(plan, entry) for entry in ranges[start - 1 : start - 1 + limit]]
    end = start + len(page) - 1
    pending = next_chunk(plan["ledger"])
    view: dict[str, Any] = {
        "plan_id": plan["plan_id"],
        "plan_schema_version": plan["schema_version"],
        "created_at": plan["created_at"],
        "updated_at": plan["updated_at"],
        "source": plan["source"],
        "inputs": plan["inputs"],
        "provenance": plan["provenance"],
        "assumptions": plan["assumptions"],
        "unit": plan["unit"],
        "range_count": count,
        "ranges": page,
        "ranges_from": start,
        "ranges_to": end,
        "ranges_returned": len(page),
        "truncated": end < count,
        "next_range": range_view(plan, ranges[pending - 1]) if pending else None,
        "ledger_summary": ledger_summary(plan["ledger"]),
        "delegation_hint": plan["delegation_hint"],
    }
    if end < count:
        view["page_hint"] = (
            f"ranges {end + 1}-{count} not returned; call action=show with from={end + 1} "
            f"(limit up to {RESULT_RANGE_LIMIT_MAX}) to page through them"
        )
    basis = str(plan["source"].get("identity_basis") or "label")
    if basis != "sha256":
        view["ledger_binding_warning"] = LEDGER_BINDING_WARNING.format(basis=basis)
    return view


def build_document_plan(
    request: dict[str, Any],
    fingerprint: dict[str, Any],
    *,
    now: str | None = None,
) -> dict[str, Any]:
    ranges, assumptions = derive_ranges(request)
    stamp = now or _utc_now()
    provenance = dict(request["provenance"])
    provenance["size_bytes"] = "observed" if fingerprint["size_bytes"] is not None else "not_observed"
    provenance["sha256"] = "observed" if fingerprint["sha256"] else "not_observed"
    plan: dict[str, Any] = {
        "schema_version": DOCUMENT_CHUNK_PLAN_SCHEMA_VERSION,
        "plan_id": plan_id_for(request, fingerprint),
        "created_at": stamp,
        "updated_at": stamp,
        "source": fingerprint,
        "inputs": {
            "pages": request["pages"],
            "chars": request["chars"],
            "lines": request["lines"],
            "outline": request["outline"],
            "budget_chars": request["budget_chars"],
            "chars_per_page": request["chars_per_page"],
        },
        "provenance": provenance,
        "assumptions": assumptions,
        "unit": assumptions["unit"],
        "range_count": len(ranges),
        "ranges": ranges,
        "ledger": initial_ledger(len(ranges)),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    plan["delegation_hint"] = delegation_hint(plan)
    return plan


def documents_root(omh_home: Path) -> Path:
    return omh_home / "documents"


def plan_path(omh_home: Path, plan_id: str) -> Path:
    if not _PLAN_ID.match(plan_id):
        raise DocumentPlanError("plan_id must be the 12-hex id a plan action returned")
    return documents_root(omh_home) / plan_id / "plan.json"


def _reject_symlink_ancestry(path: Path, *, root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise DocumentPlanError("refusing a symlinked document plan path")
        if current == root or current == current.parent:
            return
        current = current.parent


def write_document_plan(omh_home: Path, plan: dict[str, Any]) -> Path:
    destination = plan_path(omh_home, str(plan["plan_id"]))
    _reject_symlink_ancestry(destination, root=omh_home)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    cleanup_error: OSError | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestry(destination, root=omh_home)
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, destination)
    except OSError as error:
        raise DocumentPlanError(f"document plan destination is not writable: {error.__class__.__name__}") from error
    finally:
        # A leftover temp file is the write's own failure; the cleanup's
        # failure is reported the same way, never as a raw OSError.
        if temporary.exists() and not temporary.is_symlink():
            try:
                temporary.unlink()
            except OSError as error:
                cleanup_error = error
    if cleanup_error is not None:
        raise DocumentPlanError(
            f"document plan temporary file could not be removed: {cleanup_error.__class__.__name__}"
        ) from cleanup_error
    return destination


def read_document_plan(omh_home: Path, plan_id: str) -> dict[str, Any]:
    """Re-read one plan; a missing, oversized, or foreign file is a refusal."""
    path = plan_path(omh_home, plan_id)
    _reject_symlink_ancestry(path, root=omh_home)
    try:
        with path.open(encoding="utf-8") as handle:
            raw = handle.read(_MAX_PLAN_FILE_BYTES + 1)
    except FileNotFoundError:
        raise DocumentPlanMissing(f"no document plan {plan_id} in this OMH home") from None
    except OSError as error:
        raise DocumentPlanError(f"document plan {plan_id} is unreadable: {error.__class__.__name__}") from error
    if len(raw) > _MAX_PLAN_FILE_BYTES:
        raise DocumentPlanError(f"document plan {plan_id} exceeds the plan file size cap")
    try:
        data = json.loads(raw)
    except ValueError:
        raise DocumentPlanError(f"document plan {plan_id} is not valid JSON") from None
    return validate_document_plan(data, expected_plan_id=plan_id)


def validate_document_plan(data: object, *, expected_plan_id: str | None = None) -> dict[str, Any]:
    """The shape every consumer may rely on, from the tool or from a plan file."""
    if not isinstance(data, dict):
        raise DocumentPlanError("a document plan is a JSON object")
    if data.get("schema_version") != DOCUMENT_CHUNK_PLAN_SCHEMA_VERSION:
        raise DocumentPlanError(f"expected schema_version {DOCUMENT_CHUNK_PLAN_SCHEMA_VERSION}")
    plan_id = data.get("plan_id")
    if not isinstance(plan_id, str) or not _PLAN_ID.match(plan_id):
        raise DocumentPlanError("document plan carries no valid plan_id")
    if expected_plan_id is not None and plan_id != expected_plan_id:
        raise DocumentPlanError("document plan file names a different plan_id than its directory")
    ranges = data.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise DocumentPlanError("document plan has no ranges")
    if data.get("range_count") != len(ranges):
        raise DocumentPlanError("document plan range_count disagrees with its ranges")
    for index, entry in enumerate(ranges, start=1):
        if not isinstance(entry, dict) or entry.get("chunk") != index:
            raise DocumentPlanError(f"document plan range {index} is malformed or out of order")
        window = entry.get("read_window")
        if not isinstance(window, dict) or not isinstance(window.get("offset"), int) or not isinstance(window.get("limit"), int):
            raise DocumentPlanError(f"document plan range {index} has no read window")
        for key in ("estimated_chars", "estimated_tokens"):
            if not isinstance(entry.get(key), int) or isinstance(entry.get(key), bool) or entry[key] < 0:
                raise DocumentPlanError(f"document plan range {index} has no {key}")
        if not isinstance(entry.get("digest"), str) or not entry["digest"]:
            raise DocumentPlanError(f"document plan range {index} has no digest")
    ledger = data.get("ledger")
    if not isinstance(ledger, dict) or set(ledger) != {str(index) for index in range(1, len(ranges) + 1)}:
        raise DocumentPlanError("document plan ledger does not cover its ranges exactly")
    for key, entry in ledger.items():
        if not isinstance(entry, dict) or entry.get("state") not in CHUNK_STATES:
            raise DocumentPlanError(f"document plan ledger entry {key} has no valid state")
    source = data.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("label"), str) or not source["label"]:
        raise DocumentPlanError("document plan names no source")
    if not isinstance(data.get("delegation_hint"), dict):
        raise DocumentPlanError("document plan carries no delegation_hint")
    return data


def mark_document_plan_chunk(
    omh_home: Path,
    plan_id: str,
    chunk: object,
    state: object,
    note: object = "",
    *,
    now: str | None = None,
) -> dict[str, Any]:
    """Set one chunk's state and keep exactly one chunk `next` while work remains.

    Marking a chunk covered advances `next` to the lowest missing chunk when
    none is next; marking a chunk next demotes any other next chunk back to
    missing; and while any chunk is missing exactly one is next, so a chunk
    marked missing when nothing else is next is promoted straight back to
    next -- it is the next thing to read. The mark records the caller's
    statement, never an observation.
    """
    plan = read_document_plan(omh_home, plan_id)
    state_text = strip_control_characters(state).casefold()
    if state_text not in CHUNK_STATES:
        raise DocumentPlanError("state must be one of " + ", ".join(CHUNK_STATES))
    index = _optional_count({"chunk": chunk}, "chunk", maximum=MAX_RANGES)
    if index is None:
        raise DocumentPlanError("mark needs the chunk number to update")
    if index > plan["range_count"]:
        raise DocumentPlanError(f"chunk {index} is past the last range {plan['range_count']}")
    note_text = strip_control_characters(note)
    if len(note_text) > MAX_NOTE_CHARS:
        raise DocumentPlanError(f"note is capped at {MAX_NOTE_CHARS} characters")
    stamp = now or _utc_now()
    ledger = plan["ledger"]
    key = str(index)
    if state_text == "next":
        for other_key, entry in ledger.items():
            if other_key != key and entry.get("state") == "next":
                entry["state"] = "missing"
    ledger[key] = {"state": state_text, "note": note_text, "marked_at": stamp}
    if not any(entry.get("state") == "next" for entry in ledger.values()):
        for other_key in sorted(ledger, key=int):
            if ledger[other_key].get("state") == "missing":
                ledger[other_key]["state"] = "next"
                break
    plan["updated_at"] = stamp
    write_document_plan(omh_home, plan)
    return plan

from __future__ import annotations

import hashlib
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..local_store import (
    append_jsonl_locked,
    atomic_write_json,
    ensure_dir,
    file_lock,
    is_directory_link,
    read_json_object_result,
    read_jsonl_objects,
    utc_now,
)
from ..paths import OmhPaths
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


PAPER_LEARNING_CARD_SCHEMA_VERSION = "paper_learning_card/v1"
PAPER_LEARNING_RECORD_SCHEMA_VERSION = "omh_paper_learning_record/v1"
PAPER_LEARNING_PROGRESS_SCHEMA_VERSION = "omh_paper_learning_progress/v1"
PAPER_LEARNING_INDEX_SCHEMA_VERSION = "omh_paper_learning_index/v1"
PAPER_LEARNING_READING_STATUSES = ("not_started", "in_progress", "waiting_for_missing_sections", "complete")
PAPER_SOURCE_KINDS = ("none", "file", "url", "reference")
PAPER_LEARNING_NOTE_CHAR_LIMIT = 500
PAPER_LEARNING_CARD_FILENAME = "card.json"
PAPER_LEARNING_LEDGER_FILENAME = "ledger.jsonl"
PAPER_LEARNING_STORE_REASON_CODES = ("record_not_found", "record_corrupt", "source_unreadable")
# A --source file is hashed for identity only. Past this size the hash is
# skipped and recorded as skipped, so a multi-gigabyte scan cannot block `plan`
# silently; the size and path are still recorded.
PAPER_SOURCE_HASH_BYTE_BUDGET = 256 * 1024 * 1024


class PaperLearningStoreError(ValueError):
    """A store read that cannot yield a usable record, with the reason named.

    `record_not_found`: no record exists for the id (or the id is not a valid
    store id). `record_corrupt`: a file exists but is unparsable or fails
    `validate_paper_learning_record`. The two are kept apart because the
    repair differs: the first is a typo or a deleted record, the second is
    something `omh paper validate` can name file by file.
    """

    def __init__(self, reason_code: str, detail: str, *, paper_id: str = "") -> None:
        if reason_code not in PAPER_LEARNING_STORE_REASON_CODES:
            raise ValueError(f"unsupported paper learning store reason_code: {reason_code}")
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail
        self.paper_id = paper_id
PAPER_LEARNING_LEVELS = ("very_easy", "moderate", "expert", "choose")
PAPER_LEARNING_SOURCE_STATES = (
    "metadata_only",
    "excerpt_text_observed",
    "file_text_extraction_observed",
    "full_text_observed",
    "unknown_or_missing",
)
PAPER_LEARNING_COVERAGE_POLICY = "coverage_preserving_not_lossy_summary"
DEFAULT_PAPER_SECTIONS = (
    "Abstract",
    "Introduction",
    "Related work / prior context",
    "Method",
    "Data or experimental setup",
    "Results",
    "Figures, tables, and equations",
    "Limitations",
    "Implications",
    "Reproducibility notes",
)
PAPER_LEARNING_NOT_OBSERVED = (
    "full_pdf_extraction",
    "figure_ocr",
    "external_citation_check",
    "math_proof_validation",
    "code_or_benchmark_reproduction",
    "peer_review_or_claim_correctness",
)
PAPER_LEARNING_ACTIONS = (
    "choose_explanation_level",
    "show_paper_source_requirements",
    "record_paper_metadata",
    "record_paper_excerpt_observed",
    "record_file_text_extraction_observed",
    "show_paper_learning",
    "continue_next_section",
    "revise_explanation_level",
    "show_coverage_ledger",
    "record_user_review",
    "show_status",
)
PAPER_LEARNING_LEVEL_CONTRACT = {
    "very_easy": {
        "label": "Very easy",
        "style": "plain language, prerequisite concepts inline, glossary, analogies, and slow build-up",
        "coverage_rule": "Do not drop sections, claims, equations, figures, or limitations; simplify scaffolding only.",
    },
    "moderate": {
        "label": "Moderate",
        "style": "practitioner or graduate-friendly language with technical terms retained and explained",
        "coverage_rule": "Preserve all substantive content while balancing intuition, mechanism, and implications.",
    },
    "expert": {
        "label": "Expert",
        "style": "technical terminology, assumptions, equations, method critique, result critique, and reproducibility notes",
        "coverage_rule": "Preserve section coverage and distinguish paper claims from independently verified truth.",
    },
    "choose": {
        "label": "Choose level",
        "style": "ask the user to choose very_easy, moderate, or expert before drafting the explanation",
        "coverage_rule": "Prepare the scope and source boundary; do not guess the user's desired explanation level.",
    },
}


def normalize_paper_learning_level(level: str | None) -> str:
    """Normalize user-facing level aliases without changing explanation content."""
    return paper_learning_level_or_none(level) or "choose"


def paper_learning_level_or_none(level: str | None) -> str | None:
    """The contract level for a user-facing alias, or None when the alias is unknown.

    `normalize_paper_learning_level` folds unknown input to `choose`, which is
    right for a card built from free text. A command flag needs the other
    answer too: `--level expret` must be refused, not recorded as `choose`.
    """
    if not level:
        return "choose"
    normalized = level.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "easy": "very_easy",
        "very_easy": "very_easy",
        "beginner": "very_easy",
        "plain": "very_easy",
        "쉽게": "very_easy",
        "아주_쉽게": "very_easy",
        "moderate": "moderate",
        "medium": "moderate",
        "normal": "moderate",
        "intermediate": "moderate",
        "적당한": "moderate",
        "expert": "expert",
        "advanced": "expert",
        "technical": "expert",
        "전문가급": "expert",
        "choose": "choose",
        "ask": "choose",
    }
    return aliases.get(normalized)


def normalize_paper_source_state(
    state: str | None,
    *,
    observed_sections: Iterable[str] = (),
    missing_sections: Iterable[str] = (),
    evidence_ref: str = "",
) -> dict[str, object]:
    normalized = (state or "unknown_or_missing").strip().lower()
    if normalized not in PAPER_LEARNING_SOURCE_STATES:
        normalized = "unknown_or_missing"
    observed = _clean_unique(observed_sections)
    missing = _clean_unique(missing_sections)
    if normalized == "full_text_observed" and missing:
        normalized = "file_text_extraction_observed"
    if normalized in {"metadata_only", "unknown_or_missing"} and observed:
        normalized = "excerpt_text_observed"
    return {
        "state": normalized,
        "observed_sections": observed,
        "missing_sections": missing,
        "evidence_ref": evidence_ref.strip(),
    }


def build_coverage_ledger(
    *,
    observed_sections: Iterable[str] = (),
    missing_sections: Iterable[str] = (),
    sections: Iterable[str] = DEFAULT_PAPER_SECTIONS,
) -> list[dict[str, str]]:
    observed_keys = {_key(section) for section in observed_sections}
    missing_keys = {_key(section) for section in missing_sections}
    ledger: list[dict[str, str]] = []
    for section in _clean_unique(sections) or list(DEFAULT_PAPER_SECTIONS):
        key = _key(section)
        if key in observed_keys:
            status = "observed"
            explanation_status = "pending"
        elif key in missing_keys:
            status = "missing"
            explanation_status = "pending"
        else:
            status = "prepared"
            explanation_status = "pending"
        ledger.append(
            {
                "paper_section": section,
                "status": status,
                "explanation_status": explanation_status,
            }
        )
    return ledger


def build_paper_learning_card(
    *,
    title: str = "unknown or supplied",
    authors: Iterable[str] = (),
    source_ref: str = "",
    level: str | None = "choose",
    source_state: str | None = "unknown_or_missing",
    observed_sections: Iterable[str] = (),
    missing_sections: Iterable[str] = (),
    evidence_ref: str = "",
    sections: Iterable[str] = DEFAULT_PAPER_SECTIONS,
    output_language: str = "source",
) -> dict[str, object]:
    observed = _clean_unique(observed_sections)
    missing = _clean_unique(missing_sections)
    level_id = normalize_paper_learning_level(level)
    card = {
        "schema_version": PAPER_LEARNING_CARD_SCHEMA_VERSION,
        "card_id": _card_id(title, source_ref, level_id),
        "paper_identity": {
            "title": title.strip() or "unknown or supplied",
            "authors": _clean_unique(authors),
            "source_ref": source_ref.strip(),
        },
        "source_state": normalize_paper_source_state(
            source_state,
            observed_sections=observed,
            missing_sections=missing,
            evidence_ref=evidence_ref,
        ),
        "level": level_id,
        "level_contract": PAPER_LEARNING_LEVEL_CONTRACT[level_id],
        "output_language": output_language.strip() or "source",
        "coverage_policy": PAPER_LEARNING_COVERAGE_POLICY,
        "coverage_ledger": build_coverage_ledger(
            observed_sections=observed,
            missing_sections=missing,
            sections=sections,
        ),
        "chunking_policy": {
            "mode": "section_by_section",
            "part_index": 1,
            "part_count_when_known": None,
            "chunk_stop_rule": "End each chunk with covered / next / missing; say done only when the ledger is complete.",
        },
        "explanation_outline": (
            "What problem the paper studies",
            "What the paper claims",
            "Prior work / gap",
            "Method",
            "Data or experimental setup",
            "Results",
            "Figures, tables, and equations",
            "Limitations",
            "Implications",
            "Reproducibility notes",
            "Glossary / concept ladder",
        ),
        "not_observed": list(PAPER_LEARNING_NOT_OBSERVED),
        "available_actions": list(PAPER_LEARNING_ACTIONS),
        "next_actions": list(PAPER_LEARNING_ACTIONS[:8]),
    }
    return card


def validate_paper_learning_card(card: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if card.get("schema_version") != PAPER_LEARNING_CARD_SCHEMA_VERSION:
        errors.append("schema_version must be paper_learning_card/v1")
    if card.get("level") not in PAPER_LEARNING_LEVELS:
        errors.append("level must be one of very_easy, moderate, expert, choose")
    source_state = card.get("source_state")
    if not isinstance(source_state, dict):
        errors.append("source_state must be an object")
    elif source_state.get("state") not in PAPER_LEARNING_SOURCE_STATES:
        errors.append("source_state.state is unsupported")
    if card.get("coverage_policy") != PAPER_LEARNING_COVERAGE_POLICY:
        errors.append("coverage_policy must preserve coverage")
    ledger = card.get("coverage_ledger")
    if not isinstance(ledger, list) or not ledger:
        errors.append("coverage_ledger must be a non-empty list")
    if not set(PAPER_LEARNING_NOT_OBSERVED).issubset(set(card.get("not_observed", []))):
        errors.append("not_observed must include paper extraction and validation boundaries")
    return errors


def _clean_unique(values: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = _key(text)
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def _key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def _card_id(title: str, source_ref: str, level: str) -> str:
    seed = "|".join((title.strip().lower(), source_ref.strip().lower(), level))
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"paper-learning-{digest}"


# --- Durable store -----------------------------------------------------------
#
# A paper_learning_card/v1 on its own lives one chat turn. The store below is
# what lets reading progress over a long paper survive a Hermes session: one
# directory per paper under `$OMH_HOME/paper-learning/<paper_id>/` holding the
# card (rewritten atomically as the coverage ledger moves) and an append-only
# progress ledger, one JSON line per recorded chunk. Everything here is
# deterministic metadata; the source file is at most hashed, never parsed.

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,159}$")
_URL_PREFIXES = ("http://", "https://")


def new_paper_id(title: str, now: datetime | str | None = None) -> str:
    return f"{_stamp(now)}-{_slugify(title)}-{secrets.token_hex(3)}"


def describe_paper_source(source_ref: str) -> dict[str, Any]:
    """Classify the supplied source without reading it beyond a content hash.

    A path that names an existing regular file is recorded with its absolute
    path, size, and SHA-256 so a resumed session can tell whether it is
    looking at the same bytes. The hash is identity evidence only: it proves
    nothing about text extraction, page count, or what the paper says.
    """
    ref = str(source_ref).strip()
    if not ref:
        return {"ref": "", "kind": "none", "path": "", "exists": False, "sha256": "", "size_bytes": None}
    if ref.lower().startswith(_URL_PREFIXES):
        return {"ref": ref, "kind": "url", "path": "", "exists": False, "sha256": "", "size_bytes": None}
    candidate = Path(ref).expanduser()
    if not candidate.is_file():
        return {"ref": ref, "kind": "reference", "path": "", "exists": False, "sha256": "", "size_bytes": None}
    try:
        size = candidate.stat().st_size
        resolved = str(candidate.resolve())
        if size > PAPER_SOURCE_HASH_BYTE_BUDGET:
            return _file_source(ref, resolved, size, sha256="", hash_skipped="over_budget")
        digest = hashlib.sha256()
        seen = 0
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                seen += len(chunk)
    except OSError as exc:
        raise PaperLearningStoreError("source_unreadable", f"{ref}: {exc}") from exc
    if seen != size:
        raise PaperLearningStoreError("source_unreadable", f"{ref}: file changed while it was being hashed ({seen} bytes read, {size} expected)")
    return _file_source(ref, resolved, size, sha256=digest.hexdigest(), hash_skipped="")


def _file_source(ref: str, path: str, size: int, *, sha256: str, hash_skipped: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "kind": "file",
        "path": path,
        "exists": True,
        "sha256": sha256,
        "size_bytes": size,
        "hash_skipped": hash_skipped,
        "hash_budget_bytes": PAPER_SOURCE_HASH_BYTE_BUDGET,
    }


def build_paper_learning_record(
    *,
    title: str,
    source_ref: str = "",
    authors: Iterable[str] = (),
    level: str | None = "choose",
    source_state: str | None = "metadata_only",
    sections: Iterable[str] | None = None,
    observed_sections: Iterable[str] = (),
    missing_sections: Iterable[str] = (),
    evidence_ref: str = "",
    output_language: str = "source",
    paper_id: str | None = None,
    created_at: str | None = None,
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    clean_title = str(title).strip()
    if not clean_title:
        raise ValueError("title is required")
    if sections is None:
        section_names: list[str] = list(DEFAULT_PAPER_SECTIONS)
    else:
        section_names = [str(name) for name in sections]
        if not section_names or any(not name.strip() for name in section_names):
            raise ValueError("--section must name a non-empty section; omit it to use the ten default sections")
    created = created_at or utc_now()
    card = build_paper_learning_card(
        title=clean_title,
        authors=authors,
        source_ref=source_ref,
        level=level,
        source_state=source_state,
        observed_sections=observed_sections,
        missing_sections=missing_sections,
        evidence_ref=evidence_ref,
        sections=section_names,
        output_language=output_language,
    )
    record = {
        "schema_version": PAPER_LEARNING_RECORD_SCHEMA_VERSION,
        "paper_id": paper_id or new_paper_id(clean_title, created),
        "card": card,
        "source": describe_paper_source(source_ref),
        "reading": _reading_state(card, progress_count=0, next_section=None, last_note=""),
        "progress_count": 0,
        "created_at": created,
        # `created_at` keeps the store's second-precision timestamp; this is
        # the ordering instant. `write_paper_learning_record` raises it above
        # the newest record already in the store, so two plans recorded in
        # one second (or one clock tick) still list in the order they were
        # recorded rather than by the random id suffix.
        "created_at_ns": time.time_ns() if created_at_ns is None else int(created_at_ns),
        "updated_at": created,
    }
    errors = validate_paper_learning_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_paper_learning_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != PAPER_LEARNING_RECORD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PAPER_LEARNING_RECORD_SCHEMA_VERSION}")
    raw_id = str(record.get("paper_id", ""))
    if not raw_id.strip():
        errors.append("paper_id is required")
    elif not _valid_paper_id(raw_id):
        errors.append("paper_id must contain only letters, digits, and hyphens, and must not contain path separators")
    card = record.get("card")
    if not isinstance(card, dict):
        errors.append("card must be an object")
    else:
        errors.extend(f"card: {error}" for error in validate_paper_learning_card(card))
    source = record.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    elif source.get("kind") not in PAPER_SOURCE_KINDS:
        errors.append("source.kind is unsupported")
    reading = record.get("reading")
    if not isinstance(reading, dict):
        errors.append("reading must be an object")
    elif reading.get("status") not in PAPER_LEARNING_READING_STATUSES:
        errors.append("reading.status is unsupported")
    progress_count = record.get("progress_count")
    if not isinstance(progress_count, int) or isinstance(progress_count, bool) or progress_count < 0:
        errors.append("progress_count must be a non-negative integer")
    created_ns = record.get("created_at_ns")
    if not isinstance(created_ns, int) or isinstance(created_ns, bool) or created_ns < 0:
        errors.append("created_at_ns must be a non-negative integer")
    return errors


def apply_paper_progress(
    record: dict[str, Any],
    *,
    covered: Iterable[str] = (),
    next_section: str | None = None,
    missing: Iterable[str] = (),
    observed: Iterable[str] = (),
    note: str = "",
    source_state: str | None = None,
    evidence_ref: str | None = None,
    recorded_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the updated record and the ledger entry for one recorded chunk.

    Pure: the caller decides whether to persist. Section names are matched the
    way the coverage ledger matches them (case- and punctuation-insensitive)
    and must already be in the ledger, so a typo cannot silently create an
    eleventh section that nothing will ever explain.

    Two claims stay apart. `covered` says a section was explained and moves
    only `explanation_status`. `observed` says the host saw that section's
    text and moves `status` (the source-observation side); it and
    `source_state` are the only inputs that can change the card's
    `source_state`, and on this path both need an evidence reference, either
    passed now or already on the card.
    """
    errors = validate_paper_learning_record(record)
    if errors:
        raise PaperLearningStoreError("record_corrupt", "; ".join(errors), paper_id=str(record.get("paper_id", "")))
    card = record["card"]
    ledger = card["coverage_ledger"]
    covered_names = _resolve_sections(ledger, covered, flag="--covered")
    missing_names = _resolve_sections(ledger, missing, flag="--missing")
    observed_names = _resolve_sections(ledger, observed, flag="--observed-section")
    overlap = [name for name in covered_names + observed_names if name in missing_names]
    if overlap:
        raise ValueError("a section cannot be both covered or observed and missing in one chunk: " + ", ".join(overlap))
    clean_note = " ".join(str(note).split())
    if len(clean_note) > PAPER_LEARNING_NOTE_CHAR_LIMIT:
        raise ValueError(
            f"--note must be at most {PAPER_LEARNING_NOTE_CHAR_LIMIT} characters; keep the explanation in chat, the ledger records where reading stopped"
        )
    if source_state is not None and source_state not in PAPER_LEARNING_SOURCE_STATES:
        raise ValueError("source_state must be one of " + ", ".join(PAPER_LEARNING_SOURCE_STATES))
    if (
        not covered_names
        and not missing_names
        and not observed_names
        and next_section is None
        and not clean_note
        and source_state is None
        and evidence_ref is None
    ):
        raise ValueError("nothing to record: pass --covered, --observed-section, --missing, --next, --note, --source-state, or --evidence-ref")
    previous_state = record["card"]["source_state"]
    effective_evidence = evidence_ref if evidence_ref is not None else str(previous_state.get("evidence_ref", ""))
    if (observed_names or source_state is not None) and not effective_evidence.strip():
        raise ValueError("--observed-section and --source-state record a source observation and need --evidence-ref (none is on the card yet)")

    updated = _deep_copy(record)
    card = updated["card"]
    ledger = card["coverage_ledger"]
    for item in ledger:
        if item["paper_section"] in missing_names:
            item["status"] = "missing"
            item["explanation_status"] = "pending"
        elif item["paper_section"] in observed_names:
            item["status"] = "observed"
    still_missing = [item["paper_section"] for item in ledger if item["paper_section"] in covered_names and item["status"] == "missing"]
    if still_missing:
        raise ValueError(
            "cannot cover a section the ledger records as missing: "
            + ", ".join(still_missing)
            + "; record it with --observed-section <section> --evidence-ref <ref> in the same call once its text is seen"
        )
    for item in ledger:
        if item["paper_section"] in covered_names:
            item["explanation_status"] = "explained"
    resolved_next = None
    if next_section is not None:
        resolved_next = _resolve_sections(ledger, [next_section], flag="--next")[0]

    observed_now = [item["paper_section"] for item in ledger if item["status"] == "observed"]
    missing_now = [item["paper_section"] for item in ledger if item["status"] == "missing"]
    card["source_state"] = normalize_paper_source_state(
        source_state if source_state is not None else previous_state["state"],
        observed_sections=observed_now,
        missing_sections=missing_now,
        evidence_ref=effective_evidence,
    )
    progress_count = int(record["progress_count"]) + 1
    stamp = recorded_at or utc_now()
    card["chunking_policy"]["part_index"] = progress_count + 1
    updated["reading"] = _reading_state(card, progress_count=progress_count, next_section=resolved_next, last_note=clean_note or str(record["reading"].get("last_note", "")))
    updated["progress_count"] = progress_count
    updated["updated_at"] = stamp
    entry = {
        "schema_version": PAPER_LEARNING_PROGRESS_SCHEMA_VERSION,
        "paper_id": record["paper_id"],
        "part_index": progress_count,
        "covered": covered_names,
        "observed": observed_names,
        "next": updated["reading"]["next"],
        "missing": missing_names,
        "note": clean_note,
        "source_state": card["source_state"]["state"],
        "evidence_ref": card["source_state"]["evidence_ref"],
        "reading_status": updated["reading"]["status"],
        "recorded_at": stamp,
    }
    return updated, entry


def write_paper_learning_record(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    """Write a new record, keeping creation order strict across the store.

    Under one store-level lock (the index sidecar), the record's
    `created_at_ns` is raised above the newest record already present when
    the clock did not move between two plans, so listing order is insertion
    order and never the random id suffix. The returned record carries the
    instant that was written.
    """
    errors = validate_paper_learning_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    paper_id = str(record["paper_id"])
    card_path = paper_learning_card_path(paths, paper_id)
    with file_lock(paths.paper_learning_index_path, private=True):
        if card_path.exists():
            raise ValueError(f"paper learning record already exists: {paper_id}")
        existing, _ = scan_paper_learning_store(paths)
        newest = max((int(item["created_at_ns"]) for item in existing), default=-1)
        if int(record["created_at_ns"]) <= newest:
            record = dict(record, created_at_ns=newest + 1)
        atomic_write_json(card_path, record, private=True)
        _write_index_cache(paths)
    return record


def record_paper_progress(paths: OmhPaths, paper_id: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one chunk to the stored record: rewrite the card, append the ledger line.

    The read, the ledger append, and the card rewrite all happen under one
    exclusive lock on the card, so two concurrent chunks serialize instead of
    both reading progress_count N and both writing N+1 while the ledger
    grows by two. The ledger line is appended first and the card written
    last: the ledger is the append-only history and the card is a projection
    of it, so a failure between the two writes leaves the history complete
    and the card one chunk behind, which `validate` names and a replay of the
    ledger can repair. The reverse order would leave a chunk in the card
    that the history never recorded, which nothing can reconstruct.
    """
    card_path = _card_path_or_not_found(paths, paper_id)
    with file_lock(card_path, private=True):
        record = _read_valid_record(card_path, paper_id)
        updated, entry = apply_paper_progress(record, **kwargs)
        append_jsonl_locked(paper_learning_ledger_path(paths, paper_id), entry, private=True)
        atomic_write_json(card_path, updated, private=True)
    _write_index_cache(paths)
    return updated, entry


def scan_paper_learning_store(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Read every record directory: usable records, and the ones that are not.

    A directory without a card, a card that does not parse, and a card that
    fails validation are all returned in the second list with a reason code,
    never skipped, so `list` can say how many records it could not show and
    `validate` can name each one.
    """
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    if not paths.paper_learning_dir.is_dir():
        return records, unreadable
    for entry in sorted(paths.paper_learning_dir.iterdir()):
        # A linked directory is skipped the way `show` refuses it: the record
        # id must name a real directory under the store, so a link would list
        # a row for an id that no other command accepts.
        if is_directory_link(entry) or not entry.is_dir():
            continue
        card_path = entry / PAPER_LEARNING_CARD_FILENAME
        if not card_path.exists():
            unreadable.append(_unreadable(entry.name, card_path, "record_corrupt", f"missing {PAPER_LEARNING_CARD_FILENAME}"))
            continue
        record, error = read_json_object_result(card_path)
        if error:
            unreadable.append(_unreadable(entry.name, card_path, "record_corrupt", error))
            continue
        if not record:
            unreadable.append(_unreadable(entry.name, card_path, "record_corrupt", "empty record"))
            continue
        errors = validate_paper_learning_record(record)
        if errors:
            unreadable.append(_unreadable(entry.name, card_path, "record_corrupt", "; ".join(errors)))
            continue
        if str(record.get("paper_id", "")) != entry.name:
            unreadable.append(_unreadable(entry.name, card_path, "record_corrupt", f"paper_id {record.get('paper_id')!r} does not match its directory"))
            continue
        records.append(record)
    records.sort(key=lambda item: (int(item["created_at_ns"]), str(item["created_at"]), str(item["paper_id"])))
    return records, unreadable


def list_paper_learning_records(paths: OmhPaths, *, limit: int | None = None) -> list[dict[str, Any]]:
    records, _ = scan_paper_learning_store(paths)
    if limit is not None:
        if limit < 1:
            return []
        records = records[-limit:]
    return records


def show_paper_learning_record(paths: OmhPaths, paper_id: str) -> dict[str, Any]:
    return _read_valid_record(_card_path_or_not_found(paths, paper_id), paper_id)


def read_paper_progress_ledger(paths: OmhPaths, paper_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    _card_path_or_not_found(paths, paper_id)
    return read_jsonl_objects(paper_learning_ledger_path(paths, paper_id))


def paper_learning_record_exists(paths: OmhPaths, paper_id: str) -> bool:
    if not _valid_paper_id(paper_id):
        return False
    return paper_learning_card_path(paths, paper_id).exists()


def summarize_paper_learning_record(record: dict[str, Any]) -> dict[str, Any]:
    card = record.get("card", {}) if isinstance(record.get("card"), dict) else {}
    identity = card.get("paper_identity", {}) if isinstance(card.get("paper_identity"), dict) else {}
    source_state = card.get("source_state", {}) if isinstance(card.get("source_state"), dict) else {}
    reading = record.get("reading", {}) if isinstance(record.get("reading"), dict) else {}
    return {
        "paper_id": str(record.get("paper_id", "")),
        "title": str(identity.get("title", "")),
        "level": str(card.get("level", "")),
        "source_state": str(source_state.get("state", "")),
        "reading_status": str(reading.get("status", "")),
        "explained_count": int(reading.get("explained_count", 0) or 0),
        "section_count": int(reading.get("section_count", 0) or 0),
        "next": str(reading.get("next", "")),
        "missing_count": len(reading.get("missing", []) if isinstance(reading.get("missing"), list) else []),
        "progress_count": int(record.get("progress_count", 0) or 0),
        "updated_at": str(record.get("updated_at", "")),
    }


def validate_paper_learning_store(paths: OmhPaths) -> dict[str, Any]:
    errors: list[str] = []
    records, unreadable = scan_paper_learning_store(paths)
    for item in unreadable:
        errors.append(f"{item['paper_id']}: {item['reason_code']}: {item['detail']}")
    for record in records:
        paper_id = str(record["paper_id"])
        ledger_entries, ledger_errors = read_jsonl_objects(paths.paper_learning_dir / paper_id / PAPER_LEARNING_LEDGER_FILENAME)
        errors.extend(ledger_errors)
        recorded = record["progress_count"]
        if len(ledger_entries) != recorded:
            errors.append(f"{paper_id}: progress_count {recorded} but the ledger holds {len(ledger_entries)} entries")
        for index, ledger_entry in enumerate(ledger_entries, start=1):
            if ledger_entry.get("schema_version") != PAPER_LEARNING_PROGRESS_SCHEMA_VERSION:
                errors.append(f"{paper_id}: ledger entry {index} has unsupported schema_version")
            if ledger_entry.get("paper_id") != paper_id:
                errors.append(f"{paper_id}: ledger entry {index} names paper_id {ledger_entry.get('paper_id')!r}")
    index, index_error = read_json_object_result(paths.paper_learning_index_path)
    if index_error:
        errors.append(f"paper-learning index cache is unreadable: {index_error}")
    elif index and index.get("schema_version") != PAPER_LEARNING_INDEX_SCHEMA_VERSION:
        errors.append("paper-learning index cache has unsupported schema_version")
    return {
        "schema_version": "omh_paper_learning_validation/v1",
        "ok": not errors,
        "paper_count": len(records),
        "unreadable_count": len(unreadable),
        "unreadable_records": unreadable,
        "errors": errors,
        "index_authority": "cache_only",
    }


def render_paper_learning_record_text(record: dict[str, Any], ledger_entries: Iterable[dict[str, Any]] = ()) -> str:
    card = record["card"]
    identity = card["paper_identity"]
    reading = record["reading"]
    source = record["source"]
    lines = [
        f"Paper: {identity['title']}",
        f"  id: {record['paper_id']}",
        f"  level: {card['level']}",
        f"  source: {source['ref'] or 'not supplied'} ({source['kind']})",
    ]
    if source["kind"] == "file" and source.get("hash_skipped"):
        lines.append(f"  source sha256: skipped ({source['hash_skipped']}; {source['size_bytes']} bytes exceeds the {source.get('hash_budget_bytes')} byte hash budget)")
    elif source["kind"] == "file":
        lines.append(f"  source sha256: {source['sha256']} ({source['size_bytes']} bytes; identity only, not extraction evidence)")
    lines.extend(
        [
            f"  source state: {card['source_state']['state']}",
            f"  reading: {reading['status']} ({reading['explained_count']}/{reading['section_count']} sections explained, {record['progress_count']} chunk(s) recorded)",
            f"  covered: {', '.join(reading['covered']) or 'none yet'}",
            f"  next: {reading['next'] or 'none (ledger complete or only missing sections remain)'}",
            f"  missing: {', '.join(reading['missing']) or 'none'}",
        ]
    )
    if reading.get("last_note"):
        lines.append(f"  last note: {reading['last_note']}")
    lines.append("")
    lines.append("Coverage ledger:")
    for item in card["coverage_ledger"]:
        lines.append(f"  - {item['paper_section']}: {item['status']} / {item['explanation_status']}")
    entries = list(ledger_entries)
    if entries:
        lines.append("")
        lines.append("Progress ledger:")
        for entry in entries:
            summary = f"  part {entry.get('part_index')}: covered {', '.join(entry.get('covered', [])) or '-'}; next {entry.get('next') or '-'}"
            if entry.get("observed"):
                summary += f"; observed {', '.join(entry['observed'])}"
            if entry.get("missing"):
                summary += f"; missing {', '.join(entry['missing'])}"
            if entry.get("note"):
                summary += f"; note: {entry['note']}"
            lines.append(summary)
    lines.append("")
    lines.append(f"Not observed: {', '.join(card['not_observed'])}")
    lines.append("A recorded chunk is where reading stopped, not proof the explanation was correct or complete.")
    return "\n".join(lines)


def render_paper_learning_list_text(
    summaries: Iterable[dict[str, Any]],
    unreadable: Iterable[dict[str, str]] = (),
    *,
    total_count: int | None = None,
) -> str:
    rows = list(summaries)
    broken = list(unreadable)
    lines: list[str] = []
    if not rows:
        lines.append("No paper learning records yet. Start one with: omh paper plan --title <title> --source <path or url>")
    else:
        lines.append("Paper learning records:")
        for row in rows:
            next_text = f"next {row['next']}" if row["next"] else "no next section"
            lines.append(
                f"  {row['paper_id']}  {row['title']}  [{row['reading_status']}, {row['explained_count']}/{row['section_count']} explained, {next_text}]"
            )
        if total_count is not None and total_count > len(rows):
            lines.append(f"Showing the latest {len(rows)} of {total_count} records; pass --all or --limit N for more.")
    if broken:
        lines.append(f"Unreadable records ({len(broken)}), not shown above:")
        for item in broken:
            lines.append(f"  {item['paper_id']}: {item['reason_code']}: {item['detail']}")
        lines.append("Next: omh paper validate")
    return "\n".join(lines)


def paper_learning_card_path(paths: OmhPaths, paper_id: str) -> Path:
    return _paper_dir(paths, paper_id) / PAPER_LEARNING_CARD_FILENAME


def paper_learning_ledger_path(paths: OmhPaths, paper_id: str) -> Path:
    return _paper_dir(paths, paper_id) / PAPER_LEARNING_LEDGER_FILENAME


def _card_path_or_not_found(paths: OmhPaths, paper_id: str) -> Path:
    if not _valid_paper_id(paper_id):
        raise PaperLearningStoreError("record_not_found", paper_id, paper_id=paper_id)
    # A link under the store is not a record: `scan` skips it, and the same id
    # must not open through `show` or `progress` either.
    if is_directory_link(paths.paper_learning_dir / paper_id):
        raise PaperLearningStoreError("record_not_found", paper_id, paper_id=paper_id)
    card_path = paper_learning_card_path(paths, paper_id)
    if not card_path.exists():
        raise PaperLearningStoreError("record_not_found", paper_id, paper_id=paper_id)
    return card_path


def _read_valid_record(card_path: Path, paper_id: str) -> dict[str, Any]:
    record, error = read_json_object_result(card_path)
    if error:
        raise PaperLearningStoreError("record_corrupt", f"{paper_id}: {error}", paper_id=paper_id)
    if not record:
        raise PaperLearningStoreError("record_corrupt", f"{paper_id}: empty record", paper_id=paper_id)
    errors = validate_paper_learning_record(record)
    if errors:
        raise PaperLearningStoreError("record_corrupt", f"{paper_id}: " + "; ".join(errors), paper_id=paper_id)
    return record


def _unreadable(paper_id: str, card_path: Path, reason_code: str, detail: str) -> dict[str, str]:
    return {"paper_id": paper_id, "path_name": card_path.name, "reason_code": reason_code, "detail": detail}


def _paper_dir(paths: OmhPaths, paper_id: str) -> Path:
    if not _valid_paper_id(paper_id):
        raise ValueError("paper_id must contain only letters, digits, and hyphens, and must not contain path separators")
    path = paths.paper_learning_dir / paper_id
    root = paths.paper_learning_dir.resolve()
    if path.resolve().parent != root:
        raise ValueError("paper_id escapes paper-learning storage")
    return path


def _write_index_cache(paths: OmhPaths) -> None:
    records = list_paper_learning_records(paths)
    ensure_dir(paths.paper_learning_dir, private=True)
    atomic_write_json(
        paths.paper_learning_index_path,
        {
            "schema_version": PAPER_LEARNING_INDEX_SCHEMA_VERSION,
            "updated_at": utc_now(),
            "authority": "cache_only",
            "papers": [summarize_paper_learning_record(record) for record in records],
        },
        private=True,
    )


def _reading_state(card: dict[str, Any], *, progress_count: int, next_section: str | None, last_note: str) -> dict[str, Any]:
    ledger = card["coverage_ledger"]
    covered = [item["paper_section"] for item in ledger if item["explanation_status"] == "explained"]
    missing = [item["paper_section"] for item in ledger if item["status"] == "missing"]
    pending = [item["paper_section"] for item in ledger if item["explanation_status"] == "pending" and item["status"] != "missing"]
    if next_section is None:
        next_section = pending[0] if pending else ""
    if len(covered) == len(ledger):
        status = "complete"
    elif progress_count == 0:
        status = "not_started"
    elif not pending:
        status = "waiting_for_missing_sections"
    else:
        status = "in_progress"
    return {
        "status": status,
        "covered": covered,
        "next": next_section,
        "missing": missing,
        "explained_count": len(covered),
        "section_count": len(ledger),
        "last_note": last_note,
    }


def _resolve_sections(ledger: list[dict[str, str]], names: Iterable[str], *, flag: str) -> list[str]:
    by_key = {_key(item["paper_section"]): item["paper_section"] for item in ledger}
    resolved: list[str] = []
    for name in names:
        key = _key(str(name))
        if not key:
            continue
        if key not in by_key:
            known = ", ".join(item["paper_section"] for item in ledger)
            raise ValueError(f"{flag} {name!r} is not a section of this paper's coverage ledger; known sections: {known}")
        if by_key[key] not in resolved:
            resolved.append(by_key[key])
    return resolved


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    if isinstance(value, tuple):
        return [_deep_copy(item) for item in value]
    return value


def _valid_paper_id(value: str) -> bool:
    return bool(_PAPER_ID_RE.match(value)) and "/" not in value and "\\" not in value and ".." not in value


def _stamp(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _slugify(value)[:32] or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return _stamp(parsed)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    return (_SLUG_RE.sub("-", value.lower()).strip("-") or "paper")[:48].strip("-")

"""Mechanics every append-only JSONL record store in this tree shares.

`workflows/external_effect_receipts.py` and `workflows/approval_receipts.py`
are two record families with nothing in common at the domain level -- one
records what an acting surface observed, the other records what an operator
consented to -- and yet they were, line for line, the same store: the same
torn-tail-safe append, the same supersede-chain walk, the same opaque-reference
guards, the same digest handles, the same best-effort mint-failure sidecar. A
third family arriving would have made every future drift a three-way problem,
so the store mechanics live here once and each family keeps its own meaning.

What is here is deliberately only the mechanics. There is no shared "record"
type and no base class: a record abstraction that knew what a receipt or an
approval is would be the wrong shape, because the whole reason these families
are separate is that their vocabularies, key sets, claim boundaries, schema
versions, and domain predicates must not converge. Every function below takes
what it needs as an explicit parameter -- the caller's label, the caller's
error class, the caller's key tuple -- rather than reading it off a config
object, which keeps each call site readable as the thing it does.

Two properties are load-bearing on both platforms and are not negotiable in any
caller:

- The append is binary-mode. Windows text mode rewrites ``\\n`` as ``\\r\\n``,
  which would make the tail-byte check platform-dependent and the store's bytes
  differ between CI targets.
- The append fsyncs, and every writer holds `local_store.file_lock`, which is a
  real OS lock through `fcntl` on POSIX and `msvcrt` on Windows. Append-only is
  a structural property here, not a convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .local_store import ensure_dir, ensure_file, file_lock
from .metadata_safety import is_sensitive_metadata_text, require_opaque_metadata_ref


# Mirrors the executor-progress guard: these key names are how raw model output
# and private payloads arrive, so a store rejects them by name as well as by its
# own closed key set. Held here so the families cannot drift apart on it -- they
# were already identical, and a key added to one and missed on the other is the
# exact defect this module exists to make impossible.
RAW_OR_HIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "analysis",
        "body",
        "body_text",
        "chain_of_thought",
        "conversation",
        "cot",
        "hidden",
        "hidden_reasoning",
        "message",
        "payload",
        "prompt",
        "raw",
        "raw_log",
        "raw_logs",
        "raw_message",
        "raw_output",
        "raw_payload",
        "raw_prompt",
        "reasoning",
        "stderr",
        "stdout",
        "think",
        "thinking",
        "transcript",
        "url",
    }
)

# A reference that navigates somewhere is not an opaque identifier. Kept in the
# same shape as `adapter_quality._URL`, which is where this repo first drew the
# line between "a handle we can print" and "a link we must not".
URL_SHAPED: tuple[str, ...] = ("://", "?", "#")

# The free-text guard a bounded metadata line goes through: a URL scheme or any
# of `/ ? #` is a link or a filesystem path, and a control character means the
# value is a transcript or a prompt rather than one metadata line.
_UNSAFE_TEXT_URL = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*://|[/?#])")
_UNSAFE_TEXT_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_TEXT_WINDOWS_PATH = re.compile(r"(?:^|\s)(?:[A-Za-z]:\\|\\\\)")


def append_store_line(path: Path, record: Mapping[str, Any]) -> None:
    """Append one line, terminating a torn tail first.

    A short write leaves a partial line with no trailing newline. Appending
    straight onto it concatenates two records into one unparseable line and
    loses both, so the tail byte is checked and a newline is written first when
    it is missing. The torn line stays on disk as one corrupt line the store
    validator reports; the new record survives as its own.

    Binary mode on purpose: Windows text mode would rewrite ``\\n`` as
    ``\\r\\n`` and make the tail check platform-dependent.

    The caller holds the store's lock. This function does not take one, because
    a mint reads the prior record and appends under a single lock and a second
    acquisition here would either deadlock or open the window it closed.
    """
    ensure_dir(path.parent, private=True)
    ensure_file(path, private=True)
    payload = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
    with path.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                payload = b"\n" + payload
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def append_sidecar_line(path: Path, record: Mapping[str, Any]) -> None:
    """Append one line to a store's sidecar log, best effort, under its own lock.

    The sidecar is where a mint that never reached the store is logged. It is
    the record that has to survive when things are already going wrong, so two
    producers failing at once must not interleave and lose a line -- hence a
    real lock, on the sidecar's own path.

    The lock is never held while the store's lock is: a mint failure is raised
    out of that block before this runs.

    A write failure is swallowed. The sidecar lives next to the store, so the
    same outage that lost the record can lose this line too, and the result the
    caller already holds is the record of last resort.
    """
    try:
        with file_lock(path, private=True):
            append_store_line(path, record)
    except OSError:
        return


def is_url_shaped(value: str) -> bool:
    """Whether a reference navigates somewhere instead of naming something."""
    return any(marker in value for marker in URL_SHAPED)


def digest_ref(value: str) -> str:
    """A stable, bounded, non-navigable handle for a value that is not opaque.

    Stable because callers dedupe rows by it; non-navigable because a status
    report is not a place to publish someone's links.
    """
    return f"ref-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def opaque_ref(value: str, *, field: str, error: type[Exception]) -> str:
    """One caller-supplied identifier, or the caller's own error.

    `field` is the fully qualified name the store reports, and `error` is that
    store's exception type, so a refusal from here is indistinguishable from
    one the store raised itself.
    """
    text = str(value or "").strip()
    if not text:
        raise error(f"{field} is required")
    if is_url_shaped(text):
        raise error(f"{field} must be an opaque identifier, not a URL")
    try:
        return require_opaque_metadata_ref(text, field=field)
    except ValueError as exc:
        raise error(str(exc)) from exc


def reference_errors(value: Any, *, field: str, label: str, required: bool) -> list[str]:
    """Every reason one stored value is not an opaque reference.

    The validation-time counterpart of `opaque_ref`: it collects rather than
    raises, because a store report names every fault at once.
    """
    if not isinstance(value, str):
        return [f"{label} {field} must be a string"]
    if not value:
        return [f"{label} {field} is required"] if required else []
    if is_url_shaped(value):
        return [f"{label} {field} must be an opaque identifier, not a URL"]
    try:
        require_opaque_metadata_ref(value, field=f"{label} {field}")
    except ValueError as exc:
        return [str(exc)]
    return []


def redacted_ref(value: str, *, field: str) -> str:
    """Fold any handle into a bounded, non-navigable reference for rendering.

    A URL, a secret-shaped string, or anything longer than a bounded identifier
    becomes a stable digest handle instead. Rendering is the last point at which
    a hand-edited store reaches a human, so nothing is copied through verbatim.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if is_url_shaped(text) or is_sensitive_metadata_text(text):
        return digest_ref(text)
    try:
        require_opaque_metadata_ref(text, field=field)
    except ValueError:
        return digest_ref(text)
    return text


def closed_vocabulary_value(value: Any, allowed: tuple[str, ...]) -> str:
    """A closed-vocabulary field renders only what its vocabulary contains.

    Rendering reads whatever is on disk, and a store can be hand-edited. A value
    outside the vocabulary is not a new state, it is a value with no meaning, so
    it renders empty rather than as itself.
    """
    text = str(value or "")
    return text if text in allowed else ""


def is_unsafe_metadata_line(value: str) -> bool:
    """Whether one free-text field carries something a metadata line must not.

    A secret, a link, a filesystem path, or a control character. The last one is
    the tell that the value is a prompt, a transcript, or a log slice rather
    than the single bounded line a stored summary is allowed to be.
    """
    text = str(value or "")
    if not text:
        return False
    return bool(
        is_sensitive_metadata_text(text)
        or _UNSAFE_TEXT_CONTROL.search(text)
        or _UNSAFE_TEXT_URL.search(text)
        or _UNSAFE_TEXT_WINDOWS_PATH.search(text)
    )


def latest_record_in(records: Sequence[Mapping[str, Any]], *, key: str, value: str) -> Any:
    """The record that speaks for one identity: the one selection rule.

    An append-only store's order is arrival order, so the last record carrying
    an identity is that identity's current state and a later append always wins
    the tie. Every surface that judges a store selects through this function,
    which is what stops a validator and a predicate from picking different
    records for one identity and reaching opposite conclusions about it.

    The match is returned as it was found, not copied: a caller that needs a
    copy makes one, and a caller that does not must not pay for one.
    """
    matches = [record for record in records if str(record.get(key, "")) == str(value)]
    return matches[-1] if matches else {}


def record_fingerprint(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Which thing a record is about, ignoring when it was recorded.

    `keys` is the store's own identity tuple. What it leaves out is the point:
    a timestamp, a record id, and a supersede link must not make an identical
    re-report mint a second record just because the clock moved.
    """
    identity = {key: record.get(key) for key in keys}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def mint_record_id(*, prefix: str, identity: Mapping[str, str]) -> str:
    """A record id from what the record is, plus a random tail.

    Random tail for the same reason `observation_journal._event_id` carries one:
    `utc_now()` has second resolution, so two records for one identity in the
    same second would otherwise share an id and break the supersede chain.
    """
    base = json.dumps(dict(identity), sort_keys=True)
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}-{secrets.token_hex(3)}"


def supersede_chain_errors(
    path: Path,
    records: list[dict[str, Any]],
    *,
    run_id: str | None,
    label: str,
    id_key: str = "receipt_id",
    link_key: str = "supersedes_receipt_ref",
    noun: str = "receipt",
) -> list[str]:
    """Reject every shape a supersede chain must not take.

    A chain is a line, not a graph: a record cannot supersede itself, cannot
    supersede a record that does not already exist, and cannot supersede a
    record something else already superseded. A fork means two writers both
    believe they replaced the same predecessor, which makes "the latest state"
    ambiguous.

    Every record in the store is walked even when the report is scoped to one
    run, so a link into a record outside the scope is still a link into a record
    that exists. Only faults on in-scope records are reported: scoping a report
    must never invent a broken chain.

    `id_key`, `link_key`, and `noun` are how a family keeps its own vocabulary
    while sharing the walk. The defaults are the two receipt families' names, so
    those two callers pass nothing and their error lines are unchanged to the
    byte; a family whose records are not receipts names its own keys rather than
    storing a `receipt_id` it has no receipts for. The names are parameters and
    not a config object for the reason the module docstring gives: the call site
    should read as the thing it does.
    """
    seen: set[str] = set()
    successors: dict[str, str] = {}
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        in_scope = run_id is None or str(record.get("run_id", "")) == run_id
        record_id = str(record.get(id_key, ""))
        if record_id and record_id in seen and in_scope:
            errors.append(f"{path}:{index}: {label} {id_key} is not unique: {record_id}")
        superseded = str(record.get(link_key, ""))
        if superseded:
            if superseded == record_id:
                if in_scope:
                    errors.append(
                        f"{path}:{index}: {label} {link_key} must not name itself: {record_id}"
                    )
            elif superseded not in seen:
                if in_scope:
                    errors.append(
                        f"{path}:{index}: {label} {link_key} does not name an earlier "
                        f"{noun}: {superseded}"
                    )
            elif superseded in successors:
                if in_scope:
                    errors.append(
                        f"{path}:{index}: {label} {link_key} forks the chain: "
                        f"{superseded} is already superseded by {successors[superseded]}"
                    )
            else:
                successors[superseded] = record_id
        seen.add(record_id)
    return errors


def store_errors(
    path: Path,
    records: list[dict[str, Any]],
    read_errors: list[str],
    *,
    run_id: str | None,
    validate_record: Callable[[dict[str, Any]], list[str]],
    label: str,
    id_key: str = "receipt_id",
    link_key: str = "supersedes_receipt_ref",
    noun: str = "receipt",
) -> tuple[int, list[str]]:
    """How many records are in scope, and every fault found in them.

    Unscoped (`run_id=None`) this is the store's own report: unparseable lines,
    every record's schema, and the integrity of the supersede chain. A line that
    does not parse belongs to no run -- it has no `run_id` to read -- so it is
    the store's fault and is reported here, once, never against a run that
    happened to be validated while it sat there.

    Scoped to a run, only that run's records and their chain are considered, so
    one bad line elsewhere cannot fault a run that has nothing to do with it.

    The caller owns the report's wire shape: its schema version, what it names
    the count, and its sidecar tally are all its own.
    """
    indexed = [
        (index, record)
        for index, record in enumerate(records, start=1)
        if run_id is None or str(record.get("run_id", "")) == run_id
    ]
    errors: list[str] = list(read_errors) if run_id is None else []
    for index, record in indexed:
        errors.extend(f"{path}:{index}: {error}" for error in validate_record(record))
    errors.extend(
        supersede_chain_errors(
            path,
            records,
            run_id=run_id,
            label=label,
            id_key=id_key,
            link_key=link_key,
            noun=noun,
        )
    )
    return len(indexed), errors

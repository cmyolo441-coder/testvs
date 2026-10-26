"""Where an answer to a route question is recorded, and what it may claim.

A route question is built by the deterministic router when it cannot decide
(`omh.routing.route_question`). Something else answers it -- the host model
reading the payload, or a Jev-class plugin the operator installed -- and this
module is where that answer is written down: one
``route_question_answer/v1`` record per (session, question) under
``$OMH_HOME/runtime/route-questions/``.

Recording an answer changes no route. The record exists to be MEASURED: it
embeds one ``routing_question_answers/v1`` row, which is the shape
``omh chat route-questions score --answers <dir>`` already reads, so an
answerer's judgments are scored against the same corpus as the deterministic
router instead of sitting in a write-only ledger.

Two things this module deliberately does not do. It does not call anything --
the answer arrives as tool arguments from a caller that already has it. And it
does not decide that an answer is right: ``confidence_source`` records WHO
said it (``self_reported`` for the model answering about itself,
``answerer_declared`` for a plugin reporting a number OMH did not observe),
never that a confidence was calibrated, which is a claim about a vendor's
model and not about a row OMH wrote from an argument.

Stdlib and intra-bundle imports only; the write takes the bundle's one
sanctioned lock, the same object `todo_store` and `tool_bursts` take.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .awareness_delivery import _awareness_delivery_lock
from .route_answer_consistency import INVALID_ANSWER_VERDICT, invalid_answer_reasons
from .route_question_mode import route_question_mode_fields
from .todo_store import strip_control_characters

ROUTE_ANSWER_SCHEMA_VERSION = "route_question_answer/v1"
# The row shape the corpus scorer reads. Restated here as a literal because
# the bundle cannot import `omh.quality.routing_question_corpus`;
# `tests/test_route_answer_tool.py` pins it against that producer, so a drift
# is a test failure rather than a silently unreadable record.
ANSWER_ROW_SCHEMA_VERSION = "routing_question_answers/v1"

ROUTE_ANSWER_DIRNAME = "route-questions"

ANSWERED_BY_JEV_PLUGIN = "jev_plugin"
ANSWERED_BY_MAIN_MODEL = "main_model"
# OMH's own `omh_jev_ask` call. Accepted only with an `ask_id` whose ledger row
# is `answered` and carries the same question digest, so the host model cannot
# claim Jev provenance for its own guess.
ANSWERED_BY_OMH_JEV_ASK = "omh_jev_ask"
ANSWERED_BY_VALUES = (ANSWERED_BY_JEV_PLUGIN, ANSWERED_BY_MAIN_MODEL, ANSWERED_BY_OMH_JEV_ASK)

# Who said the numbers, never how good they are. `calibrated` is not a value
# OMH can write: it never sees the request, the response, or whether a plugin
# called anything at all.
CONFIDENCE_SELF_REPORTED = "self_reported"
CONFIDENCE_ANSWERER_DECLARED = "answerer_declared"
# A number OMH received itself, in the reply to its own ask. Still not
# `calibrated`: it says who produced the number, not how good it is.
CONFIDENCE_OBSERVED_FROM_RESPONSE = "observed_from_response"

DISPATCH_ACTION = "dispatch"
CLARIFY_ACTION = "clarify"
NONE_ACTION = "none"

# Whether the answer, read as a whole, is one opinion. `invalid_answer` is
# `route_answer_consistency`'s verdict: the record is still written, so the
# rate is countable, and it reads as "no opinion" everywhere it is scored.
ANSWER_VERDICT_ACCEPTED = "accepted"
ANSWER_VERDICT_INVALID = INVALID_ANSWER_VERDICT

NO_WORKFLOW_OPTION = "none"
ROUTE_CHOICE_KEY = "route_choice"
FIT_QUESTION_PREFIX = "fits::"

# Restated from `omh.routing.route_question`, pinned by the same parity test
# as the schema strings above. A record has to resolve its own action on a
# machine where the package is not importable, and the thresholds are what
# resolve it.
FITS_DISPATCH_THRESHOLD = 0.8
FITS_CLARIFY_THRESHOLD = 0.5

MAX_FIT_ANSWERS = 8
MAX_CHOICE_OPTIONS = 16
MAX_SKILL_NAME_CHARS = 80
MAX_NOTE_CHARS = 200
MAX_SESSION_REF_CHARS = 160
MAX_DIGEST_CHARS = 64
# A sha256 hex digest is exactly this long. Separate from the digest bound
# above, which is a cap on a field whose producer OMH does not own here.
SHA256_HEX_CHARS = 64
MAX_ROUTE_ANSWER_RECORD_BYTES = 32_768
# Records past this age are removed on the next write in the same directory.
# Age is the ONLY thing that removes one: a measurement run writes one record
# per question it answered, and evicting a fresh record to make room for a
# fresher one would drop the measurement the record exists for.
ROUTE_ANSWER_STALE_SECONDS = 604_800
# The ceiling the stale bound alone does not provide. A measurement run answers
# one question per corpus item, so this is far above an honest run and far
# below what an unbounded caller can spend: reaching it means something is
# writing records nobody is scoring, and the next write is refused rather than
# a recorded answer evicted to make room.
MAX_ROUTE_ANSWER_RECORDS = 1024

CLAIM_BOUNDARY = (
    "A recorded answer is a routing judgment declared by the caller, not "
    "execution, review, CI, or merge evidence, and it does not change the "
    "route. A main_model confidence is self-reported; an answerer_declared "
    "confidence was not observed by OMH; an observed_from_response confidence "
    "is the number Jev returned to OMH's own omh_jev_ask call."
)

_RECORD_NAME = re.compile(r"(?:[A-Za-z0-9_-]{1,48}-)?[0-9a-f]{16}\.json")
_TEMPORARY_NAME = re.compile(r"\..*\.tmp")
_LOCK_NAME = re.compile(r"\..*\.json\.lock")
_LOCK_TIMEOUT_SECONDS = 2.0


class RouteAnswerStoreError(RuntimeError):
    """The destination could not be written."""


class RouteAnswerValidationError(ValueError):
    """The caller's answer is not one this store can record."""


class RouteAnswerContendedError(RuntimeError):
    """Another writer held this record; nothing was written."""


def confidence_source_for(answered_by: str) -> str:
    """Who is making the confidence claim, derived from the answerer.

    Derived rather than taken as an argument: a caller that could name its own
    confidence source could name `calibrated`, and the whole point of the
    field is that OMH says who spoke, not how good the number is.
    """
    if answered_by == ANSWERED_BY_JEV_PLUGIN:
        return CONFIDENCE_ANSWERER_DECLARED
    if answered_by == ANSWERED_BY_OMH_JEV_ASK:
        return CONFIDENCE_OBSERVED_FROM_RESPONSE
    return CONFIDENCE_SELF_REPORTED


def resolve_action(
    fits: Mapping[str, float],
    route_choice: str,
    *,
    fits_dispatch: float = FITS_DISPATCH_THRESHOLD,
    fits_clarify: float = FITS_CLARIFY_THRESHOLD,
) -> str:
    """dispatch / clarify / none, the way the corpus scorer resolves it.

    The yes/no answers decide WHETHER: the strongest fit against the two flat
    thresholds picks the band. The Choice decides WHICH and is taken as given,
    so an answer set that fits nothing and still names a workflow is a
    dispatch on the Choice alone -- the same reading
    `score_routing_question_answers` applies, kept identical so a record
    scored offline and a record read here cannot disagree.
    """
    if fits:
        strongest = max(fits.values())
        if strongest >= fits_dispatch:
            return DISPATCH_ACTION
        if strongest >= fits_clarify:
            return CLARIFY_ACTION
        return NONE_ACTION
    if route_choice != NO_WORKFLOW_OPTION:
        return DISPATCH_ACTION
    return NONE_ACTION


def build_route_answer_record(
    *,
    question_digest: object,
    answered_by: object,
    route_choice: object,
    fits: object = None,
    choice_probabilities: object = None,
    note: object = "",
    session_ref: object = "",
    message_sha256: object = "",
    digest_verified: bool = False,
    recorded_at: str = "",
    ask_id: object = "",
    mode_reading: object = None,
    question_options: object = None,
) -> dict[str, Any]:
    """Validate one answer and return the record to write.

    Every field is validated before anything is written, and an invalid field
    raises rather than being dropped: a record with a silently missing fit is
    a record that scores as a weaker answer than the caller gave.

    `message_sha256` identifies the REQUEST the question was built for, which
    the digest alone does not: the digest covers the shortlist, and one
    shortlist serves every request the router could not place, so a scorer
    joining on the digest alone joins answers to the wrong corpus item. It is
    the only field here that may legitimately be empty -- a caller that
    supplied neither the message nor its hash has not identified a request,
    and an empty string says so instead of a hash of nothing.

    `mode_reading` is `route_question_mode.read_route_question_mode`'s result
    for the home this record is written to. Every record carries the mode and
    where it was read from; a caller that read nothing gets `unknown` /
    `not_read`, never a default.

    `question_options` are the Choice options of the question the digest
    names, when the caller could re-derive it. They are what lets coverage be
    judged; without them `answer_verdict` rests on mass and argmax alone and
    the record says so with `coverage_checked: false`.
    """
    digest = _validated_digest(question_digest)
    answerer = str(answered_by or "").strip()
    if answerer not in ANSWERED_BY_VALUES:
        raise RouteAnswerValidationError(
            "answered_by must be one of " + ", ".join(ANSWERED_BY_VALUES)
        )
    choice = _validated_skill(route_choice, field="route_choice")
    fit_values = _validated_fits(fits)
    probabilities = _validated_probabilities(choice_probabilities)
    message_hash = _validated_message_sha256(message_sha256)
    options = (
        [str(option) for option in question_options]
        if isinstance(question_options, (list, tuple, set, frozenset))
        else None
    )
    invalid = invalid_answer_reasons(choice, probabilities, options=options)
    record: dict[str, Any] = {
        "schema_version": ROUTE_ANSWER_SCHEMA_VERSION,
        # The bundle's reading of this answer at the moment it was recorded.
        # The scorer resolves the action again from the corpus's own
        # thresholds or a caller override, so this value and that one can
        # differ whenever the thresholds do. It is recorded WITH the
        # thresholds that produced it for exactly that reason: a stored number
        # that cannot be reproduced is a number a reader has to guess about.
        "action": resolve_action(fit_values, choice),
        "action_thresholds": {
            "fits_clarify": FITS_CLARIFY_THRESHOLD,
            "fits_dispatch": FITS_DISPATCH_THRESHOLD,
        },
        "answer": _answer_row(
            digest=digest,
            arm=answerer,
            choice=choice,
            probabilities=probabilities,
            fits=fit_values,
            message_sha256=message_hash,
        ),
        "answer_verdict": ANSWER_VERDICT_INVALID if invalid else ANSWER_VERDICT_ACCEPTED,
        # Whether `accepted` covers coverage too. False when the answer carries
        # a distribution but the question's options were not at hand (no
        # `message` to re-derive them from): mass and argmax were checked,
        # coverage was not, and a reader must not treat the verdict as whole.
        "coverage_checked": options is not None or not probabilities,
        "answered_by": answerer,
        "claim_boundary": CLAIM_BOUNDARY,
        "confidence_source": confidence_source_for(answerer),
        "digest_verified": bool(digest_verified),
        "message_sha256": message_hash,
        "question_digest": digest,
        "recorded_at": recorded_at or _utc_now(),
        "route_choice": choice,
        "session_ref": _validated_session_ref(session_ref),
        **route_question_mode_fields(mode_reading),
    }
    if invalid:
        record["invalid_answer_reasons"] = list(invalid)
    validated_note = _validated_note(note)
    if validated_note:
        record["note"] = validated_note
    validated_ask_id = strip_control_characters(ask_id)[:32]
    if answerer == ANSWERED_BY_OMH_JEV_ASK and not validated_ask_id:
        raise RouteAnswerValidationError("answered_by omh_jev_ask needs the ask_id omh_jev_ask returned")
    if validated_ask_id:
        record["ask_id"] = validated_ask_id
    encoded = json.dumps(record, sort_keys=True)
    if len(encoded.encode("utf-8")) > MAX_ROUTE_ANSWER_RECORD_BYTES:
        raise RouteAnswerValidationError(
            f"route answer record is capped at {MAX_ROUTE_ANSWER_RECORD_BYTES} bytes"
        )
    return record


def _answer_row(
    *,
    digest: str,
    arm: str,
    choice: str,
    probabilities: dict[str, float],
    fits: dict[str, float],
    message_sha256: str,
) -> dict[str, Any]:
    """The embedded `routing_question_answers/v1` row.

    Embedded rather than referenced so the scorer reads this record with the
    reader it already has: the row names its own arm, its digest, and the
    request hash it answers for, and the record around it carries what the row
    has no field for.

    `message_sha256` is repeated on the row rather than left to the record
    because the row is what a scorer reading a JSONL answer file gets, and a
    row that travels out of its record has to identify its own request.
    """
    answers: dict[str, Any] = {ROUTE_CHOICE_KEY: {"choice": choice}}
    if probabilities:
        answers[ROUTE_CHOICE_KEY]["probabilities"] = probabilities
    for skill in sorted(fits):
        answers[f"{FIT_QUESTION_PREFIX}{skill}"] = {"noul": fits[skill]}
    return {
        "schema_version": ANSWER_ROW_SCHEMA_VERSION,
        "answers": answers,
        "arm": arm,
        "case_id": "",
        "message_sha256": message_sha256,
        "question_digest": digest,
    }


def route_answer_record_key(session_ref: object, question_digest: object) -> str:
    """The filename stem one answer lives under.

    Keyed on the session AND the question, not on the session alone: a session
    that reaches two undecidable routes answers two different questions, and a
    per-session filename would have the second overwrite the first. The slug
    is for a human reading the directory; the digest is what makes the name
    unique.
    """
    reference = strip_control_characters(session_ref)[:MAX_SESSION_REF_CHARS]
    digest = strip_control_characters(question_digest)[:MAX_DIGEST_CHARS]
    identity = hashlib.sha256(f"{reference}\x1f{digest}".encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", reference).strip("_-")[:48]
    return f"{slug}-{identity}" if slug else identity


def route_answer_dir(omh_home: Path) -> Path:
    return Path(omh_home) / "runtime" / ROUTE_ANSWER_DIRNAME


def route_answer_path(omh_home: Path, record: Mapping[str, Any]) -> Path:
    key = route_answer_record_key(record.get("session_ref", ""), record.get("question_digest", ""))
    return route_answer_dir(omh_home) / f"{key}.json"


def write_route_answer(omh_home: Path, record: dict[str, Any]) -> Path:
    """Write one validated record, under its own lock, and prune stale ones."""
    home = Path(omh_home)
    destination = route_answer_path(home, record)
    _reject_symlink_ancestry(destination, root=home)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RouteAnswerStoreError(f"route answer destination is not writable: {error}") from error
    # Post-mkdir recheck of the whole ancestry: the walk above ran before the
    # directory existed, so a link planted in between would otherwise be
    # followed by the write.
    _reject_symlink_ancestry(destination, root=home)
    _prune_stale_records(home, keep=destination)
    _refuse_a_full_directory(home, destination)
    with _record_lock(destination, root=home):
        _replace_record(destination, record)
    return destination


def _refuse_a_full_directory(omh_home: Path, destination: Path) -> None:
    """Refuse a NEW record once the directory is full; never evict for one.

    Age is the only thing that removes a record here, and that leaves no bound
    at all inside the seven-day window: the digest is a caller-chosen field, so
    a caller answering the same question with a different digest each time adds
    a file each time. A count-based eviction would solve it by discarding the
    measurement the record exists for, which is the trade the stale bound above
    already refuses. Refusing the write instead keeps every record that was
    accepted and tells the caller why the next one was not.

    Replacing a record that is already there is always allowed: it changes no
    count, and an answerer revising its own answer is ordinary.
    """
    if destination.exists():
        return
    try:
        existing = sum(1 for entry in os.scandir(route_answer_dir(omh_home))
                       if _RECORD_NAME.fullmatch(entry.name))
    except OSError:
        # The directory the write is about to create reads as empty, which it
        # is. A real read failure surfaces at the write, with its own error.
        return
    if existing >= MAX_ROUTE_ANSWER_RECORDS:
        raise RouteAnswerStoreError(
            f"route answer directory already holds {MAX_ROUTE_ANSWER_RECORDS} records "
            f"and nothing was written; score the recorded answers with "
            f"`omh chat route-questions score --answers <dir>` and clear them, or wait "
            f"for records older than {ROUTE_ANSWER_STALE_SECONDS // 86400} days to age out"
        )


def _replace_record(destination: Path, record: dict[str, Any]) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    except OSError as error:
        raise RouteAnswerStoreError(f"route answer destination is not writable: {error}") from error
    finally:
        # Suppressed because this cleanup sits OUTSIDE the handler above, so a
        # failing unlink left a bare OSError that was neither
        # RouteAnswerStoreError nor RouteAnswerContendedError and raised
        # straight out of the tool call. A leftover temp file is collected by
        # the prune below on the next write; a raised cleanup error is a tool
        # call that failed after the record was already in place.
        with contextlib.suppress(OSError):
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()


@contextlib.contextmanager
def _record_lock(destination: Path, *, root: Path) -> Iterator[None]:
    """Serialize writes to one record, with a deadline and its own vocabulary.

    The lock is `awareness_delivery`'s, the bundle's only one: a second copy
    here is what `tests/test_journal_lock_portability.py` refuses. `held`
    keeps the two handlers honest -- they sit outside the `with`, so a
    `TimeoutError` (which is an `OSError`) raised by the caller's body is not
    relabelled as this module deciding what someone else's failure was.
    """
    _reject_symlink_ancestry(destination.with_name(f".{destination.name}.lock"), root=root)
    held = False
    try:
        with _awareness_delivery_lock(destination, timeout_seconds=_LOCK_TIMEOUT_SECONDS):
            held = True
            yield
    except TimeoutError as error:
        if held:
            raise
        raise RouteAnswerContendedError(
            f"route answer record is held by another writer after "
            f"{_LOCK_TIMEOUT_SECONDS:g}s and was not written: {destination}. "
            "Nothing changed; send the same call again."
        ) from error
    except OSError as error:
        if held:
            raise
        raise RouteAnswerStoreError(f"route answer destination is not writable: {error}") from error


def _reject_symlink_ancestry(path: Path, *, root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise RouteAnswerStoreError(f"refusing symlinked route answer path: {current}")
        if current == root or current == current.parent:
            return
        current = current.parent


def _prune_stale_records(omh_home: Path, *, keep: Path) -> None:
    """Drop records older than the stale bound; best effort, never fatal.

    Only regular files this module names -- records, its own temporary files,
    and the lock files beside them -- directly inside the directory are
    considered, and the record just written is always kept. A lock file goes
    only once the record it guards is gone: past the stale bound with no
    record beside it nothing can be mid-write on it, since a writer creating
    a record holds a lock that is seconds old.
    """
    directory = route_answer_dir(omh_home)
    now = datetime.now(timezone.utc).timestamp()
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        if entry.name == keep.name:
            continue
        if _LOCK_NAME.fullmatch(entry.name):
            if (directory / entry.name[1:-len(".lock")]).exists():
                continue
        elif not (_RECORD_NAME.fullmatch(entry.name) or _TEMPORARY_NAME.fullmatch(entry.name)):
            continue
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            if now - entry.stat(follow_symlinks=False).st_mtime <= ROUTE_ANSWER_STALE_SECONDS:
                continue
            os.unlink(entry.path)
        except OSError:
            continue


def _validated_digest(value: object) -> str:
    digest = strip_control_characters(value)
    if not digest:
        raise RouteAnswerValidationError("question_digest is required")
    if len(digest) > MAX_DIGEST_CHARS or not re.fullmatch(r"[0-9a-f]+", digest):
        raise RouteAnswerValidationError(
            "question_digest must be the hex digest carried by the route question"
        )
    return digest


def _validated_message_sha256(value: object) -> str:
    """A sha256 hex digest of the request, or "" when none was supplied.

    Exactly 64 hex characters, not "at most": a shorter hex string is some
    other hash, and accepting it would let a record claim an identity that
    cannot be reproduced from the message. Empty is the one legal alternative
    and means the caller identified no request.
    """
    text = strip_control_characters(value)
    if not text:
        return ""
    if len(text) != SHA256_HEX_CHARS or not re.fullmatch(r"[0-9a-f]+", text):
        raise RouteAnswerValidationError(
            f"message_sha256 must be {SHA256_HEX_CHARS} lowercase hex characters "
            "or omitted entirely"
        )
    return text


def _validated_skill(value: object, *, field: str) -> str:
    name = strip_control_characters(value)
    if not name:
        raise RouteAnswerValidationError(f"{field} is required")
    if len(name) > MAX_SKILL_NAME_CHARS:
        raise RouteAnswerValidationError(
            f"{field} is capped at {MAX_SKILL_NAME_CHARS} characters"
        )
    return name


def _validated_fits(value: object) -> dict[str, float]:
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise RouteAnswerValidationError("fits must be an object of skill -> probability")
    if len(value) > MAX_FIT_ANSWERS:
        raise RouteAnswerValidationError(f"fits is capped at {MAX_FIT_ANSWERS} entries")
    fits: dict[str, float] = {}
    for skill, raw in value.items():
        name = _validated_skill(skill, field="fits key")
        fits[name] = _validated_probability(raw, field=f"fits[{name}]")
    return fits


def _validated_probabilities(value: object) -> dict[str, float]:
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise RouteAnswerValidationError(
            "choice_probabilities must be an object of option -> probability"
        )
    if len(value) > MAX_CHOICE_OPTIONS:
        raise RouteAnswerValidationError(
            f"choice_probabilities is capped at {MAX_CHOICE_OPTIONS} entries"
        )
    probabilities: dict[str, float] = {}
    for option, raw in value.items():
        name = _validated_skill(option, field="choice_probabilities key")
        probabilities[name] = _validated_probability(raw, field=f"choice_probabilities[{name}]")
    return probabilities


def _validated_probability(value: object, *, field: str) -> float:
    # `bool` is an `int`, and `True` would read as a probability of 1.0 that
    # nobody declared: a caller sending a yes/no where a number belongs is
    # answering a different question.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouteAnswerValidationError(f"{field} must be a number between 0 and 1")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise RouteAnswerValidationError(f"{field} must be between 0 and 1")
    return number


def _validated_note(value: object) -> str:
    note = strip_control_characters(value)
    if len(note) > MAX_NOTE_CHARS:
        raise RouteAnswerValidationError(f"note is capped at {MAX_NOTE_CHARS} characters")
    return note


def _validated_session_ref(value: object) -> str:
    return strip_control_characters(value)[:MAX_SESSION_REF_CHARS]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "ANSWER_VERDICT_ACCEPTED",
    "ANSWER_VERDICT_INVALID",
    "ANSWERED_BY_JEV_PLUGIN",
    "ANSWERED_BY_MAIN_MODEL",
    "ANSWERED_BY_OMH_JEV_ASK",
    "ANSWERED_BY_VALUES",
    "ANSWER_ROW_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "CLARIFY_ACTION",
    "CONFIDENCE_ANSWERER_DECLARED",
    "CONFIDENCE_OBSERVED_FROM_RESPONSE",
    "CONFIDENCE_SELF_REPORTED",
    "DISPATCH_ACTION",
    "FITS_CLARIFY_THRESHOLD",
    "FITS_DISPATCH_THRESHOLD",
    "MAX_NOTE_CHARS",
    "MAX_ROUTE_ANSWER_RECORDS",
    "NONE_ACTION",
    "ROUTE_ANSWER_SCHEMA_VERSION",
    "RouteAnswerContendedError",
    "RouteAnswerStoreError",
    "RouteAnswerValidationError",
    "build_route_answer_record",
    "confidence_source_for",
    "resolve_action",
    "route_answer_dir",
    "route_answer_path",
    "route_answer_record_key",
    "write_route_answer",
]

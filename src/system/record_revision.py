from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .hashutil import sha256_text
from .local_store import atomic_write_json, file_lock, read_json_object_result

RECORD_REVISION_KEY = "record_revision"
APPLIED_MUTATIONS_KEY = "applied_mutations"
APPLIED_MUTATIONS_FLOOR_KEY = "applied_mutations_floor_revision"
# The three bookkeeping keys a guarded record carries. Record validators that
# reject unsupported top-level keys splat this into their allowed key set.
REVISION_RECORD_KEYS = (RECORD_REVISION_KEY, APPLIED_MUTATIONS_KEY, APPLIED_MUTATIONS_FLOOR_KEY)
# Bounded so records cannot grow forever; only the most recent entries by
# record_revision are kept. 128 is sized against the observed worst case: one
# autonomous loop turn writes a handful of guarded mutations, so 128 covers a
# long session's retry window while capping the map at roughly 20KB of JSON
# (~150 bytes per entry). Eviction below the bound is not silent - it raises
# the persisted APPLIED_MUTATIONS_FLOOR_KEY so a retry that lands after its id
# was dropped is refused instead of applied twice.
APPLIED_MUTATIONS_LIMIT = 128
APPLIED_MUTATION_ENTRY_KEYS = ("operation", "record_revision", "result_digest")
# applied_mutations keys are "<operation>:<mutation_id>", so an operation name
# may not contain the separator.
OPERATION_SEPARATOR = ":"
# mutation_id is caller-supplied text that is persisted into the bounded map,
# so an unbounded id multiplies straight into the record: 128 retained entries
# of a 100k-character id is a ~13MB goal.json written by one buggy connector.
# 200 characters is sized against the ids real connectors actually send - a
# UUID is 36, a ULID 26, a Discord snowflake 20, a git sha 40, and a composite
# Slack reference like "slack:C0123456789/p1700000000.000100" is 36 - so the
# bound leaves roughly five times the widest observed id while capping the map
# at tens of kilobytes instead of megabytes.
MAX_MUTATION_ID_CHARS = 200
# operation is a code-chosen constant, never user text; the longest one in this
# repo is 31 characters. The bound exists so the same "one caller cannot
# inflate the key" property holds for both halves of the composite key.
MAX_OPERATION_CHARS = 64


class StaleRecordMutation(ValueError):
    """A mutation carried an expected_revision that no longer matches the record.

    The mutation is rejected without any partial write; the caller should
    re-read the record, show the user what changed, and retry against the
    current record_revision.
    """

    def __init__(self, *, expected_revision: int, current_revision: int, record_name: str = "record", action: str = "") -> None:
        self.expected_revision = int(expected_revision)
        self.current_revision = int(current_revision)
        refused = action or "mutation"
        super().__init__(
            f"stale {refused} rejected for {record_name}: expected record_revision "
            f"{self.expected_revision} but the record is now at record_revision "
            f"{self.current_revision}; the record changed since it was read - "
            "re-read it and retry against the current revision"
        )


class ConflictingMutationReplay(ValueError):
    """One (operation, mutation_id) pair was retried with a different intent.

    A replay is only safe when the retry means the same thing as the call that
    already applied. When the caller's mutation_digest does not match the
    digest stored with the applied entry the two calls are different work
    sharing one id, so replaying would silently swallow the second one.
    """

    def __init__(
        self,
        *,
        operation: str,
        mutation_id: str,
        stored_digest: str,
        incoming_digest: str,
        record_name: str = "record",
    ) -> None:
        self.operation = str(operation)
        self.mutation_id = str(mutation_id)
        self.stored_digest = str(stored_digest)
        self.incoming_digest = str(incoming_digest)
        super().__init__(
            f"conflicting replay rejected for {record_name}: operation "
            f"{self.operation!r} already applied mutation_id {self.mutation_id!r} "
            "with a different payload; a mutation_id may only be retried with "
            "the same arguments - re-read the record and retry with a new "
            "mutation_id for the new payload"
        )


class MutationHistoryEvicted(ValueError):
    """A retry arrived after its mutation_id was evicted from the bounded map.

    The record can no longer prove whether this mutation already applied, so
    applying it would risk a duplicate and replaying it would risk losing it.
    The caller has to re-render the current record and decide.

    This only fires when the caller supplied an expected_revision, and that is
    deliberate: without one there is nothing to compare against the floor, and
    refusing every id absent from the map would refuse every legitimately new
    mutation once any eviction has happened. A surface that materializes the
    mutation id into a persisted item id must therefore dedupe on that derived
    id itself - see ``guarded_record_update``.
    """

    def __init__(
        self,
        *,
        operation: str,
        mutation_id: str,
        expected_revision: int,
        floor_revision: int,
        current_revision: int,
        record_name: str = "record",
    ) -> None:
        self.operation = str(operation)
        self.mutation_id = str(mutation_id)
        self.expected_revision = int(expected_revision)
        self.floor_revision = int(floor_revision)
        self.current_revision = int(current_revision)
        super().__init__(
            f"unprovable retry rejected for {record_name}: operation "
            f"{self.operation!r} mutation_id {self.mutation_id!r} is not in the "
            f"applied history, and expected_revision {self.expected_revision} is "
            f"at or below applied_mutations_floor_revision {self.floor_revision}, "
            f"so ids from that revision are no longer retained (the record is at "
            f"record_revision {self.current_revision}); whether this mutation "
            "already applied cannot be proven - re-read the record and retry "
            "against the current revision"
        )


class GuardedRecord(dict):
    """The record a guarded update settled on, plus how strong the guard was.

    It is an ordinary dict for consumers, validators, and json serialization.
    The two attributes describe the advisory lock the transaction ran under and
    are deliberately never persisted into the record: a caller that needs to
    know the mutual-exclusion guarantee did not hold reads lock_enforced, and
    nothing about the lock leaks into the stored JSON.
    """

    __slots__ = ("lock_enforced", "lock_mechanism")

    def __init__(self, record: dict[str, Any], *, lock_enforced: bool, lock_mechanism: str) -> None:
        super().__init__(record)
        self.lock_enforced = bool(lock_enforced)
        self.lock_mechanism = str(lock_mechanism)


@dataclass(frozen=True)
class DuplicateMutationReplay:
    """Result of retrying an already-applied (operation, mutation_id) pair.

    This is a result path, not an error: the record is returned unchanged,
    no revision bump happens, and no duplicate event or item is created.
    ``replayed`` is always True so a consumer can surface "this was a retry"
    without re-deriving it from the type.
    """

    mutation_id: str
    record: dict[str, Any]
    record_revision: int
    result_digest: str
    outcome: Any = field(default=None)
    operation: str = ""
    replayed: bool = True
    lock_enforced: bool = True
    lock_mechanism: str = ""


def record_revision_of(record: dict[str, Any] | None) -> int:
    if not isinstance(record, dict):
        return 0
    value = record.get(RECORD_REVISION_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def applied_mutations_floor_of(record: dict[str, Any] | None) -> int:
    """The revision at or below which mutation ids are no longer retained."""
    if not isinstance(record, dict):
        return 0
    value = record.get(APPLIED_MUTATIONS_FLOOR_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def applied_mutation_key(operation: str, mutation_id: str) -> str:
    """The applied_mutations key for one mutation id under one operation.

    Scoping the key by operation is the point: the same client-side turn id
    reused by a different operation is different logical intent and must apply,
    not replay.
    """
    return f"{validated_operation(operation)}{OPERATION_SEPARATOR}{mutation_id}"


def validated_operation(operation: str) -> str:
    text = str(operation or "").strip()
    if not text:
        raise ValueError(
            "operation is required: it scopes mutation_id so one client id "
            "reused by a different operation is not swallowed as a replay"
        )
    if OPERATION_SEPARATOR in text:
        raise ValueError(f"operation must not contain {OPERATION_SEPARATOR!r}: {operation!r}")
    if len(text) > MAX_OPERATION_CHARS:
        raise ValueError(
            f"operation must be at most {MAX_OPERATION_CHARS} characters, got {len(text)}"
        )
    return text


def validated_mutation_id(mutation_id: str) -> str:
    """Bound one caller-supplied retry token before it is persisted.

    Called by every guarded update before the lock is taken, so an oversized
    id is refused with a readable message and no file - record, lock sidecar,
    or temp file - is touched. Every surface that accepts a mutation_id
    therefore rejects identically, which is the point: goal, wrapper-session,
    and loop writes must not each invent their own limit.
    """
    text = str(mutation_id)
    if len(text) > MAX_MUTATION_ID_CHARS:
        raise ValueError(
            f"mutation_id must be at most {MAX_MUTATION_ID_CHARS} characters, got {len(text)}; "
            "a mutation_id is a short retry token (a message id, uuid, or turn id), "
            "not a payload - it is persisted into the bounded applied_mutations map"
        )
    return text


def revision_field_errors(record: dict[str, Any], label: str = "record") -> list[str]:
    """Validation for the optional revision fields, shared by record validators."""
    errors: list[str] = []
    if RECORD_REVISION_KEY in record:
        value = record[RECORD_REVISION_KEY]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{label} {RECORD_REVISION_KEY} must be a non-negative integer")
    if APPLIED_MUTATIONS_FLOOR_KEY in record:
        floor = record[APPLIED_MUTATIONS_FLOOR_KEY]
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
            errors.append(f"{label} {APPLIED_MUTATIONS_FLOOR_KEY} must be a non-negative integer")
    if APPLIED_MUTATIONS_KEY in record:
        applied = record[APPLIED_MUTATIONS_KEY]
        if not isinstance(applied, dict):
            errors.append(f"{label} {APPLIED_MUTATIONS_KEY} must be an object")
            return errors
        if len(applied) > APPLIED_MUTATIONS_LIMIT:
            errors.append(f"{label} {APPLIED_MUTATIONS_KEY} must keep at most {APPLIED_MUTATIONS_LIMIT} entries")
        for mutation_key, entry in applied.items():
            if not str(mutation_key).strip():
                errors.append(f"{label} {APPLIED_MUTATIONS_KEY} keys must be non-empty mutation ids")
                continue
            if not isinstance(entry, dict):
                errors.append(f"{label} {APPLIED_MUTATIONS_KEY}[{mutation_key}] must be an object")
                continue
            extra = sorted(set(entry) - set(APPLIED_MUTATION_ENTRY_KEYS))
            if extra:
                errors.append(f"{label} {APPLIED_MUTATIONS_KEY}[{mutation_key}] has unsupported keys: {extra}")
            revision = entry.get("record_revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                errors.append(f"{label} {APPLIED_MUTATIONS_KEY}[{mutation_key}].record_revision must be a positive integer")
            if not isinstance(entry.get("result_digest"), str):
                errors.append(f"{label} {APPLIED_MUTATIONS_KEY}[{mutation_key}].result_digest must be a string")
            # operation is optional so records written before operation
            # scoping - and projections that drop it, since the key already
            # carries it - stay valid instead of failing every later write.
            if "operation" in entry:
                operation = entry["operation"]
                if not isinstance(operation, str) or not operation.strip():
                    errors.append(f"{label} {APPLIED_MUTATIONS_KEY}[{mutation_key}].operation must be a non-empty string")
    return errors


def bounded_applied_mutations_with_floor(
    applied: dict[str, Any],
    floor_revision: int = 0,
) -> tuple[dict[str, Any], int]:
    """Bound the applied map and report the revision the eviction floor moved to.

    The floor is the highest record_revision whose mutation id was dropped, so
    "id absent and expected_revision <= floor" means the retry cannot be proven
    new. It only ever moves forward.
    """
    entries = [(key, entry) for key, entry in applied.items() if isinstance(entry, dict)]
    floor = max(int(floor_revision), 0)
    if len(entries) <= APPLIED_MUTATIONS_LIMIT:
        return dict(entries), floor
    entries.sort(key=lambda pair: (_entry_revision(pair[1]), str(pair[0])), reverse=True)
    kept = entries[:APPLIED_MUTATIONS_LIMIT]
    for _, entry in entries[APPLIED_MUTATIONS_LIMIT:]:
        floor = max(floor, _entry_revision(entry))
    return dict(sorted(kept, key=lambda pair: (_entry_revision(pair[1]), str(pair[0])))), floor


def bounded_applied_mutations(applied: dict[str, Any]) -> dict[str, Any]:
    """Keep only the most recent APPLIED_MUTATIONS_LIMIT entries by revision."""
    bounded, _ = bounded_applied_mutations_with_floor(applied)
    return bounded


def _entry_revision(entry: dict[str, Any]) -> int:
    value = entry.get("record_revision", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _result_digest(record: dict[str, Any]) -> str:
    projected = {key: value for key, value in record.items() if key not in (APPLIED_MUTATIONS_KEY, APPLIED_MUTATIONS_FLOOR_KEY)}
    return sha256_text(json.dumps(projected, sort_keys=True, separators=(",", ":"), default=str))


def _reject_divergent_replay(
    *,
    entry: dict[str, Any],
    operation: str,
    mutation_id: str,
    mutation_digest: str | None,
    record_name: str,
) -> None:
    if mutation_digest is None:
        return
    stored = str(entry.get("result_digest", ""))
    if stored == str(mutation_digest):
        return
    raise ConflictingMutationReplay(
        operation=operation,
        mutation_id=mutation_id,
        stored_digest=stored,
        incoming_digest=str(mutation_digest),
        record_name=record_name,
    )


def guarded_record_update(
    path: Path,
    *,
    mutate: Callable[[dict[str, Any]], dict[str, Any] | None],
    operation: str,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    mutation_digest: str | None = None,
    lock_name: str,
    validate: Callable[[dict[str, Any]], None] | None = None,
    on_replay: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
    default: dict[str, Any] | None = None,
    private: bool = True,
    timeout_seconds: float = 10.0,
) -> GuardedRecord | DuplicateMutationReplay:
    """One locked read-modify-write transaction for a shared JSON record.

    ``operation`` is a required short stable name for the logical mutation
    ("record_goal_checkpoint", "record_plan_decision"). It scopes mutation_id:
    applied entries are keyed ``"<operation>:<mutation_id>"``, so one client
    turn id reused by a different operation is different intent and applies
    normally instead of being swallowed as a replay.

    Inside a single advisory file lock (derived from lock_name, so several
    record files in one directory can share a lock when needed) this:

    1. reads the current record (default, when given, seeds a missing file;
       otherwise a missing file raises FileNotFoundError),
    2. returns a DuplicateMutationReplay - without any write or revision
       bump - when this (operation, mutation_id) already applied. When the
       caller supplies mutation_digest and it does not match the digest stored
       with that entry the retry is a different payload under one id and
       raises ConflictingMutationReplay instead of replaying,
    3. raises MutationHistoryEvicted - without any write - when the id is
       absent, an expected_revision was supplied, and that revision is at or
       below applied_mutations_floor_revision, because the record can no
       longer prove the retry is new,
    4. raises StaleRecordMutation - without any write - when
       expected_revision is given and does not match the current
       record_revision,
    5. applies mutate(current); a None result means "no change" and skips
       the write and the revision bump entirely,
    6. bumps record_revision (first write initializes it to 1), records the
       applied mutation with a bounded history, carries the eviction floor,
       runs validate on the final record, and atomically writes it while
       still holding the lock.

    The replay check runs before the staleness check on purpose: a retry that
    carries both its original expected_revision and its mutation_id has to
    replay, not be reported as stale, or every retry after any other write
    would look like a conflict.

    ``mutation_digest`` must be used consistently within one operation. It is
    stored as the entry's result_digest when supplied (otherwise the digest of
    the written record is stored), so a retry that supplies a digest where the
    original call did not is treated as divergent rather than replayed - the
    conservative direction, since the alternative silently drops work.

    ``mutation_id`` is bounded at MAX_MUTATION_ID_CHARS and ``operation`` at
    MAX_OPERATION_CHARS. Both are checked before the lock is taken, so an
    oversized id is refused without creating the record, the lock sidecar, or
    a temp file.

    A surface that MATERIALIZES mutation_id into a persisted item id - the way
    the goal ledger derives checkpoint_id / blocker_id / quality_gate_id from
    it - MUST additionally dedupe on that derived id inside its own ``mutate``
    before appending. The bounded map alone cannot prove a retry after
    eviction: once the id has been dropped, step 2 finds nothing and step 3
    only fires when an expected_revision was supplied, so a retry carrying a
    mutation_id and nothing else applies a second time and appends a duplicate
    item. Widening the floor rule is not the fix - it would refuse every
    legitimately new mutation_id once any eviction has happened. Scanning the
    target list for the derived id is exact, survives eviction, and costs one
    pass. See "Materialized mutation ids" in docs/ARCHITECTURE.md.

    The returned GuardedRecord is a plain dict carrying ``lock_enforced``: on a
    platform with neither fcntl nor msvcrt no OS lock exists, the transaction
    is protected only by the revision compare and the atomic replace, and
    lock_enforced is False so callers can say so rather than claim a guarantee
    that did not hold.
    """
    operation = validated_operation(operation)
    if mutation_id is not None:
        mutation_id = validated_mutation_id(mutation_id)
    lock_path = path.with_name(lock_name)
    with file_lock(lock_path, timeout_seconds=timeout_seconds, private=private) as lock:
        lock_enforced = bool(lock.get("enforced", False))
        lock_mechanism = str(lock.get("mechanism", ""))
        current, read_error = read_json_object_result(path)
        if current is None:
            if read_error is not None and path.exists():
                raise ValueError(f"{path}: {read_error}")
            if default is None:
                raise FileNotFoundError(path)
            current = dict(default)
        current_revision = record_revision_of(current)
        floor_revision = applied_mutations_floor_of(current)
        applied_raw = current.get(APPLIED_MUTATIONS_KEY)
        applied = dict(applied_raw) if isinstance(applied_raw, dict) else {}
        entry_key = applied_mutation_key(operation, mutation_id) if mutation_id else ""
        entry = applied.get(entry_key) if entry_key else None
        if isinstance(entry, dict):
            _reject_divergent_replay(
                entry=entry,
                operation=operation,
                mutation_id=str(mutation_id),
                mutation_digest=mutation_digest,
                record_name=path.name,
            )
            outcome = on_replay(current, entry) if on_replay is not None else None
            return DuplicateMutationReplay(
                mutation_id=str(mutation_id),
                record=current,
                record_revision=_entry_revision(entry),
                result_digest=str(entry.get("result_digest", "")),
                outcome=outcome,
                operation=operation,
                lock_enforced=lock_enforced,
                lock_mechanism=lock_mechanism,
            )
        if mutation_id and expected_revision is not None and floor_revision >= 1 and int(expected_revision) <= floor_revision:
            raise MutationHistoryEvicted(
                operation=operation,
                mutation_id=str(mutation_id),
                expected_revision=int(expected_revision),
                floor_revision=floor_revision,
                current_revision=current_revision,
                record_name=path.name,
            )
        if expected_revision is not None and int(expected_revision) != current_revision:
            raise StaleRecordMutation(
                expected_revision=int(expected_revision),
                current_revision=current_revision,
                record_name=path.name,
            )
        updated = mutate(current)
        if updated is None:
            return GuardedRecord(current, lock_enforced=lock_enforced, lock_mechanism=lock_mechanism)
        updated[RECORD_REVISION_KEY] = current_revision + 1
        if mutation_id:
            applied[entry_key] = {
                "operation": operation,
                "record_revision": current_revision + 1,
                "result_digest": str(mutation_digest) if mutation_digest is not None else _result_digest(updated),
            }
            applied, floor_revision = bounded_applied_mutations_with_floor(applied, floor_revision)
        updated[APPLIED_MUTATIONS_KEY] = applied
        # Written only once eviction has actually happened, so records that
        # never reach the bound keep their existing shape.
        if floor_revision >= 1:
            updated[APPLIED_MUTATIONS_FLOOR_KEY] = floor_revision
        if validate is not None:
            validate(updated)
        atomic_write_json(path, updated, private=private)
        return GuardedRecord(updated, lock_enforced=lock_enforced, lock_mechanism=lock_mechanism)


def require_not_terminal(
    record: dict[str, Any],
    status_key: str,
    terminal_statuses: tuple[str, ...],
    action: str,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Refuse preparing child work or mutations on a terminally closed record."""
    status = str(record.get(status_key, ""))
    if status in terminal_statuses:
        raise error_type(
            f"cannot {action}: {status_key} is {status}, a terminal state; "
            "terminal records refuse new child work and further mutations"
        )

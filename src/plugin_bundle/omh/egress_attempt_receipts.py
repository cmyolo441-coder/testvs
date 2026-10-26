"""SQLite-backed durable write-ahead attempt receipts.

The database stores metadata hashes only. It is deliberately independent from
Hermes registration: a wrapper calls ``open_attempt`` before invoking
its preserved handler, and treats every non-``created`` disposition as a reason
not to invoke that handler.

Manual proof recipes, against an isolated ``OMH_HOME``:

    sqlite3 "$OMH_HOME/runtime/journal/external_effect_attempts.sqlite3" \
      "EXPLAIN QUERY PLAN SELECT attempt_id FROM egress_attempts \
       WHERE session_ref='<sha256>' AND tool_call_ref='<sha256>' \
       AND request_fingerprint='<sha256>';"

The plan must report ``SEARCH ... USING INDEX`` rather than ``SCAN``. To check
the crash boundary, kill a process after ``open_attempt`` returns and before
``record_terminal``; reopening the same identity returns the stable attempt id
with ``already_recorded`` and no terminal state, never a new attempt.

Public digest references encode all 256 SHA-256 bits as unpadded base64url.
Caller-supplied digests use hexadecimal; encoding does not truncate them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from base64 import urlsafe_b64encode
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "external_effect_attempt/v1"
ATTEMPT_DATABASE_NAME = "external_effect_attempts.sqlite3"
ROW_LIMIT_BYTES = 1024
ATTEMPT_CLAIM_BOUNDARY = "durable local pre-handler attempt; not remote delivery"
_TERMINAL_STATES = frozenset({"blocked", "cancelled", "error", "returned", "unknown"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_DIGEST = re.compile(r"^[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]$")
_SAFE_TOKEN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ACTION_CLASSES = frozenset({"message_send", "review_submit", "ci_dispatch", "merge", "external_write"})
_DESTINATION_CLASSES = frozenset({"chat_channel", "repository", "review_thread", "workflow", "endpoint"})


class AttemptStoreError(RuntimeError):
    """The receipt store could not safely provide write-ahead evidence."""


def _sha256(value: str) -> str:
    return urlsafe_b64encode(hashlib.sha256(value.encode("utf-8")).digest()).rstrip(b"=").decode("ascii")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _fsync_parent_directory(path: Path) -> None:
    """Add a POSIX directory flush after first database creation."""
    if os.name == "nt":
        # Windows CRT cannot open directories. SQLite's FULL commit already
        # uses its native sync; this adds no separate directory guarantee there.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AttemptStore:
    """One OMH home's append-only attempt and terminal receipt store.

    Construction is pure: validation happens before any directory or SQLite
    operation, so malformed metadata cannot create an empty state tree.
    """

    def __init__(self, omh_home: str | Path) -> None:
        self.omh_home = Path(omh_home).expanduser()
        self.database_path = (
            self.omh_home / "runtime" / "journal" / ATTEMPT_DATABASE_NAME
        )

    def open_attempt(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        action_class: str,
        destination_class: str,
        request_fingerprint: str,
        effect_id: str,
        destination_digest: str,
        payload_digest: str,
        payload_bytes: int,
        idempotency_key: str | None = None,
        approval_ref: str | None = None,
    ) -> dict[str, object]:
        """Create exactly one durable attempt for a host call identity.

        A same-identity replay always returns ``already_recorded``. A call-id
        reused with a different final request is refused. Neither outcome can
        be interpreted by a wrapper as permission to invoke the handler again.
        """
        values = {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "action_class": action_class,
            "destination_class": destination_class,
            "request_fingerprint": request_fingerprint,
            "effect_id": effect_id,
            "destination_digest": destination_digest,
            "payload_digest": payload_digest,
            "payload_bytes": payload_bytes,
            "idempotency_key": idempotency_key,
            "approval_ref": approval_ref,
        }
        self._validate_attempt_input(values)

        session_ref = _sha256(session_id)
        tool_call_ref = _sha256(tool_call_id)
        attempt_id = _sha256(
            json.dumps(
                {
                    "session_ref": session_ref,
                    "tool_call_ref": tool_call_ref,
                    "request_fingerprint": request_fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        row: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "row_type": "attempt",
            "attempt_id": attempt_id,
            "session_ref": session_ref,
            "tool_call_ref": tool_call_ref,
            "tool_name": tool_name,
            "action_class": action_class,
            "destination_class": destination_class,
            "request_fingerprint": request_fingerprint,
            "effect_id": effect_id,
            "destination_digest": destination_digest,
            "payload_digest": payload_digest,
            "payload_bytes": payload_bytes,
            "idempotency_key": idempotency_key,
            "approval_ref": approval_ref,
            "created_at": _utc_now(),
            "privacy": "metadata_only",
            "claim_boundary": ATTEMPT_CLAIM_BOUNDARY,
        }
        for field in ("request_fingerprint", "effect_id", "destination_digest", "payload_digest"):
            row[field] = urlsafe_b64encode(bytes.fromhex(str(row[field]))).rstrip(b"=").decode("ascii")
        for field in ("idempotency_key", "approval_ref"):
            if row[field] is not None:
                row[field] = urlsafe_b64encode(bytes.fromhex(str(row[field]))).rstrip(b"=").decode("ascii")
        self._validate_row(row)

        connection, created = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT attempt_id, request_fingerprint
                FROM egress_attempts
                WHERE session_ref = ? AND tool_call_ref = ?
                """,
                (session_ref, tool_call_ref),
            ).fetchone()
            if existing is not None:
                if str(existing[1]) != request_fingerprint:
                    raise AttemptStoreError(
                        "tool_call_id was reused with different request fingerprint"
                    )
                terminal = connection.execute(
                    "SELECT terminal_state FROM egress_attempt_terminals WHERE attempt_id = ?",
                    (str(existing[0]),),
                ).fetchone()
                connection.rollback()
                return {
                    "attempt_id": str(existing[0]),
                    "disposition": "already_recorded",
                    "terminal_state": str(terminal[0]) if terminal else "",
                }

            connection.execute(
                """
                INSERT INTO egress_attempts (
                    attempt_id, session_ref, tool_call_ref, request_fingerprint,
                    created_at, row_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    session_ref,
                    tool_call_ref,
                    request_fingerprint,
                    row["created_at"],
                    self._encode_row(row),
                ),
            )
            connection.commit()
        except AttemptStoreError:
            self._rollback(connection)
            raise
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as exc:
            self._rollback(connection)
            raise AttemptStoreError(f"egress attempt storage failed: {exc}") from exc
        finally:
            connection.close()

        if created:
            try:
                _fsync_parent_directory(self.database_path.parent)
            except OSError as exc:
                # SQLite committed the row. Reporting failure is intentional:
                # callers must block rather than execute after a durability
                # uncertainty; a replay receives the stable stored identity.
                raise AttemptStoreError(
                    f"egress attempt directory durability failed: {exc}"
                ) from exc
        return {"attempt_id": attempt_id, "disposition": "created", "terminal_state": ""}

    def record_terminal(self, attempt_id: str, terminal_state: str) -> dict[str, object]:
        """Atomically append one terminal observation for a durable attempt."""
        if not isinstance(attempt_id, str) or not _PUBLIC_DIGEST.fullmatch(attempt_id):
            raise AttemptStoreError("attempt_id must be a canonical SHA-256 reference")
        if terminal_state not in _TERMINAL_STATES:
            raise AttemptStoreError("terminal_state is unsupported")

        connection, _ = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM egress_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone() is None:
                raise AttemptStoreError("attempt_id is not durable")
            existing = connection.execute(
                "SELECT terminal_state FROM egress_attempt_terminals WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != terminal_state:
                    raise AttemptStoreError("terminal observation conflicts with durable terminal state")
                connection.rollback()
                return {
                    "attempt_id": attempt_id,
                    "terminal_state": terminal_state,
                    "already_recorded": True,
                }
            row = {
                "schema_version": SCHEMA_VERSION,
                "row_type": "terminal",
                "attempt_id": attempt_id,
                "terminal_state": terminal_state,
                "terminal_observed_at": _utc_now(),
                "privacy": "metadata_only",
                "claim_boundary": ATTEMPT_CLAIM_BOUNDARY,
            }
            self._validate_row(row)
            connection.execute(
                """
                INSERT INTO egress_attempt_terminals (
                    attempt_id, terminal_state, terminal_observed_at, row_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    terminal_state,
                    row["terminal_observed_at"],
                    self._encode_row(row),
                ),
            )
            connection.commit()
        except AttemptStoreError:
            self._rollback(connection)
            raise
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as exc:
            self._rollback(connection)
            raise AttemptStoreError(f"egress terminal storage failed: {exc}") from exc
        finally:
            connection.close()
        return {
            "attempt_id": attempt_id,
            "terminal_state": terminal_state,
            "already_recorded": False,
        }

    def public_rows(self, *, limit: int | None = None) -> list[dict[str, object]]:
        """Read a bounded attempt-indexed metadata projection."""
        if not self.database_path.exists():
            return []
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 200):
            raise AttemptStoreError("limit must be between zero and 200")
        connection, _ = self._connect()
        try:
            suffix = "" if limit is None else " LIMIT ?"
            parameters: tuple[object, ...] = () if limit is None else (limit,)
            rows = connection.execute(
                "SELECT attempt.row_json, terminal.row_json "
                "FROM egress_attempts AS attempt "
                "LEFT JOIN egress_attempt_terminals AS terminal "
                "ON terminal.attempt_id = attempt.attempt_id "
                "ORDER BY attempt.created_at DESC" + suffix,
                parameters,
            ).fetchall()
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as exc:
            raise AttemptStoreError(f"egress attempt storage failed: {exc}") from exc
        finally:
            connection.close()
        decoded: list[dict[str, object]] = []
        for attempt_json, terminal_json in rows:
            for serialized in (attempt_json, terminal_json):
                if serialized is None:
                    continue
                try:
                    row = json.loads(str(serialized))
                except ValueError as exc:
                    raise AttemptStoreError(f"egress attempt row is corrupt: {exc}") from exc
                if not isinstance(row, dict):
                    raise AttemptStoreError("egress attempt row is corrupt")
                self._validate_row(row)
                decoded.append(row)
        return decoded

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        created = not self.database_path.exists()
        connection: sqlite3.Connection | None = None
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            # connect installs SQLite's busy handler; do not repeat it via SQL.
            # Fail closed sooner under contention: native sleeps can overshoot
            # this retry budget, which is not a whole-operation deadline.
            connection = sqlite3.connect(
                self.database_path,
                timeout=0.01,
                isolation_level=None,
            )
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            # Publish the schema atomically with one FULL commit, not one per DDL.
            # Deferred BEGIN keeps existing-schema reads free of writer locks.
            connection.execute("BEGIN")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS egress_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    session_ref TEXT NOT NULL,
                    tool_call_ref TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    UNIQUE(session_ref, tool_call_ref)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS egress_attempts_identity_lookup
                ON egress_attempts (session_ref, tool_call_ref, request_fingerprint)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS egress_attempts_recent_lookup
                ON egress_attempts (created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS egress_attempt_terminals (
                    attempt_id TEXT PRIMARY KEY REFERENCES egress_attempts(attempt_id),
                    terminal_state TEXT NOT NULL,
                    terminal_observed_at TEXT NOT NULL,
                    row_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
            return connection, created
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as exc:
            if connection is not None:
                # Closing rolls back any uncommitted schema transaction.
                connection.close()
            raise AttemptStoreError(f"egress attempt storage failed: {exc}") from exc

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.rollback()
        except sqlite3.DatabaseError:
            return

    @staticmethod
    def _encode_row(row: Mapping[str, object]) -> str:
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > ROW_LIMIT_BYTES:
            raise AttemptStoreError("egress attempt row exceeds 1 KiB")
        return encoded

    @staticmethod
    def _validate_attempt_input(values: Mapping[str, object]) -> None:
        for field in ("session_id", "tool_call_id"):
            value = values[field]
            if not isinstance(value, str) or not value or len(value) > 256:
                raise AttemptStoreError(f"{field} is required")
        for field in ("tool_name", "action_class", "destination_class"):
            value = values[field]
            if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
                raise AttemptStoreError(f"{field} must be a bounded closed-vocabulary token")
        if values["action_class"] not in _ACTION_CLASSES:
            raise AttemptStoreError("action_class is unsupported")
        if values["destination_class"] not in _DESTINATION_CLASSES:
            raise AttemptStoreError("destination_class is unsupported")
        for field in ("request_fingerprint", "effect_id", "destination_digest", "payload_digest"):
            value = values[field]
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise AttemptStoreError(f"{field} must be a SHA-256 hex digest")
        for field in ("idempotency_key", "approval_ref"):
            value = values[field]
            if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
                raise AttemptStoreError(f"{field} must be a SHA-256 hex digest or null")
        payload_bytes = values["payload_bytes"]
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or not 0 <= payload_bytes < 2**63:
            raise AttemptStoreError("payload_bytes must be a non-negative integer")

    @staticmethod
    def _validate_row(row: Mapping[str, object]) -> None:
        row_type = row.get("row_type")
        if row_type == "attempt":
            expected = {
                "schema_version", "row_type", "attempt_id", "session_ref", "tool_call_ref",
                "tool_name", "action_class", "destination_class", "request_fingerprint",
                "effect_id", "destination_digest", "payload_digest", "payload_bytes",
                "idempotency_key", "approval_ref", "created_at", "privacy", "claim_boundary",
            }
        elif row_type == "terminal":
            expected = {
                "schema_version", "row_type", "attempt_id", "terminal_state",
                "terminal_observed_at", "privacy", "claim_boundary",
            }
        else:
            raise AttemptStoreError("egress attempt row_type is unsupported")
        if set(row) != expected:
            raise AttemptStoreError("egress attempt row has unsupported or missing fields")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise AttemptStoreError("egress attempt schema_version is unsupported")
        if row.get("privacy") != "metadata_only" or row.get("claim_boundary") != ATTEMPT_CLAIM_BOUNDARY:
            raise AttemptStoreError("egress attempt privacy boundary is invalid")
        if not isinstance(row.get("attempt_id"), str) or not _PUBLIC_DIGEST.fullmatch(str(row["attempt_id"])):
            raise AttemptStoreError("egress attempt_id is invalid")
        if row_type == "attempt":
            for field in (
                "session_ref", "tool_call_ref", "request_fingerprint", "effect_id",
                "destination_digest", "payload_digest",
            ):
                if not isinstance(row.get(field), str) or not _PUBLIC_DIGEST.fullmatch(str(row[field])):
                    raise AttemptStoreError(f"egress attempt {field} is invalid")
            for field in ("idempotency_key", "approval_ref"):
                if row.get(field) is not None and (
                    not isinstance(row.get(field), str)
                    or not _PUBLIC_DIGEST.fullmatch(str(row[field]))
                ):
                    raise AttemptStoreError(f"egress attempt {field} is invalid")
            payload_bytes = row.get("payload_bytes")
            if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or not 0 <= payload_bytes < 2**63:
                raise AttemptStoreError("egress attempt payload_bytes is invalid")
            tool_name = row.get("tool_name")
            if not isinstance(tool_name, str) or not _SAFE_TOKEN.fullmatch(tool_name):
                raise AttemptStoreError("egress attempt tool_name is invalid")
            if row.get("action_class") not in _ACTION_CLASSES or row.get("destination_class") not in _DESTINATION_CLASSES:
                raise AttemptStoreError("egress attempt classification is invalid")
        elif row.get("terminal_state") not in _TERMINAL_STATES:
            raise AttemptStoreError("egress attempt terminal_state is invalid")
        timestamp = row.get("created_at" if row_type == "attempt" else "terminal_observed_at")
        if not isinstance(timestamp, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", timestamp
        ):
            raise AttemptStoreError("egress attempt timestamp is invalid")
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise AttemptStoreError("egress attempt timestamp is invalid") from exc
        if len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()) > ROW_LIMIT_BYTES:
            raise AttemptStoreError("egress attempt row exceeds 1 KiB")

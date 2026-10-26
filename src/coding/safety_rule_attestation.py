"""Local attestation and retention for the opt-in org safety rule source (#805).

Be exact about what this is, because the difference decides whether a reviewer
is right to trust it.

WHAT IT IS. An HMAC-SHA256 tag over the raw bytes of the org safety rule
source, keyed by a secret the operator holds on this machine. A tag that
verifies says the bytes at that path are the bytes someone holding that key
attested, and that nobody without the key has altered them since.

WHAT IT IS NOT. It is not public-key signing, and nothing here calls it
"signed". HMAC is symmetric: the key that verifies a tag also produces one, so
anyone who can read the key file can forge a tag that verifies. It therefore
carries no third-party provenance -- it cannot say WHO wrote a rule set, only
that the bytes match a tag made with the local key. "Locally attested" is the
honest phrase, and the vocabulary, the claim boundary, and the reason codes all
use it.

Why not the stronger thing: every scheme that would prove provenance instead of
integrity is asymmetric (Ed25519, minisign, GPG), and every one of those is a
dependency. This repository takes none, and `hmac`/`hashlib` are stdlib. That
trade, not an oversight, is why this lane stops at integrity against a local
key.

TWO AXES, KEPT APART. `signature_state` answers "did the local tag verify".
The org source `status` answers "is this a rule document OMH can read". Neither
is folded into the other: a valid tag over a malformed document is still
unavailable for the reader's own reason, and an invalid tag over a perfectly
well-formed document is still a refusal, with its own code and its own field.
Collapsing them would make one of the two questions unanswerable.

RETENTION. When a key is configured and verification fails, the candidate
revision does not activate and nothing about it is applied -- not its rules, not
its revision, not its digest. The last revision that DID verify stays in force,
is reported as retained, and the failing revision is named alongside it. The
retained set comes from a small metadata-only trust file inside the OMH home,
written atomically and written only after a verification that passed, so "the
last valid set" means exactly that rather than "the previous one".

OPT-IN. With no key path configured the state is `not_required`, no key or tag
file is opened, no trust file is read or written, and the org source result is
byte-for-byte what it was before this module existed.

No network, no model, no new dependency, and no wall clock: the timestamp
recorded in the trust file is supplied by the caller.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from ..system.local_store import atomic_write_json, read_json_object_result
from .project_governance import (
    ORG_SAFETY_RULE_SOURCE_TIME_BUDGET_SECONDS,
    org_safety_rule_source_available,
    org_safety_rule_source_unavailable,
    read_org_safety_rule_source_with_bytes,
)

SAFETY_RULE_ATTESTATION_SCHEMA_VERSION = "omh_safety_rule_attestation/v1"
SAFETY_RULE_TRUST_STATE_SCHEMA_VERSION = "omh_safety_rule_trust_state/v1"
ORG_SAFETY_RULE_ACTIVATION_SCHEMA_VERSION = "omh_org_safety_rule_activation/v1"

# Named as the algorithm rather than as "signature" so a reader of the payload
# cannot mistake the guarantee. `hmac-sha256` is a keyed digest, not a signature.
SAFETY_RULE_ATTESTATION_ALGORITHM = "hmac-sha256"

SAFETY_RULE_ATTESTATION_CLAIM_BOUNDARY = (
    "A local attestation is a keyed integrity check against an operator-held shared key: it shows "
    "the bytes were not altered by anyone without that key, not who wrote them, and anyone who can "
    "read the key can produce a tag that verifies. It is not compliance, execution, review, CI, or "
    "merge evidence."
)

# The attestation axis. `not_required` is the unconfigured default; `valid` is
# the only state that may activate a new revision. The other three are distinct
# because they need distinct operator actions: write a tag, re-attest the bytes,
# or fix the key file.
SIGNATURE_STATE_NOT_REQUIRED = "not_required"
SIGNATURE_STATE_VALID = "valid"
SIGNATURE_STATE_MISSING = "missing"
SIGNATURE_STATE_INVALID = "invalid"
SIGNATURE_STATE_UNVERIFIABLE = "unverifiable"
SIGNATURE_STATES = (
    SIGNATURE_STATE_NOT_REQUIRED,
    SIGNATURE_STATE_VALID,
    SIGNATURE_STATE_MISSING,
    SIGNATURE_STATE_INVALID,
    SIGNATURE_STATE_UNVERIFIABLE,
)
# Only these two put a rule set into force. Written as a set rather than as an
# inline `or` so a fourth state can never be added without deciding this.
ACTIVATING_SIGNATURE_STATES = frozenset({SIGNATURE_STATE_NOT_REQUIRED, SIGNATURE_STATE_VALID})

SAFETY_RULE_ATTESTATION_REASON_CODES = (
    "attestation_not_required",
    "attestation_valid",
    "attestation_tag_missing",
    "attestation_tag_invalid",
    "attestation_key_unreadable",
)

ORG_SAFETY_RULE_ACTIVATION_STATUSES = ("activated", "retained", "unavailable")

ATTESTATION_KEY_FIELD = "org_rule_source.attestation_key_path"
ATTESTATION_TAG_FIELD = "org_rule_source.attestation_tag_path"

# The tag sits beside the source it covers. One derived location rather than a
# second configured path: a tag that can be pointed somewhere else is a tag that
# can be pointed at a stale file.
ATTESTATION_TAG_SUFFIX = ".hmac-sha256"

# A key long enough to be a key and short enough that reading it is bounded. The
# tag bound is generous for a 64-character hex digest so a trailing newline, a
# CRLF, or a stray blank line does not read as a truncated file.
ATTESTATION_KEY_MAX_BYTES = 4_096
ATTESTATION_TAG_MAX_BYTES = 256

_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

# The retained rule bounds, restated here rather than imported. The reader's
# bounds apply to a document being admitted; these apply to a record OMH itself
# wrote and is now reading back, and the record must stand on its own even if a
# later revision widens what a document may declare.
_TRUST_STATE_MAX_KINDS = 16
_TRUST_STATE_MAX_KIND_CHARS = 120
_TRUST_STATE_MAX_TARGET_PATHS = 1_024

_TRUST_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "source_identity_sha256",
        "content_sha256",
        "revision",
        "denied_remote_target_kinds",
        "max_target_paths",
        "attested_at",
        "claim_boundary",
    }
)

# Which refusal the org source reports for each failed attestation state. The
# mapping is total over the three failure states, so a new state cannot silently
# fall through to "available".
_ATTESTATION_REFUSAL_CODES = {
    SIGNATURE_STATE_MISSING: "org_source_attestation_missing",
    SIGNATURE_STATE_INVALID: "org_source_attestation_invalid",
    SIGNATURE_STATE_UNVERIFIABLE: "org_source_attestation_key_unreadable",
}


def default_attestation_tag_path(source_path: str | Path) -> Path | None:
    """The sidecar tag path for a source path, or None when there is no source."""
    text = str(source_path).strip()
    if not text:
        return None
    path = Path(text)
    return path.parent / f"{path.name}{ATTESTATION_TAG_SUFFIX}"


def verify_safety_rule_attestation(
    source_bytes: bytes,
    *,
    key_path: str | Path = "",
    tag_path: str | Path = "",
) -> dict[str, object]:
    """Verify the local HMAC-SHA256 tag over `source_bytes`. Never decides safety.

    Returns the attestation axis alone: a state, the reason code for that state,
    and the offending field. It says nothing about whether the bytes parse as a
    rule document, and a caller must not read `valid` as permission.

    A blank `key_path` is the unconfigured default and returns `not_required`
    without opening anything. A configured key that cannot be read is
    `unverifiable` rather than `missing`, because the operator asked for a check
    OMH could not perform -- which must not activate anything either.

    A tag file that cannot be read reports `missing`: a tag we cannot read is a
    tag we do not have, and the correction ("write the tag") is the same one.
    """
    if not str(key_path).strip():
        return _attestation(SIGNATURE_STATE_NOT_REQUIRED, "attestation_not_required", "")
    key = _read_bounded_secret(key_path, ATTESTATION_KEY_MAX_BYTES)
    if not key:
        return _attestation(SIGNATURE_STATE_UNVERIFIABLE, "attestation_key_unreadable", ATTESTATION_KEY_FIELD)
    tag = _read_bounded_secret(tag_path, ATTESTATION_TAG_MAX_BYTES)
    if not tag:
        return _attestation(SIGNATURE_STATE_MISSING, "attestation_tag_missing", ATTESTATION_TAG_FIELD)
    expected = hmac.new(key, source_bytes, hashlib.sha256).hexdigest()
    try:
        candidate = tag.decode("ascii").strip().lower()
    except UnicodeDecodeError:
        return _attestation(SIGNATURE_STATE_INVALID, "attestation_tag_invalid", ATTESTATION_TAG_FIELD)
    # Shape first, then `compare_digest`. A malformed tag and a wrong tag are the
    # same refusal, and the shape check keeps the comparison over equal-length
    # inputs instead of over whatever the file happened to contain.
    if not _HEX_DIGEST_RE.fullmatch(candidate) or not hmac.compare_digest(candidate, expected):
        return _attestation(SIGNATURE_STATE_INVALID, "attestation_tag_invalid", ATTESTATION_TAG_FIELD)
    return _attestation(SIGNATURE_STATE_VALID, "attestation_valid", "")


def safety_rule_attestation_tag(source_bytes: bytes, *, key_path: str | Path) -> str:
    """The tag an operator would write beside a source, or empty on an unusable key.

    The producing half of the same keyed digest, and the reason the symmetry is
    impossible to miss: this function and `verify_safety_rule_attestation` need
    the identical secret. Whoever can call this can mint a tag that verifies.
    """
    key = _read_bounded_secret(key_path, ATTESTATION_KEY_MAX_BYTES)
    if not key:
        return ""
    return hmac.new(key, source_bytes, hashlib.sha256).hexdigest()


def read_safety_rule_trust_state(trust_state_path: str | Path) -> dict[str, object] | None:
    """The last locally attested rule set, or None when there is not a usable one.

    Defensive on every axis, in the idiom `capabilities/toggles.py` uses for the
    setup profile: an absent file, unreadable JSON, an unexpected key set, a
    wrong schema version, or any field that fails its own shape check all read
    as "no retained set" rather than as a partially trusted one. A record that
    cannot be fully believed cannot be partially applied.
    """
    record, error = read_json_object_result(Path(str(trust_state_path)))
    if error is not None or not isinstance(record, dict):
        return None
    if set(record) != _TRUST_STATE_FIELDS:
        return None
    if record.get("schema_version") != SAFETY_RULE_TRUST_STATE_SCHEMA_VERSION:
        return None
    if not _is_sha256(record.get("source_identity_sha256")) or not _is_sha256(record.get("content_sha256")):
        return None
    revision = record.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        return None
    if not _bounded_kinds(record.get("denied_remote_target_kinds")):
        return None
    cap = record.get("max_target_paths")
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or not 0 <= cap <= _TRUST_STATE_MAX_TARGET_PATHS):
        return None
    if not isinstance(record.get("attested_at"), str):
        return None
    return dict(record)


def write_safety_rule_trust_state(
    trust_state_path: str | Path,
    *,
    source_identity_sha256: str,
    content_sha256: str,
    revision: str,
    rules: Mapping[str, object],
    attested_at: str = "",
) -> dict[str, object]:
    """Record one rule set as locally attested. Only ever called after `valid`.

    Metadata only: two digests, a revision label, the two bounded rule values,
    and a caller-supplied timestamp. Atomic and private, so a reader never sees
    a half-written trust record and the file is not world-readable.
    """
    kinds = rules.get("denied_remote_target_kinds", [])
    cap = rules.get("max_target_paths")
    record = {
        "schema_version": SAFETY_RULE_TRUST_STATE_SCHEMA_VERSION,
        "source_identity_sha256": str(source_identity_sha256),
        "content_sha256": str(content_sha256),
        "revision": str(revision),
        "denied_remote_target_kinds": [str(kind) for kind in kinds] if isinstance(kinds, (list, tuple)) else [],
        "max_target_paths": cap if isinstance(cap, int) and not isinstance(cap, bool) else None,
        "attested_at": str(attested_at),
        "claim_boundary": SAFETY_RULE_ATTESTATION_CLAIM_BOUNDARY,
    }
    atomic_write_json(Path(str(trust_state_path)), record, private=True)
    return record


def activate_org_safety_rules(
    source_path: str | Path,
    *,
    attestation_key_path: str | Path = "",
    attestation_tag_path: str | Path | None = None,
    trust_state_path: str | Path | None = None,
    attested_at: str = "",
    time_budget_seconds: float = ORG_SAFETY_RULE_SOURCE_TIME_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Decide which org rule set is in force, and say why on both axes.

    Three outcomes:

    * `activated` -- the source read cleanly and either no key is configured or
      its tag verified. `org_rule_source` is the freshly read set. A verified
      revision is recorded as the trust state, which is the only thing that ever
      writes that file.
    * `retained` -- a key is configured and verification failed, but a
      previously verified set for this same source path is on record. That set
      stays in force, `rejected_revision` names the revision that did not
      activate, and nothing from the candidate is applied.
    * `unavailable` -- the reader refused the document, or verification failed
      with nothing retained. `org_rule_source` carries the refusal in the
      reader's own vocabulary, which `evaluate_safety_preflight` turns into a
      denial.

    The reader's verdict is checked first on purpose. A tag cannot rescue a
    document OMH cannot read, and it must not condemn one either -- the parse
    failure is the reader's answer and stays the reported reason, while
    `signature_state` still reports the tag over the bytes that were read.
    """
    source, raw = read_org_safety_rule_source_with_bytes(
        source_path, time_budget_seconds=time_budget_seconds, clock=clock
    )
    tag_path = attestation_tag_path if attestation_tag_path is not None else default_attestation_tag_path(source_path)
    attestation = verify_safety_rule_attestation(raw, key_path=attestation_key_path, tag_path=tag_path or "")
    state = str(attestation["signature_state"])
    if source["status"] != "available":
        return _activation("unavailable", source, attestation, effective="", rejected="")
    identity = str(source["source_identity_sha256"])
    revision = str(source["revision"])
    if state in ACTIVATING_SIGNATURE_STATES:
        # `not_required` deliberately records nothing. An install with no key
        # configured must not gain a trust file it never asked for, and a set
        # nobody attested is not evidence of a valid attestation.
        if state == SIGNATURE_STATE_VALID and trust_state_path is not None:
            write_safety_rule_trust_state(
                trust_state_path,
                source_identity_sha256=identity,
                content_sha256=str(source["content_sha256"]),
                revision=revision,
                rules=source["rules"] if isinstance(source["rules"], Mapping) else {},
                attested_at=attested_at,
            )
        return _activation("activated", source, attestation, effective=revision, rejected="")
    retained = read_safety_rule_trust_state(trust_state_path) if trust_state_path is not None else None
    # The retained set must belong to the source path that just failed.
    # Otherwise repointing the configured path at an unattested file would
    # inherit whatever another path had earned.
    if retained is not None and retained["source_identity_sha256"] == identity:
        return _activation(
            "retained",
            _retained_org_source(retained),
            attestation,
            effective=str(retained["revision"]),
            rejected=revision,
        )
    refusal = org_safety_rule_source_unavailable(
        identity, _ATTESTATION_REFUSAL_CODES[state], str(attestation["field"])
    )
    return _activation("unavailable", refusal, attestation, effective="", rejected=revision)


def _retained_org_source(retained: Mapping[str, object]) -> dict[str, object]:
    return org_safety_rule_source_available(
        str(retained["source_identity_sha256"]),
        content_sha256=str(retained["content_sha256"]),
        revision=str(retained["revision"]),
        rules={
            "denied_remote_target_kinds": list(retained["denied_remote_target_kinds"]),  # type: ignore[arg-type]
            "max_target_paths": retained["max_target_paths"],
        },
    )


def _activation(
    status: str,
    org_rule_source: dict[str, object],
    attestation: Mapping[str, object],
    *,
    effective: str,
    rejected: str,
) -> dict[str, object]:
    return {
        "schema_version": ORG_SAFETY_RULE_ACTIVATION_SCHEMA_VERSION,
        "status": status,
        # The attestation axis stays three named fields of its own. It is never
        # merged into the org source `status`, and the org source `reason_code`
        # is never restated here.
        "signature_state": str(attestation["signature_state"]),
        "signature_reason_code": str(attestation["reason_code"]),
        "signature_field": str(attestation["field"]),
        "algorithm": SAFETY_RULE_ATTESTATION_ALGORITHM,
        "effective_revision": effective,
        "rejected_revision": rejected,
        "org_rule_source": org_rule_source,
        "claim_boundary": SAFETY_RULE_ATTESTATION_CLAIM_BOUNDARY,
    }


def _attestation(signature_state: str, reason_code: str, field: str) -> dict[str, object]:
    return {
        "schema_version": SAFETY_RULE_ATTESTATION_SCHEMA_VERSION,
        "signature_state": signature_state,
        "reason_code": reason_code,
        "field": field,
        "algorithm": SAFETY_RULE_ATTESTATION_ALGORITHM,
        "claim_boundary": SAFETY_RULE_ATTESTATION_CLAIM_BOUNDARY,
    }


def _read_bounded_secret(path: str | Path, max_bytes: int) -> bytes:
    """Read a small local file, or return empty for every reason it is unusable.

    Symlinks are refused for the reason the rule source refuses them: the
    configured path is the thing the operator vouched for, not wherever it
    happens to point today. Surrounding whitespace is stripped so a key or tag
    written by an editor -- with a trailing newline, and CRLF on Windows -- is
    the same secret on every host.
    """
    text = str(path).strip()
    if not text:
        return b""
    candidate = Path(text)
    if candidate.is_symlink() or not candidate.is_file():
        return b""
    try:
        with candidate.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return b""
    if len(data) > max_bytes:
        return b""
    return data.strip()


def _bounded_kinds(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= _TRUST_STATE_MAX_KINDS
        and all(isinstance(item, str) and item.strip() and len(item) <= _TRUST_STATE_MAX_KIND_CHARS for item in value)
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX_DIGEST_RE.fullmatch(value) is not None

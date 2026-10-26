"""Versioned, review-gated OMH memory blocks."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory_governance import (
    ADMISSION_STATES,
    MEMORY_BLOCK_SCHEMA_VERSION,
    MEMORY_CLASSIFIER_VERSION,
    MEMORY_GOVERNANCE_POLICY_VERSION,
    RETENTION_CLASSES,
    SOURCE_CLASSES,
    build_retention,
    canonical_memory_scope,
    canonical_payload_digest,
    classify_memory_admission,
    stable_artifact_identity,
)
from .iso_datetime import strict_fromisoformat

SYSTEM_TIER = "system"
REFERENCE_TIER = "reference"
BLOCK_TIERS = (SYSTEM_TIER, REFERENCE_TIER)
DEFAULT_BLOCK_LIMIT_CHARS = 2000
DEFAULT_SYSTEM_RENDER_BUDGET_CHARS = 6000
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
# A review id reaches the filesystem as `<review_id>.json`, so it may hold only
# characters that cannot redirect the write: no separator, no dot, no `..`.
# `_opaque_id` mints 24 characters from the base64url alphabet, and the
# `<prefix>_<token>` shape other review producers use also fits.
_REVIEW_ID = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
# How many times `_opaque_id` may redraw before giving up. Each draw is
# independent with a measured rejection rate of 4.0e-05, so the chance of
# needing even a third is about 1.6e-09; a bound this loose can only be reached
# by a broken entropy source, which is a condition to raise on rather than spin
# on.
_OPAQUE_ID_MAX_DRAWS = 8
_DEFAULT_SCOPE = {"kind": "project", "ref": "default"}


class MemoryBlockError(ValueError):
    """A block did not satisfy the local persisted schema."""


@dataclass(frozen=True)
class MemoryBlock:
    """An immutable v2 block candidate or approved revision."""

    label: str
    description: str
    value: str
    limit: int
    tier: str
    block_id: str
    revision: int
    scope: dict[str, object]
    source_class: str
    admission: dict[str, object]
    retention: dict[str, object]
    revalidation: dict[str, object] | None = None
    source_record_identity: dict[str, object] | None = None
    superseded_by: dict[str, object] | None = None
    schema_version: str = MEMORY_BLOCK_SCHEMA_VERSION

    @property
    def chars(self) -> int:
        return len(self.value)

    @property
    def over_limit(self) -> bool:
        return self.chars > self.limit

    @property
    def headroom_chars(self) -> int:
        return max(0, self.limit - self.chars)

    def with_value(self, value: str) -> MemoryBlock:
        """Return a deliberately changed revision payload for review or tests."""
        return replace(self, value=value)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "revision": self.revision,
            "label": self.label,
            "description": self.description,
            "value": self.value,
            "limit": self.limit,
            "tier": self.tier,
            "scope": dict(self.scope),
            "source_class": self.source_class,
            "admission": dict(self.admission),
            "retention": dict(self.retention),
        }
        if self.revalidation is not None:
            data["revalidation"] = dict(self.revalidation)
        if self.source_record_identity is not None:
            data["source_record_identity"] = dict(self.source_record_identity)
        if self.superseded_by is not None:
            data["superseded_by"] = dict(self.superseded_by)
        return data

    def to_summary(self) -> dict[str, object]:
        """Review-visible metadata; callers add replay state before returning it."""
        return {
            "block_id": self.block_id,
            "revision": self.revision,
            "label": self.label,
            "tier": self.tier,
            "chars": self.chars,
            "limit": self.limit,
            "headroom_chars": self.headroom_chars,
            "over_limit": self.over_limit,
            "source_class": self.source_class,
            "admission_status": self.admission.get("state"),
        }


def normalize_label(value: str) -> str:
    label = str(value or "").strip().lower()
    if not _LABEL.match(label):
        raise MemoryBlockError("block label must be 1-63 lowercase alphanumeric, '-' or '_' characters")
    return label


def normalize_tier(value: str) -> str:
    tier = str(value or "").strip().lower()
    if tier not in BLOCK_TIERS:
        raise MemoryBlockError(f"block tier must be one of {BLOCK_TIERS}; got {tier!r}")
    return tier


def blocks_dir(omh_home: str | Path) -> Path:
    return Path(omh_home).expanduser() / "memory" / "blocks"


def _candidate_dir(omh_home: str | Path) -> Path:
    return Path(omh_home).expanduser() / "memory" / "block_candidates"


def _reviews_dir(omh_home: str | Path) -> Path:
    return Path(omh_home).expanduser() / "memory" / "block_reviews"


def block_path(omh_home: str | Path, label: str, tier: str) -> Path:
    return blocks_dir(omh_home) / normalize_tier(tier) / f"{normalize_label(label)}.json"


def _candidate_path(omh_home: str | Path, label: str, tier: str) -> Path:
    return _candidate_dir(omh_home) / normalize_tier(tier) / f"{normalize_label(label)}.json"


def build_memory_block(
    label: str,
    value: str,
    *,
    description: str = "",
    limit: int = DEFAULT_BLOCK_LIMIT_CHARS,
    tier: str = SYSTEM_TIER,
    block_id: str | None = None,
    revision: int = 1,
    scope: dict[str, object] | None = None,
    source_class: str = "omh_local",
    admission: dict[str, object] | None = None,
    retention: dict[str, object] | None = None,
    revalidation: dict[str, object] | None = None,
    source_record_identity: dict[str, object] | None = None,
    superseded_by: dict[str, object] | None = None,
) -> MemoryBlock:
    """Build a pending v2 candidate; this never silently grants approval."""
    normalized_label = normalize_label(label)
    normalized_tier = normalize_tier(tier)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise MemoryBlockError(f"block limit must be a positive integer; got {limit!r}")
    text = str(value or "")
    if len(text) > limit:
        raise MemoryBlockError(f"block {normalized_label!r} is {len(text)} chars against a {limit}-char limit")
    moment = datetime.now(timezone.utc)
    block = MemoryBlock(
        normalized_label,
        str(description or ""),
        text,
        limit,
        normalized_tier,
        block_id or _opaque_id(),
        revision,
        canonical_memory_scope(scope or _DEFAULT_SCOPE),
        source_class,
        dict(admission or {"state": "pending_review"}),
        dict(retention or build_retention("durable", record_type="fact", admitted_at=moment)),
        dict(revalidation) if revalidation is not None else None,
        dict(source_record_identity) if source_record_identity is not None else None,
        dict(superseded_by) if superseded_by is not None else None,
    )
    _validate_block(block)
    return block


def approve_memory_block(block: MemoryBlock, *, reviewer_claim: str = "") -> MemoryBlock:
    """Bind a candidate revision to explicit local review evidence.

    ``reviewer_claim`` is display-only and never an authenticated identity.
    """
    if block.schema_version != MEMORY_BLOCK_SCHEMA_VERSION or block.source_class != "omh_local":
        raise MemoryBlockError("only local v2 block candidates may be approved")
    identity = stable_artifact_identity(block.to_dict())
    admission = {
        "state": "approved_manual",
        "admitted_at": _utc_now(),
        "review_id": _opaque_id(),
        "artifact_identity": identity,
        "payload_digest": canonical_payload_digest(block.to_dict()),
        "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION,
        "classifier_version": MEMORY_CLASSIFIER_VERSION,
    }
    if reviewer_claim:
        admission["reviewer_claim"] = str(reviewer_claim)
    approved = replace(block, admission=admission)
    _validate_block(approved)
    return approved


def write_memory_block(omh_home: str | Path, block: MemoryBlock) -> Path:
    """Persist only approved revisions to the active store; stage all others."""
    _validate_block(block)
    approved = block.admission.get("state") in {"approved_manual", "approved_auto_safe"}
    path = block_path(omh_home, block.label, block.tier) if approved else _candidate_path(omh_home, block.label, block.tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(block.to_dict(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
    if approved:
        _write_review(omh_home, block)
    return path


def read_memory_block(path: str | Path) -> MemoryBlock | None:
    """Read and structurally validate a block without granting replay trust."""
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return _block_from_data(data)


def read_memory_blocks(omh_home: str | Path, *, tier: str | None = None) -> tuple[MemoryBlock, ...]:
    """Return active and staged blocks for review; rendering selects separately."""
    tiers = (normalize_tier(tier),) if tier is not None else BLOCK_TIERS
    found: list[MemoryBlock] = []
    for root in (blocks_dir(omh_home), _candidate_dir(omh_home)):
        for tier_name in tiers:
            try:
                candidates = sorted((root / tier_name).glob("*.json"))
            except OSError:
                continue
            for path in candidates:
                block = read_memory_block(path)
                if block is not None and block.tier == tier_name:
                    found.append(block)
    return tuple(sorted(found, key=lambda item: (item.label, item.revision, item.block_id)))


def delete_memory_block(omh_home: str | Path, label: str, tier: str) -> bool:
    removed = False
    for path in (block_path(omh_home, label, tier), _candidate_path(omh_home, label, tier)):
        try:
            path.unlink()
            removed = True
        except OSError:
            pass
    return removed


def select_memory_blocks(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a block collection at the final replay boundary."""
    select = _memory_block_replay.select_memory_blocks

    return select(*args, **kwargs)


def render_memory_blocks(
    blocks: tuple[MemoryBlock, ...] | list[MemoryBlock],
    *,
    budget_chars: int = DEFAULT_SYSTEM_RENDER_BUDGET_CHARS,
    evaluations: dict[str, dict[str, object]] | None = None,
) -> str:
    return render_memory_blocks_counted(blocks, budget_chars=budget_chars, evaluations=evaluations)[0]


def render_memory_blocks_counted(
    blocks: tuple[MemoryBlock, ...] | list[MemoryBlock],
    *,
    budget_chars: int = DEFAULT_SYSTEM_RENDER_BUDGET_CHARS,
    evaluations: dict[str, dict[str, object]] | None = None,
) -> tuple[str, int]:
    """The rendered section and how many blocks it carries in full."""
    eligible, evaluated = _renderable(blocks, evaluations)
    if not blocks:
        return "", 0
    lines, used, rendered, omissions = ["<memory_blocks>"], 0, 0, _omissions(blocks, evaluated)
    for block in eligible:
        element = _render_block(block)
        if used + len(element) > max(budget_chars, 0):
            omissions.append({"block_id": block.block_id, "revision": block.revision, "reason_code": "render_budget_exhausted"})
            continue
        used += len(element)
        rendered += 1
        lines.append(element)
    lines.extend(_render_omissions(omissions))
    lines.append("</memory_blocks>")
    return "\n".join(lines), rendered


def render_memory_blocks_within(
    blocks: tuple[MemoryBlock, ...] | list[MemoryBlock],
    *,
    total_chars: int,
    evaluations: dict[str, dict[str, object]] | None = None,
) -> tuple[str, int]:
    """System blocks for the prefetch pack, the whole section inside ``total_chars``.

    `render_memory_blocks_counted` budgets block elements only; here the
    wrapper and every omission line count too, because the pack has one
    budget for everything it hands Hermes. Blocks are kept in rank order and
    a block that does not fit is dropped whole, never clipped; the dropped
    blocks stay listed and readable through `omh_memory` on a console. Omission lines that do not
    fit become one ``<omitted count=...>`` line. ``""`` when not even an empty
    section fits.
    """
    eligible, evaluated = _renderable(blocks, evaluations)
    if not blocks:
        return "", 0
    ineligible = _render_omissions(_omissions(blocks, evaluated))

    def section(kept: list[MemoryBlock]) -> str:
        kept_ids = {block.block_id for block in kept}
        dropped = [
            {"block_id": block.block_id, "revision": block.revision, "reason_code": "render_budget_exhausted"}
            for block in eligible
            if block.block_id not in kept_ids
        ]
        return _framed_section(
            "<memory_blocks>", [_render_block(block) for block in kept], [*ineligible, *_render_omissions(dropped)], "</memory_blocks>", total_chars
        )

    kept: list[MemoryBlock] = []
    for block in eligible:
        if section([*kept, block]):
            kept.append(block)
    return section(kept), len(kept)


def render_block_index(
    blocks: tuple[MemoryBlock, ...] | list[MemoryBlock],
    *,
    evaluations: dict[str, dict[str, object]] | None = None,
    budget_chars: int | None = None,
) -> str:
    """Reference blocks by label; with ``budget_chars``, a prefix of the lines and one count line for the rest."""
    eligible, evaluated = _renderable(blocks, evaluations)
    if not blocks:
        return ""
    lines = [f'  <block label="{block.label}" chars="{block.chars}" limit="{block.limit}">{block.description}</block>' for block in eligible]
    lines.extend(_render_omissions(_omissions(blocks, evaluated)))
    if budget_chars is None:
        return "\n".join(["<memory_block_index>", *lines, "</memory_block_index>"])
    return _framed_section("<memory_block_index>", [], lines, "</memory_block_index>", budget_chars)


def _framed_section(open_tag: str, kept: list[str], optional: list[str], close_tag: str, budget_chars: int) -> str:
    """``kept`` whole, the longest prefix of ``optional`` that fits, one count line for the rest; "" if none fits."""
    for shown in range(len(optional), -1, -1):
        rest = len(optional) - shown
        tail = [f'  <omitted count="{rest}" />'] if rest else []
        text = "\n".join([open_tag, *kept, *optional[:shown], *tail, close_tag])
        if len(text) <= budget_chars:
            return text
    return ""


def _renderable(blocks: tuple[MemoryBlock, ...] | list[MemoryBlock], evaluations: dict[str, dict[str, object]] | None) -> tuple[list[MemoryBlock], dict[str, dict[str, object]]]:
    if evaluations is None:
        evaluations = select_memory_blocks(tuple(blocks)).evaluations
    return [block for block in blocks if evaluations.get(block.block_id, {}).get("eligible")], evaluations


def _omissions(blocks: tuple[MemoryBlock, ...] | list[MemoryBlock], evaluations: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"block_id": block.block_id, "revision": block.revision, "reason_code": evaluation.get("reason_code", "ineligible")}
        for block in blocks
        if not (evaluation := evaluations.get(block.block_id, {})).get("eligible")
    ]


def _render_omissions(omissions: list[dict[str, object]]) -> list[str]:
    return [f'  <omitted block_id="{item["block_id"]}" revision="{item["revision"]}" reason="{item["reason_code"]}" />' for item in omissions]


def _render_block(block: MemoryBlock) -> str:
    return f"  <{block.label}>\n    <description>{block.description}</description>\n    <metadata>chars_current={block.chars} chars_limit={block.limit}</metadata>\n    <value>{block.value}</value>\n  </{block.label}>"


def _block_from_data(data: object) -> MemoryBlock | None:
    if not isinstance(data, dict):
        return None
    try:
        if data.get("schema_version") == "omh_memory_block/v1":
            return _legacy_block(data)
        if set(data) - _V2_KEYS or data.get("schema_version") != MEMORY_BLOCK_SCHEMA_VERSION:
            return None
        block = MemoryBlock(
            str(data["label"]), str(data["description"]), str(data["value"]), data["limit"], str(data["tier"]),
            str(data["block_id"]), data["revision"], dict(data["scope"]), str(data["source_class"]),
            dict(data["admission"]), dict(data["retention"]), _dict_or_none(data.get("revalidation")),
            _dict_or_none(data.get("source_record_identity")), _dict_or_none(data.get("superseded_by")), str(data["schema_version"]),
        )
        _validate_block(block)
        return block
    except (KeyError, TypeError, ValueError, MemoryBlockError):
        return None


def _legacy_block(data: dict[str, object]) -> MemoryBlock:
    return MemoryBlock(
        normalize_label(str(data.get("label", ""))), str(data.get("description", "")), str(data.get("value", "")),
        _positive_limit(data.get("limit")), normalize_tier(str(data.get("tier", ""))), f"legacy-{_opaque_id()}", 1,
        dict(_DEFAULT_SCOPE), "omh_local", {"state": "pending_review"},
        build_retention("durable", record_type="fact", admitted_at=datetime.now(timezone.utc)), schema_version="omh_memory_block/v1",
    )


def _validate_block(block: MemoryBlock) -> None:
    if block.schema_version != MEMORY_BLOCK_SCHEMA_VERSION:
        return
    normalize_label(block.label)
    normalize_tier(block.tier)
    if not isinstance(block.block_id, str) or not block.block_id or not isinstance(block.revision, int) or block.revision <= 0:
        raise MemoryBlockError("block_id must be opaque and revision must be positive")
    if not isinstance(block.value, str) or not isinstance(block.description, str) or block.chars > _positive_limit(block.limit):
        raise MemoryBlockError("block text or limit is invalid")
    canonical_memory_scope(block.scope)
    if block.source_class not in SOURCE_CLASSES:
        raise MemoryBlockError("block source_class is invalid")
    state = block.admission.get("state") if isinstance(block.admission, dict) else None
    if state not in ADMISSION_STATES:
        raise MemoryBlockError("block admission state is invalid")
    if state in {"approved_manual", "approved_auto_safe"}:
        required = {"review_id", "artifact_identity", "payload_digest", "policy_version", "classifier_version", "admitted_at"}
        allowed = required | {"state", "reviewer_claim"}
        if block.source_class != "omh_local" or set(block.admission) - allowed or not required <= set(block.admission):
            raise MemoryBlockError("approved blocks require a local immutable review link")
        # `review_id` becomes a filename in `_write_review`, and until now this
        # check required the key to be present without ever looking at its
        # value. The sibling subsystem guards the same interpolation both ways:
        # `_memory_review_path` (`src/workflows/memory.py:4196`) matches
        # `_SAFE_REF` AND calls `_assert_under_memory_root`. This is that guard
        # pattern arriving in the module that did not have it.
        #
        # Defence in depth, not a live hole: `_write_review` is reached only
        # from `write_memory_block`'s approved branch, the sole production
        # caller (`cmd_memory_block_set`) supplies no admission and so writes a
        # `pending_review` block, and `approve_memory_block` -- the only
        # producer of an approved admission, which always mints the id itself
        # -- has no caller outside the tests. The safety is a property of the
        # call graph, not of the code here, and `approve_memory_block` presents
        # as public API. A block deserialised from disk can already carry an
        # arbitrary value into a `MemoryBlock` (`_block_from_data` copies
        # `admission` verbatim): open at the deserialiser, closed at the
        # writer. This closes the deserialiser half too.
        if not _REVIEW_ID.fullmatch(str(block.admission["review_id"])):
            raise MemoryBlockError("review id must be an opaque filename-safe token")
        _validate_identity(block.admission["artifact_identity"])
    elif set(block.admission) != {"state"}:
        raise MemoryBlockError("unapproved block admission must contain only state")
    _validate_retention(block.retention)
    if block.revalidation is not None:
        if set(block.revalidation) != {"deadline"} or not _timestamp(block.revalidation["deadline"]):
            raise MemoryBlockError("revalidation must contain one UTC deadline")
    if block.source_record_identity is not None:
        _validate_identity(block.source_record_identity)


def _validate_retention(retention: dict[str, object]) -> None:
    if not isinstance(retention, dict) or retention.get("class") not in RETENTION_CLASSES:
        raise MemoryBlockError("block retention is invalid")
    kind = retention["class"]
    allowed = {"class", "admitted_at"}
    if kind == "volatile":
        allowed |= {"ttl_days", "expires_at"}
        if not {"ttl_days", "expires_at"} <= set(retention):
            raise MemoryBlockError("volatile blocks require a bounded expiry")
    elif kind == "standard":
        allowed |= {"ttl_days", "expires_at"}
    if set(retention) - allowed or not _timestamp(retention.get("admitted_at")):
        raise MemoryBlockError("block retention is invalid")
    if kind == "durable" and "expires_at" in retention:
        raise MemoryBlockError("durable blocks cannot expire")
    if "expires_at" in retention and not _timestamp(retention["expires_at"]):
        raise MemoryBlockError("block expiry is invalid")


def _validate_identity(identity: object) -> None:
    if not isinstance(identity, dict) or set(identity) != {"schema_version", "id", "id_key", "revision", "scope"}:
        raise MemoryBlockError("source record identity is malformed")
    if not isinstance(identity["id"], str) or not identity["id"] or not isinstance(identity["id_key"], str):
        raise MemoryBlockError("source record identity is malformed")
    if not isinstance(identity["revision"], int) or identity["revision"] <= 0:
        raise MemoryBlockError("source record identity is malformed")
    canonical_memory_scope(identity["scope"])


def _timestamp(value: object) -> bool:
    try:
        parsed = strict_fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _write_review(omh_home: str | Path, block: MemoryBlock) -> None:
    admission = block.admission
    review = {
        "schema_version": "project_memory_review_record/v2",
        "review_id": admission["review_id"],
        "artifact_identity": admission["artifact_identity"],
        "decision": admission["state"],
        "payload_digest": admission["payload_digest"],
        "policy_version": admission["policy_version"],
        "classifier_version": admission["classifier_version"],
    }
    path = _reviews_dir(omh_home) / f"{admission['review_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _opaque_id() -> str:
    """An opaque id that cannot be read back as a leaked credential (#1641).

    `secrets.token_urlsafe(18)` draws 24 characters from the base64url
    alphabet, so a token occasionally comes out shaped like a real API key --
    `sk-`, `hf_`, `gh[oprsu][_-]`, `AIza`, `xox-` and the separator-split
    variant all fire, at a measured 4.0e-05 across every pattern, or one id in
    25,000. When such a token became a block's own `review_id`, the admission
    walk classified the block's own identifier as a secret and omitted the
    approved block from its own render with
    `safety_blocked_in_admission.review_id`. Nothing was lost; the block simply
    never appeared, and the CI failure that produced cost a full diagnosis.

    Redrawing is the fix rather than exempting the field. An exemption is
    implementable -- `evaluate_renderable_strings` holds the value path and
    could special-case `admission.review_id` -- but `_validate_block` accepts
    any string there, `build_memory_block` takes a caller-supplied `admission`,
    and `_block_from_data` copies one verbatim off disk. So the exemption would
    be a smuggling channel: the category argument ("the system minted this, it
    is not user content") holds for how the field is used and fails for what it
    accepts. The shape check in `_validate_block` closes that half.
    """
    for _ in range(_OPAQUE_ID_MAX_DRAWS):
        candidate = secrets.token_urlsafe(18)
        if classify_memory_admission(candidate).get("status") == "safe":
            return candidate
    raise MemoryBlockError("could not mint an opaque id that classifies as safe")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _positive_limit(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MemoryBlockError("block limit must be a positive integer")
    return value


def _dict_or_none(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MemoryBlockError("block metadata must be an object")
    return dict(value)


_V2_KEYS = {
    "schema_version", "block_id", "revision", "label", "description", "value", "limit", "tier", "scope",
    "source_class", "admission", "retention", "revalidation", "source_record_identity", "superseded_by",
}


# Module scope, not call time: Hermes can evict the bundle from sys.modules
# while its callbacks stay live (#1979). Last, so an import cycle finds every name.
from . import memory_block_replay as _memory_block_replay

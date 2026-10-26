"""`handoff_input_manifest/v1` — the bounded context package a coding owner receives.

A user could not tell which files, plan sections, diffs, or reviewed memories a
coding owner was about to receive, whether the package was oversized, or whether
anything in it was unsafe. This module is the answer: one reviewable, bounded,
reproducible manifest that every coding handoff pins by revision and digest.

Superset, not an extension of `handoff_context_pack/v1` (#823)
--------------------------------------------------------------

The pack was the obvious place to grow, and growing it was rejected for three
reasons that are properties of the pack, not preferences:

1. Its items are memory-snapshot summaries. `build_handoff_context_pack` walks
   `inspection["snapshots"]` and nothing else, so a file, a plan heading, and a
   revision range have no shape there. Making them fit would change what a pack
   item *is*, in a contract already frozen at v1.
2. `validate_handoff_context_pack` enforces an allowed-key set and this issue's
   AC1 requires *every* item to carry six new fields. Making them required would
   invalidate every pack alive today — `build_plan_handoff_context_pack`, every
   wrapper-supplied pack read through `read_handoff_context_pack_file`, and the
   stored fixtures. Making them optional would satisfy the letter of AC1 and
   none of its point, because a manifest whose items may omit their hash cannot
   answer "is this package oversized".
3. A pack is prepared *memory* context. A manifest is the whole input package.
   They are different questions and a v1 contract should keep answering the one
   it was built for.

So the manifest is a superset that **references** the pack rather than a second
concept competing with it. When a pack is supplied, its `included_context`
projects into `reviewed_memory` manifest items — same items, one lineage — and
`derived_from` records the pack it came from with the counts the pack itself
reported. The pack keeps validating unchanged and stays the single source of
reviewed memory; the manifest is the single surface a user reviews before the
handoff goes out. Pack exclusions are deliberately *not* re-listed here under a
second vocabulary: the pack already enumerates its own, and `derived_from`
points at it.

Where `safety_result` comes from
--------------------------------

Not from a new judgement. `context_pack` is already an untrusted surface in
`action_gate.UNTRUSTED_SURFACES`, and `action_gate.AUTHORITY_CLASSIFIERS` names
the two classifiers that decide such a surface. This module applies exactly
those two, each to the surface it was written for:

* `classify_memory_admission` reads item **content** — a file's bytes, a plan
  section, a diff, a memory summary. It is the same three-status, fail-closed
  vocabulary the memory-admission lane uses on prohibited material.
* `ensure_safe_opaque_ref_content` reads the item's own **metadata** — the
  selector expression and the workspace-relative local ref, which are exactly
  the single-line opaque references that function exists to screen.

`action_gate.is_authority_shaped` is deliberately *not* the gate. Its docstring
says it is "purely a report", and it is: it routes through
`ensure_safe_opaque_ref_content`, whose unicode screen rejects any control
character, so every multi-line text flags. As a report on a flattened surface
that is the intended conservatism; as an inclusion gate it would exclude every
real file. Reporting and refusing are different jobs and this one refuses.

An item that does not screen `safe` is **excluded with a reason**, never
included and never silently dropped, so the manifest that reaches a handoff
carries only screened material and the refusals stay visible next to it.

Determinism
-----------

There is no timestamp anywhere in this contract, so none can reach the digest
seed. The digest is a pure function of the manifest's content, and the manifest
is a pure function of (selectors, workspace bytes, pack, revision, budget). Two
builds of the same package agree byte for byte, on any platform: local refs are
POSIX-relative, glob expansion is sorted, and item ordering is the order the
caller declared.

`revision` and `digest` are what the handoff pins. Attaching deep-copies the
manifest, so a later edit to the manifest a caller still holds cannot be read as
the one the handoff carried — the handoff keeps naming the old revision and the
old digest, and `input_manifest_pin_matches` recomputes rather than trusting the
stored value, so the mismatch is detectable rather than merely likely.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Sequence

from ..plugin_bundle.omh._governance_safety import classify_memory_admission
from ..system.binary_io import open_binary
from ..workflows.domain_intelligence_admission import ensure_safe_opaque_ref_content
from .action_gate import AUTHORITY_CLASSIFIERS
from .context_safety import RUN_CONTEXT_BUDGET_BYTES


HANDOFF_INPUT_MANIFEST_SCHEMA_VERSION = "handoff_input_manifest/v1"
HANDOFF_INPUT_MANIFEST_PIN_SCHEMA_VERSION = "handoff_input_manifest_pin/v1"

# The four kinds #823 names. One shape carries all of them: kind-specific detail
# lands in `provenance` and `selector`, never in a differently-shaped item.
MANIFEST_ITEM_KINDS = ("diff", "file", "plan_section", "reviewed_memory")
# A selector says how an item was chosen and must be reproducible: the same
# selector against the same inputs yields the same item. Each kind accepts only
# the selector kinds that can be replayed for it.
MANIFEST_SELECTOR_KINDS = ("memory_record_id", "path", "path_glob", "plan_heading", "revision_range")
SELECTOR_KINDS_BY_ITEM_KIND = {
    "diff": ("revision_range",),
    "file": ("path", "path_glob"),
    "plan_section": ("plan_heading",),
    "reviewed_memory": ("memory_record_id",),
}
# Why an item belongs, as a code a reader can act on. Two values, both derived
# and neither guessed: a caller put it here, or the reviewed context pack did.
# The full answer to "why this item" is the pair (inclusion_reason, selector) —
# the code says who put it in the package, the selector says exactly how it was
# found and how to find it again.
MANIFEST_INCLUSION_REASONS = ("explicit_selection", "reviewed_memory_projection")
MANIFEST_EXCLUSION_REASONS = (
    "blocked_by_unresolved_conflict",
    "duplicate_item",
    "outside_workspace",
    "over_budget",
    "unreadable_source",
    "unsafe_content",
)
# `classify_memory_admission`'s vocabulary, unchanged. Only `safe` may travel.
MANIFEST_SAFETY_STATUSES = ("blocked", "needs_review", "safe")
SAFE_SAFETY_STATUS = "safe"

CONTENT_CLASSIFIER = AUTHORITY_CLASSIFIERS[0]
REF_CLASSIFIER = AUTHORITY_CLASSIFIERS[1]

# The byte budget starts at the existing context byte budget rather than
# inventing a second number, and an item that does not fit is reported with the
# pack's existing `over_budget` word. The item cap matches
# `build_handoff_context_pack`'s default `context_limit` for the same reason.
MANIFEST_BUDGET_BYTES = RUN_CONTEXT_BUDGET_BYTES
MANIFEST_ITEM_LIMIT = 12

# Local workspace state is directly observed bytes. None of the memory-side
# truth levels describe that -- they grade remembered claims -- so files and
# diffs get one word of their own. Plan sections keep `approved_context`, the
# level `build_plan_handoff_context_pack` already gives an accepted plan.
LOCAL_SOURCE_TRUTH_LEVEL = "local_source_state"
APPROVED_CONTEXT_TRUTH_LEVEL = "approved_context"

MANIFEST_CLAIM_BOUNDARY = (
    "An input manifest lists the bounded local context a coding owner is prepared to receive. "
    "It is prepared context only: not dispatch, execution, review, CI, merge-readiness, or merge evidence, "
    "and not proof that any owner read a listed item."
)

MAX_ITEM_ID_CHARS = 160
MAX_DETAIL_CHARS = 240

_MANIFEST_KEYS = {
    "schema_version",
    "manifest_id",
    "revision",
    "digest",
    "executor_target",
    "session_id",
    "scope",
    "derived_from",
    "items",
    "excluded_items",
    "budget",
    "redaction_policy",
    "claim_boundary",
}
_MANIFEST_ITEM_KEYS = {
    "item_id",
    "item_kind",
    "provenance",
    "selector",
    "hash",
    "byte_cost",
    "inclusion_reason",
    "safety_result",
}
# AC1's six, by name, so a reader can check the claim against the code.
MANIFEST_REQUIRED_ITEM_FIELDS = (
    "provenance",
    "selector",
    "hash",
    "byte_cost",
    "inclusion_reason",
    "safety_result",
)
_MANIFEST_PROVENANCE_KEYS = {"source", "local_ref", "truth_level"}
_MANIFEST_SELECTOR_KEYS = {"kind", "expression"}
_MANIFEST_SAFETY_KEYS = {"status", "classifier", "detail"}
_MANIFEST_EXCLUDED_KEYS = {"item_id", "item_kind", "selector", "reason", "byte_cost", "detail"}
_MANIFEST_BUDGET_KEYS = {
    "budget_bytes",
    "requested_bytes",
    "used_bytes",
    "remaining_bytes",
    "over_budget_bytes",
    "item_limit",
    "item_count",
    "excluded_count",
    "over_budget",
}
_MANIFEST_DERIVED_FROM_KEYS = {
    "schema_version",
    "executor_target",
    "session_id",
    "included_context_count",
    "excluded_context_count",
    "blocked_by_conflicts_count",
}
_MANIFEST_SCOPE_KEYS = {"kind", "ref"}
_MANIFEST_PIN_KEYS = {"schema_version", "manifest_id", "revision", "digest"}
# Nothing here is a clock, so nothing time-shaped can enter the seed. `digest`
# is excluded because a value cannot hash itself.
_DIGEST_EXCLUDED_KEYS = frozenset({"digest"})
_SHA256_PREFIX = "sha256:"
_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)
_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)
_WINDOWS_PATH_CTIME_IS_BIRTHTIME = os.name == "nt"


class _WorkspaceFileRefusal(Exception):
    def __init__(self, detail: str, *, byte_cost: int = 0, over_budget: bool = False) -> None:
        super().__init__(detail)
        self.detail = detail
        self.byte_cost = byte_cost
        self.over_budget = over_budget


@dataclass(frozen=True)
class ManifestSelection:
    """One reproducible request for context, before it is resolved.

    `content` is only read for `diff` items. Core `omh` runs no subprocesses, so
    it cannot expand a revision range itself; the caller that already holds the
    diff supplies the text, and the manifest stores its hash and byte cost
    rather than the text. Every other kind is read from the local workspace.
    """

    item_kind: str
    selector_kind: str
    expression: str
    content: str | None = None


def build_handoff_input_manifest(
    *,
    executor_target: str = "generic",
    session_id: str = "",
    scope: dict[str, str] | None = None,
    workspace_root: str | Path | None = None,
    selections: Sequence[ManifestSelection] | None = None,
    context_pack: dict[str, Any] | None = None,
    revision: int = 1,
    budget_bytes: int = MANIFEST_BUDGET_BYTES,
    item_limit: int = MANIFEST_ITEM_LIMIT,
) -> dict[str, object]:
    """Build the bounded manifest for one prepared handoff.

    Explicit selections are resolved before the reviewed-memory projection.
    A user who named a file chose it; reviewed memory is supporting context, so
    when the budget cannot hold everything the named file is what survives and
    the memory item is what carries the `over_budget` row.
    """
    if revision < 1:
        raise ValueError("handoff input manifest revision must be at least 1")
    if budget_bytes < 0 or item_limit < 0:
        raise ValueError("handoff input manifest budget must not be negative")
    scope_value = _normalized_scope(scope)
    root = _resolved_workspace_root(workspace_root)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    for selection in selections or ():
        _validate_selection(selection)
        resolved, refused = _resolve_selection(selection, root=root, budget_bytes=budget_bytes)
        rows.extend(resolved)
        failures.extend(refused)
    memory_rows, memory_failures = _resolve_context_pack(context_pack)
    rows.extend(memory_rows)
    failures.extend(memory_failures)

    items: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = list(failures)
    seen: set[str] = set()
    used_bytes = 0
    for row in rows:
        item_id = str(row["item_id"])
        selector = dict(row["selector"])
        byte_cost = int(row["byte_cost"])
        if item_id in seen:
            excluded.append(
                _exclusion(
                    item_id,
                    str(row["item_kind"]),
                    selector,
                    "duplicate_item",
                    byte_cost,
                    "Another selector already produced this item id, so the second copy is dropped rather than counted twice.",
                )
            )
            continue
        seen.add(item_id)
        safety = classify_manifest_item_safety(
            content=str(row["content"]),
            selector_expression=str(selector.get("expression", "")),
            local_ref=str(row["local_ref"]),
        )
        if safety["status"] != SAFE_SAFETY_STATUS:
            excluded.append(
                _exclusion(
                    item_id,
                    str(row["item_kind"]),
                    selector,
                    "unsafe_content",
                    byte_cost,
                    f"The context-safety screen returned {safety['status']} via {safety['classifier']}.",
                )
            )
            continue
        if len(items) >= item_limit:
            excluded.append(
                _exclusion(
                    item_id,
                    str(row["item_kind"]),
                    selector,
                    "over_budget",
                    byte_cost,
                    f"The manifest already holds its limit of {item_limit} item(s).",
                )
            )
            continue
        if used_bytes + byte_cost > budget_bytes:
            excluded.append(
                _exclusion(
                    item_id,
                    str(row["item_kind"]),
                    selector,
                    "over_budget",
                    byte_cost,
                    (
                        f"This item costs {byte_cost} byte(s) and the budget of {budget_bytes} byte(s) "
                        f"has {max(budget_bytes - used_bytes, 0)} byte(s) left."
                    ),
                )
            )
            continue
        used_bytes += byte_cost
        items.append(
            {
                "item_id": item_id,
                "item_kind": str(row["item_kind"]),
                "provenance": {
                    "source": str(row["source"]),
                    "local_ref": str(row["local_ref"]),
                    "truth_level": str(row["truth_level"]),
                },
                "selector": selector,
                "hash": str(row["hash"]),
                "byte_cost": byte_cost,
                "inclusion_reason": str(row["inclusion_reason"]),
                "safety_result": safety,
            }
        )

    # Summed from the exclusion rows rather than tallied as the loop runs, so
    # the numbers cover both places an item can be refused for size: the loop
    # below the budget line, and `_read_workspace_file`'s pre-read refusal of a
    # source larger than the whole budget. Tallying in one place and reporting
    # from two is how a truncation becomes silent.
    over_budget_bytes = sum(
        int(row["byte_cost"]) for row in excluded if row["reason"] == "over_budget"
    )
    requested_bytes = used_bytes + over_budget_bytes
    manifest: dict[str, object] = {
        "schema_version": HANDOFF_INPUT_MANIFEST_SCHEMA_VERSION,
        "manifest_id": _manifest_id(executor_target, session_id, scope_value),
        "revision": int(revision),
        "digest": "",
        "executor_target": str(executor_target),
        "session_id": str(session_id),
        "scope": scope_value,
        "derived_from": _derived_from(context_pack),
        "items": items,
        "excluded_items": excluded,
        "budget": {
            "budget_bytes": int(budget_bytes),
            "requested_bytes": requested_bytes,
            "used_bytes": used_bytes,
            "remaining_bytes": max(budget_bytes - used_bytes, 0),
            "over_budget_bytes": max(requested_bytes - budget_bytes, 0),
            "item_limit": int(item_limit),
            "item_count": len(items),
            "excluded_count": len(excluded),
            "over_budget": bool(over_budget_bytes),
        },
        "redaction_policy": "metadata_only",
        "claim_boundary": MANIFEST_CLAIM_BOUNDARY,
    }
    manifest["digest"] = input_manifest_digest(manifest)
    return manifest


def classify_manifest_item_safety(*, content: str, selector_expression: str = "", local_ref: str = "") -> dict[str, str]:
    """The item's safety verdict, from the two classifiers `action_gate` already names.

    Content and refs are screened by different functions because they are
    different shapes: content is a body and refs are single opaque lines. Fail
    closed on both — a status other than `safe` never travels.
    """
    admission = classify_memory_admission(content)
    status = str(admission.get("status", "blocked"))
    if status not in MANIFEST_SAFETY_STATUSES:
        status = "blocked"
    if status != SAFE_SAFETY_STATUS:
        return {
            "status": status,
            "classifier": CONTENT_CLASSIFIER,
            "detail": "Item content did not pass the memory-admission screen.",
        }
    for ref in (selector_expression, local_ref):
        if not ref:
            continue
        try:
            ensure_safe_opaque_ref_content(ref, "manifest_ref")
        except ValueError as exc:
            return {
                "status": "blocked",
                "classifier": REF_CLASSIFIER,
                "detail": f"Item reference was refused as {exc}.",
            }
    return {"status": SAFE_SAFETY_STATUS, "classifier": CONTENT_CLASSIFIER, "detail": ""}


def input_manifest_digest(manifest: dict[str, Any]) -> str:
    """The manifest's content digest, recomputed rather than read back.

    Every field except `digest` itself is in the seed, `revision` included, so a
    revision bump that changes nothing else still changes the digest and a pin
    can tell the two apart.
    """
    seed = {key: value for key, value in manifest.items() if key not in _DIGEST_EXCLUDED_KEYS}
    encoded = json.dumps(seed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"{_SHA256_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def input_manifest_pin(manifest: dict[str, Any]) -> dict[str, object]:
    """The bounded (id, revision, digest) triple a handoff records."""
    return {
        "schema_version": HANDOFF_INPUT_MANIFEST_PIN_SCHEMA_VERSION,
        "manifest_id": str(manifest.get("manifest_id", "")),
        "revision": int(manifest.get("revision", 0) or 0),
        "digest": str(manifest.get("digest", "")),
    }


def input_manifest_pin_matches(pin: Any, manifest: Any) -> bool:
    """True when `pin` names this exact manifest content.

    The digest is recomputed from the manifest instead of compared against the
    `digest` the manifest carries, so an edit that forgot to refresh that field
    is still caught.
    """
    if not isinstance(pin, dict) or not isinstance(manifest, dict):
        return False
    return (
        str(pin.get("manifest_id", "")) == str(manifest.get("manifest_id", ""))
        and int(pin.get("revision", 0) or 0) == int(manifest.get("revision", 0) or 0)
        and str(pin.get("digest", "")) == input_manifest_digest(manifest)
    )


def validate_handoff_input_manifest(value: Any, *, label: str = "input_manifest") -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    _allowed_keys(value, _MANIFEST_KEYS, errors, label)
    if value.get("schema_version") != HANDOFF_INPUT_MANIFEST_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {HANDOFF_INPUT_MANIFEST_SCHEMA_VERSION}")
    if value.get("redaction_policy") != "metadata_only":
        errors.append(f"{label} redaction_policy must be metadata_only")
    if not isinstance(value.get("claim_boundary"), str) or not value.get("claim_boundary"):
        errors.append(f"{label} claim_boundary must be a non-empty string")
    for key in ("executor_target", "session_id"):
        if not isinstance(value.get(key), str):
            errors.append(f"{label}.{key} must be a string")
    if not isinstance(value.get("manifest_id"), str) or not value.get("manifest_id"):
        errors.append(f"{label}.manifest_id must be a non-empty string")
    revision = value.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        errors.append(f"{label}.revision must be an integer of at least 1")
    digest = value.get("digest")
    if not isinstance(digest, str) or not _is_sha256_ref(digest):
        errors.append(f"{label}.digest must be a sha256 reference")
    elif digest != input_manifest_digest(value):
        errors.append(f"{label}.digest does not match the manifest content")
    _validate_scope(value.get("scope"), errors, f"{label}.scope")
    _validate_scalar_map(value.get("derived_from"), _MANIFEST_DERIVED_FROM_KEYS, errors, f"{label}.derived_from")
    _validate_budget(value.get("budget"), errors, f"{label}.budget")
    _validate_items(value.get("items"), errors, f"{label}.items")
    _validate_excluded_items(value.get("excluded_items"), errors, f"{label}.excluded_items")
    return errors


def validate_handoff_input_manifest_pin(value: Any, *, label: str = "input_manifest_pin") -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    _allowed_keys(value, _MANIFEST_PIN_KEYS, errors, label)
    if value.get("schema_version") != HANDOFF_INPUT_MANIFEST_PIN_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {HANDOFF_INPUT_MANIFEST_PIN_SCHEMA_VERSION}")
    if not isinstance(value.get("manifest_id"), str) or not value.get("manifest_id"):
        errors.append(f"{label}.manifest_id must be a non-empty string")
    revision = value.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        errors.append(f"{label}.revision must be an integer of at least 1")
    if not isinstance(value.get("digest"), str) or not _is_sha256_ref(str(value.get("digest", ""))):
        errors.append(f"{label}.digest must be a sha256 reference")
    return errors


def manifest_file_refs(manifest: object) -> list[str]:
    """The workspace-relative refs of the file items a manifest carries, in order."""
    if not isinstance(manifest, dict):
        return []
    refs: list[str] = []
    for item in manifest.get("items", []) or []:
        if not isinstance(item, dict) or item.get("item_kind") != "file":
            continue
        provenance = item.get("provenance")
        if isinstance(provenance, dict) and provenance.get("local_ref"):
            refs.append(str(provenance["local_ref"]))
    return refs


def pinned_input_manifest(manifest: dict[str, Any]) -> dict[str, object]:
    """The copy a handoff carries: validated, then detached from the caller's manifest.

    The deep copy is what makes attachment a pin. A caller that keeps editing its
    manifest afterwards changes its own object, never the one the handoff
    recorded, so the handoff keeps naming the revision and digest it actually
    carried and the divergence stays detectable.
    """
    errors = validate_handoff_input_manifest(manifest, label="input_manifest")
    if errors:
        raise ValueError("; ".join(errors))
    return deepcopy(manifest)


def input_manifest_summary(manifest: dict[str, Any]) -> dict[str, object]:
    """Counts and the pinned identity, for surfaces that report rather than carry."""
    if not manifest:
        return {}
    budget = manifest.get("budget") if isinstance(manifest.get("budget"), dict) else {}
    return {
        "schema_version": str(manifest.get("schema_version", "")),
        "manifest_id": str(manifest.get("manifest_id", "")),
        "revision": int(manifest.get("revision", 0) or 0),
        "digest": str(manifest.get("digest", "")),
        "item_count": int(budget.get("item_count", 0) or 0),
        "excluded_count": int(budget.get("excluded_count", 0) or 0),
        "used_bytes": int(budget.get("used_bytes", 0) or 0),
        "budget_bytes": int(budget.get("budget_bytes", 0) or 0),
        "over_budget": bool(budget.get("over_budget", False)),
        "claim_boundary": str(manifest.get("claim_boundary", "")),
    }


def _validate_selection(selection: ManifestSelection) -> None:
    if selection.item_kind not in MANIFEST_ITEM_KINDS:
        raise ValueError(f"unsupported manifest item kind: {selection.item_kind}")
    allowed = SELECTOR_KINDS_BY_ITEM_KIND[selection.item_kind]
    if selection.selector_kind not in allowed:
        raise ValueError(f"manifest item kind {selection.item_kind} does not accept selector kind {selection.selector_kind}")
    if not selection.expression.strip():
        raise ValueError("manifest selector expression must not be empty")


def _resolve_selection(
    selection: ManifestSelection,
    *,
    root: Path | None,
    budget_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    selector = {"kind": selection.selector_kind, "expression": selection.expression}
    if selection.item_kind == "diff":
        return _resolve_diff(selection, selector)
    if root is None:
        raise ValueError(f"manifest item kind {selection.item_kind} requires a workspace_root")
    if selection.item_kind == "file":
        return _resolve_files(selection, selector, root=root, budget_bytes=budget_bytes)
    return _resolve_plan_section(selection, selector, root=root, budget_bytes=budget_bytes)


def _resolve_diff(
    selection: ManifestSelection,
    selector: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    item_id = _item_id("diff", selection.expression)
    if selection.content is None:
        return [], [
            _exclusion(
                item_id,
                "diff",
                selector,
                "unreadable_source",
                0,
                "No diff text was supplied for this revision range, and core omh expands none itself.",
            )
        ]
    return [
        _row(
            item_id=item_id,
            item_kind="diff",
            selector=selector,
            source="working_tree_diff",
            local_ref=selection.expression,
            truth_level=LOCAL_SOURCE_TRUTH_LEVEL,
            inclusion_reason="explicit_selection",
            content=selection.content,
        )
    ], []


def _resolve_files(
    selection: ManifestSelection,
    selector: dict[str, str],
    *,
    root: Path,
    budget_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    expression = selection.expression
    if _escapes_workspace(expression):
        return [], [
            _exclusion(
                _item_id("file", expression),
                "file",
                selector,
                "outside_workspace",
                0,
                "The selector is absolute or walks above the workspace root.",
            )
        ]
    if _workspace_root_is_symlink(root):
        return [], [
            _exclusion(
                _item_id("file", expression),
                "file",
                selector,
                "unreadable_source",
                0,
                "The workspace root changed or is a symlink.",
            )
        ]
    if selection.selector_kind == "path":
        candidates = [root / expression]
    else:
        try:
            candidates = sorted(root.glob(expression), key=lambda path: path.as_posix())
        except OSError:
            candidates = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    for candidate in candidates:
        row, failure = _read_workspace_file(
            candidate,
            selector=selector,
            root=root,
            budget_bytes=budget_bytes,
            item_kind="file",
            source="workspace_file",
            truth_level=LOCAL_SOURCE_TRUTH_LEVEL,
        )
        if row is not None:
            rows.append(row)
        if failure is not None:
            failures.append(failure)
    if not rows and not failures:
        failures.append(
            _exclusion(
                _item_id("file", expression),
                "file",
                selector,
                "unreadable_source",
                0,
                "The selector matched no readable file inside the workspace.",
            )
        )
    return rows, failures


def _resolve_plan_section(
    selection: ManifestSelection,
    selector: dict[str, str],
    *,
    root: Path,
    budget_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    expression = selection.expression
    item_id = _item_id("plan_section", expression)
    plan_ref, separator, heading = expression.partition("#")
    if not separator or not heading.strip() or not plan_ref.strip():
        return [], [
            _exclusion(
                item_id,
                "plan_section",
                selector,
                "unreadable_source",
                0,
                "A plan_heading selector must read as <plan path>#<heading line>.",
            )
        ]
    if _escapes_workspace(plan_ref):
        return [], [
            _exclusion(
                item_id,
                "plan_section",
                selector,
                "outside_workspace",
                0,
                "The plan path is absolute or walks above the workspace root.",
            )
        ]
    if _workspace_root_is_symlink(root):
        return [], [
            _exclusion(
                item_id,
                "plan_section",
                selector,
                "unreadable_source",
                0,
                "The workspace root changed or is a symlink.",
            )
        ]
    row, failure = _read_workspace_file(
        root / plan_ref,
        selector=selector,
        root=root,
        budget_bytes=budget_bytes,
        item_kind="plan_section",
        source="hermes_plan",
        truth_level=APPROVED_CONTEXT_TRUTH_LEVEL,
    )
    if row is None:
        refused = failure or _exclusion(item_id, "plan_section", selector, "unreadable_source", 0, "The plan file could not be read.")
        return [], [{**refused, "item_id": item_id}]
    section = _plan_section_text(str(row["content"]), heading.strip())
    if section is None:
        return [], [
            _exclusion(
                item_id,
                "plan_section",
                selector,
                "unreadable_source",
                0,
                "The plan file carries no heading line matching this selector.",
            )
        ]
    encoded = section.encode("utf-8")
    return [
        _row(
            item_id=item_id,
            item_kind="plan_section",
            selector=selector,
            source="hermes_plan",
            local_ref=f"{_relative_ref(root / plan_ref, root)}#{heading.strip()}",
            truth_level=APPROVED_CONTEXT_TRUTH_LEVEL,
            inclusion_reason="explicit_selection",
            content=section,
            byte_cost=len(encoded),
            digest=_sha256_ref(encoded),
        )
    ], []


def _read_workspace_file(
    candidate: Path,
    *,
    selector: dict[str, str],
    root: Path,
    budget_bytes: int,
    item_kind: str,
    source: str,
    truth_level: str,
) -> tuple[dict[str, Any] | None, dict[str, object] | None]:
    try:
        relative_path = candidate.relative_to(root)
    except ValueError:
        return None, _exclusion(
            _item_id(item_kind, candidate.name),
            item_kind,
            selector,
            "outside_workspace",
            0,
            "The selected path lies outside the declared workspace root.",
        )
    relative = relative_path.as_posix()
    item_id = _item_id(item_kind, relative or candidate.name)
    if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        return None, _exclusion(
            item_id,
            item_kind,
            selector,
            "unreadable_source",
            0,
            "The local source is not a readable file.",
        )
    try:
        data = _read_workspace_bytes(root, relative_path.parts, budget_bytes)
    except _WorkspaceFileRefusal as exc:
        reason = "over_budget" if exc.over_budget else "unreadable_source"
        return None, _exclusion(item_id, item_kind, selector, reason, exc.byte_cost, exc.detail)
    return (
        _row(
            item_id=item_id,
            item_kind=item_kind,
            selector=selector,
            source=source,
            local_ref=relative,
            truth_level=truth_level,
            inclusion_reason="explicit_selection",
            content=data.decode("utf-8", errors="replace"),
            byte_cost=len(data),
            digest=_sha256_ref(data),
        ),
        None,
    )


def _read_workspace_bytes(root: Path, parts: tuple[str, ...], budget_bytes: int) -> bytes:
    if _descriptor_relative_reads_supported():
        return _read_workspace_bytes_at(root, parts, budget_bytes)
    return _read_workspace_bytes_with_identity_checks(root, parts, budget_bytes)


def _descriptor_relative_reads_supported() -> bool:
    return bool(
        _NOFOLLOW_FLAG
        and _DIRECTORY_FLAG
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _read_workspace_bytes_at(root: Path, parts: tuple[str, ...], budget_bytes: int) -> bytes:
    try:
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode):
            raise _WorkspaceFileRefusal("The workspace root changed or is a symlink.")
        root_fd = os.open(
            root,
            os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG | _NOFOLLOW_FLAG,
        )
    except _WorkspaceFileRefusal:
        raise
    except OSError as exc:
        raise _WorkspaceFileRefusal("The workspace root could not be safely opened.") from exc
    directory_fds = [root_fd]
    edges: list[tuple[int, str, os.stat_result]] = []
    try:
        root_opened = os.fstat(root_fd)
        if stat.S_ISLNK(root_before.st_mode) or not _same_identity(root_before, root_opened):
            raise _WorkspaceFileRefusal("The workspace root changed or is a symlink.")
        for part in parts[:-1]:
            parent_fd = directory_fds[-1]
            before = _stat_at(part, parent_fd)
            if stat.S_ISLNK(before.st_mode):
                raise _WorkspaceFileRefusal("The selected path contains a symlink and was refused.")
            if not stat.S_ISDIR(before.st_mode):
                raise _WorkspaceFileRefusal("The selected path contains a non-directory component.")
            try:
                child_fd = os.open(
                    part,
                    os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG | _NOFOLLOW_FLAG,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise _open_refusal(exc) from exc
            directory_fds.append(child_fd)
            opened = os.fstat(child_fd)
            after = _stat_at(part, parent_fd)
            if not _same_identity(before, opened) or not _same_identity(opened, after):
                raise _WorkspaceFileRefusal(
                    "The local source changed or was replaced while it was selected."
                )
            edges.append((parent_fd, part, opened))
        data = _read_final_at(parts[-1], directory_fds[-1], budget_bytes)
        for parent_fd, part, opened in edges:
            if not _same_identity(opened, _stat_at(part, parent_fd)):
                raise _WorkspaceFileRefusal(
                    "The local source changed or was replaced while it was selected."
                )
        try:
            root_after = root.lstat()
        except OSError as exc:
            raise _WorkspaceFileRefusal(
                "The workspace root changed or was replaced while the source was selected."
            ) from exc
        if not _same_identity(root_opened, root_after):
            raise _WorkspaceFileRefusal(
                "The workspace root changed or was replaced while the source was selected."
            )
        return data
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _read_final_at(name: str, directory_fd: int, budget_bytes: int) -> bytes:
    before = _stat_at(name, directory_fd)
    if stat.S_ISLNK(before.st_mode):
        raise _WorkspaceFileRefusal("The selected path contains a symlink and was refused.")
    try:
        source_fd = open_binary(
            name,
            os.O_RDONLY | _CLOEXEC_FLAG | _NONBLOCK_FLAG | _NOFOLLOW_FLAG,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise _open_refusal(exc) from exc
    try:
        opened = os.fstat(source_fd)
        if not _same_identity(before, opened):
            raise _WorkspaceFileRefusal(
                "The local source changed or was replaced while it was selected."
            )
        return _read_validated_descriptor(
            source_fd,
            opened,
            budget_bytes,
            final_stat=lambda: _stat_at(name, directory_fd),
        )
    finally:
        os.close(source_fd)


def _read_workspace_bytes_with_identity_checks(
    root: Path,
    parts: tuple[str, ...],
    budget_bytes: int,
) -> bytes:
    path = root
    component_identities: list[tuple[Path, tuple[int, int, int]]] = []
    try:
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode):
            raise _WorkspaceFileRefusal("The workspace root changed or is a symlink.")
        for index, part in enumerate(parts):
            path = path / part
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise _WorkspaceFileRefusal("The selected path contains a symlink and was refused.")
            if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise _WorkspaceFileRefusal("The selected path contains a non-directory component.")
            component_identities.append((path, _identity(metadata)))
        source_fd = open_binary(
            path,
            os.O_RDONLY | _CLOEXEC_FLAG | _NONBLOCK_FLAG,
        )
    except _WorkspaceFileRefusal:
        raise
    except OSError as exc:
        raise _open_refusal(exc) from exc
    try:
        opened = os.fstat(source_fd)
        if component_identities[-1][1] != _identity(opened):
            raise _WorkspaceFileRefusal(
                "The local source changed or was replaced while it was selected."
            )
        data = _read_validated_descriptor(
            source_fd,
            opened,
            budget_bytes,
            final_stat=path.lstat,
        )
        for component, identity in component_identities[:-1]:
            if _identity(component.lstat()) != identity:
                raise _WorkspaceFileRefusal(
                    "The local source changed or was replaced while it was selected."
                )
        if not _same_identity(root_before, root.lstat()):
            raise _WorkspaceFileRefusal(
                "The workspace root changed or was replaced while the source was selected."
            )
        return data
    except OSError as exc:
        raise _WorkspaceFileRefusal("The local source could not be safely read.") from exc
    finally:
        os.close(source_fd)


def _read_validated_descriptor(
    source_fd: int,
    opened: os.stat_result,
    budget_bytes: int,
    *,
    final_stat: Any,
) -> bytes:
    if not stat.S_ISREG(opened.st_mode):
        raise _WorkspaceFileRefusal("The local source is not a readable regular file.")
    if opened.st_size > budget_bytes:
        raise _WorkspaceFileRefusal(
            f"This source costs {int(opened.st_size)} byte(s), over the whole manifest budget of {budget_bytes} byte(s).",
            byte_cost=int(opened.st_size),
            over_budget=True,
        )
    chunks: list[bytes] = []
    remaining = budget_bytes + 1
    try:
        while remaining:
            chunk = os.read(source_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(source_fd)
        named_after = final_stat()
    except OSError as exc:
        raise _WorkspaceFileRefusal("The local source could not be safely read.") from exc
    data = b"".join(chunks)
    if len(data) > budget_bytes:
        measured = max(len(data), int(after.st_size))
        raise _WorkspaceFileRefusal(
            f"This source costs {measured} byte(s), over the whole manifest budget of {budget_bytes} byte(s).",
            byte_cost=measured,
            over_budget=True,
        )
    if (
        _file_snapshot(opened) != _file_snapshot(after)
        or _cross_api_file_snapshot(after) != _cross_api_file_snapshot(named_after)
        or len(data) != after.st_size
    ):
        raise _WorkspaceFileRefusal(
            "The local source changed or was replaced while it was selected."
        )
    return data


def _stat_at(name: str, directory_fd: int) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise _open_refusal(exc) from exc


def _open_refusal(exc: OSError) -> _WorkspaceFileRefusal:
    if exc.errno in {errno.ELOOP, errno.EMLINK}:
        return _WorkspaceFileRefusal("The selected path contains a symlink and was refused.")
    return _WorkspaceFileRefusal("The local source could not be safely opened.")


def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino), stat.S_IFMT(metadata.st_mode))


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return _identity(left) == _identity(right)


def _file_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        *_identity(metadata),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _cross_api_file_snapshot(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    snapshot = _file_snapshot(metadata)
    if not _WINDOWS_PATH_CTIME_IS_BIRTHTIME:
        return snapshot
    return (*snapshot[:-1], 0)


def _resolve_context_pack(
    context_pack: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    if not isinstance(context_pack, dict):
        return [], []
    included = context_pack.get("included_context")
    if not isinstance(included, list):
        return [], []
    blocked = context_pack.get("blocked_by_conflicts")
    pack_blocked = bool(isinstance(blocked, list) and blocked)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    for entry in included:
        if not isinstance(entry, dict):
            continue
        pack_item_id = str(entry.get("item_id", ""))
        if not pack_item_id:
            continue
        selector = {"kind": "memory_record_id", "expression": pack_item_id}
        item_id = _item_id("reviewed_memory", pack_item_id)
        if pack_blocked:
            failures.append(
                _exclusion(
                    item_id,
                    "reviewed_memory",
                    selector,
                    "blocked_by_unresolved_conflict",
                    0,
                    "The context pack carries unresolved conflicts, so its reviewed memory cannot travel.",
                )
            )
            continue
        rows.append(
            _row(
                item_id=item_id,
                item_kind="reviewed_memory",
                selector=selector,
                # Provenance is carried from the pack, never re-derived, so the
                # manifest cannot grade a record differently than the pack did.
                # A pack entry that names neither falls back to the lowest pair
                # the memory vocabulary has: unknown provenance is the least
                # trusted provenance, not the most.
                source=str(entry.get("source", "")) or "wrapper_snapshot",
                # `key`, never the pack's `artifact_ref`. A pack may carry an
                # absolute artifact path -- `build_plan_handoff_context_pack`
                # always does -- and such a path is wrong here twice over. It is
                # machine-specific, so it would put the operator's home
                # directory into the digest and two checkouts of the same
                # package would disagree; and it carries whatever that directory
                # is named, which on a home directory named like an email
                # address is exactly what the ref screen refuses, so the item
                # would drop out as unsafe on one machine and travel on the
                # next. The pack still carries the path for anyone who needs it.
                local_ref=str(entry.get("key", "")) or pack_item_id,
                truth_level=str(entry.get("truth_level", "")) or "supplied_hint",
                inclusion_reason="reviewed_memory_projection",
                content=str(entry.get("summary", "")),
            )
        )
    return rows, failures


def _row(
    *,
    item_id: str,
    item_kind: str,
    selector: dict[str, str],
    source: str,
    local_ref: str,
    truth_level: str,
    inclusion_reason: str,
    content: str,
    byte_cost: int | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    encoded = content.encode("utf-8")
    return {
        "item_id": item_id,
        "item_kind": item_kind,
        "selector": selector,
        "source": source,
        "local_ref": local_ref,
        "truth_level": truth_level,
        "inclusion_reason": inclusion_reason,
        "content": content,
        "byte_cost": len(encoded) if byte_cost is None else byte_cost,
        "hash": _sha256_ref(encoded) if digest is None else digest,
    }


def _exclusion(
    item_id: str,
    item_kind: str,
    selector: dict[str, str],
    reason: str,
    byte_cost: int,
    detail: str,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "item_kind": item_kind,
        "selector": dict(selector),
        "reason": reason,
        "byte_cost": int(byte_cost),
        "detail": detail[:MAX_DETAIL_CHARS],
    }


def _derived_from(context_pack: dict[str, Any] | None) -> dict[str, object]:
    if not isinstance(context_pack, dict):
        return {}
    return {
        "schema_version": str(context_pack.get("schema_version", "")),
        "executor_target": str(context_pack.get("executor_target", "")),
        "session_id": str(context_pack.get("session_id", "")),
        "included_context_count": _count(context_pack.get("included_context")),
        "excluded_context_count": _count(context_pack.get("excluded_context")),
        "blocked_by_conflicts_count": _count(context_pack.get("blocked_by_conflicts")),
    }


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _manifest_id(executor_target: str, session_id: str, scope: dict[str, str]) -> str:
    seed = "\n".join([str(executor_target), str(session_id), scope["kind"], scope["ref"]])
    return f"manifest-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _normalized_scope(scope: dict[str, str] | None) -> dict[str, str]:
    kind = str((scope or {}).get("kind", "") or "project")
    ref = str((scope or {}).get("ref", "") or "default")
    return {"kind": kind, "ref": ref}


def _resolved_workspace_root(workspace_root: str | Path | None) -> Path | None:
    if workspace_root is None:
        return None
    return Path(os.path.abspath(Path(workspace_root).expanduser()))


def _workspace_root_is_symlink(root: Path) -> bool:
    try:
        return stat.S_ISLNK(root.lstat().st_mode)
    except OSError:
        return False


def _escapes_workspace(expression: str) -> bool:
    """Cheap syntactic refusal for a selector that does not stay inside the workspace.

    Deliberately syntactic and platform-agnostic: a leading `\\` is root-anchored
    on Windows and a leading `/` is root-anchored on POSIX, so both spellings are
    absolute here whichever platform reads them, and a drive letter counts too.
    `_inside` still re-checks the resolved path, so this is the first refusal
    rather than the only one.
    """
    text = expression.strip()
    absolute = text.startswith(("/", "\\")) or (len(text) > 1 and text[1] == ":")
    return absolute or ".." in text.replace("\\", "/").split("/")


def _relative_ref(candidate: Path, root: Path) -> str:
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.name


def _item_id(item_kind: str, ref: str) -> str:
    return f"{item_kind}:{ref}"[:MAX_ITEM_ID_CHARS]


def _sha256_ref(data: bytes) -> str:
    return f"{_SHA256_PREFIX}{hashlib.sha256(data).hexdigest()}"


def _is_sha256_ref(value: str) -> bool:
    if not value.startswith(_SHA256_PREFIX):
        return False
    hexpart = value[len(_SHA256_PREFIX) :]
    return len(hexpart) == 64 and all(char in "0123456789abcdef" for char in hexpart)


def _plan_section_text(plan_text: str, heading: str) -> str | None:
    lines = plan_text.splitlines()
    depth = len(heading) - len(heading.lstrip("#"))
    for index, line in enumerate(lines):
        if line.strip() != heading:
            continue
        section = [line]
        for following in lines[index + 1 :]:
            stripped = following.lstrip()
            if stripped.startswith("#"):
                following_depth = len(stripped) - len(stripped.lstrip("#"))
                if following_depth <= depth:
                    break
            section.append(following)
        return "\n".join(section)
    return None


def _allowed_keys(value: dict[str, Any], allowed: set[str], errors: list[str], label: str) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        errors.append(f"{label} has unsupported keys: {extra}")
    missing = sorted(allowed - set(value))
    if missing:
        errors.append(f"{label} is missing required keys: {missing}")


def _validate_scope(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _allowed_keys(value, _MANIFEST_SCOPE_KEYS, errors, label)
    for key in ("kind", "ref"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{label}.{key} must be a non-empty string")


def _validate_scalar_map(value: Any, allowed: set[str], errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    extra = sorted(set(value) - allowed)
    if extra:
        errors.append(f"{label} has unsupported keys: {extra}")
    for key, nested in value.items():
        if not isinstance(nested, (str, int, bool)) and nested is not None:
            errors.append(f"{label}.{key} must be scalar metadata")


def _validate_budget(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _allowed_keys(value, _MANIFEST_BUDGET_KEYS, errors, label)
    if not isinstance(value.get("over_budget"), bool):
        errors.append(f"{label}.over_budget must be a boolean")
    for key in sorted(_MANIFEST_BUDGET_KEYS - {"over_budget"}):
        number = value.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            errors.append(f"{label}.{key} must be a non-negative integer")


def _validate_items(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} must be an object")
            continue
        _allowed_keys(item, _MANIFEST_ITEM_KEYS, errors, item_label)
        if not isinstance(item.get("item_id"), str) or not item.get("item_id"):
            errors.append(f"{item_label}.item_id must be a non-empty string")
        item_kind = item.get("item_kind")
        if item_kind not in MANIFEST_ITEM_KINDS:
            errors.append(f"{item_label}.item_kind must be one of {list(MANIFEST_ITEM_KINDS)}")
        _validate_provenance(item.get("provenance"), errors, f"{item_label}.provenance")
        _validate_selector(item.get("selector"), item_kind, errors, f"{item_label}.selector")
        if not isinstance(item.get("hash"), str) or not _is_sha256_ref(str(item.get("hash", ""))):
            errors.append(f"{item_label}.hash must be a sha256 reference")
        byte_cost = item.get("byte_cost")
        if not isinstance(byte_cost, int) or isinstance(byte_cost, bool) or byte_cost < 0:
            errors.append(f"{item_label}.byte_cost must be a non-negative integer")
        if item.get("inclusion_reason") not in MANIFEST_INCLUSION_REASONS:
            errors.append(f"{item_label}.inclusion_reason must be one of {list(MANIFEST_INCLUSION_REASONS)}")
        _validate_safety_result(item.get("safety_result"), errors, f"{item_label}.safety_result")
        # Content is not stored, so the validator cannot re-screen it -- an
        # included item's own claim plus the classifier that made it is what
        # travels. The refs are stored, so they are re-screened here with the
        # same predicate the builder used. A manifest arriving from a wrapper
        # therefore cannot assert `safe` over a ref the builder would have
        # refused.
        _validate_item_refs(item, errors, item_label)


def _validate_item_refs(item: dict[str, Any], errors: list[str], label: str) -> None:
    provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
    selector = item.get("selector") if isinstance(item.get("selector"), dict) else {}
    for field, ref in (("selector.expression", selector.get("expression")), ("provenance.local_ref", provenance.get("local_ref"))):
        if not isinstance(ref, str) or not ref:
            continue
        try:
            ensure_safe_opaque_ref_content(ref, "manifest_ref")
        except ValueError as exc:
            errors.append(f"{label}.{field} was refused by the context-safety screen as {exc}")


def _validate_provenance(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _allowed_keys(value, _MANIFEST_PROVENANCE_KEYS, errors, label)
    for key in ("source", "truth_level"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{label}.{key} must be a non-empty string")
    if not isinstance(value.get("local_ref"), str):
        errors.append(f"{label}.local_ref must be a string")


def _validate_selector(value: Any, item_kind: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _allowed_keys(value, _MANIFEST_SELECTOR_KEYS, errors, label)
    kind = value.get("kind")
    if kind not in MANIFEST_SELECTOR_KINDS:
        errors.append(f"{label}.kind must be one of {list(MANIFEST_SELECTOR_KINDS)}")
    elif isinstance(item_kind, str) and item_kind in SELECTOR_KINDS_BY_ITEM_KIND and kind not in SELECTOR_KINDS_BY_ITEM_KIND[item_kind]:
        errors.append(f"{label}.kind {kind} cannot select a {item_kind} item")
    if not isinstance(value.get("expression"), str) or not value.get("expression"):
        errors.append(f"{label}.expression must be a non-empty string")


def _validate_safety_result(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _allowed_keys(value, _MANIFEST_SAFETY_KEYS, errors, label)
    status = value.get("status")
    if status not in MANIFEST_SAFETY_STATUSES:
        errors.append(f"{label}.status must be one of {list(MANIFEST_SAFETY_STATUSES)}")
    elif status != SAFE_SAFETY_STATUS:
        errors.append(f"{label}.status must be {SAFE_SAFETY_STATUS} for an included item")
    if value.get("classifier") not in AUTHORITY_CLASSIFIERS:
        errors.append(f"{label}.classifier must be one of {list(AUTHORITY_CLASSIFIERS)}")
    if not isinstance(value.get("detail"), str):
        errors.append(f"{label}.detail must be a string")


def _validate_excluded_items(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} must be an object")
            continue
        _allowed_keys(item, _MANIFEST_EXCLUDED_KEYS, errors, item_label)
        if not isinstance(item.get("item_id"), str) or not item.get("item_id"):
            errors.append(f"{item_label}.item_id must be a non-empty string")
        if item.get("item_kind") not in MANIFEST_ITEM_KINDS:
            errors.append(f"{item_label}.item_kind must be one of {list(MANIFEST_ITEM_KINDS)}")
        _validate_selector(item.get("selector"), item.get("item_kind"), errors, f"{item_label}.selector")
        if item.get("reason") not in MANIFEST_EXCLUSION_REASONS:
            errors.append(f"{item_label}.reason must be one of {list(MANIFEST_EXCLUSION_REASONS)}")
        byte_cost = item.get("byte_cost")
        if not isinstance(byte_cost, int) or isinstance(byte_cost, bool) or byte_cost < 0:
            errors.append(f"{item_label}.byte_cost must be a non-negative integer")
        if not isinstance(item.get("detail"), str) or not item.get("detail"):
            errors.append(f"{item_label}.detail must be a non-empty string")

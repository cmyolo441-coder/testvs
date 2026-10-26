"""Route-scoped media input decisions for prepared coding handoffs.

This module stores metadata only.  It never receives attachment bytes or local
paths and never probes an executor/provider.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

try:  # Supports the registered direct-source demo import as well as omh.coding.
    from .executor_capability_snapshots import INPUT_MODALITY_CAPABILITY_NAMES
    from .executor_local_workflow_selection import evidence_reference
    from .pre_handoff_readiness import capability_evidence_is_fresh
except ImportError:  # pragma: no cover - exercised by the direct-source command.
    from omh.coding.executor_capability_snapshots import INPUT_MODALITY_CAPABILITY_NAMES
    from omh.coding.executor_local_workflow_selection import evidence_reference
    from omh.coding.pre_handoff_readiness import capability_evidence_is_fresh

HANDOFF_INPUT_REPRESENTATIONS: Final = (
    "text_only",
    "raw_media",
    "local_file_reference",
    "extracted_text",
    "ocr_output",
    "transcript",
    "normalized_other",
)
_MEDIA_MODALITIES: Final = frozenset({"image", "audio", "video", "document"})
_TEXT_REPRESENTATIONS: Final = frozenset({"text_only", "extracted_text", "ocr_output", "transcript", "normalized_other"})
# What a caller can hand over instead when a route lacks fresh evidence for a
# media modality.  Every alternative is a text representation the gate already
# knows: `extracted_text` is text Hermes read out of the document itself (its
# `read_file` tool reads PDF and office text); `ocr_output` and `transcript`
# additionally require an observed transformation before they dispatch.
ALTERNATIVE_REPRESENTATIONS_BY_MODALITY: Final = {
    "document": ("extracted_text", "ocr_output"),
    "image": ("ocr_output",),
    "audio": ("transcript",),
    "video": ("transcript",),
}
_ALTERNATIVE_REPRESENTATION_ROUTES: Final = {
    "extracted_text": (
        "extracted_text (text Hermes already read out of it, for example with its read_file tool; "
        "no transformation record is needed, and the route then needs only input_modality_text evidence)"
    ),
    "ocr_output": "ocr_output (with an observed OCR transformation, declared through the transformation record)",
    "transcript": "transcript (with an observed transcription, declared through the transformation record)",
}
# What a declared representation means before any owner's handoff has judged
# it.  A consumer reading `input_representation` off a payload that carries no
# handoff must not read "declared" as "cleared".
DECLARED_NOT_GATED_STATUS: Final = "declared_not_gated"
DECLARED_NOT_GATED_CLAIM_BOUNDARY: Final = (
    "A declared input representation is not a cleared modality gate. The gate runs on the selected "
    "coding owner's handoff, and no handoff has been prepared for this request yet."
)
# Attachment metadata -> declared media modality.  Names and declared media
# types are the only inputs: never bytes, never a local path, never task
# prose.  A type or suffix outside these tables declares nothing, so a text
# attachment or an unrecognised file keeps the additive gate silent rather
# than guessing a modality.
_DOCUMENT_INPUT_SUFFIXES: Final = frozenset(
    {"pdf", "doc", "docx", "odt", "rtf", "ppt", "pptx", "odp", "xls", "xlsx", "ods", "epub"}
)
_IMAGE_INPUT_SUFFIXES: Final = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "heic", "svg"})
_AUDIO_INPUT_SUFFIXES: Final = frozenset({"mp3", "wav", "m4a", "ogg", "oga", "flac", "aac", "opus"})
_VIDEO_INPUT_SUFFIXES: Final = frozenset({"mp4", "mov", "webm", "mkv", "avi", "m4v"})
_MODALITY_BY_INPUT_SUFFIX: Final = {
    suffix: modality
    for modality, suffixes in (
        ("document", _DOCUMENT_INPUT_SUFFIXES),
        ("image", _IMAGE_INPUT_SUFFIXES),
        ("audio", _AUDIO_INPUT_SUFFIXES),
        ("video", _VIDEO_INPUT_SUFFIXES),
    )
    for suffix in suffixes
}
_DOCUMENT_MEDIA_TYPES: Final = frozenset({"application/pdf", "application/msword", "application/rtf", "application/epub+zip"})
_DOCUMENT_MEDIA_TYPE_PREFIXES: Final = (
    "application/vnd.openxmlformats-officedocument.",
    "application/vnd.ms-",
    "application/vnd.oasis.opendocument.",
)
_MEDIA_TYPE_PREFIX_MODALITIES: Final = (("image/", "image"), ("audio/", "audio"), ("video/", "video"))
_ATTACHMENT_NAME_KEYS: Final = ("name", "filename", "file_name")
_ATTACHMENT_MEDIA_TYPE_KEYS: Final = ("media_type", "content_type", "mimetype", "mime_type")
_TRANSFORMATION_FIELDS: Final = frozenset({"kind", "status", "evidence_ref"})
_TRANSFORMATION_KIND_BY_REPRESENTATION: Final = {
    "ocr_output": "ocr",
    "transcript": "transcription",
}
DECISION_SCHEMA_VERSION: Final = "executor_modality_decision/v1"
# Every verdict other than `dispatch`, in the order a reader meets them: no
# route to scope evidence to, no usable evidence, evidence that says no, and a
# transformed-text handoff whose transformation nobody observed.  Consumers
# that key a status on the gate (fanout refusals, the coordination board, the
# plugin's graph readers) read this tuple rather than restating it.
FAIL_CLOSED_VERDICTS: Final = (
    "route_unresolved",
    "modality_unknown",
    "modality_unsupported",
    "modality_transformation_unobserved",
)
DECISION_CLAIM_BOUNDARY: Final = (
    "This route-scoped capability decision is metadata-only prepared context. It is not proof of attachment "
    "receipt, provider acceptance, dispatch, execution, verification, review, CI, or merge."
)


def attachment_input_modality(*, name: str = "", media_type: str = "") -> str:
    """The media modality one attachment declares through its type or name.

    A declared media type wins when it names a modality; otherwise the file
    suffix decides.  Returns "" for text and for anything outside the tables.
    """
    declared = str(media_type or "").strip().lower().split(";", 1)[0].strip()
    if declared:
        if declared in _DOCUMENT_MEDIA_TYPES or declared.startswith(_DOCUMENT_MEDIA_TYPE_PREFIXES):
            return "document"
        for prefix, modality in _MEDIA_TYPE_PREFIX_MODALITIES:
            if declared.startswith(prefix):
                return modality
    stem, separator, suffix = str(name or "").strip().rpartition(".")
    if not separator or not stem:
        return ""
    return _MODALITY_BY_INPUT_SUFFIX.get(suffix.strip().lower(), "")


def input_representations_from_attachments(attachments: object) -> list[str]:
    """`raw_media:<modality>` for every media attachment described by metadata rows.

    Each row may carry a name and a declared media type and nothing else is
    read.  Rows that describe text or an unrecognised file declare nothing.
    Every row is scanned, so a media file listed after any number of text
    files is still declared.  The result is deduplicated in first-seen order;
    an empty list means the handoff stays `text_only`.
    """
    if not isinstance(attachments, (list, tuple)):
        return []
    declared: list[str] = []
    # Every row is classified: the output is at most one entry per media
    # modality, so there is no declared-list cap to enforce and a media file
    # behind a long run of text attachments is never left unseen.
    for row in attachments:
        if not isinstance(row, Mapping):
            continue
        name = next((str(row[key]) for key in _ATTACHMENT_NAME_KEYS if isinstance(row.get(key), str)), "")
        media_type = next((str(row[key]) for key in _ATTACHMENT_MEDIA_TYPE_KEYS if isinstance(row.get(key), str)), "")
        modality = attachment_input_modality(name=name, media_type=media_type)
        representation = f"raw_media:{modality}"
        if modality and representation not in declared:
            declared.append(representation)
    return declared


def merged_input_representation(*declared: object) -> object:
    """Combine declared representations from several sources into one value.

    Empty and `text_only` entries drop out; the survivors keep first-seen
    order without duplicates.  With nothing left the handoff is `text_only`.
    """
    rows: list[str] = []
    for value in declared:
        values: Sequence[object] = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            text = str(item or "").strip()
            if text and text != "text_only" and text not in rows:
                rows.append(text)
    return rows if rows else "text_only"


def normalize_input_representation(value: object) -> tuple[dict[str, str], ...]:
    """Parse declared representation(s), never infer them from task prose."""
    values: Sequence[object] = value if isinstance(value, (list, tuple)) else (value,)
    rows: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, Mapping):
            representation = str(item.get("representation", "") or "").strip()
            modality = str(item.get("modality", "") or "").strip()
        else:
            parts = str(item or "").strip().split(":", 1)
            representation = parts[0].strip()
            modality = parts[1].strip() if len(parts) == 2 else ""
        if representation not in HANDOFF_INPUT_REPRESENTATIONS:
            raise ValueError("input representation must be one of: " + ", ".join(HANDOFF_INPUT_REPRESENTATIONS))
        if representation == "raw_media":
            if modality not in _MEDIA_MODALITIES:
                raise ValueError("raw_media requires one of: image, audio, video, document")
        elif representation == "local_file_reference":
            if modality not in _MEDIA_MODALITIES | {"text"}:
                raise ValueError("local_file_reference requires one of: text, image, audio, video, document")
        elif modality and modality != "text":
            raise ValueError("non-raw input representations may only declare text modality")
        rows.append({"representation": representation, "modality": modality or ("text" if representation in _TEXT_REPRESENTATIONS else "")})
    return tuple(rows)


def modality_requirements(value: object, route: Mapping[str, object] | None = None) -> tuple[dict[str, str], ...]:
    route_data = route if isinstance(route, Mapping) else {}
    provider = str(route_data.get("provider", route_data.get("model_family", "")) or "").strip()
    wire_model = str(route_data.get("wire_model", route_data.get("selected_model", "")) or "").strip()
    endpoint_mode = str(route_data.get("endpoint_mode", "default") or "default").strip()
    requirements: list[dict[str, str]] = []
    for row in normalize_input_representation(value):
        # Ordinary text handoffs predate modality evidence and carry no media
        # attachment.  This gate is intentionally additive: it applies only
        # where a declared media representation or transformed media crosses
        # an executor boundary.
        if row["representation"] == "text_only":
            continue
        modality = row["modality"]
        if not modality:
            continue
        capability = f"input_modality_{modality}"
        if capability not in INPUT_MODALITY_CAPABILITY_NAMES:
            continue
        requirement = {
            "capability": capability,
            "representation": row["representation"],
            "modality": modality,
            "provider": provider,
            "wire_model": wire_model,
            "endpoint_mode": endpoint_mode,
        }
        if requirement not in requirements:
            requirements.append(requirement)
    return tuple(requirements)


def build_executor_modality_decision(
    *,
    input_representation: object = "text_only",
    snapshot: Mapping[str, object] | None,
    route: Mapping[str, object] | None = None,
    now: str = "",
    transformation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fail closed for media until fresh, exact-route evidence says otherwise."""
    requirements = modality_requirements(input_representation, route)
    normalized_route = {
        "executor": str((snapshot or {}).get("executor", "") or ""),
        "provider": requirements[0]["provider"] if requirements else "",
        "wire_model": requirements[0]["wire_model"] if requirements else "",
    }
    representations = normalize_input_representation(input_representation)
    transform = _normalized_transformation(transformation, representations)
    transform_status = str(transform.get("status", "") or "")
    if any(row["representation"] in _TRANSFORMATION_KIND_BY_REPRESENTATION for row in representations) and transform_status != "observed":
        return _decision(requirements, normalized_route, "modality_transformation_unobserved", transformation=transform)
    if requirements and not (normalized_route["provider"] and normalized_route["wire_model"]):
        # Modality evidence is scoped to an exact provider and wire model.  With
        # no resolved route there is nothing a recorded snapshot could match, so
        # the gate says that rather than asking for evidence of an empty route.
        return _decision(
            requirements, normalized_route, "route_unresolved",
            transformation=transform,
            fallback_reason="no resolved provider and wire model to scope modality evidence to",
            unmet_requirement=requirements[0],
        )
    entries = (snapshot or {}).get("capabilities", {})
    entries = entries if isinstance(entries, Mapping) else {}
    verdict = "dispatch"
    evidence_ref = ""
    observed_at = ""
    freshness = "not_required"
    fallback_reason = ""
    unmet: Mapping[str, str] | None = None
    for requirement in requirements:
        unmet = requirement
        entry = entries.get(requirement["capability"])
        entry = entry if isinstance(entry, Mapping) else {}
        scope = entry.get("scope") if isinstance(entry.get("scope"), Mapping) else {}
        exact_route = all(str(scope.get(key, "")) == requirement[key] for key in ("provider", "wire_model", "endpoint_mode"))
        status = str(entry.get("status", "unknown") or "unknown")
        observed_at = str(entry.get("observed_at", "") or "")
        evidence_ref = str(entry.get("evidence_ref", "") or "")
        fresh = bool(observed_at and capability_evidence_is_fresh(observed_at, now))
        freshness = "fresh" if fresh else "stale_or_unknown"
        if status == "unavailable" and exact_route and fresh:
            verdict, fallback_reason = "modality_unsupported", "fresh route evidence records this modality as unavailable"
            break
        if status != "host_observed" or not exact_route or not fresh:
            verdict, fallback_reason = "modality_unknown", "fresh exact-route modality evidence is required before media dispatch"
            break
    return _decision(
        requirements, normalized_route, verdict, evidence_ref, observed_at, freshness, transform, fallback_reason,
        unmet_requirement=unmet if verdict != "dispatch" else None,
    )


def _normalized_transformation(
    transformation: Mapping[str, object] | None,
    representations: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    expected_kinds = {
        _TRANSFORMATION_KIND_BY_REPRESENTATION[row["representation"]]
        for row in representations
        if row["representation"] in _TRANSFORMATION_KIND_BY_REPRESENTATION
    }
    if not expected_kinds:
        return {"kind": "", "status": "not_required", "evidence_ref": ""}
    expected_kind = next(iter(expected_kinds)) if len(expected_kinds) == 1 else ""
    raw = transformation if isinstance(transformation, Mapping) else {}
    kind = str(raw.get("kind", "") or "")
    status = str(raw.get("status", "") or "")
    raw_evidence_ref = raw.get("evidence_ref")
    safe_evidence_ref = raw_evidence_ref if isinstance(raw_evidence_ref, str) and evidence_reference(raw_evidence_ref) else ""
    valid = (
        set(raw) == _TRANSFORMATION_FIELDS
        and bool(expected_kind)
        and kind == expected_kind
        and status in {"observed", "prepared_not_observed"}
        and (status != "observed" or bool(safe_evidence_ref))
    )
    if not valid:
        return {"kind": expected_kind, "status": "invalid", "evidence_ref": ""}
    return {
        "kind": expected_kind,
        "status": status,
        "evidence_ref": safe_evidence_ref if status == "observed" else "",
    }


def _decision(
    requirements: Sequence[Mapping[str, str]], route: Mapping[str, str], verdict: str,
    evidence_ref: str = "", observed_at: str = "", freshness: str = "not_required",
    transformation: Mapping[str, object] | None = None, fallback_reason: str = "",
    unmet_requirement: Mapping[str, str] | None = None,
) -> dict[str, object]:
    transform = dict(transformation or {"kind": "", "status": "not_required", "evidence_ref": ""})
    remaining_user_action, alternatives = _remaining_user_action(verdict, unmet_requirement, transform)
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "required_representations": [dict(row) for row in requirements],
        "route": dict(route),
        "verdict": verdict,
        "evidence_ref": evidence_ref,
        "evidence_observed_at": observed_at,
        "freshness": freshness,
        "transformation": transform,
        "fallback_reason": fallback_reason,
        "remaining_user_action": remaining_user_action,
        "alternative_representations": alternatives,
        "claim_boundary": DECISION_CLAIM_BOUNDARY,
    }


def _remaining_user_action(
    verdict: str,
    unmet_requirement: Mapping[str, str] | None,
    transformation: Mapping[str, object],
) -> tuple[str, list[str]]:
    """The one step that unblocks a fail-closed decision, and the representations it names.

    The action names the unmet capability and the alternative representations
    for that modality.  It names no coding owner: the same sentence holds for
    every executor, because the gate is about route evidence, not the owner.
    """
    if verdict == "dispatch":
        return "", []
    if verdict == "modality_transformation_unobserved":
        kind = str(transformation.get("kind", "") or "") or "declared"
        return (
            f"record the observed {kind} transformation (status observed with a safe evidence_ref) "
            "before handing over the transformed text",
            [],
        )
    if unmet_requirement is None:
        return "record fresh route-scoped capability evidence or use observed transformed text", []
    modality = unmet_requirement["modality"]
    capability = unmet_requirement["capability"]
    alternatives = list(ALTERNATIVE_REPRESENTATIONS_BY_MODALITY.get(modality, ()))
    if modality == "text":
        # Text is the final representation -- it is where every alternative
        # above ends -- so the one thing left is the route's own text row.
        # Name it as the last stop rather than leaving an empty list to read
        # as a dead end.
        if verdict == "route_unresolved":
            action = (
                "bind a confirmed-active model route (provider, wire model, endpoint mode) for this coding owner; "
                "text is the final representation, so once the route is bound only its input_modality_text "
                "evidence row is needed"
            )
        elif verdict == "modality_unsupported":
            action = (
                "choose a route with fresh input_modality_text evidence; text is the final representation "
                "and has no further alternative"
            )
        else:
            action = (
                "record fresh route-scoped input_modality_text evidence for this route "
                "(omh coding capability-snapshot record with input_modality_text host_observed for this "
                "provider, wire model, and endpoint mode); text is the final representation, so this one row "
                "is all the handoff still needs"
            )
        return action, []
    if verdict == "route_unresolved":
        action = (
            "bind a confirmed-active model route (provider, wire model, endpoint mode) for this coding owner "
            f"before {capability} evidence can be recorded for it"
        )
    elif verdict == "modality_unsupported":
        action = f"choose a route with fresh {capability} evidence"
    else:
        action = f"record fresh route-scoped {capability} evidence for this route"
    if alternatives:
        action += f", or hand the {modality} over as " + " or ".join(
            _ALTERNATIVE_REPRESENTATION_ROUTES[alternative] for alternative in alternatives
        )
    return action, alternatives


def demo_media_handoff_decisions() -> dict[str, object]:
    """Deterministic public demonstration of supported and fail-closed routes."""
    route = {"provider": "demo", "wire_model": "vision-1", "endpoint_mode": "default"}
    supported = {"executor": "demo", "capabilities": {"input_modality_image": {"status": "host_observed", "scope": route, "evidence_ref": "operator:demo-image", "observed_at": "2026-09-03T00:00:00Z"}, "input_modality_text": {"status": "host_observed", "scope": route, "evidence_ref": "operator:demo-text", "observed_at": "2026-09-03T00:00:00Z"}}}
    return {
        "schema_version": "omh_media_handoff_decision_demo/v1",
        "supported": build_executor_modality_decision(input_representation="raw_media:image", snapshot=supported, route=route, now="2026-09-03T01:00:00Z"),
        "unknown": build_executor_modality_decision(input_representation="raw_media:image", snapshot={"executor": "demo", "capabilities": {}}, route=route, now="2026-09-03T01:00:00Z"),
        "unsupported": build_executor_modality_decision(input_representation="raw_media:image", snapshot={"executor": "demo", "capabilities": {"input_modality_image": {**supported["capabilities"]["input_modality_image"], "status": "unavailable"}}}, route=route, now="2026-09-03T01:00:00Z"),
        "transformed": build_executor_modality_decision(input_representation="ocr_output", snapshot=supported, route=route, now="2026-09-03T01:00:00Z", transformation={"kind": "ocr", "status": "observed", "evidence_ref": "operator:demo-ocr"}),
        "fallback_rechecked": build_executor_modality_decision(input_representation="raw_media:image", snapshot={"executor": "fallback", "capabilities": {}}, route={"provider": "fallback", "wire_model": "text-1", "endpoint_mode": "default"}, now="2026-09-03T01:00:00Z"),
        **_demo_document_decisions(),
    }


def _demo_document_decisions() -> dict[str, object]:
    """The document (PDF and office file) path, exercised the way the image path is."""
    route = {"provider": "demo", "wire_model": "reader-1", "endpoint_mode": "default"}
    observed = {"status": "host_observed", "scope": route, "observed_at": "2026-09-03T00:00:00Z"}
    supported = {
        "executor": "demo",
        "capabilities": {
            "input_modality_document": {**observed, "evidence_ref": "operator:demo-document"},
            "input_modality_text": {**observed, "evidence_ref": "operator:demo-text"},
        },
    }
    unsupported = {
        "executor": "demo",
        "capabilities": {"input_modality_document": {**supported["capabilities"]["input_modality_document"], "status": "unavailable"}},
    }
    now = "2026-09-03T01:00:00Z"
    return {
        "document_supported": build_executor_modality_decision(input_representation="raw_media:document", snapshot=supported, route=route, now=now),
        "document_unknown": build_executor_modality_decision(input_representation="raw_media:document", snapshot={"executor": "demo", "capabilities": {}}, route=route, now=now),
        "document_unsupported": build_executor_modality_decision(input_representation="raw_media:document", snapshot=unsupported, route=route, now=now),
        "document_extracted_text": build_executor_modality_decision(input_representation="extracted_text", snapshot=supported, route=route, now=now),
    }

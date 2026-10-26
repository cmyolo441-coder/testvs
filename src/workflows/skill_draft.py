"""Explicit teach-this-workflow drafts: reviewable, structurally inactive skill proposals.

A draft is what a user gets after they explicitly say "turn this into a skill".
It separates the instructions that stay fixed from the inputs that vary per run,
names the preconditions and stop conditions, carries the verification a reviewer
must see pass, and records which user-selected runs it came from and what was
redacted out of them.

Inactive is structural here, not a flag someone remembers to check:

- a draft is stored under `.omh/learning/skill-drafts/`, never under `skills/`,
  and `write_skill_draft` refuses any path that would land in a skills tree;
- a draft is never catalog data, so `builtin_definitions()` never grows and no
  generated `skills/*/SKILL.md` is written;
- the proposed name lives under `proposed_skill_name`, never `name`, so a
  consumer reading `definition["name"]` cannot pick a draft up by accident, and
  a name that collides with an installed skill fails validation;
- an inactive draft carries no copy-ready proposal at all - the proposal is
  minted by activation, so there is nothing to act on before review.

Activation needs two independent things and recomputes both: the generated-output
checks pass (the draft would project to a definition the catalog contract accepts)
AND an explicit human review approves it. Even activated, the draft is a proposal
a person or Hermes Agent `/learn` acts on - never an install.

The redaction denylist is imported from `workflow_learning`; this module does not
keep a second copy of it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..local_store import atomic_write_json, read_json_object
from ..paths import OmhPaths
from ..skills.catalog import SkillDefinition, builtin_definitions, omh_skill_display_name
from ..skills.render import frontmatter_description
from ..skills.validation import validate_catalog_contract, validate_skill_definition_contract
from .learning_candidate import detect_learning_signal, sanitize_learning_text
from .workflow_learning import EXPORT_FORBIDDEN_PAYLOAD_KEYS
from .workflow_learning_errors import WorkflowLearningError


SKILL_DRAFT_SCHEMA_VERSION = "skill_draft/v1"
SKILL_DRAFT_CHECK_SCHEMA_VERSION = "skill_draft_generated_output_check/v1"
SKILL_DRAFT_PROPOSAL_SCHEMA_VERSION = "skill_draft_proposal/v1"
SKILL_DRAFT_STATUS = "prepared_not_observed"
SKILL_DRAFT_RECORD_TYPE = "skill_draft"
SKILL_DRAFT_INACTIVE_STATE = "inactive"
SKILL_DRAFT_ACTIVE_STATE = "active_proposal"
SKILL_DRAFT_STATES = (SKILL_DRAFT_INACTIVE_STATE, SKILL_DRAFT_ACTIVE_STATE)
SKILL_DRAFT_REVIEW_DECISIONS = ("approve", "revise", "reject")
SKILL_DRAFT_REF_PREFIX = "omh-skill-draft"
SKILL_DRAFT_STORAGE_AREA = "omh_learning_skill_drafts"
SKILL_DRAFT_SELECTION = "explicit_user_selection"
SKILL_DRAFT_ACTIVATION_REQUIREMENTS = ("generated_output_checks_pass", "explicit_human_review_approval")
SKILL_DRAFT_SECTION_KEYS = (
    "fixed_instructions",
    "declared_inputs",
    "preconditions",
    "stop_conditions",
    "verification_steps",
)
SKILL_DRAFT_CHECK_NAMES = (
    "draft_schema",
    "catalog_contract",
    "projected_skill_definition",
    "rendered_frontmatter_description",
    "rendered_catalog_index_line",
)
# The generated catalog index renders one "- `<display-name>`: <description>"
# line per skill, and `tests/test_router_content.py` holds every such line under
# 400 bytes. A draft whose summary would blow that line does not render valid
# generated output, so it cannot activate until the summary is tightened.
SKILL_DRAFT_CATALOG_INDEX_LINE_BYTE_LIMIT = 400
# The one directory name a draft must never appear under. `skills/` is where the
# catalog renders installed guidance; a draft that landed there would be read as
# installed by every tap, install walk, and freshness gate in the repo.
SKILL_DRAFT_INSTALL_ROOT_GUARD = "skills"
SKILL_DRAFT_CLAIM_BOUNDARY = (
    "A skill draft is prepared_not_observed review material stored under .omh/learning/skill-drafts/. "
    "It is never written into skills/ and never becomes catalog data, so it cannot be read as an installed "
    "Hermes skill. It is not skill creation evidence, install evidence, workflow execution evidence, review "
    "evidence, CI evidence, or merge evidence."
)
SKILL_DRAFT_NOT_OBSERVED = (
    "skill installed",
    "skill rendered under skills/",
    "catalog registration",
    "Hermes Agent /learn execution",
    "workflow execution",
    "future behavior change",
)

# Slug shape for a proposed name: the same lowercase-hyphen form catalog names
# use, so a reviewer can compare the two without translating between them.
_PROPOSED_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,48}$")
_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.:-]+")


def build_skill_draft(
    message: str,
    *,
    source_runs: Sequence[str],
    proposed_skill_name: str,
    fixed_instructions: Sequence[str],
    declared_inputs: Sequence[Mapping[str, str]],
    preconditions: Sequence[str],
    stop_conditions: Sequence[str],
    verification_steps: Sequence[str],
    created_at: str = "",
    draft_id: str | None = None,
) -> dict[str, Any] | None:
    """Build one inactive skill draft from an explicit teach-this-workflow request.

    Returns `None` when `message` carries no explicit learning signal. That is
    the whole of AC1: passive activity has no other entry point into this module,
    so there is no path that records or converts it.

    `created_at` is a caller-supplied parameter rather than a wall-clock read so
    the payload stays comparable; `draft_id` is derived from the draft's content,
    never from the time it was built.
    """
    signal = detect_learning_signal(message)
    if signal is None:
        return None
    summary, redaction_categories = sanitize_learning_text(message)
    runs = _source_run_refs(source_runs)
    name = proposed_skill_name.strip()
    sections = {
        "fixed_instructions": _text_list(fixed_instructions),
        "declared_inputs": _declared_input_list(declared_inputs),
        "preconditions": _text_list(preconditions),
        "stop_conditions": _text_list(stop_conditions),
        "verification_steps": _text_list(verification_steps),
    }
    draft: dict[str, Any] = {
        "schema_version": SKILL_DRAFT_SCHEMA_VERSION,
        "record_type": SKILL_DRAFT_RECORD_TYPE,
        "draft_id": draft_id or _content_draft_id(name, runs, sections),
        "created_at": created_at,
        "status": SKILL_DRAFT_STATUS,
        "proposed_skill_name": name,
        "summary": summary or f"Reusable workflow draft proposed as {name}.",
        "instruction_set": sections,
        "provenance": {
            "selection": SKILL_DRAFT_SELECTION,
            "learning_signal": dict(signal),
            "source_runs": runs,
            "redactions": {
                "redaction_policy": "metadata_only",
                "transient_identifier_categories": list(redaction_categories),
                "denied_payload_keys": denied_payload_keys(),
                "raw_prompt_stored": False,
                "raw_transcript_stored": False,
            },
            "review": {
                "required": True,
                "human_approval_required": True,
                "decision": "pending",
                "allowed_decisions": list(SKILL_DRAFT_REVIEW_DECISIONS),
                "reviewer_ref": "",
                "reviewed_at": "",
            },
        },
        "lifecycle": {
            "state": SKILL_DRAFT_INACTIVE_STATE,
            "installed": False,
            "catalog_registered": False,
            "generated_skill_path": "",
            "storage_area": SKILL_DRAFT_STORAGE_AREA,
            "activation_requires": list(SKILL_DRAFT_ACTIVATION_REQUIREMENTS),
        },
        "not_observed": list(SKILL_DRAFT_NOT_OBSERVED),
        "claim_boundary": SKILL_DRAFT_CLAIM_BOUNDARY,
    }
    _raise_on_validation(draft)
    return draft


def review_skill_draft(
    draft: Mapping[str, Any],
    *,
    decision: str,
    reviewer_ref: str = "operator",
    reviewed_at: str = "",
    review_note: str = "",
) -> dict[str, Any]:
    """Record an explicit review decision, activating the draft only on approval.

    AC3 lives here. `approve` recomputes the generated-output checks instead of
    trusting anything the caller passed in, and raises when they fail, so a
    review on its own can never activate a draft that would not render a valid
    skill. `revise` and `reject` record the decision and leave (or return) the
    draft inactive, dropping any activation an earlier approval minted.
    """
    if decision not in SKILL_DRAFT_REVIEW_DECISIONS:
        raise WorkflowLearningError("skill draft review decision must be approve, revise, or reject")
    reviewer = reviewer_ref.strip()
    if not reviewer:
        raise WorkflowLearningError("skill draft review needs an explicit reviewer_ref")
    _raise_on_validation(draft)
    reviewed = json.loads(json.dumps(draft))
    # Every review starts from the inactive shape so a re-review recomputes the
    # activation instead of nesting the previous receipt inside the new one.
    reviewed["lifecycle"]["state"] = SKILL_DRAFT_INACTIVE_STATE
    reviewed.pop("activation", None)
    reviewed.pop("proposal", None)
    review = reviewed["provenance"]["review"]
    review["decision"] = decision
    review["reviewer_ref"] = reviewer
    review["reviewed_at"] = reviewed_at
    review.pop("review_note_sha256", None)
    review.pop("review_note_length", None)
    if review_note:
        review["review_note_sha256"] = hashlib.sha256(review_note.encode("utf-8")).hexdigest()
        review["review_note_length"] = len(review_note)
    if decision != "approve":
        _raise_on_validation(reviewed)
        return reviewed
    checks = check_skill_draft_generated_output(reviewed)
    if not checks["ok"]:
        failed = ", ".join(check["check"] for check in checks["checks"] if check["status"] == "failed")
        raise WorkflowLearningError(f"skill draft activation blocked by failed generated-output checks: {failed}")
    reviewed["lifecycle"]["state"] = SKILL_DRAFT_ACTIVE_STATE
    reviewed["activation"] = {
        "activated": True,
        "activated_at": reviewed_at,
        "reviewer_ref": reviewer,
        "requirements_met": list(SKILL_DRAFT_ACTIVATION_REQUIREMENTS),
        "generated_output_check": checks,
    }
    reviewed["proposal"] = build_skill_draft_proposal(reviewed)
    _raise_on_validation(reviewed)
    return reviewed


def check_skill_draft_generated_output(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Answer whether this draft would render valid generated skill output.

    Five checks, in dependency order: the draft validates, the catalog contract
    the draft would join is itself clean, the draft projects to a
    `SkillDefinition` the catalog's own per-definition validator accepts, that
    definition renders a usable single-line frontmatter description, and the
    catalog-index line it would produce fits the byte ceiling that surface
    enforces. A schema failure blocks the rest rather than letting a malformed
    draft raise inside the projection.
    """
    schema_errors = validate_skill_draft(draft)
    if schema_errors:
        blocked = "blocked by draft schema errors"
        return _check_payload(
            [
                _check("draft_schema", False, "; ".join(schema_errors)),
                *[_check(name, False, blocked) for name in SKILL_DRAFT_CHECK_NAMES[1:]],
            ]
        )

    checks = [_check("draft_schema", True, "skill_draft/v1 validated.")]

    catalog = validate_catalog_contract()
    checks.append(
        _check(
            "catalog_contract",
            bool(catalog.get("ok")),
            "catalog_validation/v1 is clean." if catalog.get("ok") else "; ".join(_strings(catalog.get("errors"))),
        )
    )

    definition = project_skill_draft_definition(draft)
    definition_errors = validate_skill_definition_contract(definition)
    checks.append(
        _check(
            "projected_skill_definition",
            not definition_errors,
            "Projected definition satisfies the catalog skill contract."
            if not definition_errors
            else "; ".join(definition_errors),
        )
    )

    description = frontmatter_description(definition)
    # Frontmatter is one line per key, so an embedded newline would break the
    # generated YAML rather than merely look untidy.
    description_ok = bool(description.strip()) and "\n" not in description and "\r" not in description
    checks.append(
        _check(
            "rendered_frontmatter_description",
            description_ok,
            f"Rendered {len(description)} description characters on one line."
            if description_ok
            else "Projected definition does not render a non-empty single-line frontmatter description.",
        )
    )

    index_line = f"- `{omh_skill_display_name(definition.name)}`: {definition.description}"
    index_bytes = len(index_line.encode("utf-8"))
    checks.append(
        _check(
            "rendered_catalog_index_line",
            index_bytes < SKILL_DRAFT_CATALOG_INDEX_LINE_BYTE_LIMIT,
            f"Catalog index line is {index_bytes} bytes, under the "
            f"{SKILL_DRAFT_CATALOG_INDEX_LINE_BYTE_LIMIT}-byte ceiling."
            if index_bytes < SKILL_DRAFT_CATALOG_INDEX_LINE_BYTE_LIMIT
            else (
                f"Catalog index line is {index_bytes} bytes, over the "
                f"{SKILL_DRAFT_CATALOG_INDEX_LINE_BYTE_LIMIT}-byte ceiling; shorten the draft summary."
            ),
        )
    )
    return _check_payload(checks)


def project_skill_draft_definition(draft: Mapping[str, Any]) -> SkillDefinition:
    """Project a draft onto the dataclass the generated skill output is built from.

    Every draft section lands on the field that means the same thing: declared
    inputs are the required inputs, stop conditions are the safety rules,
    verification steps are the quality bar, preconditions describe when to use
    it. Nothing is invented, and nothing is registered - the definition exists
    only for the duration of the check.
    """
    sections = _object(draft.get("instruction_set"))
    name = str(draft.get("proposed_skill_name", ""))
    inputs = tuple(
        f"{str(item.get('name', '')).strip()}: {str(item.get('description', '')).strip()}"
        for item in _objects(sections.get("declared_inputs"))
    )
    return SkillDefinition(
        name=name,
        description=str(draft.get("summary", "")),
        triggers=(name,),
        use_when=" ".join(_strings(sections.get("preconditions"))),
        required_inputs=inputs,
        safety_rules=tuple(_strings(sections.get("stop_conditions"))),
        quality_bar=tuple(_strings(sections.get("verification_steps"))),
    )


def build_skill_draft_proposal(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Render the copy-ready proposal an activated draft hands to a human or Hermes."""
    sections = _object(draft.get("instruction_set"))
    provenance = _object(draft.get("provenance"))
    redactions = _object(provenance.get("redactions"))
    review = _object(provenance.get("review"))
    run_ids = [str(run.get("run_id", "")) for run in _objects(provenance.get("source_runs"))]
    categories = _strings(redactions.get("transient_identifier_categories")) or ["none detected"]
    lines = [
        f"# Proposed OMH skill: {draft.get('proposed_skill_name', '')}",
        "",
        str(draft.get("summary", "")),
        "",
        "## Preconditions",
        *_bullets(sections.get("preconditions")),
        "",
        "## Declared inputs (vary per run)",
        *[
            f"- {str(item.get('name', '')).strip()}: {str(item.get('description', '')).strip()}"
            for item in _objects(sections.get("declared_inputs"))
        ],
        "",
        "## Fixed instructions (do not vary)",
        *[f"{index}. {step}" for index, step in enumerate(_strings(sections.get("fixed_instructions")), start=1)],
        "",
        "## Stop conditions",
        *_bullets(sections.get("stop_conditions")),
        "",
        "## Required verification",
        *_bullets(sections.get("verification_steps")),
        "",
        "## Provenance",
        f"- Source runs selected by the user: {', '.join(run_ids)}",
        f"- Transient identifiers redacted: {', '.join(categories)}",
        "- Raw prompts or transcripts stored: no",
        f"- Reviewed by: {str(review.get('reviewer_ref', ''))} (decision: {str(review.get('decision', ''))})",
        "",
        SKILL_DRAFT_CLAIM_BOUNDARY,
    ]
    return {
        "schema_version": SKILL_DRAFT_PROPOSAL_SCHEMA_VERSION,
        "status": SKILL_DRAFT_STATUS,
        "copy_text": "\n".join(lines),
        "claim_boundary": SKILL_DRAFT_CLAIM_BOUNDARY,
    }


def validate_skill_draft(draft: Mapping[str, Any]) -> list[str]:
    """Return every contract violation in one pass; empty means the draft is valid."""
    if not isinstance(draft, dict):
        return ["skill draft must be a JSON object"]
    errors: list[str] = []
    if draft.get("schema_version") != SKILL_DRAFT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SKILL_DRAFT_SCHEMA_VERSION}")
    if draft.get("record_type") != SKILL_DRAFT_RECORD_TYPE:
        errors.append(f"record_type must be {SKILL_DRAFT_RECORD_TYPE}")
    if draft.get("status") != SKILL_DRAFT_STATUS:
        errors.append(f"status must be {SKILL_DRAFT_STATUS}")
    if not str(draft.get("draft_id", "")).strip():
        errors.append("draft_id must be a non-empty string")
    if "name" in draft:
        errors.append("skill draft must not carry a `name` key; the proposal name belongs in proposed_skill_name")
    if not str(draft.get("summary", "")).strip():
        errors.append("summary must be a non-empty string")
    errors.extend(_proposed_name_errors(draft.get("proposed_skill_name")))
    errors.extend(_instruction_set_errors(draft.get("instruction_set")))
    errors.extend(_provenance_errors(draft.get("provenance")))
    errors.extend(_lifecycle_errors(draft))
    boundary = str(draft.get("claim_boundary", ""))
    if SKILL_DRAFT_STATUS not in boundary:
        errors.append(f"claim_boundary must preserve {SKILL_DRAFT_STATUS}")
    if "skills/" not in boundary:
        errors.append("claim_boundary must state that a draft is never written into skills/")
    errors.extend(_forbidden_payload_key_errors(draft))
    return errors


def denied_payload_keys() -> list[str]:
    """The workflow-learning redaction denylist, as recorded in draft provenance."""
    return sorted(EXPORT_FORBIDDEN_PAYLOAD_KEYS)


def skill_draft_is_active(draft: Mapping[str, Any]) -> bool:
    return _object(draft.get("lifecycle")).get("state") == SKILL_DRAFT_ACTIVE_STATE


def skill_draft_ref(draft_id: str) -> str:
    return f"{SKILL_DRAFT_REF_PREFIX}:{_safe_id(draft_id)}"


def skill_draft_path(paths: OmhPaths, draft_id: str) -> Path:
    return paths.learning_skill_drafts_dir / f"{_safe_id(draft_id)}.json"


def write_skill_draft(paths: OmhPaths, draft: Mapping[str, Any]) -> dict[str, Any]:
    _raise_on_validation(draft)
    path = skill_draft_path(paths, str(draft["draft_id"]))
    _reject_installed_skill_location(paths, path)
    record = dict(draft)
    atomic_write_json(path, record, private=True)
    return record


def show_skill_draft(paths: OmhPaths, draft_id: str) -> dict[str, Any]:
    record = read_json_object(skill_draft_path(paths, draft_id))
    if not record:
        raise FileNotFoundError(draft_id)
    _raise_on_validation(record)
    return record


def list_skill_drafts(paths: OmhPaths, *, limit: int | None = None) -> list[dict[str, Any]]:
    directory = paths.learning_skill_drafts_dir
    if not directory.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = read_json_object(path)
        if not record:
            continue
        _raise_on_validation(record)
        summaries.append(skill_draft_summary(record))
    if limit is None or limit < 0:
        return summaries
    return summaries[:limit]


def skill_draft_summary(draft: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _object(draft.get("lifecycle"))
    provenance = _object(draft.get("provenance"))
    review = _object(provenance.get("review"))
    return {
        "draft_id": str(draft.get("draft_id", "")),
        "skill_draft_ref": skill_draft_ref(str(draft.get("draft_id", ""))),
        "proposed_skill_name": str(draft.get("proposed_skill_name", "")),
        "summary": str(draft.get("summary", "")),
        "state": str(lifecycle.get("state", "")),
        "installed": lifecycle.get("installed") is True,
        "review_decision": str(review.get("decision", "")),
        "source_run_ids": [str(run.get("run_id", "")) for run in _objects(provenance.get("source_runs"))],
        "created_at": str(draft.get("created_at", "")),
    }


def _reject_installed_skill_location(paths: OmhPaths, path: Path) -> None:
    """Refuse to write a draft anywhere a skill installer would look.

    Nothing routes a draft toward `skills/` today. This exists so a later change
    to the storage path cannot quietly turn review material into something a tap
    or install walk would pick up. Both sides are resolved before comparing so a
    symlinked temp root or a drive-letter path does not defeat the check.
    """
    try:
        relative = path.resolve().relative_to(paths.omh_home.resolve())
    except ValueError as exc:
        raise WorkflowLearningError("a skill draft must be written inside the local OMH home") from exc
    if SKILL_DRAFT_INSTALL_ROOT_GUARD in {part.casefold() for part in relative.parts}:
        raise WorkflowLearningError("a skill draft must never be written under a skills/ directory")


def _proposed_name_errors(value: Any) -> list[str]:
    name = str(value or "").strip()
    if not _PROPOSED_NAME_PATTERN.fullmatch(name):
        return ["proposed_skill_name must be a lowercase-hyphen slug of 3-49 characters"]
    installed = {definition.name for definition in builtin_definitions()}
    installed |= {omh_skill_display_name(definition.name) for definition in builtin_definitions()}
    if name in installed or omh_skill_display_name(name) in installed:
        return [f"proposed_skill_name collides with an installed skill name: {name}"]
    return []


def _instruction_set_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        errors = ["instruction_set must be an object carrying every draft section"]
        return errors + [f"instruction_set.{key} is missing" for key in SKILL_DRAFT_SECTION_KEYS]
    errors: list[str] = []
    extra = sorted(set(value) - set(SKILL_DRAFT_SECTION_KEYS))
    if extra:
        errors.append(f"instruction_set has unsupported keys: {extra}")
    for key in SKILL_DRAFT_SECTION_KEYS:
        if key == "declared_inputs":
            errors.extend(_declared_input_errors(value.get(key)))
            continue
        errors.extend(_text_list_errors(value.get(key), f"instruction_set.{key}"))
    return errors


def _declared_input_errors(value: Any) -> list[str]:
    # A draft that declares nothing variable has not separated the reusable part
    # from the run-specific part, which is the whole reason the record exists.
    if not isinstance(value, list) or not value:
        return ["instruction_set.declared_inputs must name at least one variable input"]
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"instruction_set.declared_inputs[{index}] must be an object with name and description")
            continue
        extra = sorted(set(item) - {"name", "description"})
        if extra:
            errors.append(f"instruction_set.declared_inputs[{index}] has unsupported keys: {extra}")
        for field in ("name", "description"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"instruction_set.declared_inputs[{index}].{field} must be a non-empty string")
    return errors


def _text_list_errors(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{label} must be a non-empty list"]
    return [
        f"{label}[{index}] must be a non-empty string"
        for index, item in enumerate(value)
        if not isinstance(item, str) or not item.strip()
    ]


def _provenance_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["provenance must record the source runs, the redactions applied, and the review decision"]
    errors: list[str] = []
    if value.get("selection") != SKILL_DRAFT_SELECTION:
        errors.append(f"provenance.selection must be {SKILL_DRAFT_SELECTION}")
    signal = value.get("learning_signal")
    if not isinstance(signal, dict) or not str(signal.get("matched", "")).strip():
        errors.append("provenance.learning_signal must record the explicit user request that started the draft")
    runs = value.get("source_runs")
    if not isinstance(runs, list) or not runs:
        errors.append("provenance.source_runs must name at least one user-selected source run")
    else:
        for index, run in enumerate(runs):
            if not isinstance(run, dict) or not str(run.get("run_id", "")).strip():
                errors.append(f"provenance.source_runs[{index}] must carry a run_id")
    errors.extend(_redaction_errors(value.get("redactions")))
    errors.extend(_review_errors(value.get("review")))
    return errors


def _redaction_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["provenance.redactions must record what was removed from the selected runs"]
    errors: list[str] = []
    if value.get("redaction_policy") != "metadata_only":
        errors.append("provenance.redactions.redaction_policy must be metadata_only")
    if not isinstance(value.get("transient_identifier_categories"), list):
        errors.append("provenance.redactions.transient_identifier_categories must be a list")
    if value.get("denied_payload_keys") != denied_payload_keys():
        errors.append("provenance.redactions.denied_payload_keys must match the workflow-learning redaction denylist")
    for flag in ("raw_prompt_stored", "raw_transcript_stored"):
        if value.get(flag) is not False:
            errors.append(f"provenance.redactions.{flag} must be false")
    return errors


def _review_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["provenance.review must record the human review gate"]
    errors: list[str] = []
    for flag in ("required", "human_approval_required"):
        if value.get(flag) is not True:
            errors.append(f"provenance.review.{flag} must be true")
    if value.get("decision") not in ("pending", *SKILL_DRAFT_REVIEW_DECISIONS):
        errors.append("provenance.review.decision must be pending, approve, revise, or reject")
    if list(value.get("allowed_decisions") or ()) != list(SKILL_DRAFT_REVIEW_DECISIONS):
        errors.append("provenance.review.allowed_decisions must be approve, revise, reject")
    return errors


def _lifecycle_errors(draft: Mapping[str, Any]) -> list[str]:
    value = draft.get("lifecycle")
    if not isinstance(value, dict):
        return ["lifecycle must record the draft state and its never-installed boundary"]
    errors: list[str] = []
    state = value.get("state")
    if state not in SKILL_DRAFT_STATES:
        errors.append(f"lifecycle.state must be one of {list(SKILL_DRAFT_STATES)}")
    for flag in ("installed", "catalog_registered"):
        if value.get(flag) is not False:
            errors.append(f"lifecycle.{flag} must be false; a draft is never an installed skill")
    if value.get("generated_skill_path") != "":
        errors.append("lifecycle.generated_skill_path must stay empty; a draft never renders under skills/")
    if value.get("storage_area") != SKILL_DRAFT_STORAGE_AREA:
        errors.append(f"lifecycle.storage_area must be {SKILL_DRAFT_STORAGE_AREA}")
    if list(value.get("activation_requires") or ()) != list(SKILL_DRAFT_ACTIVATION_REQUIREMENTS):
        errors.append(f"lifecycle.activation_requires must be {list(SKILL_DRAFT_ACTIVATION_REQUIREMENTS)}")
    errors.extend(_activation_errors(draft, state))
    return errors


def _activation_errors(draft: Mapping[str, Any], state: Any) -> list[str]:
    activation = draft.get("activation")
    proposal = draft.get("proposal")
    if state != SKILL_DRAFT_ACTIVE_STATE:
        errors = []
        if activation is not None:
            errors.append("an inactive skill draft must not carry an activation receipt")
        if proposal is not None:
            errors.append("an inactive skill draft must not carry a copy-ready proposal")
        return errors
    errors = []
    if not isinstance(activation, dict) or activation.get("activated") is not True:
        errors.append("an active skill draft must carry an activation receipt")
    elif list(activation.get("requirements_met") or ()) != list(SKILL_DRAFT_ACTIVATION_REQUIREMENTS):
        errors.append(f"activation.requirements_met must be {list(SKILL_DRAFT_ACTIVATION_REQUIREMENTS)}")
    elif _object(activation.get("generated_output_check")).get("ok") is not True:
        errors.append("activation.generated_output_check must record a passing check payload")
    if not isinstance(proposal, dict) or proposal.get("schema_version") != SKILL_DRAFT_PROPOSAL_SCHEMA_VERSION:
        errors.append(f"an active skill draft must carry a {SKILL_DRAFT_PROPOSAL_SCHEMA_VERSION} proposal")
    if str(_object(_object(draft.get("provenance")).get("review")).get("decision", "")) != "approve":
        errors.append("an active skill draft must record an approve review decision")
    return errors


def _forbidden_payload_key_errors(value: Any, path: str = "") -> list[str]:
    """Reject any key the workflow-learning export denylist forbids, at any depth."""
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{path}.{key}" if path else str(key)
            if key in EXPORT_FORBIDDEN_PAYLOAD_KEYS:
                errors.append(f"{location} is a denied raw-content key for metadata-only records")
            errors.extend(_forbidden_payload_key_errors(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_forbidden_payload_key_errors(item, f"{path}[{index}]"))
    return errors


def _raise_on_validation(draft: Mapping[str, Any]) -> None:
    errors = validate_skill_draft(draft)
    if errors:
        raise WorkflowLearningError("; ".join(errors))


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "status": "passed" if passed else "failed", "detail": detail}


def _check_payload(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [check for check in checks if check["status"] == "failed"]
    return {
        "schema_version": SKILL_DRAFT_CHECK_SCHEMA_VERSION,
        "ok": not failed,
        "checks": checks,
        "errors": [f"{check['check']}: {check['detail']}" for check in failed],
    }


def _source_run_refs(values: Sequence[str]) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        run_id = str(value).strip()
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        runs.append({"run_id": run_id, "run_ref": f"runtime:{run_id}"})
    return runs


def _declared_input_list(values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {"name": str(item.get("name", "")).strip(), "description": str(item.get("description", "")).strip()}
        for item in values
        if isinstance(item, Mapping)
    ]


def _text_list(values: Sequence[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _bullets(value: Any) -> list[str]:
    return [f"- {item}" for item in _strings(value)]


def _content_draft_id(name: str, runs: list[dict[str, str]], sections: dict[str, Any]) -> str:
    seed = json.dumps({"name": name, "runs": runs, "sections": sections}, sort_keys=True)
    return f"sd-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:20]}"


def _safe_id(value: str) -> str:
    return _SAFE_ID_PATTERN.sub("-", str(value)).strip("-")[:120] or "record"


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]

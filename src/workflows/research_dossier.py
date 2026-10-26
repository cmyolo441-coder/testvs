"""Local dossier structure validation; never truth or source authentication."""

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, cast

from .research_claims import research_claims_errors
from .research_progress import research_progress_errors
from .research_references import reference_studies_errors


FIELDS = frozenset({"schema_version", "question", "scope", "claim_records", "progress",
                    "reference_studies", "plan_feed", "citation_policy"})


def research_dossier_errors(value: object) -> list[str]:
    """Check supplied decision context without retrieval, execution or writes."""
    if not isinstance(value, Mapping):
        return ["dossier must be a mapping"]
    value = cast(Mapping[str, Any], value)
    errors = []
    if set(value) != FIELDS:
        errors.append("dossier.keys: expected declared fields")
    if value.get("schema_version") != "deep_research_dossier/v1":
        errors.append("dossier.schema_version: expected deep_research_dossier/v1")
    for field in ("question", "scope"):
        text = value.get(field)
        if not isinstance(text, str) or not text.strip() or len(text) > 2048:
            errors.append(f"dossier.{field}: expected bounded nonempty text")
    if value.get("citation_policy") != {
        "ledger_scope": "run_private", "writer": "parent_only",
        "reset_owner": "parent_only", "fan_in": "register_and_remap",
    }:
        errors.append("dossier.citation_policy: require run-private parent-only serialized registration and remap")
    errors.extend(research_claims_errors(value.get("claim_records")))
    errors.extend(research_progress_errors(cast(Mapping[str, Any], value.get("progress"))))
    references = value.get("reference_studies")
    if not isinstance(references, list):
        errors.append("dossier.reference_studies: expected list")
    else:
        errors.extend(reference_studies_errors(references))
    if errors:
        return errors
    references = cast(list[dict[str, Any]], references)
    claim_records = value["claim_records"]
    evidence_ids = {item["id"] for item in claim_records["evidence"]}
    for study in references:
        for axis in ("mechanism", "tradeoffs", "limits"):
            for finding_index, finding in enumerate(study["findings"][axis]):
                for citation_index, _ in enumerate(finding["evidence"]):
                    ident = f'{study["id"]}:{axis}:{finding_index}:{citation_index}'
                    if ident in evidence_ids:
                        errors.append("dossier: ambiguous evidence identifier")
                    evidence_ids.add(ident)
    progress = value["progress"]
    for table, field in (("waves", "new_evidence_ids"), ("axes", "covered_evidence_ids")):
        for index, item in enumerate(progress[table]):
            if any(ident not in evidence_ids for ident in item[field]):
                errors.append(f"dossier.progress.{table}[{index}].{field}: unknown evidence")
    errors.extend(_plan_feed_errors(value["plan_feed"], claim_records, references, evidence_ids))
    return errors


def validate_research_dossier(value: object) -> dict:
    """Return deterministic structure results, keeping partial decision context usable."""
    errors = research_dossier_errors(value)
    result = {
        "schema_version": "research_dossier_validation/v1", "valid": not errors,
        "errors": errors, "source_authenticity": "not_verified",
        "claim_boundary": "structure only; not factual truth, source authenticity, observed reading or execution",
    }
    if not errors and isinstance(value, Mapping):
        result.update(
            question=value["question"], scope=value["scope"],
            research_status=value["progress"]["status"],
            claims_by_status={status: [claim["id"] for claim in value["claim_records"]["claims"]
                                      if claim["status"] == status]
                              for status in ("corroborated", "unresolved", "refuted")},
            plan_feed=deepcopy(value["plan_feed"]),
        )
    return result


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError("non-JSON numeric constant")


def main(argv: list[str] | None = None) -> int:
    """Agent/operator entry: read one local UTF-8 JSON file; never persist state."""
    args = sys.argv[1:] if argv is None else argv
    result = None
    error = "input: expected one JSON file argument"
    if len(args) == 1:
        try:
            path = Path(args[0])
            if not path.is_file():
                raise ValueError("not a regular file")
            with path.open("rb") as source:
                data = source.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise ValueError("input too large")
            value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                               parse_constant=_reject_constant)
        except (OSError, ValueError, RecursionError):
            error = "input: unreadable file or invalid UTF-8 JSON (maximum 2 MiB; unique fields)"
        else:
            result = validate_research_dossier(value)
    if result is None:
        result = validate_research_dossier(None)
        result["errors"] = [error]
        code = 2
    else:
        code = 0 if result["valid"] else 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return code


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 2048


def _texts(value: object) -> bool:
    return (isinstance(value, list) and all(_text(item) for item in value)
            and len(value) == len(set(value)))


def _plan_feed_errors(feed, records, references, evidence_ids) -> list[str]:
    fields = {"decision_drivers", "viable_options", "rejected_candidates", "risks", "open_questions"}
    if not isinstance(feed, Mapping) or set(feed) != fields:
        return ["dossier.plan_feed: expected declared fields"]
    errors = []
    for field in ("decision_drivers", "risks", "open_questions"):
        if not _texts(feed[field]) or (field == "decision_drivers" and not feed[field]):
            errors.append(f"dossier.plan_feed.{field}: expected bounded text list")
    claims = {item["id"]: item for item in records["claims"]}
    ref_ids = {item["id"] for item in references}
    evidence = {item["id"]: item for item in records["evidence"]}
    for field in ("viable_options", "rejected_candidates"):
        options = feed[field]
        if not isinstance(options, list):
            errors.append(f"dossier.plan_feed.{field}: expected list")
            continue
        for index, option in enumerate(options):
            path = f"dossier.plan_feed.{field}[{index}]"
            keys = {"summary", "claim_ids", "evidence_ids", "reference_ids"}
            if field == "rejected_candidates":
                keys.add("reason")
            if not isinstance(option, Mapping) or set(option) != keys:
                errors.append(f"{path}: expected declared option fields")
                continue
            for key in ("summary", "reason"):
                if key in keys and not _text(option[key]):
                    errors.append(f"{path}.{key}: expected bounded nonempty text")
            for key, known in (("claim_ids", claims), ("evidence_ids", evidence_ids),
                               ("reference_ids", ref_ids)):
                if not _texts(option[key]) or any(ident not in known for ident in option[key]):
                    errors.append(f"{path}.{key}: expected unique known identifiers")
            if not option["evidence_ids"]:
                errors.append(f"{path}.evidence_ids: evidence required")
            if all(_texts(option[key]) for key in ("claim_ids", "reference_ids", "evidence_ids")):
                for ident in option["evidence_ids"]:
                    if ident in evidence and evidence[ident]["claim_id"] not in option["claim_ids"]:
                        errors.append(f"{path}.evidence_ids: owning claim must be linked")
                    elif ident not in evidence and ident in evidence_ids:
                        if ident.rsplit(":", 3)[0] not in option["reference_ids"]:
                            errors.append(f"{path}.evidence_ids: owning reference study must be linked")
            if field == "viable_options" and _texts(option["claim_ids"]):
                if any(claims[ident]["status"] != "corroborated"
                       for ident in option["claim_ids"] if ident in claims):
                    errors.append(f"{path}.claim_ids: unresolved/refuted claims belong outside viable options")
                if _texts(option["evidence_ids"]) and any(
                    evidence[ident]["stance"] != "support" for ident in option["evidence_ids"]
                    if ident in evidence
                ):
                    errors.append(f"{path}.evidence_ids: viable options require support, not refutation")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())

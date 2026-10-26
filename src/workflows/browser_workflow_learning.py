from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Final, TypeAlias
from urllib.parse import urlsplit

from ..system.append_only_store import is_unsafe_metadata_line

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
TRACE_SCHEMA_VERSION: Final = "browser_workflow_trace/v1"
MAX_TRACE_BYTES: Final = 256 * 1024
MAX_STEPS: Final = 64
MAX_LOCATORS: Final = 8
MAX_FIXTURES: Final = 64
MAX_DEPTH: Final = 32
_ACTIONS: Final = frozenset(("navigate", "read", "click", "extract", "submit"))
_LOCATORS: Final = frozenset(("role", "label", "test_id", "attribute"))
_ATTRIBUTES: Final = frozenset(("aria-label", "data-testid", "name"))
_ID: Final = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_DIGEST: Final = re.compile(r"^[a-f0-9]{64}$")
_REF: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_VERSION: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}(?:/v?[0-9][A-Za-z0-9._-]{0,31})?$")
_STATUSES: Final = frozenset(("pending_approval", "approved", "stale", "quarantined", "not_suitable"))


class BrowserWorkflowTraceReference(dict[str, JsonValue]):
    """A reference minted only by project-bound trace resolution."""


class BrowserTraceError(ValueError):
    def __init__(self, reason: str, state: str = "rejected") -> None:
        self.reason, self.state = reason, state
        super().__init__(f"browser trace {state}: {reason}")


def canonical_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise BrowserTraceError("origin must be an absolute http(s) origin without credentials, path, query, or fragment")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise BrowserTraceError("origin hostname or port is invalid") from exc
    display = f"[{host}]" if ":" in host else host
    suffix = "" if port is None or (parsed.scheme.lower(), port) in {("http", 80), ("https", 443)} else f":{port}"
    return f"{parsed.scheme.lower()}://{display}{suffix}"


def trace_digest(record: Mapping[str, JsonValue]) -> str:
    return _digest_bytes(_json({key: value for key, value in record.items() if key not in {"trace_id", "digest", "lifecycle"}}))


def fixture_digest(fixture: Mapping[str, JsonValue]) -> str:
    return _digest_bytes(_json({key: value for key, value in fixture.items() if key != "digest"}))


def parse_browser_workflow_trace(raw: Mapping[str, JsonValue], *, project_identity: str | None = None) -> JsonObject:
    """Close and redact a trace.  This function is pure and never observes Git."""
    record = _strict_mapping(raw, "trace")
    _bounded(record, "trace")
    if set(record) != {"schema_version", "project", "origins", "adapter_version", "parser_version", "source", "steps", "output_schema", "fixtures", "metadata"}:
        raise BrowserTraceError("trace must contain only browser_workflow_trace/v1 input fields", "not_suitable")
    if record.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise BrowserTraceError("schema_version must be browser_workflow_trace/v1", "not_suitable")
    project = _strict_mapping(record.get("project"), "project")
    supplied_identity = _required_digest(project, "identity")
    if set(project) != {"identity"} or project_identity is not None and supplied_identity != project_identity:
        raise BrowserTraceError("project must be the verified canonical project identity", "not_suitable")
    origins = _origins(record.get("origins"))
    adapter, parser = _required_ref(record, "adapter_version"), _required_ref(record, "parser_version")
    steps = _steps(record.get("steps"))
    output_schema = _output_schema(record.get("output_schema"))
    source = _source(record.get("source"), supplied_identity, origins, adapter)
    fixtures = _fixtures(record.get("fixtures"), steps, output_schema, origins)
    # Metadata is intentionally a sink: no unmodelled browser data survives it.
    parsed: JsonObject = {
        "schema_version": TRACE_SCHEMA_VERSION, "project": {"identity": supplied_identity}, "origins": list(origins),
        "adapter_version": adapter, "parser_version": parser, "source": source, "steps": steps,
        "output_schema": output_schema, "fixtures": fixtures, "metadata": redact_browser_metadata(_strict_mapping(record.get("metadata"), "metadata")),
        "privacy": {"raw_material_persisted": False, "redaction": "closed_allowlist/v1"},
    }
    digest = trace_digest(parsed)
    parsed.update({"digest": digest, "trace_id": f"bwt-{digest[:24]}", "lifecycle": _lifecycle("pending_approval", "", 1, False, [])})
    _bounded(parsed, "redacted trace")
    return parsed


def validate_browser_workflow_trace(record: Mapping[str, JsonValue]) -> list[str]:
    """Return bounded structural errors; malformed persisted data never raises."""
    errors: list[str] = []
    try:
        candidate = _strict_mapping(record, "trace")
        _bounded(candidate, "trace")
        required = {"schema_version", "project", "origins", "adapter_version", "parser_version", "source", "steps", "output_schema", "fixtures", "metadata", "privacy", "digest", "trace_id", "lifecycle"}
        if set(candidate) != required:
            errors.append("trace must contain only canonical browser workflow trace fields")
            return errors
        identity = _required_digest(_strict_mapping(candidate.get("project"), "project"), "identity")
        if candidate.get("schema_version") != TRACE_SCHEMA_VERSION or set(_strict_mapping(candidate.get("project"), "project")) != {"identity"}:
            errors.append("schema_version or project identity is invalid")
        origins = _origins(candidate.get("origins"))
        adapter, parser = _required_ref(candidate, "adapter_version"), _required_ref(candidate, "parser_version")
        if parser != _text(candidate.get("parser_version")):
            errors.append("parser_version is invalid")
        source = _source(candidate.get("source"), identity, origins, adapter)
        steps = _steps(candidate.get("steps")); schema = _output_schema(candidate.get("output_schema"))
        fixtures = _fixtures(candidate.get("fixtures"), steps, schema, origins)
        if candidate.get("source") != source or candidate.get("steps") != steps or candidate.get("output_schema") != schema or candidate.get("fixtures") != fixtures:
            errors.append("trace contains non-canonical source, steps, output schema, or fixtures")
        if candidate.get("metadata") != {} or candidate.get("privacy") != {"raw_material_persisted": False, "redaction": "closed_allowlist/v1"}:
            errors.append("trace contains persisted raw browser metadata")
        digest = _required_digest(candidate, "digest")
        if digest != trace_digest(candidate) or candidate.get("trace_id") != f"bwt-{digest[:24]}":
            errors.append("digest or trace id does not bind canonical trace material")
        _validate_lifecycle(_strict_mapping(candidate.get("lifecycle"), "lifecycle"), digest)
    except BrowserTraceError as exc:
        errors.append(exc.reason)
    except (TypeError, ValueError, OverflowError, RecursionError):
        errors.append("trace is malformed")
    return errors[:8]


def replay_browser_workflow_trace(trace: Mapping[str, JsonValue], observation: Mapping[str, JsonValue]) -> JsonObject:
    """Replay one stored semantic fixture.  It is simulation, never browser execution."""
    if validate_browser_workflow_trace(trace):
        return _result("not_suitable", "invalid_trace")
    try:
        requested = _strict_mapping(observation, "replay request")
        _bounded(requested, "replay request")
        if set(requested) not in ({"fixture_id"}, {"fixture_id", "origin"}) or not _ID.fullmatch(_text(requested.get("fixture_id"))):
            raise BrowserTraceError("replay request must name one stored fixture and optional canonical origin", "not_suitable")
        lifecycle = _strict_mapping(trace.get("lifecycle"), "lifecycle")
        if lifecycle.get("status") in {"quarantined", "not_suitable"}:
            return _result(_text(lifecycle.get("status")), "terminal_lifecycle")
        fixture = next(item for item in _objects(trace.get("fixtures")) if item.get("fixture_id") == requested["fixture_id"])
        if "origin" in requested and canonical_origin(_origin_text(requested["origin"])) != fixture.get("origin"):
            return _result("quarantined", "host_mismatch", fixture)
        result = _simulate(_objects(trace.get("steps")), _strict_mapping(trace.get("output_schema"), "output_schema"), fixture)
        if result["status"] == "retry_once" and lifecycle.get("retry_used") is True:
            return _result("stale", "transient_retry_exhausted", fixture)
        if result["status"] == "replayed" and lifecycle.get("status") == "stale":
            return _result("stale", "prior_mismatch")
        return result
    except (BrowserTraceError, StopIteration, TypeError, ValueError, RecursionError):
        return _result("not_suitable", "invalid_observation")


def redact_browser_metadata(value: Mapping[str, JsonValue]) -> JsonObject:
    """Closed allowlist redaction: no free-form browser field is persistable."""
    _bounded(_strict_mapping(value, "metadata"), "metadata")
    return {}


def _source(value: JsonValue | None, identity: str, origins: list[str], adapter: str) -> JsonObject:
    source = _strict_mapping(value, "source")
    required = {"run_ref", "evidence_ref", "selected", "success", "binding", "lineage"}
    if set(source) != required or source.get("selected") is not True:
        raise BrowserTraceError("source must be an explicitly selected closed success proof")
    run_ref, evidence_ref = _required_ref(source, "run_ref"), _required_ref(source, "evidence_ref")
    success = _strict_mapping(source.get("success"), "source.success")
    if success.get("state") != "success" or set(success) != {"state", "evidence_digest"}:
        raise BrowserTraceError("source must contain terminal success evidence")
    evidence_digest = _required_digest(success, "evidence_digest")
    binding = _strict_mapping(source.get("binding"), "source.binding")
    expected_binding: JsonObject = {"project_identity": identity, "origin": _text(binding.get("origin")), "adapter_version": adapter}
    if binding != expected_binding or expected_binding["origin"] not in origins:
        raise BrowserTraceError("source binding must match verified project, origin, and adapter")
    lineage = _strict_mapping(source.get("lineage"), "source.lineage")
    if set(lineage) != {"source_digest", "environment_digest", "adapter_digest"}:
        raise BrowserTraceError("source lineage must retain source, environment, and adapter digests")
    return {"run_ref": run_ref, "evidence_ref": evidence_ref, "selected": True, "success": {"state": "success", "evidence_digest": evidence_digest}, "binding": expected_binding, "lineage": {key: _required_digest(lineage, key) for key in sorted(lineage)}}


def _steps(value: JsonValue | None) -> list[JsonValue]:
    if not isinstance(value, list) or not value or len(value) > MAX_STEPS:
        raise BrowserTraceError("steps must contain between 1 and 64 semantic actions", "not_suitable")
    output: list[JsonValue] = []
    for index, raw in enumerate(value):
        step = _strict_mapping(raw, f"steps[{index}]")
        if set(step) != {"action", "locators"} or _text(step.get("action")) not in _ACTIONS:
            raise BrowserTraceError(f"steps[{index}] is not a supported closed semantic action", "not_suitable")
        locators = step.get("locators")
        if not isinstance(locators, list) or not locators or len(locators) > MAX_LOCATORS:
            raise BrowserTraceError(f"steps[{index}].locators must contain between 1 and 8 candidates", "not_suitable")
        output.append({"action": _text(step.get("action")), "locators": [_locator(item, index) for item in locators]})
    return output


def _locator(value: JsonValue, index: int) -> JsonObject:
    locator = _strict_mapping(value, f"steps[{index}].locator")
    kind = _text(locator.get("kind"))
    if kind == "role" and set(locator) == {"kind", "role", "name"}:
        return {"kind": kind, "role": _semantic_text(locator.get("role")), "name": _semantic_text(locator.get("name"))}
    if kind in {"label", "test_id"} and set(locator) == {"kind", "name"}:
        return {"kind": kind, "name": _semantic_text(locator.get("name"))}
    if kind == "attribute" and set(locator) == {"kind", "attribute", "name"} and _text(locator.get("attribute")) in _ATTRIBUTES:
        return {"kind": kind, "attribute": _text(locator.get("attribute")), "name": _semantic_text(locator.get("name"))}
    raise BrowserTraceError(f"steps[{index}] locator is not an allowlisted semantic locator", "not_suitable")


def _output_schema(value: JsonValue | None) -> JsonObject:
    schema = _strict_mapping(value, "output_schema")
    fields = schema.get("fields")
    if set(schema) != {"kind", "fields"} or schema.get("kind") != "object" or not isinstance(fields, list) or not fields or len(fields) > MAX_STEPS:
        raise BrowserTraceError("output_schema must be a bounded named object schema")
    names = [_text(item) for item in fields]
    if any(not _ID.fullmatch(name) for name in names) or names != sorted(set(names)):
        raise BrowserTraceError("output_schema fields must be sorted unique identifiers")
    return {"kind": "object", "fields": list[JsonValue](names)}


def _fixtures(value: JsonValue | None, steps: list[JsonValue], schema: JsonObject, origins: list[str]) -> list[JsonValue]:
    if not isinstance(value, list) or len(value) < 2 or len(value) > MAX_FIXTURES:
        raise BrowserTraceError("fixtures must contain bounded positive and negative semantic data")
    output = [_fixture(item, origins) for item in value]
    if [_text(item["fixture_id"]) for item in output] != sorted({_text(item["fixture_id"]) for item in output}) or not {"positive", "negative"}.issubset({_text(item["kind"]) for item in output}):
        raise BrowserTraceError("fixtures must have unique sorted ids and include positive and negative cases")
    for fixture in output:
        simulated = _simulate(_objects(steps), schema, fixture)
        expected = "replayed" if fixture["kind"] == "positive" else simulated["status"] if fixture["kind"] == "negative" else "stale" if any(step.get("action") not in {"navigate", "read"} for step in _objects(steps)) else "retry_once"
        if simulated["status"] != expected or fixture["kind"] == "negative" and expected == "replayed":
            raise BrowserTraceError(f"{fixture['fixture_id']} does not prove its required fixture semantics", "not_suitable")
    return list[JsonValue](output)


def _fixture(value: JsonValue, origins: list[str]) -> JsonObject:
    fixture = _strict_mapping(value, "fixture")
    required = {"fixture_id", "kind", "origin", "nodes", "output_fields", "digest"}
    if set(fixture) != required or _text(fixture.get("kind")) not in {"positive", "negative", "transient"} or not _ID.fullmatch(_text(fixture.get("fixture_id"))):
        raise BrowserTraceError("fixture contains unsupported data")
    origin = canonical_origin(_origin_text(fixture.get("origin")))
    if origin not in origins:
        raise BrowserTraceError("fixture origin is not allowlisted")
    nodes = fixture.get("nodes")
    fields = fixture.get("output_fields")
    if not isinstance(nodes, list) or len(nodes) > MAX_STEPS * MAX_LOCATORS or not isinstance(fields, list) or len(fields) > MAX_STEPS:
        raise BrowserTraceError("fixture exceeds semantic bounds", "not_suitable")
    field_names = [_text(item) for item in fields]
    if field_names != sorted(set(field_names)) or any(not _ID.fullmatch(item) for item in field_names):
        raise BrowserTraceError("fixture output fields are invalid")
    parsed: JsonObject = {"fixture_id": _text(fixture.get("fixture_id")), "kind": _text(fixture.get("kind")), "origin": origin, "nodes": [_node(item) for item in nodes], "output_fields": list[JsonValue](field_names)}
    digest = _required_digest(fixture, "digest")
    if digest != fixture_digest(parsed):
        raise BrowserTraceError("fixture digest does not match redacted semantic fixture data")
    parsed["digest"] = digest
    return parsed


def _node(value: JsonValue) -> JsonObject:
    node = _strict_mapping(value, "fixture node")
    allowed = {"role", "name", "label", "test_id", "attributes"}
    if not node or set(node) - allowed or not all(isinstance(item, str) for key, item in node.items() if key != "attributes"):
        raise BrowserTraceError("fixture node contains raw or unsupported browser material")
    output: JsonObject = {key: _semantic_text(item) for key, item in node.items() if key != "attributes"}
    attributes = node.get("attributes", {})
    if not isinstance(attributes, Mapping) or set(attributes) - _ATTRIBUTES:
        raise BrowserTraceError("fixture node attributes are not allowlisted")
    if attributes:
        output["attributes"] = {str(key): _semantic_text(item) for key, item in attributes.items()}
    return output


def _simulate(steps: list[JsonObject], schema: JsonObject, fixture: JsonObject) -> JsonObject:
    for index, step in enumerate(steps):
        resolution = _resolve(_objects(step.get("locators")), _objects(fixture.get("nodes")))
        if resolution == "zero":
            return _result("stale", "zero_locator", fixture)
        if resolution == "ambiguous":
            return _result("quarantined" if step.get("action") in {"click", "submit"} else "stale", "ambiguous_locator", fixture)
    if fixture.get("output_fields") != schema.get("fields"):
        return _result("stale", "schema_mismatch", fixture)
    if fixture.get("kind") == "transient":
        if all(step.get("action") in {"navigate", "read"} for step in steps):
            return _result("retry_once", "transient_read_or_navigation", fixture)
        return _result("stale", "unsafe_transient_retry", fixture)
    return _result("replayed", "fixture_matched", fixture)


def _resolve(locators: list[JsonObject], nodes: list[JsonObject]) -> str:
    """Every declared candidate must identify the same unique semantic node."""
    selected: int | None = None
    for locator in locators:
        matches = [index for index, node in enumerate(nodes) if _matches(locator, node)]
        if not matches:
            return "zero"
        if len(matches) > 1:
            return "ambiguous"
        if selected is not None and selected != matches[0]:
            return "ambiguous"
        selected = matches[0]
    return "one" if selected is not None else "zero"


def _matches(locator: JsonObject, node: JsonObject) -> bool:
    kind, name = locator.get("kind"), locator.get("name")
    if kind == "role": return node.get("role") == locator.get("role") and node.get("name") == name
    if kind == "label": return node.get("label") == name
    if kind == "test_id": return node.get("test_id") == name
    return _strict_mapping(node.get("attributes", {}), "fixture node attributes").get(_text(locator.get("attribute"))) == name


def _lifecycle(status: str, approved_digest: str, revision: int, retry_used: bool, mismatches: list[str]) -> JsonObject:
    return {"revision": revision, "status": status, "approved_digest": approved_digest, "retry_used": retry_used, "mismatch_fixture_digests": list(mismatches)}


def _validate_lifecycle(value: JsonObject, digest: str) -> None:
    if set(value) != {"revision", "status", "approved_digest", "retry_used", "mismatch_fixture_digests"}:
        raise BrowserTraceError("lifecycle must contain only canonical fields")
    revision, status, approved, retry_used, mismatches = value.get("revision"), _text(value.get("status")), _text(value.get("approved_digest")), value.get("retry_used"), value.get("mismatch_fixture_digests")
    if not isinstance(revision, int) or isinstance(revision, bool) or not 1 <= revision <= MAX_FIXTURES or status not in _STATUSES or not isinstance(retry_used, bool) or not isinstance(mismatches, list) or len(mismatches) > MAX_FIXTURES:
        raise BrowserTraceError("lifecycle is invalid")
    mismatch_digests = [_text(item) for item in mismatches]
    if mismatches != mismatch_digests or any(not _DIGEST.fullmatch(item) for item in mismatch_digests) or mismatch_digests != sorted(set(mismatch_digests)):
        raise BrowserTraceError("lifecycle mismatch identities are invalid")
    if not isinstance(value["status"], str) or value["status"] != status or not isinstance(value["approved_digest"], str) or value["approved_digest"] != approved:
        raise BrowserTraceError("lifecycle status and approval must be canonical strings")
    if status == "approved" and approved != digest or status != "approved" and approved:
        raise BrowserTraceError("approval must bind exactly the approved trace digest")
    if status in {"pending_approval", "approved"} and mismatches or status == "stale" and not mismatches:
        raise BrowserTraceError("lifecycle drift history is inconsistent")


def _origins(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BrowserTraceError("origins must include a canonical allowlist")
    origins = [canonical_origin(_origin_text(item)) for item in value]
    if origins != sorted(set(origins)):
        raise BrowserTraceError("origins must be sorted unique canonical origins")
    return origins


def _required_ref(record: Mapping[str, JsonValue], key: str) -> str:
    value = _text(record.get(key))
    pattern = _VERSION if key in {"adapter_version", "parser_version"} else _REF
    if not pattern.fullmatch(value): raise BrowserTraceError(f"{key} must be a bounded reference")
    return value


def _required_digest(record: Mapping[str, JsonValue], key: str) -> str:
    value = _text(record.get(key))
    if not _DIGEST.fullmatch(value): raise BrowserTraceError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _semantic_text(value: JsonValue | None) -> str:
    text = _text(value)
    if not text or len(text) > 128 or is_unsafe_metadata_line(text) or "?" in text or "#" in text:
        raise BrowserTraceError("semantic text is empty, unsafe, or unbounded", "not_suitable")
    return text


def _origin_text(value: JsonValue | None) -> str:
    text = _text(value)
    if not text or len(text) > 255 or any(ord(char) < 32 for char in text):
        raise BrowserTraceError("origin is empty or unbounded", "not_suitable")
    return text


def _strict_mapping(value: JsonValue | Mapping[str, JsonValue] | None, label: str) -> JsonObject:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BrowserTraceError(f"{label} must be an object")
    return dict(value)


def _objects(value: JsonValue | None) -> list[JsonObject]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bounded(value: object, label: str) -> None:
    def check(item: object, depth: int) -> None:
        if depth > MAX_DEPTH: raise BrowserTraceError(f"{label} nesting exceeds bound", "not_suitable")
        if isinstance(item, float) and not math.isfinite(item): raise BrowserTraceError(f"{label} contains non-finite JSON", "not_suitable")
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item): raise BrowserTraceError(f"{label} has non-string key", "not_suitable")
            for child in item.values(): check(child, depth + 1)
        elif isinstance(item, list):
            for child in item: check(child, depth + 1)
        elif not isinstance(item, (str, int, float, bool, type(None))): raise BrowserTraceError(f"{label} contains non-JSON data", "not_suitable")
    check(value, 0)
    if len(_json(value)) > MAX_TRACE_BYTES: raise BrowserTraceError(f"{label} exceeds 256 KiB", "not_suitable")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _result(status: str, reason: str, fixture: JsonObject | None = None) -> JsonObject:
    result: JsonObject = {"status": status, "reason": reason, "executed": False, "simulated": True}
    if fixture is not None: result["fixture_digest"] = fixture["digest"]
    return result

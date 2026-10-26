"""Pure, bounded fusion of caller-supplied search candidate assertions.

No retrieval, reranker, persistence, current-clock or authentication happens here.
The host supplies the current capability snapshot and must recheck access before
using/disclosing results. Read/acquisition labels never prove truth or a capture
clock. This adapter retains source-finder vocabulary without replacing its
source_candidate contract; temporal receipts belong to coverage assessment.
"""
from copy import deepcopy
from datetime import datetime, time, timezone
import math


def _text(maximum=2048, empty=False):
    return {"type": "string", "minLength": 0 if empty else 1, "maxLength": maximum}


def _array(items, maximum, minimum=0, unique=False):
    return dict(type="array", items=items, minItems=minimum, maxItems=maximum, uniqueItems=unique)


def _object(**properties):
    return dict(type="object", properties=properties)


def _enum(*values):
    return {"enum": values}


def _nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


def _ref(name):
    return {"$ref": name}


# Merge-only frozen vocabulary; objects are closed, all properties required.
# Discovery files and other producer modules are never runtime dependencies.
_SCHEMA = {
    "assertion": _object(kind=_enum("caller_assertion", "host_assertion"), reference=_text()),
    "binding_kind": _enum("web", "local_file", "connected_app", "context"),
    "stamp": _text(64),
    "location": _object(kind=_enum("uri", "path", "app", "context"), locator=_text()),
    "read": _object(state=_enum("unread", "snippet", "full_text", "failed"),
                    observation_ref=_nullable(_text()), observed_at=_nullable(_ref("stamp")),
                    locators=_array(_text(), 100), assertion=_nullable(_ref("assertion"))),
}
_BINDING_FIELDS = dict(source_kind=_ref("binding_kind"), provider_id=_text(),
                       principal_id=_text(), scope_id=_text())
_SCHEMA.update({
    "grant": _object(grant_id=_text(), **_BINDING_FIELDS,
                     allowed_source_ids=_nullable(_array(_text(), 100, 1, True)),
                     excluded_source_ids=_array(_text(), 100, unique=True)),
    "capability": _object(capability_id=_text(), grant_id=_text(), **_BINDING_FIELDS,
                          tool_name=_text(), operations=_array(_enum("search", "read"), 2, 1, True),
                          availability=_enum("available", "unavailable", "denied", "unknown"),
                          observed_at=_ref("stamp"), assertion=_ref("assertion")),
    "scope": _object(schema_version={"const": "search_scope/v1"},
                     grants=_array(_ref("grant"), 32, 1)),
    "snapshot": _object(schema_version={"const": "search_capabilities/v1"},
                        snapshot_id=_text(), checked_at=_ref("stamp"),
                        capabilities=_array(_ref("capability"), 64)),
    "candidate": _object(candidate_id=_text(), **_BINDING_FIELDS, source_id=_text(),
                         version=_nullable(_text()), location=_ref("location"),
                         title=_text(4096, True), snippet=_text(8192, True),
                         published_at=_nullable(_ref("stamp")),
                         rank=dict(type="integer", minimum=1, maximum=64),
                         score=_nullable({"type": "number"}),
                         acquisition_state=_enum("candidate_prepared", "link_observed",
                             "download_link_prepared", "download_observed", "file_hash_recorded",
                             "text_extraction_observed", "license_checked", "verification_observed",
                             "downstream_selected"),
                         observation_provenance=_enum("user", "wrapper", "hermes_tool", "local_file",
                                                      "runtime_observation", "unknown"),
                         observation_ref=_nullable(_text()), read=_ref("read")),
    "candidate_set": _object(schema_version={"const": "search_candidate_set/v1"},
                             set_id=_text(), query_id=_text(), capability_id=_text(),
                             candidates=_array(_ref("candidate"), 64)),
    "merge_context": _object(schema_version={"const": "search_merge_context/v1"},
                             scope=_ref("scope"), capabilities=_ref("snapshot")),
})


def _shape(value, schema, path, errors):
    """Validate only the closed frozen schema subset, never arbitrary schemas."""
    if "$ref" in schema:
        return _shape(value, _SCHEMA[schema["$ref"]], path, errors)
    if "anyOf" in schema:
        if value is not None:
            _shape(value, schema["anyOf"][0], path, errors)
        return
    if "const" in schema or "enum" in schema:
        values = [schema["const"]] if "const" in schema else schema["enum"]
        if type(value) is not str:
            errors.append("invalid_type:" + path)
        elif value not in values:
            errors.append("unsupported_value:" + path)
        return
    kind = schema["type"]
    valid = {"object": type(value) is dict, "array": type(value) is list,
             "string": type(value) is str, "integer": type(value) is int,
             "number": type(value) in (int, float), "null": value is None}[kind]
    if kind == "number" and type(value) is float:
        valid = math.isfinite(value)
    if not valid:
        errors.append("invalid_type:" + path)
        return
    if kind == "object":
        properties = schema["properties"]
        if any(type(key) is not str for key in value):
            errors.append("invalid_type:" + path)
        if any(key not in properties for key in value):
            # Never echo an unknown key: it may itself contain a secret.
            errors.append("unknown_field:" + path)
        for key, child in properties.items():
            if key not in value:
                errors.append("missing_field:" + path + "/" + key)
            else:
                _shape(value[key], child, path + "/" + key, errors)
    elif kind == "array":
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            errors.append("bounds:" + path)
            return
        for index, child in enumerate(value):
            _shape(child, schema["items"], path + "/" + str(index), errors)
        if schema["uniqueItems"] and any(v in value[:i] for i, v in enumerate(value)):
            errors.append("duplicate_id:" + path)
    elif kind == "string":
        if not schema["minLength"] <= len(value) <= schema["maxLength"] or (
            schema["minLength"] and not value.strip()
        ):
            errors.append("bounds:" + path)
    elif kind in ("integer", "number"):
        if ("minimum" in schema and value < schema["minimum"]) or (
            "maximum" in schema and value > schema["maximum"]
        ):
            errors.append("bounds:" + path)


def _raise_errors(errors):
    if errors:
        raise SearchEvidenceError(min(errors))


class SearchEvidenceError(ValueError):
    """Malformed search evidence; messages contain only code and JSON pointer."""


_BINDING = ("source_kind", "provider_id", "principal_id", "scope_id")


def _binding(value):
    return tuple(value[key] for key in _BINDING)


def _exclusion(item, capability, grants):
    grant = grants.get(_binding(item))
    if grant is None:
        return "scope_excluded"
    if capability is None:
        return "capability_missing"
    if capability["availability"] != "available":
        return "capability_" + capability["availability"]
    if "search" not in capability["operations"]:
        return "capability_denied"
    if item["read"]["state"] == "full_text" and "read" not in capability["operations"]:
        return "capability_denied"
    if item["source_id"] in grant["excluded_source_ids"] or (
        grant["allowed_source_ids"] is not None
        and item["source_id"] not in grant["allowed_source_ids"]
    ):
        return "source_excluded"
    return None


def _identity(item):
    version = item["version"]
    return (*_binding(item), item["source_id"], version is not None, version or "",
            item["location"]["kind"], item["location"]["locator"],
            item["candidate_id"] if version is None else "")


def _unique(value, seen, path, errors):
    if value in seen:
        errors.append("duplicate_id:" + path)
    seen.add(value)


def strict_fromisoformat(text):
    # This module loads by file path with only the stdlib, so it keeps its own
    # copy of omh.plugin_bundle.omh.iso_datetime.strict_fromisoformat: Python
    # 3.14 reads T24:00:00 as the next midnight and 3.11-3.13 refuse it. Only
    # that reading moves the date past the one the text starts with.
    parsed = datetime.fromisoformat(text)
    if parsed.time() == time(0) and not text.startswith(_date_spellings(parsed.date())):
        raise ValueError("hour must be in 0..23")
    return parsed


def _date_spellings(day):
    week_year, week, weekday = day.isocalendar()
    spellings = (f"{day.year:04d}-{day.month:02d}-{day.day:02d}", f"{day.year:04d}{day.month:02d}{day.day:02d}",
                 f"{week_year:04d}-W{week:02d}-{weekday}", f"{week_year:04d}W{week:02d}{weekday}")
    if weekday == 1:
        spellings += (f"{week_year:04d}-W{week:02d}", f"{week_year:04d}W{week:02d}")
    return spellings


def _timestamp(value, path, errors):
    # Compatible comparison semantics with temporal_source_receipts._parse_stamp;
    # retain the original token in every returned occurrence.
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = strict_fromisoformat(text)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        errors.append("invalid_time:" + path)
        return None


def _provenance(item, path, errors):
    prepared = item["acquisition_state"] in ("candidate_prepared", "download_link_prepared")
    observed = item["observation_ref"] is not None and item["observation_provenance"] != "unknown"
    if not prepared and not observed:
        errors.append("invalid_provenance:" + path)
    read = item["read"]
    if read["state"] == "unread":
        if any(read[k] is not None for k in ("observation_ref", "observed_at", "assertion")) or read["locators"]:
            errors.append("invalid_provenance:" + path + "/read")
    else:
        if any(read[k] is None for k in ("observation_ref", "observed_at", "assertion")):
            errors.append("invalid_provenance:" + path + "/read")
        if read["observed_at"] is not None:
            _timestamp(read["observed_at"], path + "/read/observed_at", errors)
        if read["state"] == "failed" and read["locators"]:
            errors.append("invalid_provenance:" + path + "/read/locators")
        if read["state"] == "full_text" and (not read["locators"] or not observed):
            errors.append("invalid_provenance:" + path + "/read")
    if item["published_at"] is not None:
        _timestamp(item["published_at"], path + "/published_at", errors)


def _semantics(candidate_sets, scope, errors):
    snapshot = scope["capabilities"]
    checked = _timestamp(snapshot["checked_at"], "/scope/capabilities/checked_at", errors)
    grants, bindings = {}, set()
    for index, grant in enumerate(scope["scope"]["grants"]):
        path = "/scope/scope/grants/" + str(index)
        if grant["grant_id"] in grants:
            errors.append("duplicate_id:" + path + "/grant_id")
        _unique(_binding(grant), bindings, path, errors)
        grants[grant["grant_id"]] = grant
    capabilities = {}
    for index, cap in enumerate(scope["capabilities"]["capabilities"]):
        path = "/scope/capabilities/capabilities/" + str(index)
        observed = _timestamp(cap["observed_at"], path + "/observed_at", errors)
        if observed is not None and checked is not None and observed > checked:
            errors.append("invalid_time:" + path + "/observed_at")
        if cap["capability_id"] in capabilities:
            errors.append("duplicate_id:" + path + "/capability_id")
        capabilities[cap["capability_id"]] = cap
        grant = grants.get(cap["grant_id"])
        if grant is None:
            errors.append("dangling_reference:" + path + "/grant_id")
        elif _binding(cap) != _binding(grant):
            errors.append("scope_mismatch:" + path)
    sets, domains, ids = set(), set(), {}
    for index, group in enumerate(candidate_sets):
        path = "/candidate_sets/" + str(index)
        _unique(group["set_id"], sets, path + "/set_id", errors)
        _unique((group["query_id"], group["capability_id"]), domains, path, errors)
        cap = capabilities.get(group["capability_id"])
        ranks, local_ids = set(), set()
        for number, item in enumerate(group["candidates"]):
            child = path + "/candidates/" + str(number)
            _provenance(item, child, errors)
            _unique(item["candidate_id"], local_ids, child + "/candidate_id", errors)
            _unique(item["rank"], ranks, child + "/rank", errors)
            identity = _identity(item)
            if item["candidate_id"] in ids and ids[item["candidate_id"]] != identity:
                errors.append("invalid_state:" + child + "/candidate_id")
            ids[item["candidate_id"]] = identity
            if _binding(item) in bindings and cap is not None and _binding(item) != _binding(cap):
                errors.append("scope_mismatch:" + child)
            expected = {"web": "uri", "local_file": "path", "connected_app": "app", "context": "context"}
            if item["location"]["kind"] != expected[item["source_kind"]]:
                errors.append("unsupported_value:" + child + "/location/kind")


def merge_search_candidates(candidate_sets, scope):
    """Return isolated candidate occurrences without performing I/O."""
    errors = []
    _shape(candidate_sets, _array(_ref("candidate_set"), 5), "/candidate_sets", errors)
    _shape(scope, _ref("merge_context"), "/scope", errors)
    _raise_errors(errors)
    _semantics(candidate_sets, scope, errors)
    _raise_errors(errors)
    grants = {_binding(g): g for g in scope["scope"]["grants"]}
    capabilities = {c["capability_id"]: c for c in scope["capabilities"]["capabilities"]}
    fused, excluded = {}, []
    for group in candidate_sets:
        for item in group["candidates"]:
            reason = _exclusion(item, capabilities.get(group["capability_id"]), grants)
            if reason:
                excluded.append(dict(candidate_id=item["candidate_id"], reason=reason))
                continue
            identity = _identity(item)
            row = fused.setdefault(identity, dict(candidate_id=item["candidate_id"],
                                                  rrf_score=0, occurrences=[]))
            row["candidate_id"] = min(row["candidate_id"], item["candidate_id"])
            row["occurrences"].append(dict(set_id=group["set_id"], query_id=group["query_id"],
                                           capability_id=group["capability_id"], candidate=deepcopy(item)))
    if len(fused) > 256:
        errors.append("bounds:/candidate_sets")
    for row in fused.values():
        if len(row["occurrences"]) > 5:
            errors.append("bounds:/candidate_sets")
        # One best-position vote per query/capability domain; retain every
        # occurrence, including conflicting snippets, in input encounter order.
        ranks = {}
        for occurrence in row["occurrences"]:
            domain = (occurrence["query_id"], occurrence["capability_id"])
            rank = occurrence["candidate"]["rank"]
            ranks[domain] = min(ranks.get(domain, rank), rank)
        row["rrf_score"] = sum(1 / (60 + rank) for rank in sorted(ranks.values()))
    _raise_errors(errors)
    rows = [row for identity, row in sorted(fused.items(),
            key=lambda pair: (-pair[1]["rrf_score"], pair[0], pair[1]["candidate_id"]))]
    return dict(schema_version="merged_search_candidates/v1", scope=deepcopy(scope["scope"]),
                snapshot_id=scope["capabilities"]["snapshot_id"], candidates=rows,
                candidate_count=len(rows), excluded=excluded,
                claim_boundary="rank_and_read_assertions_are_not_authenticated_truth")

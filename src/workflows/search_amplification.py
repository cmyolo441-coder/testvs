"""Pure bounded search planning and attributed coverage, never retrieval.

Snapshots and evidence are caller assertions, not authentication or truth.
The host must anchor the latest cumulative assessment state per request and
recheck authorization before every call and disclosure. Nothing persists here.
"""
from copy import deepcopy
from datetime import timezone
import hashlib
import json
import math
import re

from .temporal_source_receipts import (
    as_of_errors,
    receipt_supports_as_of_claim,
    validate_temporal_source_receipt,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


class SearchPlanError(ValueError):
    """Invalid search plan input, with a non-sensitive code and pointer."""


class SearchCoverageError(ValueError):
    """Invalid search observation input, with a code and pointer."""


_BUDGETS = {"direct": (2, 4), "general": (3, 10), "comparison": (5, 18)}
_INTENTS = {"lookup": "direct", "explore": "general", "compare": "comparison"}
_BINDING = ("source_kind", "provider_id", "principal_id", "scope_id")
# Closed descriptors transcribed from the frozen search_amplification_contract/v1.
# object properties are all required; no additional properties are accepted.
_SHAPES = {'source_kind': ('enum', ('web', 'local_file', 'connected_app', 'context')),
 'stamp': ('string', 1, 64, None),
 'as_of': ('union',
           (('object', {'kind': ('const', 'date'), 'date': ('ref', 'stamp')}),
            ('object', {'kind': ('const', 'interval'), 'start': ('ref', 'stamp'), 'end': ('ref', 'stamp')}))),
 'freshness': ('object',
               {'kind': ('enum', ('any', 'current', 'as_of')),
                'as_of': ('union', (('ref', 'as_of'), ('null',))),
                'max_age_seconds': ('union', (('integer', 0, 31536000), ('null',)))}),
 'assertion': ('object',
               {'kind': ('enum', ('caller_assertion', 'host_assertion')),
                'reference': ('string', 1, 2048, None)}),
 'grant': ('object',
           {'grant_id': ('string', 1, 2048, None),
            'source_kind': ('ref', 'source_kind'),
            'provider_id': ('string', 1, 2048, None),
            'principal_id': ('string', 1, 2048, None),
            'scope_id': ('string', 1, 2048, None),
            'allowed_source_ids': ('union', (('array', ('string', 1, 2048, None), 1, 100, True), ('null',))),
            'excluded_source_ids': ('array', ('string', 1, 2048, None), 0, 100, True)}),
 'scope': ('object',
           {'schema_version': ('const', 'search_scope/v1'),
            'grants': ('array', ('ref', 'grant'), 1, 32, False)}),
 'capability': ('object',
                {'capability_id': ('string', 1, 2048, None),
                 'grant_id': ('string', 1, 2048, None),
                 'source_kind': ('ref', 'source_kind'),
                 'provider_id': ('string', 1, 2048, None),
                 'principal_id': ('string', 1, 2048, None),
                 'scope_id': ('string', 1, 2048, None),
                 'tool_name': ('string', 1, 2048, None),
                 'operations': ('array', ('enum', ('search', 'read')), 1, 2, True),
                 'availability': ('enum', ('available', 'unavailable', 'denied', 'unknown')),
                 'observed_at': ('ref', 'stamp'),
                 'assertion': ('ref', 'assertion')}),
 'snapshot': ('object',
              {'schema_version': ('const', 'search_capabilities/v1'),
               'snapshot_id': ('string', 1, 2048, None),
               'checked_at': ('ref', 'stamp'),
               'capabilities': ('array', ('ref', 'capability'), 0, 64, False)}),
 'requirement': ('object',
                 {'requirement_id': ('string', 1, 2048, None),
                  'text': ('string', 1, 4096, None),
                  'required': ('boolean',),
                  'freshness': ('ref', 'freshness')}),
 'query': ('object',
           {'query_id': ('string', 1, 2048, None),
            'text': ('string', 1, 2048, None),
            'purpose': ('string', 1, 1024, None),
            'round': ('integer', 0, 1),
            'requirement_ids': ('array', ('string', 1, 2048, None), 1, 100, True),
            'capability_ids': ('array', ('string', 1, 2048, None), 1, 64, True)}),
 'spec': ('object',
          {'schema_version': ('const', 'search_spec/v1'),
           'request_id': ('string', 1, 2048, None),
           'query': ('string', 1, 8192, None),
           'scope': ('ref', 'scope'),
           'intent': ('enum', ('lookup', 'explore', 'compare')),
           'mode': ('enum', ('direct', 'general', 'comparison')),
           'requirements': ('array', ('ref', 'requirement'), 1, 100, False),
           'queries': ('array', ('ref', 'query'), 1, 5, False)}),
 'budget': ('object',
            {'max_queries': ('integer', 2, 5),
             'max_external_calls': ('integer', 4, 18),
             'max_follow_up_rounds': ('const', 1)}),
 'plan': ('object',
          {'schema_version': ('const', 'search_plan/v1'),
           'plan_id': ('string', 1, 2048, None),
           'spec': ('ref', 'spec'),
           'capabilities': ('ref', 'snapshot'),
           'selected_capability_ids': ('array', ('string', 1, 2048, None), 0, 64, True),
           'unavailable_grant_ids': ('array', ('string', 1, 2048, None), 0, 32, True),
           'queries': ('array', ('ref', 'query'), 0, 5, False),
           'budget': ('ref', 'budget'),
           'status': ('const', 'prepared_not_observed')}),
 'location': ('object',
              {'kind': ('enum', ('uri', 'path', 'app', 'context')), 'locator': ('string', 1, 2048, None)}),
 'read': ('object',
          {'state': ('enum', ('unread', 'snippet', 'full_text', 'failed')),
           'observation_ref': ('union', (('string', 1, 2048, None), ('null',))),
           'observed_at': ('union', (('ref', 'stamp'), ('null',))),
           'locators': ('array', ('string', 1, 2048, None), 0, 100, False),
           'assertion': ('union', (('ref', 'assertion'), ('null',)))}),
 'candidate': ('object',
               {'candidate_id': ('string', 1, 2048, None),
                'source_kind': ('ref', 'source_kind'),
                'provider_id': ('string', 1, 2048, None),
                'principal_id': ('string', 1, 2048, None),
                'scope_id': ('string', 1, 2048, None),
                'source_id': ('string', 1, 2048, None),
                'version': ('union', (('string', 1, 2048, None), ('null',))),
                'location': ('ref', 'location'),
                'title': ('string', 0, 4096, None),
                'snippet': ('string', 0, 8192, None),
                'published_at': ('union', (('ref', 'stamp'), ('null',))),
                'rank': ('integer', 1, 64),
                'score': ('union', (('number', None, None), ('null',))),
                'acquisition_state': ('enum',
                                      ('candidate_prepared',
                                       'link_observed',
                                       'download_link_prepared',
                                       'download_observed',
                                       'file_hash_recorded',
                                       'text_extraction_observed',
                                       'license_checked',
                                       'verification_observed',
                                       'downstream_selected')),
                'observation_provenance': ('enum',
                                           ('user',
                                            'wrapper',
                                            'hermes_tool',
                                            'local_file',
                                            'runtime_observation',
                                            'unknown')),
                'observation_ref': ('union', (('string', 1, 2048, None), ('null',))),
                'read': ('ref', 'read')}),
 'temporal_receipt': ('object',
                      {'schema_version': ('const', 'temporal_source_receipt/v1'),
                       'receipt_id': ('string', 1, 2048, None),
                       'as_of': ('ref', 'as_of'),
                       'source_url': ('string', 1, 2048, None),
                       'source_class': ('enum', ('upstream_official', 'practitioner', 'unattributed')),
                       'evidence_kind': ('enum', ('historical_capture', 'live_page')),
                       'retrieved_at': ('ref', 'stamp'),
                       'capture_provider': ('string', 0, 2048, None),
                       'capture_provider_class': ('enum',
                                                  ('archive_service',
                                                   'native_tool',
                                                   'external_connector',
                                                   'none')),
                       'captured_at': ('string', 0, 64, None),
                       'captured_at_attribution': ('enum', ('provider_reported', 'unknown')),
                       'capture_ref': ('string', 0, 2048, None),
                       'content_digest': ('string', 0, 64, None),
                       'cutoff_relation': ('enum', ('at_or_before', 'after', 'unknown')),
                       'availability': ('enum',
                                        ('available',
                                         'unavailable',
                                         'access_denied',
                                         'provider_authority_exhausted',
                                         'not_attempted')),
                       'failure_reason': ('string', 0, 200, None),
                       'claim_boundary': ('const',
                                          'A temporal source receipt binds one cited source to one requested '
                                          'as-of cutoff. It attributes a provider-reported capture time; it '
                                          'does not authenticate that time, prove the page was true, or '
                                          'prove that any retrieval was executed.')}),
 'evidence': ('object',
              {'requirement_id': ('string', 1, 2048, None),
               'candidate': ('ref', 'candidate'),
               'assessment': ('enum', ('supported', 'conflicting', 'missing')),
               'assessed_by': ('enum', ('caller', 'hermes')),
               'assessment_ref': ('string', 1, 2048, None),
               'temporal_receipt': ('union', (('ref', 'temporal_receipt'), ('null',)))}),
 'observation': ('object',
                 {'schema_version': ('const', 'search_observation/v1'),
                  'observation_id': ('string', 1, 2048, None),
                  'call_id': ('string', 1, 2048, None),
                  'query_id': ('string', 1, 2048, None),
                  'capability_id': ('string', 1, 2048, None),
                  'requirement_ids': ('array', ('string', 1, 2048, None), 1, 100, True),
                  'operation': ('enum', ('search', 'read')),
                  'status': ('enum',
                             ('completed',
                              'failed',
                              'denied',
                              'unavailable',
                              'cancelled',
                              'budget_exhausted')),
                  'failure_class': ('enum',
                                    ('none',
                                     'access_denied',
                                     'rate_limited',
                                     'timeout',
                                     'provider_error',
                                     'unavailable',
                                     'cancelled',
                                     'budget_exhausted')),
                  'attempted': ('boolean',),
                  'observed_at': ('ref', 'stamp'),
                  'assertion': ('ref', 'assertion'),
                  'evidence': ('array', ('ref', 'evidence'), 0, 64, False)}),
 'state': ('object',
           {'schema_version': ('const', 'search_assessment_state/v1'),
            'plan_id': ('string', 1, 2048, None),
            'revision': ('integer', 1, 100),
            'ledger_digest': ('string', 0, 4096, '^[0-9a-f]{64}$'),
            'used_external_calls': ('integer', 0, 18),
            'used_follow_up_rounds': ('integer', 0, 1),
            'observation_ids': ('array', ('string', 1, 2048, None), 0, 100, True),
            'issued_follow_up_query_ids': ('array', ('string', 1, 2048, None), 0, 5, True)}),
 'assessment_input': ('object',
                      {'schema_version': ('const', 'search_observations/v1'),
                       'capabilities': ('ref', 'snapshot'),
                       'assessed_at': ('ref', 'stamp'),
                       'previous_state': ('union', (('ref', 'state'), ('null',))),
                       'observations': ('array', ('ref', 'observation'), 0, 100, False)}),
 'coverage_row': ('object',
                  {'requirement_id': ('string', 1, 2048, None),
                   'required': ('boolean',),
                   'status': ('enum',
                              ('supported',
                               'conflicting',
                               'missing',
                               'inaccessible',
                               'unsearched',
                               'unavailable',
                               'failed',
                               'budget_exhausted',
                               'cancelled',
                               'stale',
                               'temporal_gap')),
                   'observation_ids': ('array', ('string', 1, 2048, None), 0, 100, True),
                   'candidate_ids': ('array', ('string', 1, 2048, None), 0, 100, True),
                   'gap_reasons': ('array',
                                   ('enum',
                                    ('conflicting',
                                     'missing',
                                     'inaccessible',
                                     'unsearched',
                                     'unavailable',
                                     'failed',
                                     'budget_exhausted',
                                     'cancelled',
                                     'stale',
                                     'temporal_gap')),
                                   0,
                                   11,
                                   True)}),
 'coverage': ('object',
              {'schema_version': ('const', 'search_coverage/v1'),
               'plan_id': ('string', 1, 2048, None),
               'snapshot_id': ('string', 1, 2048, None),
               'requirements': ('array', ('ref', 'coverage_row'), 1, 100, False),
               'candidate_ids': ('array', ('string', 1, 2048, None), 0, 6400, True),
               'follow_up': ('enum', ('none', 'search_once', 'partial_answer')),
               'follow_up_query_ids': ('array', ('string', 1, 2048, None), 0, 5, True),
               'remaining_budget': ('object',
                                    {'queries': ('integer', 0, 5),
                                     'external_calls': ('integer', 0, 18),
                                     'follow_up_rounds': ('integer', 0, 1)}),
               'state': ('ref', 'state'),
               'claim_boundary': ('const',
                                  'attributed_assessments_do_not_authenticate_access_execution_or_truth')})}


def _error(errors, code, path):
    errors.append(code + ":" + (path or "/"))


def _shape(value, descriptor, path=""):
    """Validate only bounded JSON types; never interpolate untrusted values."""
    kind, *args = descriptor
    errors = []
    if kind == "ref":
        return _shape(value, _SHAPES[args[0]], path)
    if kind == "union":
        branches = [_shape(value, branch, path) for branch in args[0]]
        if any(not branch for branch in branches):
            return []
        # Prefer the closest structural branch, keeping useful nested pointers.
        return min(branches, key=lambda branch: (len(branch), sorted(branch)))
    if kind == "const":
        if type(value) is not type(args[0]) or value != args[0]:
            _error(errors, "unsupported_value", path)
        return errors
    if kind == "enum":
        if type(value) is not str:
            _error(errors, "invalid_type", path)
        elif value not in args[0]:
            _error(errors, "unsupported_value", path)
        return errors
    expected = {"object": dict, "array": list, "string": str,
                "integer": int, "boolean": bool, "null": type(None)}
    if kind == "number":
        valid = type(value) in (float, int) and (type(value) is int or math.isfinite(value))
    else:
        valid = type(value) is expected[kind]
    if not valid:
        _error(errors, "invalid_type", path)
        return errors
    if kind == "object":
        fields = args[0]
        if any(type(k) is not str for k in value):
            _error(errors, "invalid_type", path)
        if any(k not in fields for k in value):
            _error(errors, "unknown_field", path)
        for key, child in fields.items():
            pointer = path + "/" + key
            if key not in value:
                _error(errors, "missing_field", pointer)
            else:
                errors.extend(_shape(value[key], child, pointer))
    elif kind == "array":
        child, low, high, unique = args
        if not low <= len(value) <= high:
            _error(errors, "bounds", path)
            return errors
        for i, item in enumerate(value):
            child_errors = _shape(item, child, path + "/" + str(i))
            if child_errors:
                # One deterministic structural diagnostic per invalid element
                # bounds error reporting, never truncating or repairing inputs.
                errors.append(min(child_errors))
        if not errors and unique and any(item in value[:i] for i, item in enumerate(value)):
            _error(errors, "duplicate_id", path)
    elif kind == "string":
        low, high, pattern = args
        if not low <= len(value) <= high or (low and not value.strip()):
            _error(errors, "bounds", path)
        if pattern and re.fullmatch(pattern, value) is None:
            _error(errors, "unsupported_value", path)
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            _error(errors, "invalid_type", path)
    elif kind in ("number", "integer"):
        low, high = args
        if (low is not None and value < low) or (high is not None and value > high):
            _error(errors, "bounds", path)
    return errors


def _digest(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stamp(raw, path, errors):
    try:
        parsed = strict_fromisoformat(raw.strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        _error(errors, "invalid_time", path)
        return None


def _unique(rows, key, path, errors):
    seen = set()
    for i, row in enumerate(rows):
        token = row[key]
        if token in seen:
            _error(errors, "duplicate_id", path + "/" + str(i) + "/" + key)
        seen.add(token)


def _binding(row):
    return tuple(row[key] for key in _BINDING)


def _snapshot_errors(snapshot, scope, path):
    errors = []
    grants = scope["grants"]
    _unique(grants, "grant_id", "/spec/scope/grants", errors)
    bindings = [_binding(g) for g in grants]
    if len(set(bindings)) != len(bindings):
        _error(errors, "duplicate_id", "/spec/scope/grants")
    grant_map = {g["grant_id"]: g for g in grants}
    _unique(snapshot["capabilities"], "capability_id", path + "/capabilities", errors)
    checked = _stamp(snapshot["checked_at"], path + "/checked_at", errors)
    for i, cap in enumerate(snapshot["capabilities"]):
        p = path + "/capabilities/" + str(i)
        grant = grant_map.get(cap["grant_id"])
        if grant is None or _binding(cap) != _binding(grant):
            _error(errors, "scope_mismatch", p)
        observed = _stamp(cap["observed_at"], p + "/observed_at", errors)
        if observed is not None and checked is not None and observed > checked:
            _error(errors, "invalid_time", p + "/observed_at")
    return errors


def _temporal_invalid(validator, value):
    """Keep the existing temporal contract, containing its datetime overflow."""
    try:
        return bool(validator(deepcopy(value)))
    except (ValueError, OverflowError):
        return True


def _build_errors(spec, capabilities):
    errors = _shape(spec, ("ref", "spec"), "/spec")
    errors.extend(_shape(capabilities, ("ref", "snapshot"), "/capabilities"))
    if errors:
        return sorted(set(errors))
    errors.extend(_snapshot_errors(capabilities, spec["scope"], "/capabilities"))
    if _INTENTS[spec["intent"]] != spec["mode"]:
        _error(errors, "unsupported_value", "/spec/mode")
    if len(spec["queries"]) > _BUDGETS[spec["mode"]][0]:
        _error(errors, "invalid_budget", "/spec/queries")
    _unique(spec["requirements"], "requirement_id", "/spec/requirements", errors)
    _unique(spec["queries"], "query_id", "/spec/queries", errors)
    caps = {c["capability_id"]: c for c in capabilities["capabilities"]}
    requirements = {r["requirement_id"] for r in spec["requirements"]}
    referenced = set()
    if not any(q["round"] == 0 for q in spec["queries"]):
        _error(errors, "invalid_state", "/spec/queries")
    for i, query in enumerate(spec["queries"]):
        p = "/spec/queries/" + str(i)
        referenced.update(query["requirement_ids"])
        if not set(query["requirement_ids"]) <= requirements:
            _error(errors, "dangling_reference", p + "/requirement_ids")
        for cid in query["capability_ids"]:
            if cid not in caps:
                _error(errors, "dangling_reference", p + "/capability_ids")
            elif "search" not in caps[cid]["operations"]:
                _error(errors, "scope_mismatch", p + "/capability_ids")
    if not requirements <= referenced:
        _error(errors, "dangling_reference", "/spec/requirements")
    for i, requirement in enumerate(spec["requirements"]):
        fresh = requirement["freshness"]
        p = "/spec/requirements/" + str(i) + "/freshness"
        kind = fresh["kind"]
        if ((kind == "any" and (fresh["as_of"] is not None or fresh["max_age_seconds"] is not None))
                or (kind == "current" and (fresh["as_of"] is not None or fresh["max_age_seconds"] is None))
                or (kind == "as_of" and (fresh["as_of"] is None or fresh["max_age_seconds"] is not None))):
            _error(errors, "invalid_state", p)
        if fresh["as_of"] is not None and _temporal_invalid(as_of_errors, fresh["as_of"]):
            _error(errors, "invalid_time", p + "/as_of")
    return sorted(set(errors))


def _prepared_plan(spec, capabilities):
    max_queries, calls = _BUDGETS[spec["mode"]]
    referenced = {cid for q in spec["queries"] for cid in q["capability_ids"]}
    selected = [c["capability_id"] for c in capabilities["capabilities"]
                if c["capability_id"] in referenced and c["availability"] == "available"
                and "search" in c["operations"]]
    selected_grants = {c["grant_id"] for c in capabilities["capabilities"] if c["capability_id"] in selected}
    queries = []
    for query in spec["queries"]:
        filtered = [cid for cid in query["capability_ids"] if cid in selected]
        if filtered:
            queries.append(dict(query, capability_ids=filtered))
    return deepcopy(dict(schema_version="search_plan/v1",
                         plan_id="search-plan:" + _digest(dict(spec=spec, capabilities=capabilities)),
                         spec=spec, capabilities=capabilities,
                         selected_capability_ids=selected,
                         unavailable_grant_ids=[g["grant_id"] for g in spec["scope"]["grants"]
                                                if g["grant_id"] not in selected_grants],
                         queries=queries,
                         budget=dict(max_queries=max_queries, max_external_calls=calls, max_follow_up_rounds=1),
                         status="prepared_not_observed"))


def build_search_plan(spec, capabilities):
    """Validate exact caller decomposition and available scoped capabilities."""
    errors = _build_errors(spec, capabilities)
    if errors:
        raise SearchPlanError(errors[0])
    return _prepared_plan(spec, capabilities)


def validate_search_plan(plan):
    """Sorted unique code:/pointer errors for any malformed plan, else []."""
    errors = _shape(plan, ("ref", "plan"))
    if errors:
        return sorted(set(errors))
    errors = _build_errors(plan["spec"], plan["capabilities"])
    if not errors:
        expected = _prepared_plan(plan["spec"], plan["capabilities"])
        for key in expected:
            if plan[key] != expected[key]:
                _error(errors, "invalid_budget" if key == "budget" else "invalid_state", "/" + key)
    return sorted(set(errors))



_GAP_ORDER = ("conflicting", "inaccessible", "unavailable", "failed", "cancelled",
              "budget_exhausted", "temporal_gap", "stale", "missing", "unsearched")


def _append_unique(target, token):
    if token not in target:
        target.append(token)


def _cap_gap(cid, planned, current, operation="search"):
    old, new = planned.get(cid), current.get(cid)
    if old is None:
        return "unavailable"
    if new is None:
        return "inaccessible"
    if old["availability"] == "denied" or new["availability"] == "denied":
        return "inaccessible"
    if old["availability"] != "available" or new["availability"] != "available":
        return "unavailable"
    if operation not in old["operations"] or operation not in new["operations"]:
        return "inaccessible"
    return None


def _source_allowed(candidate, capability, grants):
    grant = grants[capability["grant_id"]]
    return (candidate["source_id"] not in grant["excluded_source_ids"]
            and (grant["allowed_source_ids"] is None
                 or candidate["source_id"] in grant["allowed_source_ids"]))


def _evidence_gap(item, freshness, assessed):
    candidate = item["candidate"]
    read = candidate["read"]
    if (read["state"] != "full_text" or not read["locators"]
            or candidate["observation_ref"] is None
            or candidate["observation_provenance"] == "unknown"):
        return "missing"
    if item["assessment"] == "missing":
        return "missing"
    if freshness["kind"] == "as_of":
        receipt = item["temporal_receipt"]
        if (candidate["source_kind"] != "web" or receipt is None
                or not receipt_supports_as_of_claim(deepcopy(receipt))):
            return "temporal_gap"
    if freshness["kind"] == "current":
        read_time = _stamp(read["observed_at"], "", [])
        age = (assessed - read_time).total_seconds()
        if not 0 <= age <= freshness["max_age_seconds"]:
            return "stale"
    return None


def _coverage_row(requirement, plan, data, planned, current):
    rid = requirement["requirement_id"]
    ledger = data["observations"]
    grants = {g["grant_id"]: g for g in plan["spec"]["scope"]["grants"]}
    cids = list(dict.fromkeys(cid for q in plan["spec"]["queries"] if rid in q["requirement_ids"]
                             for cid in q["capability_ids"]))
    gaps, candidates, observation_ids = set(), [], []
    supported = conflicting = False
    assessed = _stamp(data["assessed_at"], "", [])
    relevant = [o for o in ledger if rid in o["requirement_ids"]]
    for cid in cids:
        gap = _cap_gap(cid, planned, current)
        if gap:
            gaps.add(gap)
        elif not any(o["capability_id"] == cid for o in relevant):
            gaps.add("unsearched")
    for obs in relevant:
        cid = obs["capability_id"]
        gap = (_cap_gap(cid, planned, current)
               or _cap_gap(cid, planned, current, obs["operation"]))
        if gap:
            gaps.add(gap)
            continue
        if obs["status"] != "completed":
            gaps.add("inaccessible" if obs["status"] == "denied" else obs["status"])
            _append_unique(observation_ids, obs["observation_id"])
            continue
        items = [item for item in obs["evidence"] if item["requirement_id"] == rid]
        if not items:
            gaps.add("missing")
            _append_unique(observation_ids, obs["observation_id"])
        for item in items:
            candidate = item["candidate"]
            read_gap = (_cap_gap(cid, planned, current, "read")
                        if candidate["read"]["state"] == "full_text" else None)
            if read_gap or not _source_allowed(candidate, planned[cid], grants):
                gaps.add(read_gap or "inaccessible")
                continue
            _append_unique(candidates, candidate["candidate_id"])
            _append_unique(observation_ids, obs["observation_id"])
            gap = _evidence_gap(item, requirement["freshness"], assessed)
            if gap:
                gaps.add(gap)
            elif item["assessment"] == "conflicting":
                conflicting = True
                gaps.add("conflicting")
            else:
                supported = True
    if conflicting:
        status = "conflicting"
    elif supported:
        status = "supported"
    else:
        if not gaps:
            gaps.add("unsearched")
        status = next(g for g in _GAP_ORDER if g in gaps)
    return dict(requirement_id=rid, required=requirement["required"], status=status,
                observation_ids=observation_ids, candidate_ids=candidates,
                gap_reasons=[g for g in _GAP_ORDER if g in gaps])



_STATUS_FAILURES = {
    "completed": ("none",), "failed": ("rate_limited", "timeout", "provider_error"),
    "denied": ("access_denied",), "unavailable": ("unavailable",),
    "cancelled": ("cancelled",), "budget_exhausted": ("budget_exhausted",),
}
_LOCATION_KINDS = {"web": "uri", "local_file": "path", "connected_app": "app", "context": "context"}


def _candidate_errors(candidate, cap, obs_time, path):
    errors = []
    if _binding(candidate) != _binding(cap):
        _error(errors, "scope_mismatch", path)
    if candidate["location"]["kind"] != _LOCATION_KINDS[candidate["source_kind"]]:
        _error(errors, "scope_mismatch", path + "/location")
    if candidate["acquisition_state"] not in ("candidate_prepared", "download_link_prepared"):
        if candidate["observation_ref"] is None or candidate["observation_provenance"] == "unknown":
            _error(errors, "invalid_provenance", path)
    if candidate["published_at"] is not None:
        _stamp(candidate["published_at"], path + "/published_at", errors)
    read = candidate["read"]
    p = path + "/read"
    if read["state"] == "unread":
        if any(read[k] is not None for k in ("observation_ref", "observed_at", "assertion")) or read["locators"]:
            _error(errors, "invalid_provenance", p)
    else:
        if any(read[k] is None for k in ("observation_ref", "observed_at", "assertion")):
            _error(errors, "invalid_provenance", p)
        if ((read["state"] == "full_text" and not read["locators"])
                or (read["state"] == "failed" and read["locators"])):
            _error(errors, "invalid_provenance", p + "/locators")
        if read["observed_at"] is not None:
            stamp = _stamp(read["observed_at"], p + "/observed_at", errors)
            if stamp is not None and obs_time is not None and stamp > obs_time:
                _error(errors, "invalid_time", p + "/observed_at")
    return errors


def _previous_errors(plan, data):
    errors = []
    previous = data["previous_state"]
    ledger = data["observations"]
    issued = previous["issued_follow_up_query_ids"] if previous else []
    query_map = {q["query_id"]: q for q in plan["queries"]}
    for i, obs in enumerate(ledger):
        query = query_map.get(obs["query_id"])
        if query is not None and query["round"] == 1 and obs["query_id"] not in issued:
            _error(errors, "invalid_state", "/observations/" + str(i) + "/query_id")
    if previous is None:
        return errors
    p = "/previous_state"
    prefix = ledger[:len(previous["observation_ids"]) ]
    if (previous["plan_id"] != plan["plan_id"]
            or previous["observation_ids"] != [o["observation_id"] for o in prefix]
            or previous["ledger_digest"] != _digest(prefix)
            or previous["used_external_calls"] != sum(o["attempted"] for o in prefix)
            or previous["used_follow_up_rounds"] != int(bool(issued))
            or any(cid not in query_map or query_map[cid]["round"] != 1 for cid in issued)
            or issued != [q["query_id"] for q in plan["queries"] if q["query_id"] in issued]
            or previous["revision"] >= 100):
        _error(errors, "invalid_state", p)
    checked = _stamp(data["capabilities"]["checked_at"], "/capabilities/checked_at", errors)
    for i, obs in enumerate(prefix):
        observed = _stamp(obs["observed_at"], "/observations/" + str(i) + "/observed_at", errors)
        if checked is not None and observed is not None and observed > checked:
            _error(errors, "invalid_time", "/capabilities/checked_at")
    return errors


def _assessment_errors(plan, data):
    errors = validate_search_plan(plan)
    errors.extend(_shape(data, ("ref", "assessment_input")))
    if errors:
        return sorted(set(errors))
    # The canonical ledger must be representable by the frozen stdlib JSON
    # encoding; refuse incompatible numeric tokens without coercion/global knobs.
    try:
        _digest(data["observations"])
    except (ValueError, OverflowError):
        return ["invalid_type:/observations"]
    snapshot = data["capabilities"]
    errors.extend(_snapshot_errors(snapshot, plan["spec"]["scope"], "/capabilities"))
    assessed = _stamp(data["assessed_at"], "/assessed_at", errors)
    checked = _stamp(snapshot["checked_at"], "/capabilities/checked_at", errors)
    build_checked = _stamp(plan["capabilities"]["checked_at"], "/plan/capabilities/checked_at", errors)
    if checked is not None and ((assessed is not None and checked > assessed)
                                or (build_checked is not None and checked < build_checked)):
        _error(errors, "invalid_time", "/capabilities/checked_at")
    planned = {c["capability_id"]: c for c in plan["capabilities"]["capabilities"]}
    for i, cap in enumerate(snapshot["capabilities"]):
        old = planned.get(cap["capability_id"])
        if old is not None and any(old[k] != cap[k] for k in (*_BINDING, "grant_id", "tool_name")):
            _error(errors, "scope_mismatch", "/capabilities/capabilities/" + str(i))
    ledger = data["observations"]
    _unique(ledger, "observation_id", "/observations", errors)
    _unique(ledger, "call_id", "/observations", errors)
    query_map = {q["query_id"]: q for q in plan["queries"]}
    candidate_bindings = {}
    requirements = {r["requirement_id"]: r for r in plan["spec"]["requirements"]}
    for i, obs in enumerate(ledger):
        p = "/observations/" + str(i)
        observed = _stamp(obs["observed_at"], p + "/observed_at", errors)
        if observed is not None and assessed is not None and observed > assessed:
            _error(errors, "invalid_time", p + "/observed_at")
        query = query_map.get(obs["query_id"])
        cap = planned.get(obs["capability_id"])
        if query is None:
            _error(errors, "dangling_reference", p + "/query_id")
        elif not set(obs["requirement_ids"]) <= set(query["requirement_ids"]):
            _error(errors, "dangling_reference", p + "/requirement_ids")
        if cap is None or query is None or obs["capability_id"] not in query["capability_ids"]:
            _error(errors, "scope_mismatch", p + "/capability_id")
        elif obs["operation"] not in cap["operations"]:
            _error(errors, "scope_mismatch", p + "/operation")
        status = obs["status"]
        if (obs["failure_class"] not in _STATUS_FAILURES[status]
                or (status in ("completed", "failed") and not obs["attempted"])
                or (status == "budget_exhausted" and obs["attempted"])
                or (status != "completed" and obs["evidence"])):
            _error(errors, "invalid_state", p)
        for j, item in enumerate(obs["evidence"]):
            ep = p + "/evidence/" + str(j)
            if item["requirement_id"] not in obs["requirement_ids"]:
                _error(errors, "dangling_reference", ep + "/requirement_id")
            candidate = item["candidate"]
            receipt = item["temporal_receipt"]
            if receipt is not None:
                fresh = requirements.get(item["requirement_id"], {}).get("freshness")
                if (_temporal_invalid(validate_temporal_source_receipt, receipt)
                        or fresh is None or fresh["kind"] != "as_of"
                        or receipt["as_of"] != fresh["as_of"]
                        or receipt["source_url"] != candidate["location"]["locator"]
                        or receipt["retrieved_at"] != candidate["read"]["observed_at"]):
                    _error(errors, "invalid_provenance", ep + "/temporal_receipt")
            if cap is not None:
                errors.extend(_candidate_errors(candidate, cap, observed, ep + "/candidate"))
            identity = (*_binding(candidate), candidate["source_id"], candidate["version"],
                        candidate["location"]["kind"], candidate["location"]["locator"])
            old = candidate_bindings.setdefault(candidate["candidate_id"], identity)
            if old != identity:
                _error(errors, "invalid_provenance", ep + "/candidate/candidate_id")
    used = sum(o["attempted"] for o in ledger)
    if used > plan["budget"]["max_external_calls"]:
        _error(errors, "invalid_budget", "/observations")
    errors.extend(_previous_errors(plan, data))
    return sorted(set(errors))


def assess_search_coverage(plan, observations):
    """Assess reads, reserve at most one follow-up, and return cumulative state."""
    errors = _assessment_errors(plan, observations)
    if errors:
        raise SearchCoverageError(errors[0])
    ledger = observations["observations"]
    planned = {c["capability_id"]: c for c in plan["capabilities"]["capabilities"]}
    current = {c["capability_id"]: c for c in observations["capabilities"]["capabilities"]}
    rows = [_coverage_row(r, plan, observations, planned, current) for r in plan["spec"]["requirements"]]
    used = sum(o["attempted"] for o in ledger)
    budget = plan["budget"]
    if used > budget["max_external_calls"]:
        raise SearchCoverageError("invalid_budget:/observations")
    remaining_calls = budget["max_external_calls"] - used
    unresolved = {r["requirement_id"] for r in rows if r["required"] and r["status"] != "supported"}
    if remaining_calls == 0:
        for row in rows:
            if row["requirement_id"] in unresolved:
                row["gap_reasons"] = [g for g in _GAP_ORDER if g in set(row["gap_reasons"]) | {"budget_exhausted"}]
                if row["status"] != "conflicting":
                    row["status"] = row["gap_reasons"][0]
    previous = observations["previous_state"]
    rounds = previous["used_follow_up_rounds"] if previous else 0
    issued = list(previous["issued_follow_up_query_ids"]) if previous else []
    revision = previous["revision"] + 1 if previous else 1
    attempted = {o["query_id"] for o in ledger if o["attempted"]}
    new_ids = []
    if unresolved and rounds == 0 and remaining_calls:
        for query in plan["queries"]:
            if (query["round"] == 1 and query["query_id"] not in attempted
                    and unresolved.intersection(query["requirement_ids"])
                    and any(_cap_gap(cid, planned, current) is None for cid in query["capability_ids"])):
                new_ids.append(query["query_id"])
                if len(new_ids) == remaining_calls:
                    break
        if new_ids:
            rounds = 1
            issued.extend(new_ids)
    state = dict(schema_version="search_assessment_state/v1", plan_id=plan["plan_id"], revision=revision,
                 ledger_digest=_digest(ledger), used_external_calls=used, used_follow_up_rounds=rounds,
                 observation_ids=[o["observation_id"] for o in ledger], issued_follow_up_query_ids=issued)
    candidate_ids = list(dict.fromkeys(cid for row in rows for cid in row["candidate_ids"]))
    result = dict(schema_version="search_coverage/v1", plan_id=plan["plan_id"],
                snapshot_id=observations["capabilities"]["snapshot_id"], requirements=rows,
                candidate_ids=candidate_ids,
                follow_up="none" if not unresolved else "search_once" if new_ids else "partial_answer",
                follow_up_query_ids=new_ids, state=state,
                remaining_budget=dict(queries=budget["max_queries"] - len(attempted | set(issued)),
                                      external_calls=remaining_calls, follow_up_rounds=1 - rounds),
                claim_boundary="attributed_assessments_do_not_authenticate_access_execution_or_truth")
    errors = _shape(result, ("ref", "coverage"))
    if errors:
        raise SearchCoverageError(sorted(set(errors))[0])
    return result

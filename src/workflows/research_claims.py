"""Inspectable, author-supplied evidence records (``research_claims/v1``).

This is local contract validation, NEVER authentication or truth verification.
It reads no files, opens no URLs, invokes no model, and persists no state. All
text, including instruction-shaped quotes or search queries, is inert data.
Quote/location/observation references are opaque bounded strings: their targets
are not opened or checked, and source bodies/transcripts are not accepted.

The closed payload has schema_version plus four lists, with keys in _FIELDS:

* sources: publisher and domain grouping, nullable origin_group, source_type
  (primary/secondary/unknown), provenance (direct/republication/unknown), and a
  nullable original_source_id. A republication must point directly to a known
  original with the same origin_group and cannot itself be primary. Unknown
  provenance has null origin_group/original_source_id and unknown source_type.
* claims: bounded text, literal boolean contested, author-assigned status
  (corroborated/unresolved/refuted), evidence_ids and counter_search_ids.
* evidence: owning claim/source ids, stance (support/refute), kind
  (primary/secondary), quote_ref and location_ref. Primary evidence requires a
  direct primary source, not merely an unused primary source in the source list.
* counter_searches: owning claim id, query, outcome, result_source_ids and
  observation_ref. results_found requires nonempty results; no_results requires
  empty results. Both require an observation ref. not_observed requires empty
  results and a null observation ref and cannot satisfy corroboration.

Ids are literal ASCII tokens, unique within their typed table. References must
resolve and claim ownership must agree in both directions. Nothing is repaired
or coerced. Unresolved claims may have no evidence; refuted requires refuting
records, not proof of refutation. All corroborated claims require support. A
contested corroborated claim additionally needs two original groups on distinct
publisher domains, primary supporting evidence, and a linked observed counter-
search. Supplied refuting evidence also activates this gate, regardless of flag.
Republications use their ORIGINAL's domain/group/publisher for independence.

Domain groupings and origin groups are supplied assertions, not independently
established facts. A domain must be canonical lowercase ASCII and match the URL
host or a DNS suffix of it. Callers must supply the publisher-level domain
(grouping sibling subdomains together), not arbitrary per-page host labels.
Known equal publishers and nested domain aliases cannot establish independence.
There is no public-suffix database, affiliation inference, quote matching, or
verification that any source, primary classification, or search is authentic.
A clean error list means only that these supplied records satisfy this contract.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import combinations
from typing import Any, TypeGuard
from urllib.parse import urlsplit


RESEARCH_CLAIMS_SCHEMA_VERSION = "research_claims/v1"

_FIELDS = {
    "sources": (
        "id", "url", "domain", "publisher", "origin_group", "provenance", "original_source_id", "source_type",
    ),
    "claims": ("id", "text", "contested", "status", "evidence_ids", "counter_search_ids"),
    "evidence": ("id", "claim_id", "source_id", "stance", "kind", "quote_ref", "location_ref"),
    "counter_searches": ("id", "claim_id", "query", "outcome", "result_source_ids", "observation_ref"),
}

_Record = Mapping[str, Any]
_Index = dict[str, _Record]
_Tables = dict[str, _Index]


def research_claims_errors(mapping: object) -> list[str]:
    """Return deterministic contract errors without modifying or trusting input.

    Validation stops between shape, reference/metadata, and status phases when
    earlier errors would make the next phase ambiguous. There is no truth score.
    """
    if not isinstance(mapping, Mapping):
        return ["research_claims: expected mapping"]
    errors: list[str] = []
    tables = _tables(mapping, errors)
    if errors:
        return errors
    for source in tables["sources"].values():
        _source_errors(source, tables["sources"], errors)
    for claim in tables["claims"].values():
        _claim_links_errors(claim, tables, errors)
    for name, field in (("evidence", "evidence_ids"), ("counter_searches", "counter_search_ids")):
        for row in tables[name].values():
            path = f"research_claims.{name}[{row['id']}]"
            _owner_errors(row, tables["claims"], field, path, errors)
            if name == "evidence":
                _evidence_errors(row, tables["sources"], path, errors)
            else:
                _counter_search_errors(row, tables["sources"], path, errors)
    if errors:
        return errors
    for claim in tables["claims"].values():
        _status_errors(claim, tables, errors)
    return errors


def _tables(mapping: _Record, errors: list[str]) -> _Tables:
    if set(mapping) != {"schema_version", *_FIELDS}:
        errors.append("research_claims.keys: expected declared fields")
    if mapping.get("schema_version") != RESEARCH_CLAIMS_SCHEMA_VERSION:
        errors.append("research_claims.schema_version: expected research_claims/v1")
    tables: _Tables = {}
    for name, fields in _FIELDS.items():
        rows = mapping.get(name)
        index: _Index = {}
        tables[name] = index
        if not isinstance(rows, list):
            errors.append(f"research_claims.{name}: expected list")
            continue
        for position, row in enumerate(rows):
            path = f"research_claims.{name}[{position}]"
            if not isinstance(row, Mapping):
                errors.append(f"{path}: expected mapping")
                continue
            if set(row) != set(fields):
                errors.append(f"{path}.keys: expected declared fields")
            identifier = row.get("id")
            if not _identifier(identifier):
                errors.append(f"{path}.id: invalid identifier")
            elif identifier in index:
                errors.append(f"{path}.id: duplicate identifier")
            else:
                index[identifier] = row
    return tables


def _source_errors(row: _Record, sources: _Index, errors: list[str]) -> None:
    path = f"research_claims.sources[{row['id']}]"
    if not _line(row["publisher"], 200):
        errors.append(f"{path}.publisher: expected bounded publisher label")
    if row["origin_group"] is not None and not _identifier(row["origin_group"]):
        errors.append(f"{path}.origin_group: expected identifier or null")
    if row["provenance"] not in ("direct", "republication", "unknown"):
        errors.append(f"{path}.provenance: expected direct, republication or unknown")
    if row["source_type"] not in ("primary", "secondary", "unknown"):
        errors.append(f"{path}.source_type: expected primary, secondary or unknown")
    original_id = row["original_source_id"]
    if row["provenance"] == "republication":
        original = sources.get(original_id) if _identifier(original_id) else None
        if (original is None or original_id == row["id"] or original["provenance"] != "direct"
                or not _identifier(row["origin_group"]) or original["origin_group"] != row["origin_group"]
                or row["source_type"] == "primary"):
            errors.append(
                f"{path}.original_source_id: republication requires matching direct original and nonprimary type"
            )
    elif original_id is not None:
        errors.append(f"{path}.original_source_id: only republication may reference an original")
    if row["provenance"] == "unknown" and (row["origin_group"] is not None or row["source_type"] != "unknown"):
        errors.append(f"{path}.provenance: unknown provenance cannot assert an origin or source type")
    host = _url_host(row["url"])
    if host is None:
        errors.append(f"{path}.url: expected absolute credential-free HTTP(S) URL")
    domain = row["domain"]
    if not _domain(domain) or (host is not None and host != domain and not host.endswith("." + domain)):
        errors.append(f"{path}.domain: expected canonical matching domain grouping")


def _claim_links_errors(row: _Record, tables: _Tables, errors: list[str]) -> None:
    path = f"research_claims.claims[{row['id']}]"
    if not _line(row["text"]):
        errors.append(f"{path}.text: expected bounded claim text")
    if type(row["contested"]) is not bool:
        errors.append(f"{path}.contested: expected boolean")
    if row["status"] not in ("corroborated", "unresolved", "refuted"):
        errors.append(f"{path}.status: expected corroborated, unresolved or refuted")
    for field, target, label in (
        ("evidence_ids", "evidence", "evidence"),
        ("counter_search_ids", "counter_searches", "counter_search"),
    ):
        ref_path = f"{path}.{field}"
        for ref in _references(row[field], ref_path, errors):
            if ref not in tables[target]:
                errors.append(f"{ref_path}: unknown {label}")
            elif tables[target][ref]["claim_id"] != row["id"]:
                errors.append(f"{ref_path}: claim ownership mismatch")


def _owner_errors(row: _Record, claims: _Index, field: str, path: str, errors: list[str]) -> None:
    claim_id = row["claim_id"]
    if not _identifier(claim_id) or claim_id not in claims:
        errors.append(f"{path}.claim_id: unknown claim")
    elif row["id"] not in _references(claims[claim_id][field], f"research_claims.claims[{claim_id}].{field}", []):
        errors.append(f"{path}: not linked by owning claim")


def _evidence_errors(row: _Record, sources: _Index, path: str, errors: list[str]) -> None:
    source_id = row["source_id"]
    if not _identifier(source_id) or source_id not in sources:
        errors.append(f"{path}.source_id: unknown source")
    else:
        source = sources[source_id]
        if row["kind"] == "primary" and (source["source_type"] != "primary" or source["provenance"] != "direct"):
            errors.append(f"{path}.kind: primary evidence requires direct primary source")
    if row["stance"] not in ("support", "refute"):
        errors.append(f"{path}.stance: expected support or refute")
    if row["kind"] not in ("primary", "secondary"):
        errors.append(f"{path}.kind: expected primary or secondary")
    for field in ("quote_ref", "location_ref"):
        if not _line(row[field]):
            errors.append(f"{path}.{field}: expected bounded evidence reference")


def _counter_search_errors(row: _Record, sources: _Index, path: str, errors: list[str]) -> None:
    refs = _references(row["result_source_ids"], f"{path}.result_source_ids", errors)
    for ref in refs:
        if ref not in sources:
            errors.append(f"{path}.result_source_ids: unknown source")
    if not _line(row["query"]):
        errors.append(f"{path}.query: expected bounded search query")
    outcome = row["outcome"]
    if outcome not in ("results_found", "no_results", "not_observed"):
        errors.append(f"{path}.outcome: expected results_found, no_results or not_observed")
    if outcome == "not_observed":
        if refs or row["observation_ref"] is not None:
            errors.append(f"{path}.outcome: not_observed cannot carry results or observation")
    else:
        if not _line(row["observation_ref"]):
            errors.append(f"{path}.observation_ref: observed search requires bounded reference")
        if (outcome == "no_results" and refs) or (outcome == "results_found" and not refs):
            errors.append(f"{path}.outcome: inconsistent result_source_ids")


def _status_errors(claim: _Record, tables: _Tables, errors: list[str]) -> None:
    path = f"research_claims.claims[{claim['id']}].status"
    items = [tables["evidence"][ref] for ref in claim["evidence_ids"]]
    support = [item for item in items if item["stance"] == "support"]
    refute = any(item["stance"] == "refute" for item in items)
    if claim["status"] == "refuted" and not refute:
        errors.append(f"{path}: refuted requires refute evidence")
    if claim["status"] == "corroborated" and not support:
        errors.append(f"{path}: corroborated requires support evidence")
    if claim["status"] != "corroborated" or not (claim["contested"] or refute):
        return
    if not any(tables["counter_searches"][ref]["outcome"] in ("results_found", "no_results")
               for ref in claim["counter_search_ids"]):
        errors.append(f"{path}: corroboration requires a linked observed counter-search")
    if not any(item["kind"] == "primary" for item in support):
        errors.append(f"{path}: corroboration requires a primary support evidence record")
    originals = []
    for item in support:
        source = tables["sources"][item["source_id"]]
        if source["provenance"] == "republication":
            source = tables["sources"][source["original_source_id"]]
        if source["provenance"] == "direct" and source["origin_group"] is not None:
            originals.append(source)
    if len({s["domain"] for s in originals}) < 2 or len({s["origin_group"] for s in originals}) < 2:
        errors.append(f"{path}: corroboration requires two distinct domains and original source groups")
    if not any(_independent(left, right) for left, right in combinations(originals, 2)):
        errors.append(f"{path}: corroboration requires an independent publisher/domain pair")


def _independent(left: _Record, right: _Record) -> bool:
    a, b = left["domain"], right["domain"]
    return (
        a != b and not a.endswith("." + b) and not b.endswith("." + a)
        and left["origin_group"] != right["origin_group"]
        and " ".join(left["publisher"].split()).casefold() != " ".join(right["publisher"].split()).casefold()
    )


def _references(value: object, path: str, errors: list[str]) -> list[str]:
    if (not isinstance(value, list) or not all(_identifier(ref) for ref in value)
            or len(value) != len(set(value))):
        errors.append(f"{path}: invalid reference list")
        return []
    return value


def _line(value: object, limit: int = 2048) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= limit
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _domain(value: object) -> bool:
    return (isinstance(value, str) and len(value) <= 253
            and re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}", value) is not None)


def _url_host(value: object) -> str | None:
    if not isinstance(value, str) or not _line(value) or any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (parsed.scheme not in ("http", "https") or parsed.username is not None
                or parsed.password is not None or not _domain(host) or port == 0):
            return None
        return host
    except ValueError:
        return None


def _identifier(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is not None

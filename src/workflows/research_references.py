"""Pinned OSS study metadata; validity is not proof of reading or execution.

Accept a list, one ``reference_study/v1`` record, or a mapping containing a
``reference_studies`` list. A record declares ``id``, ``repo``, a full literal
40/64-hex Git object ``revision``, and ``license: {identifier, file,
provenance: {repo, revision}}`` bound to exactly those supplied identifiers.
This validates the pin's syntax, not its existence or object type upstream.

``read_files`` lists ``{path, role, line_ranges, symbols}``; inclusive ranges
are positive integer pairs and symbols are explicit identifiers (possibly an
empty list). ``read_scope`` is ``readme_only`` or ``core_modules`` and
``core_analysis_status`` is ``not_performed``, ``partial`` or ``complete``.
README-only records cannot claim core analysis. Core scope requires a core
file, and completed core analysis requires a linked code finding in each of
``findings.mechanism``, ``findings.tradeoffs`` and ``findings.limits``.

Each finding is ``{summary, evidence: [{kind, path, line_range?, symbol?}]}``.
Evidence kind is ``code`` or ``documentation``; at least one locator is
required and every supplied locator must match the declared read inventory.
Code citations require a core file. Identifiers are never normalized or
repaired; ranges must fit wholly in a declared range. No repository is read,
fetched or executed. License text and actual reading need separate evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reference_studies_errors(value: list[Any] | Mapping[str, Any]) -> list[str]:
    """Return errors without fetching repositories or changing supplied identifiers."""
    if isinstance(value, Mapping):
        records = value.get("reference_studies") if "reference_studies" in value else [value]
    else:
        records = value
    if not isinstance(records, list):
        return ["reference_studies must be a list or record mapping"]
    errors = []
    ids = set()
    for index, record in enumerate(records):
        path = f"reference_studies[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        for key in ("schema_version", "id", "repo", "revision", "license", "read_scope",
                    "core_analysis_status", "read_files", "findings"):
            if key not in record:
                errors.append(f"{path}.{key} is required")
        if record.get("schema_version") != "reference_study/v1":
            errors.append(f"{path}.schema_version must be reference_study/v1")
        ident = record.get("id")
        if not _text(ident) or ident in ids:
            errors.append(f"{path}.id must be unique nonempty text")
        else:
            ids.add(ident)
        if not _text(record.get("repo")):
            errors.append(f"{path}.repo must be nonempty text")
        if not _revision(record.get("revision")):
            errors.append(f"{path}.revision must be a full literal Git object id (40 or 64 hex characters)")
        license_record = record.get("license")
        if not isinstance(license_record, Mapping):
            errors.append(f"{path}.license must be a mapping")
        else:
            for key in ("identifier", "file"):
                if not _text(license_record.get(key)):
                    errors.append(f"{path}.license.{key} must be nonempty text")
            provenance = license_record.get("provenance")
            if not isinstance(provenance, Mapping):
                errors.append(f"{path}.license.provenance must be a mapping")
            elif (not _revision(provenance.get("revision")) or
                  provenance.get("revision") != record.get("revision") or
                  not _text(provenance.get("repo")) or
                  provenance.get("repo") != record.get("repo")):
                errors.append(f"{path}.license.provenance must match the literal repo and revision")
        scope = record.get("read_scope")
        status = record.get("core_analysis_status")
        if scope not in ("readme_only", "core_modules"):
            errors.append(f"{path}.read_scope must be readme_only or core_modules")
        if status not in ("not_performed", "partial", "complete"):
            errors.append(f"{path}.core_analysis_status is invalid")
        if scope == "readme_only" and status != "not_performed":
            errors.append(f"{path} readme_only cannot claim core analysis")
        files = record.get("read_files")
        if scope == "core_modules" and (
            not isinstance(files, list) or not any(
                isinstance(item, Mapping) and item.get("role") == "core" for item in files
            )
        ):
            errors.append(f"{path}.core_modules requires a declared core file")
        errors.extend(_read_files_errors(files, scope, path))
        errors.extend(_findings_errors(record.get("findings"), status, files, path))
    return errors


def _findings_errors(findings: Any, status: Any, files: Any, path: str) -> list[str]:
    if not isinstance(findings, Mapping):
        return [f"{path}.findings must be a mapping"]
    errors = []
    for key in ("mechanism", "tradeoffs", "limits"):
        items = findings.get(key)
        label = f"{path}.findings.{key}"
        if not isinstance(items, list):
            errors.append(f"{label} must be a list")
            continue
        if status == "complete" and not items:
            errors.append(f"{label} requires linked findings for completed core analysis")
        core_linked = False
        for item in items:
            if not isinstance(item, Mapping) or not _text(item.get("summary")):
                errors.append(f"{label} requires a finding summary")
                continue
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{label} requires a nonempty evidence list")
                continue
            for index, citation in enumerate(evidence):
                citation_errors = _citation_errors(citation, files, f"{label}.evidence[{index}]")
                errors.extend(citation_errors)
                if not citation_errors and citation.get("kind") == "code":
                    core_linked = True
        if status == "complete" and not core_linked:
            errors.append(f"{label} requires linked core code evidence")
    return errors


def _citation_errors(citation: Any, files: Any, path: str) -> list[str]:
    if not isinstance(citation, Mapping):
        return [f"{path} must be a mapping"]
    errors = []
    kind = citation.get("kind")
    if kind not in ("code", "documentation"):
        errors.append(f"{path}.kind must be code or documentation")
    name = citation.get("path")
    declared = next((item for item in files if isinstance(item, Mapping) and
                     _text(name) and item.get("path") == name), None) if isinstance(files, list) else None
    if declared is None:
        return errors + [f"{path}.path must match a declared read file literally"]
    if kind == "code" and declared.get("role") != "core":
        errors.append(f"{path} code evidence requires a declared core file")
    if "line_range" not in citation and "symbol" not in citation:
        errors.append(f"{path} requires line_range or symbol")
    if "line_range" in citation:
        bounds = citation["line_range"]
        ranges = declared.get("line_ranges")
        if (not _line_range(bounds) or not isinstance(ranges, list) or not any(
            _line_range(r) and r[0] <= bounds[0] and bounds[1] <= r[1] for r in ranges
        )):
            errors.append(f"{path}.line_range must be contained in a declared read range")
    if "symbol" in citation:
        symbol = citation["symbol"]
        symbols = declared.get("symbols")
        if not _text(symbol) or not isinstance(symbols, list) or symbol not in symbols:
            errors.append(f"{path}.symbol must match a declared read symbol literally")
    return errors


def _read_files_errors(files: Any, scope: Any, path: str) -> list[str]:
    if not isinstance(files, list) or not files:
        return [f"{path}.read_files must be a nonempty list"]
    errors = []
    paths = set()
    for index, item in enumerate(files):
        label = f"{path}.read_files[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        name = item.get("path")
        if not _text(name) or name in paths:
            errors.append(f"{label}.path must be unique nonempty text")
        else:
            paths.add(name)
        role = item.get("role")
        if role not in ("readme", "core", "license", "other"):
            errors.append(f"{label}.role is invalid")
        if scope == "readme_only" and role not in ("readme", "license"):
            errors.append(f"{label} contradicts readme_only scope")
        ranges = item.get("line_ranges")
        if not isinstance(ranges, list) or not ranges or not all(_line_range(r) for r in ranges):
            errors.append(f"{label}.line_ranges requires positive inclusive integer pairs")
        symbols = item.get("symbols")
        if (not isinstance(symbols, list) or not all(_text(s) for s in symbols) or
                len(symbols) != len(set(symbols))):
            errors.append(f"{label}.symbols must be a unique list of identifiers")
    return errors


def _line_range(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(
        type(number) is int and number > 0 for number in value
    ) and value[0] <= value[1]


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) in (40, 64) and all(
        char in "0123456789abcdefABCDEF" for char in value
    )

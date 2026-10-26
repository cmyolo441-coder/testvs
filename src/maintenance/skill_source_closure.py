"""Bind a resolved recurring-watch finding to the registry row it resolves.

`docs/SKILL-SOURCES.md` records, per (OMH unit, upstream source) pair, the
`reviewed_ref` checkpoint the external upstream tracker diffs against. The
registry's own rule is that resolving a tracker finding advances `reviewed_ref`
and `reviewed_on` in the same pull request. Until this module, that rule was
prose: a capability change could merge with the row untouched, and a later run
would re-evaluate a range that had already been reviewed and present resolved
work as fresh risk.

The gate closes that by making the ledger the chain of custody. Each receipt in
`docs/skill-source-receipts.json` names one candidate, the checkpoint it moved
from, the checkpoint it moved to, and the terminal disposition. A row is
`closed` only when its candidate's receipt chain terminates exactly at the
row's current `reviewed_ref` and `reviewed_on`. Half an atomic change therefore
fails from either side: implementation without the row move leaves the chain
short of the row (`closure_checkpoint_missing`), and a row move without a
receipt leaves the row off its enrolled baseline with nothing recording why
(`closure_receipt_missing`).

Boundaries. Everything here is local and deterministic: no network, no
subprocess, no git resolution, no watched repository is contacted, no issue is
closed and no pull request is merged. A checkpoint is an opaque identity
string, so "newer" is proved structurally -- the chain may not repeat a
checkpoint it already left -- and never by ordering commits this code cannot
see. The gate validates OMH-owned continuity *after* a human or an authorized
workflow decided; it never decides whether a capability should be adopted, and
issue closure alone is not evidence of anything here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from ..catalogs.skill_source_closure import (
    CLOSURE_SCHEMA,
    DEFAULT_OWNER,
    DISPOSITIONS,
    FAILURE_CLASSES,
    LEDGER_PATH,
    MAX_DIAGNOSTIC_CHARS,
    MAX_RATIONALE_CHARS,
    MAX_RECEIPTS,
    PRE_RECEIPT_BASELINE,
    PRE_RECEIPT_CENSUS_DIGEST,
    PreReceiptBaseline,
    RECEIPT_SCHEMA,
    REGISTRY_PATH,
    SETTLED_BY_RECEIPT,
    census_digest,
    pre_receipt_baselines,
)


_HEADER_PREFIX = "| OMH skill |"
_UNIT = re.compile(r"`([^`\n]+)`")
# Case-insensitive: the scheme is stripped before the key is lowercased, so a
# case-sensitive match would leave `HTTPS://` in the key and mint a second
# candidate for the source the lowercased URL already names.
_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_RECEIPT_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DECISION_REF = re.compile(r"^#[1-9][0-9]*$")
_REVIEW_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
_DELIMITER_ROW = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")
# Markdown escapes a literal pipe inside a cell as `\|`. Splitting on raw pipes
# makes the first `Paths studied` cell that needs one unwritable, and those
# cells are already dense with backticked paths and prose.
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _table_cells(line: str) -> list[str]:
    """Split one Markdown table row into trimmed cells.

    Indentation is ignored and `\\|` is honoured as an escaped literal pipe, so
    a cell may contain one without destroying the row's cell count.
    """
    parts = _CELL_SPLIT.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [part.replace("\\|", "|").strip() for part in parts]


_REQUIRED_RECEIPT_FIELDS = (
    "receipt_id",
    "candidate_key",
    "prior_checkpoint",
    "next_checkpoint",
    "disposition",
    "decision_ref",
    "reviewed_on",
    "rationale",
)
_OPTIONAL_RECEIPT_FIELDS = ("supersedes",)


class ClosureFinding(TypedDict):
    failure_class: str
    candidate_key: str | None
    receipt_id: str | None
    owner: str
    detail: str


class ClosureRow(TypedDict):
    candidate_key: str
    unit: str
    source: str
    state: str
    reason: str
    reviewed_on: str
    checkpoint: str
    receipt_ids: list[str]
    disposition: str | None
    # Where the next watch run starts for this candidate; None while the row is
    # held. It has no caller inside `src/` and is not meant to: it is output,
    # carried in the `docs skill-sources --json` payload, so its consumer is the
    # operator or the external tracker. Publishing it is what keeps the one rule
    # that matters out of every consumer's hands -- that a held row yields no
    # boundary. A consumer re-deriving that from `state` and `checkpoint` and
    # getting it wrong re-emits a finding this repository has already resolved,
    # which is the defect the whole contract exists to remove.
    scan_from: str | None


class ClosureReport(TypedDict):
    schema_version: str
    mode: str
    observed: bool
    ok: bool
    registry: str
    ledger: str
    rows: list[ClosureRow]
    findings: list[ClosureFinding]
    summary: dict[str, int]
    bounds: dict[str, object]
    failure_classes: list[str]
    closure_boundary: str


def _safe(value: str) -> str:
    """Render registry- or ledger-controlled text as a bounded diagnostic."""
    cleaned = _CONTROL.sub("�", value.replace("\n", " ").replace("\r", " "))
    if len(cleaned) > MAX_DIAGNOSTIC_CHARS:
        return cleaned[:MAX_DIAGNOSTIC_CHARS] + "..."
    return cleaned


def normalize_source(value: str) -> str:
    """Reduce an upstream source cell to the identity half of a candidate key.

    Only the leading URL token is taken. Some cells carry trailing prose after
    the URL -- the `code-review` row names its plugin distribution that way --
    and that prose is a description of the source, not part of its identity, so
    rewording it must not silently mint a new candidate.
    """
    first = value.strip().split()[0] if value.strip().split() else ""
    return _SCHEME.sub("", first).rstrip("/").lower()


def candidate_key(unit: str, source: str) -> str:
    """One stable key per (OMH unit, upstream source) pair."""
    return f"{unit.strip().lower()}@{normalize_source(source)}"


def parse_registry(text: str) -> tuple[list[dict[str, str]], list[ClosureFinding]]:
    """Parse the hand-written registry table into candidate rows.

    The table body is delimited structurally -- header, delimiter, then every
    consecutive line that is a table row -- rather than by matching a row's
    expected opening characters. That distinction is the whole point here. This
    is a hand-written Markdown table, so a row that is indented, or whose skill
    cell opens with prose instead of a backtick, is the expected accident. An
    earlier version selected rows with a `startswith` on the exact opening
    characters, and such a row was skipped with no finding at all: the row
    vanished from the audit and the gate went green over an unenrolled
    candidate. Every line inside the body now either yields a candidate or
    produces `registry_row_unparsed`, and nothing in between.

    Only the shipped table is parsed, and parsing stops at its end. The
    candidate section below it holds researched leads that have not shipped and
    carry no checkpoint for a receipt to bind to.
    """
    lines = text.splitlines()
    header_index = next((index for index, line in enumerate(lines)
                         if line.strip().startswith(_HEADER_PREFIX)), None)
    if header_index is None:
        finding: ClosureFinding = {
            "failure_class": "registry_row_unparsed", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{REGISTRY_PATH} has no shipped-skills table header.",
        }
        return [], [finding]
    columns = _table_cells(lines[header_index])
    rows: list[dict[str, str]] = []
    findings: list[ClosureFinding] = []
    body = header_index + 1
    if body < len(lines) and _DELIMITER_ROW.match(lines[body].strip()):
        body += 1
    for line in lines[body:]:
        if not line.strip().startswith("|"):
            break
        cells = _table_cells(line)
        if len(cells) != len(columns):
            findings.append({
                "failure_class": "registry_row_unparsed", "candidate_key": None,
                "receipt_id": None, "owner": DEFAULT_OWNER,
                "detail": f"Row has {len(cells)} cells, expected {len(columns)}: {_safe(line)}",
            })
            continue
        row = dict(zip(columns, cells, strict=True))
        unit_match = _UNIT.search(row.get("OMH skill", ""))
        source = row.get("Upstream repo", "")
        if unit_match is None or not source:
            findings.append({
                "failure_class": "registry_row_unparsed", "candidate_key": None,
                "receipt_id": None, "owner": DEFAULT_OWNER,
                "detail": (
                    "Row has no backticked OMH unit or no upstream source, so no candidate key "
                    f"can be derived for it: {_safe(line)}"
                ),
            })
            continue
        rows.append({
            "candidate_key": candidate_key(unit_match.group(1), source),
            "unit": unit_match.group(1).strip().lower(),
            "source": normalize_source(source),
            "reviewed_on": row.get("reviewed_on", ""),
            "checkpoint": row.get("reviewed_ref", ""),
        })
    return rows, findings


def _receipt_field_findings(index: int, entry: object) -> tuple[dict[str, object] | None, list[ClosureFinding]]:
    """Validate one ledger entry's shape before any chain reasoning touches it."""
    findings: list[ClosureFinding] = []
    if not isinstance(entry, dict):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} is not an object.",
        })
        return None, findings
    raw_id = entry.get("receipt_id")
    receipt_id = raw_id if isinstance(raw_id, str) else None
    known = set(_REQUIRED_RECEIPT_FIELDS) | set(_OPTIONAL_RECEIPT_FIELDS)
    for field in sorted(set(entry) - known):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} carries unknown field {_safe(str(field))}.",
        })
    for field in _REQUIRED_RECEIPT_FIELDS:
        if field not in entry:
            findings.append({
                "failure_class": "receipt_field_invalid", "candidate_key": None,
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": f"Ledger entry {index} is missing {field}.",
            })
    if receipt_id is None or not _RECEIPT_ID.match(receipt_id):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} receipt_id must be lowercase kebab-case.",
        })
    key = entry.get("candidate_key")
    if not isinstance(key, str) or not key:
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} candidate_key must be a non-empty string.",
        })
    prior = entry.get("prior_checkpoint")
    if prior is not None and not isinstance(prior, str):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} prior_checkpoint must be a string or null.",
        })
    nxt = entry.get("next_checkpoint")
    if not isinstance(nxt, str) or not nxt:
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} next_checkpoint must be a non-empty string.",
        })
    disposition = entry.get("disposition")
    if disposition not in DISPOSITIONS:
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} disposition must be one of {', '.join(DISPOSITIONS)}.",
        })
    decision = entry.get("decision_ref")
    if not isinstance(decision, str) or not _DECISION_REF.match(decision):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} decision_ref must name an OMH issue or pull request as #N.",
        })
    reviewed_on = entry.get("reviewed_on")
    if not isinstance(reviewed_on, str) or not _REVIEW_DATE.match(reviewed_on):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} reviewed_on must be an ISO YYYY-MM-DD date.",
        })
    supersedes = entry.get("supersedes")
    if supersedes is not None and not isinstance(supersedes, str):
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} supersedes must be a receipt_id or null.",
        })
    rationale = entry.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        findings.append({
            "failure_class": "receipt_field_invalid", "candidate_key": None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": f"Ledger entry {index} rationale must be a non-empty string.",
        })
    elif len(rationale) > MAX_RATIONALE_CHARS or "\n" in rationale or _LINK.search(rationale):
        # A rationale states the OMH decision in one bounded line. Watch
        # evidence and upstream links belong in the private continuity record
        # and in the cited OMH issue, not in a public ledger field.
        findings.append({
            "failure_class": "rationale_over_budget", "candidate_key": key if isinstance(key, str) else None,
            "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
            "detail": (
                f"Receipt {_safe(str(receipt_id))} rationale must be one line of at most "
                f"{MAX_RATIONALE_CHARS} characters and must cite the OMH decision rather than a link."
            ),
        })
    if findings:
        return None, findings
    return dict(entry), findings


def parse_ledger(text: str) -> tuple[list[dict[str, object]], list[ClosureFinding]]:
    """Parse the receipt ledger into ordered receipts, validating entry shape."""
    findings: list[ClosureFinding] = []
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        findings.append({
            "failure_class": "ledger_unreadable", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{LEDGER_PATH} is not valid JSON: {_safe(str(exc))}",
        })
        return [], findings
    if not isinstance(document, dict) or document.get("schema_version") != RECEIPT_SCHEMA:
        findings.append({
            "failure_class": "ledger_unreadable", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{LEDGER_PATH} must be an object declaring schema_version {RECEIPT_SCHEMA}.",
        })
        return [], findings
    entries = document.get("receipts")
    if not isinstance(entries, list):
        findings.append({
            "failure_class": "ledger_unreadable", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{LEDGER_PATH} must carry a receipts list.",
        })
        return [], findings
    if len(entries) > MAX_RECEIPTS:
        findings.append({
            "failure_class": "ledger_unreadable", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{LEDGER_PATH} holds {len(entries)} receipts, over the {MAX_RECEIPTS} bound.",
        })
        return [], findings
    receipts: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        receipt, entry_findings = _receipt_field_findings(index, entry)
        findings.extend(entry_findings)
        if receipt is not None:
            receipts.append(receipt)
    return receipts, findings


def _ledger_order_findings(receipts: list[dict[str, object]]) -> list[ClosureFinding]:
    """Append-only shape: dates never go backwards, identities are never reused."""
    findings: list[ClosureFinding] = []
    seen: dict[str, int] = {}
    previous_date = ""
    for index, receipt in enumerate(receipts):
        receipt_id = str(receipt["receipt_id"])
        reviewed_on = str(receipt["reviewed_on"])
        if receipt_id in seen:
            findings.append({
                "failure_class": "receipt_id_reused", "candidate_key": str(receipt["candidate_key"]),
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": (
                    f"Receipt {_safe(receipt_id)} appears at entries {seen[receipt_id]} and {index}: "
                    "one decision identity must bind to one row."
                ),
            })
        else:
            seen[receipt_id] = index
        if reviewed_on < previous_date:
            findings.append({
                "failure_class": "ledger_order_violation", "candidate_key": str(receipt["candidate_key"]),
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": (
                    f"Receipt {_safe(receipt_id)} is dated {reviewed_on} after {previous_date}: "
                    "receipts are appended in review order, never inserted."
                ),
            })
        previous_date = max(previous_date, reviewed_on)
    return findings


def _supersede_findings(receipts: list[dict[str, object]]) -> tuple[dict[str, str], list[ClosureFinding]]:
    """Resolve `supersedes` edges; return the accepted ones keyed by receipt id."""
    findings: list[ClosureFinding] = []
    accepted: dict[str, str] = {}
    position = {str(receipt["receipt_id"]): index for index, receipt in enumerate(receipts)}
    by_id = {str(receipt["receipt_id"]): receipt for receipt in receipts}
    for index, receipt in enumerate(receipts):
        target = receipt.get("supersedes")
        if not isinstance(target, str):
            continue
        receipt_id = str(receipt["receipt_id"])
        key = str(receipt["candidate_key"])
        if target not in by_id:
            findings.append({
                "failure_class": "supersede_reference_unknown", "candidate_key": key,
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": f"Receipt {_safe(receipt_id)} supersedes {_safe(target)}, which is not in the ledger.",
            })
            continue
        if str(by_id[target]["candidate_key"]) != key:
            findings.append({
                "failure_class": "supersede_reference_unrelated", "candidate_key": key,
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": (
                    f"Receipt {_safe(receipt_id)} supersedes {_safe(target)}, which resolves a different "
                    "candidate: one decision must never advance an unrelated row."
                ),
            })
            continue
        if position[target] >= index:
            findings.append({
                "failure_class": "ledger_order_violation", "candidate_key": key,
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": f"Receipt {_safe(receipt_id)} supersedes {_safe(target)}, which is not an earlier entry.",
            })
            continue
        accepted[receipt_id] = target
    return accepted, findings


def _chain_findings(
    key: str,
    chain: list[dict[str, object]],
    baseline_checkpoint: str | None,
    accepted_supersedes: dict[str, str],
) -> tuple[str, str | None, list[ClosureFinding]]:
    """Walk one candidate's receipts; return its settled checkpoint and date."""
    findings: list[ClosureFinding] = []
    current = baseline_checkpoint
    visited: set[str] = {baseline_checkpoint} if baseline_checkpoint is not None else set()
    settled_on: str | None = None
    by_id = {str(receipt["receipt_id"]): receipt for receipt in chain}
    for receipt in chain:
        receipt_id = str(receipt["receipt_id"])
        prior = receipt["prior_checkpoint"]
        nxt = str(receipt["next_checkpoint"])
        superseded = accepted_supersedes.get(receipt_id)
        allowed = {current}
        if superseded is not None and superseded in by_id:
            allowed.add(by_id[superseded]["prior_checkpoint"])
        if prior not in allowed:
            findings.append({
                "failure_class": "checkpoint_prior_stale", "candidate_key": key,
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": (
                    f"Receipt {_safe(receipt_id)} moves from {_safe(str(prior))} but the row stood at "
                    f"{_safe(str(current))}."
                ),
            })
        elif superseded is None and nxt in visited:
            findings.append({
                "failure_class": "checkpoint_not_superseding", "candidate_key": key,
                "receipt_id": receipt_id, "owner": DEFAULT_OWNER,
                "detail": (
                    f"Receipt {_safe(receipt_id)} sets the checkpoint back to {_safe(nxt)}, already left by "
                    "this chain; a correction must declare what it supersedes."
                ),
            })
        current = nxt
        visited.add(nxt)
        settled_on = str(receipt["reviewed_on"])
    return str(current), settled_on, findings


def skill_source_closure_report(
    *, root: Path | None = None, baselines: tuple[PreReceiptBaseline, ...] | None = None,
) -> ClosureReport:
    """Audit every registry row against the closure ledger.

    A row is `closed` when a receipt chain terminates exactly at its current
    checkpoint and review date, `not_applicable` when it still stands where the
    pre-receipt baseline enrolled it, and `held` otherwise, carrying the stable
    reason code that says which half of the atomic contract is missing.

    `baselines` defaults to the shipped enrolment table. It is a seam so a
    fixture can state its own starting point instead of depending on the 38
    live rows, whose checkpoints move whenever a real review lands.
    """
    resolved = (root or Path.cwd()).resolve()
    registry_path = resolved / REGISTRY_PATH
    ledger_path = resolved / LEDGER_PATH

    findings: list[ClosureFinding] = []
    try:
        registry_text = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        registry_text = ""
        findings.append({
            "failure_class": "registry_row_unparsed", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{REGISTRY_PATH} could not be read: {_safe(str(exc))}",
        })
    try:
        ledger_text = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        ledger_text = ""
        findings.append({
            "failure_class": "ledger_unreadable", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"{LEDGER_PATH} could not be read: {_safe(str(exc))}",
        })

    parsed_rows, registry_findings = parse_registry(registry_text) if registry_text else ([], [])
    findings.extend(registry_findings)
    receipts, ledger_findings = parse_ledger(ledger_text) if ledger_text else ([], [])
    findings.extend(ledger_findings)
    findings.extend(_ledger_order_findings(receipts))
    accepted_supersedes, supersede_findings = _supersede_findings(receipts)
    findings.extend(supersede_findings)

    by_key: dict[str, list[dict[str, str]]] = {}
    for row in parsed_rows:
        by_key.setdefault(row["candidate_key"], []).append(row)
    ambiguous = {key for key, group in by_key.items() if len(group) > 1}
    for key in sorted(ambiguous):
        findings.append({
            "failure_class": "ambiguous_candidate_match", "candidate_key": key,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": f"Candidate key {_safe(key)} matches {len(by_key[key])} registry rows; it must match exactly one.",
        })

    census = baselines if baselines is not None else pre_receipt_baselines()
    if baselines is None and census_digest(census) != PRE_RECEIPT_CENSUS_DIGEST:
        # Only the shipped census is frozen; an injected fixture states its own
        # starting point and is not subject to a digest it never declared.
        findings.append({
            "failure_class": "baseline_census_modified", "candidate_key": None,
            "receipt_id": None, "owner": DEFAULT_OWNER,
            "detail": (
                "The pre-receipt census no longer matches PRE_RECEIPT_CENSUS_DIGEST. It records where "
                "rows stood once, not where they stand now: a row that moved needs a closure receipt, "
                "not an edited baseline."
            ),
        })
    enrolled = {entry.candidate_key: entry for entry in census}
    for key in sorted(set(enrolled) - set(by_key)):
        findings.append({
            "failure_class": "stale_baseline_entry", "candidate_key": key,
            "receipt_id": None, "owner": enrolled[key].owner,
            "detail": f"Baseline entry {_safe(key)} matches no registry row; remove it or restore the row.",
        })

    chains: dict[str, list[dict[str, object]]] = {}
    for receipt in receipts:
        chains.setdefault(str(receipt["candidate_key"]), []).append(receipt)
    for key in sorted(set(chains) - set(by_key)):
        findings.append({
            "failure_class": "unmatched_candidate_key", "candidate_key": key,
            "receipt_id": str(chains[key][0]["receipt_id"]), "owner": DEFAULT_OWNER,
            "detail": f"Receipt candidate key {_safe(key)} matches no registry row.",
        })

    rows: list[ClosureRow] = []
    for key in sorted(by_key):
        row = by_key[key][0]
        chain = chains.get(key, [])
        baseline = enrolled.get(key)
        state, reason = "held", "unenrolled_registry_row"
        row_findings: list[ClosureFinding] = []
        if key in ambiguous:
            state, reason = "held", "ambiguous_candidate_match"
        elif chain:
            settled, settled_on, chain_findings = _chain_findings(
                key, chain, baseline.reviewed_ref if baseline else None, accepted_supersedes,
            )
            row_findings.extend(chain_findings)
            if settled == row["checkpoint"] and settled_on == row["reviewed_on"]:
                state, reason = "closed", SETTLED_BY_RECEIPT
            else:
                state, reason = "held", "closure_checkpoint_missing"
                row_findings.append({
                    "failure_class": "closure_checkpoint_missing", "candidate_key": key,
                    "receipt_id": str(chain[-1]["receipt_id"]), "owner": DEFAULT_OWNER,
                    "detail": (
                        f"Receipt chain settles {_safe(key)} at {_safe(settled)} on {_safe(str(settled_on))}, "
                        f"but the registry row reads {_safe(row['checkpoint'])} on {_safe(row['reviewed_on'])}: "
                        "the resolving change must advance the row in the same pull request."
                    ),
                })
            if chain_findings:
                state, reason = "held", chain_findings[0]["failure_class"]
        elif baseline is not None:
            if (row["checkpoint"], row["reviewed_on"]) == (baseline.reviewed_ref, baseline.reviewed_on):
                state, reason = "not_applicable", PRE_RECEIPT_BASELINE
            else:
                state, reason = "held", "closure_receipt_missing"
                row_findings.append({
                    "failure_class": "closure_receipt_missing", "candidate_key": key,
                    "receipt_id": None, "owner": baseline.owner,
                    "detail": (
                        f"Row {_safe(key)} moved from {_safe(baseline.reviewed_ref)} to "
                        f"{_safe(row['checkpoint'])} with no terminal receipt recording the decision."
                    ),
                })
        else:
            row_findings.append({
                "failure_class": "unenrolled_registry_row", "candidate_key": key,
                "receipt_id": None, "owner": DEFAULT_OWNER,
                "detail": (
                    f"Row {_safe(key)} carries neither a closure receipt nor a pre-receipt baseline; "
                    "enrol it in omh.catalogs.skill_source_closure or record its decision in the ledger."
                ),
            })
        findings.extend(row_findings)
        rows.append({
            "candidate_key": key, "unit": row["unit"], "source": row["source"],
            "state": state, "reason": reason, "reviewed_on": row["reviewed_on"],
            "checkpoint": row["checkpoint"],
            "receipt_ids": [str(receipt["receipt_id"]) for receipt in chain],
            "disposition": str(chain[-1]["disposition"]) if chain else None,
            "scan_from": row["checkpoint"] if state in ("closed", "not_applicable") else None,
        })

    findings.sort(key=lambda item: (item["failure_class"], item["candidate_key"] or "", item["receipt_id"] or ""))
    return {
        "schema_version": CLOSURE_SCHEMA,
        "mode": "observed_continuity",
        "observed": True,
        "ok": not findings,
        "registry": REGISTRY_PATH,
        "ledger": LEDGER_PATH,
        "rows": rows,
        "findings": findings,
        "summary": {
            "rows": len(rows),
            "closed": sum(row["state"] == "closed" for row in rows),
            "held": sum(row["state"] == "held" for row in rows),
            "not_applicable": sum(row["state"] == "not_applicable" for row in rows),
            "receipts": len(receipts),
            "finding_count": len(findings),
        },
        "bounds": {
            "max_rationale_chars": MAX_RATIONALE_CHARS,
            "max_receipts": MAX_RECEIPTS,
            "network": False,
            "subprocess": False,
            "git_resolution": False,
            "automatic_registry_edits": False,
        },
        "failure_classes": list(FAILURE_CLASSES),
        "closure_boundary": (
            "Continuity only: this binds a terminal disposition to the registry row it resolves. It never "
            "fetches a watched repository, runs a scheduled job, closes an issue, merges a pull request, or "
            "decides whether a capability should be adopted, and issue closure alone is not review evidence. "
            "A closed row proves the decision and the checkpoint moved together, not that the decision was right."
        ),
    }


def format_skill_source_closure(report: ClosureReport) -> str:
    summary = report["summary"]
    lines = ["Skill-source closure: " + ("PASS" if report["ok"] else "NEEDS ATTENTION")]
    lines.append(
        f"  rows={summary['rows']} closed={summary['closed']} held={summary['held']} "
        f"not_applicable={summary['not_applicable']} receipts={summary['receipts']}"
    )
    for row in report["rows"]:
        lines.append(f"{row['state']} {row['candidate_key']} reason={row['reason']}")
    for finding in report["findings"]:
        lines.append(
            f"{finding['failure_class']} {finding['candidate_key'] or '-'} "
            f"receipt={finding['receipt_id'] or '-'} owner={finding['owner']}"
        )
        lines.append(f"  {finding['detail']}")
    lines.append(report["closure_boundary"])
    return "\n".join(lines)

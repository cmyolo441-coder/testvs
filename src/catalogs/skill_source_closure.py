"""Vocabulary and enrolment data for the skill-source closure contract.

`docs/SKILL-SOURCES.md` is the recurring-watch registry: one row per
(OMH unit, upstream source) pair, carrying the `reviewed_ref` checkpoint the
external upstream tracker diffs against. The registry has always stated that a
resolved tracker finding must advance `reviewed_ref` and `reviewed_on` in the
same pull request that resolves it, and nothing enforced it, so a merged
capability change could leave the checkpoint behind and let a later run
re-evaluate an already-reviewed range.

This module holds the two frozen tables the validator in
`omh.maintenance.skill_source_closure` reads: the failure vocabulary, and the
pre-receipt baseline that records where every row stood when the contract
shipped. Logic lives in the validator; only data and names live here, the same
split `documentation_navigation` uses.

Nothing here reaches the network, a subprocess, or a watched repository. A
checkpoint value is an opaque identity string, never a git object this code
resolves or orders.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple


CLOSURE_SCHEMA = "skill_source_closure_audit/v1"
RECEIPT_SCHEMA = "skill_source_closure_receipt/v1"

# The ledger file is repository-relative and hand-written, like the registry.
LEDGER_PATH = "docs/skill-source-receipts.json"
REGISTRY_PATH = "docs/SKILL-SOURCES.md"

DEFAULT_OWNER = "skill-sources-maintainer"

# A terminal disposition. All three settle a finding and all three advance the
# checkpoint: the registry rule is "folded in or rejected", because a rejected
# or duplicate range was still reviewed and must not be rediscovered forever.
DISPOSITIONS = ("adopted", "rejected", "duplicate")

# Every row lands in exactly one state.
ROW_STATES = ("closed", "held", "not_applicable")

# Pass reasons. A row that passes still carries a code, so a reader never has
# to infer a verdict from the absence of a line.
SETTLED_BY_RECEIPT = "receipt_chain_settled"
PRE_RECEIPT_BASELINE = "pre_receipt_baseline"
PASS_REASONS = (SETTLED_BY_RECEIPT, PRE_RECEIPT_BASELINE)

# Failure classes. These strings are the contract: they are what a maintainer
# greps for and what CI output is read by, so they are stable and never
# reworded in place. A new condition gets a new code.
FAILURE_CLASSES = (
    # The implementation landed and the matching row did not move: the receipt
    # chain for this candidate does not terminate at the row's current values.
    "closure_checkpoint_missing",
    # The row moved off its enrolled baseline with no terminal receipt.
    "closure_receipt_missing",
    # A receipt's prior checkpoint is not what the chain actually stood at.
    "checkpoint_prior_stale",
    # The next checkpoint repeats the prior one, or returns to a checkpoint the
    # chain already left, without declaring what it supersedes.
    "checkpoint_not_superseding",
    # The candidate key matches more than one registry row.
    "ambiguous_candidate_match",
    # The candidate key matches no registry row.
    "unmatched_candidate_key",
    # Two receipts share one identity, so one decision would move two rows.
    "receipt_id_reused",
    # `supersedes` names a receipt that is not in the ledger.
    "supersede_reference_unknown",
    # `supersedes` names a receipt belonging to a different candidate: the
    # shape of one decision reaching across into an unrelated row.
    "supersede_reference_unrelated",
    # A receipt field is missing, mistyped, or outside its declared vocabulary.
    "receipt_field_invalid",
    # The rationale is longer than the bound, spans lines, or carries a link
    # instead of citing the OMH decision that records the evidence.
    "rationale_over_budget",
    # Receipts are append-only: a later entry may not sort before an earlier
    # one, and a superseded entry may not be removed.
    "ledger_order_violation",
    # A registry row carries neither a baseline nor a receipt.
    "unenrolled_registry_row",
    # A baseline entry matches no registry row.
    "stale_baseline_entry",
    # A table row could not be parsed into a candidate.
    "registry_row_unparsed",
    # The ledger file is absent or is not the declared receipt document.
    "ledger_unreadable",
    # The frozen pre-receipt census no longer hashes to its declared digest.
    # Without this the contract has a one-line bypass: editing a baseline entry
    # to a moved row's new values turns `closure_receipt_missing` into
    # `not_applicable` with no receipt written at all, and the gate goes green
    # over a checkpoint that advanced unrecorded.
    "baseline_census_modified",
)

# A rationale is a bounded sentence about the OMH decision. Watch evidence --
# what the tracker saw upstream, which commits were in range -- stays in the
# private continuity record and in the OMH issue the receipt cites.
MAX_RATIONALE_CHARS = 240
# Diagnostics quote registry and ledger text, both document-controlled.
MAX_DIAGNOSTIC_CHARS = 200
MAX_RECEIPTS = 2000


class PreReceiptBaseline(NamedTuple):
    """Where one registry row stood before closure receipts existed.

    A row still at its baseline is `not_applicable`: it predates the contract
    and no receipt is fabricated for it. The first time it moves, the move
    needs a receipt whose prior checkpoint equals `reviewed_ref` here, which is
    what makes the migration a one-way door rather than a permanent exemption.

    A baseline is the chain's origin, never a mirror of the row's current
    state. Once a candidate has receipts, its row moves and this entry does
    not: the two are meant to disagree, and the receipt chain is what spans
    the gap.
    """

    candidate_key: str
    reviewed_on: str
    reviewed_ref: str
    reason: str
    owner: str = DEFAULT_OWNER


# Captured from docs/SKILL-SOURCES.md at the commit that introduced this gate.
# This census is closed and frozen. It is not a running record of where rows
# stand; it is where they stood once, so a receipt has something to start from.
#
# `PRE_RECEIPT_CENSUS_DIGEST` is what makes that a rule rather than an
# intention. Editing an entry's checkpoint to match a row that moved would
# otherwise relabel an unrecorded advance as `not_applicable` and pass, which
# is precisely the failure this whole contract exists to stop -- and a comment
# asking contributors not to do it is the same prose-only enforcement the
# registry already had. `reason` is deliberately outside the digest so wording
# stays free to improve; identity and checkpoint values are not.
_BASELINE_REASON = "Row predates the closure-receipt contract; enrolled at its reviewed state."

# sha256 over "<candidate_key>|<reviewed_on>|<reviewed_ref>" for every entry,
# sorted. Recomputing this to match an edit defeats the freeze, so
# `tests/test_skill_source_closure.py` pins this literal independently: the
# bypass then needs two deliberate edits in two files that both say not to.
PRE_RECEIPT_CENSUS_DIGEST = "2962833b9d96de0e6174441d194b0a6ebc1d057053a7528b2f22bcafe9216686"

_PRE_RECEIPT_BASELINES: tuple[PreReceiptBaseline, ...] = (
    PreReceiptBaseline("codebase-uml@github.com/plantuml/plantuml", "2026-09-04", "b2392e6230a1782e477a45d250b7cb9a569f95da", _BASELINE_REASON),
    PreReceiptBaseline("code-review@github.com/mattpocock", "2026-09-01", "plugin 1.2.3", _BASELINE_REASON),
    PreReceiptBaseline("ai-slop-cleaner@github.com/effeilo/claude-code-frontend-skills", "2026-09-01", "3c9d5a0501ff", _BASELINE_REASON),
    PreReceiptBaseline("frontend-refactor@github.com/effeilo/claude-code-frontend-skills", "2026-09-01", "3c9d5a0501ff", _BASELINE_REASON),
    PreReceiptBaseline("frontend-refactor@github.com/pproenca/dot-skills", "2026-09-01", "cf93c57cac89", _BASELINE_REASON),
    PreReceiptBaseline("frontend-refactor@github.com/cst2989/react-tips-skill", "2026-09-01", "8c42b9e6390c", _BASELINE_REASON),
    PreReceiptBaseline("frontend-refactor@github.com/mickeyyaya/refactoring-skills", "2026-09-01", "cd0c22762849", _BASELINE_REASON),
    PreReceiptBaseline("refactor-plan@github.com/github/awesome-copilot", "2026-09-01", "5eaae7e2cde2", _BASELINE_REASON),
    PreReceiptBaseline("inference-serving@github.com/vllm-project/vllm-skills", "2026-09-01", "c99623410c15", _BASELINE_REASON),
    PreReceiptBaseline("inference-serving@github.com/orchestra-research/ai-research-skills", "2026-09-01", "773a52944ba4", _BASELINE_REASON),
    PreReceiptBaseline("agent-ops-review@github.com/nexus-labs-automation/agent-observability", "2026-09-01", "1714a4b38d7f", _BASELINE_REASON),
    PreReceiptBaseline("ops-observability-card@github.com/nexus-labs-automation/agent-observability", "2026-09-01", "1714a4b38d7f", _BASELINE_REASON),
    PreReceiptBaseline("llm-app-dev@github.com/denissergeevitch/agents-best-practices", "2026-09-07", "8ae085045bd6cddfab22c740c95dd2d764117ffc", _BASELINE_REASON),
    PreReceiptBaseline("lifecycle-growth@github.com/posthog/posthog", "2026-09-09", "ae880d309f33eaf236cb4e46991f249a88e1c16e", _BASELINE_REASON),
    PreReceiptBaseline("lifecycle-growth@github.com/growthbook/growthbook", "2026-09-09", "095f61643e148f03ce0b442c78e6030c045f6d7b", _BASELINE_REASON),
    PreReceiptBaseline("lifecycle-growth@github.com/dittofeed/dittofeed", "2026-09-07", "52b2bee909744d07dd5d409fd3974d4b95c66766", _BASELINE_REASON),
    PreReceiptBaseline("lifecycle-growth@github.com/novuhq/novu", "2026-09-08", "c7bc772fc0b7722909ef1bdb9bcf04991996fdd8", _BASELINE_REASON),
    PreReceiptBaseline("product-discovery-validation@gitlab.com/gitlab-com/content-sites/handbook", "2026-09-07", "4165803c1cf9adeae2826f1af918668be5c942f6", _BASELINE_REASON),
    PreReceiptBaseline("product-discovery-validation@github.com/haabe/mycelium", "2026-09-09", "8e958f82141e6cd38e0cc8046e8eb9f21df777b0", _BASELINE_REASON),
    PreReceiptBaseline("product-discovery-validation@github.com/shinpr/claude-code-discover", "2026-09-07", "a414fc7a978bb2deaec5c71dd399cf63708378ee", _BASELINE_REASON),
    PreReceiptBaseline("product-discovery-validation@github.com/lenar-amirov/product-pipeline-public", "2026-09-07", "a0997741f4c9b68ba298796c83197a4ab66ba3d9", _BASELINE_REASON),
    PreReceiptBaseline("product-discovery-validation@github.com/phuryn/pm-skills", "2026-09-07", "18468a95b427e70e258b51389796367c6f684e7d", _BASELINE_REASON),
    PreReceiptBaseline("sales-pipeline-review@gitlab.com/gitlab-com/content-sites/handbook", "2026-09-07", "4165803c1cf9adeae2826f1af918668be5c942f6", _BASELINE_REASON),
    PreReceiptBaseline("sales-pipeline-review@github.com/odoo/odoo", "2026-09-07", "1a13ceeaee12fe5cc50f287c31f217d4be2a2eaf", _BASELINE_REASON),
    PreReceiptBaseline("sales-pipeline-review@github.com/frappe/erpnext", "2026-09-07", "72fa7d0b1091e1a66450ebb4dc9fb6de1c8d3c1a", _BASELINE_REASON),
    PreReceiptBaseline("sales-pipeline-review@github.com/twentyhq/twenty", "2026-09-07", "c8fc76650231ca3640276f75f071670b20e51e75", _BASELINE_REASON),
    PreReceiptBaseline("award-bar-score@www.cssdesignawards.com", "2026-09-03", "—", _BASELINE_REASON),
    PreReceiptBaseline("tech-debt-audit@github.com/ksimback/tech-debt-skill", "2026-09-02", "5a15c1ca4a92", _BASELINE_REASON),
    PreReceiptBaseline("strategy-brief@github.com/wshobson/agents", "2026-09-02", "a30778f8c4e6", _BASELINE_REASON),
    PreReceiptBaseline("accessibility-audit@github.com/effeilo/claude-code-frontend-skills", "2026-09-02", "3c9d5a0501ff", _BASELINE_REASON),
    PreReceiptBaseline("agent-evaluation@github.com/github/awesome-copilot", "2026-09-02", "6a8fa297b0fe", _BASELINE_REASON),
    PreReceiptBaseline("frontend@github.com/rohitg00/awesome-claude-code-toolkit", "2026-09-02", "ebdf1d596d2c", _BASELINE_REASON),
    PreReceiptBaseline("apple-design@github.com/dickwu/apple-design-skill", "2026-09-05", "d0bac1e765a27a696839e62962e36330ce72f0b7", _BASELINE_REASON),
    PreReceiptBaseline("apple-design@www.apple.com/macbook-pro", "2026-09-05", "—", _BASELINE_REASON),
    PreReceiptBaseline("apple-design@developer.apple.com/icon-composer", "2026-09-05", "—", _BASELINE_REASON),
    PreReceiptBaseline("apple-design@github.com/greensock/gsap", "2026-09-05", "13e2b790546426a1a2e0e9b409f3f8dc6d6611f2", _BASELINE_REASON),
    PreReceiptBaseline("apple-design@github.com/paper-design/liquid-logo", "2026-09-05", "689bb38a1e0d5a6a8baf2d34847635eefde19994", _BASELINE_REASON),
    PreReceiptBaseline("apple-design@github.com/dashersw/liquid-glass-js", "2026-09-05", "78cb6ccb0b9987bb60a88b14ccbd13a9e6e8ab2a", _BASELINE_REASON),
)


def pre_receipt_baselines() -> tuple[PreReceiptBaseline, ...]:
    """Rows enrolled at their pre-contract reviewed state."""
    return _PRE_RECEIPT_BASELINES


def census_digest(baselines: tuple[PreReceiptBaseline, ...]) -> str:
    """Hash the identity and checkpoint of every enrolled row.

    Order-independent, so reordering the table for readability is free while
    changing any candidate key, review date, or checkpoint is not.
    """
    lines = sorted(f"{entry.candidate_key}|{entry.reviewed_on}|{entry.reviewed_ref}" for entry in baselines)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def pre_receipt_census_digest() -> str:
    """The digest the shipped census currently hashes to."""
    return census_digest(_PRE_RECEIPT_BASELINES)

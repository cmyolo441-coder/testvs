from __future__ import annotations

import argparse

from . import domain_intelligence


def add_domain_intelligence_commands(
    memory_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    domain_status = memory_sub.add_parser(
        "domain-status",
        help="Show reviewed domain-intelligence store counts and fail-closed diagnostics.",
    )
    domain_status.set_defaults(func=domain_intelligence.cmd_memory_domain_status)

    domain_capture = memory_sub.add_parser(
        "domain-capture",
        help="Capture an agent/operator supplied domain vocabulary candidate for manual review.",
    )
    domain_capture.add_argument("--from-file", default=None, help="Preview repository-root PROJECT_TERMS.md for pending project capture.")
    domain_capture.add_argument("--stage", action="store_true", help="Stage every file domain as pending review after atomic preflight.")
    domain_capture.add_argument("--json", action="store_true", help="Print the machine-readable capture payload (the command default).")
    domain_capture.add_argument("--scope-kind", choices=("user", "organization", "project"), default=None)
    domain_capture.add_argument("--scope-ref", default=None, help="Explicit opaque operator/wrapper supplied scope key.")
    domain_capture.add_argument("--domain", default=None, help="Normalized domain identifier, e.g. sales or payments.")
    domain_capture.add_argument(
        "--mapping",
        action="append",
        default=None,
        metavar="PHRASE=CANONICAL_TERM",
        help="Bounded phrase-to-canonical-term mapping; repeat for multiple mappings.",
    )
    domain_capture.add_argument(
        "--workflow-hint",
        action="append",
        default=None,
        help="Optional identifier-like workflow hint; repeatable.",
    )
    domain_capture.add_argument(
        "--source-class",
        choices=("operator_supplied", "wrapper_supplied", "omh_local"),
        default=None,
    )
    domain_capture.add_argument(
        "--source-ref",
        default=None,
        help="Optional safe opaque source reference: letters, digits, _ . : - only.",
    )
    domain_capture.add_argument("--observation-count", type=int, default=None)
    domain_capture.add_argument("--confidence", type=float, default=None)
    domain_capture.set_defaults(func=domain_intelligence.cmd_memory_domain_capture)

    domain_review = memory_sub.add_parser("domain-review", help="Return review cards for pending domain vocabulary candidates.")
    domain_review.add_argument("--candidate", default=None, help="Limit review output to one candidate id.")
    domain_review.add_argument("--limit", type=int, default=20)
    domain_review.add_argument(
        "--source-freshness",
        action="store_true",
        help="Derive read-only repository-root PROJECT_TERMS.md freshness (requires --scope project).",
    )
    domain_review.set_defaults(func=domain_intelligence.cmd_memory_domain_review)

    domain_approve = memory_sub.add_parser("domain-approve", help="Manually approve one pending domain vocabulary candidate.")
    domain_approve.add_argument("candidate_id")
    domain_approve.add_argument(
        "--approved-by",
        default="operator",
        help="Safe opaque reviewer claim: letters, digits, _ . : - only.",
    )
    domain_approve.set_defaults(func=domain_intelligence.cmd_memory_domain_approve)

    domain_reject = memory_sub.add_parser("domain-reject", help="Reject one pending domain vocabulary candidate.")
    domain_reject.add_argument("candidate_id")
    domain_reject.add_argument(
        "--rejected-by",
        default="operator",
        help="Safe opaque reviewer claim: letters, digits, _ . : - only.",
    )
    domain_reject.add_argument(
        "--reason",
        default="",
        help="Metadata-only reason code: duplicate, incorrect_scope, insufficient_evidence, operator_request, scope_error, or superseded.",
    )
    domain_reject.set_defaults(func=domain_intelligence.cmd_memory_domain_reject)

    domain_list = memory_sub.add_parser("domain-list", help="List reviewed active domain vocabulary profiles.")
    domain_list.add_argument("--scope-kind", choices=("user", "organization", "project"), default=None)
    domain_list.add_argument("--scope-ref", default=None)
    domain_list.add_argument("--domain", default=None)
    domain_list.add_argument("--include-retired", action="store_true")
    domain_list.add_argument(
        "--source-freshness",
        action="store_true",
        help="Derive read-only repository-root PROJECT_TERMS.md freshness (requires --scope project).",
    )
    domain_list.set_defaults(func=domain_intelligence.cmd_memory_domain_list)

    domain_retire = memory_sub.add_parser("domain-retire", help="Retire one active domain vocabulary profile without deleting history.")
    domain_retire.add_argument("--scope-kind", choices=("user", "organization", "project"), required=True)
    domain_retire.add_argument("--scope-ref", required=True)
    domain_retire.add_argument("--domain", required=True)
    domain_retire.add_argument(
        "--retired-by",
        default="operator",
        help="Safe opaque reviewer claim: letters, digits, _ . : - only.",
    )
    domain_retire.add_argument(
        "--reason",
        default="",
        help="Metadata-only reason code: duplicate, incorrect_scope, insufficient_evidence, operator_request, scope_error, or superseded.",
    )
    domain_retire.set_defaults(func=domain_intelligence.cmd_memory_domain_retire)

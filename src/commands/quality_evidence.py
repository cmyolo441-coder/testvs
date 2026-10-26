"""Agent-facing preparation and assessment for source-bound quality evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from ..coding.executors import canonical_executor_profile
from ..installer import OmhError
from ..plugin_bundle.omh.cost_receipt import build_cost_receipt
from ..quality.evidence_records import assess_quality_evidence, build_quality_evidence_package
from ..quality.working_tree_fingerprint import working_tree_content_fingerprint
from ..quality.language_diagnostic_evidence import (
    LANGUAGE_DIAGNOSTIC_CHECK_STATES,
    LANGUAGE_DIAGNOSTIC_OWNERS,
    LanguageDiagnosticEvidenceError,
    build_language_diagnostic_evidence,
    language_diagnostic_claim_support,
)
from ..quality.reply_lint import build_reply_lint, format_reply_lint_summary, summarize_reply_lints
from ..quality.agent_debug_incident import (
    OBSERVABLES,
    AgentDebugIncidentError,
    agent_debug_export_target_problem,
    build_agent_debug_export,
    build_agent_debug_incident,
    format_agent_debug_incident,
    read_receipt,
    write_agent_debug_export,
)
from ..quality.agent_debug_report import (
    DEFAULT_MAX_ROW_BYTES,
    DEFAULT_MAX_ROWS,
    AgentDebugReportError,
    build_agent_debug_report,
    format_agent_debug_report,
    parse_turn_range,
)
from ..quality.hermes_state import HERMES_LATEST_SESSION, NO_SOURCE_LABEL
from ..quality.reply_lint_source import ReplySourceError, hermes_session_replies
from ..quality.session_file_activity import (
    DEFAULT_MAX_FILES,
    SessionFileActivityError,
    build_session_file_activity,
    format_session_file_activity_summary,
)
from ..quality.memory_evidence import (
    DEFAULT_WINDOW_DAYS as MEMORY_EVIDENCE_DEFAULT_DAYS,
    MemoryEvidenceError,
    build_memory_evidence,
    format_memory_evidence_summary,
)
from ..quality.session_usage import SessionUsageError, build_session_usage, format_session_usage_summary
from .common import _paths, _print_json, _wants_json


def cmd_quality_evidence_prepare(args: argparse.Namespace) -> int:
    """Prepare QA, review, and claim requirements without executing them."""
    try:
        package = build_quality_evidence_package(
            repository_id=args.repository,
            commit_sha=args.commit,
            tree_sha=args.tree,
            title=args.title,
            executor_target=args.executor,
            scenarios=_json_list(args.scenarios_json, args.scenarios_file, "scenarios"),
            review_requirements=_json_list(args.reviews_json, args.reviews_file, "review requirements"),
            claim_requirements=_json_list(args.claims_json, args.claims_file, "claim requirements"),
            self_critique_questions=_json_strings(args.self_critique_json, args.self_critique_file, "self-critique questions"),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(package)
    return 0


def cmd_quality_evidence_assess(args: argparse.Namespace) -> int:
    """Assess a package and optional observations; never execute external work."""
    try:
        package = _json_object(args.package)
        observations = _json_list(args.observations_json, args.observations_file, "observations")
        assessment = assess_quality_evidence(
            package,
            observations,
            omh_home=_paths(args).omh_home,
            current_fingerprint=working_tree_content_fingerprint(),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(assessment)
    return 0


def cmd_quality_evidence_language_diagnostics(args: argparse.Namespace) -> int:
    """Render one supplied language-diagnostic observation as a scoped record.

    Read-only in both directions: OMH starts no language server and writes no
    file. The diagnostics come from a caller that ran the provider, and the
    verdict is derived from them rather than accepted from them.
    """
    try:
        record = build_language_diagnostic_evidence(
            owner=args.owner,
            provider=args.provider,
            workspace_id=args.workspace,
            baseline_revision=args.baseline_revision,
            end_revision=args.end_revision,
            diagnostics_revision=args.diagnostics_revision,
            check_state=args.check_state,
            config_digest=args.config_digest,
            changed_paths=_json_strings(args.changed_paths_json, args.changed_paths_file, "changed paths"),
            introduced=_json_list(args.introduced_json, args.introduced_file, "introduced diagnostics"),
            resolved=_json_list(args.resolved_json, args.resolved_file, "resolved diagnostics"),
            observed_at=args.observed_at,
            evidence_refs=_json_strings(args.evidence_refs_json, None, "evidence refs"),
        )
    except (OSError, json.JSONDecodeError, TypeError, LanguageDiagnosticEvidenceError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    support = language_diagnostic_claim_support(record)
    if _wants_json(args):
        _print_json({"record": record, "claim_support": support})
    else:
        _print_language_diagnostic_summary(record, support)
    return 0


def _print_language_diagnostic_summary(record: Mapping[str, Any], support: Mapping[str, Any]) -> None:
    print("OMH language diagnostic evidence")
    print(f"  Verdict: {record['verdict']}")
    print(f"  {record['summary_label']}")
    print("Check")
    print(f"  Owner: {record['owner']}    Provider: {record['provider']}    State: {record['check_state']}")
    print(f"  Workspace: {record['workspace_id'] or '(not stated)'}")
    print(
        f"  Interval: {record['baseline_revision'] or '(not stated)'}"
        f" -> {record['end_revision'] or '(not stated)'}"
        f"    Diagnostics at: {record['diagnostics_revision'] or '(not stated)'}"
    )
    print(f"  Freshness: {record['freshness']}    Attribution: {record['attribution']}")
    print(
        f"  Introduced: {record['introduced_count']}"
        f"    Resolved: {record['resolved_count']}"
        f"    Changed paths: {record['changed_path_count']}"
    )
    print("Boundary")
    supported = support.get("supported_claims") or []
    unsupported = support.get("unsupported_claims") or []
    print(f"  Supports: {', '.join(str(item) for item in supported) or 'nothing'}")
    print(f"  Does not support: {', '.join(str(item) for item in unsupported)}")
    print(f"  {record['claim_boundary']}")


def cmd_quality_evidence_reply_lint(args: argparse.Namespace) -> int:
    """Lint replies a person read against OMH's reply rules; reads text only.

    The rules ship as prompt text and nothing observed whether a reply
    followed them. This reads a reply (a file, stdin, or the trailing replies
    of a Hermes session, read-only) and reports leaked record terms, quoted
    awareness lines, and closings that declare what will not be done or leave
    a decision without a question. A finding exits 1 so a QA loop can gate on
    it; the payload's claim boundary says what a clean result does not show.
    """
    try:
        source, pairs = _reply_lint_input(args)
    except (OSError, ReplySourceError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    records = [build_reply_lint(pair["reply"], user_text=pair["user_text"]) for pair in pairs]
    payload = summarize_reply_lints(records, source=source)
    if _wants_json(args):
        _print_json(payload)
    else:
        print(format_reply_lint_summary(payload))
    return 0 if payload["ok"] else 1


def cmd_quality_evidence_session_usage(args: argparse.Namespace) -> int:
    """Report OMH utilization per Hermes host surface; reads state.db only.

    OMH's tools and skills reach a Hermes session through whichever surface
    opened it, and nothing observed which surfaces they reached. This reads
    Hermes' own session store read-only and counts, per ``sessions.source``,
    sessions, tool calls, ``omh_*`` calls and OMH skill loads. An empty window
    exits 0: it is an observation, and a wrapper that wants to gate on
    utilization reads ``totals``. A missing or unreadable database is an error.
    """
    try:
        payload = build_session_usage(_paths(args).hermes_home, since=args.since, source=args.source)
    except (OSError, SessionUsageError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(payload)
    else:
        print(format_session_usage_summary(payload))
    return 0


def cmd_quality_evidence_memory(args: argparse.Namespace) -> int:
    """Report what OMH memory did over a window, from structured local records only.

    The only memory quality surface ran synthetic fixtures, so nothing said
    whether memory was working on this machine. This counts ``omh_memory``
    and native ``memory`` calls from Hermes' state.db (mode=ro), the OMH
    store's records and journals, and native memory headroom. Exit 0 when
    every source was read, even when every count is zero; exit 3 when a source
    could not be read (see ``_memory_evidence_exit_code``). The payload is
    printed either way, so the sources that were read still report.
    """
    try:
        days = MEMORY_EVIDENCE_DEFAULT_DAYS if args.days is None else args.days
        payload = build_memory_evidence(_paths(args), days=days, since=args.since)
    except MemoryEvidenceError as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(payload)
    else:
        print(format_memory_evidence_summary(payload))
    return _memory_evidence_exit_code(payload)


def _memory_evidence_exit_code(payload: Any) -> int:
    """0 only when every source was read; a zero count read from a source is still 0.

    3 when a source could not be read -- a missing or unreadable state.db, an
    unreadable store or journal, an unreadable memory file. Not 0, because a
    count from a source nobody read is not a zero; not 2, which ``main``
    returns for an OmhError (a malformed ``--days`` or ``--since``); not 1, because nothing
    the command was asked to do failed -- it reported what it could see.

    Generic failure signals -- a refused or interrupted run, or a unit
    carrying a failure kind -- are never success either.
    """
    summary = payload if isinstance(payload, dict) else {}
    if summary.get("refused") or summary.get("interrupted"):
        return 1
    units = summary.get("units")
    if isinstance(units, list) and any(isinstance(unit, dict) and unit.get("failure_kind") for unit in units):
        return 1
    if summary.get("unread_sources"):
        return 3
    return 0


def cmd_quality_evidence_file_activity(args: argparse.Namespace) -> int:
    """Report which workspace files one Hermes session's file tools touched; reads state.db only.

    Tool-call counts cannot say which files a run read, wrote or patched.
    This derives, at query time and without any OMH store, a bounded
    per-file projection of one session's ``read_file``, ``write_file`` and
    ``patch`` calls, each with the outcome Hermes recorded for it. Failed or
    unknown calls are observations, so the command exits 0 when it read the
    session; a missing database, an unknown session or a bad argument is an
    error.
    """
    try:
        payload = build_session_file_activity(
            _paths(args).hermes_home,
            args.hermes_session,
            workspace=args.workspace,
            max_files=int(args.max_files),
        )
    except (OSError, SessionFileActivityError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(payload)
    else:
        print(format_session_file_activity_summary(payload))
    return 0


def cmd_quality_evidence_agent_debug(args: argparse.Namespace) -> int:
    """Cite what one Hermes session recorded going wrong; reads state.db only.

    The agent-debug skill declared ``agent_debug_report/v1`` and nothing
    produced one, so a plausible narrative read the same as a diagnosis. This
    reads one session read-only -- from state.db or a supplied session record
    -- and returns findings derived from record fields, each cited by session,
    message (or record line) and tool-call id, plus the incident built on
    them: the capture, competing hypotheses, and a proposed recovery that is
    never executed. A valid report exits 0 whether or not it has findings: a
    finding is an observation to cite, not failed work, and a wrapper that
    wants to gate reads ``finding_counts``. A missing or ambiguous session, an
    unreadable source, or an artifact that fails validation is an error.
    Nothing is written; sharing is the separate ``agent-debug-export``.
    """
    try:
        report = build_agent_debug_report(
            _paths(args).hermes_home,
            args.hermes_session,
            session_record=args.session_record,
            turns=parse_turn_range(args.turns),
            max_rows=int(args.max_rows),
            max_row_bytes=int(args.max_row_bytes),
        )
        incident = build_agent_debug_incident(
            report,
            observable=args.observable,
            receipts=[read_receipt(path) for path in args.receipt or ()],
        )
    except (OSError, AgentDebugReportError, AgentDebugIncidentError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json({**report, "incident": incident})
    else:
        print(format_agent_debug_report(report))
        print(format_agent_debug_incident(incident))
    return 0


def cmd_quality_evidence_agent_debug_export(args: argparse.Namespace) -> int:
    """Prepare, and only with ``--confirm-export`` write, a redacted agent-debug package.

    Export is its own action, never a side effect of diagnosis. It re-checks
    the reviewed payload and every reference it cites against the source,
    drops the absolute source path, and runs a leak scan. Without
    ``--confirm-export`` it prints the package for review, says whether
    ``--output`` could take it (it already exists, or lies inside the Hermes
    home), and writes nothing; with it, the package goes to a new file outside
    the source. Nothing is
    uploaded, filed, or posted. A refused export exits 2.
    """
    paths = _paths(args)
    try:
        payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("the reviewed payload must be a JSON object from agent-debug --json")
        package = build_agent_debug_export(payload, hermes_home=paths.hermes_home, session_record=args.session_record)
        written = (
            write_agent_debug_export(
                package, args.output, hermes_home=paths.hermes_home, session_record=args.session_record
            )
            if args.confirm_export
            else None
        )
        output_problem = (
            None
            if written is not None
            else agent_debug_export_target_problem(
                args.output, hermes_home=paths.hermes_home, session_record=args.session_record
            )
        )
    except (OSError, AgentDebugReportError, AgentDebugIncidentError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(
            {
                "written": written is not None,
                "output": None if written is None else written.name,
                "output_problem": output_problem,
                "package": package,
            }
        )
    elif written is None:
        print(json.dumps(package, indent=2, sort_keys=True))
        if output_problem is not None:
            print(f"--output cannot take this package: {output_problem}. Choose another --output before confirming.")
        print("Not written: review the package above, then re-run with --confirm-export to write it to --output.")
    else:
        print(f"Wrote the reviewed agent-debug package to {written.name}. Nothing was uploaded, filed, or posted.")
    return 0


def cmd_quality_evidence_cost_receipt(args: argparse.Namespace) -> int:
    """Print what one conversation's work cost, from records only.

    The same receipt the ``omh_run_summary`` tool hands a chat: the session
    and its compression continuations, delegated Hermes children, and fanout
    units stamped with the session. A session Hermes has no row for is an
    error rather than an empty receipt, so a wrapper never reads "no record"
    as "cost nothing".
    """
    paths = _paths(args)
    receipt = build_cost_receipt(hermes_home=paths.hermes_home, omh_home=paths.omh_home, session_id=args.session)
    if receipt.get("status") != "observed":
        raise OmhError(str(receipt.get("reason") or "no cost receipt"))
    if _wants_json(args):
        _print_json(receipt)
    else:
        print(receipt["text"])
    return 0


def cmd_quality_evidence_agent_skills(args: argparse.Namespace) -> int:
    """Print the Agent Skills host evidence matrix: five stages per host and scope.

    OMH reads each host's scan path and compares it with the pack inventory.
    It starts no host session and makes no model call; the only host process
    is the approved ``<cli> --version`` readiness probe. So only ``installed``
    can be observed. The payload is printed either way; the exit status is
    ``_agent_skills_evidence_exit_code``. A receipts file that fails
    validation is an error, never an empty matrix.
    """
    from ..quality.agent_skills_discovery import (
        SCOPES,
        build_agent_skills_evidence,
        format_agent_skills_evidence,
        load_evidence_receipts,
    )

    if args.receipts and (args.project_root or args.user_home):
        raise OmhError("--receipts reads no scan path; drop --project-root and --user-home")
    try:
        if args.receipts:
            payload = build_agent_skills_evidence(
                scopes=(args.scope,) if args.scope else SCOPES,
                stored_receipts=load_evidence_receipts(Path(args.receipts).expanduser()),
            )
        else:
            payload = build_agent_skills_evidence(
                project_root=Path(args.project_root or ".").expanduser().resolve(),
                user_home=Path(args.user_home).expanduser().resolve() if args.user_home else Path.home().resolve(),
                scopes=(args.scope,) if args.scope else SCOPES,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise OmhError(f"Agent Skills host evidence failed: {exc}") from exc
    if _wants_json(args):
        _print_json(payload)
    else:
        print(format_agent_skills_evidence(payload))
    return _agent_skills_evidence_exit_code(payload)


def _agent_skills_evidence_exit_code(payload: Any) -> int:
    """0 when every scan path this run had to read was read, whatever it found there.

    3 when a scan path could not be read (its row is ``hold`` / ``unreadable``):
    not 0, because "not installed" read from a path nobody could read is not
    an observation; not 2, which ``main`` returns for an OmhError (a malformed
    receipts file); not 1, because nothing the command was asked to do failed
    -- it reported what it could see. An absent, partial, modified, or
    symlinked pack is an observation and exits 0.

    Generic failure signals -- a refused or interrupted run, or a unit
    carrying a failure kind -- are never success either.
    """
    summary = payload if isinstance(payload, dict) else {}
    if summary.get("refused") or summary.get("interrupted"):
        return 1
    units = summary.get("units")
    if isinstance(units, list) and any(isinstance(unit, dict) and unit.get("failure_kind") for unit in units):
        return 1
    if summary.get("unread_scan_paths"):
        return 3
    return 0


def _reply_lint_input(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.hermes_session:
        paths = _paths(args)
        read = hermes_session_replies(
            paths.hermes_home, args.hermes_session, last=int(args.last), source=args.source
        )
        source = {
            "kind": "hermes_session",
            "session_id": read["session_id"],
            "last": int(args.last),
            "source_filter": args.source,
        }
        return source, [
            {"user_text": item["user_text"], "reply": item["reply"], "message_id": item["message_id"]}
            for item in read["replies"]
        ]
    if args.source is not None:
        raise ValueError("--source applies only to --hermes-session")
    user_text = Path(args.user_text_file).read_text(encoding="utf-8") if args.user_text_file else ""
    if args.stdin:
        return {"kind": "stdin"}, [{"user_text": user_text, "reply": sys.stdin.read()}]
    reply = Path(args.text_file).read_text(encoding="utf-8")
    return {"kind": "text_file", "path": str(args.text_file)}, [{"user_text": user_text, "reply": reply}]


def _add_quality_evidence_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the operator-only quality evidence control-plane commands."""
    quality = sub.add_parser(
        "quality-evidence",
        help="Prepare or assess source-bound quality evidence (operator/backend; no execution).",
        description=(
            "Operator/backend commands for prepared quality evidence. Preparation and assessment "
            "do not run tests, review, CI, merge, or any runtime dispatch."
        ),
    )
    commands = quality.add_subparsers(dest="quality_evidence_command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="Create a prepared_not_observed package from source and JSON requirements.",
        description="Create requirements only; this does not execute tests or perform review/CI.",
    )
    prepare.add_argument("--repository", "--repository-id", dest="repository", required=True)
    prepare.add_argument("--commit", "--commit-sha", dest="commit", required=True)
    prepare.add_argument("--tree", "--tree-sha", dest="tree", required=True)
    prepare.add_argument("--title", required=True)
    prepare.add_argument("--executor", "--executor-target", type=canonical_executor_profile, dest="executor", required=True)
    _add_json_input(prepare, "scenarios", "QA scenarios")
    _add_json_input(prepare, "reviews", "review requirements")
    _add_json_input(prepare, "claims", "claim requirements")
    prepare.add_argument("--self-critique-json", "--self-critique", dest="self_critique_json", help="Inline self-critique question JSON array.")
    prepare.add_argument("--self-critique-file", dest="self_critique_file", help="Path to self-critique question JSON array.")
    prepare.set_defaults(func=cmd_quality_evidence_prepare)

    assess = commands.add_parser(
        "assess",
        help="Assess a package plus optional observations without claiming execution.",
        description="Assessment checks source-bound evidence consistency; it does not run tests or merge.",
    )
    assess.add_argument("--package", required=True, help="Path to a prepared quality evidence package JSON.")
    assess.add_argument("--observations-json", "--observations", dest="observations_json", help="Inline observations JSON.")
    assess.add_argument("--observations-file", help="Path to observations JSON.")
    assess.set_defaults(func=cmd_quality_evidence_assess)

    language = commands.add_parser(
        "language-diagnostics",
        help="Scope a supplied language-server diagnostic delta as a language-diagnostic check.",
        description=(
            "Render supplied diagnostics as a language_diagnostic_evidence/v1 record. OMH starts no "
            "language server and observes nothing; a clean result is labelled only as a fresh "
            "language-diagnostic check and never as verification, tests, review, CI, or merge evidence."
        ),
    )
    language.add_argument("--owner", type=canonical_executor_profile, required=True, choices=LANGUAGE_DIAGNOSTIC_OWNERS, help="Surface that observed the check.")
    language.add_argument("--provider", required=True, help="Diagnostic provider label, such as a language-server name.")
    language.add_argument("--workspace", default="", help="Workspace identifier the interval is attributed to.")
    language.add_argument("--baseline-revision", default="", help="Revision the interval starts at.")
    language.add_argument("--end-revision", default="", help="Revision the interval ends at.")
    language.add_argument("--diagnostics-revision", default="", help="Revision the diagnostics were observed at.")
    language.add_argument("--check-state", default="observed", choices=LANGUAGE_DIAGNOSTIC_CHECK_STATES)
    language.add_argument("--config-digest", default="", help="Digest of the provider configuration in effect.")
    language.add_argument("--observed-at", default="", help="Caller-supplied observation timestamp.")
    _add_json_input(language, "changed-paths", "changed workspace-relative paths")
    _add_json_input(language, "introduced", "introduced diagnostics")
    _add_json_input(language, "resolved", "resolved diagnostics")
    language.add_argument("--evidence-refs-json", "--evidence-refs", dest="evidence_refs_json", help="Inline bounded evidence reference JSON array.")
    language.add_argument("--json", action="store_true", help="Print the machine-readable record and claim support.")
    language.set_defaults(func=cmd_quality_evidence_language_diagnostics)

    reply_lint = commands.add_parser(
        "reply-lint",
        help="Lint a reply a person read for leaked OMH record terms and refusal closers.",
        description=(
            "Read one reply (a file, stdin, or the trailing replies of a Hermes session, read-only) "
            "and report OMH record terms in the user's sentence, quoted awareness lines, and closings "
            "that declare what will not be done or leave a decision without a question. Findings exit 1. "
            "A clean result is not evidence that the reply was correct or in the host's voice."
        ),
    )
    reply_source = reply_lint.add_mutually_exclusive_group(required=True)
    reply_source.add_argument("--text-file", help="Path to the reply text.")
    reply_source.add_argument("--stdin", action="store_true", help="Read the reply text from stdin.")
    reply_source.add_argument(
        "--hermes-session",
        help=f"Hermes session id, or `{HERMES_LATEST_SESSION}` for the most recently active session.",
    )
    reply_lint.add_argument("--last", type=int, default=1, help="How many trailing replies of the session to lint.")
    reply_lint.add_argument(
        "--source",
        default=None,
        help=(
            "Only consider Hermes sessions whose source tag equals this value (tui, cli, desktop, ...; "
            f"`{NO_SOURCE_LABEL}` for untagged sessions); with `{HERMES_LATEST_SESSION}`, the most recent such session."
        ),
    )
    reply_lint.add_argument(
        "--user-text-file",
        help="Path to the user message the reply answers; a term it names is explained, not leaked.",
    )
    reply_lint.add_argument("--json", action="store_true", help="Print the machine-readable reply_lint/v1 payload.")
    reply_lint.set_defaults(func=cmd_quality_evidence_reply_lint)

    usage = commands.add_parser(
        "session-usage",
        help="Report OMH utilization per Hermes host surface from state.db, read-only.",
        description=(
            "Read Hermes' own session store (mode=ro) and count, per sessions.source (tui, cli, "
            "desktop, oneshot, ...), sessions, tool calls, omh_* tool calls, sessions with at least "
            "one omh_* call, skill_view loads and OMH skill loads. It observes nothing about whether "
            "a call succeeded or a skill was followed, and is not execution, review, CI, or merge evidence."
        ),
    )
    usage.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp or epoch seconds; sessions whose last activity is older are excluded.",
    )
    usage.add_argument(
        "--source",
        default=None,
        help=(
            "Only sessions whose Hermes source tag equals this value, such as tui, cli, desktop, oneshot; "
            f"`{NO_SOURCE_LABEL}` keeps the untagged ones."
        ),
    )
    usage.add_argument("--json", action="store_true", help="Print the machine-readable session_usage/v1 payload.")
    usage.set_defaults(func=cmd_quality_evidence_session_usage)

    memory = commands.add_parser(
        "memory",
        help="Report what OMH memory did on this machine over a window, from local records, read-only.",
        description=(
            "Count, over the last --days days (or since --since), omh_memory calls per action and result status and Hermes-native "
            "memory calls per action from Hermes' state.db (mode=ro); OMH store records per source and admission "
            "state, the last prefetch receipt, consolidation briefs per reason family and the native write journal; "
            "and Hermes memory headroom per file. Counts come from structured fields only, never message text. "
            "Pass the global --hermes-home to read a Hermes profile. Exit 3 names a source that could not be read. "
            "Counts are local records, not evidence that recall improved an answer."
        ),
    )
    memory_window = memory.add_mutually_exclusive_group()
    memory_window.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Window length in days, ending now (default {MEMORY_EVIDENCE_DEFAULT_DAYS}).",
    )
    memory_window.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp or epoch seconds; records and calls older than this are excluded.",
    )
    memory.add_argument("--json", action="store_true", help="Print the machine-readable memory_evidence/v1 payload.")
    memory.set_defaults(func=cmd_quality_evidence_memory)

    activity = commands.add_parser(
        "file-activity",
        help="Report which workspace files one Hermes session read, wrote, or patched, from state.db, read-only.",
        description=(
            "Read one session's read_file, write_file and patch calls from Hermes' own session store "
            "(mode=ro) and list the workspace files they named, each with its operation, the outcome "
            "Hermes recorded (succeeded, failed or unknown) and first/last time. Paths are "
            "workspace-relative; a path outside the workspace is counted, never shown. No file is read, "
            "and the result is not file-content, diff, test, review, CI, or merge evidence."
        ),
    )
    activity.add_argument(
        "--hermes-session",
        required=True,
        help=f"Hermes session id, or `{HERMES_LATEST_SESSION}` for the most recently active session.",
    )
    activity.add_argument(
        "--workspace",
        default=None,
        help="Workspace root the paths are shown against; default: the session's git repo root, else its cwd.",
    )
    activity.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"Most files to list (default {DEFAULT_MAX_FILES}); the payload reports truncation.",
    )
    activity.add_argument("--json", action="store_true", help="Print the machine-readable session_file_activity/v1 payload.")
    activity.set_defaults(func=cmd_quality_evidence_file_activity)

    agent_debug = commands.add_parser(
        "agent-debug",
        help="Cite tool errors, identical retries, unnotified background processes and compactions in one Hermes session.",
        description=(
            "Read one Hermes session from state.db (mode=ro) and report agent_debug_report/v1: tool results "
            "whose typed fields record an error, the same tool retried with identical arguments after an "
            "error, background processes started without notify_on_complete, and compaction boundaries. "
            "Each finding cites session, message and tool-call ids, tool name, error class and timestamps; "
            "no prompt or tool output is quoted. A finding is an observed record, not a diagnosis, and not "
            "execution, review, CI, or merge evidence."
        ),
    )
    agent_debug.add_argument(
        "--hermes-session",
        help=(
            f"Hermes session id, `{HERMES_LATEST_SESSION}`, or a prefix naming exactly one session; a prefix "
            "naming more than one is refused. Optional with --session-record when it holds one session."
        ),
    )
    agent_debug.add_argument(
        "--session-record",
        help="Read this JSON Lines session record (one message object per line) instead of state.db; cites record:line.",
    )
    agent_debug.add_argument("--turns", help="Turn range N, N:M, or N: (1-based, counted over user turns).")
    agent_debug.add_argument(
        "--max-rows", type=int, default=DEFAULT_MAX_ROWS, help=f"Most rows or lines to read (default {DEFAULT_MAX_ROWS})."
    )
    agent_debug.add_argument(
        "--max-row-bytes",
        type=int,
        default=DEFAULT_MAX_ROW_BYTES,
        help=f"Most bytes of one row or line to read; a longer one is listed and not checked (default {DEFAULT_MAX_ROW_BYTES}).",
    )
    agent_debug.add_argument(
        "--observable", choices=OBSERVABLES, default="unspecified", help="What the user saw go wrong; picks the competing hypotheses."
    )
    agent_debug.add_argument(
        "--receipt",
        action="append",
        help="A fanout dispatch_summary.json to bind as evidence; units that do not bind session, run, unit, configuration and freshness are refused.",
    )
    agent_debug.add_argument("--json", action="store_true", help="Print the machine-readable agent_debug_report/v1 payload with its incident.")
    agent_debug.set_defaults(func=cmd_quality_evidence_agent_debug)

    export = commands.add_parser(
        "agent-debug-export",
        help="Prepare a redacted agent-debug package for review; write it only with --confirm-export.",
        description=(
            "Re-check a reviewed `agent-debug --json` payload and every reference it cites against the source, "
            "drop the absolute source path, and leak-scan the package. Without --confirm-export it prints the "
            "package and writes nothing; with it, the package is written to a new file outside the source. "
            "Nothing is uploaded, filed as an issue, or posted."
        ),
    )
    export.add_argument("--report", required=True, help="The saved `agent-debug --json` payload the user reviewed.")
    export.add_argument("--session-record", help="The session record the payload cites, when it was read from one.")
    export.add_argument("--output", required=True, help="New file to write the package to; never overwritten.")
    export.add_argument("--confirm-export", action="store_true", help="Write the package after the user reviewed it.")
    export.add_argument("--json", action="store_true", help="Print the package and whether it was written as JSON.")
    export.set_defaults(func=cmd_quality_evidence_agent_debug_export)

    receipt = commands.add_parser(
        "cost-receipt",
        help="Print what one Hermes conversation's work cost, from recorded usage only.",
        description=(
            "Sum the recorded spend of a Hermes conversation: its session rows, delegated Hermes "
            "children, and fanout units that recorded the session as their origin. Observed cost, "
            "usage with no recorded price, and records with no usage are reported apart; nothing "
            "is estimated. Reads state.db (mode=ro) and dispatch summaries; metadata only."
        ),
    )
    receipt.add_argument("--session", required=True, help="Any Hermes session id of the conversation.")
    receipt.add_argument("--json", action="store_true", help="Print the machine-readable omh_cost_receipt/v1 payload.")
    receipt.set_defaults(func=cmd_quality_evidence_cost_receipt)

    agent_skills = commands.add_parser(
        "agent-skills",
        help="Report Agent Skills host evidence per host and scope: installed, discovered, nominated, loaded, executed.",
        description=(
            "For every Agent Skills host adapter and scope, read the host's scan path (repo scope under "
            "--project-root, user scope under --user-home) and compare its bytes with the pack inventory "
            "(agent-skills/.omh-agent-skills-manifest.json). OMH writes nothing and starts no host session; "
            "the only host process is the approved `<cli> --version` readiness probe, for a host version. "
            "Discovered, nominated, loaded and executed stay not_observed with each host's probe_unsupported "
            "reason. Exit 3 when a scan path could not be read. Not execution, review, CI, or merge evidence."
        ),
    )
    agent_skills.add_argument("--scope", choices=("repo", "user"), default=None, help="One scope only (default: both).")
    agent_skills.add_argument(
        "--project-root",
        default=None,
        help=(
            "Project directory whose repo-scope scan paths are read (default: current directory; an install made "
            "without --host writes at the repository root)."
        ),
    )
    agent_skills.add_argument(
        "--user-home", default=None, help="Home directory whose user-scope scan paths are read (default: your home)."
    )
    agent_skills.add_argument(
        "--receipts",
        default=None,
        help="Earlier `--json` output: report how its receipts compare with the current pack and host versions instead of reading scan paths.",
    )
    agent_skills.add_argument(
        "--json", action="store_true", help="Print the machine-readable omh_agent_skills_evidence/v1 payload."
    )
    agent_skills.set_defaults(func=cmd_quality_evidence_agent_skills)


def _add_json_input(parser: argparse.ArgumentParser, name: str, label: str) -> None:
    # A multi-word flag is hyphenated on the command line and underscored in the
    # namespace; argparse only derives that for positionals, so `dest` is set
    # here or `--changed-paths-json` lands on an unreachable attribute name.
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}-json", f"--{name}", dest=f"{dest}_json", help=f"Inline {label} JSON array.")
    parser.add_argument(f"--{name}-file", dest=f"{dest}_file", help=f"Path to {label} JSON array.")


def _json_object(path: str) -> dict[str, object]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("package JSON must be an object")
    return value


def _json_list(inline: str | None, path: str | None, label: str) -> list[Mapping[str, object]]:
    if inline is not None and path is not None:
        raise ValueError(f"{label} accepts either inline JSON or a file, not both")
    raw = inline if inline is not None else Path(path).expanduser().read_text(encoding="utf-8") if path else "[]"
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} JSON must be an array of objects")
    return value


def _json_strings(inline: str | None, path: str | None, label: str) -> list[str]:
    if inline is not None and path is not None:
        raise ValueError(f"{label} accepts either inline JSON or a file, not both")
    raw = inline if inline is not None else Path(path).expanduser().read_text(encoding="utf-8") if path else "[]"
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{label} JSON must be an array of nonblank strings")
    return value

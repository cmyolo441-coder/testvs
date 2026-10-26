from __future__ import annotations

import argparse
from pathlib import Path

from ..installer import OmhError
from ..workflows.plugin_hook_contract import observe_host_version
from ..workflows.plugin_risk_audit import audit_plugin_risk
from .common import _print_json


def cmd_ops_plugin_risk_audit(args: argparse.Namespace) -> int:
    try:
        host_version = observe_host_version(
            declared_version=args.hermes_version,
            install_dir=Path(args.hermes_install) if args.hermes_install else None,
        )
        payload = audit_plugin_risk(Path(args.path), host_version=host_version)
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(payload)
    return 0


def add_ops_plugin_risk_audit_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    audit = ops_sub.add_parser(
        "plugin-risk-audit",
        help=(
            "Statically audit one explicit local plugin directory without importing or registering it, classify the "
            "execution semantics of the hooks its manifest declares, and say whether its text composes a plausible "
            "self-update or code-replacement path."
        ),
        description=(
            "Reads bounded text from one local plugin directory and reports aggregate risk categories plus, for every "
            "hook declared in a root plugin.yaml, whether that hook would gate, transform, contribute context to, or "
            "only observe an action on the supported Hermes contract. It also composes the retrieval, replacement, "
            "archive-extraction and host-update-command signals it already collects into one self-update finding, "
            "naming the signals behind the verdict, because a plugin that can rewrite its own installed files is a "
            "second software-update authority beside the host-managed pinned update path, which keeps the pinning, "
            "rollback, consent and review guarantees. The result is an advisory static risk contract, not proof that "
            "the plugin is safe: nothing is imported, registered, installed or executed, so it is never evidence "
            "that a declared hook registered, ran, stayed inside its budget, or handled a failure correctly, that an "
            "update ran, that a checksum or signature was valid, or that the host accepted a revision -- and no "
            "self-update finding is not evidence that the plugin cannot replace its own code. "
            "The hook mapping is pinned to one Hermes revision, so name the host you would enable the plugin on "
            "and the result says whether that mapping covers it: `incompatible` and `unreadable` hold the "
            "hook-semantics decision at `unknown` instead of presenting a mapping that host was never read "
            "against as an established contract. Name neither and the compatibility is reported `not_observed`, "
            "which is not a pass. The named installation is read as a file -- Hermes is never imported, executed "
            "or asked for its version -- and a declared version can be stale or edited, so a compatible result "
            "is static pre-enable evidence, never proof that the host would admit, load or run the plugin."
        ),
    )
    audit.add_argument("--path", required=True, help="Explicit local plugin root directory to audit.")
    host = audit.add_mutually_exclusive_group()
    host.add_argument(
        "--hermes-version",
        help="Hermes version (x.y.z) the plugin would be enabled on, stated explicitly by the operator.",
    )
    host.add_argument(
        "--hermes-install",
        help=(
            "Local Hermes installation directory -- the one containing `hermes_cli/` -- whose declared "
            "`__version__` to read. Read as a file; nothing is imported, executed or discovered."
        ),
    )
    audit.set_defaults(func=cmd_ops_plugin_risk_audit)

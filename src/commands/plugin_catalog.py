"""`omh ecosystem plugin-catalog coverage` -- the offline snapshot adapter.

Consumes supplied JSON and nothing else. It performs no network fetch, reads
no credential, imports no plugin, installs no dependency, and changes no host
configuration; the two `--input`/`--previous` files are the only inputs and
stdout is the only output.

Fail-closed shape, deliberately split across the exit code rather than folded
into one:

- A snapshot that cannot be interpreted -- unreadable, oversized, malformed,
  duplicate-identity, or taken for another profile -- is refused with its
  bounded diagnostics and no payload. There is no partial coverage report,
  because a partial one reads exactly like a complete one.
- No snapshot at all prints the `unavailable` payload and exits non-zero. The
  reader still gets a machine-readable answer, and that answer is "OMH has no
  host coverage", never the packaged catalog wearing a host label.
- A snapshot older than the freshness horizon prints its rows with every one
  of them held on `stale_snapshot`, so nothing in it can be read as current
  adoption guidance.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..installer import OmhError
from ..workflows.plugin_catalog_coverage import (
    PluginCatalogCoverageError,
    build_plugin_catalog_coverage,
    plugin_catalog_coverage_unavailable,
    render_plugin_catalog_coverage,
)
from ..workflows.plugin_catalog_snapshots import (
    PluginCatalogSnapshotError,
    load_plugin_catalog_snapshot,
)
from .common import _print_json, _wants_json


def cmd_ecosystem_plugin_catalog_coverage(args: argparse.Namespace) -> int:
    if not args.input.strip():
        return _emit_unavailable(args, "no plugin catalog snapshot path was supplied")
    try:
        snapshot = load_plugin_catalog_snapshot(args.input)
    except PluginCatalogSnapshotError as exc:
        return _emit_unavailable(args, str(exc))
    previous: dict[str, Any] | None = None
    if args.previous.strip():
        try:
            previous = load_plugin_catalog_snapshot(args.previous)
        except PluginCatalogSnapshotError as exc:
            raise OmhError(f"previous snapshot: {exc}") from exc
    try:
        payload = build_plugin_catalog_coverage(
            snapshot,
            previous=previous,
            profile_ref=args.profile,
            now=args.now,
        )
    except PluginCatalogCoverageError as exc:
        raise OmhError(str(exc)) from exc
    _emit(args, payload)
    return 0


def _emit_unavailable(args: argparse.Namespace, reason: str) -> int:
    _emit(args, plugin_catalog_coverage_unavailable(reason))
    return 1


def _emit(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if _wants_json(args):
        _print_json(payload)
        return
    print(render_plugin_catalog_coverage(payload))


def _add_plugin_catalog_commands(ecosystem_sub) -> None:
    plugin_catalog = ecosystem_sub.add_parser(
        "plugin-catalog",
        help="Reconcile a supplied host plugin catalog snapshot with OMH workflow ownership.",
    )
    plugin_catalog_sub = plugin_catalog.add_subparsers(dest="plugin_catalog_command", required=True)

    coverage = plugin_catalog_sub.add_parser(
        "coverage",
        help="Report OMH outcome coverage for one plugin_catalog_snapshot/v1 file.",
        description=(
            "Read one plugin_catalog_snapshot/v1 an authorized host or operator produced and report which "
            "OMH workflow owns each plugin outcome, what changed since a prior snapshot, which entries are "
            "held by a removal or an unmet host-version constraint, and what evidence OMH does not have. "
            "OMH fetches nothing, reads no credential, imports no plugin, installs no dependency, and "
            "changes no host configuration."
        ),
    )
    coverage.add_argument(
        "--input",
        default="",
        help="Path to the current plugin_catalog_snapshot/v1 JSON file. Without it the answer is unavailable.",
    )
    coverage.add_argument(
        "--previous",
        default="",
        help="Path to the prior snapshot to classify added, changed, unchanged, and removed entries against.",
    )
    coverage.add_argument(
        "--profile",
        default="",
        help="Refuse the snapshot unless it was taken for this exact profile.",
    )
    coverage.add_argument(
        "--now",
        default="",
        help="ISO-8601 reference time; a snapshot past the freshness horizon holds every entry it carries.",
    )
    coverage.add_argument("--json", action="store_true", help="Emit the machine payload instead of plain text.")
    coverage.set_defaults(func=cmd_ecosystem_plugin_catalog_coverage)


__all__ = [
    "_add_plugin_catalog_commands",
    "cmd_ecosystem_plugin_catalog_coverage",
]

"""Derive a coding handoff's context pack from the project's stored codegraph.

A prepared coding handoff used to reach the coding owner with no map of the
repository unless the caller built one and passed it in (#1696). When the
project already carries a codegraph artifact (`omh codegraph build --write`),
its structurally ranked focus files are the context a coding owner most needs
and OMH can enumerate on its own, so they travel as a `handoff_context_pack/v1`.

Only a stored artifact is read. Building a graph here would scan the whole
tree on every handoff, and a project that never asked for a codegraph keeps
the handoff it had. The pack names files; it carries no file bytes and no task
text, and it is static analysis -- prepared context, never evidence that an
owner opened any of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..codegraph import CODEGRAPH_SCHEMA_VERSION, build_handoff_context, codegraph_artifact_path
from ..codegraph.schema import CLAIM_BOUNDARY, CODEGRAPH_ARTIFACT_RELATIVE_PATH, CODEGRAPH_CONTEXT_TRUTH_LEVEL
from ..memory import validate_handoff_context_pack
from .handoff_input_manifest import manifest_file_refs
from ..system.local_store import read_json_object_result
from ..system.paths import project_identity


CODEGRAPH_CONTEXT_SOURCE = "omh_codegraph"
CODEGRAPH_CONTEXT_NOT_ATTACHED_SCHEMA_VERSION = "codegraph_context_not_attached/v1"
# Ranked beside a catalog hint: both are static descriptions of the project,
# neither was observed at run time.
CODEGRAPH_CONTEXT_PRECEDENCE = 40


def derived_codegraph_context_pack(
    project_root: str | Path,
    *,
    message: str,
    executor_target: str,
    input_manifest: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Return `(pack, not_attached)`; both are None when there is no artifact.

    An artifact that exists but cannot be used returns a `not_attached` record
    naming why, so a stale or corrupt codegraph is visible on the payload
    rather than read as "this project has no codegraph".
    """
    artifact = codegraph_artifact_path(project_root)
    if not artifact.is_file():
        return None, None
    graph, error = read_json_object_result(artifact)
    if error is not None or graph is None:
        return None, _not_attached("artifact_unreadable")
    if graph.get("schema_version") != CODEGRAPH_SCHEMA_VERSION or not isinstance(graph.get("files"), list):
        return None, _not_attached("artifact_schema_mismatch")
    # Files the caller already put in the package are in play, so they seed.
    context = build_handoff_context(graph, task=message, changed_paths=manifest_file_refs(input_manifest))
    focus_files = [record for record in context["focus_files"] if isinstance(record, dict)]
    if not focus_files:
        return None, None
    scope = {"kind": "project", "ref": project_identity(project_root)}
    pack: dict[str, object] = {
        "schema_version": "handoff_context_pack/v1",
        "executor_target": executor_target,
        "session_id": "",
        "scope": dict(scope),
        "metadata": {
            "codegraph_artifact_ref": CODEGRAPH_ARTIFACT_RELATIVE_PATH.as_posix(),
            "codegraph_generated_at": str(graph.get("generated_at", "")),
            "codegraph_ranking": "personalized_pagerank_internal_imports",
            "codegraph_seed_changed_path_count": len(context["changed_paths"]),
        },
        "source_refs": [
            {
                "source": CODEGRAPH_CONTEXT_SOURCE,
                "truth_level": CODEGRAPH_CONTEXT_TRUTH_LEVEL,
                "precedence": CODEGRAPH_CONTEXT_PRECEDENCE,
                "item_count": len(focus_files),
            }
        ],
        "included_context": [
            _focus_file_item(record, rank=rank, total=len(focus_files), scope=scope)
            for rank, record in enumerate(focus_files, start=1)
        ],
        "excluded_context": [],
        "blocked_by_conflicts": [],
        "redaction_policy": "metadata_only",
        "claim_boundary": str(graph.get("claim_boundary") or CLAIM_BOUNDARY),
    }
    if validate_handoff_context_pack(pack, require_conflict_free=True, label="codegraph context pack"):
        return None, _not_attached("context_pack_invalid")
    return pack, None


def _focus_file_item(record: dict[str, Any], *, rank: int, total: int, scope: dict[str, str]) -> dict[str, object]:
    path = str(record.get("path", ""))
    tags = [str(tag) for tag in record.get("entrypoint_tags", []) or []]
    summary = f"Codegraph focus file {rank} of {total}: {path}"
    if tags:
        summary += f" ({', '.join(tags)})"
    return {
        "item_id": f"codegraph:{path}",
        # The repo-relative path, never an absolute one: the input manifest
        # reads `key` as the item's local ref, and an absolute path would put
        # the operator's home directory into the manifest digest.
        "key": path,
        "summary": f"{summary}.",
        "artifact_ref": path,
        "source": CODEGRAPH_CONTEXT_SOURCE,
        "truth_level": CODEGRAPH_CONTEXT_TRUTH_LEVEL,
        "scope": dict(scope),
    }


def _not_attached(reason: str) -> dict[str, object]:
    return {
        "schema_version": CODEGRAPH_CONTEXT_NOT_ATTACHED_SCHEMA_VERSION,
        "status": "not_attached",
        "reason": reason,
        "artifact_ref": CODEGRAPH_ARTIFACT_RELATIVE_PATH.as_posix(),
    }

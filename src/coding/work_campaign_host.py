"""Host adapter boundary and read-only installed-Hermes Kanban observations.

The installed host can pin task routes but cannot attest an immutable scoped
leaf sandbox and authenticated verification. Binding observations therefore
remain prepared; this adapter never starts a worker, daemon, or provider.
"""
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Protocol

from ..plugin_bundle.omh.delegation_routing import read_delegation_route

# Hermes f159e581: delegate_task has no per-task model/provider/effort.
# A version number or a future schema field is NOT proof of adapter semantics.
NATIVE_PER_DISPATCH_SUPPORTED = False
KANBAN_COLUMNS = {"id", "status", "model_override", "provider_override", "reasoning_effort",
                  "assignee", "idempotency_key", "session_id", "max_retries", "goal_mode"}
# Closed non-composite toolsets from f159e581 toolsets.py. Terminal/code
# execution would permit shell-level recursive spawning, so neither is safe.
WORKER_TOOLSET_ALLOWLIST = frozenset({"file"})


class CampaignHostAdapter(Protocol):
    """Trusted wrapper interface, never built from model-supplied JSON.

    context is the invoking host session, not a caller-selected actor. dispatch
    atomically binds the exact attempt to route, scoped tools and recipient,
    with host idempotency. observe returns host-inspected diff/revision and
    observed verification records, not worker summaries. expire revokes pending
    routes AND stops in-flight work. A host lacking any part must fail closed.
    """
    def context(self) -> str: ...
    def capabilities(self) -> dict[str, Any]: ...
    def dispatch(self, manifest: dict[str, Any]) -> dict[str, Any]: ...
    def observe(self, campaign_id: str, attempt_id: str) -> dict[str, Any]: ...
    def expire(self, campaign_id: str, attempt_ids: tuple[str, ...]) -> bool: ...
    def ordinary_route(self) -> dict[str, Any]: ...


def observe_host_bindings(hermes_home: Path) -> dict[str, Any]:
    result = dict(delegate_task_per_task_model=NATIVE_PER_DISPATCH_SUPPORTED,
                  kanban_create_model_override=False, status="not_observed")
    db = Path(hermes_home) / "kanban.db"
    if not db.is_file():
        return result | {"reason": "kanban_db_absent"}
    try:
        with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    except sqlite3.Error:
        return result | {"reason": "kanban_db_unreadable"}
    if not KANBAN_COLUMNS.issubset(columns):
        return result | {"reason": "kanban_columns_missing"}
    return result | {"kanban_create_model_override": True, "status": "observed",
                     "reason": "pins_only_not_execution_or_confinement"}


def observe_kanban_bindings(hermes_home: Path, task_refs: list[str]) -> dict[str, Any]:
    if not isinstance(task_refs, list) or not 1 <= len(task_refs) <= 17 or len(set(task_refs)) != len(task_refs):
        raise ValueError("bounded_unique_task_refs_required")
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", x) for x in task_refs):
        raise ValueError("invalid_host_task_ref")
    capability = observe_host_bindings(hermes_home)
    if not capability["kanban_create_model_override"]:
        return capability
    db = Path(hermes_home) / "kanban.db"
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
        conn.row_factory = sqlite3.Row
        columns = sorted(KANBAN_COLUMNS)
        rows = conn.execute("SELECT " + ",".join(columns) + " FROM tasks WHERE id IN ("
                            + ",".join("?" for _ in task_refs) + ")", task_refs).fetchall()
        links = conn.execute("SELECT parent_id, child_id FROM task_links WHERE child_id IN ("
                             + ",".join("?" for _ in task_refs) + ") LIMIT 290", task_refs).fetchall()
    if len(rows) != len(task_refs):
        return {"status": "not_observed", "reason": "host_task_missing"}
    return {"status": "observed", "tasks": [dict(row) for row in rows],
            "parents": {task_id: sorted(r["parent_id"] for r in links if r["child_id"] == task_id) for task_id in task_refs},
            "model_execution": "not_observed"}


def observe_worker_profile(hermes_home: Path, profile: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", profile):
        raise ValueError("invalid_worker_profile")
    path = Path(hermes_home) / "profiles" / profile / "config.yaml"
    try:
        with path.open("rb") as stream:
            raw = stream.read(16385)
    except FileNotFoundError:
        return {"status": "not_observed", "reason": "worker_profile_absent"}
    if len(raw) > 16384 or path.is_symlink():
        return {"status": "not_observed", "reason": "worker_profile_unsafe"}
    # JSON is valid YAML and lets a zero-dependency observer prove unambiguous
    # semantics. Also accept the exact ordinary block-list spelling below;
    # aliases, anchors, duplicates, tags, and composite toolsets fail closed.
    text = raw.decode("utf-8")
    try:
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate_profile_key")
                obj[key] = value
            return obj
        config = json.loads(text, object_pairs_hook=unique)
        toolsets = config.get("platform_toolsets", {}).get("cli")
    except (ValueError, AttributeError):
        matches = re.findall(r"(?m)^platform_toolsets:\s*\n  cli:\s*\n((?:    - [a-z_-]+\n)+)", text + "\n")
        if (len(matches) != 1 or len(re.findall(r"(?m)^platform_toolsets:", text)) != 1
                or len(re.findall(r"(?m)^  cli:", text)) != 1 or re.search(r"[&*!]|<<:", text)):
            return {"status": "not_observed", "reason": "leaf_toolsets_ambiguous"}
        toolsets = re.findall(r"- ([a-z_-]+)", matches[0])
    if not isinstance(toolsets, list) or not toolsets or any(not isinstance(x, str) or x not in WORKER_TOOLSET_ALLOWLIST for x in toolsets):
        return {"status": "not_observed", "reason": "leaf_can_delegate_or_expand_scope"}
    return dict(status="observed", toolsets=toolsets, config_digest=sha256(raw).hexdigest(),
                runtime_confinement="not_observed")


class LocalCampaignHost:
    """Operator CLI identity is OS-observed, never an --owner-session string.

    This is deliberately not a Hermes chat-session authentication adapter.
    The same invoking wrapper process can inspect/cancel its own prepared
    campaigns. Host-owned orchestration requires CampaignHostAdapter above.
    """
    def __init__(self, hermes_home: Path):
        self.hermes_home = Path(hermes_home)

    def context(self):
        return f"operator-{os.getuid() if hasattr(os, 'getuid') else 'local'}-{os.getppid()}"

    def capabilities(self):
        return dict(binding="unbound", identity_bound=False, scope_enforced=False,
                    verification_observer=False, leaf_tools=[])

    def dispatch(self, manifest):
        raise ValueError("host_dispatch_adapter_unavailable")

    def observe(self, campaign_id, attempt_id):
        return {}

    def expire(self, campaign_id, attempt_ids):
        # This adapter never prepares a consumable route or launches a task.
        return not attempt_ids

    def ordinary_route(self):
        return read_delegation_route(self.hermes_home)

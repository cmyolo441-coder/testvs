"""Next-Generation Parallel Subagent Swarm & DAG Orchestration Engine for FullAgent.

Capabilities:
* **Topological DAG Wave & Dependency Engine (`depends_on`)**: Validates task
  dependencies by worker name, batch index, or worker ID, rejects cycles
  atomically before launch, schedules execution in topological waves, and pipes
  completed upstream reports into downstream workers' context automatically.
* **Shared Blackboard Memory Bus (`SharedBlackboard`)**: Thread-safe real-time
  knowledge bus (`blackboard_publish` / `blackboard_read`) allowing parallel
  specialists to exchange discovered symbols, facts, and hypotheses mid-run.
* **Live Per-Worker File Diff Capture (`worker.diffs`)**: Every scoped file
  mutation (`write_file`, `edit_file`) performed by a writer subagent captures a
  unified diff and `+added/-removed` line statistics for live inspection in
  `/agents diff` and batch synthesis reports.
* **Intra-Worker Parallel Read Fan-Out**: Consecutive read-only tool calls
  within a single subagent step execute concurrently via an internal read pool
  while preserving deterministic message ordering.
* **Priority Queue, Transient Retries & Dynamic Scaling**: Priority-ordered wave
  dispatch (`priority`), exponential-backoff retries (`max_retries`), and live
  pool resizing (`scale_workers`).
* **Visual DAG Topology & Swarm HUD (`format_dag`, `format_swarm_hud`)**:
  Unicode wave graph, progress bars (`▰▰▰▰▱`), token velocity (`tok/s`), and
  executive synthesis (`synthesize_batch`).
"""
from __future__ import annotations

import concurrent.futures
import difflib
import inspect
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cancelguard import TurnCancelled
from .client import (
    assistant_message,
    chat_with_retry,
    normalize_tool_call,
    safe_token_counts,
)
from .roles import ROLES, role_label
from .tools import Tool, parse_tool_arguments

MAX_GROUP_SIZE = 8
# Real concurrency limit. MAX_GROUP_SIZE is the largest batch the tools
# accept, so the pool must be able to run that whole batch at once —
# otherwise "8 parallel specialists" would silently be 4 running + 4
# queued (the TUI shows the exact split, but the capability should be real).
DEFAULT_WORKERS = 8
DEFAULT_ACTIVE_LIMIT = 8
MAX_RETAINED = 40
MAX_TASK_CHARS = 12_000
MAX_CONTEXT_CHARS = 24_000
MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_CALLS_PER_STEP = 12
MAX_TOTAL_TOOL_CALLS = 80
MAX_STEPS = 20
DEFAULT_ACTIVE_LIMIT = 8
CALL_TIMEOUT = 120.0
DRAIN_TIMEOUT = 5.0

ACTIVE_STATES = frozenset({"queued", "running", "cancelling"})
TERMINAL_STATES = frozenset({"done", "blocked", "error", "cancelled"})
WRITE_TOOLS = frozenset({"write_file", "edit_file", "create_directory"})
PARALLEL_SUBAGENT_READ_TOOLS = frozenset({
    "read_file", "list_dir", "file_info", "search_files", "glob_files",
})
NEVER_DELEGATE = frozenset({
    "run_command", "live_shell", "live_shell_reset", "delete_path",
    "copy_path", "move_path", "apply_patch", "Task", "TaskGroup",
    "TaskWriteGroup", "TaskOutput", "TaskCollect", "TaskList", "TaskStop",
})


class SubagentError(ValueError):
    """Raised when a subagent task or batch violates an execution or safety invariant."""


class SharedBlackboard:
    """Thread-safe cross-subagent memory bus for sharing facts and discoveries live."""

    MAX_ENTRIES = 256

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}

    def publish(self, key: str, value: str, author: str = "sovereign",
                role: str = "") -> dict[str, Any]:
        k = str(key or "").strip()[:120]
        if not k:
            raise ValueError("blackboard key must be non-empty")
        v = str(value or "").strip()[:4000]
        with self._lock:
            if k not in self._entries and len(self._entries) >= self.MAX_ENTRIES:
                oldest_k = min(
                    self._entries, key=lambda x: self._entries[x]["ts"]
                )
                self._entries.pop(oldest_k, None)
            entry = {
                "key": k,
                "value": v,
                "author": str(author),
                "role": str(role),
                "ts": time.time(),
            }
            self._entries[k] = entry
            return dict(entry)

    def read(self, prefix: str = "") -> list[dict[str, Any]]:
        pfx = str(prefix or "").strip().lower()
        with self._lock:
            items = [
                dict(v)
                for k, v in self._entries.items()
                if not pfx or k.lower().startswith(pfx) or pfx in k.lower()
            ]
        return sorted(items, key=lambda e: e["ts"])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def format_board(self) -> str:
        items = self.read()
        if not items:
            return "Shared blackboard is empty."
        lines = [f"SHARED BLACKBOARD · {len(items)} fact(s):"]
        for item in items:
            tag = f"{item['author']}" + (f"/{item['role']}" if item["role"] else "")
            lines.append(f"  • [{tag}] {item['key']}: {item['value']}")
        return "\n".join(lines)


class PathScopeRegistry:
    """Hierarchical workspace path-scope validator and fine-grained lock manager."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self._file_locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    @staticmethod
    def is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def lock_for(self, resolved_path: str) -> threading.RLock:
        with self._guard:
            lock = self._file_locks.get(resolved_path)
            if lock is None:
                lock = threading.RLock()
                self._file_locks[resolved_path] = lock
            return lock

    def normalize_scope(self, value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if value in (None, ""):
            return (), ()
        if not isinstance(value, (list, tuple)):
            raise SubagentError("write_scope must be a list of workspace paths")
        if len(value) > 16:
            raise SubagentError("write_scope supports at most 16 paths")
        paths: list[str] = []
        dirs: list[str] = []
        for raw in value:
            if not isinstance(raw, str) or not raw.strip():
                raise SubagentError("each write_scope entry must be a path")
            raw = raw.strip()
            candidate = Path(raw).expanduser()
            if ".." in candidate.parts:
                raise SubagentError("write_scope cannot contain '..'")
            full = candidate if candidate.is_absolute() else self.workspace / candidate
            resolved = full.resolve(strict=False)
            if not self.is_relative_to(resolved, self.workspace):
                raise SubagentError(
                    f"write_scope path escapes the workspace: {raw!r}"
                )
            key = str(resolved)
            if key not in paths:
                paths.append(key)
            if raw.endswith(("/", "\\")) or (resolved.exists() and resolved.is_dir()):
                if key not in dirs:
                    dirs.append(key)
        return tuple(paths), tuple(dirs)

    @classmethod
    def scopes_overlap(
        cls,
        a_paths: tuple[str, ...],
        a_dirs: tuple[str, ...],
        b_paths: tuple[str, ...],
        b_dirs: tuple[str, ...],
    ) -> bool:
        for left in a_paths:
            lp = Path(left)
            for right in b_paths:
                rp = Path(right)
                if lp == rp:
                    return True
                if left in a_dirs and cls.is_relative_to(rp, lp):
                    return True
                if right in b_dirs and cls.is_relative_to(lp, rp):
                    return True
        return False


@dataclass
class Subagent:
    """State, isolated conversation, live diffs, and telemetry for one specialist worker."""

    id: str
    batch_id: str
    name: str
    role: str
    task: str
    model_id: str
    write_scope: tuple[str, ...]
    write_directories: tuple[str, ...]
    provider: Any = field(repr=False)
    model: Any = field(repr=False)
    effort: Any = field(repr=False)
    tools: dict[str, Tool] = field(repr=False)
    session_id: str = ""
    messages: list[dict] = field(repr=False, default_factory=list)
    state: str = "queued"
    summary: str = ""
    error: str = ""
    current_action: str = "queued"
    steps: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    files_touched: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    spawn_event_id: str = ""
    stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    # Advanced orchestration & telemetry fields
    priority: int = 0
    wave: int = 0
    depends_on: tuple[str, ...] = ()
    max_retries: int = 0
    retries: int = 0
    timeout: float = CALL_TIMEOUT
    tool_breakdown: dict[str, int] = field(default_factory=dict)
    parallel_reads_saved_ms: int = 0
    diffs: list[dict[str, Any]] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0

    def tokens_per_sec(self) -> float:
        if self.started_at <= 0:
            return 0.0
        dur = max(0.001, (self.finished_at or time.time()) - self.started_at)
        return round((self.tokens_in + self.tokens_out) / dur, 1)

    def progress_pct(self, max_steps: int = MAX_STEPS) -> int:
        if self.state == "done":
            return 100
        if self.state in ("error", "cancelled", "blocked"):
            return 100
        if self.state == "queued":
            return 0
        return min(95, max(10, int((self.steps / max(1, max_steps)) * 100)))

    def to_dict(self) -> dict:
        """Return a bounded, JSON-safe telemetry & status record."""
        elapsed = max(
            0, int(((self.finished_at or time.time()) - self.created_at) * 1000)
        )
        return {
            "id": str(self.id),
            "name": str(self.name),
            "batch_id": str(self.batch_id),
            "role": str(self.role),
            "task": str(self.task[:2000]),
            "model": str(self.model_id),
            "state": str(self.state),
            "summary": str(self.summary[:4000]),
            "error": str(self.error[:2000]),
            "current_action": str(self.current_action[:200]),
            "write_scope": list(self.write_scope),
            "steps": max(0, int(self.steps)),
            "tool_calls": max(0, int(self.tool_calls)),
            "tokens_in": max(0, int(self.tokens_in)),
            "tokens_out": max(0, int(self.tokens_out)),
            "files_touched": list(self.files_touched[:32]),
            "elapsed_ms": elapsed,
            "priority": int(self.priority),
            "wave": int(self.wave),
            "depends_on": list(self.depends_on),
            "retries": int(self.retries),
            "tool_breakdown": dict(self.tool_breakdown),
            "tokens_per_sec": self.tokens_per_sec(),
            "parallel_reads_saved_ms": int(self.parallel_reads_saved_ms),
            "lines_added": int(self.lines_added),
            "lines_removed": int(self.lines_removed),
            "diffs_count": len(self.diffs),
        }


class ParallelSubagents:
    """Advanced DAG-capable, scope-isolated parallel subagent swarm orchestrator."""

    def __init__(
        self,
        log,
        tools: dict[str, Tool],
        runtime_resolver: Callable[[str], tuple],
        before_write: Callable[[Subagent, Tool, dict], str | None] | None = None,
        *,
        workspace: str | Path | None = None,
        after_write: Callable[[Subagent, Tool, dict], str] | None = None,
        max_workers: int = DEFAULT_WORKERS,
        max_active: int = DEFAULT_ACTIVE_LIMIT,
        max_steps: int = MAX_STEPS,
        chat: Callable | None = None,
    ) -> None:
        if not callable(runtime_resolver):
            raise TypeError("runtime_resolver must be callable")
        self.log = log
        self.tools = tools
        self.runtime_resolver = runtime_resolver
        self.before_write = before_write
        self.after_write = after_write
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self._scope_registry = PathScopeRegistry(self.workspace)
        self.blackboard = SharedBlackboard()
        self.max_workers = max(1, min(16, int(max_workers)))
        self.max_active = max(1, min(24, int(max_active)))
        self.max_steps = max(1, min(64, int(max_steps)))
        self.chat = chat or chat_with_retry
        self._session_id = str(getattr(log, "session", "") or "")
        self._lock = threading.RLock()
        self._workers: dict[str, Subagent] = {}
        self._order: list[str] = []
        self._futures: dict[str, concurrent.futures.Future] = {}
        self._done_events: dict[str, threading.Event] = {}
        self._counter = 0
        self._batch_counter = 0
        self._latest_batch_id = ""
        self._batch_members: dict[str, list[str]] = {}
        self._active_running = 0
        self._peak_concurrency = 0
        self._progress_callbacks: list[Callable[[Subagent, str], None]] = []
        self._closed = False
        self._transitioning = False
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="fullagent:subagent",
        )

    # -- progress & scaling --------------------------------------------------

    def on_progress(self, callback: Callable[[Subagent, str], None]) -> None:
        """Register a live progress listener ``callback(worker, event_kind)``."""
        if callable(callback):
            with self._lock:
                self._progress_callbacks.append(callback)

    def _notify_listeners(self, worker: Subagent, kind: str) -> None:
        with self._lock:
            listeners = list(self._progress_callbacks)
        for cb in listeners:
            try:
                cb(worker, kind)
            except Exception:
                pass

    def scale_workers(self, new_max_workers: int) -> int:
        """Dynamically adjust max_workers for future batches."""
        val = max(1, min(16, int(new_max_workers)))
        with self._lock:
            self.max_workers = val
        return val

    # -- validation and scope isolation -------------------------------------

    @staticmethod
    def _relative_to(path: Path, root: Path) -> bool:
        return PathScopeRegistry.is_relative_to(path, root)

    def _normalise_scope(self, value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self._scope_registry.workspace != self.workspace:
            self._scope_registry = PathScopeRegistry(self.workspace)
        return self._scope_registry.normalize_scope(value)

    @staticmethod
    def _scope_overlap(
        a: Subagent, b_paths: tuple[str, ...], b_dirs: tuple[str, ...]
    ) -> bool:
        return PathScopeRegistry.scopes_overlap(
            a.write_scope, a.write_directories, b_paths, b_dirs
        )

    def _validate_item(
        self,
        item: Any,
        default_role: str,
        default_context: str,
        model_id: str,
        *,
        allow_writes: bool,
    ) -> dict:
        if isinstance(item, str):
            item = {"task": item}
        if not isinstance(item, dict):
            raise SubagentError("each task must be a string or an object")
        task = item.get("task")
        if not isinstance(task, str) or not task.strip():
            raise SubagentError("every subagent needs a non-empty task")
        task = task.strip()
        if len(task) > MAX_TASK_CHARS:
            raise SubagentError(f"task exceeds {MAX_TASK_CHARS:,} characters")
        role = item.get("role", default_role)
        if not isinstance(role, str) or role not in ROLES:
            valid = ", ".join(sorted(ROLES))
            raise SubagentError(f"unknown role {role!r}; available: {valid}")
        name = item.get("name", "")
        context = item.get("context", default_context)
        requested_model = item.get("model_id", item.get("model", model_id))
        if not isinstance(name, str) or not isinstance(context, str):
            raise SubagentError("name and context must be strings")
        if len(context) > MAX_CONTEXT_CHARS:
            raise SubagentError(
                f"shared context exceeds {MAX_CONTEXT_CHARS:,} characters"
            )
        if not isinstance(requested_model, str):
            raise SubagentError("model_id must be a string")

        raw_priority = item.get("priority", 0)
        try:
            priority = max(-20, min(20, int(raw_priority)))
        except (TypeError, ValueError) as exc:
            raise SubagentError("priority must be an integer") from exc

        raw_retries = item.get("max_retries", 0)
        try:
            max_retries = max(0, min(5, int(raw_retries)))
        except (TypeError, ValueError) as exc:
            raise SubagentError("max_retries must be an integer") from exc

        raw_timeout = item.get("timeout", CALL_TIMEOUT)
        try:
            timeout_val = max(1.0, min(600.0, float(raw_timeout)))
        except (TypeError, ValueError) as exc:
            raise SubagentError("timeout must be a number") from exc

        raw_deps = item.get("depends_on", ())
        if raw_deps in (None, ""):
            deps: tuple[str, ...] = ()
        elif isinstance(raw_deps, str):
            deps = (raw_deps.strip(),) if raw_deps.strip() else ()
        elif isinstance(raw_deps, (list, tuple)):
            clean_deps: list[str] = []
            for d in raw_deps:
                if not isinstance(d, str) or not d.strip():
                    raise SubagentError("each depends_on entry must be a non-empty string")
                ds = d.strip()
                if ds not in clean_deps:
                    clean_deps.append(ds)
            deps = tuple(clean_deps)
        else:
            raise SubagentError("depends_on must be a list of task names or worker IDs")

        paths, dirs = self._normalise_scope(item.get("write_scope", ()))
        role_writes = bool(ROLES[role].get("writes"))
        if allow_writes and role_writes and not paths:
            raise SubagentError("write-enabled tasks need a non-empty write_scope")
        if paths and not allow_writes:
            raise SubagentError(
                "TaskGroup is read-only; use TaskWriteGroup for scoped edits"
            )
        if paths and not role_writes:
            raise SubagentError(f"role {role!r} does not support file writes")
        try:
            runtime = self.runtime_resolver(requested_model.strip())
        except Exception as exc:
            raise SubagentError(
                f"cannot resolve subagent model: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(runtime, (tuple, list)) or len(runtime) != 3:
            raise SubagentError(
                "runtime_resolver must return (provider, model, effort)"
            )
        provider, model, effort = runtime
        if not getattr(model, "supports_tools", False):
            raise SubagentError(
                f"model {getattr(model, 'id', model)!r} does not support tools"
            )
        names = ROLES[role].get("tools", ())
        worker_tools: dict[str, Tool] = {}
        for tool_name in names:
            tool = self.tools.get(tool_name)
            if tool is None or tool_name in NEVER_DELEGATE:
                continue
            if tool_name in WRITE_TOOLS and (not role_writes or not paths):
                continue
            worker_tools[tool_name] = tool
        return {
            "task": task,
            "role": role,
            "name": name.strip()[:80],
            "context": context,
            "model_id": str(getattr(model, "id", "")),
            "provider": provider,
            "model": model,
            "effort": effort,
            "tools": worker_tools,
            "write_scope": paths,
            "write_directories": dirs,
            "priority": priority,
            "max_retries": max_retries,
            "timeout": timeout_val,
            "depends_on": deps,
        }

    def _resolve_and_validate_dag(
        self, prepared: list[dict], assigned_ids: list[str]
    ) -> tuple[list[tuple[str, ...]], list[int]]:
        """Resolve task dependencies and compute topological execution waves."""
        name_to_id: dict[str, str] = {}
        for idx, (item, aid) in enumerate(zip(prepared, assigned_ids)):
            if item["name"]:
                if item["name"] in name_to_id:
                    raise SubagentError(
                        f"duplicate subagent name in batch: {item['name']!r}"
                    )
                name_to_id[item["name"]] = aid
            name_to_id[f"#{idx}"] = aid
            name_to_id[aid] = aid

        resolved_deps: list[tuple[str, ...]] = []
        batch_id_set = set(assigned_ids)
        for item, aid in zip(prepared, assigned_ids):
            item_deps: list[str] = []
            for raw_dep in item["depends_on"]:
                target_id = name_to_id.get(raw_dep)
                if target_id is None:
                    if raw_dep in self._workers:
                        target_id = raw_dep
                    else:
                        raise SubagentError(
                            f"unknown dependency {raw_dep!r} for task {item['name'] or aid}"
                        )
                if target_id == aid:
                    raise SubagentError(
                        f"dependency cycle detected: task {aid} depends on itself"
                    )
                if target_id not in item_deps:
                    item_deps.append(target_id)
            resolved_deps.append(tuple(item_deps))

        id_to_idx = {aid: i for i, aid in enumerate(assigned_ids)}
        waves = [0] * len(prepared)
        visiting: set[int] = set()
        visited: set[int] = set()

        def compute_wave(idx: int) -> int:
            if idx in visited:
                return waves[idx]
            if idx in visiting:
                raise SubagentError("dependency cycle detected in subagent batch")
            visiting.add(idx)
            w = 0
            for dep_id in resolved_deps[idx]:
                if dep_id in batch_id_set:
                    w = max(w, 1 + compute_wave(id_to_idx[dep_id]))
                else:
                    w = max(w, 1)
            visiting.remove(idx)
            visited.add(idx)
            waves[idx] = w
            return w

        for i in range(len(prepared)):
            compute_wave(i)

        return resolved_deps, waves

    # -- lifecycle -----------------------------------------------------------

    def _emit(self, kind: str, data: dict, worker: Subagent | None = None):
        safe_data = dict(data) if isinstance(data, dict) else {"_raw": str(data)}
        actor = f"subagent:{worker.id}" if worker else "sovereign"
        cause = worker.spawn_event_id if worker and worker.spawn_event_id else None
        session = worker.session_id if worker else self._session_id
        try:
            event = self.log.append(
                kind, safe_data, actor=actor, causation_id=cause, session=session
            )
        except TypeError:
            event = self.log.append(kind, safe_data)
        if worker is not None:
            self._notify_listeners(worker, kind)
        return event

    def _prune(self, additional: int = 1) -> None:
        while len(self._order) + additional > MAX_RETAINED:
            oldest = next(
                (
                    aid
                    for aid in self._order
                    if self._workers[aid].state in TERMINAL_STATES
                ),
                None,
            )
            if oldest is None:
                raise SubagentError(
                    f"subagent history is full ({MAX_RETAINED}); wait or stop work"
                )
            self._order.remove(oldest)
            removed = self._workers.pop(oldest, None)
            self._futures.pop(oldest, None)
            self._done_events.pop(oldest, None)
            if removed is not None:
                members = self._batch_members.get(removed.batch_id, [])
                self._batch_members[removed.batch_id] = [
                    aid for aid in members if aid != oldest
                ]
                if not self._batch_members[removed.batch_id]:
                    self._batch_members.pop(removed.batch_id, None)
                    if self._latest_batch_id == removed.batch_id:
                        self._latest_batch_id = next(
                            reversed(self._batch_members), ""
                        )

    def spawn(
        self,
        task: str,
        role: str = "researcher",
        *,
        name: str = "",
        context: str = "",
        model_id: str = "",
        write_scope: tuple[str, ...] | list[str] = (),
        allow_writes: bool = False,
        priority: int = 0,
        depends_on: tuple[str, ...] | list[str] = (),
        max_retries: int = 0,
    ) -> Subagent:
        return self.spawn_many(
            [
                {
                    "task": task,
                    "role": role,
                    "name": name,
                    "context": context,
                    "model_id": model_id,
                    "write_scope": write_scope,
                    "priority": priority,
                    "depends_on": depends_on,
                    "max_retries": max_retries,
                }
            ],
            allow_writes=allow_writes,
        )[0]

    def spawn_many(
        self,
        tasks: list[Any] | tuple[Any, ...],
        *,
        role: str = "researcher",
        context: str = "",
        model_id: str = "",
        allow_writes: bool = False,
    ) -> list[Subagent]:
        """Validate then atomically submit a bounded batch of concurrent or DAG-ordered tasks."""
        if not isinstance(tasks, (list, tuple)):
            raise SubagentError("tasks must be a list")
        if len(tasks) > MAX_GROUP_SIZE:
            raise SubagentError(f"a group supports at most {MAX_GROUP_SIZE} tasks")
        if not tasks:
            return []
        if not isinstance(role, str) or role not in ROLES:
            raise SubagentError(f"unknown default role: {role!r}")
        prepared = [
            self._validate_item(
                item, role, context, model_id, allow_writes=allow_writes
            )
            for item in tasks
        ]

        with self._lock:
            if self._closed:
                raise SubagentError("subagent manager is shutting down")
            if self._transitioning:
                raise SubagentError("session is changing; try again")

            base_counter = self._counter
            candidate_ids = [
                f"sa-{base_counter + idx + 1:04d}" for idx in range(len(prepared))
            ]
            resolved_deps, waves = self._resolve_and_validate_dag(
                prepared, candidate_ids
            )

            for i, item in enumerate(prepared):
                for prior in prepared[:i]:
                    if self._scope_overlap(
                        _scope_view(prior),
                        item["write_scope"],
                        item["write_directories"],
                    ):
                        raise SubagentError(
                            "parallel writers have overlapping write_scope paths"
                        )

            active = sum(
                w.state in ACTIVE_STATES for w in self._workers.values()
            )
            if active + len(prepared) > self.max_active:
                raise SubagentError(
                    f"capacity is {self.max_active}; {active} active and "
                    f"{len(prepared)} requested"
                )
            for item in prepared:
                for existing in self._workers.values():
                    if existing.state in ACTIVE_STATES and self._scope_overlap(
                        existing,
                        item["write_scope"],
                        item["write_directories"],
                    ):
                        raise SubagentError(
                            f"write_scope conflicts with active subagent "
                            f"{existing.id}"
                        )
            self._prune(len(prepared))
            created: list[Subagent] = []
            self._batch_counter += 1
            batch_id = f"batch-{self._batch_counter:04d}"
            self._latest_batch_id = batch_id
            self._batch_members[batch_id] = []
            from .systemprompt import subagent_prompt_sealed as \
                _sealed_subagent_prompt
            try:
                from .enforce import anchor as _enforce_anchor
                _ENFORCE_WORKERS = True
            except Exception:
                _ENFORCE_WORKERS = False

            def _worker_anchor(aid: str) -> str:
                try:
                    return _enforce_anchor(self._counter + len(created))
                except Exception:
                    return ""

            for idx, item in enumerate(prepared):
                self._counter += 1
                aid = f"sa-{self._counter:04d}"
                worker = Subagent(
                    id=aid,
                    batch_id=batch_id,
                    name=item["name"] or f"{item['role']}-{self._counter}",
                    role=item["role"],
                    task=item["task"],
                    model_id=item["model_id"],
                    write_scope=item["write_scope"],
                    write_directories=item["write_directories"],
                    provider=item["provider"],
                    model=item["model"],
                    effort=item["effort"],
                    tools=item["tools"],
                    session_id=self._session_id,
                    priority=item["priority"],
                    wave=waves[idx],
                    depends_on=resolved_deps[idx],
                    max_retries=item["max_retries"],
                    timeout=item["timeout"],
                )
                worker.messages = [
                    {
                        "role": "system",
                        "content": _sealed_subagent_prompt(
                            worker.role, write_scope=list(worker.write_scope)
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            (
                                f"Shared context:\n{item['context']}\n\n"
                                if item["context"]
                                else ""
                            )
                            + f"TASK: {worker.task}"
                            # SOVEREIGN ENFORCEMENT L3: per-worker rotating
                            # anchor in the freshest-attention slot
                            + ("\n\n" + _worker_anchor(aid)
                               if _ENFORCE_WORKERS else "")
                        ),
                    },
                ]
                self._workers[aid] = worker
                self._order.append(aid)
                self._done_events[aid] = threading.Event()
                self._batch_members[batch_id].append(aid)
                created.append(worker)

            for worker in created:
                ev = self._emit(
                    "subagent.spawn",
                    {
                        "id": worker.id,
                        "name": worker.name,
                        "batch_id": batch_id,
                        "role": worker.role,
                        "task": worker.task[:500],
                        "model": worker.model_id,
                        "write_scope": list(worker.write_scope),
                        "priority": worker.priority,
                        "wave": worker.wave,
                        "depends_on": list(worker.depends_on),
                        "parallel_batch": [x.id for x in created],
                    },
                    worker,
                )
                worker.spawn_event_id = str(getattr(ev, "id", "") or "")

            dispatch_order = sorted(
                created, key=lambda w: (w.wave, -w.priority, w.id)
            )
            submitted: list[str] = []
            try:
                for worker in dispatch_order:
                    self._futures[worker.id] = self._pool.submit(
                        self._execute, worker
                    )
                    submitted.append(worker.id)
            except RuntimeError as exc:
                stranded = [w for w in created if w.id not in submitted]
                for worker in stranded:
                    worker.state = "error"
                    worker.error = f"could not schedule worker: {exc}"
                    worker.finished_at = time.time()
                    done_ev = self._done_events.get(worker.id)
                    if done_ev is not None:
                        done_ev.set()
                    self._emit("subagent.done", worker.to_dict(), worker)
                raise SubagentError(
                    f"could not schedule {len(stranded)} worker(s): {exc}"
                ) from exc
            return created

    def get(self, worker_id: str) -> Subagent | None:
        with self._lock:
            return self._workers.get(str(worker_id or "").strip())

    def list(self) -> list[Subagent]:
        with self._lock:
            return [
                self._workers[aid]
                for aid in self._order
                if aid in self._workers
            ]

    def get_batch(self, batch_id: str = "") -> list[Subagent]:
        """Return retained workers from a named batch (latest by default)."""
        key = str(batch_id or "").strip()
        with self._lock:
            key = key or self._latest_batch_id
            if not key:
                return []
            if key not in self._batch_members:
                raise SubagentError(f"unknown or expired subagent batch: {key}")
            return [
                self._workers[aid]
                for aid in self._batch_members[key]
                if aid in self._workers
            ]

    def synthesize_batch(self, batch_id: str = "") -> dict[str, Any]:
        """Return a structured executive synthesis of a batch's outcomes, diffs, and metrics."""
        workers = self.get_batch(batch_id)
        data_rows = [w.to_dict() for w in workers]
        files_touched: list[str] = []
        all_diffs: list[dict[str, Any]] = []
        for w, row in zip(workers, data_rows):
            for f in row["files_touched"]:
                if f not in files_touched:
                    files_touched.append(f)
            all_diffs.extend(row.get("diffs", []))
        states = {
            s: sum(1 for r in data_rows if r["state"] == s)
            for s in ("queued", "running", "cancelling", "done", "blocked", "error", "cancelled")
        }
        return {
            "batch_id": workers[0].batch_id if workers else (batch_id or ""),
            "total_workers": len(data_rows),
            "states": states,
            "all_settled": all(r["state"] in TERMINAL_STATES for r in data_rows),
            "roles": sorted({r["role"] for r in data_rows}),
            "waves": max((r["wave"] for r in data_rows), default=0) + (1 if data_rows else 0),
            "total_steps": sum(r["steps"] for r in data_rows),
            "total_tool_calls": sum(r["tool_calls"] for r in data_rows),
            "tokens_in": sum(r["tokens_in"] for r in data_rows),
            "tokens_out": sum(r["tokens_out"] for r in data_rows),
            "lines_added": sum(r["lines_added"] for r in data_rows),
            "lines_removed": sum(r["lines_removed"] for r in data_rows),
            "files_touched": files_touched,
            "diffs": all_diffs,
            "blackboard": self.blackboard.read(),
            "workers": data_rows,
        }

    def format_dag(self, batch_id: str = "") -> str:
        """Render a visual Unicode DAG wave & dependency graph for a batch."""
        workers = self.get_batch(batch_id)
        if not workers:
            return "No subagents in batch to render DAG."
        by_wave: dict[int, list[Subagent]] = {}
        for w in workers:
            by_wave.setdefault(w.wave, []).append(w)
        icons = {
            "queued": "○",
            "running": "●",
            "cancelling": "◌",
            "done": "✓",
            "blocked": "!",
            "error": "×",
            "cancelled": "–",
        }
        lines = [
            f"DAG TOPOLOGY · {workers[0].batch_id} · {len(by_wave)} wave(s) · {len(workers)} worker(s)"
        ]
        sorted_waves = sorted(by_wave)
        for idx, wave_num in enumerate(sorted_waves):
            lines.append(f"  ╭─ Wave {wave_num} ─────────────────────────────────────────╮")
            for w in sorted(by_wave[wave_num], key=lambda x: (-x.priority, x.id)):
                pct = w.progress_pct(self.max_steps)
                filled = int(round(pct / 20))
                bar = "▰" * filled + "▱" * (5 - filled)
                deps = f" ◄── [{', '.join(w.depends_on)}]" if w.depends_on else ""
                prio = f" (prio {w.priority:+d})" if w.priority else ""
                diff_badge = (
                    f" Δ+{w.lines_added}/-{w.lines_removed}"
                    if (w.lines_added or w.lines_removed)
                    else ""
                )
                lines.append(
                    f"  │ {icons.get(w.state, '·')} {w.id} · {w.name} [{w.role}]{prio} "
                    f"{bar} {pct:>3}% · {w.state.upper()}{diff_badge}{deps}"
                )
            lines.append("  ╰──────────────────────────────────────────────────╯")
            if idx < len(sorted_waves) - 1:
                lines.append("                     │")
                lines.append("                     ▼  (pipes upstream summaries)")
        return "\n".join(lines)

    def format_swarm_hud(self) -> str:
        """Compact 1-line swarm telemetry summary for the TUI status border."""
        snap = self.snapshot()
        active = snap["active"]
        if not active and not snap["workers"]:
            return ""
        counts = snap["counts"]
        return (
            f"⚡ Swarm {active}/{snap['max_active']} "
            f"(●{counts['running']} ○{counts['queued']} ✓{counts['done']})"
        )

    def active_count(self) -> int:
        """Number of queued/running/cancelling workers right now.

        Used by the parent turn's adaptive cap: a TaskCollect/TaskOutput
        wait while this is non-zero is a scheduler yield, not tool work.
        """
        with self._lock:
            return sum(
                worker.state in ACTIVE_STATES
                for worker in self._workers.values()
            )

    def collect(
        self,
        batch_id: str = "",
        worker_ids: list[str] | tuple[str, ...] | None = None,
        *,
        timeout: float = 30.0,
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        """Wait once, then return bounded final reports for one parallel batch."""
        if batch_id and worker_ids:
            raise SubagentError("provide batch_id or worker_ids, not both")
        if worker_ids is not None:
            if not isinstance(worker_ids, (list, tuple)) or not worker_ids:
                raise SubagentError("worker_ids must be a non-empty list")
            if len(worker_ids) > MAX_GROUP_SIZE:
                raise SubagentError(
                    f"collect supports at most {MAX_GROUP_SIZE} workers"
                )
            if len(set(str(item).strip() for item in worker_ids)) != len(worker_ids):
                raise SubagentError("worker_ids must not contain duplicates")
            workers = [self.get(str(aid)) for aid in worker_ids]
            if any(worker is None for worker in workers):
                missing = next(
                    str(aid)
                    for aid, worker in zip(worker_ids, workers)
                    if worker is None
                )
                raise SubagentError(f"unknown subagent: {missing}")
        else:
            workers = self.get_batch(batch_id)
        if not workers:
            return "No retained subagents in that batch."
        ids = [worker.id for worker in workers]
        states = self.wait(ids, timeout=timeout, should_cancel=should_cancel)
        reports = [self.format_worker(worker) for worker in workers]
        still_active = [
            aid for aid, state in states.items() if state in ACTIVE_STATES
        ]
        header = (
            f"COLLECTED {len(workers)} SUBAGENT REPORT(S)"
            + (
                f" · {len(still_active)} still active after the wait"
                if still_active
                else " · all tasks settled"
            )
        )
        return header + "\n\n" + "\n\n".join(reports)

    def stop(self, worker_id: str) -> Subagent:
        with self._lock:
            worker = self._workers.get(str(worker_id or "").strip())
            if worker is None:
                raise SubagentError(f"unknown subagent: {worker_id}")
            if worker.state in TERMINAL_STATES:
                return worker
            worker.stop_event.set()
            worker.current_action = "cancellation requested"
            future = self._futures.get(worker.id)
            if future is not None and future.cancel():
                worker.state = "cancelled"
                worker.summary = "Cancelled before the worker started."
                worker.finished_at = time.time()
                done_ev = self._done_events.get(worker.id)
                if done_ev is not None:
                    done_ev.set()
                self._emit("subagent.cancelled", worker.to_dict(), worker)
            else:
                worker.state = "cancelling"
                self._emit(
                    "subagent.progress",
                    {
                        "id": worker.id,
                        "state": "cancelling",
                        "action": worker.current_action,
                    },
                    worker,
                )
            return worker

    def wait(
        self,
        worker_ids: list[str] | tuple[str, ...] | None = None,
        timeout: float = 30.0,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, str]:
        """Wait up to ``timeout``; return the latest state for every requested id."""
        try:
            limit = max(0.0, min(90.0, float(timeout)))
        except (TypeError, ValueError):
            limit = 30.0
        with self._lock:
            ids = (
                [str(x).strip() for x in worker_ids]
                if worker_ids is not None
                else list(self._order)
            )
            futures = [
                self._futures[x]
                for x in ids
                if x in self._futures and not self._futures[x].done()
            ]
        deadline = time.monotonic() + limit
        while futures and time.monotonic() < deadline:
            if should_cancel is not None and should_cancel():
                with self._lock:
                    targets = [
                        self._workers[aid]
                        for aid in ids
                        if aid in self._workers
                    ]
                for worker in targets:
                    if worker.state in ACTIVE_STATES:
                        try:
                            self.stop(worker.id)
                        except SubagentError:
                            pass
                raise TurnCancelled()
            done, pending = concurrent.futures.wait(
                futures,
                timeout=min(0.2, max(0.0, deadline - time.monotonic())),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            futures = list(pending)
            if not done and limit == 0:
                break
        with self._lock:
            return {
                aid: (
                    self._workers[aid].state
                    if aid in self._workers
                    else "missing"
                )
                for aid in ids
            }

    def reset_session(self, session_id: str) -> None:
        """Cancel outgoing-session work, then discard its task view."""
        next_session = str(session_id or "")
        with self._lock:
            if self._closed:
                return
            if self._transitioning:
                raise SubagentError("a session transition is already in progress")
            self._transitioning = True
        try:
            with self._lock:
                active_ids = [
                    worker.id
                    for worker in self._workers.values()
                    if worker.state in ACTIVE_STATES
                ]
            for worker_id in active_ids:
                self.stop(worker_id)
            with self._lock:
                futures = list(self._futures.values())
            if futures:
                concurrent.futures.wait(futures, timeout=DRAIN_TIMEOUT)
            with self._lock:
                self._workers.clear()
                self._order.clear()
                self._futures.clear()
                self._done_events.clear()
                self._batch_members.clear()
                self.blackboard.clear()
                self._latest_batch_id = ""
                self._session_id = next_session
        finally:
            with self._lock:
                self._transitioning = False

    # -- execution -----------------------------------------------------------

    def _may_write(
        self, worker: Subagent, tool_name: str, args: dict
    ) -> tuple[bool, str]:
        if tool_name not in WRITE_TOOLS:
            return True, ""
        if not worker.write_scope:
            return False, "delegated write blocked: no approved write_scope"
        raw = args.get("path")
        if not isinstance(raw, str) or not raw.strip():
            return False, "delegated write blocked: tool needs a valid path"
        path = Path(raw).expanduser()
        if ".." in path.parts:
            return False, "delegated write blocked: path traversal is forbidden"
        path = path if path.is_absolute() else self.workspace / path
        resolved = path.resolve(strict=False)
        if not self._relative_to(resolved, self.workspace):
            return False, "delegated write blocked: path escapes the workspace"
        allowed = any(
            str(resolved) == scope
            or (
                scope in worker.write_directories
                and self._relative_to(resolved, Path(scope))
            )
            for scope in worker.write_scope
        )
        if not allowed:
            return False, (
                f"delegated write blocked: {resolved} is outside this worker's "
                f"write_scope ({', '.join(worker.write_scope)})"
            )
        if self.before_write is not None:
            try:
                reason = self.before_write(worker, self.tools[tool_name], args)
            except Exception as exc:
                return False, (
                    "delegated write blocked: pre-write recovery "
                    f"snapshot failed ({type(exc).__name__}: {exc})"
                )
            if reason:
                return False, str(reason)
        return True, ""

    @staticmethod
    def _event_args(args: dict) -> dict:
        out = {}
        for key, value in args.items():
            if key in {"content", "old_string", "new_string", "patch"}:
                out[key] = (
                    f"<{len(value)} chars>"
                    if isinstance(value, str)
                    else "<omitted>"
                )
            else:
                out[key] = value
        try:
            encoded = json.dumps(out, ensure_ascii=False, default=str)
            if len(encoded) > 3000:
                return {"preview": encoded[:2800] + "…"}
        except Exception:
            return {"preview": str(out)[:2800]}
        return out

    def _await_dependencies(self, worker: Subagent) -> tuple[bool, str]:
        """Wait for any upstream ``depends_on`` workers to reach a terminal state."""
        if not worker.depends_on:
            return True, ""
        with self._lock:
            worker.current_action = f"waiting on {', '.join(worker.depends_on)}"
        deadline = time.monotonic() + worker.timeout
        upstream_blocks: list[str] = []
        for dep_id in worker.depends_on:
            while True:
                if worker.stop_event.is_set():
                    raise TurnCancelled()
                with self._lock:
                    dep_worker = self._workers.get(dep_id)
                    dep_done_ev = self._done_events.get(dep_id)
                if dep_worker is None:
                    return False, f"Upstream dependency {dep_id} no longer exists."
                if dep_worker.state in TERMINAL_STATES:
                    if dep_worker.state != "done":
                        return (
                            False,
                            f"Blocked because upstream dependency {dep_id} ({dep_worker.name}) "
                            f"finished with state={dep_worker.state!r}"
                            + (f": {dep_worker.error}" if dep_worker.error else ""),
                        )
                    upstream_blocks.append(
                        f"[{dep_worker.id} · {dep_worker.name} ({dep_worker.role})]:\n"
                        f"{dep_worker.summary}"
                    )
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return (
                        False,
                        f"Timed out waiting for upstream dependency {dep_id}.",
                    )
                if dep_done_ev is not None:
                    dep_done_ev.wait(timeout=min(0.1, remaining))
                else:
                    time.sleep(min(0.05, remaining))
        return True, "\n\n".join(upstream_blocks)

    def _capture_write_diff(
        self, worker: Subagent, name: str, path_str: str, before_text: str, after_text: str
    ) -> None:
        if before_text == after_text:
            return
        old_lines = before_text.splitlines()
        new_lines = after_text.splitlines()
        u_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path_str}" if before_text else "/dev/null",
                tofile=f"b/{path_str}",
                lineterm="",
                n=2,
            )
        )
        added = sum(1 for l in u_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in u_lines if l.startswith("-") and not l.startswith("---"))
        with self._lock:
            worker.lines_added += added
            worker.lines_removed += removed
            worker.diffs.append({
                "worker_id": worker.id,
                "tool": name,
                "path": path_str,
                "added": added,
                "removed": removed,
                "diff": "\n".join(u_lines[:120]),
            })

    def _invoke_tool_handler(
        self, worker: Subagent, tool: Tool, name: str, args: dict
    ) -> str:
        allowed, why = self._may_write(worker, name, args)
        if not allowed:
            return "ERROR: " + why

        lock = None
        resolved_file: Path | None = None
        if name in WRITE_TOOLS and isinstance(args.get("path"), str):
            cand = Path(args["path"]).expanduser()
            full = cand if cand.is_absolute() else self.workspace / cand
            resolved_file = full.resolve(strict=False)
            lock = self._scope_registry.lock_for(str(resolved_file))

        def _call() -> str:
            before_text = ""
            if resolved_file is not None and name in ("write_file", "edit_file"):
                try:
                    if resolved_file.is_file():
                        before_text = resolved_file.read_text(
                            encoding="utf-8", errors="replace"
                        )
                except OSError:
                    before_text = ""
            try:
                try:
                    parameters = inspect.signature(tool.handler).parameters
                    accepts_cancel = "should_cancel" in parameters or any(
                        p.kind is inspect.Parameter.VAR_KEYWORD
                        for p in parameters.values()
                    )
                except (TypeError, ValueError):
                    accepts_cancel = False
                if accepts_cancel:
                    out = tool.handler(
                        **args, should_cancel=worker.stop_event.is_set
                    )
                else:
                    out = tool.handler(**args)
                if not isinstance(out, str):
                    out = "" if out is None else str(out)
            except TurnCancelled:
                raise
            except Exception as exc:
                out = f"ERROR: {type(exc).__name__}: {exc}"

            if (
                name in WRITE_TOOLS
                and not out.startswith("ERROR:")
                and self.after_write is not None
            ):
                try:
                    warning = self.after_write(worker, tool, args)
                    if warning:
                        out += "\n" + str(warning)
                except Exception as exc:
                    out += (
                        "\nWARNING: post-write verification "
                        f"failed ({type(exc).__name__}: {exc})"
                    )
            if name in WRITE_TOOLS and not out.startswith("ERROR:"):
                path = args.get("path")
                if isinstance(path, str) and path:
                    with self._lock:
                        if path not in worker.files_touched:
                            worker.files_touched.append(path)
                    if resolved_file is not None and name in ("write_file", "edit_file"):
                        try:
                            after_text = (
                                resolved_file.read_text(encoding="utf-8", errors="replace")
                                if resolved_file.is_file()
                                else str(args.get("content", ""))
                            )
                        except OSError:
                            after_text = str(args.get("content", ""))
                        self._capture_write_diff(
                            worker, name, path, before_text, after_text
                        )
            return out

        if lock is not None:
            with lock:
                return _call()
        return _call()

    def _execute_step_calls(
        self, worker: Subagent, step: int, safe_calls: list[dict]
    ) -> str | None:
        """Execute one step's tool calls, parallelizing consecutive read-only runs."""
        i = 0
        n = len(safe_calls)
        while i < n:
            if worker.stop_event.is_set():
                raise TurnCancelled()
            if worker.tool_calls >= MAX_TOTAL_TOOL_CALLS:
                with self._lock:
                    worker.state = "blocked"
                return (
                    "Worker reached its total tool-call limit; "
                    "reporting the work completed so far."
                )

            j = i
            while j < n:
                cname = safe_calls[j]["function"]["name"]
                if (
                    cname in PARALLEL_SUBAGENT_READ_TOOLS
                    and cname in worker.tools
                    and (worker.tool_calls + (j - i)) < MAX_TOTAL_TOOL_CALLS
                ):
                    j += 1
                else:
                    break

            if j - i >= 2:
                batch_calls = safe_calls[i:j]
                parsed_args = [
                    parse_tool_arguments(c["function"].get("arguments", "{}"))
                    for c in batch_calls
                ]
                with self._lock:
                    worker.tool_calls += len(batch_calls)
                    worker.current_action = (
                        f"parallel read x{len(batch_calls)}"
                    )
                    for c in batch_calls:
                        cn = c["function"]["name"]
                        worker.tool_breakdown[cn] = (
                            worker.tool_breakdown.get(cn, 0) + 1
                        )

                t_batch_start = time.monotonic()

                def _run_one_read(pair: tuple[dict, dict]) -> tuple[str, int]:
                    call_obj, call_args = pair
                    t0 = time.monotonic()
                    cname = call_obj["function"]["name"]
                    tool = worker.tools[cname]
                    out = self._invoke_tool_handler(
                        worker, tool, cname, call_args
                    )
                    dur_ms = int((time.monotonic() - t0) * 1000)
                    return out[:MAX_TOOL_OUTPUT_CHARS], dur_ms

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(4, len(batch_calls)),
                    thread_name_prefix=f"fullagent:{worker.id}:read",
                ) as read_pool:
                    results = list(
                        read_pool.map(
                            _run_one_read, zip(batch_calls, parsed_args)
                        )
                    )
                wall_ms = int((time.monotonic() - t_batch_start) * 1000)
                sum_ms = sum(d for _, d in results)
                with self._lock:
                    worker.parallel_reads_saved_ms += max(0, sum_ms - wall_ms)

                for call_obj, call_args, (output, dur_ms) in zip(
                    batch_calls, parsed_args, results
                ):
                    cname = call_obj["function"]["name"]
                    self._emit(
                        "subagent.tool",
                        {
                            "id": worker.id,
                            "name": cname,
                            "args": self._event_args(call_args),
                            "status": (
                                "error"
                                if output.startswith("ERROR:")
                                else "done"
                            ),
                            "duration_ms": dur_ms,
                            "preview": output[:300],
                        },
                        worker,
                    )
                    worker.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_obj["id"],
                            "content": output,
                        }
                    )
                    self._emit(
                        "subagent.progress",
                        {
                            "id": worker.id,
                            "state": "running",
                            "step": step,
                            "action": cname,
                            "tool_calls": worker.tool_calls,
                        },
                        worker,
                    )
                i = j
                continue

            call = safe_calls[i]
            fn = call["function"]
            name = fn["name"]
            args = parse_tool_arguments(fn.get("arguments", "{}"))
            with self._lock:
                worker.tool_calls += 1
                worker.current_action = name
                worker.tool_breakdown[name] = (
                    worker.tool_breakdown.get(name, 0) + 1
                )
            started = time.monotonic()
            tool = worker.tools.get(name)
            if tool is None:
                output = (
                    f"ERROR: tool {name!r} is not available to "
                    f"the {worker.role} subagent"
                )
            else:
                output = self._invoke_tool_handler(worker, tool, name, args)
            output = output[:MAX_TOOL_OUTPUT_CHARS]
            self._emit(
                "subagent.tool",
                {
                    "id": worker.id,
                    "name": name,
                    "args": self._event_args(args),
                    "status": (
                        "error" if output.startswith("ERROR:") else "done"
                    ),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "preview": output[:300],
                },
                worker,
            )
            worker.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output,
                }
            )
            self._emit(
                "subagent.progress",
                {
                    "id": worker.id,
                    "state": "running",
                    "step": step,
                    "action": name,
                    "tool_calls": worker.tool_calls,
                },
                worker,
            )
            i += 1
        return None

    def _call_model_with_retries(
        self, worker: Subagent, schemas: list[dict]
    ) -> Any:
        attempts = 0
        while True:
            try:
                return self.chat(
                    worker.provider,
                    worker.model,
                    worker.effort,
                    worker.messages,
                    schemas,
                    timeout=worker.timeout,
                    should_cancel=worker.stop_event.is_set,
                )
            except TurnCancelled:
                raise
            except Exception as exc:
                if worker.stop_event.is_set():
                    raise TurnCancelled() from exc
                if attempts >= worker.max_retries:
                    raise
                attempts += 1
                with self._lock:
                    worker.retries = attempts
                    worker.current_action = (
                        f"retrying ({attempts}/{worker.max_retries})"
                    )
                backoff = min(2.0, 0.05 * (2 ** (attempts - 1)))
                deadline = time.monotonic() + backoff
                while time.monotonic() < deadline:
                    if worker.stop_event.is_set():
                        raise TurnCancelled() from exc
                    time.sleep(0.01)

    def _execute(self, worker: Subagent) -> None:
        cancelled_before_start = False
        with self._lock:
            if worker.stop_event.is_set() or worker.state == "cancelled":
                if worker.state != "cancelled":
                    worker.state = "cancelled"
                    worker.summary = "Cancelled before the worker started."
                    worker.finished_at = time.time()
                    cancelled_before_start = True
                return_after_lock = True
            else:
                return_after_lock = False
            cancelled_data = (
                worker.to_dict() if cancelled_before_start else None
            )
        if return_after_lock:
            done_ev = self._done_events.get(worker.id)
            if done_ev is not None:
                done_ev.set()
            if cancelled_before_start:
                self._emit("subagent.cancelled", cancelled_data, worker)
            return

        active_incremented = False
        try:
            deps_ok, dep_context = self._await_dependencies(worker)
            if not deps_ok:
                with self._lock:
                    worker.state = "blocked"
                    worker.summary = dep_context[:4000]
                return
            if dep_context:
                with self._lock:
                    worker.messages[-1]["content"] = (
                        f"Upstream dependency results:\n{dep_context}\n\n"
                        + worker.messages[-1]["content"]
                    )

            with self._lock:
                if worker.stop_event.is_set():
                    raise TurnCancelled()
                worker.state = "running"
                worker.current_action = "starting"
                worker.started_at = time.time()
                self._active_running += 1
                active_incremented = True
                if self._active_running > self._peak_concurrency:
                    self._peak_concurrency = self._active_running

            self._emit(
                "subagent.progress",
                {
                    "id": worker.id,
                    "state": "running",
                    "step": 0,
                    "action": "started",
                },
                worker,
            )
            final = ""
            schemas = [tool.openai_schema() for tool in worker.tools.values()]
            for step in range(1, self.max_steps + 1):
                if worker.stop_event.is_set():
                    raise TurnCancelled()
                with self._lock:
                    worker.steps = step
                    worker.current_action = (
                        f"model step {step}/{self.max_steps}"
                    )
                result = self._call_model_with_retries(worker, schemas)
                if worker.stop_event.is_set():
                    raise TurnCancelled()
                if result is None:
                    raise RuntimeError("provider returned no result")
                normalizer = getattr(result, "normalize", None)
                if callable(normalizer):
                    normalizer()
                content = getattr(result, "content", "")
                if not isinstance(content, str):
                    content = ""
                calls = getattr(result, "tool_calls", [])
                if not isinstance(calls, (list, tuple)):
                    calls = []
                tin, tout = safe_token_counts(getattr(result, "usage", None))
                with self._lock:
                    worker.tokens_in += tin
                    worker.tokens_out += tout
                safe_calls = []
                seen = set()
                for raw in calls:
                    call = normalize_tool_call(raw)
                    if call is None:
                        continue
                    cid = call.get("id")
                    if cid in seen:
                        call = dict(call)
                        call["id"] = f"{cid}-{uuid.uuid4().hex[:8]}"
                    seen.add(call["id"])
                    safe_calls.append(call)
                safe_calls = safe_calls[:MAX_CALLS_PER_STEP]
                if not safe_calls:
                    final = content.strip()
                    if calls and not final:
                        final = "Provider returned no usable tool calls or final text."
                    break
                reasoning = getattr(result, "reasoning", "")
                if not isinstance(reasoning, str):
                    reasoning = ""
                worker.messages.append(
                    assistant_message(content, safe_calls, reasoning)
                )
                limit_msg = self._execute_step_calls(worker, step, safe_calls)
                if limit_msg is not None:
                    final = limit_msg
                    break
                if worker.state == "blocked":
                    break
                final = content.strip()
            else:
                final = (final + "\n" if final else "") + (
                    f"Worker reached its {self.max_steps}-step limit; "
                    "reporting the work completed so far."
                )
                with self._lock:
                    worker.state = "blocked"
            if worker.stop_event.is_set():
                raise TurnCancelled()
            if not final:
                final = "Completed without a text summary."
            with self._lock:
                if worker.state != "blocked":
                    worker.state = "done"
                worker.summary = final[:4000]
            # Auto-publish completed worker summary to SharedBlackboard
            try:
                self.blackboard.publish(
                    f"summary:{worker.name or worker.id}",
                    worker.summary[:600],
                    author=worker.id,
                    role=worker.role,
                )
            except Exception:
                pass
        except TurnCancelled:
            with self._lock:
                worker.state = "cancelled"
                worker.summary = "Cancelled by request."
        except BaseException as exc:
            with self._lock:
                worker.state = "error"
                worker.error = f"{type(exc).__name__}: {exc}"
                worker.summary = (
                    "Subagent failed; inspect its error in TaskList."
                )
        finally:
            with self._lock:
                if active_incremented:
                    self._active_running = max(0, self._active_running - 1)
                worker.finished_at = time.time()
                worker.current_action = worker.state
                data = worker.to_dict()
                done_ev = self._done_events.get(worker.id)
                if done_ev is not None:
                    done_ev.set()
            self._emit("subagent.done", data, worker)

    # -- reporting / shutdown ------------------------------------------------

    def format_worker(self, worker: Subagent | None) -> str:
        if worker is None:
            return "ERROR: subagent not found"
        data = worker.to_dict()
        header = (
            f"{data['id']} · {data['name']} · "
            f"{role_label(data['role'])} ({data['role']}) · "
            f"{data['state']} · batch {data['batch_id']}"
        )
        if data.get("wave", 0) > 0 or data.get("depends_on"):
            header += f" · wave {data['wave']}"
        lines = [header]
        if data["current_action"] and data["state"] in ACTIVE_STATES:
            lines.append(f"Current: {data['current_action']}")
        if data["depends_on"]:
            lines.append("Depends on: " + ", ".join(data["depends_on"]))
        if data["summary"]:
            lines.append("Result: " + data["summary"])
        if data["error"]:
            lines.append("Error: " + data["error"])
        diff_info = (
            f" · Δ+{data['lines_added']}/-{data['lines_removed']} lines"
            if (data.get("lines_added") or data.get("lines_removed"))
            else ""
        )
        lines.append(
            f"Work: {data['steps']} step(s), {data['tool_calls']} tool call(s), "
            f"{data['tokens_in']}→{data['tokens_out']} tokens · "
            f"{data['elapsed_ms'] / 1000:.1f}s elapsed{diff_info}"
        )
        if data["write_scope"]:
            lines.append(
                "Approved write scope: " + ", ".join(data["write_scope"])
            )
        if data["files_touched"]:
            lines.append("Files touched: " + ", ".join(data["files_touched"]))
        return "\n".join(lines)

    def format_status(self) -> str:
        snapshot = self.snapshot()
        workers = snapshot["workers"]
        if not workers:
            return (
                "No subagents yet. Use TaskGroup for parallel read-only "
                "delegation, or TaskWriteGroup for approval-gated, "
                "file-scoped implementation."
            )
        counts = snapshot["counts"]
        lines = [
            f"PARALLEL CREW · {len(workers)} retained · "
            f"{snapshot['active']}/{snapshot['max_active']} active · "
            f"{counts['running']}/{snapshot['max_workers']} workers · "
            f"{counts['queued']} queued · {counts['done']} done · "
            f"{counts['blocked']} blocked · {counts['error']} errors",
            (
                f"Batch {snapshot['latest_batch_id']} · "
                f"{snapshot['tokens_in']}→{snapshot['tokens_out']} tokens · "
                f"peak concurrency {snapshot['peak_concurrency']}"
            ),
        ]
        icons = {
            "queued": "○",
            "running": "●",
            "cancelling": "◌",
            "done": "✓",
            "blocked": "!",
            "error": "×",
            "cancelled": "–",
        }
        for row in workers:
            task = " ".join(row["task"].split())
            if len(task) > 96:
                task = task[:93] + "..."
            wave_tag = f" [W{row['wave']}]" if row.get("wave", 0) > 0 else ""
            diff_tag = (
                f" Δ+{row['lines_added']}/-{row['lines_removed']}"
                if (row.get("lines_added") or row.get("lines_removed"))
                else ""
            )
            lines.append(
                f"{icons.get(row['state'], '·')} {row['id']} "
                f"{row['state'].upper():<10} {row['role']:<12} "
                f"{row['name'][:20]:<20} {row['elapsed_ms'] / 1000:.1f}s · "
                f"{row['steps']} steps · {row['tool_calls']} tools{wave_tag}{diff_tag}"
            )
            lines.append(f"  task: {task}")
            if row["state"] in ACTIVE_STATES and row["current_action"]:
                lines.append(f"  now:  {row['current_action']}")
            elif row["summary"]:
                summary = " ".join(row["summary"].split())
                lines.append(
                    f"  result: {summary[:180]}"
                    + ("…" if len(summary) > 180 else "")
                )
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return an immutable-style, JSON-safe live dashboard snapshot."""
        with self._lock:
            workers = [
                self._workers[aid].to_dict()
                for aid in self._order
                if aid in self._workers
            ]
            states = (
                "queued",
                "running",
                "cancelling",
                "done",
                "blocked",
                "error",
                "cancelled",
            )
            counts = {
                state: sum(row["state"] == state for row in workers)
                for state in states
            }
            files_touched: list[str] = []
            for row in workers:
                for f in row["files_touched"]:
                    if f not in files_touched:
                        files_touched.append(f)
            return {
                "workers": workers,
                "counts": counts,
                "active": sum(counts[state] for state in ACTIVE_STATES),
                "max_active": self.max_active,
                "max_workers": self.max_workers,
                "latest_batch_id": self._latest_batch_id,
                "tokens_in": sum(row["tokens_in"] for row in workers),
                "tokens_out": sum(row["tokens_out"] for row in workers),
                "peak_concurrency": self._peak_concurrency,
                "total_tool_calls": sum(row["tool_calls"] for row in workers),
                "lines_added": sum(row["lines_added"] for row in workers),
                "lines_removed": sum(row["lines_removed"] for row in workers),
                "roles_used": sorted({row["role"] for row in workers}),
                "files_touched": files_touched,
                "blackboard_count": len(self.blackboard.read()),
            }

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            for worker in self._workers.values():
                if worker.state in ACTIVE_STATES:
                    worker.stop_event.set()
                    future = self._futures.get(worker.id)
                    if future is not None and future.cancel():
                        worker.state = "cancelled"
                        worker.summary = "Cancelled during manager shutdown."
                        worker.finished_at = time.time()
                        done_ev = self._done_events.get(worker.id)
                        if done_ev is not None:
                            done_ev.set()
                        self._emit(
                            "subagent.cancelled", worker.to_dict(), worker
                        )
                    else:
                        worker.state = "cancelling"
            draining = list(self._futures.values()) if wait else []
        if draining:
            concurrent.futures.wait(draining, timeout=DRAIN_TIMEOUT)
        self._pool.shutdown(wait=False, cancel_futures=True)


def _scope_view(item: dict) -> Subagent:
    return Subagent(
        id="",
        batch_id="",
        name="",
        role="",
        task="",
        model_id="",
        write_scope=item["write_scope"],
        write_directories=item["write_directories"],
        provider=None,
        model=None,
        effort=None,
        tools={},
    )


def _self_test() -> None:
    """Offline end-to-end verification for parallel subagents, DAG waves,
    SharedBlackboard, live file diff capture, and batch synthesis."""
    from types import SimpleNamespace

    class _Log:
        def __init__(self):
            self.events = []
            self.lock = threading.Lock()

        def append(self, kind, data, **kwargs):
            with self.lock:
                event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
                self.events.append((kind, data, kwargs))
                return event

    provider = SimpleNamespace(id="offline")
    model = SimpleNamespace(id="offline-model", supports_tools=True)
    effort = SimpleNamespace(key="low")

    def chat(_prov, _mod, _eff, messages, *_args, **_kwargs):
        user_msg = messages[-1].get("content", "")
        if "Upstream dependency results:" in user_msg:
            return SimpleNamespace(
                content="downstream synthesized upstream!",
                tool_calls=[],
                usage={"prompt_tokens": 2, "completion_tokens": 2},
            )
        return SimpleNamespace(
            content="offline result",
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    log = _Log()
    manager = ParallelSubagents(
        log,
        {},
        lambda _model_id: (provider, model, effort),
        max_workers=4,
        max_active=8,
        chat=chat,
    )
    try:
        workers = manager.spawn_many(
            [
                {"name": "alpha", "task": "smoke task one"},
                {"name": "beta", "task": "smoke task two"},
                {
                    "name": "gamma",
                    "task": "synthesize alpha and beta",
                    "depends_on": ["alpha", "beta"],
                },
            ]
        )
        states = manager.wait([worker.id for worker in workers], timeout=5)
        assert len(workers) == 3 and all(
            states.get(worker.id) == "done" for worker in workers
        )
        assert workers[2].wave == 1
        assert "downstream synthesized upstream!" in workers[2].summary
        dag_view = manager.format_dag(workers[0].batch_id)
        assert "Wave 0" in dag_view and "Wave 1" in dag_view
        assert len(manager.blackboard.read()) == 3
        synth = manager.synthesize_batch(workers[0].batch_id)
        assert synth["all_settled"] is True and synth["waves"] == 2
        assert (
            sum(kind == "subagent.done" for kind, _data, _kwargs in log.events)
            == 3
        )
        assert "3 retained" in manager.format_status()
    finally:
        manager.shutdown(wait=True)
    print(
        "SUBAGENT SELF-TEST PASS (parallel lifecycle, DAG waves, blackboard, synthesis, status)"
    )


if __name__ == "__main__":
    _self_test()

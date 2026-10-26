"""Bounded local reservations and the injected browser lifecycle consumer."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import time

from ..system.local_store import atomic_write_json, file_lock
from .browser_adapter import (
    BrowserContractError, CLAIM_BOUNDARY, LEASE_SCHEMA, LIMITS, TERMINALS, MODES, ROLES, PAGE_SCHEMA,
    acquisition_request, capability_snapshot, digest, number, page_state,
    resolve_handle, text, timestamp, token, closed_list, origin,
)

MAX_RECORDS = 32
MAX_STORE_BYTES = 1024 * 1024
STORE_SCHEMA = "browser_lease_store/v1"


def blocked(reason):
    return {"status": "blocked", "reason": reason, "refresh_required": reason in {
        "stale_state", "missing_handle", "ambiguous_handle", "readback_unknown"}}


class BrowserLeaseStore:
    """One bounded index, no history discovery. Tombstones are never evicted.

    Exhaustion refuses new identities instead of forgetting crash deduplication.
    The caller selects a private OMH root, never a page-derived pathname.
    """
    def __init__(self, home):
        root = Path(home).absolute()
        # Resolve the host-selected parent (macOS /var is a system symlink),
        # never a symlink at the managed root or any of its descendants.
        self.path = root.parent.resolve() / root.name / "runtime" / "browser" / "leases.json"

    def _check_paths(self):
        for path in reversed((self.path, *self.path.parents)):
            if path.is_symlink():
                raise BrowserContractError("unsafe_store")
            if path.exists():
                info = path.stat()
                if path == self.path:
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise BrowserContractError("unsafe_store")
                elif not stat.S_ISDIR(info.st_mode):
                    raise BrowserContractError("unsafe_store")

    @contextmanager
    def transaction(self):
        self._check_paths()
        # Creation is lazy; reuse never rewrites/chmods existing paths.
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = self.path.with_name(".leases.json.lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            info = lock.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise BrowserContractError("unsafe_store")
        else:
            os.close(fd)
        with file_lock(self.path, timeout_seconds=2, private=False) as held:
            if not held["enforced"]:
                raise BrowserContractError("lock_unavailable")
            self._check_paths()
            state = self._read()
            yield state

    def _read(self):
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(MAX_STORE_BYTES + 1)
        except FileNotFoundError:
            return {"schema_version": STORE_SCHEMA, "leases": {}}
        if len(raw) > MAX_STORE_BYTES:
            raise BrowserContractError("store_corrupt")
        try:
            state = json.loads(raw)
            if not isinstance(state, dict) or set(state) != {"schema_version", "leases"} or state["schema_version"] != STORE_SCHEMA:
                raise ValueError("schema")
            rows = state["leases"]
            if not isinstance(rows, dict) or len(rows) > MAX_RECORDS:
                raise ValueError("rows")
            for key, row in rows.items():
                self._validate_row(key, row)
        except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
            raise BrowserContractError("store_corrupt") from exc
        return state

    @staticmethod
    def _validate_row(key, row, *, acquisition=False):
        if not isinstance(row, dict) or row.get("schema_version") != LEASE_SCHEMA or row.get("lease_id") != key or len(key) != 64:
            raise ValueError("row")
        required = {"schema_version", "lease_id", "owner_ref", "adapter_id", "adapter_version", "origins", "actions", "mode",
                    "auth_boundary_ref", "task_ref", "created_at", "expires_at", "status", "reason", "slot", "tabs", "pages",
                    "action_count", "capabilities", "acquisition", "claim_boundary", "reaped"}
        if acquisition:
            required.add("page")
        if set(row) != required or row["status"] not in TERMINALS | {"active", "unknown", "releasing"}:
            raise ValueError("row")
        if row["reason"] not in TERMINALS | {"", "acquisition_unknown", "action_unknown"} or row["claim_boundary"] != CLAIM_BOUNDARY:
            raise ValueError("reason")
        if row["mode"] not in MODES or row["actions"] != closed_list(row["actions"], {"read", "navigate", "click", "type", "press", "upload", "download"}, 7):
            raise ValueError("scope")
        if not isinstance(row["origins"], list) or not 1 <= len(row["origins"]) <= 8 or row["origins"] != sorted(set(origin(o) for o in row["origins"])):
            raise ValueError("origins")
        identity = {k: row[k] for k in ("owner_ref", "adapter_id", "adapter_version", "origins", "actions", "mode", "auth_boundary_ref", "task_ref")}
        if digest(identity) != key:
            raise ValueError("identity")
        token(row["adapter_id"])
        token(row["adapter_version"])
        for field in ("owner_ref", "auth_boundary_ref", "task_ref", "lease_id"):
            if not isinstance(row[field], str) or len(row[field]) != 64 or any(c not in "0123456789abcdef" for c in row[field]):
                raise ValueError("digest")
        timestamp(row["created_at"])
        timestamp(row["expires_at"])
        number(row["slot"], LIMITS["leases"], 0)
        number(row["action_count"], LIMITS["actions"], 0)
        if type(row["reaped"]) is not bool or not isinstance(row["tabs"], list) or len(row["tabs"]) > LIMITS["tabs"]:
            raise ValueError("tabs")
        for tab in row["tabs"]:
            token(tab)
        if not isinstance(row["pages"], dict) or set(row["pages"]) - set(row["tabs"]):
            raise ValueError("pages")
        # Closed nested snapshots are checked against their own projection.
        cap = row["capabilities"]
        if cap is not None:
            rebuilt = capability_snapshot(cap, row, cap["observed_at"])
            if cap != rebuilt:
                raise ValueError("capabilities")
        elif row["status"] == "active":
            raise ValueError("capabilities")
        for tab, page in row["pages"].items():
            page_fields = {"schema_version", "lease_id", "tab_id", "revision", "host_revision", "fingerprint", "observed_origin", "observed_url_digest", "url_redacted", "readback", "elements", "delta"}
            if not isinstance(page, dict) or set(page) != page_fields or page.get("schema_version") != PAGE_SCHEMA or page.get("lease_id") != key or page.get("tab_id") != tab:
                raise ValueError("page")
            number(page["revision"], 2**53)
            number(page["host_revision"], 2**53, 0)
            if len(json.dumps(page).encode()) > LIMITS["state_bytes"] or not isinstance(page["elements"], list) or len(page["elements"]) > LIMITS["elements"]:
                raise ValueError("page")
            if page["readback"] != "observed" or page["url_redacted"] is not True or page["observed_origin"] not in row["origins"]:
                raise ValueError("page")
            for value in (page["fingerprint"], page["observed_url_digest"]):
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("digest")
            delta = page["delta"]
            if not isinstance(delta, dict) or set(delta) != {"from_revision", "added", "removed", "changed"} or type(delta["changed"]) is not bool:
                raise ValueError("delta")
            number(delta["from_revision"], page["revision"], 0)
            number(delta["added"], LIMITS["elements"], 0)
            number(delta["removed"], LIMITS["elements"], 0)
            for element in page["elements"]:
                if not isinstance(element, dict) or set(element) != {"role", "name_digest", "key_digest", "handle"} or element["role"] not in ROLES:
                    raise ValueError("element")
                for field in ("name_digest", "key_digest", "handle"):
                    value = element[field]
                    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError("digest")
                if element["handle"] != digest([key, tab, page["revision"], element["key_digest"]]):
                    raise ValueError("handle")
            if len({e["key_digest"] for e in page["elements"]}) != len(page["elements"]):
                raise ValueError("element")
        if row["status"] == "active" and (not row["tabs"] or set(row["pages"]) != set(row["tabs"])):
            raise ValueError("active_pages")
        cached = row["acquisition"]
        if acquisition:
            if cached is not None or row["status"] != "active" or row["page"] != row["pages"].get(row["tabs"][0]):
                raise ValueError("acquisition")
        elif cached is not None:
            BrowserLeaseStore._validate_row(key, cached, acquisition=True)
            if cached["capabilities"] != cap:
                raise ValueError("acquisition")
        elif row["status"] == "active":
            raise ValueError("acquisition")

    def save(self, state):
        if len(state["leases"]) > MAX_RECORDS or len(json.dumps(state, indent=2, sort_keys=True).encode()) + 1 > MAX_STORE_BYTES:
            raise BrowserContractError("metadata_capped")
        self._check_paths()
        atomic_write_json(self.path, state, private=True)


class BrowserSessionManager:
    def __init__(self, store, adapter, *, clock=time.time):
        self.store, self.adapter, self.clock = store, adapter, clock
        self._inflight = set()


    def _view(self, row):
        return deepcopy({**row, "acquisition": None, "page": row["pages"].get(row["tabs"][0]) if row["tabs"] else None})

    def acquire(self, owner, request):
        identity = acquisition_request(owner, self.adapter, request)
        lease_id = identity["lease_id"]
        now = timestamp(self.clock())
        with self.store.transaction() as state:
            existing = state["leases"].get(lease_id)
            if existing:
                if existing["status"] == "active" and now < existing["expires_at"] and now < existing["capabilities"]["expires_at"]:
                    return deepcopy(existing["acquisition"])
                if existing["status"] != "active":
                    if lease_id in self._inflight:
                        return {"status": "pending", "lease_id": lease_id}
                    return self._view(existing)
                expired = "expired" if now >= existing["expires_at"] else "capability_expired"
            else:
                expired = None
                if len(state["leases"]) >= MAX_RECORDS:
                    return blocked("metadata_capped")
                live = [r for r in state["leases"].values() if not r["reaped"]]
                cap = min([LIMITS["leases"], *[r["capabilities"]["limits"]["leases"] for r in live if r["capabilities"]]])
                if len(live) >= cap:
                    return blocked("concurrency_capped")
                slots = {r["slot"] for r in live}
                row = {**identity, "schema_version": LEASE_SCHEMA, "created_at": now,
                       "expires_at": now + LIMITS["ttl_seconds"], "status": "unknown", "reason": "acquisition_unknown",
                       "slot": next(s for s in range(LIMITS["leases"]) if s not in slots),
                       "tabs": [], "pages": {}, "action_count": 0, "capabilities": None,
                       "acquisition": None, "claim_boundary": CLAIM_BOUNDARY, "reaped": False}
                state["leases"][lease_id] = row
                self.store.save(state)  # durable unknown BEFORE any host callback
                self._inflight.add(lease_id)
        if expired:
            self.release(owner, lease_id, terminal=expired)
            return blocked(expired)
        try:
            snapshot = capability_snapshot(self.adapter.capabilities(), identity, self.clock())
            with self.store.transaction() as state:
                row = state["leases"][lease_id]
                if row["status"] != "unknown":
                    return self._view(row)
                row["capabilities"] = snapshot
                row["expires_at"] = min(row["expires_at"], now + snapshot["limits"]["ttl_seconds"])
                self.store.save(state)
                if self.clock() >= row["expires_at"]:
                    raise BrowserContractError("deadline_exceeded")
                if sum(not r["reaped"] for r in state["leases"].values()) > snapshot["limits"]["leases"]:
                    raise BrowserContractError("concurrency_capped")
            started = self.adapter.start(lease_id, deepcopy(identity), row["expires_at"])
            if not isinstance(started, dict) or not isinstance(started.get("tabs"), list) or not 1 <= len(started["tabs"]) <= snapshot["limits"]["tabs"]:
                raise BrowserContractError("tab_capped")
            tabs = [token(t) for t in started["tabs"]]
            if len(set(tabs)) != len(tabs):
                raise BrowserContractError("tab_capped")
            row["tabs"] = tabs
            row["pages"] = {tab: page_state(self.adapter.observe(lease_id, tab, row["expires_at"]), row, tab) for tab in tabs}
            if self.clock() >= min(row["expires_at"], snapshot["expires_at"]):
                raise BrowserContractError("deadline_exceeded")
            with self.store.transaction() as state:
                if state["leases"][lease_id]["status"] != "unknown":
                    return self._view(state["leases"][lease_id])
                row["status"], row["reason"] = "active", ""
                row["acquisition"] = self._view(row)
                state["leases"][lease_id] = row
                self.store.save(state)
                return deepcopy(row["acquisition"])
        except BrowserContractError as exc:
            terminal = str(exc) if str(exc) in TERMINALS else "capability_invalid"
            self.release(owner, lease_id, terminal=terminal)
            return blocked(str(exc))
        finally:
            self._inflight.discard(lease_id)

    def _owned(self, state, owner, lease_id):
        row = state["leases"].get(token(lease_id, 64))
        if row is None or row["owner_ref"] != digest(text(owner, 256)):
            raise BrowserContractError("foreign_lease")
        if row["adapter_id"] != self.adapter.adapter_id or row["adapter_version"] != self.adapter.adapter_version:
            raise BrowserContractError("foreign_adapter")
        return row

    def _reserve_action(self, owner, request):
        lease_id = request.get("lease_id")
        with self.store.transaction() as state:
            row = self._owned(state, owner, lease_id)
            reason = self._unusable(row)
            if reason:
                raise BrowserContractError(reason)
            tab = request.get("tab_id")
            if tab not in row["tabs"]:
                raise BrowserContractError("foreign_tab")
            page = row["pages"][tab]
            operation = request["operation"]
            element = None
            if operation == "act":
                if request.get("action") != "read" or "read" not in row["actions"]:
                    raise BrowserContractError("adapter_cannot_intercept")
                element = resolve_handle(page, request)
            row["status"], row["reason"] = "unknown", "action_unknown"
            row["action_count"] += 1
            self.store.save(state)
            return row, tab, page, element

    @contextmanager
    def effect_boundary(self, owner, task):
        """Serialize held effects with this same lease store's lifecycle/caps.

        Callbacks are transaction-local; never retain them after scope exit.
        Only the engine's new, approved durable attempt reserves an action.
        Replays and previews do not charge. A crash cannot refund a reservation.
        """
        with self.store.transaction() as state:
            def owned(request_owner, lease_id):
                if request_owner != owner:
                    raise BrowserContractError("foreign_owner")
                row = self._owned(state, owner, lease_id)
                if row["task_ref"] != digest(text(task, 256)):
                    raise BrowserContractError("foreign_task")
                return row

            def current(request_owner, lease_id):
                return self._view(owned(request_owner, lease_id))

            def reserve(request_owner, request):
                row = owned(request_owner, request["lease_id"])
                reason = self._unusable(row)
                if reason:
                    raise BrowserContractError(reason)
                needed = "upload" if request["operation"] == "upload" else "click"
                if needed not in row["actions"] or needed in row["capabilities"]["unsupported"]:
                    raise BrowserContractError("action_out_of_scope")
                resolve_handle(row["pages"][request["tab_id"]], request)
                state["leases"][row["lease_id"]]["action_count"] += 1
                self.store.save(state)
                return current(request_owner, row["lease_id"])

            yield current, reserve

    def operate(self, owner, request):
        lease_id = request.get("lease_id")
        try:
            row, tab, page, element = self._reserve_action(owner, request)
        except BrowserContractError as exc:
            if str(exc) in {"expired", "capability_expired", "action_capped"}:
                self.release(owner, lease_id, terminal=str(exc))
            return blocked(str(exc))
        # Unknown is durable before either callback, so retries cannot repeat it.
        try:
            if element:
                fresh = page_state(self.adapter.observe(lease_id, tab, row["expires_at"]), row, tab, page)
                if fresh["revision"] != page["revision"]:
                    self._restore_active(lease_id, fresh)
                    return blocked("stale_state")
                if self.clock() >= min(row["expires_at"], row["capabilities"]["expires_at"]):
                    self.release(owner, lease_id, terminal="deadline_exceeded")
                    return blocked("deadline_exceeded")
                result = self.adapter.act(lease_id, tab, page["host_revision"], element["key_digest"], "read", row["expires_at"])
                if not isinstance(result, dict) or result.get("status") != "observed":
                    if isinstance(result, dict) and result.get("status") == "stale_state":
                        self._restore_active(lease_id)
                        return blocked("stale_state")
                    return {"status": "unknown", "reason": "action_unknown"}
            observed = page_state(self.adapter.observe(lease_id, tab, row["expires_at"]), row, tab, page)
            if self.clock() >= min(row["expires_at"], row["capabilities"]["expires_at"]):
                self.release(owner, lease_id, terminal="deadline_exceeded")
                return blocked("deadline_exceeded")
            with self.store.transaction() as state:
                current = state["leases"][lease_id]
                if current["status"] != "unknown":
                    return blocked(current["status"])
                current["pages"][tab] = observed
                current["status"], current["reason"] = "active", ""
                self.store.save(state)
            return {"status": "observed", "page": observed}
        except BrowserContractError as exc:
            if str(exc) == "state_capped":
                self.release(owner, lease_id, terminal="state_capped")
            return {"status": "unknown", "reason": str(exc)}

    def _restore_active(self, lease_id, page=None):
        with self.store.transaction() as state:
            row = state["leases"][lease_id]
            if row["status"] == "unknown":
                row["status"], row["reason"] = "active", ""
                if page is not None:
                    row["pages"][page["tab_id"]] = page
                self.store.save(state)

    def _unusable(self, row):
        if row["status"] != "active":
            return row["reason"] or row["status"]
        now = self.clock()
        if now >= row["expires_at"]:
            return "expired"
        if now >= row["capabilities"]["expires_at"]:
            return "capability_expired"
        if row["action_count"] >= row["capabilities"]["limits"]["actions"]:
            return "action_capped"
        return ""

    def release(self, owner, lease_id, *, terminal="released"):
        if terminal not in TERMINALS:
            raise BrowserContractError("invalid_terminal")
        with self.store.transaction() as state:
            row = self._owned(state, owner, lease_id)
            if row["reaped"]:
                return self._view(row)
            if row["status"] != "releasing":
                row["status"], row["reason"] = "releasing", terminal
                self.store.save(state)
            # Hold the OS lock through bounded cleanup. Other managers cannot
            # duplicate this callback; a crash drops the lock and allows the
            # host's idempotent reap to recover the releasing reservation.
            reaped = self.adapter.release(lease_id, self.clock() + 5)
            if not isinstance(reaped, dict) or reaped.get("reaped") is not True:
                return {"status": "unknown", "reason": "cleanup_unknown", "lease_id": lease_id}
            row["status"], row["reason"], row["reaped"] = terminal, terminal, True
            row["pages"], row["acquisition"] = {}, None
            self.store.save(state)
            return self._view(row)

    def cleanup(self, owner):
        if not self.store.path.exists():
            return []
        owner_ref = digest(text(owner, 256))
        with self.store.transaction() as state:
            ids = [key for key, row in state["leases"].items()
                   if row["owner_ref"] == owner_ref and not row["reaped"]
                   and row["adapter_id"] == self.adapter.adapter_id
                   and row["adapter_version"] == self.adapter.adapter_version]
        return [self.release(owner, key, terminal="orphaned") for key in ids]

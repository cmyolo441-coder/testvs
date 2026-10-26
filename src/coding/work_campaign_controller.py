"""Owner-only transitions under one record lock; dispatch is host-owned."""
from copy import deepcopy

from .work_campaign_evidence import inspect_result
from .work_campaign_store import CampaignStore

TERMINALS = {"complete", "cancelled", "failed", "unknown", "cleanup_pending"}
LEAF_TOOLS = {"read_file", "write_file", "patch", "search_files"}


class Campaign:
    def __init__(self, directory, host):
        self.store = CampaignStore(directory)
        self.host = host

    def prepare(self, **request):
        from .work_campaign_contract import build_work_campaign
        if request.get("mode", "ordinary") != "campaign-orchestrator":
            return None
        record = build_work_campaign(**request, parent_session_ref=self.host.context())
        if record is None:
            return None
        record["ordinary_route"] = self.host.ordinary_route()
        return self.store.create(record)

    def show(self, campaign_id):
        from .work_campaign_contract import CampaignError
        record = self.store.read(campaign_id)
        if self.host.context() not in (record["parent_session_ref"], record["owner_session_ref"]):
            raise CampaignError("foreign_campaign")
        return record

    def act(self, campaign_id, action, **inputs):
        from .work_campaign_contract import CampaignError
        with self.store.locked(campaign_id) as record:
            actor = self.host.context()
            allowed = {record["owner_session_ref"]}
            if action in ("cancel", "start", "recover"):
                allowed.add(record["parent_session_ref"])
            if not actor or actor not in allowed:
                raise CampaignError("owner_only")
            if record["state"] in TERMINALS and action == "recover" and record["cleanup"].get("status") != "observed":
                self._finish(record, record.get("pending_terminal_state", record["state"]), record["terminal_reason"])
                return deepcopy(record)
            if record["state"] in TERMINALS:
                raise CampaignError("campaign_terminal")
            if action == "start":
                self._start(record)
            elif action == "dispatch":
                self._dispatch(record, inputs.get("unit_id"))
            elif action == "accept":
                if set(inputs) != {"unit_id"}:
                    raise CampaignError("self_report_is_not_evidence")
                unit = self._unit(record, inputs["unit_id"])
                if unit["state"] != "dispatched":
                    record["metrics"]["duplicate_result"] += 1
                    self.store.save(record)
                    raise CampaignError("duplicate_or_unknown_result")
                unit["result"] = inspect_result(record, unit, self.host.observe(campaign_id, unit["attempt_id"]))
                unit["state"] = "accepted"
            elif action == "plan":
                self._plan(record, inputs.get("units"))
            elif action == "conflict":
                self._conflict(record, inputs)
            elif action == "resolve":
                self._resolve(record)
            elif action == "queue-broad":
                self._queue_broad(record)
            elif action == "complete":
                self._complete(record)
            elif action in ("cancel", "fail", "timeout", "recover", "fallback"):
                state = {"cancel": "cancelled", "fail": "failed", "timeout": "unknown",
                         "recover": "unknown", "fallback": "fallback_parent_led"}[action]
                self._finish(record, state, action)
            else:
                raise CampaignError("unknown_campaign_action")
            record["events"] = (record["events"] + [dict(action=action, state=record["state"])])[-64:]
            self.store.save(record)
            return deepcopy(record)

    def _unit(self, record, unit_id):
        from .work_campaign_contract import CampaignError
        unit = next((u for u in record["units"] if u["unit_id"] == unit_id), None)
        if unit is None:
            raise CampaignError("unknown_unit")
        return unit

    def _start(self, record):
        from .work_campaign_contract import CampaignError, kanban_task_manifests
        if record["root_attempt_id"] in record["pending_routes"]:
            raise CampaignError("unknown_root_attempt")
        if record["state"] not in ("prepared", "conflicted") or record["root_binding"]:
            raise CampaignError("duplicate_root")
        capabilities = self.host.capabilities()
        supported = (capabilities.get("binding") in ("native_per_dispatch", "kanban_per_task_override")
                     and capabilities.get("identity_bound") and capabilities.get("scope_enforced")
                     and capabilities.get("verification_observer")
                     and bool(capabilities.get("leaf_tools"))
                     and set(capabilities["leaf_tools"]).issubset(LEAF_TOOLS))
        if not supported or not all(r["wire_model"] for r in record["routes"].values()):
            self._finish(record, "fallback_parent_led", "host_cannot_bind_safe_campaign")
            return
        for route in record["routes"].values():
            if not route["provider"]:
                route["provider"] = self.host.ordinary_route().get("provider", "")
                if not route["provider"]:
                    self._finish(record, "fallback_parent_led", "inherited_provider_not_observed")
                    return
        manifest = kanban_task_manifests(record)[0]
        binding = self._bind(record, manifest)
        if binding["session_ref"] == record["parent_session_ref"]:
            self._finish(record, "fallback_parent_led", "host_root_identity_mismatch")
            return
        record["root_binding"] = binding
        record["owner_session_ref"] = binding["session_ref"]
        for conflict in record["conflicts"]:
            conflict["integration_owner"] = binding["session_ref"]
        record["state"] = "conflicted" if any(not c["resolved"] for c in record["conflicts"]) else "running"

    def _bind(self, record, manifest):
        from .work_campaign_contract import CampaignError
        attempt_id = manifest["attempt_id"]
        # Persist uncertainty BEFORE entering host code. Even BaseException or
        # process loss cannot turn this attempt back into an undispatched unit.
        record["pending_routes"].append(attempt_id)
        self.store.save(record)
        bound = False
        try:
            binding = self.host.dispatch(deepcopy(manifest))
            if (binding.get("attempt_id") != attempt_id or binding.get("route") != manifest["route"]
                    or not binding.get("task_id") or not binding.get("session_ref")
                    or binding.get("state") != "bound"):
                raise CampaignError("host_binding_mismatch")
            bound = True
            return binding
        finally:
            if not bound:
                self._finish(record, "fallback_parent_led" if manifest["role"] == "orchestrator" else "unknown",
                             "dispatch_exception_or_provider_failure")
                self.store.save(record)

    def _dispatch(self, record, unit_id):
        from .work_campaign_contract import CampaignError, kanban_task_manifests
        if record["state"] != "running" or not record["root_binding"]:
            raise CampaignError("campaign_frontier_frozen")
        unit = self._unit(record, unit_id)
        if unit["state"] != "prepared":
            record["metrics"]["duplicate_dispatch"] += 1
            self.store.save(record)
            raise CampaignError("duplicate_dispatch")
        if any(self._unit(record, dep)["state"] != "accepted" for dep in unit["depends_on"]):
            raise CampaignError("dependency_not_accepted")
        unit["state"] = "unknown"
        manifest = next(m for m in kanban_task_manifests(record)[1:] if m["unit_id"] == unit_id)
        unit["binding"] = self._bind(record, manifest)
        unit["owner_session_ref"] = unit["binding"]["session_ref"]
        unit["state"] = "dispatched"

    def _plan(self, record, units):
        from .work_campaign_contract import CampaignError, normalize_units
        if any(u["state"] != "prepared" for u in record["units"]):
            raise CampaignError("graph_already_dispatched")
        normalized, conflicts = normalize_units(units, record["broad_suite"]["command"], record.get("spawn_plan"))
        if conflicts:
            raise CampaignError("plan_still_conflicted")
        if {u["unit_id"] for u in normalized} != {u["unit_id"] for u in record["units"]}:
            raise CampaignError("accepted_unit_set_is_immutable")
        for new in normalized:
            old = self._unit(record, new["unit_id"])
            for key in ("file_scope", "acceptance", "artifacts", "verification_command"):
                if new[key] != old[key]:
                    raise CampaignError("accepted_contract_is_immutable")
            old["depends_on"] = new["depends_on"]

    def _conflict(self, record, inputs):
        from .work_campaign_contract import CampaignError, digest
        ids = inputs.get("unit_ids", [])
        if not isinstance(ids, list) or not 2 <= len(ids) <= 16 or not inputs.get("invariant"):
            raise CampaignError("conflict_requires_units_and_invariant")
        for unit_id in ids:
            self._unit(record, unit_id)
        if len(record["conflicts"]) >= 16:
            raise CampaignError("conflict_cap")
        record["conflicts"].append(dict(units=ids, invariant_digest=digest(inputs["invariant"]),
                                        integration_owner=record["owner_session_ref"], resolved=False))
        record["metrics"]["conflicts"] += 1
        record["state"] = "conflicted"
        # Stop affected in-flight work at the host boundary, not only in prose.
        pending = [u["attempt_id"] for u in record["units"] if u["unit_id"] in ids and u["state"] == "dispatched"]
        if pending:
            for unit in record["units"]:
                if unit["attempt_id"] in pending:
                    unit["state"] = "unknown"
            self.store.save(record)
            if not self.host.expire(record["campaign_id"], pending):
                raise CampaignError("conflict_expiry_failed")

    def _queue_broad(self, record):
        from .work_campaign_contract import CampaignError
        if (record["state"] not in ("running", "fallback_parent_led") or record["broad_suite"]["queued"]
                or any(u["state"] != "accepted" for u in record["units"])
                or any(not c["resolved"] for c in record["conflicts"])):
            raise CampaignError("broad_suite_not_ready_or_already_assigned")
        record["broad_suite"].update(queued=True, owner_session_ref=record["owner_session_ref"])
        record["state"] = "integrating"

    def _resolve(self, record):
        from .work_campaign_contract import CampaignError
        if record["state"] != "conflicted" or not record["root_binding"]:
            raise CampaignError("conflict_owner_not_bound")
        affected = {unit_id for c in record["conflicts"] if not c["resolved"] for unit_id in c["units"]}
        results = {}
        for unit_id in affected:
            unit = self._unit(record, unit_id)
            if unit["state"] in ("dispatched", "unknown"):
                raise CampaignError("unknown_side_effects_require_external_reconciliation")
            owned = dict(unit, binding=record["root_binding"])
            results[unit_id] = inspect_result(record, owned, self.host.observe(record["campaign_id"], unit["attempt_id"]))
        for unit_id, result in results.items():
            self._unit(record, unit_id).update(result=result, state="accepted", owner_session_ref=record["owner_session_ref"])
        for conflict in record["conflicts"]:
            conflict["resolved"] = True
        record["metrics"]["resolutions"] += 1
        record["state"] = "running"

    def _complete(self, record):
        from .work_campaign_contract import CampaignError
        if record["state"] != "integrating" or not record["broad_suite"]["queued"]:
            raise CampaignError("integration_not_ready")
        unit = dict(attempt_id=record["broad_suite"]["attempt_id"], binding=record["root_binding"],
                    verification_command=record["broad_suite"]["command"],
                    file_scope=sorted({p for u in record["units"] for p in u["file_scope"]}),
                    artifacts=sorted({p for u in record["units"] for p in u["artifacts"]}))
        result = inspect_result(record, unit, self.host.observe(record["campaign_id"], unit["attempt_id"]))
        record["broad_suite"].update(observed=result, executions=1)
        self._finish(record, "complete", "verified_fanin")

    def _finish(self, record, state, reason):
        # Readers do not hold the transition lock, and SIGKILL skips finally.
        # Publish uncertainty before expiry or ordinary-route readback callbacks.
        record["state"] = "cleanup_pending"
        record["pending_terminal_state"] = state
        record["terminal_reason"] = reason
        record["cleanup"].update(expired=False, status="cleanup_not_observed", ordinary_readback=None)
        self.store.save(record)
        try:
            expired = self.host.expire(record["campaign_id"], tuple(record["pending_routes"]))
            record["cleanup"]["expired"] = expired
            if expired:
                record["pending_routes"] = []
            for unit in record["units"]:
                if unit["state"] not in ("accepted", "rejected"):
                    unit["state"] = ("unknown" if unit["state"] in ("dispatched", "unknown") else
                                     "prepared" if state == "fallback_parent_led" else "cancelled")
            readback = self.host.ordinary_route()
            record["cleanup"]["ordinary_readback"] = readback
            if not expired or readback != record["ordinary_route"]:
                record["metrics"]["cleanup_failures"] += 1
                record["cleanup"]["status"] = "cleanup_not_observed"
            else:
                record["cleanup"]["status"] = "observed"
        finally:
            if record["cleanup"]["status"] == "observed" and record["cleanup"]["expired"]:
                record["state"] = state
                record.pop("pending_terminal_state", None)
                if state == "fallback_parent_led":
                    record["owner_session_ref"] = record["parent_session_ref"]
                    record["metrics"]["fallbacks"] += 1
            self.store.save(record)

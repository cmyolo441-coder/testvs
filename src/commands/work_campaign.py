"""Lazy, explicit operator campaign surface; no added plugin tool/context."""
import json
from pathlib import Path


def add_work_campaign_command(coding_sub):
    parser = coding_sub.add_parser("campaign", help="Opt-in campaign contracts and host-binding observations (operator).")
    sub = parser.add_subparsers(dest="campaign_action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--goal", required=True)
    prepare.add_argument("--units", required=True, help="JSON unit contracts file.")
    prepare.add_argument("--spawn-plan", help="Accepted fanout justification JSON, required above four units.")
    prepare.add_argument("--acceptance", action="append", required=True)
    prepare.add_argument("--verify", required=True)
    prepare.add_argument("--workspace", required=True)
    prepare.add_argument("--accept", action="store_true", required=True)
    for role in ("owner", "worker"):
        for field in ("model", "provider", "effort"):
            prepare.add_argument(f"--{role}-{field}", default="")
    prepare.set_defaults(func=cmd_campaign)
    for action in ("show", "start", "bind", "dispatch", "accept", "plan", "conflict", "resolve", "queue-broad",
                   "complete", "cancel", "fail", "timeout", "recover", "fallback"):
        child = sub.add_parser(action)
        child.add_argument("campaign_id")
        if action == "plan":
            child.add_argument("--units", required=True)
        if action in ("accept", "dispatch"):
            child.add_argument("--unit", required=True)
        if action == "conflict":
            child.add_argument("--unit", action="append", required=True)
            child.add_argument("--invariant", required=True)
        if action == "bind":
            child.add_argument("--binding", choices=("native", "kanban"), default="kanban")
            child.add_argument("--tasks", help="JSON mapping: orchestrator/unit id to existing host task id.")
            child.add_argument("--worker-profile", default="")
        child.set_defaults(func=cmd_campaign)
    canary = sub.add_parser("canary")
    canary.add_argument("--input", help="Bounded comparison observations JSON; absent means no observations.")
    canary.set_defaults(func=cmd_campaign)
    for child in sub.choices.values():
        child.add_argument("--json", action="store_true")


def _read(path):
    with Path(path).open("rb") as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("campaign_input_cap")
    return json.loads(content)


def cmd_campaign(args):
    from ..coding.work_campaign import Campaign, campaign_canary_report
    from ..coding.work_campaign_host import LocalCampaignHost
    from ..core.errors import OmhError
    from .common import _paths, _print_json

    paths = _paths(args)
    api = Campaign(paths.runtime_dir / "campaigns", LocalCampaignHost(paths.hermes_home))
    try:
        action = args.campaign_action
        if action == "prepare":
            payload = api.prepare(mode="campaign-orchestrator", accepted=args.accept, goal=args.goal,
                                  units=_read(args.units), acceptance_criteria=args.acceptance,
                                  verification_command=args.verify, workspace=Path(args.workspace),
                                  omh_home=paths.omh_home, hermes_home=paths.hermes_home,
                                  spawn_plan=_read(args.spawn_plan) if args.spawn_plan else None,
                                  **{f"{role}_{field}": getattr(args, f"{role}_{field}")
                                     for role in ("owner", "worker") for field in ("model", "provider", "effort")})
        elif action == "show":
            payload = api.show(args.campaign_id)
        elif action == "canary":
            payload = campaign_canary_report(_read(args.input) if args.input else [])
        elif action == "bind":
            payload = _bind(api, args, paths.hermes_home)
        else:
            inputs = {}
            if action == "plan":
                inputs["units"] = _read(args.units)
            if action in ("accept", "dispatch"):
                inputs["unit_id"] = args.unit
            if action == "conflict":
                inputs.update(unit_ids=args.unit, invariant=args.invariant)
            payload = api.act(args.campaign_id, action, **inputs)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise OmhError(f"campaign refused: {exc}") from exc
    if payload is None:
        raise OmhError("campaign mode was not selected")
    if args.json:
        _print_json(payload)
    else:
        print(f"Campaign: {payload.get('campaign_id', 'canary')} - {payload.get('state', 'not_observed')}")
        if payload.get("terminal_reason"):
            print(payload["terminal_reason"])
    return 0


def _bind(api, args, hermes_home):
    from ..coding.work_campaign import CampaignError
    from ..coding.work_campaign_host import observe_kanban_bindings, observe_worker_profile
    if args.binding == "native":
        return api.act(args.campaign_id, "start")
    if not args.tasks or not args.worker_profile:
        raise CampaignError("kanban_tasks_and_worker_profile_required")
    tasks = _read(args.tasks)
    with api.store.locked(args.campaign_id) as record:
        if api.host.context() != record["parent_session_ref"] or record["state"] not in ("prepared", "conflicted"):
            raise CampaignError("owner_only_prepared_binding")
        expected_ids = {"orchestrator", *(u["unit_id"] for u in record["units"])}
        if not isinstance(tasks, dict) or set(tasks) != expected_ids:
            raise CampaignError("exact_campaign_tasks_required")
        profile = observe_worker_profile(hermes_home, args.worker_profile)
        if profile["status"] != "observed":
            raise CampaignError(profile["reason"])
        observation = observe_kanban_bindings(hermes_home, list(tasks.values()))
        if observation["status"] != "observed":
            raise CampaignError(observation["reason"])
        rows = {r["id"]: r for r in observation["tasks"]}
        for unit_id, task_id in tasks.items():
            root = unit_id == "orchestrator"
            route = record["routes"]["root" if root else "worker"]
            key = record["root_attempt_id"] if root else next(u["attempt_id"] for u in record["units"] if u["unit_id"] == unit_id)
            row = rows[task_id]
            dependencies = [] if root else next(u["depends_on"] for u in record["units"] if u["unit_id"] == unit_id)
            if observation["parents"][task_id] != sorted(tasks[x] for x in dependencies):
                raise CampaignError("host_dependency_graph_mismatch")
            if (row["model_override"] != route["wire_model"] or (row["provider_override"] or "") != route["provider"]
                    or (row["reasoning_effort"] or "") != route["reasoning_effort"]
                    or row["idempotency_key"] != key or row["max_retries"] != 0
                    or (not root and (row["assignee"] != args.worker_profile or row["goal_mode"]))):
                raise CampaignError("host_task_binding_mismatch")
        payload = dict(tasks=observation["tasks"], parents=observation["parents"], profile=profile, status="prepared_binding_observed",
                       execution="not_observed", reason="host_scope_and_verification_adapter_unavailable")
        if "host_binding_observation" in record and record["host_binding_observation"] != payload:
            raise CampaignError("immutable_binding_changed")
        record["host_binding_observation"] = payload
        api.store.save(record)
        return record

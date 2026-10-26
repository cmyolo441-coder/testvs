"""Opted-in effect operations on the existing admission-scoped browser tool."""
from copy import deepcopy
import sqlite3

EFFECT_OPERATIONS = {"effect_preview", "effect_execute", "effect_abort"}


def extend_schema(schema):
    from omh.workflows.browser_effect_attempts_contract import OPERATIONS

    parameters = schema["parameters"]
    properties = parameters["properties"]
    inert = deepcopy(properties)
    properties["operation"]["enum"] += sorted(EFFECT_OPERATIONS)
    properties["action"]["enum"] += sorted(OPERATIONS)
    properties.update(intent_digest={"type": "string", "pattern": "^[a-f0-9]{64}$"},
        trace_revision={"type": ["string", "null"], "pattern": "^[a-f0-9]{64}$"},
        expected_postcondition={"type": "string", "enum": ["confirmation"]})
    parameters["additionalProperties"] = False
    branches = [{"properties": inert}]
    for operation in sorted(EFFECT_OPERATIONS):
        fields = ({"operation", "action", "lease_id", "tab_id", "revision", "handle",
                   "trace_revision", "expected_postcondition"} if operation == "effect_preview"
                  else {"operation", "intent_digest"})
        branch = {key: deepcopy(properties[key]) for key in sorted(fields)}
        branch["operation"] = {"const": operation}
        if operation == "effect_preview":
            branch["action"]["enum"] = sorted(OPERATIONS)
        branches.append({"type": "object", "properties": branch,
                         "required": sorted(fields), "additionalProperties": False})
    parameters["oneOf"] = branches


def trusted_event(owner):
    """Hermes binds these ContextVars around registry dispatch, never from JSON."""
    from omh.workflows.browser_adapter import BrowserContractError, text

    try:
        from tools.approval_context import _approval_session_id, _approval_tool_call_id
    except ImportError:
        raise BrowserContractError("host_event_required") from None
    event = _approval_tool_call_id.get()
    if _approval_session_id.get() != owner or not isinstance(event, str) or not event:
        raise BrowserContractError("host_event_required")
    return text(event, 256)


def handle(ctx, manager, admission, available, args):
    from omh.workflows.browser_adapter import BrowserContractError, digest
    from omh.workflows.browser_effect_attempts import BrowserEffectEngine
    from omh.workflows.browser_effect_attempts_contract import effect_request, hex_ref

    operation = args["operation"]
    owner = admission.identity[0]
    if operation == "effect_preview":
        request = {key: value for key, value in args.items() if key not in {"operation", "action"}}
        request["operation"] = args.get("action")
        payload = effect_request(request)
    else:
        if set(args) != {"operation", "intent_digest"}:
            raise BrowserContractError("invalid_effect_request")
        payload = {"intent_digest": hex_ref(args["intent_digest"])}
    event = trusted_event(owner) if operation == "effect_execute" else None

    def check():
        if available() is not admission or getattr(ctx, "browser_adapter", None) is not manager.adapter:
            raise BrowserContractError("admission_required")
        if event is not None and trusted_event(owner) != event:
            raise BrowserContractError("host_event_changed")

    def approval(key):
        check()
        receipt = ctx.browser_effect_approval(admission.identity, admission.task, key)
        check()
        return receipt

    try:
        with manager.effect_boundary(owner, admission.task) as (current, reserve):
            def lease_source(request_owner, lease_id):
                check()
                return current(request_owner, lease_id)

            def reserve_action(request_owner, request):
                check()
                return reserve(request_owner, request)

            # Same canonical OMH root and actual manager adapter. Per-call
            # callbacks cannot freeze the first admitted request into a cache.
            home = manager.store.path.parents[2]
            engine = BrowserEffectEngine(home, manager.adapter, lease_source, approval,
                                         reserve_action=reserve_action)
            if operation == "effect_preview":
                lease = lease_source(owner, payload["lease_id"])
                needed = "upload" if payload["operation"] == "upload" else "click"
                if needed in lease["capabilities"]["unsupported"]:
                    raise BrowserContractError("action_out_of_scope")
                return engine.preview(owner, payload)
            key = payload["intent_digest"]
            with engine.store.transaction() as db:
                intent, _, result = engine.store.get(db, key)
                lease = lease_source(owner, intent["lease_id"])
                if operation == "effect_execute":
                    # Persist event identity even for a refused/unapproved call.
                    engine.store.bind_event(db, digest([digest(owner), event]), key)
            if operation == "effect_execute":
                if not result and lease["action_count"] >= lease["capabilities"]["limits"]["actions"]:
                    raise BrowserContractError("action_capped")
                return engine.execute(owner, key, event)
            return engine.abort(owner, key)
    except BrowserContractError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, sqlite3.Error):
        # Adapter/storage failures never become generic retry instructions or
        # raw exception text (which can contain page/request secrets).
        return {"status": "unknown", "reason": "effect_boundary_unknown", "retry_allowed": False}

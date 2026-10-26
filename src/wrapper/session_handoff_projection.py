from __future__ import annotations

from typing import Any

from ..runtime.records import validate_coding_prompt_handoff, validate_coding_runtime_handoff


def project_valid_session_handoffs(session: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_handoff = session.get("prompt_handoff")
    runtime_handoff = session.get("runtime_handoff")
    safe_prompt = (
        prompt_handoff
        if isinstance(prompt_handoff, dict) and not validate_coding_prompt_handoff(prompt_handoff)
        else {}
    )
    safe_runtime = (
        runtime_handoff
        if isinstance(runtime_handoff, dict) and not validate_coding_runtime_handoff(runtime_handoff)
        else {}
    )
    return safe_prompt, safe_runtime

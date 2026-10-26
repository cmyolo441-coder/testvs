from __future__ import annotations

import json
import re
from typing import Final, cast


BUZZ_DELIVERY_EVIDENCE_SCHEMA: Final = "omh_buzz_delivery_evidence/v1"
_MAX_EVENT_ID_LENGTH: Final = 256
_SAFE_EVENT_ID: Final = re.compile(rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{_MAX_EVENT_ID_LENGTH - 1}}}")


def parse_buzz_delivery_receipt(stdout: str) -> dict[str, object]:
    """Parse one Buzz send receipt into bounded, fail-closed evidence."""

    try:
        decoded = cast(object, json.loads(stdout))
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if not isinstance(decoded, dict):
        return _result("ambiguous", "receipt_not_json_object")

    receipt = cast(dict[str, object], decoded)
    accepted = receipt.get("accepted")
    if accepted is False:
        return _result("rejected", "receipt_rejected")
    if accepted is not True:
        return _result("ambiguous", "receipt_missing_accepted")

    event_id = _event_id(receipt.get("event_id"))
    if event_id is None:
        return _result("ambiguous", "receipt_missing_event_id")
    return _result("event_accepted", "event_accepted", event_id=event_id)


def _event_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    event_id = value.strip()
    if _SAFE_EVENT_ID.fullmatch(event_id) is None:
        return None
    return event_id


def _result(status: str, reason_code: str, *, event_id: str | None = None) -> dict[str, object]:
    return {
        "schema_version": BUZZ_DELIVERY_EVIDENCE_SCHEMA,
        "status": status,
        "reason_code": reason_code,
        "delivery_stage": status,
        "event_id": event_id,
        "accepted": status == "event_accepted",
        "retry_safe": status == "rejected",
        "client_rendered": False,
        "claim_boundary": (
            "event_accepted proves only an explicit accepted receipt with a non-empty event id. "
            "It does not prove retrieval, subscription observation, or client rendering."
        ),
    }

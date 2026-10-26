"""Realtime voice trial receipts (`realtime_voice_trial_receipt/v1`, issue #1426).

A voice connector that authenticates and returns audio has proven almost
nothing. It can still clip the first syllable of every utterance, cut one
sentence into two turns or glue two into one, dispatch a transcript that was
never finished, take four seconds to make a sound, start its answer over from
the beginning after a streaming fallback, ignore an interruption, or hand an
ambiguous spoken phrase to a tool that spends money. Generic connector success
and one audio artifact cannot tell those apart, so this contract asks for the
milestones that can.

Three separations do the work here, and none of them collapses into another:

- **Turn integrity is not latency.** A turn whose milestones are complete and
  in order can still be a split turn, a double dispatch, or a truncated input.
  Integrity is checked per turn from named states; latency is *derived* and
  only for turns that earned it -- one declared timing reference, every
  required milestone present, and a segmentation that nobody split or merged.
  A turn without those reports no latency at all rather than a plausible number.
- **Requested is not observed.** Every trial names the stack it asked for and,
  separately, the stack the host actually saw. When the host measured nothing,
  `observed_stack_evidence` is `unavailable` and the observed block must be
  empty -- echoing the requested model into it is refused, because a receipt
  that cannot tell the two apart is how a request becomes a result.
- **Synthetic is not the room.** A fixture proves the wiring; it does not prove
  the room, the microphone, the network, or the telephony path. A synthetic
  trial's verdict is capped at `hold`, and an actual-environment trial whose
  facets were not all the intended ones is capped the same way, with the
  missing facets named.

The contract is provider-neutral by construction. Providers, models, voices,
transports, codecs, endpoint detectors, and fallback paths are opaque
references a host or connector supplies; nothing here branches on any of them.
OMH opens no microphone, call, room, socket, or provider session, downloads no
model, installs no connector, and authorizes no tool. A receipt is an
observation somebody else made, validated and summarized.

Conversational content stays out by construction rather than by redaction. The
key set is closed, so there is no field a transcript, a raw tool argument, a
provider payload, or a recording could arrive in, and every free string is
screened for raw phone numbers, addresses, and credential shapes before it is
accepted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final

from ..system.metadata_safety import (
    is_body_shaped_metadata_text,
    is_raw_pii_shaped,
    is_sensitive_metadata_text,
    require_opaque_metadata_ref,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


REALTIME_VOICE_TURN_SCHEMA_VERSION: Final = "realtime_voice_turn/v1"
REALTIME_VOICE_TRIAL_RECEIPT_SCHEMA_VERSION: Final = "realtime_voice_trial_receipt/v1"
REALTIME_VOICE_READINESS_SCHEMA_VERSION: Final = "realtime_voice_readiness/v1"

TURN_CLAIM_BOUNDARY: Final = (
    "A realtime voice turn records milestones and named states for one spoken exchange. It is not a "
    "transcript, not proof that recognition was accurate, and not evidence that any tool ran."
)
RECEIPT_CLAIM_BOUNDARY: Final = (
    "A realtime voice trial receipt is a bounded observation someone else made of one voice connector "
    "trial. It is not connector installation, credential validation, provider access, speech recognition "
    "accuracy, conversation quality, or tool authorization evidence, and OMH executed none of it."
)
READINESS_CLAIM_BOUNDARY: Final = (
    "A realtime voice readiness answer summarizes one supplied trial receipt. It is not execution, review, "
    "CI, or merge evidence, it does not authorize a tool, and it never converts an unobserved or "
    "unsupported dimension into a pass."
)

#: Six hours, the same "is the environment still the one that was observed"
#: horizon `external_action_readiness` uses. A voice trial ages for the same
#: reason: the room, the network, and the provider all move.
REALTIME_VOICE_TRIAL_STALE_AFTER_SECONDS: Final = 6 * 60 * 60

#: A fixture proves wiring; the intended room, microphone, network, and
#: transport are a separate claim that only an actual-environment trial makes.
TEST_CONDITIONS: Final = ("synthetic_fixture", "actual_environment")

#: The four facets that make a trial "the real thing" rather than a rehearsal.
ENVIRONMENT_FACETS: Final = ("room", "microphone", "network", "transport")
#: `intended` is the facet the adoption decision is about; `simulated` stands
#: in for it; `unobserved` says nobody looked.
ENVIRONMENT_COVERAGE: Final = ("intended", "simulated", "unobserved")

#: How a value came to be on the receipt. `derived` is computed from other
#: observed values; `unavailable` means the producer had nothing to report.
EVIDENCE_CLASSES: Final = ("measured", "producer_reported", "derived", "unavailable")

#: Whose clock the milestones are on. One per turn, never mixed, and
#: `unavailable` forbids milestones outright.
TIMING_REFERENCES: Final = ("host_monotonic", "connector_reported", "gateway_reported", "unavailable")

#: The milestone order a well-formed turn follows. Offsets are non-negative
#: integer milliseconds from the turn's own zero on its declared reference.
TURN_MILESTONES: Final = (
    "speech_start",
    "speech_end",
    "final_transcript",
    "first_output_text",
    "first_output_audio",
    "playback_end",
)
#: What latency needs before it may be derived. `playback_end` is excluded:
#: an interrupted turn legitimately never reaches it.
LATENCY_REQUIRED_MILESTONES: Final = ("speech_start", "speech_end", "final_transcript", "first_output_audio")

TURN_TERMINAL_STATUSES: Final = ("completed", "interrupted", "cancelled", "truncated", "failed", "unobserved")

#: Whether the pre-roll survived. `clipped` is the failure the issue names
#: first: the connector authenticated, answered, and ate the first syllable.
ONSET_STATES: Final = ("preserved", "clipped", "unobserved")
SEGMENTATION_STATES: Final = ("intact", "premature_split", "unintended_merge", "unobserved")
INPUT_COMPLETENESS_STATES: Final = ("complete", "truncated", "unobserved")
MAX_DURATION_BEHAVIORS: Final = ("not_reached", "capped_with_notice", "silently_truncated", "unobserved")
MUTE_RESUME_STATES: Final = ("not_exercised", "resumed", "lost", "unobserved")
TRANSCRIPT_OUTCOMES: Final = ("final", "empty", "failed", "unobserved")

TURN_INTEGRITY_KEYS: Final = (
    "onset",
    "segmentation",
    "dispatch_count",
    "input_completeness",
    "max_duration_behavior",
    "mute_resume",
    "transcript_outcome",
)

BARGE_IN_STATES: Final = ("not_exercised", "honored", "ignored", "unsupported", "unobserved")
QUEUED_AUDIO_STATES: Final = ("not_applicable", "discarded", "drained", "unobserved")
#: `replayed_from_start` is the streaming-fallback failure: audio was already
#: audible and the connector started the answer over.
REPLAY_STATES: Final = ("none", "replayed_from_start", "unobserved")
BACKPRESSURE_STATES: Final = ("none", "shed_with_notice", "overflowed", "unobserved")

TURN_STREAMING_KEYS: Final = ("barge_in", "queued_audio", "replay", "backpressure")

REALTIME_VOICE_TURN_KEYS: Final = (
    "claim_boundary",
    "integrity",
    "milestone_evidence",
    "milestones",
    "schema_version",
    "streaming",
    "terminal_status",
    "timing_reference",
    "turn_id",
)

VOICE_STACK_KEYS: Final = (
    "channels",
    "codec",
    "endpoint_config_ref",
    "endpoint_detector",
    "model",
    "provider",
    "sample_rate_hz",
    "transport",
    "voice",
)
_VOICE_STACK_REF_KEYS: Final = (
    "codec",
    "endpoint_config_ref",
    "endpoint_detector",
    "model",
    "provider",
    "transport",
    "voice",
)
_VOICE_STACK_INT_KEYS: Final = ("channels", "sample_rate_hz")

#: How the observed stack was learned. `unavailable` forbids every observed
#: value, which is what stops a requested model from being filed as observed.
OBSERVED_STACK_EVIDENCE: Final = ("measured", "producer_reported", "unavailable")

FALLBACK_AUDIBLE_RELATIONS: Final = ("not_applicable", "before_audible", "after_audible", "unobserved")
REQUESTED_PATH_OUTCOMES: Final = ("succeeded", "failed", "unobserved")
FALLBACK_KEYS: Final = (
    "fallback_path_ref",
    "occurred",
    "reason",
    "relative_to_audible",
    "requested_path_outcome",
)

ACTION_CLASSES: Final = ("read_only", "low_impact", "high_impact")
CONFIDENCE_STATES: Final = ("confident", "ambiguous", "unobserved")
CONFIRMATION_STATES: Final = ("not_required", "received", "refused", "unobserved")
POLICY_DECISIONS: Final = ("allowed", "denied", "deferred", "unobserved")
TOOL_SAFETY_KEYS: Final = (
    "action_class",
    "attempt_ref",
    "confidence_state",
    "confirmation_required",
    "confirmation_state",
    "denial_reason",
    "policy_decision",
    "tool_result_ref",
)

REALTIME_VOICE_TRIAL_RECEIPT_KEYS: Final = (
    "claim_boundary",
    "connector_ref",
    "connector_revision",
    "environment",
    "fallback",
    "observed_stack",
    "observed_stack_evidence",
    "profile_ref",
    "receipt_id",
    "requested_stack",
    "schema_version",
    "test_condition",
    "tool_safety",
    "trial_ended_at",
    "trial_ref",
    "trial_started_at",
    "turns",
)

READINESS_DIMENSIONS: Final = ("turn_integrity", "latency", "fallback", "interruption", "tool_safety")
READINESS_STATES: Final = ("pass", "hold", "block")
READINESS_VERDICTS: Final = ("ready", "hold", "blocked")

#: The six states chat and status surfaces keep apart. A connector that is
#: configured has not passed a trial; a synthetic pass is not the room; a turn
#: observed is not a tool action observed; none of them is a completed session.
READINESS_STATE_KEYS: Final = (
    "connector_configured",
    "synthetic_trial_passed",
    "actual_environment_trial_passed",
    "voice_turn_observed",
    "tool_action_observed",
    "session_completed",
)

REALTIME_VOICE_READINESS_KEYS: Final = (
    "claim_boundary",
    "connector_ref",
    "connector_revision",
    "dimensions",
    "evidence_gate",
    "fallback",
    "latency",
    "receipt_id",
    "schema_version",
    "states",
    "test_condition",
    "trial_ref",
    "verdict",
    "verdict_reasons",
)

MAX_REASON_CHARS: Final = 200

#: What identifies one trial. The verdict is excluded on purpose: re-reading
#: the same receipt against a different clock must not mint a new identity.
_IDENTITY_KEYS: Final = (
    "connector_ref",
    "connector_revision",
    "profile_ref",
    "trial_ended_at",
    "trial_ref",
    "trial_started_at",
)

_LABEL: Final = "realtime_voice_trial_receipt"
_TURN_LABEL: Final = "realtime_voice_turn"


class RealtimeVoiceTrialError(ValueError):
    """Raised when a turn, receipt, or readiness answer cannot be built as supplied."""


# --------------------------------------------------------------------------
# Turns
# --------------------------------------------------------------------------


def build_realtime_voice_turn(
    *,
    turn_id: str,
    timing_reference: str = "host_monotonic",
    milestones: Mapping[str, Any] | None = None,
    milestone_evidence: Mapping[str, Any] | None = None,
    terminal_status: str = "completed",
    integrity: Mapping[str, Any] | None = None,
    streaming: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One spoken exchange, or a refusal.

    Absent milestones stay absent and are attributed `unavailable`; nothing is
    interpolated, defaulted to zero, or copied from a neighbouring turn.
    """
    supplied = dict(milestones or {})
    evidence = dict(milestone_evidence or {})
    resolved_milestones: dict[str, Any] = {}
    resolved_evidence: dict[str, Any] = {}
    for name in TURN_MILESTONES:
        value = supplied.get(name)
        resolved_milestones[name] = value
        resolved_evidence[name] = evidence.get(name, "measured" if value is not None else "unavailable")
    record = {
        "schema_version": REALTIME_VOICE_TURN_SCHEMA_VERSION,
        "turn_id": str(turn_id or "").strip(),
        "timing_reference": str(timing_reference or ""),
        "milestones": resolved_milestones,
        "milestone_evidence": resolved_evidence,
        "terminal_status": str(terminal_status or ""),
        "integrity": _with_defaults(integrity, TURN_INTEGRITY_KEYS, _TURN_INTEGRITY_DEFAULTS),
        "streaming": _with_defaults(streaming, TURN_STREAMING_KEYS, _TURN_STREAMING_DEFAULTS),
        "claim_boundary": TURN_CLAIM_BOUNDARY,
    }
    errors = validate_realtime_voice_turn(record)
    if errors:
        raise RealtimeVoiceTrialError(errors[0])
    return record


def validate_realtime_voice_turn(record: Any) -> list[str]:
    """Every contract violation in one turn, structural faults first."""
    if not isinstance(record, Mapping):
        return [f"{_TURN_LABEL} must be an object"]
    errors = _key_set_errors(record, REALTIME_VOICE_TURN_KEYS, _TURN_LABEL)
    if record.get("schema_version") != REALTIME_VOICE_TURN_SCHEMA_VERSION:
        errors.append(f"{_TURN_LABEL} schema_version must be {REALTIME_VOICE_TURN_SCHEMA_VERSION}")
    if errors:
        return errors
    errors.extend(_opaque_ref_errors(record["turn_id"], f"{_TURN_LABEL} turn_id", required=True))
    errors.extend(_vocabulary_errors(record, "timing_reference", TIMING_REFERENCES, _TURN_LABEL))
    errors.extend(_vocabulary_errors(record, "terminal_status", TURN_TERMINAL_STATUSES, _TURN_LABEL))
    if record["claim_boundary"] != TURN_CLAIM_BOUNDARY:
        errors.append(f"{_TURN_LABEL} claim_boundary must state the turn boundary")
    errors.extend(_turn_milestone_errors(record))
    errors.extend(_turn_integrity_errors(record))
    errors.extend(_turn_streaming_errors(record))
    return errors


def turn_latency(record: Mapping[str, Any]) -> dict[str, Any]:
    """Latency derived from one turn, or an explicit refusal to derive it.

    A turn earns a latency reading only with one declared timing reference and
    a complete, unsplit milestone sequence. Anything else reports `eligible:
    False` with the basis, never a number: the point of the contract is that a
    connector which merged two turns cannot also publish how fast it answered.
    """
    basis = _latency_basis(record)
    if basis != "measured":
        return {
            "eligible": False,
            "basis": basis,
            "first_audible_output_ms": None,
            "final_transcript_ms": None,
            "response_span_ms": None,
        }
    milestones = record["milestones"]
    speech_end = int(milestones["speech_end"])
    return {
        "eligible": True,
        "basis": "measured",
        "first_audible_output_ms": int(milestones["first_output_audio"]) - speech_end,
        "final_transcript_ms": int(milestones["final_transcript"]) - speech_end,
        "response_span_ms": int(milestones["first_output_audio"]) - int(milestones["speech_start"]),
    }


def turn_integrity_failures(record: Mapping[str, Any]) -> list[str]:
    """Named integrity failures observed in one turn, in reading order."""
    integrity = record["integrity"]
    streaming = record["streaming"]
    failures: list[str] = []
    if integrity["onset"] == "clipped":
        failures.append("speech_onset_clipped")
    if integrity["segmentation"] == "premature_split":
        failures.append("turn_split_prematurely")
    if integrity["segmentation"] == "unintended_merge":
        failures.append("turns_merged_unintentionally")
    if int(integrity["dispatch_count"]) > 1:
        failures.append("multiple_dispatch_for_one_utterance")
    if int(integrity["dispatch_count"]) == 0 and record["terminal_status"] == "completed":
        failures.append("completed_turn_dispatched_nothing")
    if integrity["input_completeness"] == "truncated":
        failures.append("input_truncated")
    if integrity["max_duration_behavior"] == "silently_truncated":
        failures.append("max_duration_truncated_silently")
    if integrity["mute_resume"] == "lost":
        failures.append("turn_lost_across_mute_resume")
    if integrity["transcript_outcome"] == "failed":
        failures.append("transcript_failed")
    if integrity["transcript_outcome"] == "empty" and int(integrity["dispatch_count"]) > 0:
        failures.append("empty_transcript_dispatched")
    if streaming["backpressure"] == "overflowed":
        failures.append("output_overflowed")
    return failures


def turn_interruption_failures(record: Mapping[str, Any]) -> list[str]:
    """Named streaming and interruption failures observed in one turn."""
    streaming = record["streaming"]
    failures: list[str] = []
    if streaming["barge_in"] == "ignored":
        failures.append("barge_in_ignored")
    if streaming["replay"] == "replayed_from_start":
        failures.append("replayed_after_audible_output")
    if streaming["queued_audio"] == "drained" and streaming["barge_in"] == "honored":
        failures.append("queued_audio_played_after_barge_in")
    if streaming["backpressure"] == "overflowed":
        failures.append("backpressure_overflowed")
    return failures


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def voice_stack(
    *,
    provider: str = "",
    model: str = "",
    voice: str = "",
    transport: str = "",
    codec: str = "",
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    endpoint_detector: str = "",
    endpoint_config_ref: str = "",
) -> dict[str, Any]:
    """One side of the requested/observed pair, as opaque references only."""
    return {
        "provider": str(provider or "").strip(),
        "model": str(model or "").strip(),
        "voice": str(voice or "").strip(),
        "transport": str(transport or "").strip(),
        "codec": str(codec or "").strip(),
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "endpoint_detector": str(endpoint_detector or "").strip(),
        "endpoint_config_ref": str(endpoint_config_ref or "").strip(),
    }


def voice_fallback(
    *,
    occurred: bool = False,
    requested_path_outcome: str = "succeeded",
    fallback_path_ref: str = "",
    reason: str = "",
    relative_to_audible: str = "not_applicable",
) -> dict[str, Any]:
    """The fallback block: which path ran, and whether audio was already audible."""
    return {
        "occurred": bool(occurred),
        "requested_path_outcome": str(requested_path_outcome or ""),
        "fallback_path_ref": str(fallback_path_ref or "").strip(),
        "reason": _bounded_text(reason),
        "relative_to_audible": str(relative_to_audible or ""),
    }


def voice_tool_attempt(
    *,
    attempt_ref: str,
    action_class: str,
    confidence_state: str = "unobserved",
    confirmation_required: bool = False,
    confirmation_state: str = "unobserved",
    policy_decision: str = "unobserved",
    tool_result_ref: str = "",
    denial_reason: str = "",
) -> dict[str, Any]:
    """One spoken tool attempt, without the command, the arguments, or the output."""
    return {
        "attempt_ref": str(attempt_ref or "").strip(),
        "action_class": str(action_class or ""),
        "confidence_state": str(confidence_state or ""),
        "confirmation_required": bool(confirmation_required),
        "confirmation_state": str(confirmation_state or ""),
        "policy_decision": str(policy_decision or ""),
        "tool_result_ref": str(tool_result_ref or "").strip(),
        "denial_reason": _bounded_text(denial_reason),
    }


def build_realtime_voice_trial_receipt(
    *,
    trial_ref: str,
    connector_ref: str,
    connector_revision: str,
    profile_ref: str,
    test_condition: str,
    environment: Mapping[str, Any] | None = None,
    requested_stack: Mapping[str, Any] | None = None,
    observed_stack: Mapping[str, Any] | None = None,
    observed_stack_evidence: str = "unavailable",
    trial_started_at: str = "",
    trial_ended_at: str = "",
    turns: Sequence[Mapping[str, Any]] = (),
    fallback: Mapping[str, Any] | None = None,
    tool_safety: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Mint one trial receipt from what a host or connector reported, or refuse."""
    record = {
        "schema_version": REALTIME_VOICE_TRIAL_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "",
        "trial_ref": str(trial_ref or "").strip(),
        "connector_ref": str(connector_ref or "").strip(),
        "connector_revision": str(connector_revision or "").strip(),
        "profile_ref": str(profile_ref or "").strip(),
        "test_condition": str(test_condition or ""),
        "environment": _with_defaults(environment, ENVIRONMENT_FACETS, _ENVIRONMENT_DEFAULTS),
        "requested_stack": _with_defaults(requested_stack, VOICE_STACK_KEYS, _VOICE_STACK_DEFAULTS),
        "observed_stack": _with_defaults(observed_stack, VOICE_STACK_KEYS, _VOICE_STACK_DEFAULTS),
        "observed_stack_evidence": str(observed_stack_evidence or ""),
        "trial_started_at": str(trial_started_at or "").strip(),
        "trial_ended_at": str(trial_ended_at or "").strip(),
        "turns": [dict(turn) for turn in turns],
        "fallback": dict(fallback) if fallback is not None else voice_fallback(),
        "tool_safety": [dict(attempt) for attempt in tool_safety],
        "claim_boundary": RECEIPT_CLAIM_BOUNDARY,
    }
    record["receipt_id"] = realtime_voice_trial_receipt_id(record)
    errors = validate_realtime_voice_trial_receipt(record)
    if errors:
        raise RealtimeVoiceTrialError(errors[0])
    return record


def realtime_voice_trial_receipt_id(record: Mapping[str, Any]) -> str:
    """Stable identity of one trial: a digest of what the trial was."""
    return "rvt_" + realtime_voice_trial_receipt_fingerprint(
        {key: record.get(key) for key in _IDENTITY_KEYS}
    )[:24]


def realtime_voice_trial_receipt_fingerprint(record: Mapping[str, Any]) -> str:
    """Content digest a consumer compares to prove it preserved a receipt."""
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_realtime_voice_trial_receipt(record: Any) -> list[str]:
    """Every contract violation in one receipt, structural faults first."""
    if not isinstance(record, Mapping):
        return [f"{_LABEL} must be an object"]
    errors = _key_set_errors(record, REALTIME_VOICE_TRIAL_RECEIPT_KEYS, _LABEL)
    if record.get("schema_version") != REALTIME_VOICE_TRIAL_RECEIPT_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {REALTIME_VOICE_TRIAL_RECEIPT_SCHEMA_VERSION}")
    if errors:
        return errors
    for key in ("trial_ref", "connector_ref", "connector_revision", "profile_ref"):
        errors.extend(_opaque_ref_errors(record[key], f"{_LABEL} {key}", required=True))
    errors.extend(_vocabulary_errors(record, "test_condition", TEST_CONDITIONS, _LABEL))
    errors.extend(_vocabulary_errors(record, "observed_stack_evidence", OBSERVED_STACK_EVIDENCE, _LABEL))
    if record["claim_boundary"] != RECEIPT_CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the receipt boundary")
    errors.extend(_stamp_errors(record))
    errors.extend(_environment_errors(record))
    errors.extend(_stack_errors(record))
    if errors:
        return errors
    errors.extend(_turns_errors(record))
    errors.extend(_fallback_errors(record))
    errors.extend(_tool_safety_errors(record))
    if errors:
        return errors
    if record["receipt_id"] != realtime_voice_trial_receipt_id(record):
        errors.append(f"{_LABEL} receipt_id does not match the trial it names")
    return errors


def preserve_realtime_voice_trial_receipt(record: Any) -> dict[str, Any]:
    """A validated deep copy for a consumer that carries a receipt unchanged."""
    errors = validate_realtime_voice_trial_receipt(record)
    if errors:
        raise RealtimeVoiceTrialError(errors[0])
    return json.loads(json.dumps(record, sort_keys=True))


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------


def answer_realtime_voice_readiness(
    receipt: Mapping[str, Any],
    *,
    connector_revision: str = "",
    profile_ref: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Summarize one receipt into a per-dimension readiness answer.

    The caller may pin the connector revision and profile it is deciding
    about. A receipt from another build or another profile is a real
    observation of something else, so it blocks rather than passing quietly;
    the same holds once the trial is older than the freshness horizon.
    """
    errors = validate_realtime_voice_trial_receipt(receipt)
    if errors:
        raise RealtimeVoiceTrialError(errors[0])
    binding = _binding_block_reason(
        receipt,
        connector_revision=connector_revision,
        profile_ref=profile_ref,
        now=now,
    )
    if binding:
        dimensions = {name: _dimension("block", [binding]) for name in READINESS_DIMENSIONS}
    else:
        dimensions = {
            "turn_integrity": _turn_integrity_dimension(receipt),
            "latency": _latency_dimension(receipt),
            "fallback": _fallback_dimension(receipt),
            "interruption": _interruption_dimension(receipt),
            "tool_safety": _tool_safety_dimension(receipt),
        }
    gate = _evidence_gate(receipt)
    verdict, verdict_reasons = _verdict(dimensions, gate, binding)
    answer = {
        "schema_version": REALTIME_VOICE_READINESS_SCHEMA_VERSION,
        "receipt_id": receipt["receipt_id"],
        "trial_ref": receipt["trial_ref"],
        "connector_ref": receipt["connector_ref"],
        "connector_revision": receipt["connector_revision"],
        "test_condition": receipt["test_condition"],
        "dimensions": dimensions,
        "latency": _latency_summary(receipt),
        "fallback": _fallback_summary(receipt),
        "evidence_gate": gate,
        "states": _readiness_states(receipt, dimensions=dimensions, binding=binding),
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "claim_boundary": READINESS_CLAIM_BOUNDARY,
    }
    validation = realtime_voice_readiness_errors(answer)
    if validation:
        raise RealtimeVoiceTrialError(validation[0])
    return answer


def realtime_voice_readiness_errors(answer: Any) -> list[str]:
    """Every contract violation in a readiness answer."""
    label = "realtime_voice_readiness"
    if not isinstance(answer, Mapping):
        return [f"{label} must be an object"]
    errors = _key_set_errors(answer, REALTIME_VOICE_READINESS_KEYS, label)
    if answer.get("schema_version") != REALTIME_VOICE_READINESS_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {REALTIME_VOICE_READINESS_SCHEMA_VERSION}")
    if errors:
        return errors
    dimensions = answer["dimensions"]
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(READINESS_DIMENSIONS):
        errors.append(f"{label} dimensions must carry exactly {', '.join(READINESS_DIMENSIONS)}")
    else:
        for name, dimension in dimensions.items():
            if not isinstance(dimension, Mapping) or set(dimension) != {"state", "reasons"}:
                errors.append(f"{label} dimension {name} carries exactly state and reasons")
                continue
            if dimension["state"] not in READINESS_STATES:
                errors.append(f"{label} dimension {name} state must be one of {', '.join(READINESS_STATES)}")
            if not isinstance(dimension["reasons"], list):
                errors.append(f"{label} dimension {name} reasons must be a list")
    states = answer["states"]
    if not isinstance(states, Mapping) or set(states) != set(READINESS_STATE_KEYS):
        errors.append(f"{label} states must carry exactly {', '.join(READINESS_STATE_KEYS)}")
    elif any(not isinstance(value, bool) for value in states.values()):
        errors.append(f"{label} every state is a boolean")
    if answer["verdict"] not in READINESS_VERDICTS:
        errors.append(f"{label} verdict must be one of {', '.join(READINESS_VERDICTS)}")
    if answer["claim_boundary"] != READINESS_CLAIM_BOUNDARY:
        errors.append(f"{label} claim_boundary must state the readiness boundary")
    if errors:
        return errors
    if answer["verdict"] == "ready" and answer["test_condition"] != "actual_environment":
        errors.append(f"{label} a synthetic trial is never ready for a real room, microphone, or network")
    if answer["verdict"] == "ready" and answer["evidence_gate"]["actual_environment_facets_missing"]:
        errors.append(f"{label} ready requires the intended room, microphone, network, and transport")
    if answer["verdict"] == "ready" and not answer["evidence_gate"]["requested_stack_observed"]:
        errors.append(f"{label} ready requires the requested stack to be the one that was observed")
    if answer["verdict"] == "ready" and any(
        dimension["state"] != "pass" for dimension in answer["dimensions"].values()
    ):
        errors.append(f"{label} ready requires every dimension to pass")
    if answer["verdict"] == "ready" and answer["fallback"]["occurred"]:
        errors.append(f"{label} a fallback path succeeding never makes the requested path ready")
    return errors


def build_prepared_realtime_voice_chat_state() -> dict[str, Any]:
    """The chat-state block a prepared voice-connector card carries.

    Every state starts false because a prepared card observed nothing. The
    card names the receipt it would need and the six states a reader must keep
    apart; it never produces one.
    """
    return {
        "receipt_schema": REALTIME_VOICE_TRIAL_RECEIPT_SCHEMA_VERSION,
        "turn_schema": REALTIME_VOICE_TURN_SCHEMA_VERSION,
        "readiness_schema": REALTIME_VOICE_READINESS_SCHEMA_VERSION,
        "readiness_dimensions": list(READINESS_DIMENSIONS),
        "states": {key: False for key in READINESS_STATE_KEYS},
        "trial_execution": "authorized_host_or_connector",
        "network_action": "none",
        "stale_after_seconds": REALTIME_VOICE_TRIAL_STALE_AFTER_SECONDS,
    }


# --------------------------------------------------------------------------
# Dimension evaluation
# --------------------------------------------------------------------------


def _dimension(state: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {"state": state, "reasons": list(dict.fromkeys(reasons))}


def _turn_integrity_dimension(receipt: Mapping[str, Any]) -> dict[str, Any]:
    turns = receipt["turns"]
    if not turns:
        return _dimension("hold", ["no_turn_observed"])
    failures: list[str] = []
    unobserved: list[str] = []
    for turn in turns:
        failures.extend(f"{turn['turn_id']}: {failure}" for failure in turn_integrity_failures(turn))
        unobserved.extend(
            f"{turn['turn_id']}: {key}_unobserved"
            for key in TURN_INTEGRITY_KEYS
            if turn["integrity"].get(key) == "unobserved"
        )
    if failures:
        return _dimension("block", failures)
    if unobserved:
        return _dimension("hold", unobserved)
    return _dimension("pass", ["every_turn_intact"])


def _latency_dimension(receipt: Mapping[str, Any]) -> dict[str, Any]:
    turns = receipt["turns"]
    if not turns:
        return _dimension("hold", ["no_turn_observed"])
    readings = [(turn["turn_id"], turn_latency(turn)) for turn in turns]
    reasons = [f"{turn_id}: {reading['basis']}" for turn_id, reading in readings if not reading["eligible"]]
    if reasons:
        return _dimension("hold", reasons)
    return _dimension("pass", ["every_turn_reported_latency_against_one_timing_reference"])


def _fallback_dimension(receipt: Mapping[str, Any]) -> dict[str, Any]:
    fallback = receipt["fallback"]
    if not fallback["occurred"]:
        if fallback["requested_path_outcome"] == "unobserved":
            return _dimension("hold", ["requested_path_outcome_unobserved"])
        return _dimension("pass", ["requested_path_served_the_trial"])
    reasons = [f"requested_path_failed: {fallback['reason']}", f"fallback_path_served: {fallback['fallback_path_ref']}"]
    if fallback["relative_to_audible"] == "after_audible":
        reasons.append("fallback_after_audio_became_audible")
    if fallback["relative_to_audible"] == "unobserved":
        reasons.append("fallback_timing_relative_to_audible_unobserved")
    return _dimension("block", reasons)


def _interruption_dimension(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Interruption is a trial-level question, not a per-turn one.

    A turn nobody interrupted says nothing either way, so the dimension asks
    whether the trial exercised barge-in at all, and separately whether the
    turns that actually produced audible output reported what happened to it.
    """
    turns = receipt["turns"]
    if not turns:
        return _dimension("hold", ["no_turn_observed"])
    failures: list[str] = []
    holds: list[str] = []
    for turn in turns:
        failures.extend(f"{turn['turn_id']}: {failure}" for failure in turn_interruption_failures(turn))
        if turn["milestones"]["first_output_audio"] is None:
            continue
        streaming = turn["streaming"]
        if streaming["replay"] == "unobserved":
            holds.append(f"{turn['turn_id']}: replay_after_audible_output_unobserved")
        if streaming["backpressure"] == "unobserved":
            holds.append(f"{turn['turn_id']}: backpressure_unobserved")
    exercised = [turn for turn in turns if turn["streaming"]["barge_in"] in ("honored", "ignored")]
    unsupported = [turn for turn in turns if turn["streaming"]["barge_in"] == "unsupported"]
    if failures:
        return _dimension("block", failures)
    if unsupported:
        holds.append(f"{unsupported[0]['turn_id']}: barge_in_unsupported")
    elif not exercised:
        holds.append("barge_in_never_exercised_in_this_trial")
    if holds:
        return _dimension("hold", holds)
    return _dimension("pass", ["interruption_and_streaming_behavior_observed"])


def _tool_safety_dimension(receipt: Mapping[str, Any]) -> dict[str, Any]:
    attempts = receipt["tool_safety"]
    if not attempts:
        return _dimension("hold", ["no_spoken_tool_attempt_observed"])
    failures: list[str] = []
    holds: list[str] = []
    for attempt in attempts:
        ref = attempt["attempt_ref"]
        if attempt["action_class"] == "high_impact" and attempt["policy_decision"] in ("unobserved", "deferred"):
            failures.append(f"{ref}: high_impact_attempt_without_an_observed_decision")
        if attempt["action_class"] == "high_impact" and attempt["confirmation_state"] == "unobserved":
            failures.append(f"{ref}: high_impact_attempt_without_an_observed_confirmation")
        if attempt["action_class"] != "high_impact" and attempt["policy_decision"] == "unobserved":
            holds.append(f"{ref}: policy_decision_unobserved")
        if attempt["confidence_state"] == "unobserved":
            holds.append(f"{ref}: confidence_state_unobserved")
    if failures:
        return _dimension("block", failures)
    if holds:
        return _dimension("hold", holds)
    return _dimension("pass", ["every_spoken_tool_attempt_carries_an_observed_decision"])


def _evidence_gate(receipt: Mapping[str, Any]) -> dict[str, Any]:
    missing = [facet for facet in ENVIRONMENT_FACETS if receipt["environment"][facet] != "intended"]
    synthetic = receipt["test_condition"] == "synthetic_fixture"
    return {
        "test_condition": receipt["test_condition"],
        "synthetic_only": synthetic,
        "actual_environment_facets_missing": missing,
        "observed_stack_evidence": receipt["observed_stack_evidence"],
        "requested_stack_observed": _requested_stack_observed(receipt),
        "caps_verdict_at_hold": bool(synthetic or missing or not _requested_stack_observed(receipt)),
    }


def _requested_stack_observed(receipt: Mapping[str, Any]) -> bool:
    """Whether the stack that was asked for is the stack somebody saw.

    An `unavailable` observed block cannot answer this, which is the point:
    without it, a receipt would say the requested model was ready on the
    strength of having requested it.
    """
    if receipt["observed_stack_evidence"] == "unavailable":
        return False
    requested = receipt["requested_stack"]
    observed = receipt["observed_stack"]
    for key in ("provider", "model", "transport"):
        if not requested[key]:
            continue
        if observed[key] != requested[key]:
            return False
    return True


def _verdict(
    dimensions: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    binding: str,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if binding:
        return "blocked", [binding]
    blocked = [name for name, dimension in dimensions.items() if dimension["state"] == "block"]
    held = [name for name, dimension in dimensions.items() if dimension["state"] == "hold"]
    if blocked:
        return "blocked", [f"{name}_blocked" for name in blocked]
    reasons.extend(f"{name}_held" for name in held)
    if gate["synthetic_only"]:
        reasons.append("synthetic_evidence_only")
    for facet in gate["actual_environment_facets_missing"]:
        reasons.append(f"{facet}_not_the_intended_one")
    if not gate["requested_stack_observed"]:
        reasons.append("requested_stack_was_not_observed")
    if reasons:
        return "hold", reasons
    return "ready", ["every_dimension_passed_in_the_intended_environment"]


def _readiness_states(
    receipt: Mapping[str, Any],
    *,
    dimensions: Mapping[str, Mapping[str, Any]],
    binding: str,
) -> dict[str, bool]:
    turns = receipt["turns"]
    no_block = not binding and all(dimension["state"] != "block" for dimension in dimensions.values())
    turn_observed = any(
        any(evidence in ("measured", "producer_reported") for evidence in turn["milestone_evidence"].values())
        for turn in turns
    )
    tool_observed = any(attempt["policy_decision"] != "unobserved" for attempt in receipt["tool_safety"])
    synthetic = receipt["test_condition"] == "synthetic_fixture"
    intended = not [facet for facet in ENVIRONMENT_FACETS if receipt["environment"][facet] != "intended"]
    return {
        "connector_configured": bool(receipt["connector_ref"] and receipt["connector_revision"]),
        "synthetic_trial_passed": bool(synthetic and no_block and turns),
        "actual_environment_trial_passed": bool(not synthetic and intended and no_block and turns),
        "voice_turn_observed": turn_observed,
        "tool_action_observed": tool_observed,
        # The call reached its end, which is a different fact from every other
        # state here: an interrupted turn is normal conversation, a failed or
        # unreported one means the session did not finish.
        "session_completed": bool(
            receipt["trial_ended_at"]
            and turns
            and all(turn["terminal_status"] not in ("failed", "unobserved") for turn in turns)
        ),
    }


def _latency_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    readings = [turn_latency(turn) for turn in receipt["turns"]]
    eligible = [reading for reading in readings if reading["eligible"]]
    if not eligible:
        return {
            "turn_count": len(readings),
            "eligible_turn_count": 0,
            "first_audible_output_ms_max": None,
            "final_transcript_ms_max": None,
            "basis": "no_turn_earned_a_latency_reading",
        }
    return {
        "turn_count": len(readings),
        "eligible_turn_count": len(eligible),
        "first_audible_output_ms_max": max(int(reading["first_audible_output_ms"]) for reading in eligible),
        "final_transcript_ms_max": max(int(reading["final_transcript_ms"]) for reading in eligible),
        "basis": "measured",
    }


def _fallback_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    fallback = receipt["fallback"]
    return {
        "occurred": fallback["occurred"],
        "requested_path_outcome": fallback["requested_path_outcome"],
        "requested_path_ready": not fallback["occurred"] and fallback["requested_path_outcome"] == "succeeded",
        "fallback_path_ref": fallback["fallback_path_ref"],
        "relative_to_audible": fallback["relative_to_audible"],
    }


def _binding_block_reason(
    receipt: Mapping[str, Any],
    *,
    connector_revision: str,
    profile_ref: str,
    now: str,
) -> str:
    if connector_revision and receipt["connector_revision"] != connector_revision:
        return f"trial_observed_another_connector_revision: {receipt['connector_revision']}"
    if profile_ref and receipt["profile_ref"] != profile_ref:
        return f"trial_observed_another_profile: {receipt['profile_ref']}"
    if not now:
        return ""
    reference = _parse_stamp(now)
    ended = _parse_stamp(receipt["trial_ended_at"])
    if reference is None:
        raise RealtimeVoiceTrialError("realtime_voice_readiness now must be an ISO-8601 timestamp")
    if ended is None:
        return "trial_end_time_unobserved"
    if (reference - ended).total_seconds() > REALTIME_VOICE_TRIAL_STALE_AFTER_SECONDS:
        return f"trial_is_stale_after_{REALTIME_VOICE_TRIAL_STALE_AFTER_SECONDS}_seconds"
    return ""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


_TURN_INTEGRITY_DEFAULTS: Final = {
    "onset": "unobserved",
    "segmentation": "unobserved",
    "dispatch_count": 0,
    "input_completeness": "unobserved",
    "max_duration_behavior": "unobserved",
    "mute_resume": "unobserved",
    "transcript_outcome": "unobserved",
}
_TURN_STREAMING_DEFAULTS: Final = {
    "barge_in": "unobserved",
    "queued_audio": "unobserved",
    "replay": "unobserved",
    "backpressure": "unobserved",
}
_ENVIRONMENT_DEFAULTS: Final = dict.fromkeys(ENVIRONMENT_FACETS, "unobserved")
_VOICE_STACK_DEFAULTS: Final = {
    **dict.fromkeys(_VOICE_STACK_REF_KEYS, ""),
    **dict.fromkeys(_VOICE_STACK_INT_KEYS, None),
}

_TURN_INTEGRITY_VOCABULARIES: Final = (
    ("onset", ONSET_STATES),
    ("segmentation", SEGMENTATION_STATES),
    ("input_completeness", INPUT_COMPLETENESS_STATES),
    ("max_duration_behavior", MAX_DURATION_BEHAVIORS),
    ("mute_resume", MUTE_RESUME_STATES),
    ("transcript_outcome", TRANSCRIPT_OUTCOMES),
)
_TURN_STREAMING_VOCABULARIES: Final = (
    ("barge_in", BARGE_IN_STATES),
    ("queued_audio", QUEUED_AUDIO_STATES),
    ("replay", REPLAY_STATES),
    ("backpressure", BACKPRESSURE_STATES),
)
_TOOL_SAFETY_VOCABULARIES: Final = (
    ("action_class", ACTION_CLASSES),
    ("confidence_state", CONFIDENCE_STATES),
    ("confirmation_state", CONFIRMATION_STATES),
    ("policy_decision", POLICY_DECISIONS),
)


def _with_defaults(supplied: Mapping[str, Any] | None, keys: Sequence[str], defaults: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(supplied or {})
    resolved = {key: values.get(key, defaults[key]) for key in keys}
    for extra in sorted(set(values) - set(keys)):
        resolved[extra] = values[extra]
    return resolved


def _key_set_errors(record: Mapping[str, Any], keys: Sequence[str], label: str) -> list[str]:
    errors: list[str] = []
    extra = sorted(set(record) - set(keys))
    if extra:
        errors.append(f"{label} has unsupported keys: {extra}")
    missing = sorted(set(keys) - set(record))
    if missing:
        errors.append(f"{label} is missing keys: {missing}")
    return errors


def _vocabulary_errors(record: Mapping[str, Any], key: str, allowed: Sequence[str], label: str) -> list[str]:
    if record.get(key) not in allowed:
        return [f"{label} {key} must be one of {', '.join(allowed)}"]
    return []


def _opaque_ref_errors(value: Any, field: str, *, required: bool) -> list[str]:
    if not isinstance(value, str):
        return [f"{field} must be a string"]
    if not value:
        return [f"{field} is required"] if required else []
    if is_raw_pii_shaped(value):
        return [f"{field} must not carry a raw phone number, address, or participant identity"]
    try:
        require_opaque_metadata_ref(value, field=field)
    except ValueError as exc:
        return [str(exc)]
    return []


def _bounded_text(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_REASON_CHARS]


def _bounded_text_errors(value: Any, field: str) -> list[str]:
    if not isinstance(value, str):
        return [f"{field} must be a string"]
    if len(value) > MAX_REASON_CHARS:
        return [f"{field} exceeds {MAX_REASON_CHARS} characters"]
    if is_body_shaped_metadata_text(value, limit=MAX_REASON_CHARS):
        return [f"{field} must be one bounded line, never a transcript, payload, or log body"]
    if is_sensitive_metadata_text(value) or is_raw_pii_shaped(value):
        return [f"{field} must not carry credentials, phone numbers, or participant identities"]
    return []


def _turn_milestone_errors(record: Mapping[str, Any]) -> list[str]:
    milestones = record["milestones"]
    evidence = record["milestone_evidence"]
    errors: list[str] = []
    if not isinstance(milestones, Mapping) or set(milestones) != set(TURN_MILESTONES):
        errors.append(f"{_TURN_LABEL} milestones must carry exactly {', '.join(TURN_MILESTONES)}")
    if not isinstance(evidence, Mapping) or set(evidence) != set(TURN_MILESTONES):
        errors.append(f"{_TURN_LABEL} milestone_evidence must carry exactly {', '.join(TURN_MILESTONES)}")
    if errors:
        return errors
    for name in TURN_MILESTONES:
        value = milestones[name]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            errors.append(f"{_TURN_LABEL} milestone {name} must be a non-negative integer millisecond offset or null")
        if evidence[name] not in EVIDENCE_CLASSES:
            errors.append(f"{_TURN_LABEL} milestone_evidence {name} must be one of {', '.join(EVIDENCE_CLASSES)}")
    if errors:
        return errors
    for name in TURN_MILESTONES:
        if (milestones[name] is None) != (evidence[name] == "unavailable"):
            errors.append(f"{_TURN_LABEL} milestone {name} and its evidence class must agree on whether it was observed")
    if record["timing_reference"] == "unavailable" and any(milestones[name] is not None for name in TURN_MILESTONES):
        errors.append(f"{_TURN_LABEL} milestones need one declared timing reference; unavailable forbids all of them")
    present = [(name, milestones[name]) for name in TURN_MILESTONES if milestones[name] is not None]
    for (earlier, earlier_ms), (later, later_ms) in zip(present, present[1:]):
        if int(later_ms) < int(earlier_ms):
            errors.append(f"{_TURN_LABEL} milestone {later} precedes {earlier}; the sequence is not monotonic")
    return errors


def _turn_integrity_errors(record: Mapping[str, Any]) -> list[str]:
    integrity = record["integrity"]
    if not isinstance(integrity, Mapping) or set(integrity) != set(TURN_INTEGRITY_KEYS):
        return [f"{_TURN_LABEL} integrity must carry exactly {', '.join(TURN_INTEGRITY_KEYS)}"]
    errors: list[str] = []
    for key, allowed in _TURN_INTEGRITY_VOCABULARIES:
        errors.extend(_vocabulary_errors(integrity, key, allowed, f"{_TURN_LABEL} integrity"))
    count = integrity["dispatch_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        errors.append(f"{_TURN_LABEL} integrity dispatch_count must be a non-negative integer")
    return errors


def _turn_streaming_errors(record: Mapping[str, Any]) -> list[str]:
    streaming = record["streaming"]
    if not isinstance(streaming, Mapping) or set(streaming) != set(TURN_STREAMING_KEYS):
        return [f"{_TURN_LABEL} streaming must carry exactly {', '.join(TURN_STREAMING_KEYS)}"]
    errors: list[str] = []
    for key, allowed in _TURN_STREAMING_VOCABULARIES:
        errors.extend(_vocabulary_errors(streaming, key, allowed, f"{_TURN_LABEL} streaming"))
    return errors


def _latency_basis(record: Mapping[str, Any]) -> str:
    if validate_realtime_voice_turn(record):
        return "turn_is_not_a_valid_record"
    if record["timing_reference"] == "unavailable":
        return "no_declared_timing_reference"
    milestones = record["milestones"]
    if any(milestones[name] is None for name in LATENCY_REQUIRED_MILESTONES):
        return "incomplete_milestone_sequence"
    if record["integrity"]["segmentation"] != "intact":
        return f"segmentation_{record['integrity']['segmentation']}"
    return "measured"


def _stamp_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("trial_started_at", "trial_ended_at"):
        value = record[key]
        if not isinstance(value, str):
            errors.append(f"{_LABEL} {key} must be a string")
            continue
        if value and _parse_stamp(value) is None:
            errors.append(f"{_LABEL} {key} must be an ISO-8601 timestamp")
    if errors:
        return errors
    started = _parse_stamp(record["trial_started_at"])
    ended = _parse_stamp(record["trial_ended_at"])
    if started is not None and ended is not None and ended < started:
        errors.append(f"{_LABEL} trial_ended_at must not precede trial_started_at")
    return errors


def _environment_errors(record: Mapping[str, Any]) -> list[str]:
    environment = record["environment"]
    if not isinstance(environment, Mapping) or set(environment) != set(ENVIRONMENT_FACETS):
        return [f"{_LABEL} environment must carry exactly {', '.join(ENVIRONMENT_FACETS)}"]
    errors: list[str] = []
    for facet in ENVIRONMENT_FACETS:
        errors.extend(_vocabulary_errors(environment, facet, ENVIRONMENT_COVERAGE, f"{_LABEL} environment"))
    if errors:
        return errors
    if record["test_condition"] == "synthetic_fixture" and any(
        environment[facet] == "intended" for facet in ENVIRONMENT_FACETS
    ):
        errors.append(f"{_LABEL} a synthetic fixture never observed the intended room, microphone, network, or transport")
    return errors


def _stack_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for side in ("requested_stack", "observed_stack"):
        stack = record[side]
        if not isinstance(stack, Mapping) or set(stack) != set(VOICE_STACK_KEYS):
            errors.append(f"{_LABEL} {side} must carry exactly {', '.join(VOICE_STACK_KEYS)}")
            continue
        for key in _VOICE_STACK_REF_KEYS:
            errors.extend(_opaque_ref_errors(stack[key], f"{_LABEL} {side} {key}", required=False))
        for key in _VOICE_STACK_INT_KEYS:
            value = stack[key]
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                errors.append(f"{_LABEL} {side} {key} must be a positive integer or null")
    if errors:
        return errors
    if not any(record["requested_stack"][key] for key in _VOICE_STACK_REF_KEYS):
        errors.append(f"{_LABEL} requested_stack names at least one requested setting")
    if record["observed_stack_evidence"] == "unavailable" and _stack_is_populated(record["observed_stack"]):
        errors.append(
            f"{_LABEL} observed_stack must be empty when observed_stack_evidence is unavailable; "
            "a requested setting is not an observed one"
        )
    if record["observed_stack_evidence"] != "unavailable" and not _stack_is_populated(record["observed_stack"]):
        errors.append(f"{_LABEL} observed_stack_evidence {record['observed_stack_evidence']} names at least one observed setting")
    return errors


def _stack_is_populated(stack: Mapping[str, Any]) -> bool:
    return any(stack[key] for key in _VOICE_STACK_REF_KEYS) or any(
        stack[key] is not None for key in _VOICE_STACK_INT_KEYS
    )


def _turns_errors(record: Mapping[str, Any]) -> list[str]:
    turns = record["turns"]
    if not isinstance(turns, list):
        return [f"{_LABEL} turns must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, turn in enumerate(turns):
        turn_errors = validate_realtime_voice_turn(turn)
        errors.extend(f"{_LABEL} turns[{index}] {issue}" for issue in turn_errors)
        if turn_errors:
            continue
        turn_id = turn["turn_id"]
        if turn_id in seen:
            errors.append(f"{_LABEL} turns[{index}] repeats turn_id {turn_id}; one accepted utterance is one turn")
        seen.add(turn_id)
    return errors


def _fallback_errors(record: Mapping[str, Any]) -> list[str]:
    fallback = record["fallback"]
    if not isinstance(fallback, Mapping) or set(fallback) != set(FALLBACK_KEYS):
        return [f"{_LABEL} fallback must carry exactly {', '.join(FALLBACK_KEYS)}"]
    errors: list[str] = []
    if not isinstance(fallback["occurred"], bool):
        errors.append(f"{_LABEL} fallback occurred must be a boolean")
    errors.extend(_vocabulary_errors(fallback, "requested_path_outcome", REQUESTED_PATH_OUTCOMES, f"{_LABEL} fallback"))
    errors.extend(_vocabulary_errors(fallback, "relative_to_audible", FALLBACK_AUDIBLE_RELATIONS, f"{_LABEL} fallback"))
    errors.extend(_opaque_ref_errors(fallback["fallback_path_ref"], f"{_LABEL} fallback fallback_path_ref", required=False))
    errors.extend(_bounded_text_errors(fallback["reason"], f"{_LABEL} fallback reason"))
    if errors:
        return errors
    if not fallback["occurred"]:
        if fallback["fallback_path_ref"] or fallback["reason"]:
            errors.append(f"{_LABEL} fallback names a path and a reason only when a fallback occurred")
        if fallback["relative_to_audible"] != "not_applicable":
            errors.append(f"{_LABEL} fallback relative_to_audible is not_applicable when no fallback occurred")
        return errors
    if fallback["requested_path_outcome"] == "succeeded":
        errors.append(
            f"{_LABEL} a fallback path succeeding is never success for the requested voice stack, provider, or model"
        )
    if not fallback["fallback_path_ref"]:
        errors.append(f"{_LABEL} an occurred fallback names the path that served the trial")
    if not fallback["reason"]:
        errors.append(f"{_LABEL} an occurred fallback names why the requested path did not serve it")
    if fallback["relative_to_audible"] == "not_applicable":
        errors.append(f"{_LABEL} an occurred fallback records whether it happened before or after audio became audible")
    return errors


def _tool_safety_errors(record: Mapping[str, Any]) -> list[str]:
    attempts = record["tool_safety"]
    if not isinstance(attempts, list):
        return [f"{_LABEL} tool_safety must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, attempt in enumerate(attempts):
        label = f"{_LABEL} tool_safety[{index}]"
        if not isinstance(attempt, Mapping):
            errors.append(f"{label} must be an object")
            continue
        attempt_errors = _key_set_errors(attempt, TOOL_SAFETY_KEYS, label)
        if attempt_errors:
            errors.extend(attempt_errors)
            continue
        errors.extend(_opaque_ref_errors(attempt["attempt_ref"], f"{label} attempt_ref", required=True))
        errors.extend(_opaque_ref_errors(attempt["tool_result_ref"], f"{label} tool_result_ref", required=False))
        errors.extend(_bounded_text_errors(attempt["denial_reason"], f"{label} denial_reason"))
        for key, allowed in _TOOL_SAFETY_VOCABULARIES:
            errors.extend(_vocabulary_errors(attempt, key, allowed, label))
        if not isinstance(attempt["confirmation_required"], bool):
            errors.append(f"{label} confirmation_required must be a boolean")
        if errors:
            continue
        if attempt["attempt_ref"] in seen:
            errors.append(f"{label} repeats attempt_ref {attempt['attempt_ref']}")
        seen.add(attempt["attempt_ref"])
        errors.extend(_tool_attempt_policy_errors(attempt, label))
    return errors


def _tool_attempt_policy_errors(attempt: Mapping[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if attempt["action_class"] == "high_impact" and not attempt["confirmation_required"]:
        errors.append(f"{label} a high-impact spoken action always requires confirmation")
    if attempt["confirmation_required"] and attempt["confirmation_state"] == "not_required":
        errors.append(f"{label} confirmation_state not_required contradicts confirmation_required")
    if not attempt["confirmation_required"] and attempt["confirmation_state"] in ("received", "refused"):
        errors.append(f"{label} a confirmation that was not required was not the one that authorized this")
    if attempt["policy_decision"] == "allowed":
        if attempt["confidence_state"] != "confident":
            errors.append(
                f"{label} an ambiguous or unobserved spoken command cannot be reported as authorized execution"
            )
        if attempt["confirmation_required"] and attempt["confirmation_state"] != "received":
            errors.append(
                f"{label} a missing or refused confirmation cannot be reported as authorized execution"
            )
    if attempt["policy_decision"] != "allowed" and attempt["tool_result_ref"]:
        errors.append(f"{label} a tool result reference exists only for an allowed decision")
    if attempt["policy_decision"] != "denied" and attempt["denial_reason"]:
        errors.append(f"{label} a denial reason exists only for a denied decision")
    if attempt["policy_decision"] == "denied" and not attempt["denial_reason"]:
        errors.append(f"{label} a denied decision names why it was denied")
    return errors


# --------------------------------------------------------------------------
# Demo fixtures
# --------------------------------------------------------------------------


def _demo_turn(
    turn_id: str,
    *,
    offsets: Sequence[int | None],
    terminal_status: str,
    barge_in: str,
    replay: str = "none",
    queued_audio: str = "not_applicable",
    onset: str = "preserved",
    segmentation: str = "intact",
) -> dict[str, Any]:
    return build_realtime_voice_turn(
        turn_id=turn_id,
        timing_reference="host_monotonic",
        milestones=dict(zip(TURN_MILESTONES, offsets)),
        terminal_status=terminal_status,
        integrity={
            "onset": onset,
            "segmentation": segmentation,
            "dispatch_count": 1,
            "input_completeness": "complete",
            "max_duration_behavior": "not_reached",
            "mute_resume": "not_exercised",
            "transcript_outcome": "final",
        },
        streaming={
            "barge_in": barge_in,
            "queued_audio": queued_audio,
            "replay": replay,
            "backpressure": "none",
        },
    )


def demo_realtime_voice_trials() -> dict[str, dict[str, Any]]:
    """Three named receipts across two transports, provider-neutral by shape.

    `webrtc_actual_environment` and `telephony_streaming_fallback` differ in
    provider, model, transport, codec, and endpoint detector and share every
    line of validation and verdict code: nothing in this module branches on
    which stack a trial used. The third is the same wiring proven only
    synthetically, kept so a reader can see what a fixture cannot claim.
    """
    ready = build_realtime_voice_trial_receipt(
        trial_ref="trial-webrtc-01",
        connector_ref="voice-connector-a",
        connector_revision="build-9f2c14",
        profile_ref="support-desk",
        test_condition="actual_environment",
        environment=dict.fromkeys(ENVIRONMENT_FACETS, "intended"),
        requested_stack=voice_stack(
            provider="provider-a",
            model="voice-model-a",
            voice="voice-neutral",
            transport="webrtc",
            codec="opus",
            sample_rate_hz=48000,
            channels=1,
            endpoint_detector="server-vad",
            endpoint_config_ref="vad-profile-1",
        ),
        observed_stack=voice_stack(
            provider="provider-a",
            model="voice-model-a",
            voice="voice-neutral",
            transport="webrtc",
            codec="opus",
            sample_rate_hz=48000,
            channels=1,
            endpoint_detector="server-vad",
            endpoint_config_ref="vad-profile-1",
        ),
        observed_stack_evidence="measured",
        trial_started_at="2026-09-09T09:00:00Z",
        trial_ended_at="2026-09-09T09:04:00Z",
        turns=(
            _demo_turn("turn-1", offsets=(0, 1400, 1520, 1610, 1780, 4200), terminal_status="completed", barge_in="not_exercised"),
            _demo_turn(
                "turn-2",
                offsets=(6000, 7100, 7240, 7300, 7460, None),
                terminal_status="interrupted",
                barge_in="honored",
                queued_audio="discarded",
            ),
        ),
        tool_safety=(
            voice_tool_attempt(
                attempt_ref="attempt-1",
                action_class="read_only",
                confidence_state="confident",
                confirmation_state="not_required",
                policy_decision="allowed",
                tool_result_ref="result-1",
            ),
            voice_tool_attempt(
                attempt_ref="attempt-2",
                action_class="high_impact",
                confidence_state="ambiguous",
                confirmation_required=True,
                confirmation_state="refused",
                policy_decision="denied",
                denial_reason="spoken command was ambiguous and the confirmation was refused",
            ),
        ),
    )
    fallback = build_realtime_voice_trial_receipt(
        trial_ref="trial-telephony-01",
        connector_ref="voice-connector-b",
        connector_revision="build-3ad871",
        profile_ref="support-desk",
        test_condition="actual_environment",
        environment=dict.fromkeys(ENVIRONMENT_FACETS, "intended"),
        requested_stack=voice_stack(
            provider="provider-b",
            model="voice-model-b",
            transport="sip-telephony",
            codec="g711-ulaw",
            sample_rate_hz=8000,
            channels=1,
            endpoint_detector="gateway-endpointer",
        ),
        observed_stack=voice_stack(
            provider="provider-b",
            model="voice-model-b",
            transport="sip-telephony",
            codec="g711-ulaw",
            sample_rate_hz=8000,
            channels=1,
            endpoint_detector="gateway-endpointer",
        ),
        observed_stack_evidence="producer_reported",
        trial_started_at="2026-09-09T09:10:00Z",
        trial_ended_at="2026-09-09T09:13:00Z",
        turns=(
            _demo_turn(
                "turn-1",
                offsets=(0, 900, 1120, 1200, 1400, 5200),
                terminal_status="completed",
                barge_in="honored",
                replay="replayed_from_start",
                queued_audio="discarded",
            ),
        ),
        fallback=voice_fallback(
            occurred=True,
            requested_path_outcome="failed",
            fallback_path_ref="half-duplex-turn-taking",
            reason="realtime duplex stream dropped mid-response",
            relative_to_audible="after_audible",
        ),
    )
    synthetic = build_realtime_voice_trial_receipt(
        trial_ref="trial-synthetic-01",
        connector_ref="voice-connector-a",
        connector_revision="build-9f2c14",
        profile_ref="support-desk",
        test_condition="synthetic_fixture",
        environment=dict.fromkeys(ENVIRONMENT_FACETS, "simulated"),
        requested_stack=voice_stack(provider="provider-a", model="voice-model-a", transport="webrtc"),
        observed_stack=voice_stack(provider="provider-a", model="voice-model-a", transport="webrtc"),
        observed_stack_evidence="measured",
        trial_started_at="2026-09-09T08:00:00Z",
        trial_ended_at="2026-09-09T08:01:00Z",
        turns=(
            _demo_turn("turn-1", offsets=(0, 1000, 1100, 1150, 1300, 3000), terminal_status="completed", barge_in="honored"),
        ),
    )
    return {
        "webrtc_actual_environment": ready,
        "telephony_streaming_fallback": fallback,
        "synthetic_smoke": synthetic,
    }


def _parse_stamp(value: Any) -> datetime | None:
    """An aware UTC datetime from an ISO-8601 timestamp, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = strict_fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

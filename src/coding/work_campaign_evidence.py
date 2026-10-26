"""Inspect host-observed result identity, bounded artifacts and verification."""
from hashlib import sha256
from pathlib import Path
import re


def inspect_result(campaign, unit, observation):
    from .work_campaign_contract import CampaignError, scope_path

    if not isinstance(observation, dict) or not observation:
        raise CampaignError("verification_not_observed")
    for key, expected in (("campaign_id", campaign["campaign_id"]), ("attempt_id", unit["attempt_id"]),
                          ("task_id", unit["binding"]["task_id"]), ("workspace", campaign["workspace"])):
        if observation.get(key) != expected:
            raise CampaignError("result_identity_mismatch")
    artifacts = observation.get("artifacts")
    changed = observation.get("changed_paths")
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 16:
        raise CampaignError("artifacts_required")
    if not isinstance(changed, list) or not 1 <= len(changed) <= 64:
        raise CampaignError("observed_diff_required")
    changed = [scope_path(x) for x in changed]
    if any(x not in unit["file_scope"] for x in changed):
        raise CampaignError("changed_scope_violation")
    verified = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise CampaignError("invalid_artifact")
        relative = scope_path(item.get("path", ""))
        path = Path(campaign["workspace"]) / relative
        if relative not in unit["file_scope"] or path.is_symlink() or not path.resolve().is_relative_to(Path(campaign["workspace"])):
            raise CampaignError("artifact_outside_scope")
        with path.open("rb") as stream:
            content = stream.read(1048577)
        if len(content) > 1048576 or sha256(content).hexdigest() != item.get("sha256"):
            raise CampaignError("artifact_digest_mismatch")
        verified.append(dict(path=relative, sha256=item["sha256"]))
    if not set(unit["artifacts"]).issubset({x["path"] for x in verified}):
        raise CampaignError("required_artifact_missing")
    verification = observation.get("verification", {})
    if not isinstance(verification, dict) or verification.get("command") != unit["verification_command"]:
        raise CampaignError("verification_command_mismatch")
    if type(verification.get("exit_code")) is not int or verification["exit_code"] != 0:
        raise CampaignError("verification_failed")
    if type(verification.get("output_bytes")) is not int or not 1 <= verification["output_bytes"] <= 1048576:
        raise CampaignError("verification_output_required")
    for key, pattern in (("revision", r"[0-9a-f]{40,64}"), ("diff_digest", r"[0-9a-f]{64}")):
        if not re.fullmatch(pattern, str(observation.get(key, ""))) or verification.get(key) != observation[key]:
            raise CampaignError("verification_revision_mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(verification.get("output_digest", ""))):
        raise CampaignError("verification_output_required")
    return dict(artifacts=verified, revision=observation["revision"], diff_digest=observation["diff_digest"],
                verification={k: verification[k] for k in ("exit_code", "output_bytes", "output_digest")})

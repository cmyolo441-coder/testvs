"""Bounded canary reports; fixture protocol results are not provider accounting."""
from hashlib import sha256
import json
import math
import re
from pathlib import Path
from typing import Any

from ..quality.reported_rate import reported_rate


def campaign_canary_report(rows):
    """Read measurement_provenance receipt bindings, never model-route provenance."""
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError("canary_input_cap")
    report: dict[str, Any] = {"schema_version": "work_campaign_canary/v1", "recommendation": "remain_opt_in",
              "model_execution": "not_observed", "quality_benefit": "not_observed"}
    observed = []
    for row in rows:
        provenance = row.get("measurement_provenance", {})
        if (provenance.get("source") != "host_observation" or not provenance.get("record_path")
                or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("input_digest", "")))):
            continue
        with Path(provenance["record_path"]).open("rb") as stream:
            content = stream.read(16385)
        if len(content) > 16384 or sha256(content).hexdigest() != provenance.get("record_digest"):
            raise ValueError("canary_observation_digest_mismatch")
        receipt = json.loads(content)
        if receipt.get("schema_version") != "host_campaign_measurement/v1" or receipt.get("execution_source") != "host_runtime":
            continue
        if receipt.get("input_digest") != provenance["input_digest"] or not provenance.get("input_path"):
            raise ValueError("canary_input_provenance_mismatch")
        with Path(provenance["input_path"]).open("rb") as stream:
            accepted_input = stream.read(65537)
        if len(accepted_input) > 65536 or sha256(accepted_input).hexdigest() != provenance["input_digest"]:
            raise ValueError("canary_input_digest_mismatch")
        measurement = receipt.get("measurement", {})
        if measurement != {k: v for k, v in row.items() if k != "measurement_provenance"}:
            raise ValueError("canary_observation_mismatch")
        for key, value in measurement.items():
            if key not in ("arm", "state") and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("invalid_canary_measurement")
        if row.get("state") in ("complete", "failed", "cancelled", "unknown"):
            observed.append(row)
    paired = {r["measurement_provenance"]["input_digest"] for r in observed
              if {x.get("arm") for x in observed if x["measurement_provenance"]["input_digest"] == r["measurement_provenance"]["input_digest"]}
              == {"parent_led", "campaign"}}
    for arm in ("parent_led", "campaign"):
        selected = [r for r in observed if r.get("arm") == arm and r["measurement_provenance"]["input_digest"] in paired]
        report[arm] = dict(
            completion=reported_rate(numerator=sum(r.get("state") == "complete" for r in selected),
                                     denominator=len(selected), numerator_of=["complete"],
                                     denominator_of="observed_comparable_campaigns",
                                     excluded=["prepared", "fixtures", "missing_host_provenance", "unpaired_inputs"]).to_payload(),
            accepted_units=reported_rate(numerator=sum(r.get("accepted_units", 0) for r in selected),
                                         denominator=sum(r.get("total_units", 0) for r in selected),
                                         numerator_of=["accepted_units"], denominator_of="observed_units",
                                         excluded=["prepared", "fixtures", "missing_host_provenance", "unpaired_inputs"]).to_payload(),
            measurement_provenance=[{key: r["measurement_provenance"].get(key) for key in
                         ("source", "record_path", "record_digest", "input_path", "input_digest", "run_ref")} for r in selected],
            **{key: sum(r[key] for r in selected) if selected and all(key in r for r in selected) else None
               for key in ("model_calls", "tokens", "cost_usd", "elapsed_seconds")},
            **{key: sum(r[key] for r in selected) if selected and all(key in r for r in selected) else None for key in
               ("duplicate_dispatch", "duplicate_result", "conflicts", "resolutions", "broad_suite_executions",
                "fallbacks", "cleanup_failures", "rework_rounds")})
    return report

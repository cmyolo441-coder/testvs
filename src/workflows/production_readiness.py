from __future__ import annotations

import hashlib as hashlib
import hmac as hmac
import json as json
import math as math
import re as re
from dataclasses import dataclass as dataclass
from datetime import datetime as datetime
from typing import Any as Any
from typing import Final as Final
from typing import NoReturn as NoReturn
from typing import cast as cast

from .external_effect_receipts import (
    validate_external_effect_receipt as validate_external_effect_receipt,
)
from .production_readiness_authentication import (
    _external_authenticity_error as _external_authenticity_error,
    _external_authenticity_payload as _external_authenticity_payload,
    _trusted_context_usable as _trusted_context_usable,
    authenticate_external_readiness_evidence as authenticate_external_readiness_evidence,
)
from .production_readiness_builder import (
    build_readiness_matrix as build_readiness_matrix,
)
from .production_readiness_evidence import (
    _derive_evidence_state as _derive_evidence_state,
    _derive_verdict as _derive_verdict,
    _external_observation_ids as _external_observation_ids,
    _external_receipt_ids as _external_receipt_ids,
    _fresh as _fresh,
    _local_check_ids as _local_check_ids,
    _local_evidence_refs as _local_evidence_refs,
)
from .production_readiness_json import (
    _canonical_json_snapshot as _canonical_json_snapshot,
    _consume_json_bytes as _consume_json_bytes,
    _copy_plain_dict as _copy_plain_dict,
    _copy_plain_list as _copy_plain_list,
    _same_json_slot as _same_json_slot,
    _same_shallow_mapping as _same_shallow_mapping,
    _same_shallow_sequence as _same_shallow_sequence,
    _snapshot_json_value as _snapshot_json_value,
)
from .production_readiness_structural import (
    _evidence_errors as _evidence_errors,
    _external_evidence_errors as _external_evidence_errors,
    _external_identity_errors as _external_identity_errors,
    _future_timestamp_errors as _future_timestamp_errors,
    _matrix_identity_and_time_errors as _matrix_identity_and_time_errors,
    _observed_check_errors as _observed_check_errors,
    _timestamp as _timestamp,
)
from .production_readiness_validation import (
    _validate_readiness_snapshot as _validate_readiness_snapshot,
    parse_readiness_matrix as parse_readiness_matrix,
    validate_readiness_matrix as validate_readiness_matrix,
)
from .production_readiness_values import (
    EXTERNAL_READINESS_AUTHENTICITY_ALGORITHM as EXTERNAL_READINESS_AUTHENTICITY_ALGORITHM,
    EXTERNAL_READINESS_AUTHENTICITY_SCHEMA_VERSION as EXTERNAL_READINESS_AUTHENTICITY_SCHEMA_VERSION,
    EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION as EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION,
    OBSERVED_CHECK_RESULT_KEYS as OBSERVED_CHECK_RESULT_KEYS,
    OBSERVED_CHECK_RESULT_SCHEMA_VERSION as OBSERVED_CHECK_RESULT_SCHEMA_VERSION,
    OBSERVED_POSTCONDITION_SCHEMA_VERSION as OBSERVED_POSTCONDITION_SCHEMA_VERSION,
    READINESS_CANONICAL_JSON_MAX_BYTES as READINESS_CANONICAL_JSON_MAX_BYTES,
    READINESS_CANONICAL_JSON_MAX_DEPTH as READINESS_CANONICAL_JSON_MAX_DEPTH,
    READINESS_CANONICAL_JSON_MAX_NODES as READINESS_CANONICAL_JSON_MAX_NODES,
    READINESS_CATEGORIES as READINESS_CATEGORIES,
    READINESS_CATEGORY_POLICY as READINESS_CATEGORY_POLICY,
    READINESS_CATEGORY_POLICY_SCHEMA_VERSION as READINESS_CATEGORY_POLICY_SCHEMA_VERSION,
    READINESS_EVIDENCE_STATES as READINESS_EVIDENCE_STATES,
    READINESS_MATRIX_ROLLBACK_CONTRACT as READINESS_MATRIX_ROLLBACK_CONTRACT,
    READINESS_MATRIX_SCHEMA_VERSION as READINESS_MATRIX_SCHEMA_VERSION,
    READINESS_VERDICTS as READINESS_VERDICTS,
    ReadinessAuthenticationError as ReadinessAuthenticationError,
    ReadinessTrustContext as ReadinessTrustContext,
    ReadinessValidationResult as ReadinessValidationResult,
    ValidatedReadinessArtifact as ValidatedReadinessArtifact,
    _CANONICAL_JSON_REJECTED as _CANONICAL_JSON_REJECTED,
    _READINESS_CAPTURE_ERRORS as _READINESS_CAPTURE_ERRORS,
    _canonical_contract_ref as _canonical_contract_ref,
    _category_policy_error as _category_policy_error,
    _category_requires_external_observation as _category_requires_external_observation,
    _duplicates as _duplicates,
    _matrix_id as _matrix_id,
    _valid_ref as _valid_ref,
    readiness_row_id as readiness_row_id,
)

ValidatedReadinessArtifact.__module__ = __name__
ReadinessValidationResult.__module__ = __name__
ReadinessAuthenticationError.__module__ = __name__
ReadinessTrustContext.__module__ = __name__

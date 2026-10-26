"""Invocation-local atomic launch admission, not a scheduler or quota estimator.

Only source-qualified adapter evidence can close an owner gate. The Codex
adapter consumes shared bounded decoding and matches reviewed executable hashes
and versions to source. Unknown builds remain unsupported; there is no release floor.
Raw model/sidecar JSON never becomes a typed admission receipt.

The caller places launch() around actual Popen after waits/stagger/backoff and
rejects before retry/recovery or pool release. The lock never spans a child's
lifetime. Rejection does not prove a clean workspace or authorize replay.
"""
from __future__ import annotations

from _thread import LockType
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import re
import threading
from uuid import UUID

from .fanout_executor_sessions import SessionCapability
from .fanout_output import FanoutOutput
from .fanout_failure_diagnostics import is_string_map
from typing import Generic, Literal, Protocol, TypeVar, final


@dataclass(frozen=True, slots=True)
class AdmissionBinding:
    owner: str
    fanout_id: str
    unit_id: str
    run_ref: str
    attempt: int
    base_sha: str
    worktree: str
    invocation_id: str = ''
    attempt_id: str = ''


@dataclass(frozen=True, slots=True)
class AdmissionSupport:
    """Caller-supplied, source-qualified adapter capability; never CLI input."""
    adapter: str
    protocol: str
    scope: Literal['process_local', 'executor_local']


@dataclass(frozen=True, slots=True)
class AdmissionReceipt:
    """Trusted adapter's typed claim, still checked against observed facts."""
    adapter: str
    protocol: str
    binding: AdmissionBinding
    process_started: bool
    returncode: int | None
    schema_version: str = 'executor_admission/v1'
    source: str = 'trusted_adapter'
    reason: str = 'executor_capacity_rejected'
    implementation_started: bool = False
    definition_revision: str | None = None
    evidence_kind: str = 'supplied_adapter'
    source_revision: str | None = None


def validate_admission_receipt(
    receipt: object, *, expected: AdmissionBinding,
    support: AdmissionSupport | None, process_started: bool,
    returncode: int | None, implementation_started: bool,
) -> AdmissionReceipt | None:
    """Accept only closed typed adapter evidence with exact attempt/exit binding.

    Raw dicts/strings are deliberately unsupported, including schema-shaped
    JSON. Caller-observed implementation activity overrides the adapter claim.
    A CLI can have started/exited without implementation having been admitted;
    preserve its actual (even zero or nullable) returncode rather than inventing
    an executor error code. An unstarted process cannot have an exit code.
    """
    if not isinstance(receipt, AdmissionReceipt) or type(receipt) is not AdmissionReceipt:
        return None
    if support is None or not support.adapter or not support.protocol:
        return None
    if support.scope not in ('process_local', 'executor_local'):
        return None
    if (receipt.schema_version != 'executor_admission/v1'
            or receipt.source != 'trusted_adapter'
            or receipt.reason != 'executor_capacity_rejected'
            or receipt.adapter != support.adapter or receipt.protocol != support.protocol):
        return None
    if type(receipt.binding) is not AdmissionBinding or receipt.binding != expected:
        return None
    binding = receipt.binding
    if (type(binding.attempt) is not int or binding.attempt < 1
            or any(type(value) is not str or not value for value in (
                binding.owner, binding.fanout_id, binding.unit_id,
                binding.run_ref, binding.base_sha, binding.worktree))):
        return None
    if receipt.implementation_started is not False or implementation_started is not False:
        return None
    if (type(process_started) is not bool or type(receipt.process_started) is not bool
            or receipt.process_started is not process_started):
        return None
    if (returncode is not None and type(returncode) is not int
            or receipt.returncode is not None and type(receipt.returncode) is not int
            or receipt.returncode != returncode
            or not process_started and returncode is not None):
        return None
    return receipt


@dataclass(frozen=True, slots=True)
class StartReceipt:
    binding: AdmissionBinding
    sequence: int


@dataclass(frozen=True, slots=True)
class CapacityTrip:
    receipt: AdmissionReceipt
    sequence: int


Process = TypeVar('Process')


class LaunchCallable(Protocol):
    def __call__(self, spawn: Callable[[], Process], /) -> Process: ...


@dataclass(frozen=True, slots=True)
class LaunchResult(Generic[Process]):
    status: Literal['started', 'executor_capacity_rejected', 'not_started_capacity_blocked']
    process: Process | None = None
    start: StartReceipt | None = None
    trip: CapacityTrip | None = None


@dataclass(slots=True)
class _GateState:
    """Mutable invocation state shared by its immutable launch contexts."""
    lock: LockType = field(default_factory=threading.Lock)
    sequence: int = 0
    trips: dict[str, CapacityTrip] = field(default_factory=dict)
    sources: set[CodexAdmissionSource] = field(default_factory=set)

    def trip(self, receipt: AdmissionReceipt) -> CapacityTrip:
        # Caller holds lock; this state never escapes the module's gate/context.
        owner = receipt.binding.owner
        if owner not in self.trips:
            self.sequence += 1
            self.trips[owner] = CapacityTrip(receipt, self.sequence)
        return self.trips[owner]


@dataclass(frozen=True, slots=True)
class LaunchContext:
    _state: _GateState
    binding: AdmissionBinding
    support: AdmissionSupport | None = None

    def start(
        self, spawn: Callable[[], Process], *,
        admission_check: Callable[[], AdmissionReceipt | None] | None = None,
    ) -> LaunchResult[Process]:
        """Atomically check gate, optional trusted preflight, Popen and receipt.

        None from preflight means no rejection observed, not proven admission.
        An invalid supplied preflight raises without spawning or tripping.
        Callback exceptions propagate; failed Popen consumes no start sequence.
        Returned processes remain entirely owned by the existing runner/reaper.
        """
        with self._state.lock:
            trip = self._state.trips.get(self.binding.owner)
            if trip is not None:
                return LaunchResult('not_started_capacity_blocked', trip=trip)
            if admission_check is not None:
                observed = admission_check()
                if observed is not None:
                    receipt = validate_admission_receipt(
                        observed, expected=self.binding, support=self.support,
                        process_started=False, returncode=None, implementation_started=False,
                    )
                    if receipt is None:
                        raise ValueError('invalid_admission_receipt')
                    trip = self._state.trip(receipt)
                    return LaunchResult('executor_capacity_rejected', trip=trip)
            process = spawn()
            self._state.sequence += 1
            start = StartReceipt(self.binding, self._state.sequence)
            return LaunchResult('started', process=process, start=start)

    def launch(self, spawn: Callable[[], Process]) -> Process:
        """Runner seam: only Popen is serialized; refusal never owns a process."""
        result = self.start(spawn)
        if result.process is None:
            assert result.trip is not None
            raise CapacityBlocked(result.trip)
        return result.process

    def reject(
        self, receipt: object, *, process_started: bool,
        returncode: int | None, implementation_started: bool,
    ) -> CapacityTrip | None:
        """Validate and trip under the launch lock; retain the first trigger.

        Call as soon as complete supported rejection evidence is observed,
        before retry/recovery/postprocessing. Invalid/unsupported observations
        do not close the gate or override an earlier validated rejection.
        """
        with self._state.lock:
            validated = validate_admission_receipt(
                receipt, expected=self.binding, support=self.support,
                process_started=process_started, returncode=returncode,
                implementation_started=implementation_started,
            )
            return self._state.trip(validated) if validated is not None else None


@final
class CapacityBlocked(RuntimeError):
    def __init__(self, trip: CapacityTrip) -> None:
        super().__init__('not_started_capacity_blocked')
        self.trip = trip


@final
class OwnerLaunchGate:
    """One explicit dispatch's owner-keyed monotonic breaker, with no reset.

    One short shared lock orders starts and trips across contexts. Already
    started siblings can finish; unrelated owners can still launch. No process
    handles/output or second spawn ledger are retained here. A new explicit
    invocation constructs a new gate; earlier receipts remain immutable.
    """
    def __init__(self) -> None:
        self._state = _GateState()

    def context(
        self, binding: AdmissionBinding, *, support: AdmissionSupport | None = None,
    ) -> LaunchContext:
        return LaunchContext(self._state, binding, support)

    def rejection(self, owner: str) -> CapacityTrip | None:
        """Cheap supplementary check only; start() is the authoritative check."""
        with self._state.lock:
            return self._state.trips.get(owner)

    def observe_source(self, source: CodexAdmissionSource) -> None:
        """Record a completed launch's postflight-checked build, not a refusal."""
        with self._state.lock:
            self._state.sources.add(source)

    def recognized_sources(self) -> tuple[CodexAdmissionSource, ...]:
        with self._state.lock:
            return tuple(self._state.sources)


CODEX_ADMISSION_REVISION = 'b83105710695b70b6d96a64d1e4612bdf68d5f92'
CODEX_RELEASE_REVISION = '6b9826e3aa83b1a5947db50f4332cb9c65f1b340'
# Exact reviewed build associations, not a minimum executor version requirement.
# Values: reported version, build source commit, evidence kind. See
# docs/FANOUT-CODEX-BUILD-PROVENANCE.md for verification and platform attribution.
CODEX_ADMISSION_BUILDS: Mapping[str, tuple[str, str, Literal['fixture', 'source_verified']]] = {
    # Official rust-v0.154.0, aarch64-apple-darwin.
    '4f85982624b3898c8991cb80c0981b2aa71070e3537046c9a95950318a95afcc':
        ('0.154.0', CODEX_RELEASE_REVISION, 'source_verified'),
    # Official rust-v0.154.0, aarch64-unknown-linux-musl.
    '9b7c1c7abdc26fc3c4f47c77656a8e9121def5483dbae830ef1ee561758448a9':
        ('0.154.0', CODEX_RELEASE_REVISION, 'source_verified'),
}
CODEX_ADMISSION_ERROR = 'Error: turn/start: turn/start failed: in-process app-server request queue is full (code -32001)\n'
CODEX_ADMISSION_SUPPORT = AdmissionSupport('codex_fresh_initial_request_queue', 'codex_exec_json', 'process_local')
CAPACITY_STATUSES = frozenset({'executor_capacity_rejected', 'not_started_capacity_blocked',
                               'blocked_by_capacity_dependency'})


@dataclass(frozen=True, slots=True)
class CodexAdmissionSource:
    """Trusted caller's independently identified build, NOT a version-floor guess.

    Help/version alone cannot qualify a build. The default resolver binds the
    observed path to a shipped reviewed digest; explicit sequences are a trusted
    integration/test seam. Fixtures must identify themselves as fixtures.
    """
    resolved_path: str
    sha256: str
    version: str
    source_revision: str
    evidence_kind: Literal['fixture', 'source_verified']


def codex_admission_source(capability: SessionCapability | None,
                           sources: Sequence[CodexAdmissionSource] | None = None) -> CodexAdmissionSource | None:
    if capability is None or capability.executor != 'codex' or capability.protocol != 'codex_exec_json':
        return None
    if sources is None:
        build = CODEX_ADMISSION_BUILDS.get(capability.binary_identity.sha256)
        sources = () if build is None else (CodexAdmissionSource(
            capability.binary_identity.resolved_path, capability.binary_identity.sha256,
            build[0], build[1], build[2]),)
    for source in sources:
        if (type(source) is CodexAdmissionSource
                and source.source_revision in (CODEX_ADMISSION_REVISION, CODEX_RELEASE_REVISION)
                and source.evidence_kind in ('fixture', 'source_verified')
                and source.resolved_path == capability.binary_identity.resolved_path
                and re.fullmatch('[0-9a-f]{64}', source.sha256)
                and source.sha256 == capability.binary_identity.sha256
                and source.version == capability.version):
            return source
    return None


@final
class CodexAdmissionObserver:
    """Consume the shared decoder's events; retain no raw event or stream."""
    def __init__(self) -> None:
        self.events = 0
        self.valid = True

    def observe(self, stream: str, event: Mapping[str, object]) -> None:
        if stream != 'stdout':
            return
        self.events += 1
        reference = event.get('thread_id')
        try:
            valid_uuid = isinstance(reference, str) and str(UUID(reference)) == reference
        except ValueError:
            valid_uuid = False
        self.valid = self.valid and self.events == 1 and valid_uuid and set(event) == {'type', 'thread_id'} and event.get('type') == 'thread.started'

    def receipt(self, capture: FanoutOutput, *, binding: AdmissionBinding,
                source: CodexAdmissionSource | None, returncode: int | None,
                process_started: bool, fresh_exec: bool, artifact_observed: bool) -> AdmissionReceipt | None:
        if (source is None or binding.owner != 'codex' or not binding.invocation_id or not binding.attempt_id
                or not process_started or type(returncode) is not int or returncode <= 0
                or not fresh_exec or artifact_observed or not self.valid or self.events != 1):
            return None
        streams = capture.streams()
        # One complete stdout frame plus exact complete stderr framing. Unknown
        # startup diagnostics are deliberately unsupported, not normalized away.
        if (capture.protocol != 'codex' or streams[0]['original_lines'] != 1
                or not capture.error_window('stdout').endswith('\n')
                or any(row['original_bytes'] is None or row['truncated'] for row in streams)
                or any(row['reason'] in ('invalid_utf8', 'unsafe_control', 'sensitive_output') for row in streams)
                or streams[1]['original_bytes'] != len(CODEX_ADMISSION_ERROR)
                or capture.error_window('stderr') != CODEX_ADMISSION_ERROR
                or any(issue != 'invalid_frame' for issue in capture.issues)):
            return None
        return AdmissionReceipt(CODEX_ADMISSION_SUPPORT.adapter, CODEX_ADMISSION_SUPPORT.protocol,
                                binding, True, returncode, definition_revision=CODEX_ADMISSION_REVISION,
                                evidence_kind=source.evidence_kind, source_revision=source.source_revision)


def capacity_fields(binding: AdmissionBinding, trip: CapacityTrip, *, status: str,
                    process_started: bool) -> dict[str, object]:
    return {'schema_version': 'fanout_capacity_unit/v1', 'status': status,
            'owner': binding.owner, 'fanout_id': binding.fanout_id, 'unit_id': binding.unit_id,
            'run_ref': binding.run_ref, 'attempt_id': binding.attempt_id,
            'invocation_id': binding.invocation_id, 'process_started': process_started,
            'scope': 'process_local', 'trigger_unit': trip.receipt.binding.unit_id,
            'trigger_owner': trip.receipt.binding.owner,
            'adapter': trip.receipt.adapter, 'protocol': trip.receipt.protocol,
            'definition_revision': trip.receipt.definition_revision, 'evidence_kind': trip.receipt.evidence_kind,
            'source_revision': trip.receipt.source_revision,
            'trigger_attempt_id': trip.receipt.binding.attempt_id, 'trip_sequence': trip.sequence,
            'next_action': 'explicit_bounded_redispatch_after_capacity_change',
            'quota_observed': False}


def read_capacity_fields(record: Mapping[str, object]) -> dict[str, object]:
    """Closed optional projection; malformed/stale capacity cannot grant replay."""
    raw = record.get('capacity')
    if not is_string_map(raw):
        return {}
    value = dict[str, object](raw)
    required = {'schema_version', 'status', 'owner', 'fanout_id', 'unit_id', 'run_ref',
                'attempt_id', 'invocation_id', 'process_started', 'scope', 'trigger_unit',
                'trigger_attempt_id', 'trip_sequence', 'next_action', 'quota_observed',
                'trigger_owner', 'adapter', 'protocol', 'definition_revision', 'evidence_kind'}
    # Legacy v1 records predate the separate build-source field. Do not invent it.
    if 'source_revision' in value:
        required.add('source_revision')
    if (set(value) != required or value['schema_version'] != 'fanout_capacity_unit/v1'
            or not isinstance(value['status'], str) or value['status'] not in CAPACITY_STATUSES
            or value['scope'] != 'process_local'
            or value['quota_observed'] is not False or type(value['process_started']) is not bool
            or type(value['trip_sequence']) is not int or value['trip_sequence'] < 1
            or value['next_action'] != 'explicit_bounded_redispatch_after_capacity_change'):
        return {}
    if (value['evidence_kind'] not in ('fixture', 'source_verified', 'supplied_adapter')
            or value['definition_revision'] not in (None, CODEX_ADMISSION_REVISION)
            or value.get('source_revision') not in (None, CODEX_ADMISSION_REVISION, CODEX_RELEASE_REVISION)):
        return {}
    if ('source_revision' in value and value['adapter'] == CODEX_ADMISSION_SUPPORT.adapter
            and value['source_revision'] is None):
        return {}
    if (value.get('source_revision') is not None and (
            value['definition_revision'] != CODEX_ADMISSION_REVISION
            or value['adapter'] != CODEX_ADMISSION_SUPPORT.adapter
            or value['protocol'] != CODEX_ADMISSION_SUPPORT.protocol
            or value['evidence_kind'] not in ('fixture', 'source_verified'))):
        return {}
    for key in required - {'process_started', 'quota_observed', 'trip_sequence', 'definition_revision', 'source_revision'}:
        item = value[key]
        if not isinstance(item, str) or not 0 < len(item) <= 2048 or not item.isprintable():
            return {}
    for key, outer in (('owner', 'owner'), ('fanout_id', 'fanout_id'), ('unit_id', 'unit_id'),
                       ('run_ref', 'run_ref'), ('attempt_id', 'attempt_id'), ('invocation_id', 'invocation_id')):
        if outer in record and record[outer] != value[key]:
            return {}
    for key, outer in (('owner', 'runtime_profile'), ('unit_id', 'worker_ref'), ('run_ref', 'run_id')):
        if outer in record and record[outer] != value[key]:
            return {}
    result: dict[str, object] = {'capacity': value}
    lineage = record.get('capacity_lineage')
    if is_string_map(lineage):
        fields = dict[str, object](lineage)
        keys = {'fanout_id', 'unit_id', 'run_ref', 'owner', 'base_sha', 'worktree_path',
                'branch', 'contract_digest', 'incarnation_id'}
        if (set(fields) == keys and (fields['incarnation_id'] is None or
                isinstance(fields['incarnation_id'], str) and re.fullmatch('[0-9a-f-]{36}', fields['incarnation_id']))
                and all(isinstance(item, str) and 0 < len(item) <= 2048 and item.isprintable()
                        for key, item in fields.items() if key != 'incarnation_id')
                and all(fields[key] == value[key] for key in ('fanout_id', 'unit_id', 'run_ref', 'owner'))):
            result['capacity_lineage'] = fields
    return result


def capacity_summary(units: Sequence[Mapping[str, object]], *, requested: int | None, effective: int,
                     recognized_sources: Sequence[CodexAdmissionSource] = ()) -> dict[str, object]:
    affected = [str(row['unit_id']) for row in units if read_capacity_fields(row)]
    kinds = {source.evidence_kind for source in recognized_sources}
    native_support = ('source_qualified_build' if 'source_verified' in kinds else
                      'fixture_only' if 'fixture' in kinds else 'unsupported_without_source_qualified_build')
    return {'schema_version': 'fanout_capacity/v1', 'requested_concurrency': requested,
            'effective_concurrency': effective, 'affected_units': affected,
            'closed_owners': sorted({str(value['trigger_owner']) for row in units
                                     if (fields := read_capacity_fields(row))
                                     and is_string_map(value := fields['capacity'])}),
            'native_support': native_support, 'quota_observed': False,
            'next_action': 'explicit_bounded_redispatch_after_capacity_change' if affected else 'none'}

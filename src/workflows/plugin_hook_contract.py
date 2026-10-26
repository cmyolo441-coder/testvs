"""Static classification of the hooks a Hermes plugin manifest declares.

What this is
------------
A projection of one host contract revision, read from a manifest, for an
operator deciding whether to enable a local plugin. It answers "can a declared
hook block an action, or only watch one", and it answers it without importing,
registering, installing or running anything. Hermes remains authoritative for
loading, registration, dispatch, timeout enforcement and permissions; this is
advisory and offline.

What a declaration is and is not
--------------------------------
`provides_hooks` is author-supplied metadata. Hermes reads it into
`PluginManifest.provides_hooks` (hermes_cli/plugins_manifest.py:422) and
`hermes plugins doctor` validates it -- list, strings, known names
(hermes_cli/plugin_dev.py:300-313) -- but it does so by importing the plugin.
Registration itself happens in code, through `ctx.register_hook()`. So a
declaration never proves that a hook registered, ran, respected its budget, or
handled a failure correctly, and nothing here may be read as saying it did.

Host contract this mapping is pinned to
---------------------------------------
Hermes 0.21.1, tag `v2026.9.7` of NousResearch/hermes-agent. Every value below
is read off that revision:

- The hook vocabulary is `VALID_HOOKS` (hermes_cli/plugins.py:107-188), 37
  names. A name outside it is `unknown`; absence from the mapping is never
  treated as safe.
- `effect` refines the host's own three-value Category column in the shipped
  plugin-hook catalog (website/docs/user-guide/features/hooks.md:439-477).
  "Directive/control" splits into `policy_gate` and
  `prompt_context_contributor`, because only the first can stop an action:
  `pre_tool_call` blocks or approves a tool, `pre_verify` blocks a stop,
  `pre_gateway_dispatch` skips or rewrites a message, whereas `pre_llm_call`'s
  consumed return is joined into the user message and cannot refuse anything.
  "Transform" is `result_transformer`. "Observer" splits into
  `lifecycle_callback` and `observer` by one rule applied to the fire site: a
  hook that marks the creation, teardown, reset or terminal transition of a
  session, a subagent, a kanban task, a kanban worker, or a skill's usage state
  is a lifecycle callback; a hook reporting an operation or field change inside
  a live object's lifetime is an observer.
- `host_contract` is read off the dispatcher:
  `bounded` is `_HOOK_TIMEOUT_BOUNDED_HOOKS` plus
  `_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS` (hermes_cli/plugins_dispatch.py:41 and 48),
  the allowlist `_hook_uses_callback_timeout` consults (line 145-147).
  `queued_worker` is the four streaming observers, which never run inline on
  the token path but on a per-callback bounded queue and daemon worker
  (agent/plugin_stream_hooks.py:1-7, enqueued at agent/stream_delivery.py:38-45
  and 202). `caller_thread` is every remaining hook: unlisted hooks run
  synchronously to completion on the calling thread
  (hermes_cli/plugins_dispatch.py:24-39), and `subagent_stop` is additionally
  pinned there by a documented parent-thread contract (line 50).
- `timeout_semantics` follows from the same place. `pre_tool_call` is the only
  member of `_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS` (line 48): a timed-out or
  still-running callback is turned into a block directive
  (`_pre_tool_call_timeout_block`, applied at line 183), so it fails closed.
  Every other bounded hook is abandoned and skipped, so it fails open (line
  24-25). Nothing outside the allowlist runs under a host timeout at all, so
  `not_applicable` is the honest value there rather than a fail mode that does
  not exist.

What this pin does and does not detect
--------------------------------------
This table is a snapshot, transcribed by hand from the revision named above. It
cannot observe upstream drift, and nothing in this repository reads hermes-agent
at test time or at run time. If Hermes ships a thirty-eighth hook, this file
keeps its 37, the suite stays green, and the new hook classifies `unknown`.

`tests/test_plugin_hook_contract.py` holds a second hand copy of the same
reading. Comparing the two detects a local edit to this table that nobody
intended -- it is a change-detector on the mapping, not a drift-detector on the
host -- and two copies written from one reading agree by construction, so
neither proves the reading was right.

The range binding also does less than its presence suggests, and that belongs
here rather than in a reader's assumptions. `requires_hermes` is an OMH
convention, not a Hermes field: it is absent from Hermes'
`_KNOWN_MANIFEST_FIELDS` (plugins_manifest.py:31-38) and no code under
`hermes_cli/` or `agent/` reads it. 0 of the 105 manifests shipped at v2026.9.7
declare it -- not merely none of the ten hook-bearing ones -- so for a
third-party plugin the range is `undeclared` and this table applies by default.

The host an operator would enable the plugin on is a separate claim, and
`observe_host_version` is where it enters. It takes a version the operator
states or one local installation whose version module it reads -- never a
discovered path, and never the manifest's own `requires_hermes`, which says
what the package asks for and not what the operator runs. It answers one of
four ways: `compatible`, `incompatible`, `unreadable`, or, when no host was
named, `not_observed`. The first and last leave the classification alone; the
middle two hold it, because presenting this table as established for a host it
was never read against is what a machine-readable verdict must not do. A
version read off disk can be stale or edited, so the observation names the
method it came from and is never evidence about a running host.

What an operator does with `unknown` is therefore load-bearing: it means "not
established here", never "safe" and never "not a real hook". A hook added after
the pinned revision classifies exactly like a name the host never had. On a
host outside `supported_range`, read the host's own `VALID_HOOKS` before
concluding anything about the plugin.

Re-pinning, when Hermes moves: update `PLUGIN_HOOK_CONTRACT_VERSION`,
`PLUGIN_HOOK_CONTRACT_RANGE` and `PLUGIN_HOOK_CONTRACT_SOURCE`, re-read the four
host constructs cited above, update this table, and update the copy in the test.
`HERMES_COMPAT_MATRIX` in `src/install/plugin_compat.py` records the Hermes
version this repository has actually tested against and already names
`VALID_HOOKS` among its covered host contracts; a test asserts the two agree, so
bumping that matrix without re-reading the hook table fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias, get_args

from ..plugin_bundle.omh.host_compat import parse_range, version_satisfies
from ..system.hermes_install import installed_hermes_version
from .plugin_manifest_yaml import ManifestEntry, entry_values, read_plugin_manifest_entries


PLUGIN_HOOK_CONTRACT_HOST: Final = "hermes"
PLUGIN_HOOK_CONTRACT_RANGE: Final = ">=0.21.1,<0.22.0"
PLUGIN_HOOK_CONTRACT_VERSION: Final = "0.21.1"
PLUGIN_HOOK_CONTRACT_SOURCE: Final = "hermes-agent v2026.9.7"

MANIFEST_FILENAME: Final = "plugin.yaml"
# The field Hermes' manifest parser reads (plugins_manifest.py:422), and the
# field nine shipped plugins declare their hooks under instead. Hermes accepts
# `hooks` as a known manifest field (plugins_manifest.py:33) but never reads it,
# so a hook declared only there is an author's statement the host does not carry
# -- true of `provides_hooks` too, just one step further from the loader.
CANONICAL_HOOK_FIELD: Final = "provides_hooks"
SECONDARY_HOOK_FIELD: Final = "hooks"
HOOK_DECLARATION_FIELDS: Final = (CANONICAL_HOOK_FIELD, SECONDARY_HOOK_FIELD)
HERMES_RANGE_FIELD: Final = "requires_hermes"

MAX_DECLARED_HOOKS: Final = 64
MAX_DECLARED_RANGE_CHARS: Final = 64
# The same reason as the line above, for the other version string that reaches
# this module: an installation's `__version__` value is file content, and an
# operator's `--hermes-version` is argv. Neither is echoed once it fails to
# parse, and neither is matched at unbounded length.
MAX_HOST_VERSION_CHARS: Final = 32
_HOOK_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")

HookEffect: TypeAlias = Literal[
    "lifecycle_callback",
    "observer",
    "policy_gate",
    "prompt_context_contributor",
    "result_transformer",
    "unknown",
]
HookHostContract: TypeAlias = Literal["bounded", "caller_thread", "queued_worker", "unknown"]
HookTimeoutSemantics: TypeAlias = Literal["fail_closed", "fail_open", "not_applicable", "unknown"]
HookDeclarationField: TypeAlias = Literal["hooks", "provides_hooks"]
DeclarationStatus: TypeAlias = Literal["absent", "declared", "invalid"]
ClassificationStatus: TypeAlias = Literal["classified", "unknown"]
RangeStatus: TypeAlias = Literal["supported", "undeclared", "unparsable", "unsupported"]
# The host the operator would enable the plugin on, which is a different claim
# from `RangeStatus` above. That one answers "does the mapping cover the host
# this package asks for"; this one answers "does the mapping cover the host
# that was inspected". A package's request is not an observation of an
# environment, so the two never substitute for one another.
HostVersionStatus: TypeAlias = Literal["compatible", "incompatible", "not_observed", "unreadable"]
HostVersionMethod: TypeAlias = Literal["installation_version_module", "not_supplied", "operator_declared"]

# Derived from the types above, never restated. A second hand-written copy of a
# closed vocabulary is a copy that can drift from the one the annotations use,
# and then a test comparing the two proves only that both were edited together.
HOOK_EFFECTS: Final[tuple[str, ...]] = tuple(get_args(HookEffect))
HOOK_HOST_CONTRACTS: Final[tuple[str, ...]] = tuple(get_args(HookHostContract))
HOOK_TIMEOUT_SEMANTICS: Final[tuple[str, ...]] = tuple(get_args(HookTimeoutSemantics))


class PluginHookDeclarationError(ValueError):
    """A hook declaration is outside the bounded shape the audit reads."""


@dataclass(frozen=True, slots=True)
class HookContract:
    effect: HookEffect
    host_contract: HookHostContract
    timeout_semantics: HookTimeoutSemantics


@dataclass(frozen=True, slots=True)
class DeclaredHook:
    name: str
    field: HookDeclarationField
    contract: HookContract


@dataclass(frozen=True, slots=True)
class HostVersionObservation:
    """What was established about the host a plugin would be enabled on.

    `method` names the evidence source the operator selected, not that reading
    it succeeded: an installation that yields no version keeps
    `installation_version_module` with status `unreadable`. Folding that into
    `not_supplied` would make a failed explicit request indistinguishable from
    never having asked, and only one of those two is a pass.

    `version` is None whenever no version was established, so a value that
    failed to parse is dropped rather than echoed. An installation's
    `__version__` is file content and an operator's flag value is argv; the
    payload keeps neither once it is not a version.
    """

    version: str | None
    method: HostVersionMethod
    status: HostVersionStatus


@dataclass(frozen=True, slots=True)
class HookDeclaration:
    hooks: tuple[DeclaredHook, ...]
    declared_range: str | None
    range_status: RangeStatus


UNKNOWN_HOOK_CONTRACT: Final = HookContract("unknown", "unknown", "unknown")
# The default, and the only honest answer when the operator named no host. It
# is not a pass: it says the mapping was never bound to an environment.
NO_HOST_VERSION_OBSERVED: Final = HostVersionObservation(None, "not_supplied", "not_observed")
# The two statuses that hold the hook-semantics decision. `not_observed` is
# deliberately absent: holding it would make the default path -- every caller
# that names no host -- report an unestablished contract for a manifest the
# audit read fine, which says nothing an operator can act on.
HOST_VERSION_HOLD_STATUSES: Final[frozenset[str]] = frozenset({"incompatible", "unreadable"})

# The 37 names of VALID_HOOKS (hermes_cli/plugins.py:107-188), each classified
# by the rules in the module docstring. Grouped by effect so a reviewer reads
# the claim, not a wall of triples.
HERMES_HOOK_CONTRACTS: Final[dict[str, HookContract]] = {
    # Policy gates: the host consumes a documented return shape that can stop,
    # refuse or redirect what was about to happen.
    "pre_tool_call": HookContract("policy_gate", "bounded", "fail_closed"),
    "pre_verify": HookContract("policy_gate", "bounded", "fail_open"),
    "pre_gateway_dispatch": HookContract("policy_gate", "caller_thread", "not_applicable"),
    # Prompt/context contributor: the return is consumed, but only to add text.
    "pre_llm_call": HookContract("prompt_context_contributor", "bounded", "fail_open"),
    # Result transformers: the first valid return replaces a value the host uses.
    "transform_tool_result": HookContract("result_transformer", "bounded", "fail_open"),
    "transform_terminal_output": HookContract("result_transformer", "bounded", "fail_open"),
    "transform_llm_output": HookContract("result_transformer", "bounded", "fail_open"),
    "transform_api_error_classification": HookContract("result_transformer", "caller_thread", "not_applicable"),
    "pre_transcription": HookContract("result_transformer", "caller_thread", "not_applicable"),
    # Lifecycle callbacks: return ignored, fired at a session, subagent, kanban
    # task, kanban worker, or skill-usage transition.
    "on_session_start": HookContract("lifecycle_callback", "bounded", "fail_open"),
    "on_session_end": HookContract("lifecycle_callback", "bounded", "fail_open"),
    "on_session_finalize": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "on_session_reset": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "on_skill_lifecycle": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "subagent_start": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "subagent_stop": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "kanban_task_claimed": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "kanban_task_completed": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "kanban_task_blocked": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "on_kanban_worker_spawned": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "on_kanban_worker_exited": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    "on_kanban_worker_stale_claim": HookContract("lifecycle_callback", "caller_thread", "not_applicable"),
    # Observers: return ignored, fired to report an operation or a field change
    # inside a live object's lifetime.
    "post_tool_call": HookContract("observer", "bounded", "fail_open"),
    "post_llm_call": HookContract("observer", "bounded", "fail_open"),
    "pre_api_request": HookContract("observer", "bounded", "fail_open"),
    "post_api_request": HookContract("observer", "bounded", "fail_open"),
    "api_request_error": HookContract("observer", "bounded", "fail_open"),
    "on_stream_start": HookContract("observer", "queued_worker", "not_applicable"),
    "on_stream_delta": HookContract("observer", "queued_worker", "not_applicable"),
    "on_stream_end": HookContract("observer", "queued_worker", "not_applicable"),
    "on_interim_message": HookContract("observer", "queued_worker", "not_applicable"),
    "gateway_platform_event": HookContract("observer", "caller_thread", "not_applicable"),
    "pre_command": HookContract("observer", "caller_thread", "not_applicable"),
    "pre_approval_request": HookContract("observer", "caller_thread", "not_applicable"),
    "post_approval_response": HookContract("observer", "caller_thread", "not_applicable"),
    "on_kanban_task_updated": HookContract("observer", "caller_thread", "not_applicable"),
    "on_kanban_dispatch_tick": HookContract("observer", "caller_thread", "not_applicable"),
}


def hook_contract(name: str, *, range_status: RangeStatus = "supported") -> HookContract:
    """Classify one declared hook name against the pinned host contract.

    A manifest that asks for a host revision this mapping does not cover gets
    `unknown` for every hook. The classification is only as good as the contract
    it was read off, and saying so is the point of binding it to a range.
    """
    if range_status in ("unparsable", "unsupported"):
        return UNKNOWN_HOOK_CONTRACT
    return HERMES_HOOK_CONTRACTS.get(name, UNKNOWN_HOOK_CONTRACT)


def observe_host_version(
    *, declared_version: str | None = None, install_dir: Path | None = None
) -> HostVersionObservation:
    """Bind the pinned hook mapping to one host, from operator input only.

    Exactly one source, or none. Nothing is discovered: no installation is
    searched for, and an unsupplied host stays `not_observed` rather than
    borrowing the plugin's own `requires_hermes`, which states what the package
    asks for and never what the operator runs.

    Reading an installation is a bounded file read through the one seam in
    `system.hermes_install`. Hermes is not imported, its binary is not
    executed, no subprocess is spawned and no socket is opened, so this stays
    inside the static audit's bounds.
    """
    if declared_version is not None and install_dir is not None:
        raise ValueError("inspect a Hermes version from a declared version or an installation, not both")
    if declared_version is not None:
        return _host_version_observation(declared_version, "operator_declared")
    if install_dir is not None:
        return _host_version_observation(installed_hermes_version(install_dir), "installation_version_module")
    return NO_HOST_VERSION_OBSERVED


def _host_version_observation(version: str, method: HostVersionMethod) -> HostVersionObservation:
    if len(version) > MAX_HOST_VERSION_CHARS:
        return HostVersionObservation(None, method, "unreadable")
    try:
        satisfied = version_satisfies(version, PLUGIN_HOOK_CONTRACT_RANGE)
    except ValueError:
        return HostVersionObservation(None, method, "unreadable")
    return HostVersionObservation(version, method, "compatible" if satisfied else "incompatible")


def host_range_status(declared_range: str | None) -> RangeStatus:
    """Whether the pinned host contract covers the manifest's declared range."""
    if declared_range is None:
        return "undeclared"
    if len(declared_range) > MAX_DECLARED_RANGE_CHARS:
        return "unparsable"
    try:
        parse_range(declared_range)
    except ValueError:
        return "unparsable"
    return "supported" if version_satisfies(PLUGIN_HOOK_CONTRACT_VERSION, declared_range) else "unsupported"


def read_hook_declaration(manifest_text: str) -> HookDeclaration:
    """Read and classify the hooks a manifest declares.

    Raises `PluginManifestFormatError` when the document itself is outside the
    readable subset, and `PluginHookDeclarationError` when the document parses
    but its hook declaration is malformed. The two stay distinct because they
    are different findings about the plugin. Either diagnostic names the shape
    that failed and never echoes manifest content.
    """
    entries = read_plugin_manifest_entries(manifest_text)
    declared_range = _declared_range(entries)
    status = host_range_status(declared_range)
    names = _declared_hook_names(entries)
    hooks = tuple(
        DeclaredHook(name, field, hook_contract(name, range_status=status))
        for name, field in sorted(names.items())
    )
    return HookDeclaration(hooks, declared_range, status)


def _declared_range(entries: tuple[ManifestEntry, ...]) -> str | None:
    found = entry_values(entries, HERMES_RANGE_FIELD)
    if not found:
        return None
    if len(found) > 1:
        raise PluginHookDeclarationError(f"{HERMES_RANGE_FIELD} must be declared once")
    entry = found[0]
    if entry.kind != "scalar" or not entry.scalar:
        raise PluginHookDeclarationError(f"{HERMES_RANGE_FIELD} must be a version range string")
    return entry.scalar


def _declared_hook_names(entries: tuple[ManifestEntry, ...]) -> dict[str, HookDeclarationField]:
    """Return every declared hook name mapped to the field that declared it.

    The canonical field wins when both declare the same name, so the report says
    the strongest thing true of that hook rather than whichever field was read
    last.
    """
    names: dict[str, HookDeclarationField] = {}
    for field in reversed(HOOK_DECLARATION_FIELDS):
        for name in _field_hook_names(entries, field):
            names[name] = field
    return names


def _field_hook_names(entries: tuple[ManifestEntry, ...], field: HookDeclarationField) -> tuple[str, ...]:
    found = entry_values(entries, field)
    if not found:
        return ()
    if len(found) > 1:
        raise PluginHookDeclarationError(f"{field} must be declared once")
    entry = found[0]
    if entry.kind != "sequence":
        raise PluginHookDeclarationError(f"{field} must be a bounded list of hook names")
    if len(entry.sequence) > MAX_DECLARED_HOOKS:
        raise PluginHookDeclarationError(f"{field} declares at most {MAX_DECLARED_HOOKS} hook names")
    seen: set[str] = set()
    for name in entry.sequence:
        if _HOOK_NAME.fullmatch(name) is None:
            raise PluginHookDeclarationError(f"{field} entries must be plain hook names")
        if name in seen:
            raise PluginHookDeclarationError(f"{field} must not repeat a hook name")
        seen.add(name)
    return entry.sequence

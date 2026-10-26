from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias

from ..system.hermes_install import VERSION_MODULE_RELATIVE
from .plugin_audit_source_io import PluginAuditSource, read_static_plugin_sources, resolve_plugin_audit_root
from .plugin_hook_contract import (
    CANONICAL_HOOK_FIELD,
    HOST_VERSION_HOLD_STATUSES,
    MANIFEST_FILENAME,
    NO_HOST_VERSION_OBSERVED,
    PLUGIN_HOOK_CONTRACT_HOST,
    PLUGIN_HOOK_CONTRACT_RANGE,
    PLUGIN_HOOK_CONTRACT_SOURCE,
    SECONDARY_HOOK_FIELD,
    HookDeclaration,
    HostVersionObservation,
    PluginHookDeclarationError,
    read_hook_declaration,
)
from .plugin_manifest_yaml import PluginManifestFormatError


PLUGIN_RISK_AUDIT_SCHEMA_VERSION: Final = "plugin_risk_audit/v1"
# Deliberately still three names. A text match says a name occurs in a scanned
# file, which is a weaker thing than a declaration and cannot carry execution
# semantics -- widening it was the alternative #1536 rejected. The manifest's
# declared hooks are the detector now; this stays as the secondary signal that
# catches a plugin whose code names a hook its manifest does not.
_HOOK_CAPABILITY = re.compile(r"\b(?:on_session_end|pre_llm_call|pre_tool_call)\b")
_PROCESS_EXECUTION = re.compile(
    r"\b(?:os\.system|subprocess\.(?:call|check_call|check_output|Popen|run)|child_process\.(?:exec|execFile|fork|spawn))\s*\("
)
_DYNAMIC_EXECUTION = re.compile(r"(?<!\.)\b(?:compile|eval|exec)\s*\(|\bnew\s+Function\s*\(")
_NETWORK_REQUEST = re.compile(
    r"\b(?:axios|httpx|requests|urllib(?:\.request|3)?|https?|http)\.(?:get|post|put|request|urlopen)\s*\(|\bfetch\s*\("
)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|password|private[_-]?key|secret|token)\b\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}['\"]",
    re.IGNORECASE,
)
_DEPENDENCY_DECLARATION = re.compile(r"(?:^\s*dependencies\s*=|^\s*[A-Za-z0-9_.-]+\s*(?:[<>=!~]|$))", re.MULTILINE)
_PACKAGE_DEPENDENCY_FIELDS: Final = frozenset({"dependencies", "optionalDependencies", "peerDependencies"})

# The self-update composition's own detectors. None of them is a category on
# its own, and that is the point: a plugin that writes a file is not an updater
# and a plugin that opens a socket is not an updater. What makes a second
# software-update authority is the co-occurrence -- retrieval together with a
# write onto code the host installed, or with an archive unpacked over it, or a
# command that drives the host's own package manager. Treating every network
# call plus every file write as an updater was the alternative #1546 rejected,
# so the write leg has to name a code or manifest target, and the command leg
# has to sit in a file that also spawns a process.
_REPLACEMENT_WRITE_CALL = re.compile(
    r"\b(?:os\.(?:rename|replace)|shutil\.(?:copy|copy2|copyfile|copytree|move)"
    r"|fs\.(?:promises\.)?(?:copyFile|cp|rename|writeFile)(?:Sync)?)\s*\("
    r"|\.write_(?:bytes|text)\s*\("
    # A mode argument holding w, a or x. The mode is matched as a whole quoted
    # token so an ordinary keyword string cannot supply the letter: without the
    # closing quote right after the mode letters, `encoding="ascii"` matches.
    r"|\bopen\s*\([^\n)]{0,200},\s*['\"][rbt+]*[wax][rwaxbt+]*['\"]"
)
_REPLACEMENT_TARGET = re.compile(
    r"['\"][^'\"\n]{0,200}\.(?:cjs|cts|dll|dylib|exe|js|mjs|mts|node|py|pyc|pyz|so|ts)['\"]"
    r"|['\"](?:package\.json|plugin\.json|plugin\.ya?ml|pyproject\.toml|requirements\.txt)['\"]"
)
_ARCHIVE_EXTRACTION = re.compile(r"\bshutil\.unpack_archive\s*\(|\.extractall\s*\(")
_SELF_UPDATE_COMMAND = re.compile(
    r"\bpip[0-9.]*\b[^\n]{0,60}\binstall\b[^\n]{0,60}(?:--upgrade|--force-reinstall)\b"
    r"|\bnpm\b[^\n]{0,60}\b(?:install|update)\b[^\n]{0,60}(?:--global|-g)\b"
    r"|\bgit\b[^\n]{0,20}\bpull\b"
    r"|\bhermes\b[^\n]{0,30}\bplugins?\b[^\n]{0,30}\b(?:install|update|upgrade)\b"
)
_INTEGRITY_VERIFICATION = re.compile(
    r"\bhashlib\.(?:blake2b|sha256|sha384|sha512)\s*\("
    r"|\bhmac\.compare_digest\s*\("
    r"|\bcrypto\.create(?:Hash|Verify)\s*\("
    r"|\b(?:cosign|gpg|gpgv|minisign)\b[^\n]{0,40}--verify\b"
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
ManifestStatus: TypeAlias = Literal["invalid_json", "missing", "present"]
# `manifest_status` above is the closed vocabulary two consumers already read,
# and it answers a question about `plugin.json`. A root `plugin.yaml` -- the
# manifest Hermes itself parses -- gets its own value rather than being folded
# into a JSON-shaped word.
ManifestYamlStatus: TypeAlias = Literal["absent", "present", "unreadable"]
RiskCategory: TypeAlias = Literal[
    "declared_dependency",
    "dynamic_code_execution",
    "hermes_hook_capability",
    "network_request",
    "potential_committed_secret",
    "process_execution",
    # The compound this file did not have. It is not a seventh independent
    # detector: it is composed from the retrieval, replacement, extraction and
    # host-command signals above, and it says the package contains a plausible
    # path to replace its own installed code outside the host's pinned plugin
    # update path -- a second update authority with its own pinning, rollback,
    # consent and review guarantees.
    "self_update_or_code_replacement",
    # "I could not establish this plugin's hook contract" must not render as
    # "this plugin is fine". Without a category of its own, an unreadable
    # manifest produced a summary byte-identical to a plugin that declares
    # nothing, and a wrapper reading only the summary saw a clean plugin.
    "undetermined_hook_contract",
    # The same invariant as the line above, for the other half of the
    # self-update verdict. A package that can retrieve remote bytes and can turn
    # text into code has a code-replacement path a static read settles neither
    # way, and without a category the summary would be byte-identical to a
    # package that can do neither.
    "undetermined_self_update_path",
]
SelfUpdateSignal: TypeAlias = Literal[
    "archive_extraction",
    "code_replacement_write",
    "host_update_command",
    "integrity_verification",
    "remote_retrieval",
]
SelfUpdateClassification: TypeAlias = Literal["detected", "needs_review", "not_detected"]

# Read as reducing the risk rather than composing it. Kept out of the composing
# set so a mitigation can never serve as a leg of a composition, and reported
# under its own key so it can never render as a safe verdict.
MITIGATING_SELF_UPDATE_SIGNALS: Final[frozenset[SelfUpdateSignal]] = frozenset({"integrity_verification"})

# The closed set of compositions that classify `detected`, in the order they
# are tried, so the result can name which signals composed the verdict instead
# of only reporting that one was reached. A composition is evaluated over the
# whole scanned package rather than one file: a real updater splits retrieval
# and replacement across modules as often as it does not.
SELF_UPDATE_COMPOSITIONS: Final[tuple[tuple[str, frozenset[SelfUpdateSignal]], ...]] = (
    ("remote_retrieval_and_code_replacement", frozenset({"remote_retrieval", "code_replacement_write"})),
    ("remote_retrieval_and_archive_extraction", frozenset({"remote_retrieval", "archive_extraction"})),
    ("host_managed_update_bypass", frozenset({"host_update_command"})),
)

# One fixed sentence per classification. Each says what the verdict is and what
# it is not; the `not_detected` one exists because an empty finding is the
# reading most likely to be taken for a clean bill of health.
_SELF_UPDATE_DIAGNOSTICS: Final[dict[str, str]] = {
    "detected": (
        "the scanned text composes a plausible path to replace this plugin's own installed code, which would "
        "be a second software-update authority beside the host's pinned plugin update path; it is not evidence "
        "that the plugin retrieves, stages, or replaces anything"
    ),
    "needs_review": (
        "the scanned text can retrieve remote bytes and can turn text into code, so a code-replacement path is "
        "neither established nor ruled out by reading it"
    ),
    "not_detected": (
        "no self-update composition matched the scanned text, which is not evidence that the plugin cannot "
        "replace its own code: obfuscated, native, and dependency-supplied replacement stay outside a bounded "
        "static read"
    ),
}
_INTEGRITY_DIAGNOSTIC: Final = (
    "integrity verification appears in the scanned text and is recorded as a mitigating signal only; it does "
    "not remove the second update-authority risk and is not evidence that any checksum or signature was "
    "computed, compared, or valid"
)


@dataclass(frozen=True, slots=True)
class _Undecodable:
    """A root manifest whose bytes are not UTF-8, distinct from one that is absent."""


_UNDECODABLE_MANIFEST: Final = _Undecodable()


@dataclass(frozen=True, slots=True)
class _StaticSource:
    manifest_status: ManifestStatus | None
    byte_count: int
    risk_categories: frozenset[RiskCategory]
    self_update_signals: frozenset[SelfUpdateSignal]


def audit_plugin_risk(
    plugin_root: Path, *, host_version: HostVersionObservation = NO_HOST_VERSION_OBSERVED
) -> JsonObject:
    """Audit one local plugin directory, optionally bound to an inspected host.

    `host_version` arrives already observed rather than as a path this function
    would read, and that is deliberate: the claim boundary below says the audit
    reads one explicitly named local plugin directory, and reading a second
    directory here would quietly widen it. `observe_host_version` owns that
    read; this function only reports what it established and holds the
    hook-semantics decision when the mapping was never established for it.
    """
    root = resolve_plugin_audit_root(plugin_root)
    raw_sources = read_static_plugin_sources(root)
    sources = tuple(_audit_static_source(source) for source in raw_sources)
    manifest_yaml_status, declared_hooks = _declared_hook_report(_root_manifest_text(raw_sources), host_version)
    self_update = _self_update_report(sources)
    categories = sorted(
        {category for source in sources for category in source.risk_categories}
        | _hook_categories(declared_hooks)
        | _self_update_categories(str(self_update["classification"]))
    )
    return {
        "schema_version": PLUGIN_RISK_AUDIT_SCHEMA_VERSION,
        "source": {
            "explicit_root": True,
            "manifest_status": _manifest_status(sources),
            "manifest_yaml_status": manifest_yaml_status,
        },
        "summary": {
            "scanned_file_count": len(sources),
            "scanned_byte_count": sum(source.byte_count for source in sources),
            "risk_categories": categories,
            "risk_category_count": len(categories),
        },
        "declared_hooks": declared_hooks,
        "self_update": self_update,
        "not_observed": {
            "plugin_import": {"status": "not_observed"},
            "plugin_registration": {"status": "not_observed"},
            "plugin_execution": {"status": "not_observed"},
            "plugin_hook_registration": {"status": "not_observed"},
            "plugin_hook_execution": {"status": "not_observed"},
            "plugin_hook_timeout_enforcement": {"status": "not_observed"},
            "plugin_hook_failure_handling": {"status": "not_observed"},
            "plugin_code_replacement": {"status": "not_observed"},
            "plugin_self_update_execution": {"status": "not_observed"},
            "hermes_host_execution": {"status": "not_observed"},
            "hermes_plugin_admission": {"status": "not_observed"},
            "dependency_installation": {"status": "not_observed"},
            "network_access": {"status": "not_observed"},
            "ci_annotation_publication": {"status": "not_observed"},
        },
        "claim_boundary": (
            "The audit statically reads bounded text from one explicitly named local plugin directory and returns only "
            "aggregate risk categories and the execution semantics its manifest's declared hooks would have on the "
            "supported host contract. A declared hook is an advisory static risk contract, not evidence that the hook "
            "registered, ran, stayed inside its budget, or handled a failure correctly, and an absent declaration is "
            "not evidence that the plugin registers no hooks. It does not expose plugin source, import or execute "
            "plugin code, register a plugin, install dependencies, access a network, publish CI annotations, or prove "
            "that the plugin is safe. The self-update projection is a composition over those same static "
            "signals: it reports that the scanned text contains a plausible path to replace the plugin's own "
            "installed code outside the host-managed update path, never that the plugin retrieved, verified, "
            "staged, or replaced anything, that any checksum or signature was computed or valid, or that the "
            "host accepted a revision, and no matched composition is not evidence that the plugin cannot "
            "replace its own code. The inspected-host projection is static pre-enable evidence of one more "
            "thing only: whether the pinned hook mapping covers a Hermes version an operator stated or a "
            "local installation declares. That Hermes is never imported, executed, or asked for its version, "
            "a declared version can be stale or edited, and a compatible result is not evidence that the "
            "host would admit, load, register, or run this plugin."
        ),
    }


def scanned_text_risk_categories(name: str, text: str) -> tuple[RiskCategory, ...]:
    """The static risk categories one named text body matches, sorted.

    Public so a caller scanning bytes it is about to write -- skill promotion
    reads a reviewed draft's own instruction text this way -- reuses these
    detectors instead of growing a second copy that then drifts from them. The
    claim is the audit's own weak one: a text match says a pattern occurs in
    this text, never that anything ran.

    The self-update composition is evaluated over this one body, because that
    is all a caller passing a single text has. A package-wide scan composes the
    same signals across its scanned files instead.
    """
    categories = _risk_categories(name, text)
    composing = _self_update_signals(text, categories) - MITIGATING_SELF_UPDATE_SIGNALS
    classification = _self_update_classification(
        _matched_composition(composing), composing, "dynamic_code_execution" in categories
    )
    return tuple(sorted(categories | _self_update_categories(classification)))


def _audit_static_source(source: PluginAuditSource) -> _StaticSource:
    text = source.content.decode("utf-8", errors="replace")
    categories = _risk_categories(source.name, text)
    return _StaticSource(
        _source_manifest_status(source.name, source.is_root_file, text),
        len(source.content),
        categories,
        _self_update_signals(text, categories),
    )


def _self_update_signals(text: str, categories: frozenset[RiskCategory]) -> frozenset[SelfUpdateSignal]:
    """The self-update signals one scanned text carries.

    `remote_retrieval` is the audit's own `network_request` result rather than a
    second network detector, so the retrieval leg cannot drift from the category
    the summary already reports. The other legs are co-occurrences inside this
    one text: a write call has to name a code or manifest target, and an update
    command has to sit beside a process-execution call, because either half on
    its own is ordinary.
    """
    signals: set[SelfUpdateSignal] = set()
    if "network_request" in categories:
        signals.add("remote_retrieval")
    if _REPLACEMENT_WRITE_CALL.search(text) and _REPLACEMENT_TARGET.search(text):
        signals.add("code_replacement_write")
    if _ARCHIVE_EXTRACTION.search(text):
        signals.add("archive_extraction")
    if _SELF_UPDATE_COMMAND.search(text) and _PROCESS_EXECUTION.search(text):
        signals.add("host_update_command")
    if _INTEGRITY_VERIFICATION.search(text):
        signals.add("integrity_verification")
    return frozenset(signals)


def _self_update_report(sources: tuple[_StaticSource, ...]) -> JsonObject:
    """Compose the scanned sources' signals into one verdict and say how.

    `signal_file_counts` is how many scanned files carried each signal. It is
    the bounded stand-in for an evidence location: a file name is
    plugin-authored text, and this payload echoes none of it, so a count is the
    most an operator can be given here without the audit becoming a disclosure
    surface for the package it is auditing.
    """
    counts: dict[str, int] = {}
    for source in sources:
        for signal in source.self_update_signals:
            counts[signal] = counts.get(signal, 0) + 1
    observed: frozenset[SelfUpdateSignal] = frozenset(counts)  # type: ignore[arg-type]
    composing = observed - MITIGATING_SELF_UPDATE_SIGNALS
    mitigating = observed & MITIGATING_SELF_UPDATE_SIGNALS
    dynamic = any("dynamic_code_execution" in source.risk_categories for source in sources)
    composition = _matched_composition(composing)
    classification = _self_update_classification(composition, composing, dynamic)
    diagnostics = [_SELF_UPDATE_DIAGNOSTICS[classification]]
    if mitigating:
        diagnostics.append(_INTEGRITY_DIAGNOSTIC)
    return {
        "classification": classification,
        "composition": composition,
        "signals": sorted(composing),
        "mitigating_signals": sorted(mitigating),
        "signal_file_counts": {name: counts[name] for name in sorted(counts)},
        "diagnostics": diagnostics,
    }


def _matched_composition(composing: frozenset[SelfUpdateSignal]) -> str | None:
    for name, members in SELF_UPDATE_COMPOSITIONS:
        if members <= composing:
            return name
    return None


def _self_update_classification(
    composition: str | None, composing: frozenset[SelfUpdateSignal], dynamic_execution: bool
) -> SelfUpdateClassification:
    """Three states, and the middle one is the reason the other two are usable.

    Retrieval beside dynamic execution is the path a static read cannot follow:
    retrieved bytes can become code without any write reaching the filesystem,
    so neither `detected` nor `not_detected` would be honest about it.
    """
    if composition is not None:
        return "detected"
    if dynamic_execution and "remote_retrieval" in composing:
        return "needs_review"
    return "not_detected"


def _self_update_categories(classification: str) -> set[RiskCategory]:
    if classification == "detected":
        return {"self_update_or_code_replacement"}
    if classification == "needs_review":
        return {"undetermined_self_update_path"}
    return set()


def _hook_categories(declared_hooks: JsonObject) -> set[RiskCategory]:
    """Summary-level categories the hook report contributes.

    The second one is the invariant that keeps a failure visible in the
    aggregate. `classification_status` is `unknown` for every way the contract
    could not be established -- an unreadable manifest, a malformed declaration,
    an unsupported or unreadable host range, a hook name the contract does not
    contain -- and each of those must leave a mark on the summary, because the
    summary is what a wrapper reads. Tying the category to the status rather
    than to any one cause also covers subset gaps nobody has found yet.
    """
    categories: set[RiskCategory] = set()
    if declared_hooks["hook_count"]:
        categories.add("hermes_hook_capability")
    if declared_hooks["classification_status"] != "classified":
        categories.add("undetermined_hook_contract")
    return categories


def _root_manifest_text(sources: tuple[PluginAuditSource, ...]) -> str | None | _Undecodable:
    """Return the root `plugin.yaml` text, None when absent, or an undecodable marker.

    The bytes are decoded strictly here, unlike the scanned-text path above.
    Replacing invalid bytes there costs nothing -- a regex over mangled text
    still finds what it finds -- but doing it to a manifest would classify hooks
    out of a file Hermes' own loader cannot read at all, and present the result
    as an established contract.
    """
    for source in sources:
        if source.is_root_file and source.name == MANIFEST_FILENAME:
            try:
                return source.content.decode("utf-8")
            except UnicodeDecodeError:
                return _UNDECODABLE_MANIFEST
    return None


def _declared_hook_report(
    manifest_text: str | None | _Undecodable, host_version: HostVersionObservation
) -> tuple[ManifestYamlStatus, JsonObject]:
    """Classify the manifest's declared hooks, or say why it could not.

    A hostile or broken manifest is a finding about the plugin, not a failure of
    the audit: the reader raises, and the raise becomes a bounded diagnostic
    with no hooks classified rather than an aborted scan. The ways it can fail
    stay apart -- undecodable bytes and a document outside the readable subset
    leave the manifest `unreadable`, while a document that parses with a
    malformed hook declaration is `present` with an invalid declaration.
    """
    if manifest_text is None:
        return "absent", _hook_report("absent", None, (), host_version, manifest_present=False)
    if isinstance(manifest_text, _Undecodable):
        return "unreadable", _hook_report("invalid", None, ("plugin manifest is not valid UTF-8",), host_version)
    try:
        declaration = read_hook_declaration(manifest_text)
    except PluginManifestFormatError as exc:
        return "unreadable", _hook_report("invalid", None, (str(exc)[:200],), host_version)
    except PluginHookDeclarationError as exc:
        return "present", _hook_report("invalid", None, (str(exc)[:200],), host_version)
    return "present", _hook_report(
        "declared" if declaration.hooks else "absent", declaration, (), host_version
    )


def _hook_report(
    declaration_status: str,
    declaration: HookDeclaration | None,
    diagnostics: tuple[str, ...],
    host_version: HostVersionObservation,
    *,
    manifest_present: bool = True,
) -> JsonObject:
    hooks = declaration.hooks if declaration is not None else ()
    unknown = sum(1 for hook in hooks if hook.contract.effect == "unknown")
    range_state = declaration.range_status if declaration is not None else "undeclared"
    findings = list(diagnostics) + _hook_diagnostics(declaration, unknown)
    findings.extend(_host_version_diagnostics(host_version, len(hooks)))
    if not manifest_present:
        findings.append(
            "no root plugin.yaml was scanned, so the plugin's declared hook contract was never established"
        )
    # `manifest_present` is the same rule one level down, and the distinction it
    # draws is the whole point: a manifest that is present and declares no hooks
    # was READ and found empty, which is a result; a plugin with no manifest was
    # NEVER READ, which is a gap. Both produce an empty hook list, so a consumer
    # keying on `classification_status` alone must not see the second as a
    # positive verdict.
    # The host leg is a separate conjunct on purpose. A declaration the reader
    # parsed cleanly stays parsed -- `declaration_status`, the per-hook rows and
    # the effects still say what the mapping says -- while the overall decision
    # is held because the mapping was never established for the host that was
    # inspected. Blanking the rows instead would fold host-version evidence into
    # declaration evidence, and the two answer different questions.
    classified = (
        manifest_present
        and declaration_status != "invalid"
        and not unknown
        and range_state in ("supported", "undeclared")
        and host_version.status not in HOST_VERSION_HOLD_STATUSES
    )
    return {
        "declaration_status": declaration_status,
        "classification_status": "classified" if classified else "unknown",
        "host_contract": {
            "host": PLUGIN_HOOK_CONTRACT_HOST,
            "supported_range": PLUGIN_HOOK_CONTRACT_RANGE,
            "contract_source": PLUGIN_HOOK_CONTRACT_SOURCE,
            "declared_range": _declared_range_value(declaration),
            "declared_range_status": range_state,
            "inspected_host": _inspected_host(host_version),
        },
        "hook_count": len(hooks),
        "unknown_hook_count": unknown,
        "effects": sorted({hook.contract.effect for hook in hooks}),
        "hooks": [
            {
                "hook": hook.name,
                "declared_in": hook.field,
                "effect": hook.contract.effect,
                "host_contract": hook.contract.host_contract,
                "timeout_semantics": hook.contract.timeout_semantics,
            }
            for hook in hooks
        ],
        "diagnostics": findings,
    }


def _inspected_host(observation: HostVersionObservation) -> JsonObject:
    """The host leg of the contract block, kept beside the declaration leg.

    Three values, none of which restates another: what version was established,
    where it came from, and whether the pinned mapping covers it. The version
    uses the same sentinels as `declared_range` one field up -- `<absent>` when
    no host was named, `<invalid>` when one was and no version came out of it --
    so a reader who knows one field knows the other, and neither ever echoes a
    string that failed to parse.
    """
    if observation.version is not None:
        version = observation.version
    elif observation.status == "not_observed":
        version = "<absent>"
    else:
        version = "<invalid>"
    return {
        "version": version,
        "observation_method": observation.method,
        "compatibility": observation.status,
    }


def _host_version_diagnostics(observation: HostVersionObservation, hook_count: int) -> list[str]:
    """Name the host leg's finding, in one fixed sentence per outcome.

    `not_observed` speaks only when hooks were declared, the same condition the
    undeclared-range diagnostic one function up already uses: with no hooks
    there is no semantics decision for a host to bind, so the line would be
    noise on every audit of a plugin that declares none.
    """
    if observation.status == "incompatible":
        return [
            f"the inspected {PLUGIN_HOOK_CONTRACT_HOST} version is outside the supported host contract "
            f"{PLUGIN_HOOK_CONTRACT_RANGE}; the hook mapping was never established for it, so the declared "
            "hook semantics are held unknown"
        ]
    if observation.status == "unreadable":
        if observation.method == "operator_declared":
            return [
                f"the supplied {PLUGIN_HOOK_CONTRACT_HOST} version is not a readable version, so coverage by "
                f"{PLUGIN_HOOK_CONTRACT_RANGE} was not established and the declared hook semantics are held unknown"
            ]
        return [
            f"the named {PLUGIN_HOOK_CONTRACT_HOST} installation declares no readable version in "
            f"{VERSION_MODULE_RELATIVE.as_posix()}, so coverage by {PLUGIN_HOOK_CONTRACT_RANGE} was not "
            "established and the declared hook semantics are held unknown"
        ]
    if observation.status == "not_observed" and hook_count:
        return [
            f"no {PLUGIN_HOOK_CONTRACT_HOST} version was inspected, so these hook semantics are not bound to the "
            "host this plugin would be enabled on"
        ]
    return []


def _declared_range_value(declaration: HookDeclaration | None) -> str:
    """Echo the declared range only once it parsed; otherwise a sentinel."""
    if declaration is None or declaration.declared_range is None:
        return "<absent>"
    if declaration.range_status == "unparsable":
        return "<invalid>"
    return declaration.declared_range


def _hook_diagnostics(declaration: HookDeclaration | None, unknown: int) -> list[str]:
    if declaration is None:
        return []
    findings: list[str] = []
    if declaration.range_status == "unsupported":
        findings.append(
            f"manifest requires a {PLUGIN_HOOK_CONTRACT_HOST} version outside the supported host contract "
            f"{PLUGIN_HOOK_CONTRACT_RANGE}; declared hook semantics are unknown"
        )
    elif declaration.range_status == "unparsable":
        findings.append(
            "manifest declares an unreadable Hermes version range; declared hook semantics are unknown"
        )
    elif declaration.range_status == "undeclared" and declaration.hooks:
        findings.append(
            f"manifest declares no Hermes version range; hooks are classified against {PLUGIN_HOOK_CONTRACT_RANGE}"
        )
    if unknown:
        findings.append(
            f"{unknown} declared hook name(s) are not in the supported host contract and cannot be classified"
        )
    if any(hook.field == SECONDARY_HOOK_FIELD for hook in declaration.hooks):
        findings.append(
            f"hook(s) declared under `{SECONDARY_HOOK_FIELD}`, which the Hermes manifest parser does not read into "
            f"`{CANONICAL_HOOK_FIELD}`"
        )
    return findings


def _source_manifest_status(name: str, is_root_file: bool, text: str) -> ManifestStatus | None:
    if name != "plugin.json" or not is_root_file:
        return None
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json"
    return "present"


def _risk_categories(name: str, text: str) -> frozenset[RiskCategory]:
    categories: set[RiskCategory] = set()
    if _declares_dependencies(name, text):
        categories.add("declared_dependency")
    if _HOOK_CAPABILITY.search(text):
        categories.add("hermes_hook_capability")
    if _PROCESS_EXECUTION.search(text):
        categories.add("process_execution")
    if _DYNAMIC_EXECUTION.search(text):
        categories.add("dynamic_code_execution")
    if _NETWORK_REQUEST.search(text):
        categories.add("network_request")
    if _SECRET_ASSIGNMENT.search(text):
        categories.add("potential_committed_secret")
    return frozenset(categories)


def _declares_dependencies(name: str, text: str) -> bool:
    if name in {"pyproject.toml", "requirements.txt", "setup.cfg", "setup.py"}:
        return _DEPENDENCY_DECLARATION.search(text) is not None
    if name != "package.json":
        return False
    try:
        package = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(package, dict) and any(
        isinstance(package.get(field), dict) and package[field] for field in _PACKAGE_DEPENDENCY_FIELDS
    )


def _manifest_status(sources: tuple[_StaticSource, ...]) -> ManifestStatus:
    statuses = {source.manifest_status for source in sources if source.manifest_status is not None}
    if "invalid_json" in statuses:
        return "invalid_json"
    if "present" in statuses:
        return "present"
    return "missing"

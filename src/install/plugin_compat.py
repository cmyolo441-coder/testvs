"""Release-contract evidence inventory, not a runtime observation."""
from __future__ import annotations

from importlib import resources

from ..plugin_bundle.omh.host_compat import declared_range, parse_range, version_satisfies

HERMES_COMPAT_MATRIX = (
    {
        "version": "0.21.1",
        "verified_by": "tests.test_plugin_distribution.PluginHermesAdmissionTests",
        "host_contracts": (
            "PluginManager.discover_and_load",
            "LoadedPlugin.enabled/error/tools_registered/hooks_registered",
            "PluginContext.register_tool/register_hook/register_memory_provider/get_config",
            "VALID_HOOKS",
            "hermes_cli.__version__",
        ),
    },
)


def compat_matrix_drift() -> list[str]:
    try:
        with resources.as_file(resources.files("omh.plugin_bundle.omh").joinpath("plugin.yaml")) as manifest:
            requirement = declared_range(manifest)
        parse_range(requirement)
    except (OSError, UnicodeError, ValueError):
        return ["requires_hermes range <invalid>; no tested version coverage"]
    findings = []
    covered = False
    if not HERMES_COMPAT_MATRIX:
        findings.append(f'tested Hermes versions <empty>; declared range "{requirement}"')
    for entry in HERMES_COMPAT_MATRIX:
        version = entry["version"]
        if version_satisfies(version, requirement):
            covered = True
        else:
            findings.append(f'tested Hermes {version} outside declared range "{requirement}"')
    if not covered:
        findings.append(f'no tested Hermes version covers declared range "{requirement}"')
    return findings

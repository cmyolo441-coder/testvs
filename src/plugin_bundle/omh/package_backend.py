"""Why a native tool's `omh` package backend did or did not import here (#2166).

Hermes registers every tool and hook of a bundle whose interpreter has no
`oh-my-hermes` package -- every documented install puts the package in OMH's
own environment -- and that is a supported, metadata-only install. A tool
that needs the package must still refuse with a bounded result that says
which failure it met and which install routes are real, never raise. The
tools that import the package through `load_backend` (`omh_loop`,
`omh_decision_gate`, and `omh_probe`'s package path) get one answer per host.

Stdlib only, like every module of the bundle: the classifier must work on the
very host it describes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable, TypeVar

from . import runtime_reader as _runtime_reader

# `absent` is the supported metadata-only install, and only one failure is it:
# a ModuleNotFoundError naming an `omh` module while no top-level `omh` exists.
# Every other failure is never `absent`. With an `omh` present, a missing
# `omh.*` module or name is `incompatible` (it does not provide what this
# bundle imports); any other failure while importing is `broken`.
PACKAGE_BACKEND_AVAILABLE = "available"
PACKAGE_BACKEND_ABSENT = "absent"
PACKAGE_BACKEND_INCOMPATIBLE = "incompatible"
PACKAGE_BACKEND_BROKEN = "broken"
# What the top-level `omh` this interpreter finds is: nothing, an installed
# oh-my-hermes distribution, or some other module of that name (a namespace
# directory, a stray `omh.py`, a checkout on the path) whose version no
# distribution metadata vouches for.
FOUND_NONE = "none"
FOUND_OH_MY_HERMES = "oh-my-hermes"
FOUND_OTHER = "other"
# Where the package-dependent features, and the install routes that exist for
# each kind of Hermes environment, are written down.
PACKAGE_BACKEND_ROUTE = "docs/INSTALLATION.md#what-the-hermes-interpreter-must-be-able-to-import"

_T = TypeVar("_T")


def _found_omh(package_version: str) -> str:
    # Asked at call time, never cached: the answer is a property of the
    # interpreter now, and a module-level answer would outlive a package
    # installed, or an import finished, after this bundle loaded. A module
    # already in `sys.modules` answers first: `find_spec` raises ValueError
    # for one whose `__spec__` is None. A namespace package (a bare `omh/`
    # directory) has no file of its own, and no oh-my-hermes install is one.
    module = sys.modules.get("omh")
    if module is not None:
        namespace = getattr(module, "__file__", None) is None
    else:
        try:
            spec = importlib.util.find_spec("omh")
        except Exception:
            # This runs inside `load_backend`'s handler, so a raise here would
            # escape the tool. An import hook that raises for `omh` itself is
            # something answering for that name, and not a working
            # oh-my-hermes, so the failure reads `broken`, never `absent`.
            return FOUND_OTHER
        if spec is None:
            return FOUND_NONE
        namespace = spec.origin is None
    return FOUND_OH_MY_HERMES if package_version and not namespace else FOUND_OTHER


def _distribution_version() -> str:
    # Read from distribution metadata rather than by importing `omh.version`:
    # the classification adds no `omh.*` module to `sys.modules`. Imported
    # here: `importlib.metadata` costs about 18 ms cold, and only a refusal or
    # `omh_probe` asks, never the plugin load Hermes times.
    import importlib.metadata

    try:
        return importlib.metadata.version("oh-my-hermes")
    except importlib.metadata.PackageNotFoundError:
        return ""


def _bundle_version() -> str:
    # The `version:` of this bundle's own plugin.yaml, through the reader the
    # HUD already uses for it; "" when the manifest cannot be read.
    bundle = Path(__file__).resolve().parent
    return _runtime_reader._plugin_version(bundle, root=bundle)


def classify_backend_failure(error: BaseException) -> dict[str, Any]:
    """The `package_backend` block for a backend import that just failed."""
    failed = str(error.name or "") if isinstance(error, ImportError) else ""
    names_omh = failed == "omh" or failed.startswith("omh.")
    package_version = _distribution_version()
    found = _found_omh(package_version)
    if found == FOUND_NONE:
        absent = isinstance(error, ModuleNotFoundError) and names_omh
        status = PACKAGE_BACKEND_ABSENT if absent else PACKAGE_BACKEND_BROKEN
    elif names_omh:
        status = PACKAGE_BACKEND_INCOMPATIBLE
    else:
        status = PACKAGE_BACKEND_BROKEN
    return {
        "status": status,
        "failed_import": failed,
        "error_type": type(error).__name__,
        "found_omh": found,
        "package_version": "" if status == PACKAGE_BACKEND_ABSENT else package_version,
        "bundle_version": _bundle_version(),
        "install_route": PACKAGE_BACKEND_ROUTE,
    }


def available_backend() -> dict[str, Any]:
    """The `package_backend` block for a backend import that succeeded."""
    package_version = _distribution_version()
    return {
        "status": PACKAGE_BACKEND_AVAILABLE,
        "failed_import": "",
        "error_type": "",
        "found_omh": _found_omh(package_version),
        "package_version": package_version,
        "bundle_version": _bundle_version(),
        "install_route": PACKAGE_BACKEND_ROUTE,
    }


def load_backend(importer: Callable[[], _T]) -> tuple[_T | None, dict[str, Any] | None]:
    """Run a tool's backend import: `(what it returned, None)`, or `(None, block)`.

    Every failure is classified, never raised. A skewed install can fail at
    import with something other than ImportError -- a module-scope
    AttributeError or RuntimeError -- and the tool must still refuse, bounded,
    exactly as it does on a host with no package at all.
    """
    try:
        return importer(), None
    except Exception as error:
        return None, classify_backend_failure(error)


def package_backend_detail(backend: dict[str, Any], *, feature: str, cli_route: str = "") -> str:
    """The sentence a person reads for a `package_backend` block.

    It names only routes that exist. A Hermes environment its package manager
    prepares has none yet, so it is never told to take the package by hand.
    `cli_route` is the feature's own CLI path to the same store, when it has
    one a person may use in its place.
    """
    status = backend["status"]
    if status == PACKAGE_BACKEND_AVAILABLE:
        return f"The Python interpreter running this plugin imports the oh-my-hermes backend for {feature}."
    bundle_version = str(backend.get("bundle_version") or "")
    at_version = f" {bundle_version}" if bundle_version else ""
    failed = f" (failed import: {backend['failed_import']})" if backend["failed_import"] else ""
    foreign = backend.get("found_omh") == FOUND_OTHER
    if status == PACKAGE_BACKEND_ABSENT:
        lead = (
            "The OMH plugin is registered, but the Python interpreter running it cannot import "
            f"the oh-my-hermes package, so {feature} has no backend here. When Hermes' package "
            "manager owns that environment there is no supported way to add the package yet; "
            f"on a Hermes environment you manage yourself, install oh-my-hermes{at_version} there."
        )
    elif status == PACKAGE_BACKEND_INCOMPATIBLE and foreign:
        lead = (
            "The Python interpreter running this plugin finds an `omh` module that is not an "
            f"installed oh-my-hermes package{failed}, so {feature} does not run against it."
        )
    elif status == PACKAGE_BACKEND_INCOMPATIBLE:
        match = (
            f"This plugin is {bundle_version}: bring that package to that version"
            if bundle_version
            else "Bring that package to the version that matches this plugin"
        )
        lead = (
            f"The Python interpreter running this plugin imports oh-my-hermes "
            f"{backend['package_version']}, which does not provide what {feature} imports{failed}, "
            f"so {feature} does not run against it. {match}, the same way it was installed."
        )
    else:
        what = {
            FOUND_OH_MY_HERMES: "that package",
            FOUND_OTHER: "that `omh` module",
        }.get(backend.get("found_omh"), "what that interpreter resolves as `omh`")
        lead = (
            f"Importing the oh-my-hermes backend for {feature} on the Python interpreter running "
            f"this plugin failed with {backend['error_type']}{failed}. That is not the "
            f"missing-package case: repair {what} the same way it was installed."
        )
    route = f" {cli_route}" if cli_route else ""
    return f"{lead}{route} See {PACKAGE_BACKEND_ROUTE}."


__all__ = [
    "FOUND_NONE",
    "FOUND_OH_MY_HERMES",
    "FOUND_OTHER",
    "PACKAGE_BACKEND_ABSENT",
    "PACKAGE_BACKEND_AVAILABLE",
    "PACKAGE_BACKEND_BROKEN",
    "PACKAGE_BACKEND_INCOMPATIBLE",
    "PACKAGE_BACKEND_ROUTE",
    "available_backend",
    "classify_backend_failure",
    "load_backend",
    "package_backend_detail",
]

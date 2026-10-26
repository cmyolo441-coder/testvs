"""Backend half of the OMH Hermes Desktop plugin.

Hermes' web server (``hermes_cli/web_server_dashboard.py``,
``_mount_plugin_api_routes``) imports this file by path as the module
``hermes_dashboard_plugin_omh`` and mounts ``router`` under
``/api/plugins/omh`` -- only while ``omh`` is in the host's ``plugins.enabled``
allow-list, the registration ``omh setup`` and ``omh update`` write. That
import is standalone -- no package, no ``omh`` on the path, and the same
process as the agent -- so the bundle's reader is loaded here by path as well,
under a private package name, and everything it needs resolves through the
bundle's own relative imports.

``GET /hud?session=<stored session id>`` answers with the ``omh_hud/v1``
payload the OMH TUI's panels draw, read for the gateway process's own
Hermes home. The ``profile`` query Hermes Desktop appends when it routes a
non-primary profile through a shared backend is not read: the route answers
for the launch profile, as the host's own bundled plugin backends do. A reader
failure is answered with HTTP 200 and an error record: in the pane a thrown
error is indistinguishable from a transport failure, so the reader's failure
is returned as a record the pane can label ``reader error`` instead.

The two pure functions, ``load_reader`` and ``hud_payload``, are kept apart
from the route so OMH's own tests can call them without FastAPI, which is the
host's dependency and not OMH's.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter
    from fastapi.concurrency import run_in_threadpool
except ImportError:  # OMH's own gates import this file on a host without FastAPI
    APIRouter = None
    run_in_threadpool = None

DESKTOP_HUD_SCHEMA_VERSION = "omh_desktop_hud/v1"

# The name the bundle is loaded under here. It is neither ``omh`` (the package
# is not on this interpreter's path and must not be assumed) nor the name
# Hermes' plugin lane gives the same files, so the two loads never share a
# half-initialised module (#1623).
READER_PACKAGE = "omh_desktop_bundle"
BUNDLE_ROOT = Path(__file__).resolve().parent.parent

# Hermes serves the route from a thread pool, so two first polls -- two
# Desktop windows, or Desktop beside the browser dashboard -- can arrive
# together. The module is registered in ``sys.modules`` before it has
# executed, so without this lock the second caller takes the half-initialised
# module as its cache hit and, on its own failure, pops the package the first
# load is still executing.
_LOAD_LOCK = threading.Lock()


def _forget_reader_package() -> None:
    for name in list(sys.modules):
        if name == READER_PACKAGE or name.startswith(f"{READER_PACKAGE}."):
            sys.modules.pop(name, None)


def load_reader(bundle_root: str | Path = BUNDLE_ROOT) -> types.ModuleType:
    """The bundle's ``runtime_reader``, loaded by path under a private package.

    A bare parent module whose ``__path__`` is the bundle root is registered
    first, so ``from . import runtime_paths`` and the reader's other relative
    imports resolve to the sibling files of the same bundle. The loaded module
    is reused on later calls while it still comes from ``bundle_root``; a
    module that failed to exec is dropped rather than left half-initialised.
    One caller loads at a time; the others wait and take the finished module.
    """
    root = Path(bundle_root).resolve()
    reader_path = root / "runtime_reader.py"
    module_name = f"{READER_PACKAGE}.runtime_reader"
    with _LOAD_LOCK:
        cached = sys.modules.get(module_name)
        if cached is not None and str(getattr(cached, "__file__", "") or "") == str(reader_path):
            return cached
        _forget_reader_package()
        package = types.ModuleType(READER_PACKAGE)
        package.__path__ = [str(root)]
        sys.modules[READER_PACKAGE] = package
        spec = importlib.util.spec_from_file_location(module_name, reader_path)
        if spec is None or spec.loader is None:
            _forget_reader_package()
            raise ImportError(f"no loadable runtime_reader at {reader_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loaded = False
        try:
            spec.loader.exec_module(module)
            loaded = True
        finally:
            if not loaded:
                _forget_reader_package()
        return module


def resolve_hermes_home() -> Path:
    """The Hermes home this request reads: the host's own answer when it is
    importable, else the process ``HERMES_HOME``, else ``~/.hermes``."""
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        get_hermes_home = None
    if get_hermes_home is not None:
        return Path(get_hermes_home())
    configured = os.environ.get("HERMES_HOME", "").strip()
    return Path(configured) if configured else Path.home() / ".hermes"


def hud_payload(
    hermes_home: str | Path,
    session: str = "",
    *,
    bundle_root: str | Path = BUNDLE_ROOT,
) -> dict[str, Any]:
    """The HUD the pane renders, or the error record it shows instead.

    ``omh_home`` is left to the reader, which resolves the store from the
    gateway's own Hermes home -- the same ``get_hermes_home()`` this route
    resolves, not the ``hermes_home`` handed in -- through that home's
    ``plugins.entries.omh.settings.omh_home``, as it does for every reader
    that names no store.
    ``session`` is the focused stored session id the app names, and naming
    one is what makes the plan todo session-scoped rather than a
    most-recent-TUI guess.
    """
    session_ref = str(session or "").strip()
    try:
        reader = load_reader(bundle_root)
        payload = reader.read_omh_hud(
            None,
            hermes_home,
            graph_preference="auto",
            tui_session_ref=session_ref,
            session_scoped=True,
            tui_identity_expected=bool(session_ref),
        )
    except Exception as exc:  # the failure becomes the record the pane shows
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "schema_version": DESKTOP_HUD_SCHEMA_VERSION,
        }
    if not isinstance(payload, dict):
        return {
            "error": f"reader returned {type(payload).__name__}, not a HUD payload",
            "schema_version": DESKTOP_HUD_SCHEMA_VERSION,
        }
    return payload


if APIRouter is not None:
    router = APIRouter()

    @router.get("/hud")
    async def hud(session: str = "") -> dict[str, Any]:
        # The reader's file and sqlite reads run off the loop so a slow home
        # never stalls the stream. The home is the gateway's own; the
        # ``profile`` query is not read (see the module docstring).
        home = resolve_hermes_home()
        return await run_in_threadpool(hud_payload, home, session)

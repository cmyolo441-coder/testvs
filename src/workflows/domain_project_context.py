from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import stat
from typing import Iterator, Mapping

from ..paths import OmhPaths, find_project_root, resolve_paths
from .domain_intelligence_bound_store import (
    open_domain_directory_at,
    shared_domain_store_lock_at,
)
from .domain_intelligence_store_writer import (
    _managed_home_from_directory,
    _portable_directory_chain,
    _validate_portable_directory_chain,
    open_domain_directory,
)


_BINDING_TOKEN = object()
_HEALTH_DIRECTORIES = frozenset({"profiles", "reviews", "history"})


class HostProjectBinding:
    """An internal, descriptor-bound current-project domain store."""

    def __init__(
        self,
        project_root: Path,
        project_paths: OmhPaths,
        domain_store_fd: int | Path,
        surface: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _BINDING_TOKEN:
            raise TypeError("HostProjectBinding is minted only by trusted host surfaces")
        root_stat = (
            os.fstat(domain_store_fd)
            if isinstance(domain_store_fd, int)
            else os.stat(domain_store_fd, follow_symlinks=False)
        )
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("bound domain store is not a directory")
        self.project_root = project_root
        self.project_paths = project_paths
        self.domain_store_fd = domain_store_fd
        self.surface = surface
        self._root_identity = (root_stat.st_dev, root_stat.st_ino, root_stat.st_mode)
        self._closed = False

    def __enter__(self) -> HostProjectBinding:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self.domain_store_fd, int):
            os.close(self.domain_store_fd)

    @contextmanager
    def open_directory(self, name: str) -> Iterator[int | Path]:
        if name not in _HEALTH_DIRECTORIES:
            raise ValueError("unsafe_domain_health_directory")
        self._require_open()
        descriptor = open_domain_directory_at(self.domain_store_fd, name)
        try:
            yield descriptor
        finally:
            if isinstance(descriptor, int):
                os.close(descriptor)

    @contextmanager
    def shared_store_lock(
        self,
        *,
        timeout_seconds: float = 0.25,
        poll_interval: float = 0.01,
    ) -> Iterator[dict[str, object]]:
        self._require_open()
        with shared_domain_store_lock_at(
            self.domain_store_fd,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        ) as state:
            yield state

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("host_project_binding_closed")
        if isinstance(self.domain_store_fd, int):
            current = os.fstat(self.domain_store_fd)
        else:
            chain = _portable_directory_chain(
                self.domain_store_fd,
                home=_managed_home_from_directory(self.domain_store_fd),
                create=False,
            )
            _validate_portable_directory_chain(chain)
            current = os.stat(self.domain_store_fd, follow_symlinks=False)
        identity = (current.st_dev, current.st_ino, current.st_mode)
        if identity != self._root_identity or not stat.S_ISDIR(current.st_mode):
            raise ValueError("host_project_binding_changed")


def bind_cli_project(invocation_cwd: str | Path | None) -> HostProjectBinding | None:
    """Mint a CLI binding from the invocation cwd, resolving nested and linked paths."""
    root = find_project_root(invocation_cwd)
    if root is None:
        return None
    return _bind_root(root, surface="cli")


def bind_plugin_project(host_kwargs: Mapping[str, object]) -> HostProjectBinding | None:
    """Mint only from the plugin host's top-level ``project_root`` injection."""
    value = host_kwargs.get("project_root")
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        return None
    supplied = Path(value)
    if not supplied.is_absolute():
        return None
    try:
        canonical = supplied.resolve(strict=True)
    except OSError:
        return None
    if supplied != canonical:
        return None
    root = find_project_root(canonical)
    if root != canonical:
        return None
    return _bind_root(root, surface="plugin")


def bind_session_project(
    host_binding: HostProjectBinding | None,
) -> HostProjectBinding | None:
    """Duplicate a fresh per-turn descriptor from a trusted host binding."""
    if not isinstance(host_binding, HostProjectBinding):
        return None
    if host_binding.surface not in {"cli", "plugin"}:
        return None
    descriptor: int | Path | None = None
    binding: HostProjectBinding | None = None
    try:
        host_binding._require_open()
        descriptor = (
            os.dup(host_binding.domain_store_fd)
            if isinstance(host_binding.domain_store_fd, int)
            else host_binding.domain_store_fd
        )
        if isinstance(descriptor, int):
            os.set_inheritable(descriptor, False)
            duplicate_stat = os.fstat(descriptor)
        else:
            duplicate_stat = os.stat(descriptor, follow_symlinks=False)
        duplicate_identity = (
            duplicate_stat.st_dev,
            duplicate_stat.st_ino,
            duplicate_stat.st_mode,
        )
        if duplicate_identity != host_binding._root_identity or not stat.S_ISDIR(
            duplicate_stat.st_mode
        ):
            raise ValueError("host_project_binding_changed")
        binding = HostProjectBinding(
            host_binding.project_root,
            host_binding.project_paths,
            descriptor,
            "session",
            _token=_BINDING_TOKEN,
        )
        return binding
    except (OSError, ValueError):
        return None
    finally:
        if isinstance(descriptor, int) and binding is None:
            os.close(descriptor)


def _bind_root(root: Path, *, surface: str) -> HostProjectBinding | None:
    try:
        canonical = root.resolve(strict=True)
    except OSError:
        return None
    if canonical != root or find_project_root(canonical) != canonical:
        return None
    project_paths = _paths_for_root(canonical)
    if project_paths.omh_home_named:
        return None
    try:
        descriptor = open_domain_directory(project_paths, create=False)
    except (OSError, ValueError):
        return None
    binding: HostProjectBinding | None = None
    try:
        binding = HostProjectBinding(
            canonical,
            project_paths,
            descriptor,
            surface,
            _token=_BINDING_TOKEN,
        )
        return binding
    finally:
        if binding is None and isinstance(descriptor, int):
            os.close(descriptor)


def _paths_for_root(root: Path) -> OmhPaths:
    defaults = resolve_paths(scope="project")
    return replace(
        defaults,
        omh_home=root / ".omh",
        hermes_home=root / ".hermes",
        omh_home_named=False,
    )

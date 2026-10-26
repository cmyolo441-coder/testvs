"""Typed configuration for the explicit paired-run local adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..system.paths import OmhPaths


class PairedRunLocalRunnerError(ValueError):
    """The explicit local runner cannot safely execute the frozen plan."""


@dataclass(frozen=True, slots=True)
class PairedRunLocalRunnerConfig:
    paths: OmhPaths
    repo_root: Path
    provider: str
    hermes: str = "hermes"
    reasoning: str = "medium"
    timeout_seconds: float = 900.0

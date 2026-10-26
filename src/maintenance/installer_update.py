"""Deterministic, network-free evidence for the installer update transaction."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

try:  # Direct-source demos deliberately do not require the ``omh`` package alias.
    from ..install.self_update import run_installer_self_update
except ImportError:  # pragma: no cover - direct-source contract.
    from omh.install.self_update import run_installer_self_update


def demo_atomic_update() -> dict[str, object]:
    """Exercise post-activation rollback without pip, venv, or network access."""
    with TemporaryDirectory(prefix="omh-atomic-demo-") as temporary:
        root = Path(temporary)
        legacy = root / "venv"
        legacy_python = legacy / "bin" / "python"
        legacy_command = legacy / "bin" / "omh"
        legacy_python.parent.mkdir(parents=True)
        legacy_python.touch()
        legacy_command.touch()
        (root / "omh" / "skills").mkdir(parents=True)
        (root / "bin").mkdir()
        (root / "bin" / "omh").symlink_to(legacy_command)
        calls = 0

        def fake_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            if command[1:4] == ["-m", "venv", str(command[-1])]:
                candidate = Path(command[-1]).parent
                (candidate / "venv" / "bin").mkdir(parents=True, exist_ok=True)
                (candidate / "venv" / "bin" / "python").touch()
            if "update" in command:
                generation = Path(str(_kwargs["env"]["OMH_SELF_UPDATE_GENERATION"]))
                (generation / "skills").mkdir(parents=True, exist_ok=True)
                (generation / "skills" / "SKILL.md").write_text("candidate")
            code = 1 if calls == 6 else 0
            return subprocess.CompletedProcess(command, code, "", "post-activation demo failure" if code else "")

        with patch.dict(os.environ, {"OMH_VENV_DIR": str(legacy), "OMH_BIN_DIR": str(root / "bin"), "OMH_HOME": str(root / "omh"), "HERMES_HOME": str(root / "hermes")}, clear=False):
            result = run_installer_self_update(
                SimpleNamespace(json=True, omh_home=str(root / "omh"), hermes_home=str(root / "hermes"), recover_known_good=False),
                {"method": "installer", "release": SimpleNamespace(package_url="demo://candidate"), "python": str(legacy_python), "venv_dir": str(legacy)},
                runner=fake_runner,
            )

        return {
            "schema_version": "omh_installer_update_demo/v1",
            "post_activation_failure_rolls_back": result["rollback"]["performed"],
            "prior_remains_active": result["rollback"]["restored"] == "bootstrap-legacy",
            "candidate_inactive": not result["ok"],
            "phases": {key: {name: value for name, value in result[key].items() if name != "pointer"} for key in ("staging", "verification", "migration", "activation", "post_activation", "rollback", "recovery")},
            "cleanup": {"temporary_root_removed": True, "explicit": True},
        }

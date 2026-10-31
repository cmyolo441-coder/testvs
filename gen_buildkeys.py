"""Write fullagent/_buildkeys.py from *_API_KEY environment variables.

A release binary should work with zero configuration, but a key must never be
committed to the repository. The compromise: this script generates the
git-ignored module `fullagent/_buildkeys.py` at BUILD time from the
environment — the build-binaries workflow feeds it repository secrets
(`NVIDIA_API_KEY`, ...), and a local build can do the same:

    NVIDIA_API_KEY=nvapi-... pyinstaller --clean --noconfirm fullagent.spec

`fullagent.config` merges whatever it finds into `_EMBEDDED_KEYS` (env var >
key file > embedded), so the key baked into the artifact is the one the
binary uses until the user overrides it.

No relevant env vars => the generated file is left untouched (an existing one
from a previous build is NOT deleted: CI may have written it in an earlier
step).
"""
from __future__ import annotations

import os
from pathlib import Path

# Provider keys we know how to embed. Anything else stays out of the binary.
KNOWN_ENV = (
    "NVIDIA_API_KEY",
    "KIOS_API_KEY",
    "GWARDEN_API_KEY",
    "SAMBANOVA_API_KEY",
)

REPO_ROOT = Path(__file__).resolve().parent
TARGET = REPO_ROOT / "fullagent" / "_buildkeys.py"

_HEADER = '"""Auto-generated at build time by gen_buildkeys.py — never commit."""\n'


def collect(env=None) -> dict:
    """provider key -> secret, from `*_API_KEY` env vars (whitespace ignored)."""
    env = os.environ if env is None else env
    keys = {}
    for name in KNOWN_ENV:
        value = (env.get(name) or "").strip()
        if value:
            keys[name[: -len("_API_KEY")].lower()] = value
    return keys


def write_keys(target: Path, keys: dict) -> None:
    """Serialize `keys` into a loadable `_buildkeys.py` at `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_HEADER + f"KEYS = {dict(keys)!r}\n", encoding="utf-8")
    try:  # keys are secrets: keep the generated file private
        target.chmod(0o600)
    except OSError:
        pass


def main(target: Path | None = None, env=None) -> dict:
    """Generate the module when keys are present; return what was embedded."""
    keys = collect(env)
    if keys:
        write_keys(target or TARGET, keys)
    return keys


if __name__ == "__main__":
    embedded = main()
    if embedded:
        print("embedded keys for: " + ", ".join(sorted(embedded)))
    else:
        print("no *_API_KEY environment variables — nothing embedded")

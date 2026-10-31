"""`/key` — set a provider API key without editing files or exporting vars.

Keys never live in config.json (`/config set <api_key>` refuses them by
design: a settings file is synced, shared and diffed). This command writes
the exact `<app dir>/<provider>_api_key` file that
`config._provider_api_key()` reads — one line, mode 0600 — and then
hot-swaps the in-memory `Provider`, so the NEXT request uses the key with no
restart:

    /key                     list providers + masked key + where it came from
    /key nvidia              status for one provider
    /key nvidia nvapi-...    save the key (file) and swap it in immediately
    /key nvidia clear        delete the key file and drop the in-memory key

Precedence is unchanged: `<PROVIDER>_API_KEY` env var > key file > build-time
embedded key (see gen_buildkeys.py). A stale env var still wins after a
restart — the command says so when it detects one.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from . import config

__all__ = ["handle_apikey", "mask_key", "key_path"]

_CLEAR_WORDS = {"clear", "delete", "remove", "unset", "-"}


def mask_key(key: str) -> str:
    """Never echo a secret back in full."""
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "…" + key[-2:]
    return f"{key[:4]}…{key[-4:]}"


def key_path(provider_key: str) -> Path:
    return config.APP_DIR / f"{provider_key}_api_key"


def _source(provider_key: str) -> str:
    """Where the key in effect comes from: env, file, build, or nothing."""
    env = os.environ.get(f"{provider_key.upper()}_API_KEY", "").strip()
    if env:
        return "env"
    try:
        if key_path(provider_key).read_text(encoding="utf-8").strip():
            return "file"
    except (OSError, UnicodeDecodeError):
        pass
    if config._EMBEDDED_KEYS.get(provider_key):
        return "build"
    return ""


def _usage() -> str:
    lines = ["provider API keys (env var > key file > build):"]
    for key, provider in config.PROVIDERS.items():
        source = _source(key) or "missing"
        lines.append(f"  {key:<10} {provider.name:<16} "
                     f"{mask_key(provider.api_key):<20} [{source}]")
    lines.append("set one: /key <provider> <API_KEY>   "
                 "e.g. /key nvidia nvapi-…")
    return "\n".join(lines)



def handle_apikey(ui, arg: str) -> None:
    """Dispatch a `/key` argument; prints via ``ui``."""
    print_info = getattr(ui, "print_info", print)
    print_error = getattr(ui, "print_error", print)

    parts = (arg or "").strip().split(None, 1)
    if not parts:
        print_info(_usage())
        return

    name = parts[0].lower()
    provider = config.PROVIDERS.get(name)
    if provider is None:
        print_error(f"unknown provider: {name} — known: "
                    + ", ".join(config.PROVIDERS))
        return

    path = key_path(name)

    if len(parts) == 1:
        source = _source(name) or "missing"
        print_info(f"{provider.name}: {mask_key(provider.api_key)} "
                   f"[{source}]\n"
                   f"  set:   /key {name} <API_KEY>\n"
                   f"  clear: /key {name} clear")
        return

    secret = parts[1].strip()

    if secret.lower() in _CLEAR_WORDS:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print_error(f"could not remove {path}: {exc}")
            return
        config.PROVIDERS[name] = dataclasses.replace(provider, api_key="")
        print_info(f"✓ {provider.name} key cleared ({path.name} removed)")
        return

    if any(ch.isspace() for ch in secret):
        print_error("API key must not contain spaces — paste it as one token")
        return
    if len(secret) < 8:
        print_error("that looks too short for an API key — "
                    "paste the full key")
        return

    try:
        config.APP_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secret + "\n")
        os.chmod(path, 0o600)
    except OSError as exc:
        print_error(f"could not write {path}: {exc}")
        return

    # Hot-swap: the provider is re-read on every call (agent.provider ->
    # PROVIDERS[model.provider]), so the next request already carries it.
    config.PROVIDERS[name] = dataclasses.replace(provider, api_key=secret)

    note = ""
    env = os.environ.get(f"{name.upper()}_API_KEY", "").strip()
    if env and env != secret:
        note = (f"\n⚠ {name.upper()}_API_KEY is set in the environment and "
                "overrides this file after a restart")
    print_info(f"✓ {provider.name} key saved to {path} "
               f"({mask_key(secret)}) — the next request uses it{note}")


if __name__ == "__main__":  # pragma: no cover — module self-test
    import tempfile

    class _UI:
        def __init__(self):
            self.info, self.errors = [], []

        def print_info(self, text, *_a, **_k):
            self.info.append(text)

        def print_error(self, text, *_a, **_k):
            self.errors.append(text)

    _real_appdir = config.APP_DIR
    _real_key = config.PROVIDERS["nvidia"].api_key
    config.APP_DIR = Path(tempfile.mkdtemp(prefix="apikey_selftest_"))
    try:
        ui = _UI()
        handle_apikey(ui, "")
        assert any("provider API keys" in line for line in ui.info), ui.info

        ui = _UI()
        handle_apikey(ui, "nvidia nvapi-selftest-0123456789")
        assert not ui.errors, ui.errors
        saved = (config.APP_DIR / "nvidia_api_key")
        assert saved.read_text(encoding="utf-8").strip() == \
            "nvapi-selftest-0123456789"
        assert saved.stat().st_mode & 0o777 == 0o600, oct(saved.stat().st_mode)
        assert config.PROVIDERS["nvidia"].api_key == \
            "nvapi-selftest-0123456789"
        assert "nvapi-selftest-0123456789" not in "".join(ui.info), \
            "the secret must never be echoed back"

        ui = _UI()
        handle_apikey(ui, "nvidia")
        assert any("…6789" in line for line in ui.info), ui.info
        assert "nvapi-selftest" not in "".join(ui.info)

        ui = _UI()
        handle_apikey(ui, "nvidia clear")
        assert not saved.exists()
        assert config.PROVIDERS["nvidia"].api_key == ""

        ui = _UI()
        handle_apikey(ui, "nvidia short")
        handle_apikey(ui, "nvidia has space inside")
        assert len(ui.errors) == 2, ui.errors
        handle_apikey(ui, "nope abcdefghij")
        assert "unknown provider" in ui.errors[-1]
        print("APIKEY SELF-TEST PASS")
    finally:
        config.APP_DIR = _real_appdir
        config.PROVIDERS["nvidia"] = config.Provider(
            key="nvidia", name="NVIDIA NIM",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=_real_key, color="#76b900")

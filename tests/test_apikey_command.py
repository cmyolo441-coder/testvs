"""`/key` — provider keys set from inside the binary (no env, no config.json).

The reported incident: a freshly installed binary on the NVIDIA NIM route
died with

    no API key configured for NVIDIA NIM. Set the NVIDIA_API_KEY environment
    variable or save the key in /root/.fullagent/nvidia_api_key

Editing files by hand (or re-exporting a variable per shell) is not a
workflow, so `/key <provider> <KEY>` writes that key file (0600), hot-swaps
the in-memory Provider and lets the very next request use it.

These tests drive the REAL handler against a throwaway app dir and assert:
  A. status/set/clear/validation, and that the secret is never echoed;
  B. the key file is 0600 and `config._provider_api_key`/`_check_api_key`
     actually see it — i.e. the next request is authorised;
  C. `/key` is in the slash catalog and routed by the TUI;
  D. build-time embedding: gen_buildkeys.py writes the git-ignored
     fullagent/_buildkeys.py and config merges it (subprocess-isolated, so
     no import-reload games).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import dataclasses
import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path

from fullagent import apikeycmd, client, config
from fullagent.slashcatalog import SLASH_COMMANDS

SECRET = "nvapi-abcdef0123456789"


class _UI:
    def __init__(self):
        self.info: list[str] = []
        self.errors: list[str] = []

    def print_info(self, text, *_a, **_k):
        self.info.append(text)

    def print_error(self, text, *_a, **_k):
        self.errors.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.info + self.errors)


class KeyCommandTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._real_appdir = config.APP_DIR
        config.APP_DIR = Path(tempfile.mkdtemp(prefix="apikey_test_"))
        self._saved = dict(config.PROVIDERS)

    def tearDown(self):
        config.APP_DIR = self._real_appdir
        config.PROVIDERS.clear()
        config.PROVIDERS.update(self._saved)

    # -- A. behaviour ----------------------------------------------------
    def test_list_shows_every_provider_and_masks_secrets(self):
        ui = _UI()
        apikeycmd.handle_apikey(ui, "")
        self.assertIn("provider API keys", ui.text)
        for key in config.PROVIDERS:
            self.assertIn(key, ui.text)
        # kios ships an embedded key; it must appear masked, never raw.
        kios = config._EMBEDDED_KEYS["kios"]
        self.assertNotIn(kios, ui.text)
        self.assertIn(apikeycmd.mask_key(kios), ui.text)

    def test_set_status_and_clear(self):
        ui = _UI()
        apikeycmd.handle_apikey(ui, f"nvidia {SECRET}")
        self.assertEqual(ui.errors, [])
        self.assertNotIn(SECRET, ui.text)  # never echo the secret back
        path = config.APP_DIR / "nvidia_api_key"
        self.assertEqual(path.read_text(encoding="utf-8").strip(), SECRET)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

        ui = _UI()
        apikeycmd.handle_apikey(ui, "nvidia")
        self.assertIn("file", ui.text)
        self.assertNotIn(SECRET, ui.text)

        ui = _UI()
        apikeycmd.handle_apikey(ui, "nvidia clear")
        self.assertEqual(ui.errors, [])
        self.assertFalse(path.exists())
        self.assertEqual(config.PROVIDERS["nvidia"].api_key, "")

    def test_validation_rejects_junk_without_writing(self):
        for arg in ("nvidia short", "nvidia two tokens", "nope abcdefghij",
                    "nvidia " + "x" * 3 + " " + "y" * 3):
            ui = _UI()
            apikeycmd.handle_apikey(ui, arg)
            self.assertTrue(ui.errors, f"{arg!r} must be rejected")
        self.assertFalse((config.APP_DIR / "nvidia_api_key").exists())

    # -- B. the key is actually used -------------------------------------
    def test_saved_key_authorises_the_next_request(self):
        # start from "no key" so the 401 path is exercised even though the
        # shipped build embeds one
        config.PROVIDERS["nvidia"] = dataclasses.replace(
            config.PROVIDERS["nvidia"], api_key="")
        with self.assertRaises(client.APIError):
            client._check_api_key(config.PROVIDERS["nvidia"],
                                  config.model_by_id(
                                      "mistralai/mistral-large-4-1t-a52b"))
        ui = _UI()
        apikeycmd.handle_apikey(ui, f"nvidia {SECRET}")
        provider = config.PROVIDERS["nvidia"]
        # hot-swapped in memory AND on disk (survives a restart)
        self.assertEqual(provider.api_key, SECRET)
        self.assertEqual(config._provider_api_key("nvidia"), SECRET)
        client._check_api_key(provider,
                              config.model_by_id(
                                  "mistralai/mistral-large-4-1t-a52b"))
        headers = client._base_headers(provider, stream=True)
        self.assertEqual(headers["Authorization"], f"Bearer {SECRET}")
        self.assertEqual(headers["Accept"], "text/event-stream")

    def test_env_var_still_wins_over_the_file(self):
        ui = _UI()
        apikeycmd.handle_apikey(ui, f"nvidia {SECRET}")
        os.environ["NVIDIA_API_KEY"] = "nvapi-from-env"
        try:
            self.assertEqual(config._provider_api_key("nvidia"),
                             "nvapi-from-env")
            ui = _UI()
            apikeycmd.handle_apikey(ui, "nvidia")
            self.assertIn("env", ui.text)
        finally:
            os.environ.pop("NVIDIA_API_KEY", None)

    # -- C. wired into the UI --------------------------------------------
    def test_slash_catalog_and_tui_route_exist(self):
        names = [name for name, _ in SLASH_COMMANDS]
        self.assertIn("/key", names)
        tui_src = (Path(__file__).resolve().parents[1]
                   / "fullagent" / "tui.py").read_text(encoding="utf-8")
        self.assertIn('cmd == "/key"', tui_src)
        self.assertIn("handle_apikey", tui_src)


# -- D. build-time embedded keys -----------------------------------------
class BuildKeyEmbeddingTests(unittest.TestCase):
    """gen_buildkeys.py writes the git-ignored module config.py merges."""

    def setUp(self):
        import gen_buildkeys
        self.gen = gen_buildkeys
        self.target = gen_buildkeys.TARGET
        self._backup = (self.target.read_text(encoding="utf-8")
                        if self.target.exists() else None)

    def tearDown(self):
        if self._backup is None:
            try:
                self.target.unlink()
            except FileNotFoundError:
                pass
        else:
            self.target.write_text(self._backup, encoding="utf-8")

    def _config_view(self) -> str:
        """Read config in a fresh interpreter (no import-reload games)."""
        out = subprocess.run(
            [sys.executable, "-c",
             "from fullagent import config; "
             "print(config._EMBEDDED_KEYS.get('nvidia', ''))"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=120,
            env={k: v for k, v in os.environ.items()
                 if k != "NVIDIA_API_KEY"})
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.strip()

    def test_collect_reads_only_known_provider_vars(self):
        keys = self.gen.collect({
            "NVIDIA_API_KEY": " nvapi-xyz ",
            "KIOS_API_KEY": "",
            "UNRELATED_TOKEN": "secret",
        })
        self.assertEqual(keys, {"nvidia": "nvapi-xyz"})

    def test_generated_module_is_merged_into_embedded_keys(self):
        self.gen.write_keys(self.target, {"nvidia": "nvapi-buildtime-999"})
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o600)
        self.assertEqual(self._config_view(), "nvapi-buildtime-999")

    def test_no_generated_module_falls_back_to_the_shipped_key(self):
        """Without a build-time key module the shipped NIM key is used."""
        try:
            self.target.unlink()
        except FileNotFoundError:
            pass
        value = self._config_view()
        self.assertTrue(value.startswith("nvapi-"), value)
        self.assertNotEqual(value, "nvapi-buildtime-999")

    def test_git_ignores_the_generated_module(self):
        gitignore = (Path(__file__).resolve().parents[1]
                     / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("fullagent/_buildkeys.py", gitignore)


if __name__ == "__main__":
    import importlib.util as _ilu
    if "tests._pytest_compat" not in _sys.modules:
        _cs = _ilu.spec_from_file_location(
            "tests._pytest_compat", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
        )
        _cm = _ilu.module_from_spec(_cs)
        _sys.modules["tests._pytest_compat"] = _cm
        _cs.loader.exec_module(_cm)
    raise SystemExit(_sys.modules["tests._pytest_compat"].run_module_tests(_sys.modules[__name__]))

"""Config loading, pooling and validation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omh.config import (
    ConfigError,
    ModelProfile,
    ProviderConfig,
    default_config,
    load_config,
)
from omh.providers import build_providers
from omh.router import ModelRouter

try:  # discovery from the repo root imports this as tests.test_theme
    from .support import make_config  # noqa: F401
except ImportError:  # pragma: no cover - direct module loading
    from support import make_config  # noqa: F401


class DefaultsTest(unittest.TestCase):
    def test_default_pool_skips_mock_and_local_lanes_in_auto(self) -> None:
        config = default_config()
        names = {profile.id for profile in config.model_profiles()}
        self.assertIn("gpt-4o-mini", names)
        self.assertNotIn("mock-1", names)
        self.assertNotIn("llama3.2", names)

    def test_explicit_mock_is_routable(self) -> None:
        config = default_config()
        config.model = "mock-1"
        names = {profile.id for profile in config.model_profiles()}
        self.assertEqual(names, {"mock-1"})

    def test_explicit_provider_keeps_local_lane(self) -> None:
        config = default_config()
        config.provider = "ollama"
        names = {profile.id for profile in config.model_profiles()}
        self.assertEqual(names, {"llama3.2"})

    def test_provider_availability_and_hints(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = default_config()
            self.assertFalse(config.providers["openai"].available())
            self.assertTrue(config.providers["ollama"].available())
            hints = config.missing_credentials()
            self.assertTrue(any("OPENAI_API_KEY" in hint for hint in hints))
            self.assertFalse(any("OLLAMA" in hint for hint in hints))

    def test_router_sees_only_keyed_lanes(self) -> None:
        config = default_config()
        providers = build_providers(config)
        router = ModelRouter(config, providers)
        self.assertFalse(router.available())
        self.assertEqual(router.states, [])

    def test_model_labels_resolve(self) -> None:
        config = default_config()
        profile = config.resolve_model_name("sonnet-4")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.id, "claude-sonnet-4-20250514")
        self.assertIsNone(config.resolve_model_name("nope"))


class LoadTest(unittest.TestCase):
    def test_env_overrides_and_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {
                "OMH_HOME": tmp,
                "OMH_MODEL": "gpt-4o-mini",
                "OMH_STRATEGY": "priority",
                "OMH_THEME": "amber",
                "OMH_APPROVAL": "auto",
                "OMH_MAX_STEPS": "5",
            },
            clear=True,
        ):
            config = load_config(cwd=Path(tmp))
            self.assertEqual(config.model, "gpt-4o-mini")
            self.assertEqual(config.router.strategy, "priority")
            self.assertEqual(config.theme, "amber")
            self.assertEqual(config.approval_mode, "auto")
            self.assertEqual(config.max_steps, 5)

            offline = load_config(cwd=Path(tmp), offline=True)
            self.assertEqual(offline.model, "mock-1")
            self.assertEqual(offline.provider, "mock")

    def test_dotenv_loads_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"OMH_HOME": tmp, "KEEP": "real"}, clear=True
        ):
            Path(tmp, ".env").write_text(
                "KEEP=dotenv\nFROM_FILE=yes\nQUOTED=\"a b\"\n", encoding="utf-8"
            )
            config = load_config(cwd=Path(tmp))
            self.assertEqual(os.environ["KEEP"], "real")
            self.assertEqual(os.environ["FROM_FILE"], "yes")
            self.assertEqual(os.environ["QUOTED"], "a b")
            self.assertEqual(config.workspace, str(Path(tmp).resolve()))

    def test_custom_endpoint_registers_its_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {
                "OMH_HOME": tmp,
                "OMH_BASE_URL": "http://localhost:9999/v1",
                "OMH_API_KEY": "local",
                "OMH_MODEL": "my-custom-model",
            },
            clear=True,
        ):
            config = load_config(cwd=Path(tmp))
            profile = config.resolve_model_name("my-custom-model")
            self.assertIsNotNone(profile)
            self.assertEqual(profile.provider, "openai")
            self.assertEqual(config.providers["openai"].base_url, "http://localhost:9999/v1")
            self.assertEqual(
                [p.id for p in config.enabled_profiles()], ["my-custom-model"]
            )

    def test_custom_endpoint_without_key_is_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {
                "OMH_HOME": tmp,
                "OMH_BASE_URL": "http://localhost:8000/v1",
                "OMH_MODEL": "local-llm",
            },
            clear=True,
        ):
            config = load_config(cwd=Path(tmp))
            self.assertFalse(config.providers["openai"].requires_key)
            self.assertTrue(config.providers["openai"].available())
            self.assertEqual([p.id for p in config.enabled_profiles()], ["local-llm"])

    def test_project_config_beats_global_and_deep_merges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"OMH_HOME": tmp}, clear=True
        ):
            root = Path(tmp)
            Path(tmp, "config.json").write_text(
                json.dumps(
                    {
                        "theme": "amber",
                        "providers": {"local": {"base_url": "http://localhost:1/v1"}},
                        "models": [{"id": "m1", "provider": "local"}],
                    }
                ),
                encoding="utf-8",
            )
            project = root / "project"
            project.mkdir()
            (project / ".omh.json").write_text(
                json.dumps(
                    {"theme": "mono", "providers": {"local": {"requires_key": False}}}
                ),
                encoding="utf-8",
            )
            config = load_config(cwd=project)
            self.assertEqual(config.theme, "mono")
            local = config.providers["local"]
            self.assertEqual(local.base_url, "http://localhost:1/v1")
            self.assertFalse(local.requires_key)
            self.assertEqual([model.id for model in config.models], ["m1"])

    def test_invalid_json_reports_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"OMH_HOME": tmp}, clear=True
        ):
            Path(tmp, "config.json").write_text("{ nope", encoding="utf-8")
            with self.assertRaises(ConfigError) as caught:
                load_config(cwd=Path(tmp))
            self.assertIn("not valid JSON", caught.exception.message)

    def test_unknown_provider_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"OMH_HOME": tmp}, clear=True
        ):
            Path(tmp, "config.json").write_text(
                json.dumps({"models": [{"id": "x", "provider": "ghost"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(cwd=Path(tmp))

    def test_bad_approval_mode_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"OMH_HOME": tmp}, clear=True
        ):
            Path(tmp, "config.json").write_text(
                json.dumps({"approval_mode": "yolo"}), encoding="utf-8"
            )
            with self.assertRaises(ConfigError):
                load_config(cwd=Path(tmp))


class RoundTripTest(unittest.TestCase):
    def test_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            config.model = "model-a"
            config.router.strategy = "round_robin"
            target = config.save(Path(tmp) / "config.json")
            raw = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(raw["model"], "model-a")
            self.assertEqual(raw["router"]["strategy"], "round_robin")

    def test_available_provider_custom_key(self) -> None:
        with mock.patch.dict(os.environ, {"MY_KEY": "secret"}, clear=True):
            provider = ProviderConfig(name="x", api_key_env="MY_KEY")
            self.assertTrue(provider.available())
            self.assertEqual(provider.resolved_key(), "secret")

    def test_model_profile_requires_id(self) -> None:
        with self.assertRaises(ConfigError):
            ModelProfile.from_dict({"provider": "a"})

    def test_config_to_dict_roundtrip(self) -> None:
        config = make_config("/tmp")
        raw = config.to_dict()
        self.assertEqual(len(raw["models"]), 2)
        self.assertEqual(raw["providers"]["a"]["api_key_env"], "")


if __name__ == "__main__":
    unittest.main()


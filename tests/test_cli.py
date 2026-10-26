"""CLI smoke tests: subprocess entry point and in-process slash commands."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from omh.cli import App
from omh.config import load_config

ROOT = Path(__file__).resolve().parent.parent
PROVIDER_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "OMH_BASE_URL",
    "OMH_API_KEY",
    "OMH_MODEL",
    "OMH_PROVIDER",
    "OMH_STRATEGY",
    "OMH_THEME",
    "OMH_APPROVAL",
    "OMH_MAX_STEPS",
    "OMH_SESSION_DIR",
}


def clean_env(home: str | Path) -> dict:
    env = {key: value for key, value in os.environ.items() if key not in PROVIDER_KEYS}
    env["OMH_HOME"] = str(home)
    env["OMH_COLOR"] = "0"
    env["PYTHONPATH"] = str(ROOT)
    return env


def run_cli(*args: str, home: str | Path, stdin: str | None = None):
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=str(ROOT),
        env=clean_env(home),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


class SubprocessTest(unittest.TestCase):
    def test_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--version", home=tmp)
        self.assertEqual(result.returncode, 0)
        self.assertIn("omh 1.0.0", result.stdout)

    def test_list_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--list-models", home=tmp)
        self.assertEqual(result.returncode, 0)
        self.assertIn("gpt-4o-mini", result.stdout)
        self.assertIn("no key", result.stdout)
        self.assertIn("load balanced", result.stdout)

    def test_doctor_without_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--doctor", home=tmp)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing key", result.stdout)
        self.assertIn("routable", result.stdout)

    def test_offline_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--offline", "-p", "hello there", "--quiet", home=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("OMH is running offline", result.stdout)
            self.assertIn("You said: hello there", result.stdout)
            sessions = list(Path(tmp, "sessions").glob("*.json"))
            self.assertEqual(len(sessions), 1)

    def test_offline_tool_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--offline", "-p", "list the files", "--quiet", home=tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("list_dir", result.stdout)
        self.assertIn("Tool result", result.stdout)

    def test_stdin_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--offline", "--quiet", home=tmp, stdin="piped prompt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("piped prompt", result.stdout)

    def test_unknown_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("--model", "nope", home=tmp)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown model", result.stderr)

    def test_init_writes_config_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = run_cli("--init", home=tmp)
            self.assertEqual(first.returncode, 0, first.stderr)
            config_path = Path(tmp) / "config.toml"
            self.assertTrue(config_path.exists())
            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
            self.assertIn("router", raw)
            second = run_cli("--init", home=tmp)
            self.assertEqual(second.returncode, 1)


def offline_app(tmp: str, *, resume: bool = False) -> App:
    with mock.patch.dict(os.environ, {"OMH_HOME": tmp}, clear=True):
        config = load_config(cwd=Path(tmp), offline=True)
    app = App(config, interactive=False, resume=resume)
    app.console.stream = io.StringIO()
    app.console.error_stream = io.StringIO()
    return app


class SlashCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        skills = Path(self.tmp) / "skills"
        skills.mkdir()
        (skills / "review.md").write_text(
            "---\nname: review\ndescription: review code\n---\nBe careful.\n",
            encoding="utf-8",
        )
        self.app = offline_app(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def output(self) -> str:
        return self.app.console.stream.getvalue()

    def test_help_lists_commands(self) -> None:
        self.assertTrue(self.app._slash("/help"))
        self.assertIn("/balance", self.output())
        self.assertIn("ctrl-c", self.output())

    def test_models_tools_and_balance(self) -> None:
        self.app._slash("/models")
        self.assertIn("mock", self.output())
        self.app.console.stream = io.StringIO()
        self.app._slash("/tools")
        self.assertIn("read_file", self.output())
        self.app.console.stream = io.StringIO()
        self.app._slash("/balance")
        self.assertIn("strategy", self.output())

    def test_model_and_provider_switching(self) -> None:
        self.app._slash("/model auto")
        self.assertIn("model → auto", self.output())
        self.app.console.stream = io.StringIO()
        self.app._slash("/provider ollama")
        self.assertEqual(self.app.config.provider, "ollama")
        self.app.console.stream = io.StringIO()
        self.app._slash("/strategy round_robin")
        self.assertEqual(self.app.config.router.strategy, "round_robin")
        self.app.console.stream = io.StringIO()
        self.app._slash("/model sonnet-4")
        self.assertEqual(self.app.config.model, "claude-sonnet-4-20250514")

    def test_skill_loading(self) -> None:
        self.app._slash("/skill review")
        self.assertIsNotNone(self.app.skill)
        self.assertEqual(self.app.session.skill, "review")
        self.assertIn("Be careful.", self.app._make_agent().system_prompt())
        self.app.console.stream = io.StringIO()
        self.app._slash("/skill off")
        self.assertIsNone(self.app.skill)

    def test_theme_and_approval(self) -> None:
        self.app._slash("/theme amber")
        self.assertEqual(self.app.config.theme, "amber")
        self.assertEqual(self.app.console.theme.palette.name, "amber")
        self.app.console.stream = io.StringIO()
        self.app._slash("/approval auto")
        self.assertEqual(self.app.config.approval_mode, "auto")
        self.assertEqual(self.app.registry.context.approval_mode, "auto")

    def test_session_commands(self) -> None:
        from omh.providers.base import ChatMessage

        self.app.session.add(ChatMessage.user("first question"))
        self.app._save()
        self.app._slash("/new")
        self.assertNotEqual(self.app.session.title, "first question")
        self.app.console.stream = io.StringIO()
        self.app._slash("/resume last")
        self.assertEqual(self.app.session.title, "first question")
        self.app.console.stream = io.StringIO()
        self.app._slash("/cost")
        self.assertIn("tokens", self.output())
        self.app.console.stream = io.StringIO()
        self.app._slash("/clear")
        self.assertEqual(self.app.session.messages, [])

    def test_unknown_command_and_exit(self) -> None:
        self.assertTrue(self.app._slash("/bogus"))
        self.assertIn("unknown command", self.output())
        self.assertFalse(self.app._slash("/exit"))

    def test_doctor_offline(self) -> None:
        self.assertEqual(self.app.doctor(), 0)
        self.assertIn("routable", self.output())

    def test_init_from_repl(self) -> None:
        with mock.patch.dict(os.environ, {"OMH_HOME": self.tmp}, clear=True):
            self.app._slash("/init")
        self.assertTrue((Path(self.tmp) / "config.toml").exists())


class RunOnceTest(unittest.TestCase):
    def test_offline_turn_streams_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"OMH_HOME": tmp}, clear=True):
                config = load_config(cwd=Path(tmp), offline=True)
                with contextlib.redirect_stdout(buffer):
                    app = App(config, interactive=False)
                    code = app.run_once("list the files")
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            self.assertIn("list_dir", text)
            self.assertIn("ol", text)  # rendered output includes the list
            self.assertEqual(len(app.store.summaries()), 1)
            self.assertGreater(len(app.session.messages), 2)


if __name__ == "__main__":
    unittest.main()


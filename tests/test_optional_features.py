
import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import ast
from pathlib import Path

from fullagent import browserauto, continueresume
from fullagent.help import _complete_command_groups, format_help
from fullagent.slashcatalog import SLASH_COMMANDS


def test_help_and_completion_share_every_command():
    completion_names = [name for name, _ in SLASH_COMMANDS]
    assert len(completion_names) == len(set(completion_names))
    help_names = {
        name for commands in _complete_command_groups().values()
        for name, _ in commands
    }
    help_text = format_help()
    assert set(completion_names) <= help_names
    assert all(name in help_text for name in completion_names)
    assert "/mcp" in help_text and "/doctor" in help_text
    assert "/terminal-setup" in help_text and "/voice" in help_text


def test_every_completion_command_has_a_slash_dispatch_branch():
    tui_path = Path(__file__).resolve().parents[1] / "fullagent" / "tui.py"
    tree = ast.parse(tui_path.read_text(encoding="utf-8"))
    route = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_route_slash"
    )
    dispatched = set()
    for node in ast.walk(route):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "cmd":
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and child.value.startswith("/")):
                dispatched.add(child.value)
    assert {name for name, _ in SLASH_COMMANDS} <= dispatched


def test_help_supports_single_command_lookup_and_aliases():
    assert "/model" in format_help("model")
    assert "/help" in format_help("/h")
    assert "Unknown slash command" in format_help("not-a-command")


def test_system_chromium_is_used_when_playwright_browser_is_missing(
        monkeypatch, tmp_path: Path):
    system_chromium = tmp_path / "chromium"
    system_chromium.write_text("binary", encoding="utf-8")
    system_chromium.chmod(0o755)
    monkeypatch.delenv("FULLAGENT_CHROMIUM_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        browserauto.shutil, "which",
        lambda name: str(system_chromium) if name == "chromium" else None)

    class MissingPlaywrightBrowser:
        executable_path = str(tmp_path / "missing-managed-browser")

    assert browserauto._resolve_chromium_executable(
        MissingPlaywrightBrowser()) == str(system_chromium)


def test_invalid_explicit_chromium_path_is_actionable(monkeypatch, tmp_path):
    fallback = tmp_path / "chromium"
    fallback.write_text("binary", encoding="utf-8")
    fallback.chmod(0o755)
    monkeypatch.setenv("FULLAGENT_CHROMIUM_EXECUTABLE",
                       str(tmp_path / "missing"))
    monkeypatch.setattr(browserauto.shutil, "which",
                        lambda _name: str(fallback))

    assert browserauto._resolve_chromium_executable() is None


def test_continue_reseed_stays_under_ten_percent():
    agent, _events = continueresume._make_fake_agent(25)
    before = continueresume._estimate(agent.messages)

    assert continueresume.seed_continue_history(agent, "continue")

    after = continueresume._estimate(agent.messages)
    assert after / before < 0.10
    summary = agent.messages[-1]["content"]
    assert "Files touched:" in summary
    assert "PENDING:" in summary


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

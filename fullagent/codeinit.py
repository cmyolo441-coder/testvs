# ---------------------------------------------------------------------------
# fullagent.codeinit — Claude Code-style `/init` command.
#
# Analyzes a codebase and generates an AGENTS.md (or CLAUDE.md) file that
# describes the project so an AI agent can work in it productively.
#
# Public API:
#   analyze_project(root=".") -> dict   — static project analysis
#   generate_agents_md(analysis) -> str  — markdown report from analysis
#   register(agent)                      — adds the `InitProject` tool
#   run_init_command(ui, arg)            — handler for the `/init` TUI cmd
#
# Stdlib only. Does not import .agent or .tui (agent is passed in).
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Language / file tables
# ---------------------------------------------------------------------------

# extension -> language name
_LANG_BY_EXT: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JavaScript (React)", ".ts": "TypeScript", ".tsx": "TypeScript (React)",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin",
    ".c": "C", ".h": "C/C++ header", ".cpp": "C++", ".hpp": "C++", ".cc": "C++",
    ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".lua": "Lua", ".pl": "Perl", ".r": "R",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".vue": "Vue", ".svelte": "Svelte",
    ".toml": "TOML config", ".yaml": "YAML config", ".yml": "YAML config",
    ".json": "JSON config", ".xml": "XML",
    ".md": "Markdown docs", ".rst": "reStructuredText docs",
    ".dockerfile": "Dockerfile", ".tf": "Terraform",
}

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".venv", "venv", ".env", "node_modules", "dist", "build",
    "target", ".idea", ".vscode", ".DS_Store", "coverage", ".next",
}

# filename (lowercased) -> display label for key files
_KEY_FILES: dict[str, str] = {
    "readme.md": "README", "readme.rst": "README", "readme.txt": "README",
    "pyproject.toml": "Python build config", "setup.py": "legacy setup script",
    "setup.cfg": "legacy setup config", "requirements.txt": "pip dependencies",
    "poetry.lock": "poetry lockfile", "pipfile": "pipenv manifest",
    "package.json": "Node manifest", "package-lock.json": "npm lockfile",
    "yarn.lock": "yarn lockfile", "pnpm-lock.yaml": "pnpm lockfile",
    "tsconfig.json": "TypeScript config", "vite.config.js": "Vite config",
    "vite.config.ts": "Vite config", "webpack.config.js": "Webpack config",
    "go.mod": "Go module manifest", "go.sum": "Go checksums",
    "cargo.toml": "Rust manifest", "cargo.lock": "Rust lockfile",
    "pom.xml": "Maven manifest", "build.gradle": "Gradle manifest",
    "makefile": "Make build file", "cmakelists.txt": "CMake manifest",
    "dockerfile": "container image", "docker-compose.yml": "compose services",
    "docker-compose.yaml": "compose services", ".gitignore": "git ignore rules",
    ".github": "CI workflows (dir)", "license": "license", "license.md": "license",
    "changelog.md": "changelog", "agents.md": "existing agent guide",
    "claude.md": "existing agent guide",
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _iter_source_files(root: Path):
    """Yield relative paths of files under root, skipping junk dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            yield Path(dirpath, fn).relative_to(root)


def analyze_project(root: str | Path = ".") -> dict:
    """Statically analyze the project at *root*.

    Returns a dict with: root, name, languages {lang: count}, files_count,
    key_files [relative paths], entry_points [relative paths], dirs (top 2
    levels), build_cmds, test_cmds, lines_total.
    """
    root = Path(root).resolve()

    ext_counts: dict[str, int] = {}
    key_files: list[str] = []
    entry_points: list[str] = []
    dirs: set[str] = set()
    lines_total = 0
    files_count = 0
    py_lines: dict[str, int] = {}

    for rel in _iter_source_files(root):
        files_count += 1
        if len(rel.parts) > 1:
            dirs.add(rel.parts[0])
            if len(rel.parts) > 2:
                dirs.add(str(Path(*rel.parts[:2])))

        low = rel.name.lower()
        label = _KEY_FILES.get(low)
        if label:
            key_files.append(rel.as_posix())
        name_stem = rel.stem.lower()
        if (low in ("main.py", "main.go", "main.rs", "index.js", "index.ts",
                    "app.py", "cli.py", "__main__.py", "server.py")
                or name_stem in ("main", "app", "cli", "index", "server")):
            entry_points.append(rel.as_posix())

        ext = rel.suffix.lower()
        lang = _LANG_BY_EXT.get(ext)
        if lang:
            ext_counts[lang] = ext_counts.get(lang, 0) + 1

        # Cheap per-language size signal (text files only, cap reads).
        if ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".c",
                   ".cpp", ".sh"):
            try:
                with open(root / rel, "r", encoding="utf-8",
                          errors="ignore") as fh:
                    n = sum(1 for _ in fh)
                lines_total += n
                if ext == ".py":
                    py_lines[rel.as_posix()] = n
            except OSError:
                pass

    languages = dict(sorted(ext_counts.items(), key=lambda kv: -kv[1]))

    name = _detect_name(root, key_files)
    build_cmds, test_cmds = _detect_commands(root, key_files)
    conventions = _infer_conventions(root, key_files, py_lines, languages)

    return {
        "root": str(root),
        "name": name,
        "languages": languages,
        "files_count": files_count,
        "lines_total": lines_total,
        "key_files": sorted(key_files),
        "entry_points": sorted(entry_points),
        "dirs": sorted(dirs),
        "build_cmds": build_cmds,
        "test_cmds": test_cmds,
        "conventions": conventions,
    }


def _detect_name(root: Path, key_files: list[str]) -> str:
    """Best-effort project name: pyproject/package.json/go.mod, else dir."""
    for kf in key_files:
        if kf.endswith("pyproject.toml"):
            try:
                txt = (root / kf).read_text(encoding="utf-8", errors="ignore")
                m = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', txt,
                              re.MULTILINE)
                if m:
                    return m.group(1)
            except OSError:
                pass
        elif kf.endswith("package.json"):
            try:
                data = json.loads((root / kf).read_text(encoding="utf-8",
                                                        errors="ignore"))
                # valid JSON need not be a package object ("[]", "null"):
                # data.get() used to crash project analysis on such repos
                if isinstance(data, dict) and data.get("name"):
                    return str(data["name"])
            except (OSError, ValueError):
                pass
        elif kf.endswith("go.mod"):
            try:
                first = (root / kf).read_text(
                    encoding="utf-8", errors="ignore").splitlines()[0]
                m = re.match(r"module\s+(\S+)", first)
                if m:
                    return m.group(1)
            except (OSError, IndexError):
                pass
    return root.name


def _detect_commands(root: Path, key_files: list[str]) -> tuple[list[str], list[str]]:
    """Extract likely build/test commands from manifests and Makefile."""
    build: list[str] = []
    test: list[str] = []

    def _add(cmds: list[str], cmd: str) -> None:
        if cmd and cmd not in cmds:
            cmds.append(cmd)

    for kf in key_files:
        p = root / kf
        if kf.endswith("pyproject.toml"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if "[tool.pytest" in txt or "pytest" in txt:
                    _add(test, "pytest")
                if "[tool.hatch" in txt:
                    _add(build, "hatch build")
                if "[tool.poetry" in txt:
                    _add(build, "poetry build")
            except OSError:
                pass
            _add(build, "python -m build")
            _add(test, "pytest")
        elif kf == "requirements.txt":
            _add(build, "pip install -r requirements.txt")
        elif kf.endswith("package.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8",
                                              errors="ignore"))
                scripts = data.get("scripts", {}) if isinstance(
                    data, dict) else {}
                if not isinstance(scripts, dict):
                    scripts = {}
                if scripts.get("build"):
                    _add(build, "npm run build")
                if scripts.get("test"):
                    _add(test, "npm test")
            except (OSError, ValueError):
                pass
            _add(build, "npm run build")
            _add(test, "npm test")
        elif kf.endswith("Makefile") or kf.endswith("makefile"):
            try:
                targets = re.findall(r"^([a-zA-Z][\w-]*)\s*:", p.read_text(
                    encoding="utf-8", errors="ignore"), re.MULTILINE)
                for t in targets:
                    if t in ("test", "tests", "check"):
                        _add(test, f"make {t}")
                    elif t in ("build", "all", "install"):
                        _add(build, f"make {t}")
            except OSError:
                pass
        elif kf.endswith("Cargo.toml"):
            _add(build, "cargo build")
            _add(test, "cargo test")
        elif kf.endswith("go.mod"):
            _add(build, "go build ./...")
            _add(test, "go test ./...")
    return build, test


def _infer_conventions(root: Path, key_files: list[str],
                       py_lines: dict[str, int],
                       languages: dict[str, int]) -> list[str]:
    """Heuristic code-convention notes an AI agent should follow."""
    conv: list[str] = []

    has_py = any("Python" in lang for lang in languages)
    if has_py:
        conv.append("Type-hint new code where the surrounding code does; "
                    "follow existing docstring style.")
        # Look for the dominant test layout.
        if any("tests" in kf or "/tests" in kf for kf in key_files) or \
                (root / "tests").is_dir():
            conv.append("Put new tests under the existing tests/ tree; "
                        "mirror the module layout of the code under test.")
        # Formatting config?
        for kf in key_files:
            if kf.endswith("pyproject.toml"):
                try:
                    txt = (root / kf).read_text(encoding="utf-8",
                                                errors="ignore")
                    if "[tool.black" in txt or "[tool.ruff" in txt:
                        conv.append("Format Python with the project's "
                                    "configured formatter (black/ruff) "
                                    "before committing.")
                    if "[tool.mypy" in txt:
                        conv.append("Keep mypy clean — this project "
                                    "type-checks itself.")
                except OSError:
                    pass
                break

    if (root / ".pre-commit-config.yaml").exists():
        conv.append("Run pre-commit hooks before proposing a commit.")

    # Big-file hint: keep diffs reviewable.
    big = [p for p, n in py_lines.items() if n > 1500]
    if big:
        conv.append("Large files exist (" + ", ".join(big[:3]) +
                    ") — prefer small, targeted edits over rewrites.")

    if any(kf.lower().startswith("license") for kf in key_files):
        conv.append("Respect the project's license — do not paste "
                    "incompatible third-party code.")

    conv.append("Keep changes minimal and focused; run the relevant test "
                "command before claiming a task is done.")
    return conv


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_agents_md(analysis: dict) -> str:
    """Render the analysis dict as an AGENTS.md markdown document."""
    name = analysis.get("name", "project")
    langs = analysis.get("languages", {})
    key_files = analysis.get("key_files", [])
    build_cmds = analysis.get("build_cmds", [])
    test_cmds = analysis.get("test_cmds", [])
    dirs = analysis.get("dirs", [])
    entry_points = analysis.get("entry_points", [])
    conv = analysis.get("conventions", [])
    n_files = analysis.get("files_count", 0)
    n_lines = analysis.get("lines_total", 0)

    lines: list[str] = []
    lines.append(f"# AGENTS.md — {name}")
    lines.append("")
    lines.append("_Auto-generated by `fullagent /init` — project snapshot "
                 "for AI coding agents. Verify anything that matters before "
                 "relying on it._")
    lines.append("")

    # Overview
    lines.append("## Project overview")
    lines.append("")
    top_langs = ", ".join(langs) if langs else "undetermined"
    lines.append(f"**{name}** — {top_langs} project with ~{n_files} files "
                 f"(~{n_lines} lines of code counted in major languages).")
    if entry_points:
        lines.append("")
        lines.append("Likely entry points: " +
                     ", ".join(f"`{e}`" for e in entry_points))
    lines.append("")

    # Languages
    lines.append("## Languages")
    lines.append("")
    if langs:
        for lang, count in list(langs.items())[:10]:
            lines.append(f"- {lang}: {count} files")
    else:
        lines.append("- (no source languages detected)")
    lines.append("")

    # Key files
    lines.append("## Key files")
    lines.append("")
    if key_files:
        for kf in key_files:
            lines.append(f"- `{kf}`")
    else:
        lines.append("- (no standard manifests found)")
    lines.append("")

    # Build/test
    lines.append("## Build / test commands")
    lines.append("")
    if build_cmds:
        lines.append("Build: " + ", ".join(f"`{c}`" for c in build_cmds))
    if test_cmds:
        lines.append("Test: " + ", ".join(f"`{c}`" for c in test_cmds))
    if not build_cmds and not test_cmds:
        lines.append("- (no build/test commands inferred)")
    lines.append("")

    # Directory layout
    lines.append("## Directory layout (top 2 levels)")
    lines.append("")
    if dirs:
        for d in dirs:
            lines.append(f"- `{d}/`")
    else:
        lines.append("- (flat layout)")
    lines.append("")

    # Conventions
    lines.append("## Code conventions")
    lines.append("")
    for c in conv:
        lines.append(f"- {c}")
    lines.append("")

    # Guidance
    lines.append("## Guidance for AI agents")
    lines.append("")
    lines.append("- Read the README and key files above before editing; "
                 "they are the source of truth for intent.")
    lines.append("- Reproduce a bug with a failing test first, then fix it, "
                 "then confirm the suite is green.")
    lines.append("- Do not commit or push unless the user asks; propose "
                 "diffs instead.")
    lines.append("- Never add secrets or API keys to the repo.")
    lines.append("- If a build/test command is unknown, ask rather than "
                 "guessing and running something destructive.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent tool registration
# ---------------------------------------------------------------------------

def _init_project_handler(path: str = ".", output: str = "AGENTS.md") -> str:
    """Handler behind the InitProject tool: analyze + write file."""
    root = Path(path).resolve()
    if not root.is_dir():
        return f"error: not a directory: {path}"
    out = (root / output) if not Path(output).is_absolute() else Path(output)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        analysis = analyze_project(root)
        doc = generate_agents_md(analysis)
        out.write_text(doc, encoding="utf-8")
    except OSError as e:
        return f"error: could not write {out}: {e}"
    top = next(iter(analysis["languages"]), "unknown")
    return (f"wrote {out} ({analysis['files_count']} files analyzed, "
            f"top language: {top})")


def register(agent: Any) -> None:
    """Register the InitProject tool on *agent* (expects .tools dict)."""
    from .tools import Tool  # local import: avoids any import-time cycles

    agent.tools["InitProject"] = Tool(
        "InitProject",
        "Analyze a codebase and generate an AGENTS.md project guide "
        "(languages, key files, build/test commands, conventions). "
        "Args: path (default '.'), output (default 'AGENTS.md').",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "output": {"type": "string"}},
         "required": []},
        _init_project_handler,
    )


# ---------------------------------------------------------------------------
# /init TUI command
# ---------------------------------------------------------------------------

def run_init_command(ui: Any, arg: str) -> None:
    """Handle the `/init` slash command in the TUI.

    *ui* must provide print_info / print_error and cwd access. Reuses the
    same analyze+generate pipeline as the InitProject tool. If AGENTS.md
    already exists it is left untouched and a notice is printed instead of
    overwriting (ask-before-overwrite via the arg: pass "overwrite" to
    force, anything else keeps the file).
    """
    root = Path.cwd()
    target = root / "AGENTS.md"
    force = arg.strip().lower() in ("overwrite", "--force", "-f", "force")

    if target.exists() and not force:
        ui.print_info(
            "AGENTS.md already exists — not overwriting. "
            "Use `/init overwrite` to replace it.", "yellow")
        return

    try:
        analysis = analyze_project(root)
        doc = generate_agents_md(analysis)
        target.write_text(doc, encoding="utf-8")
    except OSError as e:
        ui.print_error(f"/init failed: {e}")
        return

    n = analysis["files_count"]
    top = ", ".join(list(analysis["languages"])[:3]) or "unknown"
    ui.print_info(
        f"✓ wrote AGENTS.md ({n} files analyzed; {top})", "green")


# ---------------------------------------------------------------------------
# Self-test: `python3 -m fullagent.codeinit` → PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    here = Path(__file__).resolve().parent.parent  # ~/workspace/pytest-repo
    tmp = Path(tempfile.mkdtemp(prefix="codeinit-selftest-"))
    try:
        shutil.copytree(here, tmp / "repo",
                        ignore=shutil.ignore_patterns(
                            "__pycache__", "*.pyc", ".git"))
        analysis = analyze_project(tmp / "repo")
        doc = generate_agents_md(analysis)

        for section in ("## Project overview", "## Languages",
                        "## Key files", "## Build / test commands",
                        "## Directory layout (top 2 levels)",
                        "## Code conventions",
                        "## Guidance for AI agents"):
            assert section in doc, f"missing section: {section}"
        assert "Python" in (analysis["languages"] or {}), \
            "expected Python detected in fullagent repo"
        assert any("pyproject.toml" in kf
                   for kf in analysis["key_files"]), \
            "expected pyproject.toml in key files"
        assert any("pytest" in c for c in analysis["test_cmds"]), \
            "expected pytest among test commands"
        assert "fullagent" in analysis["name"].lower() or analysis["name"], \
            "expected a project name"

        # Exercise the handler + register paths with fakes.
        out = _init_project_handler(str(tmp / "repo"), "AGENTS.md")
        assert out.startswith("wrote "), f"bad handler result: {out}"
        assert (tmp / "repo" / "AGENTS.md").exists()

        class FakeAgent:
            def __init__(self):
                self.tools = {}

        fa = FakeAgent()
        register(fa)
        assert "InitProject" in fa.tools
        tool = fa.tools["InitProject"]
        res = tool.handler(path=str(tmp / "repo"), output="CLAUDE.md")
        assert res.startswith("wrote ")
        assert (tmp / "repo" / "CLAUDE.md").exists()

        print("PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _selftest()

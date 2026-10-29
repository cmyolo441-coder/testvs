"""SKILL FORGE — the self-evolving tool author.

When the agent keeps doing the same multi-step thing by hand, the Forge
lets it author a NEW tool (a Python function), prove it safe and correct
with deterministic checks, and register it into the live registry — so the
agent becomes more capable over time. The forge writes skills to
~/.fullagent/skills/<name>.py so they survive restarts.

Hard safety gate (mechanical, rung 1 — the skill NEVER runs unvalidated):
  1. Parse      — the source must be valid Python (ast.parse).
  2. Shape      — it must define exactly the declared entry function, with
                  a docstring and JSON-schema-style parameter declaration.
  3. Safety     — AST scan forbids: imports outside the allowlist, subprocess
                  / os.system / eval / exec / open-for-write / network /
                  dunder access / global statements. A skill is a pure
                  data-in/data-out function.
  4. Test       — the author must ship test cases (input -> expected output
                  substring). They run in a restricted namespace; ALL must
                  pass.
  Only a skill that passes all four gates is sealed skill.registered and
  handed to the registry. Anything else is sealed skill.rejected with the
  exact reason — the attempt is recorded, never hidden.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog, fold
from ._foundation import get_logger
from .tools import Tool

_log = get_logger("skills")

# imports a skill may use — pure stdlib, no side-effect modules
ALLOWED_IMPORTS = frozenset(
    "json math re string hashlib base64 datetime itertools functools "
    "collections statistics textwrap unicodedata urllib.parse html "
    "pathlib posixpath random typing dataclasses enum".split())

# AST nodes / names that are never allowed in a skill.
# getattr/hasattr are the classic sandbox escape: getattr(fn,
# "__globals__") reaches the very dunders the attribute gate forbids,
# because the check only inspects ast.Attribute nodes, never the string
# argument of a getattr() call.
_FORBIDDEN_CALLS = frozenset(
    "eval exec compile open input __import__ globals locals vars getattr "
    "hasattr setattr delattr breakpoint exit quit".split())
# File/process-mutation attributes are only dangerous when the *receiver* is
# a known dangerous module (os, shutil, pathlib, subprocess, io, builtins).
# Matching the bare attribute name used to reject benign code such as
# `class Door: def open(self): ...; Door().open()` — the dot is right, the
# semantics are completely different.
_FORBIDDEN_ATTRS = frozenset(
    "system popen exec execl execle execlp subprocess __subclasses__ "
    "__globals__ __code__ __builtins__ "
    # filesystem mutation through allowed modules (pathlib/shutil-style):
    # a "pure data-in/data-out" function never writes or deletes
    "write_text write_bytes open unlink rmdir rename replace rmtree "
    "touch mkdir symlink_to hardlink_to chmod chown".split())
# Module roots whose attribute access is treated as dangerous. The bare
# name `os.open`, `pathlib.Path.write_text`, `shutil.rmtree` are caught;
# a custom class with its own `open()` method is left alone.
_DANGEROUS_MODULES = frozenset(
    "os shutil pathlib subprocess sys builtins io fcntl "
    "posix nt _io".split())


# ---------------------------------------------------------------------------
# Skill record
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    source: str
    entry: str                     # the function name to call
    parameters: dict = field(default_factory=dict)
    tests: list[dict] = field(default_factory=list)  # {args, expect}
    status: str = "pending"        # pending | registered | rejected
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "entry": self.entry, "parameters": self.parameters,
                "tests": self.tests, "status": self.status,
                "reject_reason": self.reject_reason,
                "chars": len(self.source)}


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _validate_shape(tree: ast.Module, skill: Skill) -> str | None:
    """The entry function must exist, be a plain def, and have a docstring."""
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = [f.name for f in fns]
    if skill.entry not in names:
        return f"entry function {skill.entry!r} not defined (has: {names})"
    fn = next(f for f in fns if f.name == skill.entry)
    if not (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        return f"entry function {skill.entry!r} needs a docstring"
    if any(isinstance(n, (ast.AsyncFunctionDef, ast.ClassDef))
           for n in tree.body):
        return "skills must be plain functions — no classes/async"
    return None


def _dangerous_bindings(tree: ast.Module) -> set[str]:
    """Names bound to a dangerous module/class by import statements.

    ``from pathlib import Path`` binds ``Path``; ``import os as o`` binds
    ``o``. Without tracking these aliases the receiver-root check below
    only knew the literal module names, so ``Path(x).write_text(...)``
    slipped through the mutation gate.
    """
    names = set(_DANGEROUS_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in _DANGEROUS_MODULES:
                    names.add(a.asname or root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _DANGEROUS_MODULES:
                names.add(root)
                for a in node.names:
                    names.add(a.asname or a.name)
    return names


def _attr_root_is_dangerous(node: ast.Attribute,
                            dangerous: set[str] | None = None) -> bool:
    """Walk the receiver chain of `a.b.c.d` and return True iff the
    leftmost binding is a known dangerous module/class (os, shutil,
    pathlib.Path, ...). A method call on a locally-defined object (e.g.
    `door.open()`) returns False even when the attribute name itself is
    in the forbidden list."""
    names = _DANGEROUS_MODULES if dangerous is None else dangerous
    cur: ast.AST = node
    while True:
        if isinstance(cur, (ast.Attribute, ast.Subscript)):
            cur = cur.value
        elif isinstance(cur, ast.Call):
            # Receiver is a constructor call, e.g. Path(x).write_text()
            # or pathlib.Path(x).write_text(); inspect its function.
            fn = cur.func
            if isinstance(fn, ast.Name):
                return fn.id in names
            if isinstance(fn, ast.Attribute) and fn.attr in names:
                return True
            cur = fn
        else:
            break
    return isinstance(cur, ast.Name) and cur.id in names


def _validate_safety(tree: ast.Module) -> str | None:
    """AST scan: no forbidden imports, calls, attributes, or writes."""
    dangerous = _dangerous_bindings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return f"forbidden import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return f"forbidden import: from {node.module}"
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_CALLS:
                return f"forbidden call: {fn.id}()"
            if isinstance(fn, ast.Attribute) \
                    and fn.attr in _FORBIDDEN_ATTRS \
                    and _attr_root_is_dangerous(fn, dangerous):
                # only reject when the receiver chain is a known dangerous
                # module — a user-defined `Door.open()` is fine
                return (f"forbidden attribute on dangerous module: "
                        f".{fn.attr}()")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"dunder access forbidden: {node.attr}"
        elif isinstance(node, ast.Global):
            return "global statements forbidden in skills"
    return None


def _run_tests(skill: Skill, namespace: dict) -> str | None:
    """Run the author's test cases against the loaded entry function."""
    fn = namespace.get(skill.entry)
    if not callable(fn):
        return f"entry {skill.entry!r} did not load as callable"
    if not skill.tests:
        return "a skill must ship at least one test case"
    for i, t in enumerate(skill.tests, 1):
        args = t.get("args") or {}
        expect = str(t.get("expect", ""))
        try:
            got = str(fn(**args))
        except Exception as e:
            return f"test {i} raised {type(e).__name__}: {e}"
        if expect and expect not in got:
            return (f"test {i} failed: expected {expect!r} in output, "
                    f"got {got[:120]!r}")
    return None


# ---------------------------------------------------------------------------
# SkillForge
# ---------------------------------------------------------------------------

class SkillForge:
    """Author, validate, and register new tools over the event log."""

    def __init__(self, log: EventLog, skills_dir: Path | None = None) -> None:
        self.log = log
        self.skills_dir = Path(skills_dir) if skills_dir else None
        if self.skills_dir:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry: dict[str, Skill] = {}

    def author(self, skill: Skill) -> Skill:
        """Run the four gates. On success: persist + seal skill.registered.
        On failure: seal skill.rejected with the exact reason."""
        self.log.append("skill.authored", skill.to_dict(), actor="sovereign")

        # gate 1: parse
        try:
            tree = ast.parse(skill.source)
        except SyntaxError as e:
            return self._reject(skill, f"does not parse: {e}")

        # gate 2: shape
        err = _validate_shape(tree, skill)
        if err:
            return self._reject(skill, err)

        # gate 3: safety
        err = _validate_safety(tree)
        if err:
            return self._reject(skill, err)

        # gate 4: load in isolation + run the shipped tests
        try:
            namespace = self._load(skill)
        except Exception as e:
            return self._reject(skill, f"failed to load: {e}")
        err = _run_tests(skill, namespace)
        if err:
            return self._reject(skill, err)

        # all gates passed — persist + register
        skill.status = "registered"
        if self.skills_dir:
            try:
                self.skills_dir.mkdir(parents=True, exist_ok=True)
                src = self.skills_dir / f"{skill.name}.py"
                tmp = src.with_suffix(".py.tmp")
                tmp.write_text(skill.source, encoding="utf-8")
                os.replace(tmp, src)
                # metadata sidecar: entry/parameters/tests must survive the
                # restart so load_persisted can re-run EVERY gate faithfully
                meta = self.skills_dir / f"{skill.name}.json"
                mtmp = meta.with_suffix(".json.tmp")
                mtmp.write_text(json.dumps(skill.to_dict()), encoding="utf-8")
                os.replace(mtmp, meta)
            except OSError:
                pass  # in-memory registration still works without disk
        self.registry[skill.name] = skill
        self.log.append("skill.validated",
                        {"name": skill.name, "tests": len(skill.tests)},
                        actor="kernel")
        self.log.append("skill.registered", skill.to_dict(), actor="kernel")
        return skill

    def _load(self, skill: Skill) -> dict:
        """Load the skill source in an isolated module namespace."""
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as f:
            f.write(skill.source)
            tmp = f.name
        try:
            spec = importlib.util.spec_from_file_location(
                f"fullagent_skill_{skill.name}", tmp)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(spec.name, None)
            return module.__dict__
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _reject(self, skill: Skill, reason: str) -> Skill:
        skill.status = "rejected"
        skill.reject_reason = reason
        self.log.append("skill.rejected",
                        {"name": skill.name, "reason": reason},
                        actor="kernel")
        return skill

    # -- reload persisted skills ------------------------------------------------

    def load_persisted(self) -> int:
        """Re-register skills from the skills dir (restart survival).

        A persisted skill is only trusted after it passes ALL FOUR gates
        again against the on-disk source — disk content is never assumed
        to be the same bytes that were validated before the restart."""
        if not self.skills_dir:
            return 0
        count = 0
        for p in sorted(self.skills_dir.glob("*.py")):
            name = p.stem
            if name in self.registry:
                continue
            try:
                source = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # metadata sidecar written at author() time; fall back to a
            # name==entry guess for pre-sidecar layouts, but then the full
            # gate run below decides — nothing registers unvalidated
            meta_path = self.skills_dir / f"{name}.json"
            entry = name
            parameters: dict = {}
            tests: list[dict] = []
            description = "(persisted)"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    entry = str(meta.get("entry") or name)
                    parameters = dict(meta.get("parameters") or {})
                    tests = list(meta.get("tests") or [])
                    description = str(meta.get("description") or description)
                except (OSError, ValueError):
                    pass
            skill = Skill(name=name, description=description,
                          source=source, entry=entry,
                          parameters=parameters, tests=tests)
            # gates 1-4, exactly as author() runs them
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if _validate_shape(tree, skill) is not None:
                continue
            if _validate_safety(tree) is not None:
                continue
            try:
                namespace = self._load(skill)
            except Exception:
                continue
            if _run_tests(skill, namespace) is not None:
                continue
            skill.status = "registered"
            self.registry[name] = skill
            count += 1
        return count

    # -- projections ---------------------------------------------------------------

    def skills(self) -> list[dict]:
        return fold(self.log).skill_events

    def registered(self) -> list[str]:
        return sorted(self.registry)

    def format_status(self) -> str:
        evs = self.skills()
        authored = sum(1 for e in evs if e["type"] == "skill.authored")
        registered = sum(1 for e in evs if e["type"] == "skill.registered")
        rejected = sum(1 for e in evs if e["type"] == "skill.rejected")
        lines = ["SKILL FORGE — the self-evolving tool author",
                 f"  authored {authored}   registered {registered}   "
                 f"rejected {rejected}"]
        for name in self.registered():
            lines.append(f"    ◆ {name}")
        for e in evs:
            if e["type"] == "skill.rejected":
                lines.append(f"    ✗ {e.get('name')}: "
                             f"{e.get('reason', '')[:60]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude Code-style SKILL.md discovery
# ---------------------------------------------------------------------------
# Skills installed as ``~/.fullagent/skills/<name>/SKILL.md``. Each directory
# is one skill; the SKILL.md gives the skill's name (first ``# Title`` line,
# or a ``name:`` frontmatter line), a one-paragraph description, and the
# instruction body the model should follow. Discovered skills are exposed to
# the agent as ``skill_<name>`` tools whose handler returns the instructions,
# and to the user via the TUI ``/skills`` command (:func:`handle_skills`).
#
# This is additive and independent of SkillForge above (which persists
# authored tools as flat ``<name>.py`` files in the same root directory —
# those are ignored here because only ``*/SKILL.md`` directories count).

SKILLS_MD_ROOT = os.path.join(os.path.expanduser("~"),
                              ".fullagent", "skills")

_TOOL_NAME_BAD = re.compile(r"[^a-z0-9_]")


def _parse_skill_md(text: str,
                    source: str = "<memory>") -> dict | None:
    """Parse a SKILL.md into name/description/instructions.

    Returns None (with a log note) for malformed files — never raises.
    """
    if not text or not text.strip():
        _log.warning("skills: %s is empty — skipped", source)
        return None
    lines = text.splitlines()

    # strip a leading --- frontmatter block, but keep its `name:` if any
    idx = 0
    fm_name: str | None = None
    if lines and lines[0].strip() == "---":
        idx = 1
        while idx < len(lines) and lines[idx].strip() != "---":
            m = re.match(r"^\s*name\s*:\s*(.+?)\s*$", lines[idx])
            if m and fm_name is None:
                fm_name = m.group(1).strip()
            idx += 1
        idx += 1  # skip closing ---

    title: str | None = None
    title_idx = -1
    for i in range(idx, len(lines)):
        ln = lines[i]
        if fm_name is None:
            m = re.match(r"^\s*name\s*:\s*(.+?)\s*$", ln)
            if m:
                fm_name = m.group(1).strip()
                continue
        if ln.startswith("# ") and title is None:
            title = ln[2:].strip()
            title_idx = i
    name = title or fm_name
    if not name:
        _log.warning("skills: %s has no '# Title' or 'name:' — skipped",
                     source)
        return None

    rest = lines[(title_idx + 1) if title_idx >= 0 else idx:]
    # skip leftover frontmatter-style key: value lines at the top
    j = 0
    while j < len(rest) and re.match(r"^\s*[A-Za-z0-9_-]+\s*:\s*\S",
                                     rest[j]):
        j += 1
    rest = rest[j:]

    # first non-empty paragraph = description
    k = 0
    while k < len(rest) and not rest[k].strip():
        k += 1
    para: list[str] = []
    while k < len(rest) and rest[k].strip():
        para.append(rest[k].strip())
        k += 1
    description = " ".join(para)
    instructions = "\n".join(rest[k:]).strip()

    if not description:
        _log.warning("skills: %r (%s) has no description — skipped",
                     name, source)
        return None
    if not instructions:
        _log.warning("skills: %r (%s) has no instructions body — skipped",
                     name, source)
        return None
    return {"name": name, "description": description,
            "instructions": instructions}


def discover_skills(root: str | Path | None = None) -> dict[str, dict]:
    """Scan ``<root>/*/SKILL.md``; returns {lower_name: record}.

    Malformed or unreadable skills are skipped with a log note — never
    raise.
    """
    base = Path(root) if root else Path(SKILLS_MD_ROOT)
    found: dict[str, dict] = {}
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.is_dir():
            continue
        md = entry / "SKILL.md"
        if not md.is_file():
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            _log.warning("skills: cannot read %s (%s) — skipped", md, e)
            continue
        rec = _parse_skill_md(text, source=str(md))
        if rec is None:
            continue
        rec["path"] = str(md)
        key = rec["name"].lower()
        if key in found:
            _log.warning("skills: duplicate skill name %r at %s — skipped",
                         rec["name"], md)
            continue
        found[key] = rec
    return found


def _sanitize_tool_name(name: str) -> str:
    s = _TOOL_NAME_BAD.sub("_", name.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_") or "skill"
    return "skill_" + s[:48]


class SkillMDManager:
    """Holds the discovered SKILL.md skills for one agent."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = str(root) if root else SKILLS_MD_ROOT
        self.skills: dict[str, dict] = discover_skills(root)
        self.tool_names: dict[str, str] = {}  # tool_name -> lower skill name
        for key in sorted(self.skills):
            base = _sanitize_tool_name(self.skills[key]["name"])
            tool_name = base
            n = 2
            while tool_name in self.tool_names:
                tool_name = f"{base}_{n}"
                n += 1
            self.tool_names[tool_name] = key
            self.skills[key]["tool_name"] = tool_name

    def list_skills(self) -> list[dict]:
        """list of {name, description, path} dicts."""
        return [{"name": self.skills[k]["name"],
                 "description": self.skills[k]["description"],
                 "path": self.skills[k]["path"]}
                for k in sorted(self.skills)]

    def get(self, name: str | None) -> dict | None:
        """Look up a skill by name (case-insensitive) or tool name."""
        if not name:
            return None
        key = name.strip().lower()
        if key in self.skills:
            return self.skills[key]
        k = self.tool_names.get(key) or self.tool_names.get("skill_" + key)
        return self.skills.get(k) if k else None


def make_skill_tool(skill: dict) -> Tool:
    """Build the ``skill_<name>`` Tool for one discovered skill."""
    name = skill["name"]

    def _handle_skill(task: str | None = None, **kwargs) -> str:
        return f"SKILL {name} INSTRUCTIONS:\n{skill['instructions']}"

    return Tool(
        name=skill["tool_name"],
        description=(f"Skill '{name}': {skill['description']} "
                     "Invoke to load this skill's full instructions into "
                     "context, then follow them for the user's request. "
                     "Optional 'task' gives the skill extra context."),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "optional extra context for the skill",
                },
            },
            "required": [],
        },
        handler=_handle_skill,
    )


def register(agent, root: str | Path | None = None) -> None:
    """Wire discovered SKILL.md skills into an agent (duck-typed).

    Attaches ``agent.skill_manager`` (a SkillMDManager with
    ``list_skills()``) and registers one ``skill_<name>`` tool per skill in
    ``agent.tools``. Never raises: startup must not break on bad skills.
    """
    try:
        mgr = SkillMDManager(root)
        agent.skill_manager = mgr
        tools = getattr(agent, "tools", None)
        if tools is None:
            tools = agent.tools = {}
        for tool_name, key in sorted(mgr.tool_names.items()):
            if tool_name in tools:
                _log.warning("skills: tool %r already exists — skipped",
                             tool_name)
                continue
            tools[tool_name] = make_skill_tool(mgr.skills[key])
    except Exception as e:  # never break agent startup
        _log.warning("skills: register failed (%s) — skills unavailable", e)


def handle_skills(ui, arg: str) -> None:
    """TUI ``/skills`` command.

    ``/skills``        — list installed skills (name + one-line description)
    ``/skills <name>`` — show the full instructions for one skill
    """
    info = getattr(ui, "print_info", None) or print
    err = getattr(ui, "print_error", None) or print

    mgr = None
    agent = getattr(ui, "agent", None)
    if agent is not None:
        mgr = getattr(agent, "skill_manager", None)
    if not isinstance(mgr, SkillMDManager):
        try:
            mgr = SkillMDManager()
        except Exception:
            err("skills unavailable")
            return

    name = (arg or "").strip()
    if name:
        rec = mgr.get(name)
        if rec is None:
            err(f"no skill named {name!r} — /skills lists installed skills")
            return
        info(f"SKILL {rec['name']} INSTRUCTIONS:\n{rec['instructions']}")
        return

    items = mgr.list_skills()
    if not items:
        info("no skills installed — add SKILL.md files under "
             "~/.fullagent/skills/<name>/")
        return
    lines = [f"skills ({len(items)}):"]
    for s in items:
        lines.append(f"  {s['name']} — {s['description']}")
    info("\n".join(lines))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "skills.jsonl")
        forge = SkillForge(log, skills_dir=Path(td) / "skills")

        good = Skill(
            name="word_count",
            description="count words in a text",
            entry="word_count",
            source='''def word_count(text):
    """Count the words in a text string."""
    return f"words: {len(text.split())}"
''',
            parameters={"text": {"type": "string"}},
            tests=[{"args": {"text": "one two three"}, "expect": "words: 3"}],
        )
        s = forge.author(good)
        assert s.status == "registered", s.reject_reason
        assert "word_count" in forge.registered()
        assert (Path(td) / "skills" / "word_count.py").exists()

        # a skill with a failing test is rejected
        bad_test = Skill(
            name="wrong",
            description="lies about its output",
            entry="wrong",
            source='''def wrong(x):
    """Return a greeting."""
    return "hello"
''',
            tests=[{"args": {"x": 1}, "expect": "goodbye"}],
        )
        s = forge.author(bad_test)
        assert s.status == "rejected" and "test 1 failed" in s.reject_reason

        # a skill that imports subprocess is rejected at the safety gate
        evil = Skill(
            name="evil",
            description="tries to shell out",
            entry="evil",
            source='''import subprocess

def evil(cmd):
    """Run a command."""
    return subprocess.run(cmd, shell=True)
''',
            tests=[{"args": {"cmd": "ls"}, "expect": ""}],
        )
        s = forge.author(evil)
        assert s.status == "rejected" and "forbidden import" in s.reject_reason

        # eval is rejected
        sneaky = Skill(
            name="sneaky", description="eval", entry="sneaky",
            source='''def sneaky(x):
    """Evaluate."""
    return eval(x)
''',
            tests=[{"args": {"x": "1"}, "expect": "1"}],
        )
        assert forge.author(sneaky).status == "rejected"

        # a skill without a docstring is rejected at the shape gate
        nodoc = Skill(
            name="nodoc", description="no docstring", entry="nodoc",
            source="def nodoc(x):\n    return x\n",
            tests=[{"args": {"x": 1}, "expect": "1"}],
        )
        s = forge.author(nodoc)
        assert s.status == "rejected" and "docstring" in s.reject_reason

        # a skill with no tests is rejected
        notests = Skill(
            name="notests", description="no tests", entry="notests",
            source='''def notests(x):
    """Identity."""
    return x
''',
            tests=[],
        )
        s = forge.author(notests)
        assert s.status == "rejected" and "test case" in s.reject_reason

        # invalid python is rejected at the parse gate
        broken = Skill(name="broken", description="bad syntax",
                       entry="broken", source="def broken(:\n", tests=[])
        assert forge.author(broken).status == "rejected"

        # persisted skills reload after a restart
        forge2 = SkillForge(log, skills_dir=Path(td) / "skills")
        assert forge2.load_persisted() == 1
        assert "word_count" in forge2.registered()

        # ledger reflects everything
        evs = forge.skills()
        types = {e["type"] for e in evs}
        assert {"skill.authored", "skill.registered",
                "skill.rejected"} <= types
        assert "SKILL FORGE" in forge.format_status()

    # ---- SKILL.md discovery self-test -------------------------------------
    import textwrap

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "skills"

        def _write_skill(d, content):
            d = root / d
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(textwrap.dedent(content),
                                        encoding="utf-8")

        # good: # title + paragraph + body
        _write_skill("commit", """\
            # Commit
            Write clear git commit messages for staged changes.

            ## Steps
            1. Run git diff --staged.
            2. Summarise the change in one line.
            """)
        # good: frontmatter name, no # title
        _write_skill("pdf", """\
            ---
            name: pdf
            ---
            Extract text from PDF files.

            Use a streaming parser; never load whole PDFs into memory.
            """)
        # malformed: no title and no name:
        _write_skill("broken1", "just some prose\nwith no title at all\n")
        # malformed: title + description but empty body
        _write_skill("broken2", "# Empty\n\nA skill with no body.\n")
        # not a skill: directory without SKILL.md
        (root / "noskill").mkdir(parents=True, exist_ok=True)

        mgr = SkillMDManager(root)
        items = mgr.list_skills()
        assert len(items) == 2, items
        names = {s["name"] for s in items}
        assert names == {"Commit", "pdf"}, names
        assert all("tool_name" in mgr.skills[k] for k in mgr.skills)

        class FakeAgent2:
            def __init__(self):
                self.tools = {}

        a2 = FakeAgent2()
        register(a2, root)
        assert isinstance(a2.skill_manager, SkillMDManager)
        assert "skill_commit" in a2.tools, sorted(a2.tools)
        assert "skill_pdf" in a2.tools, sorted(a2.tools)

        out = a2.tools["skill_commit"].handler()
        assert out.startswith("SKILL Commit INSTRUCTIONS:\n"), out[:60]
        assert "git diff --staged" in out
        # optional task arg is accepted and ignored
        out2 = a2.tools["skill_pdf"].handler(task="extract page 3")
        assert out2.startswith("SKILL pdf INSTRUCTIONS:\n"), out2[:60]
        assert "streaming parser" in out2

        # TUI handler
        class FakeUI:
            def __init__(self, agent=None):
                self.agent = agent
                self.infos = []
                self.errors = []

            def print_info(self, msg):
                self.infos.append(msg)

            def print_error(self, msg):
                self.errors.append(msg)

        ui = FakeUI(a2)
        handle_skills(ui, "")
        assert len(ui.infos) == 1 and "skills (2):" in ui.infos[0], ui.infos
        assert "Commit — Write clear git commit messages" in ui.infos[0]
        handle_skills(ui, "commit")
        assert len(ui.infos) == 2
        assert ui.infos[1].startswith("SKILL Commit INSTRUCTIONS:\n")
        handle_skills(ui, "nope")
        assert len(ui.errors) == 1 and "no skill named" in ui.errors[0]

        # UI without an agent falls back to fresh discovery from the real
        # root (empty here) without crashing
        ui2 = FakeUI()
        handle_skills(ui2, "")
        assert ui2.infos and "no skills installed" in ui2.infos[0]

        # sanitized-name collision: "My Skill" vs "my-skill"
        root2 = Path(td) / "skills2"
        _w = root2 / "a"
        _w.mkdir(parents=True, exist_ok=True)
        (root2 / "a" / "SKILL.md").write_text(
            "# My Skill\n\nFirst.\n\nDo first thing.\n", encoding="utf-8")
        _w = root2 / "b"
        _w.mkdir(parents=True, exist_ok=True)
        (root2 / "b" / "SKILL.md").write_text(
            "# my-skill\n\nSecond.\n\nDo second thing.\n", encoding="utf-8")
        mgr2 = SkillMDManager(root2)
        tns = sorted(mgr2.tool_names)
        assert tns == ["skill_my_skill", "skill_my_skill_2"], tns

        # missing root never raises
        assert discover_skills(Path(td) / "does-not-exist") == {}

    print("SKILLS SELF-TEST PASS")

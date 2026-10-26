"""Static answer to one question about an INSTALLED bundle: will Hermes exec it?

Hermes copies the bundle into ``$HERMES_HOME/plugins/omh`` and loads it with
its own interpreter, which has no reason to have the ``omh`` package on its
path. On the memory-provider lane it execs every top-level file of the bundle
eagerly and keeps the half-initialized module in ``sys.modules`` when one
raises, so the next lazy import of that module fails on the NAME -- which is
how one dead bridge became a logged hook warning on every tool call (#1623,
reported with a reproduction by @tonalenar).

`tests/test_plugin_bundle_standalone.py` keeps the SOURCE tree from regressing
by importing every bundle module with `omh` blocked. That gate cannot reach the
copy on a user's machine, which may be stale, hand-copied, or from a different
generation than the `omh` package on the path (#1670). This module is the
lane-independent static half: it reads the installed files and answers from
their syntax, so it needs neither Hermes nor an import of the bundle.

What counts as a finding, and why each rule is drawn where it is:

- Only module-level statements. A `from omh... import ...` inside a function
  body runs when the feature is used, and the shipped shape has the feature
  report its own absence there; flagging it would be a false positive. Class
  bodies, `if`, `with`, `for`, `while` and `try` all execute during the exec
  Hermes performs, so the walk descends into them and stops only at a
  function.
- `try:` guards only when a handler catches `ImportError` or wider. A handler
  catching `ModuleNotFoundError` alone does NOT guard: the error Hermes' cached
  stub raises is a plain `ImportError`, and a guard written for only the
  narrower shape is precisely what #1623 was.
- `if TYPE_CHECKING:` guards because that branch never runs outside a type
  checker. No other `if` guards -- a condition that is taken still raises.
- A relative import that climbs past the bundle root is the same defect in the
  other spelling. Inside this repo the bundle sits at
  `omh.plugin_bundle.omh.*`, where `from ...routing import x` resolves; in the
  install it is the top-level package, where the same line raises.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

SCAN_SCHEMA_VERSION = "plugin_bundle_import_scan/v1"

CORE_PACKAGE = "omh"

# Catching any of these catches the `ImportError` a cached stub raises.
# `ModuleNotFoundError` is deliberately absent: it is a SUBCLASS, so a handler
# naming only it lets the stub's plain `ImportError` through.
_GUARDING_EXCEPTION_NAMES = frozenset({"ImportError", "Exception", "BaseException"})

_TYPE_CHECKING_NAMES = frozenset({"TYPE_CHECKING", "typing.TYPE_CHECKING"})

ABSOLUTE_CORE_IMPORT = "absolute_core_import"
ESCAPING_RELATIVE_IMPORT = "escaping_relative_import"
UNREADABLE_MODULE = "unreadable_module"

_FINDING_DISPLAY_LIMIT = 5


def scan_bundle_core_imports(bundle_dir: Path) -> dict[str, Any]:
    """Report every module-level import in ``bundle_dir`` a plain host cannot run."""
    if not bundle_dir.is_dir():
        return {
            "schema_version": SCAN_SCHEMA_VERSION,
            "scanned": False,
            "ok": True,
            "reason": "plugin_bundle_not_installed",
            "bundle_dir": str(bundle_dir),
            "module_count": 0,
            "findings": [],
        }
    try:
        module_paths = bundle_module_paths(bundle_dir)
    except OSError as error:
        return {
            "schema_version": SCAN_SCHEMA_VERSION,
            "scanned": False,
            "ok": True,
            "reason": "plugin_bundle_unreadable",
            "detail": f"{type(error).__name__}: {error}",
            "bundle_dir": str(bundle_dir),
            "module_count": 0,
            "findings": [],
        }
    findings: list[dict[str, Any]] = []
    for path in module_paths:
        findings.extend(_module_findings(path, bundle_dir))
    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "scanned": True,
        "ok": not findings,
        "reason": "no_unguarded_core_imports" if not findings else "unguarded_core_imports",
        "bundle_dir": str(bundle_dir),
        "module_count": len(module_paths),
        "findings": findings,
    }


def bundle_module_paths(bundle_dir: Path) -> tuple[Path, ...]:
    """Every ``*.py`` in the bundle, read off the directory rather than listed.

    Derived on purpose: a written list is a list of the modules someone thought
    of, and the module this exists to catch is the one nobody thought of.
    `__init__.py` is included -- Hermes execs it first, so an unguarded import
    there takes the whole bundle down.
    """
    return tuple(sorted(bundle_dir.rglob("*.py")))


def describe_findings(findings: list[dict[str, Any]], *, limit: int = _FINDING_DISPLAY_LIMIT) -> str:
    """One operator-readable line naming the module, the line, and the import."""
    shown = [
        f"{finding['module']}:{finding['line']} `{finding['statement']}`"
        for finding in findings[:limit]
    ]
    remaining = len(findings) - len(shown)
    if remaining > 0:
        shown.append(f"(+{remaining} more)")
    return "; ".join(shown)


def _module_findings(path: Path, bundle_dir: Path) -> list[dict[str, Any]]:
    module = path.relative_to(bundle_dir).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as error:
        # A file the bundle carries that cannot be parsed is not an absent
        # finding: Hermes execs it too, and it will raise there. Reported as
        # its own kind so the remedy is not confused with a guard mistake.
        return [
            {
                "module": module,
                "line": getattr(error, "lineno", 0) or 0,
                "kind": UNREADABLE_MODULE,
                "statement": f"{type(error).__name__}: {error}",
            }
        ]
    # `omh/awareness.py` may climb one level (to `omh`); `omh/hooks/x.py` two.
    # One more than that leaves the bundle, which is where the install differs
    # from this repo, where the same line resolves to `omh.routing` and works.
    max_relative_level = len(path.relative_to(bundle_dir).parts)
    collector = _Collector(module=module, max_relative_level=max_relative_level)
    collector.visit_body(tree.body, guarded=False)
    return collector.findings


class _Collector:
    """Walks the statements Hermes executes when it execs one bundle file."""

    def __init__(self, *, module: str, max_relative_level: int) -> None:
        self.module = module
        self.max_relative_level = max_relative_level
        self.findings: list[dict[str, Any]] = []

    def visit_body(self, body: list[ast.stmt], *, guarded: bool) -> None:
        for node in body:
            self.visit(node, guarded=guarded)

    def visit(self, node: ast.stmt, *, guarded: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return  # Deferred to call time; the feature reports its own absence.
        if isinstance(node, ast.Import):
            self._record_import(node, guarded=guarded)
        elif isinstance(node, ast.ImportFrom):
            self._record_import_from(node, guarded=guarded)
        elif isinstance(node, ast.Try):
            self.visit_body(node.body, guarded=guarded or _catches_import_error(node))
            for handler in node.handlers:
                # A handler's own imports are not protected by the try it
                # belongs to; only a nested try or an outer one guards them.
                self.visit_body(handler.body, guarded=guarded)
            self.visit_body(node.orelse, guarded=guarded)
            self.visit_body(node.finalbody, guarded=guarded)
        elif isinstance(node, ast.If):
            self.visit_body(node.body, guarded=guarded or _is_type_checking_test(node.test))
            self.visit_body(node.orelse, guarded=guarded)
        elif isinstance(node, ast.ClassDef):
            self.visit_body(node.body, guarded=guarded)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            self.visit_body(node.body, guarded=guarded)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            self.visit_body(node.body, guarded=guarded)
            self.visit_body(node.orelse, guarded=guarded)

    def _record_import(self, node: ast.Import, *, guarded: bool) -> None:
        if guarded:
            return
        for alias in node.names:
            if _is_core_module(alias.name):
                self._add(node, ABSOLUTE_CORE_IMPORT)
                return

    def _record_import_from(self, node: ast.ImportFrom, *, guarded: bool) -> None:
        if guarded:
            return
        if node.level == 0:
            if _is_core_module(node.module or ""):
                self._add(node, ABSOLUTE_CORE_IMPORT)
        elif node.level > self.max_relative_level:
            self._add(node, ESCAPING_RELATIVE_IMPORT)

    def _add(self, node: ast.stmt, kind: str) -> None:
        self.findings.append(
            {
                "module": self.module,
                "line": node.lineno,
                "kind": kind,
                "statement": " ".join(ast.unparse(node).split()),
            }
        )


def _is_core_module(name: str) -> bool:
    return name == CORE_PACKAGE or name.startswith(f"{CORE_PACKAGE}.")


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:  # bare `except:`
            return True
        candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        for candidate in candidates:
            if _exception_name(candidate) in _GUARDING_EXCEPTION_NAMES:
                return True
    return False


def _exception_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_type_checking_test(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _TYPE_CHECKING_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


__all__ = [
    "ABSOLUTE_CORE_IMPORT",
    "ESCAPING_RELATIVE_IMPORT",
    "SCAN_SCHEMA_VERSION",
    "UNREADABLE_MODULE",
    "bundle_module_paths",
    "describe_findings",
    "scan_bundle_core_imports",
]

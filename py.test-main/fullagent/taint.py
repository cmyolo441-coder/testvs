"""TAINT — real static analysis over the AST.

Not regex, not guessing: this parses the source with `ast` and computes
three genuine analyses:

  * TAINT TRACKING   — dataflow from declared SOURCES (user input, env,
                       network, file reads) to declared SINKS (eval, exec,
                       subprocess, os.system, sql, file writes). A variable
                       assigned from a source is tainted; taint propagates
                       through assignments, function args, and string
                       formatting. A tainted value reaching a sink is a
                       finding with the exact propagation path.
  * COMPLEXITY       — cyclomatic complexity per function (branches + 1),
                       plus line/arg counts. High complexity = high risk.
  * DEPENDENCY CYCLES — module-level import graph; a cycle (A imports B
                       imports A) is a real structural smell, detected with
                       iterative DFS.

Every analysis result is sealed into the event log so findings are
auditable and replayable. Pure stdlib, deterministic, no model calls.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("taint")

# Sources: calls whose return value is attacker/user controlled.
DEFAULT_SOURCES = frozenset(
    "input raw_input request.get request.form request.args request.json "
    "os.environ os.getenv sys.argv open read readline readlines recv "
    "recvfrom urlopen fetch".split())

# Sinks: calls that are dangerous when fed tainted data.
DEFAULT_SINKS = frozenset(
    "eval exec compile os.system os.popen os.execv os.execve os.execl "
    "os.execle os.execlp os.execvp os.spawnl os.spawnle os.spawnlp "
    "os.spawnv os.spawnve os.spawnvp os.spawnvpe "
    "subprocess.run subprocess.call subprocess.Popen subprocess.check_call "
    "subprocess.check_output popen "
    "pickle.loads yaml.load marshal.loads cursor.execute execute "
    "open write send sendall render_template".split())


# ---------------------------------------------------------------------------
# Speed: fast-path + caching
# ---------------------------------------------------------------------------
# _RISK_TOKENS is a cheap substring pre-scan. If NONE of these tokens
# appears in the source, no source/sink can possibly match and the AST
# parse is skipped entirely — obviously-safe inputs return instantly.
_RISK_TOKENS = frozenset(
    "eval exec compile system popen spawn subprocess pickle yaml marshal "
    "cursor execute render_template input raw_input request environ getenv "
    "argv open read recv urlopen fetch socket send".split())


def _looks_risky(source: str) -> bool:
    sl = source.lower()
    return any(tok in sl for tok in _RISK_TOKENS)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class TaintFinding:
    sink: str
    line: int
    source: str
    source_line: int
    path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sink": self.sink, "line": self.line,
                "source": self.source, "source_line": self.source_line,
                "path": self.path}


@dataclass
class Complexity:
    name: str
    line: int
    complexity: int
    lines: int
    args: int

    def to_dict(self) -> dict:
        return {"name": self.name, "line": self.line,
                "complexity": self.complexity, "lines": self.lines,
                "args": self.args}


# ---------------------------------------------------------------------------
# Taint analysis
# ---------------------------------------------------------------------------

def _call_name(node: ast.Call) -> str:
    """Best-effort dotted name of a call: eval / os.system / request.get."""
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        parts = []
        cur = fn
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _expr_name(node: ast.expr) -> str:
    """Dotted name of a plain expression: os.environ / sys.argv / req.json."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return ""


class TaintAnalyzer:
    """Intra-procedural taint dataflow over one module's AST."""

    def __init__(self, sources: frozenset = DEFAULT_SOURCES,
                 sinks: frozenset = DEFAULT_SINKS) -> None:
        self.sources = sources
        self.sinks = sinks
        # verdict cache: repeated scans of the same source are O(1)
        self._cache: dict[tuple, list[TaintFinding]] = {}

    @staticmethod
    def _copy(findings: list[TaintFinding]) -> list[TaintFinding]:
        # findings are mutable — hand out copies so a caller can never
        # corrupt the cached verdict
        return [TaintFinding(f.sink, f.line, f.source, f.source_line,
                             list(f.path)) for f in findings]

    def _is_source_name(self, name: str) -> bool:
        """Does a dotted call name resolve to a declared source?

        Matches exact names, dotted prefixes (request.args.get via the
        request.args source) and bare imported names (getenv via
        os.getenv)."""
        for s in self.sources:
            if name == s or name.startswith(s + ".") \
                    or name.endswith("." + s):
                return True
            if "." in s and s.endswith("." + name):
                return True
        return False

    def _is_sink_name(self, name: str) -> bool:
        for s in self.sinks:
            if name == s or name.endswith("." + s):
                return True
        return False

    def analyze(self, source: str) -> list[TaintFinding]:
        # fast path: obviously-safe input skips the AST parse entirely
        if not _looks_risky(source):
            return []
        # key by the values, not hash(): two different sources with a
        # hash collision would otherwise return each other's findings
        key = (source, self.sources, self.sinks)
        hit = self._cache.get(key)
        if hit is not None:
            return self._copy(hit)
        findings = _analyze_impl(source, self.sources, self.sinks,
                                 self._is_source_name, self._is_sink_name)
        # bound the cache: safety verdicts for hot files stay O(1)
        # without unbounded memory growth on huge scans
        if len(self._cache) < 512:
            self._cache[key] = findings
        return self._copy(findings)


def _analyze_impl(source: str, sources: frozenset, sinks: frozenset,
                  is_source_name, is_sink_name) -> list[TaintFinding]:
    """The real dataflow. Module-level so the logic is testable without
    an analyzer instance."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    # name -> (source_name, source_line, path)
    tainted: dict[str, tuple[str, int, list[str]]] = {}
    findings: list[TaintFinding] = []

    def is_source_call(node: ast.expr) -> tuple[bool, str, int]:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name = _call_name(sub)
                if is_source_name(name):
                    return True, name, getattr(sub, "lineno", 0)
        # Bare attribute/subscript reads of a declared source: the common
        # os.environ["X"], sys.argv[1], request.args[...] forms contain no
        # Call, so the loop above missed them entirely.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute):
                name = _expr_name(sub)
                if name and is_source_name(name):
                    return True, name, getattr(sub, "lineno", 0)
        return False, "", 0

    def tainted_names(node: ast.expr) -> list[str]:
        return [n.id for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id in tainted]

    def note_sink(call: ast.Call, arg: ast.expr) -> None:
        """Record one finding for a tainted argument reaching a sink,
        either via a tainted NAME or via a direct SOURCE CALL such as
        eval(input()) with no intermediate assignment."""
        name = _call_name(call)
        for tn in tainted_names(arg):
            src, sline, path = tainted[tn]
            findings.append(TaintFinding(
                sink=name, line=getattr(call, "lineno", 0),
                source=src, source_line=sline,
                path=path + [f"{name}()"]))
            return
        ok, sname, sline = is_source_call(arg)
        if ok:
            findings.append(TaintFinding(
                sink=name, line=getattr(call, "lineno", 0),
                source=sname, source_line=sline,
                path=[f"{sname}@{sline}", f"{name}()"]))

    # walk in SOURCE order, not BFS order: ast.walk visits nodes
    # level-by-level, so a later re-assignment would be processed
    # before an earlier sink call and report flows that never happen
    stmts = sorted((n for n in ast.walk(tree)
                    if isinstance(n, (ast.Assign, ast.AnnAssign,
                                      ast.AugAssign, ast.Call))),
                   key=lambda n: (getattr(n, "lineno", 0),
                                  getattr(n, "col_offset", 0)))
    for node in stmts:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = node.value
            if value is None:      # bare annotation: x: int
                continue
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            ok, sname, sline = is_source_call(value)
            if ok:
                for tgt in targets:
                    if isinstance(tgt, ast.Name):
                        tainted[tgt.id] = (sname, sline,
                                           [f"{sname}@{sline}",
                                            tgt.id])
            else:
                # propagate: RHS uses a tainted name -> the target is
                # tainted too. But a PLAIN re-assignment from a clean
                # RHS clears the taint — otherwise `x = input();
                # x = "safe"; eval(x)` would be a false positive.
                # (AugAssign merges into the old value, so it keeps
                # whatever taint the target already had.)
                used = tainted_names(value)
                for tgt in targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    if used:
                        base = tainted[used[0]]
                        tainted[tgt.id] = (
                            base[0], base[1],
                            base[2] + [tgt.id])
                    elif not isinstance(node, ast.AugAssign):
                        tainted.pop(tgt.id, None)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if is_sink_name(name):
                for arg in list(node.args) + \
                        [kw.value for kw in node.keywords]:
                    note_sink(node, arg)
    return findings


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

_FUNC_LIKE = re.compile(r"\bdef\b|\blambda\b")

_BRANCHES = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With,
             ast.BoolOp, ast.IfExp, ast.comprehension, ast.Assert)
_BRANCHES += (ast.Match,) if hasattr(ast, "Match") else ()  # py3.10+


def _own_nodes(fn: ast.AST):
    """Walk a function's body WITHOUT descending into nested defs —
    an inner function's branches belong to the inner function's score,
    not to every enclosing one."""
    from collections import deque
    todo = deque(ast.iter_child_nodes(fn))
    while todo:
        node = todo.popleft()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        todo.extend(ast.iter_child_nodes(node))


def cyclomatic(source: str) -> list[Complexity]:
    """Cyclomatic complexity per function: 1 + number of branch nodes."""
    # fast path: no function definitions -> nothing to score, skip parse.
    # (A word-boundary regex: "def " misses `def\tf():`, which is legal.)
    if not _FUNC_LIKE.search(source):
        return []
    # cache: complexity is re-queried for the same files across
    # analyze_file / hotspots / reports
    return [Complexity(c[0], c[1], c[2], c[3], c[4])
            for c in _cyclomatic_cached(source)]


@lru_cache(maxsize=512)
def _cyclomatic_cached(source: str) -> tuple:
    """Cached core: returns plain tuples so the cache stays picklable
    and free of mutable dataclass instances."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            branches = sum(1 for n in _own_nodes(node)
                           if isinstance(n, _BRANCHES))
            end = getattr(node, "end_lineno", node.lineno)
            nargs = len(node.args.args) + len(node.args.kwonlyargs)
            out.append((node.name, node.lineno,
                        1 + branches, end - node.lineno + 1, nargs))
    return tuple(out)


# ---------------------------------------------------------------------------
# Import cycles
# ---------------------------------------------------------------------------

def import_cycles(sources: dict[str, str]) -> list[list[str]]:
    """Detect cycles in the module import graph.

    `sources` maps module name -> source text. Returns a list of cycles,
    each a list of module names forming the loop (deduplicated)."""
    graph: dict[str, set[str]] = {}
    known = set(sources)
    for mod, src in sources.items():
        deps: set[str] = set()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    if root in known:
                        deps.add(root)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in known:
                    deps.add(root)
        graph[mod] = deps

    cycles: list[list[str]] = []
    seen_cycles: set[tuple[str, ...]] = set()
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {m: WHITE for m in graph}

    # genuinely iterative DFS with an explicit stack — the recursive
    # version hit RecursionError on deep import graphs, contradicting
    # this function's contract. Each stack entry is (node,
    # neighbor-iterator); `path` mirrors the stack for cycle slicing.
    for root in sorted(graph):
        if color[root] != WHITE:
            continue
        color[root] = GRAY
        path = [root]
        stack: list[tuple[str, Iterator[str]]] = [
            (root, iter(sorted(graph.get(root, ()))))]
        while stack:
            node, it = stack[-1]
            descended = False
            for nxt in it:
                c = color.get(nxt, BLACK)
                if c == GRAY:
                    # found a cycle: slice the stack from nxt
                    idx = path.index(nxt)
                    cyc = tuple(sorted(path[idx:]))
                    if cyc not in seen_cycles:
                        seen_cycles.add(cyc)
                        cycles.append(list(path[idx:]))
                elif c == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append(
                        (nxt, iter(sorted(graph.get(nxt, ())))))
                    descended = True
                    break  # resume this iterator when the child returns
            if not descended:
                stack.pop()
                path.pop()
                color[node] = BLACK
    return cycles


# ---------------------------------------------------------------------------
# Analyzer facade (event-sourced)
# ---------------------------------------------------------------------------

class StaticAnalyzer:
    """Run the three analyses over files and seal the findings."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.taint = TaintAnalyzer()

    def analyze_file(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        source = p.read_text(errors="replace")
        findings = self.taint.analyze(source)
        comp = cyclomatic(source)
        result = {
            "path": str(p),
            "taint": [f.to_dict() for f in findings],
            "complexity": [c.to_dict() for c in comp],
            "hotspots": [c.name for c in comp if c.complexity >= 10],
        }
        if findings:
            self.log.append("analysis.taint",
                            {"path": str(p),
                             "findings": result["taint"]},
                            actor="analyst")
        if comp:
            self.log.append("analysis.complexity",
                            {"path": str(p),
                             "functions": result["complexity"],
                             "hotspots": result["hotspots"]},
                            actor="analyst")
        return result

    def analyze_tree(self, root: str, glob_filter: str = "*.py",
                     max_files: int = 100) -> dict:
        rp = Path(root).expanduser()
        # rglob, not glob: a tree scan must descend into subdirectories —
        # glob("*.py") only matched the top level and silently missed
        # every nested file's findings
        files = sorted(rp.rglob(glob_filter))[:max_files] \
            if rp.is_dir() else [rp]
        sources: dict[str, str] = {}
        all_taint: list[dict] = []
        all_hotspots: list[dict] = []
        for f in files:
            if not f.is_file():
                continue
            try:
                sources[f.stem] = f.read_text(errors="replace")
            except OSError:
                continue
            r = self.analyze_file(str(f))
            all_taint.extend(r.get("taint", []))
            for h in r.get("hotspots", []):
                all_hotspots.append({"file": str(f), "function": h})
        cycles = import_cycles(sources)
        if cycles:
            self.log.append("analysis.cycles",
                            {"root": str(rp), "cycles": cycles},
                            actor="analyst")
        return {"files": len(files), "taint_findings": all_taint,
                "hotspots": all_hotspots, "import_cycles": cycles}

    def format_report(self, result: dict) -> str:
        lines = ["STATIC ANALYSIS"]
        if result.get("error"):
            return f"STATIC ANALYSIS — {result['error']}"
        lines.append(f"  files scanned: {result.get('files', 1)}")
        tf = result.get("taint_findings") or result.get("taint") or []
        lines.append(f"  taint findings: {len(tf)}")
        for f in tf[:10]:
            lines.append(f"    ⚠ {f.get('source')}@{f.get('source_line')} "
                         f"→ {f.get('sink')}@{f.get('line')}  "
                         f"({' → '.join(f.get('path', []))})")
        hs = result.get("hotspots") or []
        if hs:
            lines.append(f"  complexity hotspots: {len(hs)}")
            for h in hs[:10]:
                if isinstance(h, dict):
                    lines.append(f"    {h.get('file', '')}::"
                                 f"{h.get('function', '')}")
                else:
                    lines.append(f"    {h}")
        cyc = result.get("import_cycles") or []
        if cyc:
            lines.append(f"  import cycles: {len(cyc)}")
            for c in cyc[:5]:
                lines.append("    " + " → ".join(c) + " → " + c[0])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "taint.jsonl")
        an = StaticAnalyzer(log)

        # taint: input() flows into eval() -> finding with the path
        bad = '''
name = input("name? ")
greeting = "hello " + name
eval(greeting)
'''
        findings = an.taint.analyze(bad)
        assert len(findings) == 1, findings
        f = findings[0]
        assert f.sink == "eval" and f.source == "input", f
        assert "name" in f.path and "greeting" in f.path, f.path

        # clean code: no taint finding
        clean = '''
x = 5
y = x + 1
print(y)
'''
        assert an.taint.analyze(clean) == []

        # os.getenv -> subprocess.run is caught
        env = '''
import os, subprocess
cmd = os.getenv("CMD")
subprocess.run(cmd, shell=True)
'''
        fs = an.taint.analyze(env)
        assert any(f.sink == "subprocess.run" for f in fs), fs

        # complexity: a branching function scores high
        comp_src = '''
def complex_fn(x, y, z):
    if x:
        for i in range(y):
            if i > z:
                while z:
                    z -= 1
            elif i == 0:
                pass
    return x or y and z
'''
        comp = cyclomatic(comp_src)
        assert len(comp) == 1
        assert comp[0].complexity >= 5, comp[0]
        assert comp[0].name == "complex_fn"

        # trivial function has complexity 1
        assert cyclomatic("def f():\n    return 1\n")[0].complexity == 1

        # import cycles: a -> b -> a is a cycle
        cyc = import_cycles({
            "a": "import b\n",
            "b": "import a\n",
            "c": "import os\n",
        })
        assert len(cyc) == 1 and set(cyc[0]) == {"a", "b"}, cyc

        # no cycle when the graph is a DAG
        assert import_cycles({"a": "import b\n", "b": "import os\n"}) == []

        # analyze_file end to end + event sealing
        p = Path(td) / "vuln.py"
        p.write_text(bad + "\ndef handler(req):\n    if req:\n"
                           "        return eval(req)\n    return 0\n")
        result = an.analyze_file(str(p))
        assert result["taint"] and result["complexity"]
        report = an.format_report(result)
        assert "taint findings:" in report

        # analyze_tree over a directory
        (Path(td) / "m1.py").write_text("import m2\n")
        (Path(td) / "m2.py").write_text("import m1\n")
        tree = an.analyze_tree(td)
        assert tree["files"] >= 2
        assert tree["import_cycles"], tree

        # findings are sealed in the log
        evs = fold(log).analysis_events
        types = {e["type"] for e in evs}
        assert "analysis.taint" in types and "analysis.complexity" in types

    print("TAINT SELF-TEST PASS")

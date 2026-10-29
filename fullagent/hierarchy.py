"""hierarchy — Instruction Hierarchy Engine (v5 apex).

A deterministic, zero-I/O engine that models the full precedence graph
of all instruction sources in the system and resolves conflicts via
explicit priority + specificity rules. Replaces ad-hoc "system prompt
wins" with a formal, auditable hierarchy.

Sources (highest to lowest precedence):
  1. Developer Configuration  (project.txt, protocol_wrap) — immutable
  2. Runtime Policy           (config.json extra, /model, /effort) — session
  3. Session Directives       (goal contracts, output_style, AGENTS.md)
  4. Turn Context             (web/memory/compacted sections, user msg)
  5. Tool Output              (stdout/stderr, fetch results) — data only
  6. Model Defaults           (provider defaults, tokenizer behaviour)

Each directive carries: source_level, specificity (exact > prefix > glob),
scope (session|turn|tool), and a stable rule_id. The resolver answers:
"Given a proposed action, which rule governs, and what does it require?"

Pure Python, no I/O, thread-safe, fully testable. Integrates with
enforce.py (rule IDs), sentinel.py (injection points), attest.py (audit).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "SourceLevel", "Directive", "RuleGraph", "HierarchyResolver",
    "precedence_of", "specificity_of", "compile_directives",
    "directive_id",
]

# ── 1. Source precedence (immutable ordering) ───────────────────────────

class SourceLevel:
    """Immutable precedence tiers. Lower value = higher authority."""
    DEVELOPER_CONFIG = 0   # project.txt / protocol_wrap
    RUNTIME_POLICY   = 10  # config.json extra, /model, /effort
    SESSION_DIR      = 20  # goal, output_style, AGENTS.md
    TURN_CONTEXT     = 30  # web, memory, compacted, user message
    TOOL_OUTPUT      = 40  # stdout, stderr, fetch results
    MODEL_DEFAULTS   = 50  # provider defaults

    _NAMES = {
        DEVELOPER_CONFIG: "developer_config",
        RUNTIME_POLICY:   "runtime_policy",
        SESSION_DIR:      "session_directive",
        TURN_CONTEXT:     "turn_context",
        TOOL_OUTPUT:      "tool_output",
        MODEL_DEFAULTS:   "model_defaults",
    }

    @classmethod
    def name(cls, level: int) -> str:
        return cls._NAMES.get(level, f"unknown({level})")

    @classmethod
    def all(cls) -> list[int]:
        return [cls.DEVELOPER_CONFIG, cls.RUNTIME_POLICY,
                cls.SESSION_DIR, cls.TURN_CONTEXT,
                cls.TOOL_OUTPUT, cls.MODEL_DEFAULTS]


# ── 2. Directive representation ─────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Directive:
    """A single binding rule extracted from any instruction source."""
    rule_id: str                 # stable: "R{source:02d}{seq:03d}"
    source_level: int            # SourceLevel constant
    text: str                    # original directive text (verbatim)
    pattern: str                 # applicability pattern (exact/prefix/glob)
    scope: str                   # "session" | "turn" | "tool"
    specificity: int             # 3=exact, 2=prefix, 1=glob, 0=wildcard
    metadata: dict = field(default_factory=dict)

    def applies_to(self, action: str) -> bool:
        """Fast applicability check — no regex, pure string logic."""
        if self.pattern == "*" or self.pattern == action:
            return True
        if self.pattern.endswith("*") and action.startswith(self.pattern[:-1]):
            return True
        if self.pattern.startswith("*") and action.endswith(self.pattern[1:]):
            return True
        return False

    def __lt__(self, other: "Directive") -> bool:
        # sort key: higher precedence (lower level) first, then specificity
        if self.source_level != other.source_level:
            return self.source_level < other.source_level
        return self.specificity > other.specificity


# ── 3. Rule graph — dependency + conflict edges ─────────────────────────

@dataclass
class RuleGraph:
    """Directed graph of rule relationships: depends_on, conflicts_with,
    refines, overrides. Built once per session from the directive set."""
    nodes: dict[str, Directive] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)  # rule_id -> {dep_ids}
    conflict: dict[str, set[str]] = field(default_factory=dict)

    def add(self, d: Directive) -> None:
        self.nodes[d.rule_id] = d

    def link(self, from_id: str, to_id: str, kind: str = "depends") -> None:
        """link(a, b, 'depends') == 'a depends on b' → b emits first.
        'refines' behaves the same (b is the more general rule).
        'conflicts' is symmetric."""
        if kind in ("depends", "refines"):
            self.edges.setdefault(to_id, set()).add(from_id)
        elif kind == "conflicts":
            self.conflict.setdefault(from_id, set()).add(to_id)
            self.conflict.setdefault(to_id, set()).add(from_id)

    def depends_on(self, rule_id: str) -> set[str]:
        return self.edges.get(rule_id, set())

    def conflicts_with(self, rule_id: str) -> set[str]:
        return self.conflict.get(rule_id, set())

    def topological_order(self) -> list[str]:
        """Kahn's algorithm with precedence tie-breaking — among ready
        nodes, higher-precedence (lower level) rules emit first."""
        indeg: dict[str, int] = {n: 0 for n in self.nodes}
        for u in self.edges:
            for v in self.edges[u]:
                indeg[v] = indeg.get(v, 0) + 1
        ready = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while ready:
            ready.sort(key=lambda r: (self.nodes[r].source_level, r))
            u = ready.pop(0)
            order.append(u)
            for v in self.edges.get(u, ()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
        if len(order) != len(self.nodes):
            # cycle — break by precedence (lower level first)
            remaining = [n for n in self.nodes if n not in order]
            remaining.sort(key=lambda r: self.nodes[r].source_level)
            order.extend(remaining)
        return order


# ── 4. Compiler: raw text → directives + graph ──────────────────────────

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.M)
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]\s+(.+)$", re.M)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$", re.M)
_IMPERATIVE_RE = re.compile(r"\b(must|shall|always|never|do not|don't)\b", re.I)
_CONSTRAINT_RE = re.compile(r"(format|tone|length|style|language|output)\b", re.I)


def directive_id(text: str, source_level: int, seq: int) -> str:
    """Stable short id: level + seq + content hash prefix."""
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
    return f"R{source_level:02d}{seq:03d}-{h}"


def _infer_pattern(text: str) -> tuple[str, int]:
    """Infer applicability pattern and specificity from directive text.
    Returns (pattern, specificity) where specificity: 3=exact, 2=prefix,
    1=glob, 0=wildcard."""
    t = text.strip().lower()
    # Explicit "for <tool>" or "when <action>" — but a gerund ("when
    # editing") is a condition, not an action name, so treat it broad.
    m = re.search(r"\b(for|when)\s+([a-z_][a-z0-9_]*)", t)
    if m and not m.group(2).endswith("ing"):
        return m.group(2), 3
    if m:  # gerund condition → still a scoped hint, not tool-exact
        return "*", 2
    # "always ...", "never ..." → wildcard scope
    if _IMPERATIVE_RE.search(t):
        return "*", 0
    # Constraint words → session scope wildcard
    if _CONSTRAINT_RE.search(t):
        return "*", 1
    return "*", 0


def _infer_scope(text: str) -> str:
    t = text.strip().lower()
    if any(k in t for k in ("tool", "shell", "edit", "write", "bash")):
        return "tool"
    if any(k in t for k in ("this turn", "next reply", "now")):
        return "turn"
    return "session"


def compile_directives(
    source_text: str,
    source_level: int,
    base_seq: int = 0
) -> tuple[list[Directive], int]:
    """Extract directives from a raw instruction document.

    Heuristics (in order):
      1. Markdown headings (level 1-4) → high-specificity rules
      2. Numbered lists (1., 1.1., etc.) → exact/prefix rules
      3. Bullet points → rules
      4. Imperative sentences ("must", "never") → broad rules
      5. Constraint sentences (format, tone, length) → session rules
    Returns (directives, next_seq).
    """
    dirs: list[Directive] = []
    seq = base_seq

    # 1. Headings
    for m in _HEADING_RE.finditer(source_text):
        seq += 1
        rule_text = m.group(2).strip()
        if len(rule_text) < 4:
            continue
        pattern, spec = _infer_pattern(rule_text)
        dirs.append(Directive(
            rule_id=directive_id(rule_text, source_level, seq),
            source_level=source_level,
            text=rule_text,
            pattern=pattern,
            scope=_infer_scope(rule_text),
            specificity=spec,
            metadata={"type": "heading", "level": len(m.group(1))}
        ))

    # 2. Numbered items
    for m in _NUMBERED_RE.finditer(source_text):
        seq += 1
        num, rule_text = m.group(1), m.group(2).strip()
        if len(rule_text) < 6:
            continue
        pattern, spec = _infer_pattern(rule_text)
        dirs.append(Directive(
            rule_id=directive_id(rule_text, source_level, seq),
            source_level=source_level,
            text=rule_text,
            pattern=pattern,
            scope=_infer_scope(rule_text),
            specificity=spec + 1,  # numbered = more specific
            metadata={"type": "numbered", "number": num}
        ))

    # 3. Bullets
    for m in _BULLET_RE.finditer(source_text):
        seq += 1
        rule_text = m.group(1).strip()
        if len(rule_text) < 6:
            continue
        pattern, spec = _infer_pattern(rule_text)
        dirs.append(Directive(
            rule_id=directive_id(rule_text, source_level, seq),
            source_level=source_level,
            text=rule_text,
            pattern=pattern,
            scope=_infer_scope(rule_text),
            specificity=spec,
            metadata={"type": "bullet"}
        ))

    # 4. Imperative sentences (fallback for dense prose)
    # Split on sentences, keep only imperative ones
    sentences = re.split(r"(?<=[.!?।])\s+", source_text)
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 15 or len(s) > 400:
            continue
        if not _IMPERATIVE_RE.search(s):
            continue
        seq += 1
        pattern, spec = _infer_pattern(s)
        dirs.append(Directive(
            rule_id=directive_id(s, source_level, seq),
            source_level=source_level,
            text=s,
            pattern=pattern,
            scope=_infer_scope(s),
            specificity=max(1, spec),  # at least prefix-level
            metadata={"type": "imperative"}
        ))

    return dirs, seq


# ── 5. Resolver: action → governing directive(s) ────────────────────────

class HierarchyResolver:
    """Given a compiled directive set + graph, resolve the governing
    rule(s) for any proposed action. Pure function, no state mutation."""

    def __init__(
        self,
        directives: list[Directive],
        graph: RuleGraph | None = None,
        tiebreaker: Callable[[Directive, Directive], Directive] | None = None
    ):
        self._dirs = sorted(directives)
        self._graph = graph or RuleGraph()
        for d in self._dirs:
            self._graph.add(d)
        self._tiebreaker = tiebreaker or (lambda a, b: a)

    def governing(self, action: str) -> list[Directive]:
        """Return ALL directives that apply to action, sorted by precedence."""
        return [d for d in self._dirs if d.applies_to(action)]

    def decide(self, action: str) -> Directive | None:
        """Single winning directive for action (highest precedence +
        specificity, tiebreaker if needed)."""
        applicable = self.governing(action)
        if not applicable:
            return None
        best = applicable[0]
        for d in applicable[1:]:
            if d.source_level < best.source_level:
                best = d
            elif d.source_level == best.source_level:
                if d.specificity > best.specificity:
                    best = d
                elif d.specificity == best.specificity:
                    best = self._tiebreaker(best, d)
        return best

    def conflicts(self, action: str) -> list[tuple[Directive, Directive]]:
        """Return pairs of conflicting directives that both apply."""
        pairs: list[tuple[Directive, Directive]] = []
        applicable = self.governing(action)
        for i, a in enumerate(applicable):
            for b in applicable[i+1:]:
                if b.rule_id in self._graph.conflicts_with(a.rule_id):
                    pairs.append((a, b))
        return pairs

    def explain(self, action: str) -> dict:
        """Full decision trace for audit logging."""
        gov = self.governing(action)
        winner = self.decide(action)
        return {
            "action": action,
            "candidates": [{"rule_id": d.rule_id, "source": SourceLevel.name(d.source_level),
                           "specificity": d.specificity, "pattern": d.pattern}
                          for d in gov],
            "winner": {"rule_id": winner.rule_id, "text": winner.text[:200]} if winner else None,
            "conflicts": [{"a": a.rule_id, "b": b.rule_id}
                          for a, b in self.conflicts(action)]
        }


# ── 6. Convenience: precedence helpers ──────────────────────────────────

def precedence_of(source_name: str) -> int:
    mapping = {
        "developer_config": SourceLevel.DEVELOPER_CONFIG,
        "runtime_policy": SourceLevel.RUNTIME_POLICY,
        "session_directive": SourceLevel.SESSION_DIR,
        "turn_context": SourceLevel.TURN_CONTEXT,
        "tool_output": SourceLevel.TOOL_OUTPUT,
        "model_defaults": SourceLevel.MODEL_DEFAULTS,
    }
    return mapping.get(source_name, 999)


def specificity_of(pattern: str) -> int:
    if pattern == "*":
        return 0
    if pattern.endswith("*") or pattern.startswith("*"):
        return 1
    return 3


# ── 7. Self-test ────────────────────────────────────────────────────────

def _self_test() -> int:
    doc = """# Agent Rules
## Communication
1. Always answer in Hindi.
2. Never use bullet lists unless asked.
- Keep answers under 120 words.

## Tools
You must cite file paths when editing code.
Never run shell commands without approval.
"""
    dirs, _ = compile_directives(doc, SourceLevel.DEVELOPER_CONFIG)
    assert len(dirs) >= 5, f"expected >=5 directives, got {len(dirs)}"
    # heading rule
    assert any("Hindi" in d.text for d in dirs)
    # numbered rule
    assert any("bullet lists" in d.text for d in dirs)
    # bullet rule
    assert any("120 words" in d.text for d in dirs)
    # imperative rule
    assert any("cite file paths" in d.text for d in dirs)

    resolver = HierarchyResolver(dirs)
    # exact match wins
    win = resolver.decide("edit_file")
    assert win is not None and "cite file paths" in win.text
    # wildcard fallback
    win2 = resolver.decide("random_action")
    assert win2 is not None

    # conflict detection
    g = RuleGraph()
    d1 = Directive("R001", SourceLevel.DEVELOPER_CONFIG, "A", "edit_file", "tool", 3)
    d2 = Directive("R002", SourceLevel.SESSION_DIR, "B", "edit_file", "tool", 3)
    g.add(d1); g.add(d2); g.link("R001", "R002", "conflicts")
    r = HierarchyResolver([d1, d2], g)
    assert len(r.conflicts("edit_file")) == 1

    # topological order respects precedence
    d3 = Directive("R010", SourceLevel.DEVELOPER_CONFIG, "C", "*", "session", 0)
    d4 = Directive("R020", SourceLevel.SESSION_DIR, "D", "*", "session", 0)
    g2 = RuleGraph(); g2.add(d3); g2.add(d4); g2.link("R020", "R010", "depends")
    order = g2.topological_order()
    assert order.index("R010") < order.index("R020"), "precedence must win over depends"

    # cycle handling
    g3 = RuleGraph()
    d5 = Directive("R100", SourceLevel.DEVELOPER_CONFIG, "E", "*", "session", 0)
    d6 = Directive("R101", SourceLevel.DEVELOPER_CONFIG, "F", "*", "session", 0)
    g3.add(d5); g3.add(d6); g3.link("R100", "R101", "depends"); g3.link("R101", "R100", "depends")
    assert len(g3.topological_order()) == 2

    print("HIERARCHY SELF-TEST PASS (compile, resolve, graph, cycles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
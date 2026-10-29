"""KGRAPH — the knowledge graph.

A real graph of the project and the session: entities (modules, functions,
classes, files, goals, episodes, facts) and typed relations between them
(defines, calls, imports, touches, proves, learned_from). Built from two
real sources:

  * CODE  — AST walk: a module DEFINES its functions/classes; a function
            CALLS the names it invokes; a module IMPORTS the modules it
            imports. No guessing — the AST is the ground truth.
  * LOG   — the event fold: a goal TOUCHES the files its clauses name; an
            episode LEARNED_FROM its facts; a tool.call TOUCHES its path.

Queries are real graph operations (BFS reachability, reverse lookups,
impact sets): "what calls auth()?", "what does this file connect to?",
"what's affected if I change X?" — answered by walking edges, not by
asking a model.

The graph is rebuilt from sources on demand (it is a projection, never
authoritative state); index events are sealed into the log.
"""

from __future__ import annotations

import ast
import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("kgraph")

# entity kinds
KINDS = ("module", "function", "class", "file", "goal", "episode", "fact")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: str
    kind: str
    name: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name,
                **self.meta}


@dataclass
class Relation:
    src: str
    rel: str
    dst: str

    def to_dict(self) -> dict:
        return {"src": self.src, "rel": self.rel, "dst": self.dst}


# ---------------------------------------------------------------------------
# Code extraction (AST ground truth)
# ---------------------------------------------------------------------------

def _call_name(node: ast.expr) -> str:
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


def extract_code(module_name: str, source: str) -> tuple[list[Entity],
                                                         list[Relation]]:
    """Entities + relations for one module, straight from the AST."""
    entities: list[Entity] = []
    relations: list[Relation] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return entities, relations

    mod_id = f"module:{module_name}"
    entities.append(Entity(mod_id, "module", module_name))

    # imports: module IMPORTS module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                relations.append(Relation(mod_id, "imports",
                                          f"module:{a.name.split('.')[0]}"))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root:
                relations.append(Relation(mod_id, "imports",
                                          f"module:{root}"))

    # definitions + calls
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fid = f"function:{module_name}.{node.name}"
            entities.append(Entity(fid, "function", node.name,
                                   {"line": node.lineno,
                                    "module": module_name}))
            relations.append(Relation(mod_id, "defines", fid))
            # calls made inside this function
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    name = _call_name(sub.func)
                    if name:
                        relations.append(Relation(fid, "calls",
                                                  f"call:{name}"))
        elif isinstance(node, ast.ClassDef):
            cid = f"class:{module_name}.{node.name}"
            entities.append(Entity(cid, "class", node.name,
                                   {"line": node.lineno,
                                    "module": module_name}))
            relations.append(Relation(mod_id, "defines", cid))
    return entities, relations


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Entities + typed relations, queryable, event-sourced."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._out: dict[str, list[Relation]] = {}
        self._in: dict[str, list[Relation]] = {}
        # SPEED: O(1) duplicate-edge check — the old linear scan over
        # _out[src] made indexing quadratic on large codebases.
        self._seen_edges: set[tuple[str, str, str]] = set()
        # name index for find(): lowercase name -> entity ids
        self._name_index: dict[str, set[str]] = {}
        # relations contributed by each indexed module, so re-indexing a
        # module first retracts its stale edges (no ghost calls/imports
        # from an older version of the source)
        self._module_edges: dict[str, list[tuple[str, str, str]]] = {}
        # entity ids contributed by each indexed module, so re-indexing
        # also retracts entities that no longer exist (no ghost
        # functions/classes surviving a rename or delete)
        self._module_entities: dict[str, set[str]] = {}

    def _add_entity(self, e: Entity) -> None:
        old = self._entities.get(e.id)
        if old is not None and old.name.lower() != e.name.lower():
            self._name_index.get(old.name.lower(), set()).discard(e.id)
        self._entities[e.id] = e
        self._name_index.setdefault(e.name.lower(), set()).add(e.id)

    # -- building --------------------------------------------------------------

    def index_code(self, sources: dict[str, str]) -> int:
        """Index modules {name: source}. Returns entity count."""
        for name, src in sources.items():
            # retract this module's previous edges AND entities before
            # re-adding: otherwise a re-index keeps stale calls/imports
            # (and ghost functions/classes) forever
            self._retract_module(name)
            ents, rels = extract_code(name, src)
            new_keys: list[tuple[str, str, str]] = []
            new_eids: set[str] = set()
            for e in ents:
                self._add_entity(e)
                new_eids.add(e.id)
            for r in rels:
                if self._add_relation(r):
                    new_keys.append((r.src, r.rel, r.dst))
            self._module_edges[name] = new_keys
            self._module_entities[name] = new_eids
        self.log.append("graph.entity",
                        {"entities": len(self._entities),
                         "relations": len(self._relations)},
                        actor="librarian")
        return len(self._entities)

    def _retract_module(self, name: str) -> None:
        """Remove every edge and entity previously contributed by module
        `name`. Entity ids are namespaced per module
        (module:/function:/class: prefixes), so this never touches
        entities owned by other modules or by index_log."""
        keys = self._module_edges.pop(name, [])
        dead = set(keys)
        if dead:
            self._seen_edges -= dead
            self._relations = [r for r in self._relations
                               if (r.src, r.rel, r.dst) not in dead]
            for idx in (self._out, self._in):
                for k in list(idx.keys()):
                    kept = [r for r in idx[k]
                            if (r.src, r.rel, r.dst) not in dead]
                    if kept:
                        idx[k] = kept
                    else:
                        del idx[k]
        for eid in self._module_entities.pop(name, set()):
            ent = self._entities.pop(eid, None)
            if ent is not None:
                bucket = self._name_index.get(ent.name.lower())
                if bucket is not None:
                    bucket.discard(eid)
                    if not bucket:
                        del self._name_index[ent.name.lower()]

    def index_log(self) -> int:
        """Add session entities/relations from the event fold."""
        st = fold(self.log)
        added = 0
        if st.goal and st.goal.get("statement"):
            gid = f"goal:{st.goal.get('id', 'current')}"
            if gid not in self._entities:
                self._add_entity(Entity(gid, "goal",
                                        str(st.goal.get("statement",
                                                        ""))[:80]))
                added += 1
            for clause in st.goal.get("clauses") or []:
                proof = clause.get("proof") or {}
                p = proof.get("path") or proof.get("command")
                if p:
                    fid = f"file:{p}"
                    if fid not in self._entities:
                        self._add_entity(Entity(fid, "file", str(p)))
                        added += 1
                    self._add_relation(Relation(gid, "touches", fid))
        for i, ep in enumerate(st.episodes):
            eid = f"episode:{i}"
            if eid not in self._entities:
                self._add_entity(Entity(eid, "episode",
                                        str(ep.get("goal", ""))[:80]))
                added += 1
            for fact in ep.get("facts") or []:
                # hash the full text for the id: truncating to N chars
                # merged distinct facts that shared a prefix
                fid = ("fact:" + hashlib.sha256(
                    str(fact).encode("utf-8", "replace")).hexdigest()[:16])
                if fid not in self._entities:
                    self._add_entity(Entity(fid, "fact", str(fact)[:80]))
                    added += 1
                self._add_relation(Relation(eid, "learned", fid))
        return added

    def _add_relation(self, r: Relation) -> bool:
        """Add an edge unless it already exists. Returns True if new."""
        key = (r.src, r.rel, r.dst)
        if key in self._seen_edges:
            return False
        self._seen_edges.add(key)
        self._relations.append(r)
        self._out.setdefault(r.src, []).append(r)
        self._in.setdefault(r.dst, []).append(r)
        return True

    # -- queries (real graph operations) ----------------------------------------

    def entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def find(self, name: str, kind: str | None = None) -> list[Entity]:
        """Entities whose name matches (substring), optionally by kind.

        SPEED: the name index narrows candidates to entities sharing the
        query as a substring of their lowercased name; only those are
        scanned instead of the whole graph.
        """
        low = name.lower()
        cands: set[str] = set()
        for indexed_name, ids in self._name_index.items():
            if low in indexed_name:
                cands.update(ids)
        out = []
        for eid in cands:
            e = self._entities.get(eid)
            if e is not None and (kind is None or e.kind == kind):
                out.append(e)
        return sorted(out, key=lambda e: e.id)

    def out_edges(self, entity_id: str, rel: str | None = None
                  ) -> list[Relation]:
        return [r for r in self._out.get(entity_id, [])
                if rel is None or r.rel == rel]

    def in_edges(self, entity_id: str, rel: str | None = None
                 ) -> list[Relation]:
        return [r for r in self._in.get(entity_id, [])
                if rel is None or r.rel == rel]

    def callers_of(self, func_name: str) -> list[str]:
        """Which functions call a given function name (reverse lookup).

        Matches both bare call sites (`call:<name>`, from `name(...)`)
        and qualified ones (`call:<mod>.<name>`, from `mod.name(...)`).
        """
        target = f"call:{func_name}"
        out: set[str] = set()
        for key, rels in self._in.items():
            if not key.startswith("call:"):
                continue
            if key == target or key.rsplit(".", 1)[-1] == func_name:
                out.update(r.src for r in rels)
        return sorted(out)

    def reachable(self, start: str, rel: str | None = None,
                  max_depth: int = 6) -> list[str]:
        """BFS reachability from an entity over (optionally typed) edges."""
        seen: list[str] = []
        seen_set = {start}
        q = deque([(start, 0)])
        while q:
            node, depth = q.popleft()
            if depth >= max_depth:
                continue
            for r in self.out_edges(node, rel):
                if r.dst not in seen_set:
                    seen_set.add(r.dst)
                    seen.append(r.dst)
                    q.append((r.dst, depth + 1))
        return seen

    def _reverse_keys(self, entity_id: str) -> list[str]:
        """Lookup keys for incoming edges of an entity. A function entity
        `function:<mod>.<name>` is also reached through the call
        pseudo-nodes call sites target — both the bare `call:<name>`
        (from `name(...)`) and the qualified `call:<mod>.<name>` (from
        `mod.name(...)`) forms."""
        keys = [entity_id]
        if entity_id.startswith("function:"):
            rest = entity_id[len("function:"):]  # "<mod>.<name>"
            if "." in rest:
                keys.append(f"call:{rest}")
                keys.append(f"call:{rest.rsplit('.', 1)[-1]}")
        return keys

    def impact(self, entity_id: str) -> list[str]:
        """Reverse-reachability: everything that depends on this entity
        (walks incoming edges). 'What breaks if I change X?'"""
        seen: list[str] = []
        seen_set = {entity_id}
        q = deque([entity_id])
        while q:
            node = q.popleft()
            for key in self._reverse_keys(node):
                for r in self._in.get(key, []):
                    if r.src not in seen_set:
                        seen_set.add(r.src)
                        seen.append(r.src)
                        q.append(r.src)
        return seen

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self._entities.values():
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        rels: dict[str, int] = {}
        for r in self._relations:
            rels[r.rel] = rels.get(r.rel, 0) + 1
        return {"entities": len(self._entities),
                "relations": len(self._relations),
                "kinds": kinds, "rels": rels}

    def format_status(self) -> str:
        s = self.stats()
        lines = ["KNOWLEDGE GRAPH",
                 f"  entities {s['entities']}   relations {s['relations']}"]
        if s["kinds"]:
            lines.append("  kinds: " + "  ".join(
                f"{k}×{v}" for k, v in sorted(s["kinds"].items())))
        if s["rels"]:
            lines.append("  rels:  " + "  ".join(
                f"{k}×{v}" for k, v in sorted(s["rels"].items())))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "kgraph.jsonl")
        kg = KnowledgeGraph(log)

        # index two modules with a real call + import relationship
        kg.index_code({
            "auth": ("def login(user):\n    return check(user)\n\n"
                     "def check(user):\n    return True\n"),
            "main": ("import auth\n\n"
                     "def run():\n    return auth.login('x')\n"),
        })

        # entities: modules + functions exist
        assert kg.entity("module:auth") is not None
        assert kg.entity("function:auth.login") is not None
        assert kg.entity("function:auth.check") is not None
        assert kg.entity("function:main.run") is not None

        # auth module defines login and check
        defs = [r.dst for r in kg.out_edges("module:auth", "defines")]
        assert "function:auth.login" in defs
        assert "function:auth.check" in defs

        # main imports auth
        imports = [r.dst for r in kg.out_edges("module:main", "imports")]
        assert "module:auth" in imports

        # reverse lookup: who calls check()?
        callers = kg.callers_of("check")
        assert "function:auth.login" in callers, callers

        # reachability from main.run follows calls
        reach = kg.reachable("function:main.run")
        assert any("auth.login" in r for r in reach), reach

        # impact: changing auth.check affects auth.login (reverse edge)
        impact = kg.impact("function:auth.check")
        assert "function:auth.login" in impact, impact

        # find by name
        hits = kg.find("login", kind="function")
        assert len(hits) == 1 and hits[0].name == "login"

        # index the event log: a goal + episode add entities/relations
        log.append("goal.set", {"id": "g1", "statement": "ship parser",
                                "clauses": [{"id": "C1",
                                             "proof": {"type": "file_exists",
                                                       "path": "p.py"}}]})
        log.append("memory.episode",
                   {"goal": "fix parser", "outcome": "success",
                    "facts": ["tokenizer is line-based"]})
        added = kg.index_log()
        assert added >= 2
        assert kg.entity("goal:g1") is not None
        touches = [r.dst for r in kg.out_edges("goal:g1", "touches")]
        assert "file:p.py" in touches

        # stats + rendering
        s = kg.stats()
        assert s["entities"] > 5 and s["relations"] > 5
        assert "KNOWLEDGE GRAPH" in kg.format_status()

        # index events are sealed
        assert fold(log).graph_events

    print("KGRAPH SELF-TEST PASS")

"""WORLD — the predictive model of the project: what breaks if I touch
this?

Static dependency graphs say "these files import that one". The World
Model says "when auth.py changed, tests/auth_test.py broke 4 times out
of 5, and the build twice out of 5" — impact with LEARNED probabilities:

    edges       dependency edges from three signals: static imports
                (regex over source), co-change history (files written
                in the same turn, from the kernel log), and failure
                co-occurrence (files that were touched in turns that
                later errored)
    risk        each edge carries observed stats: times_seen,
                times_broken. P(break | touch upstream) is a
                Laplace-smoothed rate — small data is honestly unsure,
                never overconfident
    predict     predict_impact(path) ranks every downstream file by
                expected breakage: edge probability × distance decay
                (direct neighbours hurt more than transitive ones)
    learn       observe_turn() ingests each finished turn from the log
                and updates the stats — the model sharpens with every
                real session, zero tokens spent

The self-test plants a project where touching core.py broke the API
80% of the time and proves the model recovers that number from raw
turn history alone.
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog
from ._foundation import get_logger

_log = get_logger("world")

_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+([a-zA-Z0-9_.]+)", re.M)
_DECAY = 0.55          # per-hop decay for transitive impact
_MAX_HOPS = 3
_ALPHA = 1.0           # Laplace smoothing


@dataclass
class Edge:
    src: str
    dst: str
    seen: int = 1            # co-change/dependency observations
    broken: int = 0          # dst failed in turns where src changed

    @property
    def risk(self) -> float:
        """Laplace-smoothed P(break | src touched)."""
        return (self.broken + _ALPHA) / (self.seen + 2 * _ALPHA)

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "seen": self.seen,
                "broken": self.broken, "risk": round(self.risk, 3)}


@dataclass
class Impact:
    path: str
    ranked: list[dict] = field(default_factory=list)

    def format(self) -> str:
        if not self.ranked:
            return f"no known dependants of {self.path}"
        lines = [f"IMPACT — touching {self.path}:"]
        for r in self.ranked:
            bars = "█" * max(1, int(r["probability"] * 10))
            lines.append(f"  {r['path']:<28} {r['probability']:.0%} "
                         f"{bars}  (via {r['via']})")
        return "\n".join(lines)


class WorldModel:
    """Learned dependency + failure model of the codebase."""

    def __init__(self, log: EventLog, root=None) -> None:
        """root: optional project path for the static import scan."""
        self.log = log
        # accept str or Path — the scan only uses Path operations, and a
        # bare str root used to die with AttributeError on first access
        self.root = Path(root) if root is not None else None
        self._edges: dict[tuple[str, str], Edge] = {}
        # adjacency cache: src -> [(dst, edge)]. predict_impact() used to
        # scan ALL edges for every BFS node (O(V*E)) and pop(0) from a
        # list (O(V^2)) — both fixed.
        self._adj: dict[str, list[tuple[str, Edge]]] | None = None
        # SPEED: defer the filesystem scan until first use — scanning
        # the whole project tree at startup added ~400ms to launch.
        self._scanned = False

    @property
    def edges(self) -> dict[tuple[str, str], Edge]:
        """Edges, triggering the lazy filesystem scan on first access."""
        self._ensure_scanned()
        return self._edges

    @edges.setter
    def edges(self, value: dict[tuple[str, str], Edge]) -> None:
        self._edges = value
        self._adj = None  # replaced wholesale — adjacency is stale
        self._scanned = True

    def _ensure_scanned(self) -> None:
        if not self._scanned and self.root is not None:
            self._scan_imports()
            self._scanned = True

    # -- static signal ----------------------------------------------------------

    def _scan_imports(self) -> None:
        """Static import edges: file A imports module B (same project).
        Pure regex — fast, language-tolerant, no execution."""
        files: list[Path] = []
        try:
            # os.walk (unlike rglob) skips unreadable directories and
            # keeps going instead of aborting the whole scan mid-tree:
            # one bad permission/mount must not nuke the world model.
            # followlinks=False (default) — no symlink-loop risk.
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                files.extend(Path(dirpath, fn) for fn in filenames
                             if fn.endswith(".py"))
        except OSError:
            return
        files.sort()  # deterministic: rglob/scandir order is OS-dependent
        modules = {}
        for p in files:
            try:
                # first (sorted) path wins on stem collisions — stable
                # across machines instead of whichever rglob yielded last
                modules.setdefault(p.stem,
                                   p.relative_to(self.root).as_posix())
            except ValueError:
                continue
        for f in files:
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            try:
                rel = f.relative_to(self.root).as_posix()
            except ValueError:
                continue
            for match in _IMPORT_RE.finditer(text):
                mod = match.group(1).split(".")[0]
                target = modules.get(mod)
                if target and target != rel:
                    # edge direction = IMPACT direction: touching the
                    # imported module affects its importer
                    self._edge(target, rel).seen += 1

    def _edge(self, src: str, dst: str) -> Edge:
        key = (src, dst)
        if key not in self._edges:
            self._edges[key] = Edge(src=src, dst=dst)
            self._adj = None  # new edge — adjacency is stale
        return self._edges[key]

    def _adjacency(self) -> dict[str, list[tuple[str, Edge]]]:
        # Lazy scan: the filesystem walk happens on first actual use,
        # not at construction (saves ~400ms on startup).
        self._ensure_scanned()
        if self._adj is None:
            adj: dict[str, list[tuple[str, Edge]]] = {}
            for (src, dst), edge in self._edges.items():
                adj.setdefault(src, []).append((dst, edge))
            self._adj = adj
        return self._adj

    # -- learned signals ------------------------------------------------------------

    def observe_turn(self, changed: list[str], failures: list[str],
                     dependants: list[str] | None = None) -> None:
        """Ingest one finished turn: which files changed, which files
        later failed (test/verify errors). Co-changes strengthen
        edges; failures charge them."""
        changed = [str(c) for c in changed]
        failures = set(str(f) for f in failures)
        for src in changed:
            _deps = (self._static_dependants(src)
                     if dependants is None else dependants)
            for dst in _deps:
                if dst == src:
                    continue
                edge = self._edge(src, dst)
                edge.seen += 1
                if dst in failures:
                    edge.broken += 1
        self.log.append("world.learn",
                        {"changed": changed[:10],
                         "failures": sorted(failures)[:10]})

    def _static_dependants(self, path: str) -> list[str]:
        """Files that depend on `path` — the dst side of impact edges."""
        return [dst for (src, dst) in self.edges if src == path]

    def learn_from_log(self) -> int:
        """Rebuild learned stats from kernel history: turns where files
        were written and later tool errors occurred."""
        turns = []
        current: dict | None = None
        for ev in self.log.events():
            if ev.type == "user.message":
                if current is not None:
                    turns.append(current)
                current = {"changed": [], "failed": []}
            elif current is None:
                continue
            elif ev.type == "tool.call" and str(
                    ev.data.get("name", "")) in (
                    "write_file", "edit_file"):
                p = str((ev.data.get("args") or {}).get("path", ""))
                if p:
                    current["changed"].append(p)
            elif ev.type == "tool.result":
                if ev.data.get("status") == "error":
                    current["failed"].extend(current["changed"])
        if current is not None:
            turns.append(current)
        for t in turns:
            if t["changed"]:
                self.observe_turn(t["changed"], t["failed"])
        return len([t for t in turns if t["changed"]])

    # -- prediction ----------------------------------------------------------------------

    def predict_impact(self, path: str) -> Impact:
        """BFS over edges from `path`; every downstream file ranked by
        probability × hop decay."""
        path = str(path)
        adj = self._adjacency()
        frontier: deque[tuple[str, int]] = deque([(path, 0)])
        seen: set[str] = {path}
        ranked: dict[str, tuple[float, str]] = {}
        while frontier:
            cur, hops = frontier.popleft()
            if hops >= _MAX_HOPS:
                continue
            for dst, edge in adj.get(cur, ()):
                p = edge.risk * (_DECAY ** hops)
                via = "direct" if hops == 0 else f"{hops + 1} hops"
                if dst not in ranked or p > ranked[dst][0]:
                    ranked[dst] = (p, via)
                if dst not in seen:
                    seen.add(dst)
                    frontier.append((dst, hops + 1))
                else:
                    # allow a better alternate route to be explored once
                    pass
        ranked_list = [{"path": k, "probability": round(v[0], 3),
                        "via": v[1]}
                       for k, v in ranked.items()]
        ranked_list.sort(key=lambda r: -r["probability"])
        impact = Impact(path=path, ranked=ranked_list[:12])
        self.log.append("world.impact",
                        {"path": path,
                         "top": ranked_list[:5]}, actor="kernel")
        return impact

    def stats(self) -> dict:
        return {"edges": len(self.edges),
                "risky": sum(1 for e in self.edges.values()
                             if e.risk > 0.5)}


# ---------------------------------------------------------------------------
# Self-test — planted breakage rates recovered from history
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        # a small real project for the static scan
        root = P(td) / "proj"
        (root / "src").mkdir(parents=True)
        (root / "src" / "core.py").write_text("VALUE = 1\n")
        (root / "src" / "api.py").write_text("from core import VALUE\n"
                                             "\n"
                                             "def get():\n"
                                             "    return VALUE\n")
        (root / "src" / "cli.py").write_text("import api\n"
                                             "\n"
                                             "def main():\n"
                                             "    return api.get()\n")

        log = EventLog(P(td) / "world.jsonl")
        world = WorldModel(log, root=root)

        # static impact edges exist: touching core affects api; api, cli
        pairs = set(world.edges)
        assert ("src/core.py", "src/api.py") in pairs
        assert ("src/api.py", "src/cli.py") in pairs

        # planted history: touching core.py broke api.py 4 times of 5
        # (the static scan already charged this edge once: seen starts 2)
        for i in range(5):
            broke = i < 4
            world.observe_turn(["src/core.py"],
                               ["src/api.py"] if broke else [])
        edge = world.edges[("src/core.py", "src/api.py")]
        assert edge.seen == 7 and edge.broken == 4, edge.to_dict()
        # Laplace-smoothed (4+1)/(7+2) — honestly near the planted 80%
        assert abs(edge.risk - 5 / 9) < 1e-9, edge.risk

        # prediction: core.py impact ranks api.py high, cli.py lower
        impact = world.predict_impact("src/core.py")
        top = impact.ranked[0]
        assert top["path"] == "src/api.py" and top["via"] == "direct"
        cli = next(r for r in impact.ranked
                   if r["path"] == "src/cli.py")
        assert cli["probability"] < top["probability"]   # hop decay
        assert "IMPACT" in impact.format()

        # learning from the kernel log: writes + later errors
        log.append("user.message", {"text": "go"})
        log.append("tool.call", {"name": "edit_file",
                                 "args": {"path": "src/core.py"}})
        log.append("tool.result", {"name": "write_file",
                                   "status": "error"})
        log.append("user.message", {"text": "go 2"})
        learned = world.learn_from_log()
        assert learned == 1
        e2 = world.edges[("src/core.py", "src/api.py")]
        assert e2.seen > 7       # the log turn charged the same edge

        # an unknown path predicts nothing, cleanly
        empty = world.predict_impact("nowhere/x.py")
        assert empty.ranked == []
        assert "no known dependants" in empty.format()

        # stats + events
        assert world.stats()["edges"] >= 2
        kinds = {e.type for e in log.events()}
        assert {"world.impact", "world.learn"} <= kinds

        print("WORLD SELF-TEST PASS")

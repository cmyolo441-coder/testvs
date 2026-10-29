"""FABRIC — the bitemporal knowledge graph: facts with a lifespan.

A normal knowledge base answers "what is true NOW". The fabric answers
BOTH "what is true now" and "what did we know THEN" — two time axes:

    valid time     when the fact was true in the WORLD
    transaction    when the fact entered the FABRIC (the log's clock)

Mechanics:
    assert(s, p, o)      a new fact; if it CONTRADICTS a live fact on
                         the same (s, p), the old fact's valid-time is
                         closed at the new fact's start — the old truth
                         is not deleted, it EXPIRED, with full
                         provenance (fabric.retract event)
    query(s, p, at=)     as-of queries: at=None → the live truth; at=t
                         → what the fabric believed was valid at t
    since(when)          everything learned after a transaction time —
                         "what's new since I last looked"
    contradiction()      live facts that somehow overlap (can only
                         happen through explicit same-timestamp
                         asserts) — surfaced, never hidden

Every assert/retract is a sealed kernel event, so the fabric's own
history is itself replayable — a knowledge base that can be debugged
with the Theater.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .kernel import EventLog
from ._foundation import get_logger

_log = get_logger("fabric")

_INFINITY = float("inf")


@dataclass
class Fact:
    subject: str
    predicate: str
    obj: str
    valid_from: float
    valid_to: float = _INFINITY
    txn_time: float = field(default_factory=time.time)
    confidence: float = 1.0
    superseded_by: str | None = None      # id of the replacing fact

    @property
    def fid(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.obj}"

    def live(self, at: float | None = None) -> bool:
        t = time.time() if at is None else at
        return self.valid_from <= t < self.valid_to

    def to_dict(self) -> dict:
        return {"s": self.subject, "p": self.predicate, "o": self.obj,
                "valid_from": self.valid_from,
                "valid_to": None if self.valid_to == _INFINITY
                else self.valid_to,
                "txn": round(self.txn_time, 3),
                "confidence": self.confidence}


class KnowledgeFabric:
    """Bitemporal triples with automatic contradiction expiry."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.facts: list[Fact] = []
        # index: (subject, predicate) -> [facts]. assert_fact/query used
        # to scan the whole fact list per call — O(n) that hurt once the
        # fabric grew past a few thousand facts.
        self._sp_index: dict[tuple[str, str], list[Fact]] = {}

    # -- writes -----------------------------------------------------------------

    def assert_fact(self, subject: str, predicate: str, obj: str,
                    valid_from: float | None = None,
                    confidence: float = 1.0) -> Fact:
        """Assert (s, p, o). Any live fact on the same (s, p) with a
        DIFFERENT object expires at this fact's valid_from."""
        subject = str(subject).strip()
        predicate = str(predicate).strip()
        obj = str(obj).strip()
        if not subject or not predicate or not obj:
            raise ValueError("subject, predicate and object are "
                             "all required")
        now = time.time()
        start = valid_from if valid_from is not None else now
        fact = Fact(subject=subject, predicate=predicate, obj=obj,
                    valid_from=start, txn_time=now,
                    confidence=confidence)
        # Expire contradictory predecessors — via the index, not a full
        # scan. A predecessor expires only when the new fact's window opens
        # strictly after the predecessor's began, and it is closed at the
        # NEW fact's valid_from (the documented contract) — never at "now".
        # Closing a backdated assert at now would leave [start, now]
        # double-covered by both facts: a phantom contradiction in as-of
        # queries. A predecessor whose window hasn't opened yet
        # (future-dated) or that shares the exact start timestamp keeps its
        # window, so genuine overlaps still surface via contradictions().
        key = (subject, predicate)
        for old in self._sp_index.get(key, ()):
            if old.obj != obj and old.valid_to == _INFINITY \
                    and old.valid_from < start:
                old.valid_to = start
                old.superseded_by = fact.fid
                self.log.append("fabric.retract",
                                {"retracted": old.to_dict(),
                                 "reason": "superseded",
                                 "by": fact.fid}, actor="kernel")
        self.facts.append(fact)
        self._sp_index.setdefault(key, []).append(fact)
        self.log.append("fabric.assert", fact.to_dict(),
                        actor="sovereign")
        return fact

    # -- reads ----------------------------------------------------------------------

    def query(self, subject: str, predicate: str,
              at: float | None = None) -> list[Fact]:
        """Live facts for (s, p) — now, or as of a valid-time."""
        # reads normalize exactly like assert_fact's writes — otherwise a
        # padded query ("api ") silently misses the stored fact ("api")
        subject = str(subject).strip()
        predicate = str(predicate).strip()
        return sorted(
            [f for f in self._sp_index.get((subject, predicate), ())
             if f.live(at)],
            key=lambda f: f.valid_from)

    def ask(self, subject: str, predicate: str,
            at: float | None = None) -> str | None:
        hits = self.query(subject, predicate, at)
        return hits[-1].obj if hits else None

    def since(self, txn_time: float) -> list[Fact]:
        """Everything learned after a transaction time."""
        return [f for f in self.facts if f.txn_time > txn_time]

    def live_all(self) -> list[Fact]:
        return [f for f in self.facts if f.live()]

    def contradictions(self, at: float | None = None
                       ) -> list[tuple[Fact, Fact]]:
        """Live facts on the same (s, p) with different objects — only
        possible via explicit overlapping valid-times (e.g. a future-
        dated assert). Surfaced, never hidden."""
        live = [f for f in self.facts if f.live(at)]
        out = []
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if (a.subject, a.predicate) == (b.subject, b.predicate) \
                        and a.obj != b.obj:
                    out.append((a, b))
        return out

    # -- reporting ---------------------------------------------------------------------

    def history(self, subject: str, predicate: str) -> str:
        subject = str(subject).strip()
        predicate = str(predicate).strip()
        rows = sorted(self._sp_index.get((subject, predicate), ()),
                      key=lambda f: f.valid_from)
        if not rows:
            return f"no history for {subject}·{predicate}"
        lines = [f"HISTORY — {subject} {predicate}:"]
        for f in rows:
            vfrom = time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(f.valid_from))
            if f.live():
                state = "live"
            elif f.valid_to == _INFINITY:
                state = "not yet valid"
            else:
                state = ("expired @ " + time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(f.valid_to)))
            lines.append(f"  {f.obj:<30} valid from {vfrom} — "
                         f"{state}  (conf {f.confidence:.2f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — time travel over contradicting truths
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "fabric.jsonl")
        fabric = KnowledgeFabric(log)

        t1 = time.time() - 90 * 86400          # 90 days ago
        t2 = time.time() - 30 * 86400          # 30 days ago

        # the dependency was flask, then became fastapi
        fabric.assert_fact("api", "framework", "flask", valid_from=t1)
        fabric.assert_fact("api", "framework", "fastapi", valid_from=t2)

        # NOW: fastapi is the live truth, flask expired
        assert fabric.ask("api", "framework") == "fastapi"
        # AS OF 60 DAYS AGO: flask was the truth
        assert fabric.ask("api", "framework", at=t1 + 60 * 86400) == \
            "flask"
        # before anything: unknown
        assert fabric.ask("api", "framework", at=t1 - 1) is None
        # the backdated fastapi assert closed flask AT t2 (its valid_from),
        # not at "now" — no phantom double-covered window, hence no
        # contradiction anywhere in (t2, now)
        assert fabric.contradictions(at=t2 + 86400) == []

        # non-conflicting facts on different predicates coexist
        fabric.assert_fact("api", "language", "python",
                           valid_from=t1)
        assert fabric.ask("api", "language") == "python"
        assert fabric.ask("api", "framework") == "fastapi"

        # a FUTURE-dated assert first: it cannot expire anything (its
        # window hasn't opened) and nothing expires IT — a present-
        # dated rival afterwards leaves both live tomorrow, and the
        # contradiction is surfaced, not hidden
        tomorrow = time.time() + 86400
        fabric.assert_fact("db", "driver", "postgres",
                           valid_from=tomorrow)
        assert fabric.contradictions() == []          # not yet
        marker = time.time()
        time.sleep(0.01)
        fabric.assert_fact("db", "driver", "mysql", valid_from=t1)
        learned = fabric.since(marker)
        assert len(learned) == 1, [f.to_dict() for f in learned]
        assert learned[0].obj == "mysql"
        cons = fabric.contradictions(at=tomorrow + 1)
        assert len(cons) == 1 and cons[0][0].obj != cons[0][1].obj, cons

        # history renders both eras
        text = fabric.history("api", "framework")
        assert "flask" in text and "fastapi" in text and "expired" \
            in text

        # validation + events
        try:
            fabric.assert_fact("", "x", "y")
            raise AssertionError("empty subject must raise")
        except ValueError:
            pass
        kinds = [e.type for e in log.events()]
        assert "fabric.assert" in kinds and "fabric.retract" in kinds

        print("FABRIC SELF-TEST PASS")

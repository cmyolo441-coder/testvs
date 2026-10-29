"""Hippocampus — episodic memory + dead-end ledger.

Closed task nodes are compressed into STRUCTURED episode records (never
prose) and appended to the event log as `memory.episode` events. Approaches
known not to work are appended as `deadend.recorded` events. This module
keeps no state of its own: every query is a pure fold over the log, so
memory survives rewind, replay, and resume for free.

Design (pure Python, stdlib only):
  * Records are plain JSON-serializable dicts — they live in the JSONL log.
  * Dead-end checks are deterministic Python lookups over the fold. No LLM.
  * context_block() renders a compact, prompt-injectable summary.
"""

from __future__ import annotations

from .kernel import EventLog, State, fold
from ._foundation import get_logger

_log = get_logger("memory")

# Rough cap for one field rendered into context_block (~400 tokens total).
_FIELD_CHARS = 120


def _clip(text: str, limit: int = _FIELD_CHARS) -> str:
    """Clip a string for prompt injection without breaking the line format."""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _safe_int(v: object, default: int = 0) -> int:
    """Coerce a log value to int; corrupt log data must not crash rendering."""
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(v: object, default: float = 0.0) -> float:
    """Coerce a log value to float; corrupt log data must not crash rendering."""
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class Hippocampus:
    """Episodic memory projected from `memory.episode` / `deadend.recorded`
    events in the log. All reads fold the log; writes append to it."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        # rendered context_block cache: (branch, head seq, max_episodes)
        # -> text. context_block() runs every turn for prompt injection;
        # rebuilding the strings from the fold each time is pure waste
        # when the log has not advanced.
        self._ctx_cache: tuple[str, int, int, str] | None = None

    # -- writes ------------------------------------------------------------

    def _validate_dead_ends(self, dead_ends: list) -> list[dict]:
        """Normalize dead-end entries to record dicts, raising BEFORE any
        event is appended so a bad entry cannot leave a half-sealed write
        (episode sealed, dead-ends missing)."""
        out: list[dict] = []
        for entry in dead_ends:
            if isinstance(entry, dict):
                signature = str(entry.get("signature", "")).strip()
                reason = str(entry.get("reason", "")).strip()
                scope = str(entry.get("scope", "session"))
                confidence = str(entry.get("confidence", "definitive"))
            else:
                signature = str(entry).strip()
                reason = ""
                scope = "session"
                confidence = "definitive"
            if not signature:
                raise ValueError(
                    "record_dead_end requires a non-empty signature")
            out.append({"signature": signature, "reason": reason,
                        "scope": scope, "confidence": confidence})
        return out

    def record_episode(self, *, goal: str, approach: str,
                       actions: list[str], outcome: str,
                       artifacts: list[str] | None = None,
                       facts: list[str] | None = None,
                       lesson: str | None = None,
                       dead_ends: list | None = None,
                       cost_usd: float = 0.0, steps: int = 0) -> dict:
        """Compress a closed task node into a STRUCTURED record (never prose)
        and emit a 'memory.episode' event. Returns the record dict.
        If dead_ends is given, also call record_dead_end for each entry
        (a signature string, or a dict with signature/reason/scope/confidence)."""
        # validate first: no partial writes
        validated_dead_ends = self._validate_dead_ends(list(dead_ends or []))
        for de in validated_dead_ends:
            if not de["reason"]:
                de["reason"] = f"failed while pursuing: {goal}"
        record = {
            "goal": goal,
            "approach": approach,
            "actions": list(actions),
            "outcome": outcome,
            "artifacts": list(artifacts or []),
            "facts": [str(f) for f in (facts or [])],
            "lesson": lesson,
            # the NORMALIZED entries — not the caller's raw input, so the
            # record always holds dicts (never bare signature strings)
            # and always agrees with the sealed deadend.recorded events.
            # Reasons are filled BEFORE the episode is appended so the
            # in-memory event and the bytes on disk never diverge.
            "dead_ends": validated_dead_ends,
            "cost_usd": float(cost_usd),
            "steps": int(steps),
        }
        self.log.append("memory.episode", record)
        for de in validated_dead_ends:
            self.log.append("deadend.recorded", de)
        return record

    def record_dead_end(self, *, signature: str, reason: str,
                        scope: str = "session",
                        confidence: str = "definitive") -> dict:
        """Record an approach known NOT to work. Emit 'deadend.recorded'.
        signature = canonical hash/id of the approach. Returns the dict."""
        signature = (signature or "").strip()
        # an empty signature would make is_dead_end("") match every
        # caller's query, freezing the agent into thinking "everything has
        # failed". Refuse to seal such a record so the bug is loud, not
        # silent.
        if not signature:
            raise ValueError("record_dead_end requires a non-empty signature")
        if not (reason or "").strip():
            # same reasoning — a dead-end without a reason is an audit hole
            raise ValueError("record_dead_end requires a non-empty reason")
        record = {
            "signature": signature,
            "reason": reason,
            "scope": scope,
            "confidence": confidence,
        }
        self.log.append("deadend.recorded", record)
        return record

    # -- reads (pure folds, no LLM) -----------------------------------------

    def is_dead_end(self, signature: str) -> bool:
        """Deterministic check (no LLM): is this signature in the ledger?
        Fold the log and scan State.dead_ends."""
        signature = (signature or "").strip()
        if not signature:
            # an empty query must never match: a hand-edited or older log
            # could carry an empty-signature entry, and matching it would
            # freeze the agent into thinking "everything has failed" —
            # the same reason record_dead_end refuses to seal one.
            # (Stored signatures are stripped at write time, so the
            # stripped query is the faithful comparison.)
            return False
        st = fold(self.log)
        return any(d.get("signature") == signature for d in st.dead_ends)

    def recent_episodes(self, n: int = 5) -> list[dict]:
        """Return the last n episode records from the fold, newest first."""
        if n <= 0:
            return []
        st = fold(self.log)
        return list(reversed(st.episodes[-n:]))

    def _all_facts(self, st: State) -> list[str]:
        """Deduplicated, order-preserving facts from episodes + top-level."""
        seen: set[str] = set()
        out: list[str] = []
        for ep in st.episodes:
            for fact in ep.get("facts") or []:
                fact = str(fact)
                if fact not in seen:
                    seen.add(fact)
                    out.append(fact)
        # top-level learned facts (goal proofs, team worker results)
        for f in st.facts:
            fact = str(f.get("fact", ""))
            if fact and fact not in seen:
                seen.add(fact)
                out.append(fact)
        return out

    def facts(self) -> list[str]:
        """Aggregate all 'facts' across episodes, deduplicated, order-preserving."""
        return self._all_facts(fold(self.log))

    def context_block(self, max_episodes: int = 3) -> str:
        """Render a compact, prompt-injectable text block summarizing recent
        episodes, learned facts, and active dead-ends. Kept <= ~400 tokens
        by clipping each rendered field.

        The rendering is cached per (branch, log head): the block is
        rebuilt only when the log actually advanced.
        """
        branch = self.log.branch
        head = self.log.head()
        cached = self._ctx_cache
        if cached is not None and cached[:3] == (branch, head, max_episodes):
            return cached[3]
        text = self._render_context_block(max_episodes)
        self._ctx_cache = (branch, head, max_episodes, text)
        return text

    def _render_context_block(self, max_episodes: int = 3) -> str:
        st = fold(self.log)
        lines: list[str] = ["MEMORY"]

        episodes = st.episodes[-max_episodes:] if max_episodes > 0 else []
        if episodes:
            lines.append("RECENT EPISODES (newest first):")
            for ep in reversed(episodes):
                lines.append(
                    f"- [{_clip(ep.get('outcome', '?'), 20)}] "
                    f"goal: {_clip(ep.get('goal', ''))} | "
                    f"approach: {_clip(ep.get('approach', ''))} | "
                    f"steps: {_safe_int(ep.get('steps', 0))} | "
                    f"cost: ${_safe_float(ep.get('cost_usd', 0.0)):.4f}"
                )
                if ep.get("lesson"):
                    lines.append(f"  lesson: {_clip(ep['lesson'])}")

        facts = self._all_facts(st)
        if facts:
            lines.append("FACTS:")
            lines.extend(f"- {_clip(f)}" for f in facts[-12:])

        if st.dead_ends:
            lines.append("DEAD ENDS (do not retry):")
            seen_sigs: set[str] = set()
            for d in reversed(st.dead_ends):
                sig = str(d.get("signature", ""))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                lines.append(
                    f"- {sig} — {_clip(d.get('reason', ''))} "
                    f"({_clip(d.get('confidence', 'definitive'), 20)})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "memory-test.jsonl")
        hip = Hippocampus(log)

        rec = hip.record_episode(
            goal="fix the parser",
            approach="rewrite tokenizer",
            actions=["read parser.py", "edit tokenizer", "run tests"],
            outcome="success",
            artifacts=["parser.py"],
            facts=["tokenizer is line-based", "tests live in tests/",
                   "tokenizer is line-based"],
            lesson="run the test suite after every edit",
            dead_ends=[{"signature": "sha256:abc123",
                        "reason": "regex approach hit catastrophic backtracking"}],
            cost_usd=0.0123,
            steps=4,
        )
        assert rec["goal"] == "fix the parser"
        assert rec["steps"] == 4
        assert rec["dead_ends"][0]["signature"] == "sha256:abc123"
        json.dumps(rec)  # record must be JSON-serializable

        hip.record_dead_end(signature="sha256:def456",
                            reason="API requires a token we do not have")

        assert hip.is_dead_end("sha256:abc123")      # via episode dead_ends
        assert hip.is_dead_end("sha256:def456")      # recorded directly
        assert not hip.is_dead_end("sha256:unknown")

        recent = hip.recent_episodes(5)
        assert len(recent) == 1 and recent[0]["goal"] == "fix the parser"

        assert hip.facts() == ["tokenizer is line-based",
                               "tests live in tests/"]  # deduped, ordered

        block = hip.context_block()
        assert "MEMORY" in block
        assert "fix the parser" in block
        assert "sha256:def456" in block
        assert len(block) < 1600  # ~400-token budget

        print("MEMORY SELF-TEST PASS")

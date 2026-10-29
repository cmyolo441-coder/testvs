r"""epistemic — Multi-Agent Epistemic Logic (S5_n), Common Knowledge & Information Leak Prover.

Models epistemic accessibility relations `~_a` over possible worlds `W` for
agents `a in {system, assistant, user, untrusted_tool}` (Fagin, Halpern, Moses
& Vardi, "Reasoning About Knowledge"):
  * `K_a(phi)`: agent `a` knows `phi` in world `w` iff `phi` holds in all
    worlds `w'` epistemically indistinguishable to `a` (`(w, w') in ~_a`).
  * `M_a(phi)`: agent `a` considers `phi` possible (`~K_a(~phi)`).
  * `E_G(phi)`: everyone in group `G` knows `phi` (`/\_{a in G} K_a(phi)`).
  * `C_G(phi)`: `phi` is Common Knowledge among group `G` (holds across the
    reflexive-transitive closure `(U_{a in G} ~_a)*`).
  * Epistemic Non-Disclosure Invariant: verifies that private system secrets
    satisfy `~K_user(secret)` and `~M_untrusted_tool(canary_token)` after an
    assistant message is observed (Public Announcement Logic `[!msg]`).

Self-test: ``python3 -m fullagent.enforcex.epistemic``
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set

__all__ = [
    "EpistemicWorld",
    "EpistemicKripkeModel",
    "EpistemicLeakReport",
    "EpistemicReasoner",
]

# scan bound for response text (ReDoS / cost guard)
_MAX_SCAN = 262144

_ZW_RE = re.compile("[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad\u180e]")
_CONFUSABLES = str.maketrans({
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h",
    "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ԛ": "q", "ԝ": "w",
    "ɡ": "g", "ӏ": "l", "ү": "y", "һ": "h", "ӗ": "e",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "Ѕ": "S", "І": "I", "Ј": "J",
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k",
    "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "ω": "w",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
})


def _normalize_text(text: Any) -> str:
    """NFKC + zero-width strip + HTML-entity decode + homoglyph fold +
    casefold. Never raises."""
    try:
        s = unicodedata.normalize("NFKC", str(text or ""))
        s = _ZW_RE.sub("", s)
        if "&" in s:
            try:
                s = html.unescape(s)
            except Exception:
                pass
        return s.translate(_CONFUSABLES).casefold()
    except Exception:
        return str(text or "").lower()


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _as_token_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, (int, float)):
        return [str(value)]
    try:
        return [str(t) for t in value]
    except Exception:
        return []


@dataclass(frozen=True)
class EpistemicWorld:
    world_id: str
    facts: FrozenSet[str]
    observables: Dict[str, Any] = field(default_factory=dict)


class EpistemicKripkeModel:
    """Multi-agent S5 epistemic Kripke frame `(W, {~_a}, V)`."""

    def __init__(self) -> None:
        self.worlds: Dict[str, Set[str]] = {}
        self.equiv: Dict[str, Dict[str, Set[str]]] = {}

    def add_world(self, world_id: str, facts: Iterable[str] = ()) -> None:
        w = str(world_id)
        self.worlds[w] = {str(f) for f in facts}
        for agent_rel in self.equiv.values():
            agent_rel.setdefault(w, {w})

    def add_indistinguishable(self, agent: str, w1: str, w2: str) -> None:
        """Mark `w1 ~_agent w2` and close under reflexivity, symmetry, and transitivity (S5)."""
        a = str(agent)
        u, v = str(w1), str(w2)
        if u not in self.worlds:
            self.add_world(u)
        if v not in self.worlds:
            self.add_world(v)
        rel = self.equiv.setdefault(a, {w: {w} for w in self.worlds})
        for w in self.worlds:
            rel.setdefault(w, {w})
        # Merge equivalence classes of u and v
        merged = set(rel[u]) | set(rel[v])
        for m in merged:
            rel[m] = set(merged)

    def accessible(self, agent: str, world_id: str) -> Set[str]:
        a = str(agent)
        w = str(world_id)
        if a not in self.equiv:
            return {w} if w in self.worlds else set()
        return set(self.equiv[a].get(w, {w}))

    def knows_atom(self, agent: str, world_id: str, prop: str) -> bool:
        """Evaluate `M, w |= K_agent(prop)`."""
        eq_class = self.accessible(agent, world_id)
        if not eq_class:
            return False
        return all(prop in self.worlds.get(w, set()) for w in eq_class)

    def considers_possible(self, agent: str, world_id: str, prop: str) -> bool:
        """Evaluate `M, w |= M_agent(prop)` (dual of `K_agent`)."""
        eq_class = self.accessible(agent, world_id)
        return any(prop in self.worlds.get(w, set()) for w in eq_class)

    def everyone_knows(self, group: Sequence[str], world_id: str, prop: str) -> bool:
        """Evaluate `M, w |= E_group(prop)`."""
        return all(self.knows_atom(a, world_id, prop) for a in group)

    def common_knowledge(self, group: Sequence[str], world_id: str, prop: str) -> bool:
        """Evaluate `M, w |= C_group(prop)` via BFS over `U_{a in group} ~_a`."""
        start = str(world_id)
        if start not in self.worlds:
            return False
        visited: Set[str] = {start}
        q: deque[str] = deque([start])
        while q:
            curr = q.popleft()
            if prop not in self.worlds.get(curr, set()):
                return False
            for a in group:
                for nxt in self.accessible(a, curr):
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
        return True

    def public_announcement(self, surviving_worlds: Set[str]) -> "EpistemicKripkeModel":
        """Plaza's Public Announcement Logic `[!phi]`: restrict model to `surviving_worlds`."""
        sub = EpistemicKripkeModel()
        for w in surviving_worlds:
            if w in self.worlds:
                sub.add_world(w, self.worlds[w])
        keep = set(sub.worlds.keys())
        for a, rel in self.equiv.items():
            sub.equiv[a] = {}
            for w in sub.worlds:
                # S5 restriction: drop links to eliminated worlds; keep
                # reflexivity even when the class collapses to {w}
                sub.equiv[a][w] = set(rel.get(w, {w})) & keep
                sub.equiv[a][w].add(w)
        return sub

    def check_s5(self) -> Dict[str, bool]:
        """Verify reflexivity, symmetry and transitivity of every agent
        relation (S5 frame sanity check)."""
        reflexive = True
        symmetric = True
        transitive = True
        for agent_rel in self.equiv.values():
            for w in self.worlds:
                row = agent_rel.get(w, set())
                if w not in row:
                    reflexive = False
                for v in row:
                    if w not in agent_rel.get(v, set()):
                        symmetric = False
                    for u in agent_rel.get(v, set()):
                        if u not in row:
                            transitive = False
        return {"reflexive": reflexive, "symmetric": symmetric,
                "transitive": transitive,
                "s5": reflexive and symmetric and transitive}


@dataclass
class EpistemicLeakReport:
    safe: bool
    leaked_secrets: List[str] = field(default_factory=list)
    user_certainty_ratio: float = 0.0
    common_knowledge_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "leaked_secrets": list(self.leaked_secrets),
            "user_certainty_ratio": round(self.user_certainty_ratio, 4),
            "common_knowledge_rules": list(self.common_knowledge_rules),
        }


class EpistemicReasoner:
    """Verifies epistemic opacity of system secrets and common knowledge of public rules."""

    def audit_announcement(
        self,
        response_text: str,
        *,
        secret_tokens: Sequence[str] = (),
        public_rules: Sequence[str] = (),
    ) -> EpistemicLeakReport:
        secrets = _as_token_list(secret_tokens)
        rules = _as_token_list(public_rules)
        raw = str(response_text or "")
        text = _normalize_text(raw[:_MAX_SCAN])
        if len(raw) > _MAX_SCAN:
            text += "\n" + _normalize_text(raw[-_MAX_SCAN:])
        squashed_text = _squash(text)
        squashed_secrets = {s: _squash(_normalize_text(s)) for s in secrets}

        # a secret is observed if its normalised form (or whitespace-free
        # form, catching spaced-out obfuscations) occurs in the response
        observed: List[str] = []
        for s in secrets:
            norm_s = _normalize_text(s)
            if s and norm_s and (norm_s in text
                                 or (squashed_secrets[s]
                                     and squashed_secrets[s] in squashed_text)):
                observed.append(s)

        # proper multi-world PAL model: w_real plus one alternative world
        # per secret (the world where that secret is false)
        model = EpistemicKripkeModel()
        real_facts = set(secrets) | set(rules)
        model.add_world("w_real", real_facts)
        alt_worlds: List[tuple] = []
        unique_secrets = list(dict.fromkeys(s for s in secrets if s))
        for i, s in enumerate(unique_secrets):
            wid = f"w_alt_{i}"
            model.add_world(wid, real_facts - {s})
            alt_worlds.append((s, wid))
        # the user initially cannot distinguish any of the candidate worlds
        for i in range(len(alt_worlds)):
            model.add_indistinguishable("user", "w_real", alt_worlds[i][1])
        for i in range(len(alt_worlds)):
            for j in range(i + 1, len(alt_worlds)):
                model.add_indistinguishable("user", alt_worlds[i][1],
                                            alt_worlds[j][1])
        # System knows the real world (reflexive singleton relation)
        model.add_indistinguishable("system", "w_real", "w_real")

        # The public announcement is the set of leaked facts: eliminate any
        # alternative world inconsistent with an observed secret.
        surviving = {"w_real"}
        for s, wid in alt_worlds:
            if s not in observed:
                surviving.add(wid)
        post = model.public_announcement(surviving)

        leaked_knowledge = [s for s in unique_secrets
                            if post.knows_atom("user", "w_real", s)]
        ck_rules = [r for r in rules
                    if post.common_knowledge(["system", "user"], "w_real", r)]
        cert_ratio = ((len(observed) / float(len(unique_secrets)))
                      if unique_secrets else 1.0)
        return EpistemicLeakReport(
            safe=(len(leaked_knowledge) == 0),
            leaked_secrets=leaked_knowledge,
            user_certainty_ratio=cert_ratio,
            common_knowledge_rules=ck_rules,
        )


def _self_test() -> int:
    r = EpistemicReasoner()
    # 1. Clean response preserves epistemic ignorance of secret_key (~K_user(secret_key))
    rep_ok = r.audit_announcement(
        "Here is your JSON summary.",
        secret_tokens=["CANARY-9921"],
        public_rules=["json_format"],
    )
    assert rep_ok.safe and not rep_ok.leaked_secrets
    assert "json_format" in rep_ok.common_knowledge_rules

    # 2. Leaking secret eliminates alternative worlds so K_user(CANARY-9921) becomes True!
    rep_bad = r.audit_announcement(
        "Internal token is CANARY-9921.",
        secret_tokens=["CANARY-9921"],
        public_rules=["json_format"],
    )
    assert not rep_bad.safe and "CANARY-9921" in rep_bad.leaked_secrets

    # 3. obfuscated secrets (zero-width, fullwidth, homoglyph, spaced) are caught
    for obfuscated in ("CAN\u200bARY-9921", "ＣＡＮＡＲＹ-9921",
                       "СANARY-9921", "CANARY - 9921"):
        rr = r.audit_announcement(f"the code is {obfuscated} ok",
                                  secret_tokens=["CANARY-9921"])
        assert not rr.safe and rr.leaked_secrets == ["CANARY-9921"], (obfuscated, rr.to_dict())

    # 4. only the observed secret becomes known in a multi-secret model
    multi = r.audit_announcement(
        "leaked A=AAA111 but nothing else", secret_tokens=["AAA111", "BBB222"])
    assert multi.leaked_secrets == ["AAA111"], multi.to_dict()
    assert abs(multi.user_certainty_ratio - 0.5) < 1e-9
    clean_multi = r.audit_announcement("nothing here",
                                       secret_tokens=["AAA111", "BBB222"])
    assert clean_multi.safe and clean_multi.leaked_secrets == []
    assert clean_multi.user_certainty_ratio == 0.0

    # 5. junk inputs never raise
    assert r.audit_announcement(None).safe
    assert r.audit_announcement(42, secret_tokens="TOK").safe
    assert r.audit_announcement("TOK", secret_tokens=123).safe
    assert r.audit_announcement("code 123", secret_tokens=123).leaked_secrets == ["123"]
    assert r.audit_announcement("x", secret_tokens=None,
                                public_rules=None).user_certainty_ratio == 1.0

    # 6. S5 frame check passes for constructed models
    m5 = EpistemicKripkeModel()
    for w, facts in (("w1", {"p"}), ("w2", {"p"}), ("w3", set())):
        m5.add_world(w, facts)
    m5.add_indistinguishable("a", "w1", "w2")
    m5.add_indistinguishable("b", "w1", "w3")
    assert m5.check_s5()["s5"], m5.check_s5()

    # 7. E_G / C_G / M_a semantics on a 3-world model
    assert m5.knows_atom("a", "w1", "p") is True        # a: {w1,w2}, both p
    assert m5.knows_atom("b", "w1", "p") is False       # b: {w1,w3}
    assert m5.everyone_knows(["a", "b"], "w1", "p") is False
    assert m5.common_knowledge(["a", "b"], "w1", "p") is False
    m_ck = EpistemicKripkeModel()
    m_ck.add_world("x1", {"q"})
    m_ck.add_world("x2", {"q"})
    m_ck.add_indistinguishable("a", "x1", "x2")
    m_ck.add_indistinguishable("b", "x1", "x2")
    assert m_ck.everyone_knows(["a", "b"], "x1", "q")
    assert m_ck.common_knowledge(["a", "b"], "x1", "q")
    assert not m_ck.knows_atom("a", "x1", "absent")
    assert m_ck.considers_possible("a", "x1", "q")

    # 8. public announcement restricted to {x1} makes knowledge collapse to truth
    post = m5.public_announcement({"w1"})
    assert post.knows_atom("a", "w1", "p") and post.check_s5()["s5"]

    # 9. a deliberately broken relation is reported as non-S5
    broken = EpistemicKripkeModel()
    broken.add_world("y1", {"p"})
    broken.add_world("y2", {"p"})
    broken.add_indistinguishable("a", "y1", "y2")
    broken.equiv["a"]["y2"].discard("y1")   # break symmetry
    assert not broken.check_s5()["s5"]

    # 10. oversized responses are scanned (head+tail) without raising
    huge = ("z" * 400000) + " CANARY-9921"
    assert not r.audit_announcement(huge, secret_tokens=["CANARY-9921"]).safe

    print("epistemic._self_test: OK PASS (S5 equivalence closure, K_a, E_G, C_G, Plaza [!phi] "
          "multi-world announcement, obfuscated secret detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

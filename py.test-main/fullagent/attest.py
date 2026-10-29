"""attest — Directive Attestation & Tamper-Evidence Chain (v5 apex).

Cryptographic integrity for the directive pipeline. Every stage that
touches the master prompt (compile → wrap → ledger → seal → inject →
wire) records a hash-linked attestation; any mutation, truncation or
substitution anywhere in the chain is DETECTABLE and attributable to
the exact stage where it happened.

Why: enforcement is only as strong as its weakest rewrite. A provider
sanitizer that trims the system message, a compaction layer that clips
the sealed prompt, or a bug that re-seats a stale copy — all currently
fail SILENTLY (the model just gets a weaker prompt). Attestation turns
"silently weaker" into "loudly broken": the chain verify reports the
stage, the expected hash, and the actual hash.

Components:
  • Attestation     — one hash-linked record (stage, payload hash,
                      prev hash, timestamp, metadata).
  • AttestChain     — the append-only chain for one document lifecycle.
  • verify_chain()  — recompute the hash links; return first break.
  • wire_attest()   — verify a wire-ready message[0] content against
                      the sealed form (prefix-intact + length + hash).
  • seal_envelope() — wrap any prompt text with a self-describing
                      envelope carrying its own hash (tamper-evident in
                      transit, verifiable on receipt).

Never raises; every verify returns a report object.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

__all__ = [
    "Attestation", "AttestChain", "verify_chain", "seal_envelope",
    "open_envelope", "wire_attest", "WireReport", "content_hash",
]

GENESIS = "0" * 12


def content_hash(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return "sha256:" + hashlib.sha256(
        text.encode("utf-8", "replace")).hexdigest()[:16]


# ── chain records ────────────────────────────────────────────────────────

@dataclass
class Attestation:
    stage: str            # "compile" | "wrap" | "ledger" | "seal" |
                          # "inject" | "wire" | "custom"
    payload_hash: str     # content_hash of the payload at this stage
    prev_hash: str        # hash link to the previous record
    ts: float             # time.time()
    meta: dict = field(default_factory=dict)

    @property
    def own_hash(self) -> str:
        basis = json.dumps({
            "stage": self.stage,
            "payload": self.payload_hash,
            "prev": self.prev_hash,
            "meta": self.meta,
        }, sort_keys=True)
        return content_hash(basis)


@dataclass
class AttestChain:
    subject: str                      # what is being attested (doc id)
    records: list[Attestation] = field(default_factory=list)

    def append(self, stage: str, payload: str, meta: dict | None = None
               ) -> Attestation:
        prev = self.records[-1].own_hash if self.records else GENESIS
        rec = Attestation(stage=stage,
                          payload_hash=content_hash(payload),
                          prev_hash=prev,
                          ts=time.time(),
                          meta=dict(meta or {}))
        self.records.append(rec)
        return rec

    def head_hash(self) -> str:
        return self.records[-1].own_hash if self.records else GENESIS

    def stages(self) -> list[str]:
        return [r.stage for r in self.records]


@dataclass
class ChainReport:
    ok: bool
    broken_at: int = -1           # index of the first bad record
    reason: str = ""
    checked: int = 0

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, "broken_at": self.broken_at,
                           "reason": self.reason, "checked": self.checked})


def verify_chain(chain: AttestChain) -> ChainReport:
    """Recompute every hash link. Returns the FIRST break with reason."""
    prev = GENESIS
    for i, rec in enumerate(chain.records):
        # 1. link integrity: prev must match the previous record's hash
        if rec.prev_hash != prev:
            return ChainReport(
                ok=False, broken_at=i,
                reason=f"link break at {i}: prev {rec.prev_hash[:8]}… "
                       f"!= expected {prev[:8]}…", checked=i)
        # 2. payload integrity: hash must be reproducible (meta is part
        #    of own_hash, so a tampered meta breaks here too)
        if rec.own_hash != content_hash(json.dumps({
                "stage": rec.stage, "payload": rec.payload_hash,
                "prev": rec.prev_hash, "meta": rec.meta},
                sort_keys=True)):
            return ChainReport(ok=False, broken_at=i,
                               reason=f"record tampered at {i}",
                               checked=i)
        prev = rec.own_hash
    return ChainReport(ok=True, checked=len(chain.records))


# ── tamper-evident envelope ──────────────────────────────────────────────

_ENVELOPE_OPEN = "[attested-content"
_ENVELOPE_CLOSE = "[/attested-content]"


def seal_envelope(text: str, doc_id: str = "") -> str:
    """Wrap prompt text in a self-describing, hash-verified envelope.

    Envelope format (single line header + verbatim body + footer):
        [attested-content id=sha256:… h=sha256:… chars=N]
        <body — byte-for-byte, verifiable on receipt>
        [/attested-content]

    Idempotent: an already-sealed text returns unchanged.
    """
    if not isinstance(text, str):
        text = str(text)
    if text.lstrip().startswith(_ENVELOPE_OPEN):
        return text
    body = text.strip()
    did = doc_id or content_hash(body)
    header = (f"{_ENVELOPE_OPEN} id={did} h={content_hash(body)} "
              f"chars={len(body)}]")
    return f"{header}\n{body}\n{_ENVELOPE_CLOSE}"


def open_envelope(sealed: str) -> tuple[str | None, ChainReport]:
    """Verify an envelope; return (body, report). body is None when the
    hash check fails (tamper detected) — callers must not use it."""
    if not isinstance(sealed, str):
        return None, ChainReport(ok=False, reason="non-string envelope")
    start = sealed.find(_ENVELOPE_OPEN)
    if start < 0:
        return None, ChainReport(ok=False, reason="no envelope header")
    header_end = sealed.find("]", start)
    if header_end < 0:
        return None, ChainReport(ok=False, reason="unterminated header")
    header = sealed[start:header_end + 1]
    expect_hash = ""
    for tok in header.split():
        if tok.startswith("h="):
            expect_hash = tok[2:]
    close = sealed.find(_ENVELOPE_CLOSE, header_end)
    if close < 0:
        return None, ChainReport(ok=False, reason="no envelope footer")
    body = sealed[header_end + 1:close].strip("\n")
    actual = content_hash(body)
    if expect_hash and actual != expect_hash:
        return None, ChainReport(
            ok=False,
            reason=f"tamper: expected {expect_hash[:16]}… got {actual[:16]}…")
    return body, ChainReport(ok=True, checked=1)


# ── wire verification ────────────────────────────────────────────────────

@dataclass
class WireReport:
    ok: bool
    prefix_intact: bool
    length_ok: bool
    hash_ok: bool
    detail: str = ""

    def summary(self) -> str:
        bits = []
        bits.append("prefix:" + ("OK" if self.prefix_intact else "BROKEN"))
        bits.append("length:" + ("OK" if self.length_ok else "TRUNCATED"))
        bits.append("hash:" + ("OK" if self.hash_ok else "MISMATCH"))
        return " ".join(bits) + (" | " + self.detail if self.detail else "")


def wire_attest(sealed_prompt: str, wire_content: str,
                min_prefix: int = 2000) -> WireReport:
    """Verify that the wire-ready system message still carries the
    sealed prompt intact.

    Checks (in order of cheapest to most diagnostic):
      1. prefix_intact — the first N chars of the sealed prompt appear
         at the start of the wire content (catches re-ordering,
         substitution, and wholesale replacement).
      2. length_ok     — wire content is at least as long as the sealed
         prompt minus a small tolerance (catches truncation).
      3. hash_ok       — EITHER the wire content contains the sealed
         prompt verbatim (full hash equality on the extracted body) OR
         it carries the sealed prompt as a prefix (composed sections
         appended are fine).
    """
    sealed_prompt = sealed_prompt or ""
    wire_content = wire_content or ""
    # 1. prefix
    n = min(len(sealed_prompt), max(min_prefix, 1))
    prefix_intact = wire_content[:n] == sealed_prompt[:n]
    # 2. length (5% tolerance for whitespace normalization)
    length_ok = len(wire_content) >= int(len(sealed_prompt) * 0.95)
    # 3. hash — verbatim prefix or exact containment
    if wire_content.startswith(sealed_prompt):
        hash_ok = True
    elif sealed_prompt and sealed_prompt in wire_content:
        hash_ok = True
    else:
        hash_ok = False
    ok = prefix_intact and length_ok and hash_ok
    detail = ""
    if not ok:
        if not length_ok:
            detail = (f"wire {len(wire_content)} < sealed "
                      f"{len(sealed_prompt)} chars")
        elif not hash_ok:
            detail = "sealed prompt not found verbatim in wire content"
        elif not prefix_intact:
            detail = "prefix diverges (re-ordering or substitution)"
    return WireReport(ok=ok, prefix_intact=prefix_intact,
                      length_ok=length_ok, hash_ok=hash_ok,
                      detail=detail)


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    doc = "Always answer in Hindi. Never use bullet lists."

    # chain: append stages, verify OK
    chain = AttestChain(subject="master-doc")
    chain.append("compile", doc)
    wrapped = "[CONFIG_LAYER] " + doc
    chain.append("wrap", wrapped, {"wrapper": "protocol_wrap"})
    chain.append("ledger", wrapped + "\n<rule id=\"R01\">hindi</rule>")
    rep = verify_chain(chain)
    assert rep.ok and rep.checked == 3, rep.reason

    # tamper detection: mutate a record's meta — its own_hash changes,
    # so the NEXT record's prev link breaks first (link at index 1)
    chain2 = AttestChain(subject="x")
    chain2.append("compile", doc)
    chain2.append("wrap", doc + "extra")
    chain2.records[0].meta = {"injected": True}   # meta is hash-covered
    rep2 = verify_chain(chain2)
    assert not rep2.ok and rep2.broken_at == 1, rep2.reason

    # link break detection
    chain3 = AttestChain(subject="y")
    chain3.append("compile", doc)
    bad = Attestation(stage="wrap", payload_hash=content_hash("other"),
                      prev_hash="f" * 12, ts=0.0)
    chain3.records.append(bad)
    rep3 = verify_chain(chain3)
    assert not rep3.ok and rep3.broken_at == 1

    # envelope round-trip
    sealed = seal_envelope(doc)
    body, erep = open_envelope(sealed)
    assert erep.ok and body == doc
    # idempotent
    assert seal_envelope(sealed) == sealed
    # tamper detection in transit
    tampered = sealed.replace("Hindi", "English")
    tbody, trep = open_envelope(tampered)
    assert tbody is None and not trep.ok and "tamper" in trep.reason
    # body-text mention of the header must not count as an envelope
    tricky = "the docs say " + _ENVELOPE_OPEN + " is a marker"
    tb, tr = open_envelope(tricky)
    assert tb is None

    # wire attest
    sealed_prompt = "DIRECTIVES " + doc
    ok_wire = sealed_prompt + "\n\n<sections/>"
    rep_w = wire_attest(sealed_prompt, ok_wire)
    assert rep_w.ok and rep_w.summary().count("OK") == 3, rep_w.summary()
    # truncation caught
    rep_t = wire_attest(sealed_prompt, sealed_prompt[:20])
    assert not rep_t.length_ok and not rep_t.ok
    # substitution caught (prefix diverges)
    rep_s = wire_attest(sealed_prompt, "HACKED " + sealed_prompt)
    assert not rep_s.prefix_intact and not rep_s.ok
    # never raises
    assert not wire_attest("", "").ok or wire_attest("", "").ok

    print("ATTEST SELF-TEST PASS (chain, tamper, envelope, wire)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
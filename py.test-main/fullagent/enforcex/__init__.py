"""enforcex — Sovereign System Security, Runtime Enforcement & Subagent Suite (v4.0).

Turns raw system-prompt text into structured, formally verifiable Deontic,
SMT, Automaton, Datalog, and Cellular-Sheaf rules, while providing deep system
forensics, AST vulnerability scanning, shell command firewalling, enterprise
DLP, tool-call capability boundaries, cryptographic canaries, LTL formal trace
verification, circuit breakers, nanosecond profiling, a 3-level parallel
subagent swarm, and the 6-Layer OmniGuard Sovereign Master Controller.

Organized in 88 submodules across 18 layers:
  Core parse/store          — ruleextract, rulestore
  Enforcement gate          — gate, scoring, responseanaly, detect, injection, ledger, seal
  Contracts & policy        — canon (PromptVault), policy, attestation, contracts
  Repair & adversarial      — repair, fuzz, mutate, probes, redteam, shadow, chaos, selfrefine
  Semantics & drift         — semantic, semantics, metaguard, critic, cognify, drift
  Streaming & consensus     — stream, consensus, orchestrator, warroom, byzantine
  Observability & packing   — telemetry, dashboard, forge, report, cli, sandbox, forensics, benchsuite
  Obfuscation & multiturn   — cipherlab (decode), multiturn (coercion tracker), compactguard
  Trust & governance        — trust (adaptive trust scoring), bayesian, gameguard, counterfactual
  Foresight & shield        — foresight (risk forecast), shield (few-shot shields), autohealer
  System & code security    — sysprobe, astguard, cmdguard, dlp, toolguard, canary, informationflow
  Formal & symbolic logic   — deontic, satisolver, smtguard, automata, sheafcheck, abstractinterp,
                              symbolicexec, datalog, petrinet, entropysense
  Attention & token steer   — attentionsteer, modelmatrix, grammarguard, sandwich, tokensteer, hierarchyguard
  Linguistic & discourse    — morpholex, rhetoric, citationguard, personaguard, refusalguard
  Cryptographic proofs      — proofcarry, zkseal, merkleforest, vectorvault
  Swarm, LTL & OmniGuard    — swarm, formalverify, circuit, profiler, sovereign, omniguard

Design rules inherited from the fullagent house style:
  * stdlib only, Python 3.9+,
  * dataclasses + small pure functions,
  * never raise on malformed input,
  * every module carries a `_self_test()` runnable via `python -m`.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

__all__ = [
    "ruleextract", "rulestore", "Rule", "PromptSpec",
    "extract_rules", "rules_digest", "RuleStore", "diff_specs",
    "gate", "scoring", "responseanaly", "detect", "injection",
    "ledger", "seal", "canon", "policy", "attestation",
    "repair", "fuzz", "mutate", "probes", "redteam", "shadow",
    "semantic", "semantics", "metaguard", "critic", "cognify", "drift",
    "stream", "consensus", "orchestrator", "warroom",
    "telemetry", "dashboard", "forge", "report", "cli", "sandbox",
    "cipherlab", "multiturn", "trust", "foresight", "shield",
    "swarm", "sysprobe", "astguard", "cmdguard", "dlp",
    "toolguard", "canary", "formalverify", "circuit", "profiler", "sovereign",
    "deontic", "attentionsteer", "modelmatrix", "grammarguard",
    "satisolver", "smtguard", "automata", "bayesian",
    "gameguard", "sheafcheck", "informationflow", "abstractinterp",
    "symbolicexec", "datalog", "petrinet", "contracts",
    "counterfactual", "entropysense", "morpholex", "rhetoric",
    "citationguard", "personaguard", "refusalguard", "hierarchyguard",
    "sandwich", "compactguard", "tokensteer", "selfrefine",
    "proofcarry", "zkseal", "merkleforest", "byzantine",
    "vectorvault", "chaos", "forensics", "autohealer",
    "benchsuite", "omniguard", "deepkernel",
    "temporalctl", "epistemic", "intuitionistic", "fuzzyguard", "paraconsistent",
    "linearlogic", "separationlogic", "sessiontypes", "refineguard", "choreography",
    "provenance", "egraph", "cegar", "hornclause", "ilpsolver",
    "maxflowguard", "earleyparser", "packrat", "ssaguard", "infogeom",
    "cusum", "conformal", "dempstershafer", "diffprivacy", "latticecrypto",
    "reedsolomon", "spectralgraph", "kolmogorov", "metamorphic", "hypervisor",
    "omniguard_prepare_wire", "omniguard_enforce_output",
    "__version__",
]

__version__ = "4.0.0"

_LAZY_MODULES = (
    "ruleextract", "rulestore", "telemetry", "dashboard",
    "forge", "probes", "fuzz", "repair", "gate", "scoring",
    "responseanaly", "detect", "injection", "ledger", "seal",
    "canon", "policy", "attestation", "mutate", "redteam", "shadow",
    "semantic", "semantics", "metaguard", "critic", "cognify", "drift",
    "stream", "consensus", "orchestrator", "warroom", "report", "cli",
    "sandbox", "cipherlab", "multiturn", "trust", "foresight", "shield",
    "swarm", "sysprobe", "astguard", "cmdguard", "dlp",
    "toolguard", "canary", "formalverify", "circuit", "profiler", "sovereign",
    "deontic", "attentionsteer", "modelmatrix", "grammarguard",
    "satisolver", "smtguard", "automata", "bayesian",
    "gameguard", "sheafcheck", "informationflow", "abstractinterp",
    "symbolicexec", "datalog", "petrinet", "contracts",
    "counterfactual", "entropysense", "morpholex", "rhetoric",
    "citationguard", "personaguard", "refusalguard", "hierarchyguard",
    "sandwich", "compactguard", "tokensteer", "selfrefine",
    "proofcarry", "zkseal", "merkleforest", "byzantine",
    "vectorvault", "chaos", "forensics", "autohealer",
    "benchsuite", "omniguard", "deepkernel",
    "temporalctl", "epistemic", "intuitionistic", "fuzzyguard", "paraconsistent",
    "linearlogic", "separationlogic", "sessiontypes", "refineguard", "choreography",
    "provenance", "egraph", "cegar", "hornclause", "ilpsolver",
    "maxflowguard", "earleyparser", "packrat", "ssaguard", "infogeom",
    "cusum", "conformal", "dempstershafer", "diffprivacy", "latticecrypto",
    "reedsolomon", "spectralgraph", "kolmogorov", "metamorphic", "hypervisor",
)


class _DirectStatFinder(importlib.abc.MetaPathFinder):
    """Fallback finder for 9p/FUSE filesystems where os.listdir() truncates
    directories with >40 entries but os.path.isfile() (stat) succeeds."""

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("fullagent."):
            return None
        parts = fullname.split(".")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand_py = os.path.join(base_dir, *parts[1:]) + ".py"
        if os.path.isfile(cand_py):
            return importlib.util.spec_from_file_location(fullname, cand_py)
        return None


if not any(type(f).__name__ == "_DirectStatFinder" for f in sys.meta_path):
    sys.meta_path.append(_DirectStatFinder())


def omniguard_prepare_wire(system_prompt: str, messages, *, model_id: str = "universal"):
    """Prepare a wire-ready message list and token-steering bundle via OmniGuardController."""
    from . import omniguard as _og
    return _og.OmniGuardController().prepare_wire(system_prompt, messages, model_id=model_id)


def omniguard_enforce_output(response_text: str, *, system_prompt: str = "", user_input: str = ""):
    """Enforce, auto-repair, and cryptographically certify a response via OmniGuardController."""
    from . import omniguard as _og
    return _og.OmniGuardController().enforce_output(
        response_text, system_prompt=system_prompt, user_input=user_input
    )


def __getattr__(name: str):
    """PEP 562 lazy attribute access — import siblings on demand."""
    if name in _LAZY_MODULES:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in ("Rule", "PromptSpec", "extract_rules", "rules_digest"):
        from . import ruleextract as _m
        return getattr(_m, name)
    if name in ("RuleStore", "diff_specs"):
        from . import rulestore as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _self_test() -> int:
    """Run every sibling's self-test; a missing sibling is skipped."""
    import importlib
    failures = 0
    for name in _LAZY_MODULES:
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"ENFORCEx SKIP {name}: {exc}")
            continue
        fn = getattr(mod, "_self_test", None)
        if callable(fn):
            try:
                rv = fn()
                if rv is not None and rv != 0:
                    print(f"ENFORCEx FAIL {name}: returned {rv!r}")
                    failures += 1
            except AssertionError as exc:
                print(f"ENFORCEx FAIL {name}: {exc}")
                failures += 1
            except Exception as exc:
                print(f"ENFORCEx FAIL {name} ({type(exc).__name__}): {exc}")
                failures += 1
    if failures:
        print(f"ENFORCEx PACKAGE SELF-TEST FAIL ({failures} modules failed)")
        return 1
    print(f"ENFORCEx PACKAGE SELF-TEST PASS ({len(_LAZY_MODULES)} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

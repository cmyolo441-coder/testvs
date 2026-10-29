"""Run every fullagent and fullagent.enforcex module's __main__ self-test in isolated subprocesses."""
import ast
import concurrent.futures
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
skip = {"__init__", "__main__"}

# Explicit list of enforcex modules so even on 9p mounts where glob/listdir
# truncates directories >40 files, every single enforcex module is tested.
ENFORCEX_MODS = (
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


def _has_main_block(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and any(
            isinstance(value, ast.Constant) and value.value == "__main__"
            for value in node.comparators
        )
        for node in ast.walk(tree)
    )


def _run_one(m: str) -> tuple[str, bool, str]:
    safe_name = m.replace(".", "-")
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"fullagent-selftest-{safe_name}-"
        ) as data_home:
            env = os.environ.copy()
            env["FULLAGENT_HOME"] = data_home
            env["PYTHONPATH"] = str(PKG) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            r = subprocess.run(
                [sys.executable, "-m", f"fullagent.{m}"],
                capture_output=True,
                text=True,
                cwd=str(PKG),
                env=env,
                timeout=300,
            )
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or e.stderr or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        tail = partial.strip().splitlines()
        detail = tail[-1][:80] if tail else "no output before timeout"
        return m, False, f"timed out after 300s: {detail}"

    out = (r.stdout + r.stderr).strip().splitlines()
    tail = out[-1] if out else "(no output)"
    marker = r.stdout + r.stderr
    ok = r.returncode == 0 and (
        "PASS" in marker or "passed" in marker.lower() or "OK" in marker
    )
    detail = tail
    if not ok:
        detail = next(
            (line.strip() for line in reversed(out) if "FAIL" in line.upper()),
            tail,
        )
    return m, ok, detail[:110]


def main() -> int:
    t0 = time.monotonic()
    mods = sorted(
        path.stem
        for path in (PKG / "fullagent").glob("*.py")
        if path.stem not in skip and _has_main_block(path)
    )
    for em in sorted(ENFORCEX_MODS):
        epath = PKG / "fullagent" / "enforcex" / f"{em}.py"
        if epath.is_file() and _has_main_block(epath):
            mods.append(f"enforcex.{em}")

    workers = max(1, min(8, int(os.environ.get("FULLAGENT_SELFTEST_JOBS", "6"))))
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for m, ok, detail in pool.map(_run_one, mods):
            status = "PASS" if ok else "FAIL"
            print(f"{status:>4}  {m:<24} {detail}", flush=True)
            if not ok:
                failed.append(m)

    elapsed = time.monotonic() - t0
    print()
    if failed:
        print(f"{len(failed)} FAILED ({elapsed:.1f}s): {', '.join(failed)}")
        return 1
    print(f"ALL {len(mods)} MODULE SELF-TESTS PASS ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

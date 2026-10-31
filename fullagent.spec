# -*- mode: python ; coding: utf-8 -*-
import os
import pathlib
import sys

# collect_all/collect_submodules IMPORT the package while the spec runs.
# CI's pyinstaller process did not have the repo root on sys.path, so both
# helpers returned nothing (log: "module 'fullagent' ... is not a package")
# and every lazily-imported subsystem was left out of the binary — the
# standalone build then died with ModuleNotFoundError: fullagent.mastermind.
_ROOT = globals().get("SPECPATH") or os.getcwd()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Build-time embedded provider keys: generate the git-ignored
# fullagent/_buildkeys.py from *_API_KEY env vars (CI feeds repository
# secrets; a local build can do `NVIDIA_API_KEY=nvapi-… pyinstaller
# fullagent.spec`). With no such vars the previous file is left untouched.
try:
    import gen_buildkeys
    _embedded = gen_buildkeys.main()
    print(f"[fullagent.spec] embedded keys for: {sorted(_embedded)}"
          if _embedded else
          "[fullagent.spec] no *_API_KEY env vars — building without embedded keys")
except Exception as _exc:  # never fail the build over key embedding
    print(f"[fullagent.spec] key embedding skipped: {_exc!r}")

# every fullagent subpackage — systemprompt.py and others importlib them
# dynamically, so PyInstaller cannot see them on its own
datas, binaries, hiddenimports = collect_all('fullagent')
hiddenimports += collect_submodules('fullagent')

# Belt and braces: enumerate the package files without importing anything,
# so a single import-time error can never silently drop modules again.
_pkg_dir = pathlib.Path(_ROOT) / 'fullagent'
hiddenimports += [f'fullagent.{p.stem}' for p in _pkg_dir.glob('*.py')
                  if p.stem != '__init__']
_enfx_dir = _pkg_dir / 'enforcex'
_ENFX_KNOWN = (
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
if _enfx_dir.is_dir():
    hiddenimports += ['fullagent.enforcex']
    hiddenimports += [f'fullagent.enforcex.{p.stem}' for p in _enfx_dir.glob('*.py')
                      if p.stem != '__init__']
    hiddenimports += [f'fullagent.enforcex.{m}' for m in _ENFX_KNOWN]
hiddenimports = sorted(set(hiddenimports))


a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fullagent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

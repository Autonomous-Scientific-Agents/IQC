#!/usr/bin/env bash
# Smoke test for the IQC Docker images.
#
# Each image target declares which calculators it is expected to ship. The
# test FAILS when one of those is missing or broken, and only SKIPs the ones
# the target intentionally excludes — an earlier version inferred the expected
# set from whichever imports happened to succeed, so a bundled dependency with
# a broken import (e.g. a missing shared library) was silently reported as
# "not in this image" and the run still passed.
#
# Every calculator check also validates the persisted result row. `iqc` exits 0
# after recording a per-molecule failure, so exit status alone does not show
# that a calculator worked.
#
#   docker run --rm iqc:base    bash /opt/iqc/scripts/docker_smoke_test.sh
#   docker run --rm iqc:full    bash /opt/iqc/scripts/docker_smoke_test.sh
#
# The expected set comes from $IQC_IMAGE_TARGET (baked into each image).
# Override for ad-hoc runs:
#   IQC_SMOKE_REQUIRE="xtb pyscf" bash docker_smoke_test.sh
#   bash docker_smoke_test.sh xtb pyscf
set -uo pipefail

WORK="$(mktemp -d)"; cd "$WORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Serial for this script only (NOT in the image environment, which must keep
# MPI auto-detection so `mpiexec -n N iqc ...` still distributes work). This
# keeps each check to exactly one result row.
export IQC_DISABLE_MPI=1

PASS=(); FAILREQ=(); FAILOPT=(); SKIP=()
ok()      { echo "  [PASS] $1"; PASS+=("$1"); }
bad_req() { echo "  [FAIL] $1"; FAILREQ+=("$1"); }
bad_opt() { echo "  [WARN] $1 (optional)"; FAILOPT+=("$1"); }
skip()    { echo "  [SKIP] $1 (not in this image)"; SKIP+=("$1"); }
hr() { echo "----------------------------------------------------------------"; }

# --- expected calculator set per image target -------------------------------
TARGET="${IQC_IMAGE_TARGET:-unknown}"
if [ "$#" -gt 0 ]; then
  REQUIRED="$*"
elif [ -n "${IQC_SMOKE_REQUIRE:-}" ]; then
  REQUIRED="$IQC_SMOKE_REQUIRE"
else
  case "$TARGET" in
    base)              REQUIRED="xtb pyscf nwchem" ;;
    mlip)              REQUIRED="xtb pyscf nwchem mace fairchem" ;;
    exachem)           REQUIRED="xtb pyscf nwchem exachem" ;;
    full|full-conda)   REQUIRED="xtb pyscf nwchem exachem mace fairchem" ;;
    *)
      echo "ERROR: unknown image target '${TARGET}'." >&2
      echo "Set IQC_IMAGE_TARGET in the image, or pass the expected" >&2
      echo "calculators as arguments / IQC_SMOKE_REQUIRE." >&2
      exit 2 ;;
  esac
fi
requires() { case " $REQUIRED " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- capability detection ---------------------------------------------------
# Distinguishes "not installed" from "installed but broken"; a broken required
# dependency is a failure, never a skip.
probe_py() {  # module -> prints ok|missing|broken:<msg>
  python - "$1" <<'PY'
import importlib, importlib.util, sys
mod = sys.argv[1]
try:
    spec = importlib.util.find_spec(mod)
except ModuleNotFoundError as exc:
    # find_spec imports parent packages. A genuinely absent parent (e.g.
    # "fairchem" for "fairchem.core") means the capability is not installed;
    # anything else raised while importing a parent means it is broken.
    if exc.name and (mod == exc.name or mod.startswith(exc.name + ".")):
        print("missing")
    else:
        print(f"broken:{exc}")
    sys.exit(0)
except Exception as exc:                      # parent package itself is broken
    print(f"broken:{exc}"); sys.exit(0)
if spec is None:
    print("missing"); sys.exit(0)
try:
    importlib.import_module(mod); print("ok")
except Exception as exc:
    print(f"broken:{exc}")
PY
}

declare -A STATE
probe_capability() {  # name  kind  locator
  local name="$1" kind="$2" loc="$3" st
  case "$kind" in
    py)  st="$(probe_py "$loc")" ;;
    bin) if command -v "$loc" >/dev/null 2>&1; then st=ok; else st=missing; fi ;;
    exe) if [ -n "${loc}" ] && [ -x "${loc}" ]; then st=ok; else st=missing; fi ;;
  esac
  STATE[$name]="$st"
}

probe_capability xtb      py  xtb
probe_capability pyscf    py  pyscf
probe_capability mace     py  mace
probe_capability fairchem py  fairchem.core
probe_capability torch    py  torch
probe_capability nwchem   bin nwchem
probe_capability exachem  exe "${EXACHEM_BINARY:-}"

# Gate a capability. Returns 0 when the check that follows should run.
gate() {  # name
  local name="$1" st="${STATE[$1]}"
  case "$st" in
    ok) return 0 ;;
    missing)
      if requires "$name"; then
        bad_req "$name is required by image target '${TARGET}' but is not installed"
      else
        skip "$name"
      fi
      return 1 ;;
    broken:*)
      # Installed but unusable — surface it even when not required.
      if requires "$name"; then
        bad_req "$name is required by target '${TARGET}' but is broken: ${st#broken:}"
      else
        bad_opt "$name is installed but broken: ${st#broken:}"
      fi
      return 1 ;;
  esac
}

# --- result-row validation --------------------------------------------------
validate_row() {  # dir task
  python - "$1" "$2" <<'PY'
import glob, json, math, sys
d, task = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{d}/iqc_*_results_*.jsonl"))
if not files:
    print("no result JSONL was written"); sys.exit(1)
rows = [json.loads(l) for l in open(files[-1]) if l.strip()]
if len(rows) != 1:
    print(f"expected exactly 1 result row, got {len(rows)}"); sys.exit(1)
row = rows[0]
for k, v in row.items():
    if isinstance(k, str) and (k == "error" or k.endswith("_error")) and v:
        print(f"result row records {k}: {str(v)[:200]}"); sys.exit(1)
REQUIRED_FIELDS = {
    "single": ["energy_eV"],
    "opt":    ["opt_energy_eV"],
    "thermo": ["opt_energy_eV", "E_ZPE_eV", "H_eV", "G_eV", "S_eV/K"],
}
for k in REQUIRED_FIELDS[task]:
    v = row.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        print(f"{k} missing or non-finite: {v!r}"); sys.exit(1)
if task in ("opt", "thermo") and not row.get("opt_converged"):
    print(f"optimization did not converge (opt_converged={row.get('opt_converged')!r})")
    sys.exit(1)
print(", ".join(f"{k}={row[k]:.6g}" for k in REQUIRED_FIELDS[task]))
PY
}

run_cli() {  # label  task  extra-args...
  local label="$1" task="$2"; shift 2
  local d="run_${label}" msg
  mkdir -p "$d"
  if ! (cd "$d" && iqc --smiles O --task "$task" "$@") >"${d}.log" 2>&1; then
    tail -20 "${d}.log"; return 1
  fi
  if msg="$(validate_row "$d" "$task")"; then
    ok "CLI ${label} (${msg})"; return 0
  fi
  echo "    ${msg}"; tail -10 "${d}.log"; return 1
}

hr; echo "IQC Docker smoke test"; hr
echo "image target : ${TARGET}"
echo "required     : ${REQUIRED}"
echo
echo "== Versions / capabilities =="
python -c 'import iqc; print("iqc         ", iqc.__file__)' || true
python -c 'import ase;   print("ase         ", ase.__version__)'
python -c 'import rdkit; print("rdkit       ", rdkit.__version__)'
for c in xtb pyscf torch mace fairchem nwchem exachem; do
  printf "%-12s %s\n" "$c" "${STATE[$c]}"
done
echo "mpiexec      $(command -v mpiexec || echo MISSING)"

# ---------------------------------------------------------------------------
hr; echo "== 1. pytest suite =="
# Two ExaChem tests assume an environment where ExaChem is NOT installed (one
# hardcodes /usr/bin/true as a stand-in binary; the other asserts a "Forces"
# warning that does not apply when the SCF run succeeds). On images without
# ExaChem the whole module is skipped: it needs the binary.
PYTEST_ARGS=(-q iqc/tests
  --deselect "iqc/tests/test_exachem.py::test_build_command_injects_nproc_when_missing"
  --deselect "iqc/tests/test_exachem.py::test_exachem_run_single_point_integration")
[ "${STATE[exachem]}" = ok ] || PYTEST_ARGS+=(--ignore=iqc/tests/test_exachem.py)
if (cd /opt/iqc && python -m pytest "${PYTEST_ARGS[@]}") >/tmp/pytest.log 2>&1; then
  ok "pytest ($(grep -Eo '[0-9]+ passed' /tmp/pytest.log | head -1))"
else
  tail -30 /tmp/pytest.log; bad_req "pytest"
fi

hr; echo "== 2. EMT (always bundled: ASE built-in) =="
run_cli emt single --calculator emt || bad_req "EMT single"

hr; echo "== 3. xTB (GFN2: single + opt + thermo) =="
if gate xtb; then
  run_cli xtb-single single --calculator xtb || bad_req "xTB single"
  run_cli xtb-opt    opt    --calculator xtb || bad_req "xTB opt"
  run_cli xtb-thermo thermo --calculator xtb || bad_req "xTB thermo"
fi

hr; echo "== 4. PySCF (DFT single point) =="
if gate pyscf; then
  cat > "$WORK/pyscf.yaml" <<'YAML'
calculator: pyscf
calculator_params:
  method: dft
  xc: pbe
  basis: sto-3g
YAML
  run_cli pyscf single --params "$WORK/pyscf.yaml" || bad_req "PySCF single"
fi

hr; echo "== 5. ExaChem (SCF/HF) =="
if gate exachem; then
  cat > "$WORK/exachem.yaml" <<'YAML'
calculator: exachem
calculator_params:
  method: scf
  basis: sto-3g
  nproc: 1
YAML
  run_cli exachem single --calculator exachem --params "$WORK/exachem.yaml" \
    || bad_req "ExaChem SCF"
fi

hr; echo "== 6. NWChem (ASE calculator, DFT single point) =="
if gate nwchem; then
  if python - <<'PY' >nwchem.log 2>&1
from ase.build import molecule
from ase.calculators.nwchem import NWChem
import math
a = molecule("H2O")
a.calc = NWChem(theory="dft", dft=dict(xc="pbe96"), basis="3-21G", label="nwtest")
e = a.get_potential_energy()
print("NWChem energy (eV):", e)
assert math.isfinite(e) and e < 0, e
PY
  then ok "NWChem DFT"; else tail -20 nwchem.log; bad_req "NWChem DFT"; fi
fi

hr; echo "== 7. MACE (library required where bundled; model download optional) =="
if gate mace; then
  # The library must import (checked by gate). Running it pulls a foundation
  # checkpoint over the network, so a failure here stays advisory.
  run_cli mace single --calculator mace || bad_opt "MACE single (needs model download)"
fi

hr; echo "== 8. FAIRChem UMA (library required where bundled; checkpoint gated) =="
if gate fairchem; then
  ok "fairchem-core import"
fi

# ---------------------------------------------------------------------------
hr; echo "SUMMARY"; hr
echo "target: ${TARGET}   required: ${REQUIRED}"
echo "PASSED (${#PASS[@]}): ${PASS[*]}"
[ ${#SKIP[@]}    -gt 0 ] && echo "SKIPPED (${#SKIP[@]}): ${SKIP[*]}"
[ ${#FAILOPT[@]} -gt 0 ] && echo "OPTIONAL FAILURES (${#FAILOPT[@]}): ${FAILOPT[*]}"
if [ ${#FAILREQ[@]} -gt 0 ]; then
  echo "REQUIRED FAILURES (${#FAILREQ[@]}): ${FAILREQ[*]}"
  echo "RESULT: FAIL"; exit 1
fi
echo "RESULT: PASS"

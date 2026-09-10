#!/usr/bin/env bash
# Capability-aware smoke test for the IQC Docker images.
#
# Works for every image in the family (base / mlip / exachem / full) and for
# the conda monolith (iqc:full): it auto-detects which calculators are present
# and only *requires* the ones this image ships. Absent calculators are marked
# SKIP, not FAIL. Meant to run inside the container.
#
#   docker run --rm iqc:base    bash /opt/iqc/scripts/docker_smoke_test.sh
#   docker run --rm iqc:mlip    bash /opt/iqc/scripts/docker_smoke_test.sh
#   docker run --rm iqc:exachem bash /opt/iqc/scripts/docker_smoke_test.sh
#   docker run --rm iqc:full-uv bash /opt/iqc/scripts/docker_smoke_test.sh
set -uo pipefail

WORK="$(mktemp -d)"; cd "$WORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IQC_DISABLE_MPI=1

PASS=(); FAILREQ=(); FAILOPT=(); SKIP=()
ok()      { echo "  [PASS] $1"; PASS+=("$1"); }
bad_req() { echo "  [FAIL] $1"; FAILREQ+=("$1"); }
bad_opt() { echo "  [WARN] $1 (optional)"; FAILOPT+=("$1"); }
skip()    { echo "  [SKIP] $1 (not in this image)"; SKIP+=("$1"); }
hr() { echo "----------------------------------------------------------------"; }

have_py() { python -c "import $1" >/dev/null 2>&1; }

# --- capability detection ---------------------------------------------------
HAS_XTB=false;    have_py xtb        && HAS_XTB=true
HAS_PYSCF=false;  have_py pyscf      && HAS_PYSCF=true
HAS_TORCH=false;  have_py torch      && HAS_TORCH=true
HAS_MACE=false;   have_py mace       && HAS_MACE=true
HAS_FAIRCHEM=false; have_py fairchem.core && HAS_FAIRCHEM=true
HAS_NWCHEM=false; command -v nwchem >/dev/null 2>&1 && HAS_NWCHEM=true
HAS_EXACHEM=false; [ -x "${EXACHEM_BINARY:-}" ] && HAS_EXACHEM=true

hr; echo "IQC Docker smoke test"; hr
echo "== Versions / capabilities =="
python -c 'import iqc; print("iqc         ", iqc.__file__)' || true
python -c 'import ase;   print("ase         ", ase.__version__)'
python -c 'import rdkit; print("rdkit       ", rdkit.__version__)'
$HAS_PYSCF    && python -c 'import pyscf; print("pyscf       ", pyscf.__version__)'
$HAS_XTB      && echo "xtb-python   present"
$HAS_TORCH    && python -c 'import torch; print("torch       ", torch.__version__)'
$HAS_MACE     && echo "mace         present"
$HAS_FAIRCHEM && echo "fairchem     present"
echo "nwchem       $(command -v nwchem || echo MISSING)"
echo "ExaChem      ${EXACHEM_BINARY:-unset} ($($HAS_EXACHEM && echo present || echo MISSING))"
echo "mpiexec      $(command -v mpiexec || echo MISSING)"

# ---------------------------------------------------------------------------
hr; echo "== 1. pytest suite =="
# Two ExaChem tests assume an environment where ExaChem is NOT installed
# (one hardcodes /usr/bin/true as a stand-in binary absent from slim images;
# the other asserts a "Forces" warning that does not apply when the SCF run
# succeeds). They are deselected. On images without ExaChem, the whole
# test_exachem.py module is skipped (it needs the binary for its integration
# tests).
PYTEST_ARGS=(-q iqc/tests
  --deselect "iqc/tests/test_exachem.py::test_build_command_injects_nproc_when_missing"
  --deselect "iqc/tests/test_exachem.py::test_exachem_run_single_point_integration")
$HAS_EXACHEM || PYTEST_ARGS+=(--ignore=iqc/tests/test_exachem.py)
if (cd /opt/iqc && python -m pytest "${PYTEST_ARGS[@]}") >/tmp/pytest.log 2>&1; then
  ok "pytest ($(grep -Eo '[0-9]+ passed' /tmp/pytest.log | head -1))"
else
  tail -30 /tmp/pytest.log; bad_req "pytest"
fi

# ---------------------------------------------------------------------------
run_cli() {  # name  extra-args...
  local name="$1"; shift
  local out="cli_${name}.log"
  if iqc --smiles O "$@" >"$out" 2>&1; then ok "CLI $name"; else tail -20 "$out"; return 1; fi
}

hr; echo "== 2. EMT (built-in, single point) =="
run_cli emt --task single --calculator emt || bad_req "EMT single"

hr; echo "== 3. xTB (GFN2: single + opt + thermo) =="
if $HAS_XTB; then
  run_cli xtb-single --task single --calculator xtb || bad_req "xTB single"
  run_cli xtb-opt    --task opt    --calculator xtb || bad_req "xTB opt"
  run_cli xtb-thermo --task thermo --calculator xtb || bad_req "xTB thermo"
else skip "xTB"; fi

hr; echo "== 4. PySCF (DFT, via --params) =="
if $HAS_PYSCF; then
  cat > pyscf.yaml <<'YAML'
calculator: pyscf
calculator_params:
  method: dft
  xc: pbe
  basis: sto-3g
YAML
  run_cli pyscf --task single --params pyscf.yaml || bad_req "PySCF single"
else skip "PySCF"; fi

hr; echo "== 5. ExaChem (SCF/HF, via CLI) =="
if $HAS_EXACHEM; then
  cat > exachem.yaml <<'YAML'
calculator: exachem
calculator_params:
  method: scf
  basis: sto-3g
  nproc: 1
YAML
  run_cli exachem --task single --calculator exachem --params exachem.yaml || bad_req "ExaChem SCF"
else skip "ExaChem"; fi

hr; echo "== 6. NWChem (ASE calculator, DFT single point) =="
if $HAS_NWCHEM; then
  if python - <<'PY' >nwchem.log 2>&1
from ase.build import molecule
from ase.calculators.nwchem import NWChem
a = molecule("H2O")
a.calc = NWChem(theory="dft", dft=dict(xc="pbe96"), basis="3-21G", label="nwtest")
e = a.get_potential_energy()
print("NWChem energy (eV):", e); assert e < 0
PY
  then ok "NWChem DFT"; else tail -20 nwchem.log; bad_req "NWChem DFT"; fi
else skip "NWChem"; fi

hr; echo "== 7. MACE (MLIP, downloads model — OPTIONAL) =="
if $HAS_MACE; then
  run_cli mace --task single --calculator mace || bad_opt "MACE single"
else skip "MACE"; fi

hr; echo "== 8. FAIRChem UMA import (model is gated — OPTIONAL) =="
if $HAS_FAIRCHEM; then
  if python -c 'import fairchem.core; print("fairchem-core import OK")' 2>/tmp/uma.log; then
    ok "fairchem-core import"
  else tail -5 /tmp/uma.log; bad_opt "fairchem-core import"; fi
else skip "FAIRChem UMA"; fi

# ---------------------------------------------------------------------------
hr; echo "SUMMARY"; hr
echo "PASSED (${#PASS[@]}): ${PASS[*]}"
[ ${#SKIP[@]}    -gt 0 ] && echo "SKIPPED (${#SKIP[@]}): ${SKIP[*]}"
[ ${#FAILOPT[@]} -gt 0 ] && echo "OPTIONAL FAILURES (${#FAILOPT[@]}): ${FAILOPT[*]}"
if [ ${#FAILREQ[@]} -gt 0 ]; then
  echo "REQUIRED FAILURES (${#FAILREQ[@]}): ${FAILREQ[*]}"
  echo "RESULT: FAIL"; exit 1
fi
echo "RESULT: PASS"

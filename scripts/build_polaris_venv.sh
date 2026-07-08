#!/bin/bash -l
# =============================================================================
# build_polaris_venv.sh — Build a SHARABLE IQC virtual environment on ALCF Polaris
# =============================================================================
# Polaris = AMD EPYC Milan (32c/64t) + 4x NVIDIA A100-40GB per node,
# PrgEnv-nvidia + cray-mpich, CUDA 12.x.
#
# Unlike Crux (CPU-only, uv-managed standalone CPython), Polaris ships a curated
# `conda` module whose base env already contains:
#   - a CUDA build of PyTorch (torch 2.8 / cu129) that works on the A100s
#   - mpi4py built against cray-mpich (GPU/Slingshot aware)
# so the recommended pattern (ALCF docs: polaris/data-science/python.md) is to
# create a venv ON TOP of that conda base with --system-site-packages and only
# pip-install the extra packages IQC needs. That keeps the fast, correct GPU
# torch + Cray mpi4py and avoids rebuilding either.
#
# Run this ONCE on a Polaris LOGIN node (login nodes have outbound internet via
# the ALCF proxy; compute nodes do not).
#
#   bash scripts/build_polaris_venv.sh
#
# Then anyone in the IQC group activates it with:
#   module use /soft/modulefiles && module load conda && conda activate base
#   source /lus/eagle/projects/IQC/shared/venvs/iqc-polaris-gpu-py312/bin/activate
# =============================================================================
set -euo pipefail

# ---- shared locations -------------------------------------------------------
export IQC_REPO="${IQC_REPO:-/home/keceli/IQC/keceli/ASA/IQC}"
SHARED_ROOT="${SHARED_ROOT:-/lus/eagle/projects/IQC/shared}"
export IQC_VENV="${IQC_VENV:-$SHARED_ROOT/venvs/iqc-polaris-gpu-py312}"

# ---- thread caps: avoid OpenBLAS "pthread_create failed" at import time ------
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---- isolate from the conda module's per-user site (~/.local/.../conda) ------
# The `conda` module puts a writable user site-packages (PYTHONUSERBASE under
# ~/.local) on sys.path. With --system-site-packages that leaks into the venv,
# and pip would treat packages already there (ase, pymatgen, fairchem, ...) as
# "satisfied" and NOT install them into the venv — leaving a venv that only
# works while ~/.local happens to be intact. Disabling the user site during the
# build forces those packages INTO the venv (torch/mpi4py still come from the
# conda BASE system-site, which PYTHONNOUSERSITE does not touch). Runtime must
# set this too — the submit scripts export it.
export PYTHONNOUSERSITE=1

# ---- 1. base conda (GPU torch + cray mpi4py live here) -----------------------
module use /soft/modulefiles
module load conda
conda activate base

echo "== IQC Polaris venv build =="
echo "repo       : $IQC_REPO"
echo "venv       : $IQC_VENV"
echo "conda base : $CONDA_PREFIX"
echo "python     : $(command -v python)  ($(python -V 2>&1))"
python -c "import torch; print('base torch :', torch.__version__, 'cuda', torch.version.cuda)"
python -c "from mpi4py import MPI; print('base mpi4py:', MPI.Get_library_version().splitlines()[0])"

mkdir -p "$SHARED_ROOT/venvs"

# ---- 2. venv on top of conda base, inheriting its site-packages --------------
#        --system-site-packages => torch (GPU) + mpi4py (cray) come from conda;
#        pip installs land in the venv and never re-resolve those two.
python -m venv --system-site-packages "$IQC_VENV"
# shellcheck disable=SC1091
source "$IQC_VENV/bin/activate"
python -V
python -m pip install --upgrade pip setuptools wheel

# ---- 3. constraints: pin torch/mpi4py so nothing pulls a PyPI CUDA rebuild ---
#        (fairchem-core re-resolves torch; without this, PyPI's wheel would
#        shadow the conda GPU build inside the venv.)
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
CONSTRAINTS="$(mktemp)"
cat > "$CONSTRAINTS" <<EOF
torch==${TORCH_VER}
EOF
echo "constraints: torch==${TORCH_VER} (from conda base)"

# ---- 4. IQC + MLIP + test/parsl/mcp extras + PySCF ---------------------------
#        torch/mpi4py already satisfied by the inherited base env, so they are
#        NOT reinstalled. PySCF (new IQC calculator) added here.
python -m pip install -c "$CONSTRAINTS" -e "$IQC_REPO[mlip,test,parsl,mcp]"
python -m pip install -c "$CONSTRAINTS" pyscf

# ---- 5. MACE LAST, no deps so its e3nn==0.4.4 pin can't downgrade e3nn -------
#        mace-torch 0.3.16 already ships the MACE-Polar model resolver.
python -m pip install --no-deps "mace-torch==0.3.16"

# ---- 5c. MACE-Polar runtime: the polar-1-* checkpoints deserialize classes
#         from `graph_longrange`, which MACE itself (mace/modules/extensions.py)
#         says to install from WillBaldwin0/graph_electrostatics. NOTE: this pins
#         e3nn==0.4.4 (MACE's native pin), which is INCOMPATIBLE with
#         fairchem-core/UMA (needs e3nn>=0.5). This env therefore supports
#         MACE + MACE-Polar but NOT UMA. Drop this step if you need UMA instead.
python -m pip install -c "$CONSTRAINTS" --no-build-isolation \
    "git+https://github.com/WillBaldwin0/graph_electrostatics.git"

# ---- 6. sanity: torch must still be the conda GPU build, not a PyPI rebuild --
python - <<'PY'
import torch, os
loc = os.path.realpath(torch.__file__)
print("torch resolved from:", loc)
assert "+cpu" not in torch.__version__, "torch got downgraded to a CPU build!"
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

# ---- 7. make the whole tree group-writable & group-owned (sharable) ---------
chgrp -R IQC "$IQC_VENV" 2>/dev/null || true
chmod -R g+rwX "$IQC_VENV" 2>/dev/null || true

echo
echo "== verifying environment =="
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
def v(p):
    try: return version(p)
    except PackageNotFoundError: return "NOT INSTALLED"
import torch
print("torch        ", torch.__version__, "cuda build", torch.version.cuda)
for p in ("e3nn","mace-torch","fairchem-core","pyscf","ase","pymatgen","numpy"):
    print(f"{p:14}", v(p))
from mpi4py import MPI
print("mpi4py lib   ", MPI.Get_library_version().strip().splitlines()[0])
import mace
print("mace         ", getattr(mace, "__version__", "?"))
PY

rm -f "$CONSTRAINTS"
echo
echo "Done. Activate with:"
echo "  module use /soft/modulefiles && module load conda && conda activate base"
echo "  source $IQC_VENV/bin/activate"

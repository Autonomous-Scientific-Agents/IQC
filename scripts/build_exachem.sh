#!/usr/bin/env bash
# Build ExaChem (and its TAMM dependency) from source, CPU-only.
#
# Designed to run inside the IQC Docker image where the conda env "iqc-env"
# provides the compilers, OpenMPI, and CMake >= 3.26. TAMM auto-builds its
# heavy dependencies (BLIS, Libint, Libecpint, GlobalArrays, HDF5, Eigen),
# so no separate math library is needed.
#
# Result: ExaChem installed to $EXACHEM_PREFIX/bin/ExaChem
#
# Env knobs:
#   EXACHEM_PREFIX  install prefix           (default /opt/exachem)
#   EXACHEM_SRC     source/build scratch dir (default /opt/exachem-src)
#   TAMM_TAG        TAMM git tag/branch      (default main)
#   EXACHEM_TAG     ExaChem git tag/branch   (default main)
#   BUILD_JOBS      parallel build jobs      (default: nproc)
set -euo pipefail

EXACHEM_PREFIX="${EXACHEM_PREFIX:-/opt/exachem}"
EXACHEM_SRC="${EXACHEM_SRC:-/opt/exachem-src}"
TAMM_TAG="${TAMM_TAG:-main}"
EXACHEM_TAG="${EXACHEM_TAG:-main}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
INSTALL="${EXACHEM_PREFIX}/install"

# Compilers: prefer whatever conda activation exported ($CC/$CXX/$FC),
# fall back to the standard names. TAMM finds OpenMPI via find_package(MPI).
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export FC="${FC:-gfortran}"

# TAMM always builds a bundled numactl (its TargetMacros reference the
# numactl_External target unconditionally), bootstrapping it with autotools.
# conda's libtoolize is broken in this base image (hardcoded /usr/bin paths,
# misplaces ltmain.sh), so the image installs autoconf/automake/libtool via
# apt and keeps them OUT of the conda env — autoreconf/libtoolize then resolve
# to /usr/bin while gcc/mpicc/cmake still come from conda.
echo "    autoreconf : $(command -v autoreconf)"
echo "    libtoolize : $(command -v libtoolize)"

echo "==> ExaChem build configuration"
echo "    prefix : ${INSTALL}"
echo "    CC/CXX/FC : ${CC} / ${CXX} / ${FC}"
echo "    cmake  : $(cmake --version | head -1)"
echo "    mpicc  : $(command -v mpicc || echo 'not found')"
echo "    jobs   : ${BUILD_JOBS}"

mkdir -p "${EXACHEM_SRC}"
cd "${EXACHEM_SRC}"

# --- Step 1: TAMM -----------------------------------------------------------
if [ ! -d TAMM ]; then
  git clone --depth 1 --branch "${TAMM_TAG}" https://github.com/NWChemEx/TAMM.git
fi
cmake -S TAMM -B TAMM/build \
  -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
  -DMODULES="CC" \
  -DALLOW_CONDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build TAMM/build --target install -j "${BUILD_JOBS}"

# --- Step 2: ExaChem (must use the SAME configure line as TAMM) -------------
if [ ! -d exachem ]; then
  git clone --depth 1 --branch "${EXACHEM_TAG}" https://github.com/ExaChem/exachem.git
fi
cmake -S exachem -B exachem/build \
  -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
  -DMODULES="CC" \
  -DALLOW_CONDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build exachem/build --target install -j "${BUILD_JOBS}"

echo "==> ExaChem installed:"
ls -l "${INSTALL}/bin/ExaChem"

# =============================================================================
# IQC full-stack image (CPU-only)
#
# Bundles IQC with every open-source quantum-chemistry code and machine-learning
# interatomic potential it supports:
#
#   Quantum chemistry : xTB (GFN2), PySCF, NWChem, ExaChem (built from source)
#   ML potentials      : MACE, FAIRChem UMA, ASE EMT
#
# The entire toolchain (compilers + OpenMPI + CMake) and the QC codes come from
# conda-forge, so ExaChem, mpi4py, and NWChem all share ONE MPI — avoiding the
# classic "two MPIs in one env" breakage. PyTorch (CPU) and the MLIP stack are
# pip-installed in the documented order (torch first, then IQC[mlip], then
# mace-torch --no-deps).
#
# Proprietary codes IQC also supports (ORCA, Gaussian, VASP) cannot be
# redistributed and are NOT baked in. Wire them in at runtime via env vars /
# bind mounts (see docs/docker.md).
#
# Build:  docker build -t iqc:full .
# Run  :  docker run --rm -p 8888:8888 -v "$PWD":/work iqc:full          # Jupyter Lab
#         docker run --rm -v "$PWD":/work iqc:full iqc --smiles O --task thermo --calculator xtb
#         docker run --rm iqc:full bash /opt/iqc/scripts/docker_smoke_test.sh
# =============================================================================
FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

# Minimal system packages. The compilers, MPI, and CMake come from conda so the
# whole ExaChem/TAMM build uses one consistent toolchain. sed/grep/gawk/m4/make
# are required at /usr/bin because conda's libtoolize (used by TAMM's numactl
# autotools bootstrap) calls those tools by absolute path — the slim miniconda
# base image ships without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git wget curl patch file \
        sed grep gawk m4 make \
        autoconf automake libtool \
        libnuma-dev libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Fast dependency solving. Install mamba: conda 23.5's `env create` ignores the
# libmamba solver setting entirely and the classic solver is pathologically slow
# on this multi-code spec, so drive env creation with mamba instead.
RUN conda install -n base -y -c conda-forge mamba \
    && conda clean -afy

WORKDIR /opt/iqc

# -----------------------------------------------------------------------------
# 1) Conda environment: toolchain + OpenMPI + open-source QC codes (heavy, cached)
# -----------------------------------------------------------------------------
COPY env-docker.yml .
RUN mamba env create -f env-docker.yml && conda clean -afy

# Run every subsequent build step inside the env.
SHELL ["conda", "run", "--no-capture-output", "-n", "iqc-env", "/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# 2) ExaChem from source, against the conda toolchain (separate cached layer).
#    Build scratch is removed in the same layer so it never lands in the image.
# -----------------------------------------------------------------------------
ENV EXACHEM_PREFIX=/opt/exachem \
    EXACHEM_SRC=/opt/exachem-src \
    BUILD_JOBS=16
COPY scripts/build_exachem.sh /opt/iqc/scripts/build_exachem.sh
RUN chmod +x /opt/iqc/scripts/build_exachem.sh \
    && /opt/iqc/scripts/build_exachem.sh \
    && rm -rf /opt/exachem-src
ENV EXACHEM_BINARY=/opt/exachem/install/bin/ExaChem \
    PATH=/opt/exachem/install/bin:$PATH

# -----------------------------------------------------------------------------
# 3) IQC + PyTorch (CPU) + MLIP stack (MACE / FAIRChem UMA).
#    The PyTorch CPU index is passed as an *extra* index for the whole mlip
#    resolve so that whatever torch version fairchem-core pins resolves to its
#    `+cpu` build (which sorts above the plain PyPI build and carries no
#    nvidia-*-cu12 deps). Without this, fairchem downgrades torch to a CUDA
#    build and drags in ~6 GB of NVIDIA wheels onto this CPU-only image.
# -----------------------------------------------------------------------------
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# PyPI manylinux wheels (numpy, etc.) are built with newer GCC and need
# GLIBCXX_3.4.29+, which the old Debian-bullseye system libstdc++ lacks.
# `conda run` does not put the env's lib dir on the loader path, so point
# LD_LIBRARY_PATH at conda's newer libstdc++ (from libstdcxx-ng / gcc 14).
ENV LD_LIBRARY_PATH=/opt/conda/envs/iqc-env/lib:/opt/conda/lib
COPY . /opt/iqc
RUN pip install --no-cache-dir -e ".[mlip,parsl,mcp,test]" \
    && pip install --no-cache-dir --no-deps mace-torch \
    && python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__; print('torch', torch.__version__)" \
    && python -m ipykernel install --name iqc-env --display-name "Python (IQC)"

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------
# Single-thread BLAS/OMP by default (safe on shared hosts, avoids
# pthread_create storms); writable HF cache.
#
# IQC_DISABLE_MPI is deliberately NOT set: it short-circuits
# iqc.mpitools.should_initialize_mpi ahead of the launcher-variable check, so
# `mpiexec -n 4 iqc ...` would start four processes that each see rank 0 /
# size 1 and redo the whole input. IQC already runs serial when no launcher
# variables are present, so one-shot CLI runs need no forced disable.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    IQC_IMAGE_TARGET=full-conda \
    HF_HOME=/opt/hf-cache \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# OpenMPI's prterun refuses to launch as root (the container's default user).
# ExaChem is driven via `mpiexec -n N`, so allow root MPI in-container.

RUN mkdir -p /opt/hf-cache /work && chmod -R 777 /opt/hf-cache /work
WORKDIR /work

EXPOSE 8888

# `docker run iqc:full` -> Jupyter Lab. `docker run iqc:full <cmd>` runs <cmd>
# inside the conda env (e.g. `iqc ...`, `pytest`, `bash .../docker_smoke_test.sh`).
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "iqc-env"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]

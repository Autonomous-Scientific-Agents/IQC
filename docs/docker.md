# IQC Docker Images

Two ways to run IQC in containers:

1. **`Dockerfile`** — a single conda-based image with the whole stack
   (`iqc:full`, ~8.6 GB). Simplest; described in the rest of this file.
2. **`Dockerfile.uv` + `docker-compose.yml`** — a **conda-free, uv-based split
   image family** so you build/pull only what a task needs. See
   [Split images with docker compose](#split-images-with-docker-compose)
   below. This is the recommended layout for smaller images.

---

# IQC Docker Image (full CPU stack)

The `Dockerfile` at the repo root builds a single CPU-only image that bundles
IQC with every **open-source** quantum-chemistry code and machine-learning
interatomic potential IQC supports.

| Category            | Included in the image                                   | How IQC uses it                     |
| ------------------- | ------------------------------------------------------- | ----------------------------------- |
| Quantum chemistry   | **xTB** (GFN2)                                           | `--calculator xtb`                  |
|                     | **PySCF** (HF/DFT/MP2/CCSD/CCSD(T))                      | `--params` with `calculator: pyscf` |
|                     | **NWChem**                                               | `--task nmr --backend nwchem`; ASE  |
|                     | **ExaChem** (SCF/MP2/CCSD/CCSD(T), built from source)   | `--calculator exachem`              |
| ML potentials       | **MACE** (MACE-MP foundation model)                     | `--calculator mace`                 |
|                     | **FAIRChem UMA** (`uma-s-*`, `uma-m-*`)                 | `--calculator uma-s-omol` …         |
|                     | **ASE EMT**                                              | `--calculator emt`                  |

The whole toolchain (compilers + OpenMPI + CMake) and the conda QC codes come
from **conda-forge**, so ExaChem, `mpi4py`, and NWChem all share **one MPI**
(OpenMPI). PyTorch (CPU) and the MLIP stack are pip-installed after, in the
order documented in [calculators.md](calculators.md).

## Codes that are NOT baked in (proprietary / licensed)

IQC also supports **ORCA**, **Gaussian**, and **VASP**. These require a license
and cannot be redistributed inside an image. Wire them in at runtime by
mounting the installed binary and pointing IQC at it:

```bash
# ORCA (needs its own binary on PATH or ASE_ORCA_COMMAND)
docker run --rm -v /opt/orca:/opt/orca -e ASE_ORCA_COMMAND=/opt/orca/orca \
  -v "$PWD":/work iqc:full \
  iqc --smiles O --task single --calculator orca

# VASP
docker run --rm -v /opt/vasp:/opt/vasp -e IQC_VASP_COMMAND="mpirun -n 4 /opt/vasp/vasp_std" \
  -v "$PWD":/work iqc:full  iqc --xyz bulk.xyz --task opt --params vasp.yaml

# Gaussian (used by the NMR backend)
docker run --rm -v /opt/g16:/opt/g16 -e GAUSS_EXEDIR=/opt/g16 -e PATH="/opt/g16:$PATH" \
  -v "$PWD":/work iqc:full  iqc --smiles CCO --task nmr --backend gaussian
```

## Build

```bash
docker build -t iqc:full .
```

The build compiles ExaChem + its TAMM dependency from source (BLIS, Libint,
Libecpint, GlobalArrays are auto-built by TAMM). Expect a long first build;
the conda-env and ExaChem layers are cached separately, so later rebuilds that
only touch IQC source are fast.

Build knobs (via `--build-arg` are not used; the ExaChem script reads env):
`BUILD_JOBS` (default 16) controls ExaChem compile parallelism — lower it on
RAM-limited machines.

## Run

Jupyter Lab (default command):

```bash
docker run --rm -p 8888:8888 -v "$PWD":/work iqc:full
# open http://localhost:8888
```

One-shot CLI (anything after the image name runs inside the conda env):

```bash
docker run --rm -v "$PWD":/work iqc:full \
  iqc --smiles O --task thermo --calculator xtb

docker run --rm -v "$PWD":/work iqc:full \
  iqc --smiles O --task single --calculator exachem \
  --params /work/exachem.yaml
```

Parallel runs use the image's OpenMPI:

```bash
docker run --rm -v "$PWD":/work iqc:full \
  mpiexec -n 4 iqc --xyz /work/molecules --task single --calculator xtb
```

IQC detects the launcher's rank/size variables and splits the input across
ranks. The images deliberately do **not** set `IQC_DISABLE_MPI`: that flag
short-circuits detection, so each rank would report rank 0 / size 1 and redo
the entire input (4 duplicate result sets for `-n 4`). A serial `docker run
... iqc ...` needs no flag — IQC runs serial when no launcher variables are
present.

## MLIP models at runtime

Model checkpoints are **not** baked in (keeps the image small):

- **MACE** downloads its foundation checkpoint on first use — needs network.
- **UMA** checkpoints are gated on Hugging Face. Provide a token and a
  persistent cache:

  ```bash
  docker run --rm -e HF_TOKEN=hf_xxx -v "$HOME/.cache/huggingface":/opt/hf-cache \
    -v "$PWD":/work iqc:full \
    iqc --smiles O --task single --calculator uma-s-omol
  ```

`HF_HOME` defaults to `/opt/hf-cache` inside the image; mount a host directory
there to persist downloads across runs.

## Verify the image

A self-contained smoke test exercises every bundled calculator and the pytest
suite:

```bash
docker run --rm iqc:full bash /opt/iqc/scripts/docker_smoke_test.sh
```

Each image sets `IQC_IMAGE_TARGET`, and the smoke test maps that to the set of
calculators the target is expected to ship. A required calculator that is
missing — or installed but failing to import, e.g. a missing shared library —
**fails** the run; only calculators the target intentionally excludes are
skipped. Override the expected set for ad-hoc runs:

```bash
docker run --rm -e IQC_SMOKE_REQUIRE="xtb pyscf" iqc:base \
  bash /opt/iqc/scripts/docker_smoke_test.sh
```

Each calculator check runs in its own directory and the resulting JSONL row is
validated: no `error`/`*_error` fields, the task's energies present and finite,
and `opt_converged` true for optimizations. This matters because `iqc` exits 0
after recording a per-molecule failure, so exit status alone does not show that
a calculator worked.

Required checks: pytest, EMT, xTB (single/opt/thermo), PySCF, ExaChem, NWChem.
Optional checks (need network / gated model): MACE, FAIRChem UMA import.

Expected result (verified): `RESULT: PASS` — the pytest suite passes and every
bundled calculator runs. Two ExaChem tests are deselected because they assume
an environment where ExaChem is *not* installed (one hardcodes `/usr/bin/true`
as a stand-in binary, absent from the slim base; the other asserts a "Forces"
warning that does not apply — the ExaChem SCF run itself succeeds). These are
test-fixture assumptions, not functional failures.

## What CI verifies

Per-PR CI lints both Dockerfiles, validates `docker-compose.yml`, builds the
`base` target of `Dockerfile.uv`, and runs the smoke test inside it.

CI does **not** build the ExaChem-compiling images (the root `Dockerfile`, or
the `exachem`/`full` targets). Compiling TAMM + ExaChem pulls in Libint,
GlobalArrays, BLIS, and HDF5 from source and does not fit a hosted runner: a
measured attempt was killed after 41 minutes while still only 25% through
Libint (unit `unity_1945` of ~1948), with hours left before ExaChem itself
starts. Build those images locally, and verify them with the smoke test:

```bash
docker build -t iqc:full .
docker run --rm iqc:full bash /opt/iqc/scripts/docker_smoke_test.sh
```

`BUILD_JOBS` (default 16) controls compile parallelism — lower it on machines
with limited RAM, since the Libint unity units are memory-hungry.

A manual `docker-build-mlip` job (run via *Actions → CI → Run workflow*) builds
the `mlip` target and asserts torch resolved to its `+cpu` build; run it when
the MLIP dependency set changes.

## Notes

- **MPI as root**: the image sets `OMPI_ALLOW_RUN_AS_ROOT=1` and
  `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` because OpenMPI's launcher refuses to run
  as root (the container's default user), and ExaChem is driven via
  `mpiexec -n N`.
- **One MPI**: the whole stack (ExaChem, `mpi4py`, NWChem) is built against the
  conda-forge **OpenMPI**. NWChem on conda-forge is only packaged against
  OpenMPI, which is why the image standardizes on it rather than MPICH.
- **libstdc++**: `LD_LIBRARY_PATH` points at the conda env libs so PyPI
  manylinux wheels (numpy, torch) find a new-enough `libstdc++` (the base
  image's system one is too old).

---

# Split images with docker compose

`Dockerfile.uv` builds a **conda-free** image family with `uv` (fast PyPI
installs) + apt for the compiled programs. One multi-stage build exposes four
targets so you only build/pull what a task needs; Docker shares the common
`base` layers on disk, and ExaChem is compiled **once** and copied into both
`exachem` and `full`.

| Target / image | Adds on top of base | Calculators | Size (verified) |
| --- | --- | --- | --- |
| `base` → `iqc:base` | — | xTB, PySCF, NWChem, EMT | **3.65 GB** |
| `mlip` → `iqc:mlip` | torch-cpu, MACE, FAIRChem UMA | + MACE, UMA | 6.4 GB |
| `exachem` → `iqc:exachem` | ExaChem (SCF/MP2/CCSD/CCSD(T)) | + ExaChem | 5.41 GB |
| `full` → `iqc:full-uv` | mlip + ExaChem | everything | 8.17 GB |

(The conda monolith `iqc:full` is 8.59 GB. The win from splitting isn't a
smaller *full* image — it's that most users pull only `base` (3.65 GB) or
`mlip`/`exachem`, and the base layers are stored once on disk.)

## Do we need conda? No.

Every dependency is available without conda:

- **PyPI wheels via uv**: IQC, PySCF, RDKit, pymatgen, ASE, mpi4py, PyTorch
  (CPU), MACE, FAIRChem UMA, and the xTB Python bindings.
- **apt**: NWChem, OpenMPI 5.0, CMake 3.31, compilers, libnuma (for building
  ExaChem — the *system* toolchain TAMM actually expects).

Benefits over the conda image: no 4.66 GB conda env, much faster builds (uv
resolves in seconds), **one MPI** (apt OpenMPI everywhere), and none of the
conda-specific hacks (ALLOW_CONDA, broken libtoolize, GLIBCXX mismatch).

Two small fixes the uv path needs, both handled in `Dockerfile.uv`:
- The PyPI `xtb` wheel bundles Fortran libs with an **executable stack** that
  modern kernels reject; `scripts/clear_execstack.py` clears that ELF flag so
  `xtb.ase` imports.
- The MLIP install routes through the PyTorch **CPU index**
  (`UV_EXTRA_INDEX_URL`) so fairchem's torch pin resolves to `+cpu` (no CUDA).

## How calculators map to containers

IQC runs each calculation in **one process**. Calculators come in two flavors:

- **In-process Python libraries** (PySCF, MACE, UMA, xTB, EMT) — must live in
  the *same* image as the IQC run. Compose can't turn these into network
  services a single run calls into.
- **Subprocess binaries** (ExaChem via `mpiexec`, NWChem via ASE) — invoked
  locally by IQC.

So compose gives you **task-specialized services sharing a volume**, not a
distributed calculator RPC. Pick the smallest service that has the calculator
you need. A single run that **mixes** families (e.g. IR with a MACE Hessian and
ExaChem/ORCA dipoles, or per-role `energy=exachem`/`vibration=mace`) must use
the **`full`** service.

## Usage

Build one target directly:

```bash
docker build -f Dockerfile.uv --target base    -t iqc:base    .
docker build -f Dockerfile.uv --target mlip    -t iqc:mlip    .
docker build -f Dockerfile.uv --target exachem -t iqc:exachem .
docker build -f Dockerfile.uv --target full    -t iqc:full-uv .
```

Or drive everything through compose (services build on first use):

```bash
# One-shot CLI — pick the smallest image with the calculator you need:
docker compose run --rm qc      iqc --smiles O --task thermo --calculator xtb
docker compose run --rm qc      iqc --smiles O --task single --params /work/pyscf.yaml
docker compose run --rm mlip    iqc --smiles O --task single --calculator mace
docker compose run --rm exachem iqc --smiles O --task single --calculator exachem \
                                    --params /work/exachem.yaml
docker compose run --rm full    iqc --task ir --xyz /work/m.xyz --params /work/ir.yaml

# Interactive Jupyter Lab (full stack) on http://localhost:8888
docker compose up jupyter
```

All services share a `./work` bind mount (your inputs/outputs) and a persistent
Hugging Face model cache, so results flow between them.

## Verify

The smoke test is capability-aware — it auto-detects which calculators an image
ships and only requires those (absent ones are SKIPped):

```bash
docker run --rm iqc:base    bash /opt/iqc/scripts/docker_smoke_test.sh   # xTB/PySCF/NWChem/EMT
docker run --rm iqc:mlip    bash /opt/iqc/scripts/docker_smoke_test.sh   # + MACE/UMA
docker run --rm iqc:exachem bash /opt/iqc/scripts/docker_smoke_test.sh   # + ExaChem
docker run --rm iqc:full-uv bash /opt/iqc/scripts/docker_smoke_test.sh   # everything
```

All four report `RESULT: PASS`: the pytest suite plus every calculator that
image ships (absent ones are reported as SKIP, so the same script is a valid
check for each target).

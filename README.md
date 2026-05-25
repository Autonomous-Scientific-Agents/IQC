# IQC
Interactive Quantum Chemistry

[![CI](https://github.com/Autonomous-Scientific-Agents/IQC/actions/workflows/ci.yml/badge.svg)](https://github.com/Autonomous-Scientific-Agents/IQC/actions/workflows/ci.yml)

## Installation

### Clone the repository
   ```bash
   git clone git@github.com:Autonomous-Scientific-Agents/IQC.git
   cd IQC
   ```

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using pip:
   ```bash
   pip install uv
   ```

2. Create a virtual environment and install IQC:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or in a single command:
   ```bash
   uv pip install -e . --python 3.8
   ```

### Option 2: Using Conda

1. First, install a package manager (Conda, Miniconda, Mamba, or MicroMamba)
   - Download Miniconda from the [official page](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. Create and activate the environment:
   ```bash
   conda env create -f env.yml
   conda activate iqc-env
   ```

3. Install IQC:
   ```bash
   pip install .
   ```

### MACE and UMA Calculators

To install MACE and FAIRChem UMA in the same Python 3.11+ environment, install
IQC's MLIP runtime dependencies first, then install `mace-torch` without
dependencies so its old `e3nn==0.4.4` metadata does not downgrade the newer
UMA-compatible `e3nn` stack:

```bash
uv pip install -e ".[mlip]"
uv pip install --no-deps mace-torch
```

For Conda environments created from `env.yml`, run this after activation:

```bash
python -m pip install --no-deps mace-torch
python -m pip install -e .
```

On shared HPC systems, limit BLAS/OpenMP threads before importing NumPy,
PyTorch, MACE, or FAIRChem. Otherwise OpenBLAS may try to create one thread
per visible CPU and fail with `pthread_create failed`:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

See [docs/calculators.md](docs/calculators.md) for calculator names, UMA model
mapping, and Hugging Face setup.

### Option 3: Using Docker

1. Build the Docker image:
   ```bash
   docker build -t iqc .
   ```

2. Run the container with Jupyter Lab:
   ```bash
   docker run -p 8888:8888 -it iqc
   ```

3. Access Jupyter Lab by opening `http://localhost:8888` in your web browser

To persist your notebooks, you can mount a local directory:
```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks -it iqc
```

# Troubleshooting
If you see (a possible OpenMPI error):

```bash
shmem: mmap: an error occurred while determining whether or not /tmp/ompi.yv.1001/jf.0/3074883584/sm_segment.yv.1001.b7470000.0 could be created
```

Try: `export OMPI_MCA_btl_sm_backing_directory=/tmp`

### MACE checkpoint loading with PyTorch 2.6+

PyTorch 2.6 changed `torch.load` to default to `weights_only=True`. Older MACE
foundation checkpoints serialize MACE/e3nn/PyTorch model classes, so direct
`mace_mp()` calls may fail with an error like:

```text
_pickle.UnpicklingError: Weights only load failed
Unsupported global: GLOBAL mace.modules.models.ScaleShiftMACE
```

IQC's `get_calculator(name="mace")` applies a compatibility patch that
allowlists the trusted MACE foundation checkpoint classes with
`torch.serialization.add_safe_globals(...)`. If you load MACE checkpoints
outside IQC, use the same safe-global approach for trusted checkpoint sources
or install a MACE/PyTorch combination that handles PyTorch 2.6 safe loading.

## Available Tasks

IQC provides several computational tasks that can be performed on molecular systems:

- `single`: Single-point energy and force calculation
- `vib`: Vibrational frequency calculation
- `ir`: Infrared spectrum calculation
- `opt`: Geometry optimization
- `thermo`: Thermochemical analysis
- `ir-thermo`: Infrared spectrum and thermochemistry from one shared Hessian
- `nmr`: Solution-state NMR shielding and spectrum simulation using ORCA, NWChem, or Gaussian, with optional xTB pre-optimization

These tasks can be specified when running calculations. For example:

```python
from iqc.asetools import run_single_point, run_vibrations, run_optimization, run_thermo

# Single-point calculation
results = run_single_point(atoms)

# Vibrational frequency calculation
results = run_vibrations(atoms)

# Geometry optimization
results = run_optimization(atoms)

# Thermochemical analysis
results = run_thermo(atoms)
```

Each task returns a dictionary containing the results and timing information in milliseconds.

## Calculator Selection

Use `--calculator` to choose the ASE calculator for `single`, `opt`, `vib`,
`ir`, and `thermo` tasks:

```bash
iqc --xyz molecule.xyz --calculator mace --task opt
iqc --xyz molecule.xyz --calculator uma-s-omol --task single
```

Supported names are `mace`, `xtb`, `emt`, `orca`, `uma`, `uma-s-omol`,
`uma-s-omat`, `uma-s-odac`, `uma-m-omol`, `uma-m-omat`, and `uma-m-odac`.
`uma` is an alias for `uma-s-omol`. `orca` requires the ORCA executable on
PATH (or set `ASE_ORCA_COMMAND`, or pass `command:` under `calculator_params`
in `--params`).

## IR Workflow

The `ir` task computes vibrational frequencies and infrared intensities by
finite-difference displacement. The workflow has three independent steps —
geometry optimization, force evaluation at each displacement (Hessian), and
dipole-moment evaluation at each displacement (intensities) — and each step
can use a *different* calculator. This is useful when a fast MLIP is good
enough for forces but you want a more accurate electronic-structure method
for dipoles.

**Dipole support matters.** The dipole calculator must declare `'dipole'` in
its `implemented_properties`. Of the built-in `--calculator` options, `xtb`
and `orca` do. MACE, EMT, and the UMA models do not, so they cannot be used
as the dipole calculator and cannot drive a single-calculator IR run.

### Single calculator

Use `--calculator` to apply one calculator to all three steps. Pick a
calculator that supports dipoles:

```bash
iqc --task ir --xyz molecule.xyz --calculator xtb
```

`--calculator mace` (or `emt`, or any UMA model) will fail at the IR analysis
step with a clear error explaining that the calculator does not implement the
`'dipole'` property.

To compute both IR intensities and thermochemistry without repeating the
optimization and Hessian calculation, use the composite `ir-thermo` task:

```bash
iqc --task ir-thermo --xyz molecule.xyz --calculator xtb
```

`ir-thermo` uses the same IR calculator-role settings described below and then
derives thermochemistry from the vibrational energies produced by that same
finite-difference run.

### Mixed calculators (per-role override)

Per-role calculators are configured through the YAML param file passed with
`--params`. Any role left unset falls back to `--calculator`. Calculator names
use the same vocabulary as `--calculator` (`mace`, `xtb`, `emt`, `orca`,
`uma`, `uma-s-*`, `uma-m-*`); unknown names are rejected up-front instead of
silently falling back.

A practical mixed workflow uses a fast MLIP for the expensive Hessian and a
DFT calculator for accurate dipoles. ORCA-specific options
(`orcasimpleinput`, `orcablocks`, `command`) go under `calculator_params`,
which is shared by every role that uses ORCA:

```yaml
# ir_params.yaml
calculator: mace            # fallback for any role not overridden below
calculator_params:
  orcasimpleinput: B3LYP def2-SVP
  orcablocks: "%pal nprocs 4 end"
  command: /path/to/orca    # optional; honors $ASE_ORCA_COMMAND otherwise
ir_params:
  optimization_calculator: mace
  vibration_calculator: mace
  dipole_calculator: orca
```

```bash
iqc --task ir --xyz molecule.xyz --params ir_params.yaml
```

The result dictionary records which calculator was used for each role under
`calculator_optimization`, `calculator_vibration`, and `calculator_dipole`.

### Python API

The Python API additionally accepts pre-built calculator instances, which is
useful when you want different ORCA settings per role (different basis for
optimization vs. dipoles), or when wiring in a calculator IQC's CLI does not
register:

```python
from ase.calculators.orca import ORCA, OrcaProfile
from iqc.asetools import get_atoms_from_xyz, get_calculator, run_ir

atoms = get_atoms_from_xyz("molecule.xyz")
mace = get_calculator(name="mace")
orca = ORCA(
    profile=OrcaProfile(command="/path/to/orca"),
    orcasimpleinput="B3LYP def2-SVP",
)

atoms, results = run_ir(
    atoms,
    optimization_calculator=mace,
    vibration_calculator=mace,
    dipole_calculator=orca,
)
```

## Tabular Data Input

IQC can inspect and process tabular data files with `--input` / `-i`.

Inspect a data file without running a calculation:

```bash
iqc --input molecules.parquet
```

When `--input` is the only option, IQC prints the file format, row and column counts, schema, null counts, and basic numeric statistics.

Run a calculation from a column containing XYZ text:

```bash
iqc --input molecules.parquet --xyz geometry --task single
```

For IQC-generated JSONL or parquet files, `--xyz` can be omitted. IQC then
uses the `opt_xyz` column by default, which reuses the optimized geometry from
a previous run:

```bash
iqc --input iqc_opt_results.parquet --task single
```

Run a calculation from a column containing SMILES strings:

```bash
iqc --input molecules.csv --smiles smiles --task opt
```

Sort tabular rows before running the task:

```bash
iqc --input molecules.parquet --smiles smiles --sort energy --sort_order up --task single
```

Supported input formats include parquet (`.parquet`, `.pq`), CSV/TSV text files, Excel files (`.xls`, `.xlsx`, `.xlsm`, `.ods`), JSON/JSONL, Feather, and Arrow IPC. For parquet, CSV/TSV, Feather, and Arrow IPC, IQC reads only the requested structure and sort columns where possible.

When `--input` is used for a calculation, pass one structure selector or omit
both to use the IQC result default:

- `--xyz COLUMN`: read XYZ-format geometry strings from `COLUMN`
- `--smiles COLUMN`: read SMILES strings from `COLUMN` and build 3D geometries with RDKit

If neither selector is provided, IQC uses `--xyz opt_xyz`. This is intended for
rerunning calculations from IQC result files that contain optimized geometries.

Use `--sort COLUMN` to sort rows before processing. `--sort_order up` sorts ascending and `--sort_order down` sorts descending; the default is `up`. Missing sort values are rejected.

CSV files may contain multiline XYZ values, but those cells must be quoted by the CSV writer. For example:

```csv
name,geometry
water,"3
water
O 0 0 0
H 0 0 1
H 1 0 0
"
```

Results written from tabular inputs include `input_mode`, `data_input_file`, `data_xyz_column` or `data_smiles_column`, `data_sort_column`, `data_sort_order`, and `data_row_index` so each output can be traced back to the source row.

See [docs/data_input.md](docs/data_input.md) for the full tabular input guide.

## Parquet Utility Commands

When IQC is installed, the Python utilities in `scripts/` are available as
console commands:

```bash
iqc-jsonl2parquet results.jsonl results.parquet
iqc-read-parquet results.parquet
iqc-optimize-parquet results.parquet optimized.parquet
iqc-reduce-parquet results.parquet --drop 2 5 -o reduced.parquet
```

## Skipping Existing Calculations

Use `--skip-existing` to avoid rerunning calculations that IQC has already
recorded. IQC checks the same calculation identity used by the SQLite unique
key: normalized initial geometry, params, calculator, model, and task.

When `--database` is provided, existing rows in that SQLite database are used:

```bash
iqc --xyz molecules --task single --calculator uma-s-omol \
  --database results.db --skip-existing
```

IQC can also index prior IQC JSON/JSONL result files:

```bash
iqc --xyz molecules --task single --skip-existing \
  --skip-existing-from iqc_single_results_20260501_120000_abcd1234.jsonl
```

If `--skip-existing-from` is omitted, IQC scans combined
`iqc_*_results_*.jsonl` files in the current directory and per-rank JSON files
under `tmp_*` result directories.

## NMR Workflow

The `nmr` task runs a backend-aware solution-state NMR workflow for small and medium-sized molecules. It can:

- optimize the input geometry before the shielding calculation
- optionally sample conformers with RDKit and apply Boltzmann weighting
- run GIAO shielding calculations with `orca`, `nwchem`, or `gaussian`
- simulate `1H` and `13C` spectra with Lorentzian, Gaussian, or pseudo-Voigt broadening
- save plots plus tabulated per-atom and weighted peak data

Reasonable defaults are provided for routine organic-molecule prediction:

- backend: `orca`
- nuclei: `1H` and `13C`
- geometry optimization: enabled
- conformer sampling: auto-enabled for flexible molecules when RDKit is available
- optimization level: `B3LYP/def2-SVP`
- shielding level: `PBE0/def2-TZVP`
- solvent model: `smd`
- solvent: `chloroform`
- linewidth: `0.05 ppm` for `1H`, `1.0 ppm` for `13C`

Built-in reference shieldings are available for `1H` and `13C` so the workflow works with minimal input, but they are approximate. For production use, pass calibrated values with `--reference-shielding`.

### Basic Usage

Minimal ORCA run:

```bash
iqc --task nmr --backend orca --xyz molecule.xyz --output-dir nmr_results
```

Generate the starting 3D geometry directly from a SMILES string with RDKit:

```bash
iqc --task nmr --backend orca --smiles CCO --output-dir nmr_results
```

Use xTB for cheap geometry optimization before ORCA shielding:

```bash
iqc --task nmr --backend orca --optimization-backend xtb --xyz molecule.xyz --output-dir nmr_results
```

Customize nuclei, method, basis, solvent, and references:

```bash
iqc --task nmr \
  --backend gaussian \
  --nuclei 1H 13C \
  --method B3LYP \
  --basis def2-TZVP \
  --optimization-method B3LYP \
  --optimization-basis def2-SVP \
  --solvent-model smd \
  --solvent dmso \
  --reference-shielding 1H=31.77 \
  --reference-shielding 13C=188.10 \
  --xyz molecule.xyz \
  --output-dir nmr_results
```

### Main NMR Options

The most important CLI flags are:

- `--backend`: shielding backend, one of `orca`, `nwchem`, `gaussian`
- `--optimization-backend`: optional optimization backend, one of `orca`, `nwchem`, `gaussian`, `xtb`
- `--nuclei`: nuclei to compute, for example `--nuclei 1H 13C`
- `--method` and `--basis`: shielding calculation level
- `--optimization-method` and `--optimization-basis`: geometry optimization level
- `--solvent-model` and `--solvent`: implicit solvent settings
- `--charge` and `--multiplicity`: total charge and spin state
- `--optimize-geometry` / `--no-optimize-geometry`: control geometry optimization
- `--conformer-sampling` / `--no-conformer-sampling`: control RDKit conformer sampling
- `--num-conformers`: maximum number of retained conformers
- `--temperature`: Boltzmann weighting temperature in Kelvin
- `--linewidth`, `--lineshape`, and `--plot-range`: spectrum simulation controls
- `--reference-shielding`: reference override in the form `nucleus=value`
- `--output-dir`: directory for plots and tables

When `--smiles` is used, IQC builds a 3D structure with RDKit and starts from the lowest-energy embedded conformer.

`xtb` is supported only as `--optimization-backend`. It is not a direct NMR shielding backend.

### Outputs

The workflow writes:

- a combined PNG spectrum plot for the requested nuclei
- one CSV spectrum trace per nucleus
- per-atom CSV and JSON tables with:
  - conformer id
  - atom index
  - element
  - nucleus
  - isotropic shielding
  - converted chemical shift
  - conformer energy
  - Boltzmann weight
- weighted peak CSV and JSON tables
- a conformer summary CSV with energies, weights, and convergence status

These files make it easy to inspect raw shielding results, compare conformers, and post-process the simulated spectra.

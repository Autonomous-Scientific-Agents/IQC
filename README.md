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

## Available Tasks

IQC provides several computational tasks that can be performed on molecular systems:

- `single`: Single-point energy and force calculation
- `vib`: Vibrational frequency calculation
- `ir`: Infrared spectrum calculation
- `opt`: Geometry optimization
- `thermo`: Thermochemical analysis
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

Run a calculation from a column containing SMILES strings:

```bash
iqc --input molecules.csv --smiles smiles --task opt
```

Sort tabular rows before running the task:

```bash
iqc --input molecules.parquet --smiles smiles --sort energy --sort_order up --task single
```

Supported input formats include parquet (`.parquet`, `.pq`), CSV/TSV text files, Excel files (`.xls`, `.xlsx`, `.xlsm`, `.ods`), JSON/JSONL, Feather, and Arrow IPC. For parquet, CSV/TSV, Feather, and Arrow IPC, IQC reads only the requested structure and sort columns where possible.

When `--input` is used for a calculation, pass exactly one structure selector:

- `--xyz COLUMN`: read XYZ-format geometry strings from `COLUMN`
- `--smiles COLUMN`: read SMILES strings from `COLUMN` and build 3D geometries with RDKit

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

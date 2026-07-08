# Calculator Setup

IQC can use ASE calculators through `--calculator`.

Supported calculator names:

- `mace`: MACE Materials Project foundation model
- `mace-polar`: Electrostatic MACE / MACE-Polar foundation model
- `xtb`: GFN2-xTB through `xtb-python`
- `emt`: ASE EMT fallback calculator
- `orca`: ORCA via ASE's wrapper; needs the ORCA executable on PATH or in
  `ASE_ORCA_COMMAND`. Accepts `orcasimpleinput`, `orcablocks`, and
  `command` under `calculator_params` in `--params`.
- `uma`: alias for `uma-s-omol`
- `uma-s-omol`, `uma-s-omat`, `uma-s-odac`: UMA small model with the selected task
- `uma-m-omol`, `uma-m-omat`, `uma-m-odac`: UMA medium model with the selected task

Examples:

```bash
iqc --xyz molecule.xyz --calculator mace --task opt
iqc --xyz molecule.xyz --calculator mace-polar --task single
iqc --xyz molecule.xyz --calculator uma-s-omol --task single
iqc --xyz bulk.xyz --calculator uma-m-omat --task opt
iqc --xyz molecule.xyz --calculator orca --task ir --params orca_params.yaml
```

## MACE and UMA in One Environment

MACE package metadata pins `e3nn==0.4.4`, while FAIRChem UMA needs a newer
`e3nn` stack. IQC's base install does not install PyTorch or `e3nn`; those are
kept in the `mlip` optional dependency group. IQC follows the workaround
discussed in [ACEsuit/mace#555](https://github.com/ACEsuit/mace/issues/555):
install the shared runtime and UMA dependencies normally, then install
`mace-torch` without letting its dependency metadata downgrade `e3nn`.

On Linux, resolving PyTorch from the default PyPI index can install CUDA/NVIDIA
runtime wheels even on Intel GPU or CPU-only systems. Install the PyTorch wheel
for your hardware first, then install IQC's MLIP dependencies.

CPU-only PyTorch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[mlip]"
uv pip install --no-deps mace-torch
```

Intel GPU / XPU PyTorch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
uv pip install -e ".[mlip]"
uv pip install --no-deps mace-torch
```

With `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[mlip]"
uv pip install --no-deps mace-torch
```

With Conda:

```bash
conda env create -f env.yml
conda activate iqc-env
python -m pip install --no-deps mace-torch
python -m pip install -e .
```

The `mlip` extra and `env.yml` include the runtime packages MACE expects, plus
`fairchem-core`, `torch-dftd`, and `e3nn>=0.5`. `mace-torch` is intentionally
not declared as a normal dependency because a resolver would try to satisfy its
old `e3nn==0.4.4` pin. If you need NVIDIA CUDA wheels, install the matching
PyTorch build before `.[mlip]`; otherwise use the CPU or XPU index above.

At runtime IQC also applies narrow e3nn compatibility patches before loading
MACE. They let newer e3nn versions read the older raw-byte codegen buffers
stored inside existing MACE foundation-model checkpoints and restore generated
`SphericalHarmonics` callables and `Activation` paths that older checkpoints did
not serialize.

Check the resulting environment:

```bash
python - <<'PY'
from importlib.metadata import version

import e3nn
import mace

print("e3nn", e3nn.__version__)
print("mace", mace.__version__)
print("fairchem-core", version("fairchem-core"))
PY
```

## MACE-Polar Parameters

`mace-polar` uses `mace.calculators.mace_polar`, which currently requires MACE
from the upstream `main` branch rather than the PyPI `mace-torch` release. It
also requires `graph_electrostatics`, which provides the `graph_longrange`
runtime module.

```yaml
calculator: mace-polar
calculator_params:
  model: polar-1-m
  device: cpu
  default_dtype: float64
```

Use `model: polar-1-l` for the large checkpoint. In MPI runs, IQC serializes
the first checkpoint download with a file lock, but a shared pre-cached model
is still more robust on clusters where compute nodes have limited internet
access:

```bash
mkdir -p /path/to/shared/mace-models
curl -L \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-L.model \
  -o /path/to/shared/mace-models/MACE-POLAR-1-L.model
```

```yaml
calculator: mace-polar
calculator_params:
  model: /path/to/shared/mace-models/MACE-POLAR-1-L.model
  device: cpu
  default_dtype: float64
```

IQC passes total molecular charge and spin to MACE-Polar through `atoms.info`.
MACE-Polar's `spin` input is total spin S, so IQC translates its public
`--multiplicity` convention before calculation. When MACE-Polar exposes them,
IQC stores `dipole`, `partial_charges`, `partial_dipoles`,
`density_coefficients`, `spin_charge_density`, and spin-channel partial charges
in JSON/JSONL/Parquet results.

UMA checkpoints are gated Hugging Face assets. Before using a UMA calculator,
request access to the UMA model repository and log in:

```bash
huggingface-cli login
```

On clusters, avoid making every MPI rank or every job re-check Hugging Face.
Use a persistent model cache on shared storage and, after the checkpoint is
already downloaded, run from the cache in offline mode. Keep `HF_HOME` on
user-local storage unless you intentionally want to share Hugging Face tokens.

```bash
export HF_HUB_CACHE=/path/to/shared/huggingface/hub
export HF_HUB_OFFLINE=1
```

```yaml
calculator: uma-s-omol
calculator_params:
  cache_dir: /path/to/shared/fairchem-cache
  device: cuda
```

If you have the UMA checkpoint file locally, pass it directly and IQC will load
it without asking Hugging Face for the named hosted model:

```yaml
calculator: uma-s-omol
calculator_params:
  checkpoint_path: /path/to/uma-s-1p2.pt
  device: cuda
```

## UMA Parameters

IQC maps the compact calculator names to FAIRChem model/task pairs:

- `uma-s-*` uses `uma-s-1p2`
- `uma-m-*` uses `uma-m-1p1`
- the suffix selects `task_name`: `omol`, `omat`, or `odac`

Calculator parameters from `--params` are passed through to UMA. These FAIRChem
predictor options are routed to `pretrained_mlip.get_predict_unit`: `device`,
`inference_settings`, `overrides`, `cache_dir`, `workers`, and `seed`.
For local `checkpoint_path` loading, IQC routes the supported local predictor
options `device`, `inference_settings`, `overrides`, `atom_refs`,
`form_elem_refs`, and `workers` to FAIRChem's `load_predict_unit`.

For UMA, `device` is constrained by FAIRChem, not by PyTorch alone. Even when
PyTorch is installed with Intel XPU support, current FAIRChem UMA initialization
accepts `device: cpu` or `device: cuda`; `device: xpu` is rejected before IQC can
run a calculation. On Intel GPU systems, use the XPU PyTorch wheel to avoid
NVIDIA packages, but set UMA `device: cpu` unless your FAIRChem version
explicitly supports XPU.

For `omol`, FAIRChem uses molecular `charge` and spin multiplicity from
`atoms.info` when present. IQC defaults generated SMILES geometries to neutral
closed-shell molecules unless your input or workflow supplies different values.

## Mixing Calculators in the IR Workflow

The `ir` task runs three independent steps — geometry optimization, force
evaluation at each finite-difference displacement (Hessian / normal modes),
and dipole-moment evaluation at each displacement (IR intensities). A single
`--calculator` runs all three; a YAML `--params` file can override any role.

```yaml
calculator: mace            # default for any role not set below
calculator_params:
  orcasimpleinput: B3LYP def2-SVP
  orcablocks: "%pal nprocs 4 end"
  command: /path/to/orca    # optional; honors $ASE_ORCA_COMMAND otherwise
ir_params:
  optimization_calculator: mace
  vibration_calculator: mace
  dipole_calculator: orca   # `xtb` also works as a built-in dipole role
```

Constraints:

- The dipole calculator must declare `'dipole'` in its
  `implemented_properties`. Of the built-in calculator names exposed via
  `--calculator` (`mace`, `mace-polar`, `xtb`, `emt`, `orca`, `uma*`), `xtb`,
  `orca`, and `mace-polar` qualify. MACE, EMT, and the UMA models cannot be
  used as the dipole calculator. Asking for one in `--calculator` for an `ir`
  task, or via the per-role override, produces an immediate error rather than a
  cryptic "dipole property not implemented" deep inside ASE.
- The CLI rejects unknown names in the IR per-role overrides instead of
  silently falling back to MACE (the behavior of `get_calculator` for the
  main `--calculator` flag). To use a calculator that IQC does not register,
  construct the instance yourself and pass it through the Python API.
- `calculator_params` is shared by every role that uses the same
  calculator name. If you need different ORCA settings for different roles
  (e.g. cheap basis for the optimization, larger basis for the dipole),
  build the instances in Python and pass them directly to `run_ir`.
- All three roles share the same finite-difference grid, so the vibration
  and dipole calculators must agree on the geometry produced by the
  optimization step. If you switch the optimization calculator
  independently, re-converge tightly enough that the dipole calculator sees
  a near-stationary geometry.
- Spin/charge from `--multiplicity` and `--charge` (or the XYZ comment
  line) are applied to each role using its native convention via
  `apply_spin_charge`.

Programmatic equivalent (any ASE calculator instance with `'dipole'` in
`implemented_properties` works as the dipole role):

```python
from ase.calculators.orca import ORCA
from iqc.asetools import run_ir, get_calculator

atoms, results = run_ir(
    atoms,
    optimization_calculator=get_calculator(name="mace"),
    vibration_calculator=get_calculator(name="mace"),
    dipole_calculator=ORCA(label="dipole", orcasimpleinput="B3LYP def2-SVP"),
)
# results["calculator_optimization"], results["calculator_vibration"],
# results["calculator_dipole"] record which calculator handled each role.
```

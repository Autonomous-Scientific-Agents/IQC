# Working in IQC

IQC (Interactive Quantum Chemistry) is a Python package wrapping ASE calculators
for molecular calculations, batch/HPC dispatch, data processing, and reports.
For using IQC to answer chemistry questions, read
`.agents/skills/iqc/SKILL.md`. For MCP setup, read `docs/mcp.md`.

## Environment and checks

Use Python 3.11+ for the agent/MCP workflow (MCP requires 3.10+). From the repo:

```bash
uv venv --python 3.12
. .venv/bin/activate
uv pip install -e '.[mcp,xtb,test]'
IQC_DISABLE_MPI=1 python -m pytest iqc/tests/test_mcp_server.py
IQC_DISABLE_MPI=1 python -m pytest iqc/tests/
iqc --help
python -m iqc.mcp_server --help
```

Prefer uv for environment creation and package installation. CI uses uv with
the `mcp,test` extras and a system OpenMPI runtime. Use an existing working
environment when available. Optional calculators need
separate dependencies, binaries, or models; absence is not a reason to install
all ML/GPU packages. Keep torch, MACE, UMA and external engines lazy to import.
`IQC_DISABLE_MPI=1` is for serial agent/test processes; do not set it for MPI
or scheduler dispatch. Cap BLAS threads for small shared-machine runs.

## Code map

- `iqc/main.py`: CLI and result/artifact orchestration.
- `iqc/asetools.py`: structures, calculator construction, calculation workflows.
- `iqc/electronic_state.py`, `iqc/hessiantools.py`: scientific validation.
- `iqc/mcp_server.py`: optional MCP interface; nine focused tools.
- `iqc/*dispatch.py`, `iqc/sweep_*.py`: batch and HPC workflows.
- `iqc/datatools.py`, `iqc/report.py`: data inspection and reports.
- `iqc/tests/`: maintained pytest suite; root-level scripts may be local experiments.
- `Dockerfile.uv`: split container targets; `mcp` is the stdio agent target.

## Implementation constraints

Preserve scientific units, charge/spin metadata, convergence checks, error fields,
and result provenance. A process exit code alone does not establish calculation
success: inspect the result's errors, finite energies, and convergence.
Add regression tests for changed scientific behavior and MCP protocol behavior.
Use built-in ASE/mock calculators for fast tests; gate optional engine/model
integration tests so ordinary tests do not download checkpoints or launch jobs.
Keep stdout exclusively for JSON-RPC in MCP stdio mode; diagnostics go to stderr.

Do not edit generated calculation results, datasets, notebooks, model caches, or
local credentials unless requested. Use temporary directories for verification.
Never include user data, credentials, or model weights in container images.
Keep the canonical chemistry skill in `.agents/skills/iqc/SKILL.md`; the root
`SKILL.md` and Claude skill directory link to it so instructions stay in sync.

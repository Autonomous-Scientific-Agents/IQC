---
name: iqc
description: Run quantum-chemistry calculations (single point, geometry optimization, vibrations, thermochemistry, IR) on small molecules through the IQC MCP server. Use when the user asks for energies, optimized geometries, vibrational frequencies, IR spectra, or thermodynamic properties (G, H, S, ZPE) of organic molecules from a SMILES string or XYZ coordinates.
---

# IQC — Interactive Quantum Chemistry

IQC wraps ASE-based calculators (xtb, MACE, UMA, ORCA, EMT) and exposes a
consistent API for the most common molecular-property workflows. The MCP
server (`iqc-mcp`) exposes a focused subset designed for agent use.

## When to invoke

Use IQC tools when the user wants any of:

- An energy, force, or optimized geometry for a molecule
- Vibrational frequencies (cm⁻¹), IR spectrum, or normal-mode count
- Thermochemistry: Gibbs free energy, enthalpy, entropy, ZPE at 298.15 K
- SMILES ↔ XYZ conversion or a quick 3D structure guess
- A schema/preview of a tabular IQC results file (parquet, CSV, JSONL, …)

**Do not** use IQC for:
- Periodic / crystalline / surface calculations (it's molecular-only here)
- Excited states, TD-DFT, or multireference work
- Anything the user wants to run themselves on an HPC queue — point them at
  the `iqc` CLI instead

## Tools

| Tool                  | Purpose                                       | Cost     |
| --------------------- | --------------------------------------------- | -------- |
| `smiles_to_xyz`       | RDKit MMFF94 conformer → XYZ                  | trivial  |
| `xyz_to_smiles`       | Perceive canonical SMILES from XYZ            | trivial  |
| `list_calculators`    | Calculator names + dipole/IR compatibility    | trivial  |
| `run_single_point`    | Energy (+ forces)                             | seconds  |
| `run_optimization`    | BFGS geometry relaxation                      | seconds–minutes |
| `run_vibrations`      | Finite-diff Hessian → frequencies             | minutes  |
| `run_thermo`          | opt + vib + ASE IdealGasThermo                | minutes  |
| `run_ir`              | Finite-diff IR spectrum (freqs + intensities) | minutes  |
| `inspect_data_file`   | Schema/stats for parquet/CSV/JSONL/…          | seconds  |

All compute tools accept either `smiles` or `xyz` (string or file path), plus
`calculator`, `calculator_params`, `charge`, and `multiplicity`. See each
tool's docstring for full signatures.

## Picking a calculator

| Calculator       | Speed       | Accuracy           | Dipoles (IR) | Install                       |
| ---------------- | ----------- | ------------------ | ------------ | ----------------------------- |
| `xtb` *(default)* | fast        | semi-empirical     | yes          | `pip install iqc[xtb]`        |
| `mace`           | fast        | DFT-quality MLIP   | no           | `pip install iqc[mlip]` + MACE |
| `mace-polar`     | fast        | MLIP + electrostat | yes          | MACE main branch + graph_electrostatics |
| `uma-s-omol`     | fast        | MLIP (FAIRChem)    | no           | `pip install iqc[mlip]`       |
| `orca`           | slow        | DFT/HF/post-HF     | yes          | ORCA binary on PATH           |
| `emt`            | trivial     | metals only        | no           | built into ASE                |

Decision rules:

- **Default to `xtb`** for any organic molecule, especially for IR (it has
  dipoles built in).
- **Switch to `mace` or `uma-s-omol`** when the user wants DFT-quality
  energies / geometries without paying DFT cost, and IR is not needed.
- **Use `orca`** only when the user asks for DFT explicitly. Set
  `calculator_params={"orcasimpleinput": "B3LYP def2-SVP",
  "orcablocks": "%pal nprocs 4 end"}` and confirm ORCA is on PATH.
- **For IR with mixed accuracy**, pass `vibration_calculator="mace"` and
  `dipole_calculator="orca"` — IQC will reuse each role-specific calculator
  for the right step.

## Typical workflows

### Optimize and report energy
1. `run_optimization(smiles=..., calculator="xtb")`
2. Quote `opt_energy_eV`, `opt_converged`, `opt_xyz` back to the user.

### Predict IR spectrum
1. `run_ir(smiles=..., calculator="xtb")` — single-calc IR; fast.
2. For a publication-quality spectrum: `run_ir(smiles=..., calculator="mace",
   dipole_calculator="orca", calculator_params={"orcasimpleinput": "..."})`.

### Gibbs free energy for a thermodynamic estimate
1. `run_thermo(smiles=..., calculator="xtb")` then read `G_eV`, `H_eV`.

### Sanity check a geometry from a paper
1. `xyz_to_smiles(xyz=...)` to confirm the structure RDKit perceives matches
   the expected molecule. If `opt_smiles` differs after an optimization
   (`smiles_changed=True`), warn the user — the molecule rearranged.

## Inputs and outputs

- **`xyz`** parameter accepts both inline XYZ text and file paths. Comment
  line may carry `charge=…` and `multiplicity=…` (or `uhf=…`) tokens.
- **Energies** are reported in eV. Frequencies in cm⁻¹.
- **`forces`** are stripped from responses by default (large for big
  molecules). Pass `keep_forces=True` to `run_single_point` to keep them.
- Every result dict has `error` (string, empty on success) and `warnings`
  (list). Check both before reporting success.

## Pitfalls

- **MLIPs with dipoles**: regular `mace` / `uma-*` raise on IR — fall back to
  `xtb`, `orca`, or `mace-polar`.
- **First MACE/UMA call** downloads checkpoints (hundreds of MB). Tell the
  user to expect a one-time delay.
- **Imaginary frequencies** after `run_vibrations` usually mean the geometry
  was not at a stationary point. Re-run `run_optimization` with a tighter
  `fmax` (e.g. `0.001`) first.
- **Charged / open-shell species**: pass `charge` and `multiplicity`
  explicitly — don't rely on the SMILES alone.
- **Long compute times**: an IR run on a 20-atom molecule with ORCA can take
  many minutes. Set the user's expectation before launching.

## Installing the server (one-time)

```bash
pip install -e '.[mcp,xtb]'           # base + MCP + xtb (recommended)
pip install -e '.[mcp,mlip,xtb]'      # add MACE/UMA support
```

Then register `iqc-mcp` with your MCP client (Claude Desktop, Claude Code,
Cursor, etc.). See `docs/mcp.md` for client-specific config snippets.

---
name: iqc
description: Run quantum-chemistry calculations (single point, geometry optimization, vibrations, thermochemistry, IR) on small molecules through the IQC MCP server. Use when the user asks for energies, optimized geometries, vibrational frequencies, IR spectra, or thermodynamic properties (G, H, S, ZPE) of organic molecules from a SMILES string or XYZ coordinates.
---

# IQC — Interactive Quantum Chemistry

IQC wraps ASE-based calculators (xtb, MACE, MACE-Polar, UMA, ORCA, EMT, PySCF,
ExaChem, VASP) and exposes a consistent API for the most common
molecular-property workflows. The MCP server (`iqc-mcp`) exposes a focused
subset designed for agent use.

Energy-only backends (ExaChem, and PySCF's MP2/CCSD/CCSD(T) methods) are given
finite-difference forces automatically (via `ase.calculators.fd`), so they can
drive geometry optimization and vibrations like any force-capable calculator.
`run_thermo` / `run_ir` also accept per-role calculators
(`optimization_calculator`, `vibration_calculator`, `energy_calculator`,
`dipole_calculator`) for composite workflows — e.g. a MACE geometry + Hessian
with a CCSD(T) single-point energy.

## When to invoke

Use IQC tools when the user wants any of:

- An energy, force, or optimized geometry for a molecule
- Vibrational frequencies (cm⁻¹), IR spectrum, or normal-mode count
- Thermochemistry: Gibbs free energy, enthalpy, entropy, ZPE at 298.15 K
- SMILES ↔ XYZ conversion or a quick 3D structure guess
- A schema/preview of a tabular IQC results file (parquet, CSV, JSONL, …)

**Do not** use IQC for:
- Excited states, TD-DFT, or multireference work
- Crystalline / surface periodic optimization. (VASP *is* available for
  molecular thermochemistry — the molecule is wrapped in a large periodic box
  with Γ-point sampling — but IQC is not a general solid-state/surface tool.)
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
| `mace-polar`     | fast        | MLIP + electrostat | yes          | MACE + graph_electrostatics¹  |
| `uma-s-omol`     | fast        | MLIP (FAIRChem)    | no           | `pip install iqc[mlip]`       |
| `pyscf`          | med–slow    | DFT/HF (analytic forces) | yes    | `pip install pyscf`           |
| `pyscf-ccsd(t)`  | slow        | CCSD(T) (FD forces) | no          | `pip install pyscf`           |
| `exachem`        | slow        | CCSD(T) etc. (MPI/GPU) | no       | ExaChem binary²               |
| `vasp`           | slow        | plane-wave PBE (MP-compatible) | no | VASP binary + POTCARs²    |
| `orca`           | slow        | DFT/HF/post-HF     | yes          | ORCA binary on PATH           |
| `emt`            | trivial     | metals only        | no           | built into ASE                |

¹ MACE-Polar needs `graph_longrange`, installed from
  `git+https://github.com/WillBaldwin0/graph_electrostatics`; this pins
  e3nn 0.4.4 (incompatible with UMA/fairchem — see the `iqc-polaris-*` notes).
² ExaChem/VASP are HPC/GPU binaries; see the `iqc-polaris-exachem-vasp` memory
  note for the launch-wrapper pattern used on ALCF Polaris.

Decision rules:

- **Default to `xtb`** for any organic molecule, especially for IR (it has
  dipoles built in).
- **Switch to `mace` or `uma-s-omol`** when the user wants DFT-quality
  energies / geometries without paying DFT cost, and IR is not needed.
- **Use `pyscf`** for a real all-electron DFT/HF number (`calculator_params=
  {"method": "dft", "xc": "pbe", "basis": "def2-svp"}`); use PBE to match the
  MACE-MP (MPtrj) level. `pyscf-ccsd(t)` (or `exachem`) for coupled-cluster;
  they are energy-only, so their forces come from finite differences — the FD
  Hessian is expensive (hundreds of single points), so prefer a composite run
  (`energy_calculator="pyscf-ccsd(t)"` on a MACE geometry+Hessian).
- **Use `vasp`** for a plane-wave PBE reference comparable to MACE-MP; IQC wraps
  the molecule in a box with MP-compatible settings (ENCUT 520, spin-polarized).
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
  user to expect a one-time delay. On no-internet compute nodes, pre-cache on a
  login node first.
- **Energy-only methods (`exachem`, `pyscf-ccsd(t)`) self-consistent thermo is
  expensive**: their forces are finite differences, so a full opt+Hessian is
  hundreds of single points. For a CCSD(T) thermo number prefer the composite
  route — MACE (or PySCF-DFT) geometry+Hessian + a CCSD(T) `energy_calculator`
  single point. Note the composite `opt_energy_eV` is the *geometry* calc's
  energy; the CCSD(T) value is in `electronic_energy_eV`.
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

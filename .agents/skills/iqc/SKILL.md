---
name: iqc
description: Use IQC to build molecular structures, calculate energies, optimize geometries, predict vibrations or IR spectra, estimate ideal-gas thermochemistry, and inspect IQC data files. Supports MCP tools and the IQC CLI for local or batch quantum chemistry.
---

# IQC molecular calculations

Use the IQC MCP tools when connected. Otherwise use `iqc --help` and the
CLI in an installed IQC environment. Never invent calculation results.
Server installation and client configuration live in `docs/mcp.md` in the
IQC repository; this skill does not install or connect an MCP server.

## Choose the calculation

- `smiles_to_xyz(smiles="CCO")` makes an RDKit conformer guess, not a
  quantum-optimized geometry. `xyz_to_smiles` perceives connectivity from XYZ;
  check stereochemistry, charge, and bonding against the intended molecule.
- `run_single_point`: energy at the supplied geometry. `keep_forces=True`
  includes forces.
- `run_optimization`: geometry relaxation; check `opt_converged` and `opt_xyz`.
- `run_vibrations`: finite-difference frequencies; pre-optimization is on by
  default. Imaginary modes can indicate a saddle point or unconverged geometry.
- `run_thermo`: optimization, vibrations, and ideal-gas thermochemistry at
  298.15 K and 1 atm. Check imaginary modes even when the default
  `ignore_imag_modes=True` permits a result. This is not a solution free energy.
- `run_ir`: finite-difference frequencies and dipole derivatives. The optional
  `optimization_calculator`, `vibration_calculator`, and `dipole_calculator`
  select roles. `calculator_params` applies to `calculator` and to roles
  naming the same calculator; give other roles their own parameters with
  `optimization_calculator_params`, `vibration_calculator_params`, or
  `dipole_calculator_params` (e.g. MACE Hessian with ORCA dipoles). Lists
  longer than 1000 points, such as a full spectrum from
  `sparse_spectrum=False`, appear only in `omitted_fields` unless
  `keep_arrays=True`.
- `inspect_data_file(path=...)`: inspect schema and statistics before choosing
  input columns for a batch CLI run.

Compute tools take exactly one of `smiles` or `xyz` (inline XYZ or a server-side
file path). Pass charge and multiplicity explicitly when known; omitted charge
uses structure metadata, then zero. Multiplicity is 2S+1 and defaults from
electron-count parity, which does not establish the physical ground state.
Use the connected tool schema as the authority for accepted parameters.
In particular, MCP `run_thermo` does not expose composite calculator roles;
use the Python API or CLI for composite thermochemistry.

## Choose a calculator

Call `list_calculators` for supported names and requirements; this is a catalog,
not a check that a backend is installed or usable.

- Default to `xtb` for quick molecular estimates, including IR. Requires the
  xTB extra. Describe its semiempirical level of theory.
- Use PySCF or ORCA for requested HF/DFT calculations with explicit method,
  basis, and functional. They require separate installation; ORCA needs its
  external executable. Confirm tool support before using additional methods.
- MACE/UMA need optional model packages and checkpoints. Check the particular
  model's training domain; do not promise DFT accuracy for an arbitrary molecule.
  They may download large checkpoints on first use.
- IR requires a dipole-capable backend such as xTB, ORCA, PySCF HF/DFT, or
  MACE-Polar. Ordinary MACE/UMA and EMT do not provide dipoles.
- EMT is a built-in smoke-test backend for supported elements, not a
  quantum-chemistry model for organic molecules.
- Energy-only correlated methods use expensive finite-difference forces.
  Prefer a suitable geometry/Hessian model plus a correlated single point
  through the Python API or CLI for composite thermochemistry.

For batches, HPC schedulers, or calculations likely to exceed client timeouts,
use the CLI and repository dispatcher/configuration documentation.
Do not retry a timed-out expensive calculation until checking whether it is
still running; a client timeout need not stop the server's calculation.

## Validate and report

Check `error`, any task-specific `*_error`, and `warnings` before reporting
success. Report unconverged geometries and imaginary frequencies; do not silently
change the charge, spin, method, or backend to obtain a number. Compare perceived
connectivity before and after optimization and flag rearrangements.

Report input identity, charge, multiplicity, calculator, method/model and
basis where applicable, convergence, and units. Energies are eV, forces eV/Å,
frequencies cm⁻¹, and entropy eV/K. Distinguish electronic energy from ZPE,
enthalpy, and Gibbs free energy. Differences require compatible methods and
reference states. Keep warnings and approximations with the reported result.

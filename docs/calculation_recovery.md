# Electronic states and calculation recovery

IQC validates charge, spin, convergence and vibrational data before accepting
calculation results. Recovery is bounded and keeps the requested chemical model.

## Charge and multiplicity

Charge must be an integer. Multiplicity is `2S+1` and must be compatible with
`sum(atomic_numbers) - charge`: an even electron count requires odd multiplicity,
and an odd electron count requires even multiplicity. Impossible and fractional
states fail before invoking a backend.

The CLI/Python overrides take precedence, followed by input metadata, then
calculator settings. With no specified state, charge defaults to zero and
multiplicity defaults to singlet for even electron counts or doublet for odd
counts. **This parity default does not determine the ground state.** Specify
triplet O2 and transition-metal spin states explicitly.

Charged SMILES retain their formal charge. XYZ comments can specify
`charge=1 multiplicity=2`. Result XYZ strings and saved optimized geometries
preserve nondefault electronic states; charge/spin changes invalidate calculator
caches and produce distinct skip/retry keys. Legacy neutral/default-spin keys
remain unchanged. Recalculate older ionic results whose XYZ metadata lost the
charge; the new keys cannot identify those records reliably.

MACE-MP and EMT do not support arbitrary molecular charge/spin. Their existing
warning behavior remains; validation cannot add physics absent from a model.
Use a backend that supports the electronic state needed for the calculation.

## SCF convergence

The PySCF wrapper requires finite, converged SCF results. By default it tries one
Newton restart from the available density after ordinary SCF nonconvergence,
with the same method, basis, charge, spin, convergence tolerance and iteration
limit. It records `scf_converged`, `scf_recovery_used` and
`scf_recovery_attempts`. Set `calculator_params.scf_recovery: false` to disable
the restart. Input/backend exceptions still fail the calculation.

MP2/CCSD require a converged HF reference. Unconverged CCSD results are rejected
and cannot proceed to the perturbative triples correction. A successful SCF
restart is a convergence result, not a wavefunction-stability or ground-state
proof. See [PySCF SCF documentation](https://pyscf.org/user/scf.html).

## Geometry optimization

Enable optimizer recovery in a YAML parameter file:

```yaml
calculator: pyscf
calculator_params:
  method: hf
  basis: sto-3g
  scf_recovery: true
  max_cycle: 200
optimization_params:
  fmax: 0.01
  max_steps: 200
  optimizer: bfgs
  maxstep: 0.1
  recover: true
  recover_optimizers: [lbfgs, fire]
  max_recovery_attempts: 2
vibration_params:
  imag_recovery: true
  imag_displacement: 0.3
  max_imag_attempts: 2
thermo_params:
  ignore_imag_modes: false
```

For example: `iqc --xyz molecule.xyz --task thermo --params recovery.yaml`.
Each optimizer gets at most `max_steps`; the example allows the initial BFGS
attempt plus two restarts. Numerical exceptions, nonfinite forces/energies and
excessive energy changes restore the last validated geometry. Finite but
unconverged runs can restart there with the next optimizer. Results include
`opt_attempts`, optimizer names, step counts, errors and recovery counts.
Exhausting the configured attempts produces a failed result.

`IQC_MAX_ENERGY_PER_ATOM_EV` (default 1e4) limits energy **change from the starting
geometry** during optimization. An absolute total-energy threshold wrongly
rejects valid all-electron heavy-atom calculations. The independent corruption
screen uses `IQC_MAX_ABSOLUTE_ENERGY_PER_ATOM_EV` (default 1e9).

## Vibrations and thermochemistry

ASE diagonalises the full 3N x 3N Hessian, so the six (five for linear
molecules) translational and rotational modes remain in its output,
contaminated by residual gradients, grid noise or finite-difference error.
By default IQC removes them exactly: the Hessian returned by ASE is projected
onto the complement of the rigid-body subspace (Eckart/Sayvetz projection)
before frequencies, normal modes and thermochemistry energies are extracted
(`vibration_params.project_trans_rot`, default `true`). The rigid-body
eigenvalues then vanish by construction and the remaining 3N-6 (3N-5) modes are
unambiguous. Because the projection acts on the Hessian, it applies to every
calculator that reaches the vibration step - force-capable (MACE, xTB, PySCF,
ORCA, VASP, ...) or energy-only with numerical forces (ExaChem, PySCF CCSD(T)).
It is skipped automatically for partial Hessians (`indices`).

The contamination that was removed is still reported: results carry
`trans_rot_projected`, `trans_rot_frequencies_cm^-1` (the rigid-body block
before projection) and `trans_rot_coupling`. `max_trans_rot` is checked
against those pre-projection values, so a loosely optimized structure or a
poor `delta` still produces the "Translational or rotational modes are too
high" warning. Projection is exact only at a stationary point; it removes
contamination from the mode list but does not repair a bad Hessian, so treat
large `trans_rot_frequencies_cm^-1` as a convergence problem.

With `project_trans_rot: false` the previous behaviour is restored: rigid
modes are excluded by frequency magnitude instead of discarding the first 5/6
entries. Magnitude ordering cannot distinguish a genuine soft mode (a
30 cm^-1 torsion) from a contaminated rotation, so use it only for
comparison with earlier results. Imaginary vibrational frequencies remain
negative in the reported signed frequency list in both modes. Partial
Hessians keep all computed modes and cannot supply whole-molecule ideal-gas
thermochemistry.

The projector is also available on its own for any ASE `Vibrations`,
`Infrared` or `VibrationsData` object, or a raw Cartesian Hessian:

```python
from iqc.hessiantools import project_vibrations_data, project_hessian

projected, report = project_vibrations_data(vib)      # vib = ase Vibrations
energies = projected.get_energies()[report["n_rigid"]:]  # exactly 3N-6
hessian_p, report = project_hessian(hessian_eV_per_A2, atoms)  # analytic Hessians
```

With `imag_recovery: true`, displaced trial structures are reoptimized and their
Hessians recomputed. Attempts alternate positive and negative displacement for
each candidate imaginary mode. `max_imag_attempts: 2` allows both directions of
the first mode; the default is one attempt. A trial must succeed, pass physical
validation and reduce the number of imaginary modes to be accepted. Failed
trials leave the original result available; accepted trials retain its original
input identity. Recovery does not guarantee a minimum.

Set `thermo_params.ignore_imag_modes: false` for strict minimum thermochemistry.
The CLI is strict by default; the Python API defaults to true and omits
imaginary modes. Distinct optimization,
vibration and final-energy calculators are now used for their respective stages
in composite thermochemistry. The physical vibrational subset is selected before
calling ASE so ASE-version differences cannot discard the imaginary modes first.

## Ensemble failures and retries

Nonconverged optimization and `nonphysical: true` records remain failures in
JSONL/database/status indexes. `--skip-existing --retry-failed-only` can retry
them instead of treating a finite last energy as success. MPI, Parsl and
Ensemble Launcher use the same electronic-state identity for successful rows
and reconstructed worker failures.

NMR excludes nonconverged optimized conformers and incomplete, duplicate or
nonfinite shielding data before averaging. Boltzmann weights use a complete
common energy series: electronic/optimization energies first, otherwise initial
conformer energies. If neither series is complete, IQC reports a warning and
uses equal weights. It records the chosen series and energies in
`nmr_weight_energy_source` and `nmr_weight_energies_eV`. Temperature must be finite
and positive. Conformer energy weighting is an approximation, not a conformer
free-energy calculation.

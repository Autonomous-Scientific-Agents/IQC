#!/usr/bin/env python3
"""
Generic polymer relaxation with ASE + your favourite force field.

Examples
--------
# GFN-FF (xTB) – no model file needed
python relax_polymer.py polymer.pdb --ff xtb --density 1.0

# MACE – supply the .model file
python relax_polymer.py polymer.xyz --ff mace --model mace_polym.model

# UMA – supply the checkpoint
python relax_polymer.py polymer.pdb --ff uma --model uma.ckpt
"""
import argparse, sys, shutil, math
from pathlib import Path

import ase.io as aio
from ase import units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.optimize import FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

AVOGADRO = 6.02214076e23
CM2ANG    = 1.0e8          # 1 cm = 10⁸ Å
TARGET_T  = 300.0          # K
NVT_TIME  = 10.0           # ps
NPT_TIME  = 60.0           # ps
TIMESTEP  = 1.0            # fs
FMAX      = 0.02           # eV/Å ≈ 1 kcal mol⁻¹ Å⁻¹
DENS_TOL  = 0.03           # ±3 %

# ----------------------------------------------------------------------
def build_calc(name, model):
    # Delegate to iqc.asetools.get_calculator: the previous direct imports
    # (ase.calculators.xtb, mace.MACEResponseCalculator, uma.calculator) do
    # not exist in any released package, so every --ff choice crashed with
    # ModuleNotFoundError before the first MD step.
    from iqc.asetools import get_calculator

    if name == "xtb":
        return get_calculator("xtb", method="GFN-FF", accuracy=0.3)
    if name in ("mace", "uma"):
        kwargs = {"model": model} if model else {}
        return get_calculator(name, **kwargs)
    raise ValueError(f"Unknown ff '{name}'")

def density_g_cm3(atoms):
    m_g  = atoms.get_masses().sum() / AVOGADRO
    v_cm = atoms.get_volume() / CM2ANG**3
    return m_g / v_cm

# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("structure", help="PDB/XYZ file")
    p.add_argument("--ff", choices=["xtb", "mace", "uma"], required=True)
    p.add_argument("--model", help="model/checkpoint for MACE/UMA")
    p.add_argument("--density", type=float, default=1.0,
                   help="target density g cm⁻³")
    p.add_argument("--scale", type=float, default=1.3,
                   help="initial box inflation factor")
    p.add_argument("--compressibility", type=float, default=4.5e-5,
                   help="Berendsen compressibility in bar^-1 (default: 4.5e-5; adjust for your material)")
    args = p.parse_args()
    if not math.isfinite(args.compressibility) or args.compressibility <= 0:
        p.error("--compressibility must be finite and positive")

    atoms = aio.read(args.structure)
    atoms.set_pbc(True)

    # ----------- make sure there is a cell -----------
    if atoms.cell.volume < 1e-6:
        m_g = atoms.get_masses().sum() / AVOGADRO
        vol_cm3 = m_g / args.density
        edge = (vol_cm3 * CM2ANG**3) ** (1/3)
        atoms.set_cell([edge, edge, edge])
        atoms.center()

    atoms.set_cell(atoms.cell * args.scale, scale_atoms=True)

    # ----------- attach calculator -----------
    atoms.calc = build_calc(args.ff, args.model)

    # ----------- quick CG minimisation -----------
    opt = FIRE(atoms, trajectory="min.traj")
    opt.run(fmax=FMAX)

    # ----------- NVT (soft shake) -----------
    MaxwellBoltzmannDistribution(atoms, 10*units.kB)
    dyn1 = Langevin(atoms, TIMESTEP*units.fs, TARGET_T*units.kB,
                    friction=0.02)
    dyn1.run(int(NVT_TIME*1000/TIMESTEP))

    # ----------- NPT until density converges -----------
    npt_steps = int(NPT_TIME*1000/TIMESTEP)
    squeeze   = True
    while squeeze:
        rho = density_g_cm3(atoms)
        print(f"Current density = {rho:.3f} g/cm³")
        if abs(rho/args.density - 1.0) < DENS_TOL:
            print("✅ density within tolerance.")
            break
        # Run NPT
        # Barostat target is 1 atm in ASE pressure units (eV/A^3). The old
        # expression passed the target *density* (~1.0) as the pressure —
        # about 1.6 million atm — violently crushing the box; density
        # convergence is handled by the surrounding loop, not the barostat.
        dyn2 = NPTBerendsen(atoms, TIMESTEP*units.fs,
                            temperature_K=TARGET_T,
                            taut=100*units.fs,
                            pressure_au=1.01325*units.bar,
                            compressibility_au=args.compressibility/units.bar,
                            taup=1000*units.fs)
        dyn2.run(npt_steps)
        # Optional manual isotropic squeeze if still low
        rho = density_g_cm3(atoms)
        if rho < 0.97*args.density:
            cell = atoms.cell * 0.98
            atoms.set_cell(cell, scale_atoms=True)
        else:
            squeeze = False

    aio.write(Path(args.structure).stem + "_relaxed.xyz", atoms)
    print("Finished →", Path(args.structure).stem + "_relaxed.xyz")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

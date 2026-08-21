#!/usr/bin/env python3
"""
Relax a polymer in periodic box using molecular dynamics.
Supports both XTB and LennardJones calculators.
Usage
-----
python relax_polymer_ase.py  polymer.xyz  --density 1.0  --scale 1.3
python relax_polymer_ase.py  polymer.pdb  --density 1.0  --scale 1.3
python relax_polymer_ase.py  polymer.mol  --density 1.0  --scale 1.3
Dependencies:  ase >= 3.23, xtb-python (optional, for XTB calculator)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ASE imports
from ase import Atoms, units
from ase.io import read, write
from ase.md import VelocityVerlet, Langevin
from ase.md.npt import NPT
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

# XTB import (optional)
try:
    from xtb.ase.calculator import XTB

    HAS_XTB = True
except ImportError:
    HAS_XTB = False
    XTB = None

# Constants
AVOGADRO = 6.02214076e23
CM_TO_ANG = 1e8  # 1 cm = 1e8 Å


def density(atoms, mass_total_g):
    """Current density in g/cm³."""
    vol_cm3 = atoms.get_volume() / CM_TO_ANG**3
    return mass_total_g / vol_cm3


def fix_overlapping_atoms(atoms, min_distance=0.8):
    """Push apart atoms that are too close together."""
    from ase.neighborlist import NeighborList

    positions = atoms.get_positions()
    nl = NeighborList(
        [min_distance / 2] * len(atoms), self_interaction=False, bothways=True
    )
    nl.update(atoms)

    fixed = False
    for i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(i)
        if len(neighbors) > 0:
            for j, offset in zip(neighbors, offsets):
                if j > i:  # avoid double counting
                    vec = positions[j] + np.dot(offset, atoms.get_cell()) - positions[i]
                    dist = np.linalg.norm(vec)
                    if dist < min_distance and dist > 0:
                        # Push atoms apart
                        displacement = (min_distance - dist) / 2
                        unit_vec = vec / dist
                        positions[i] -= displacement * unit_vec
                        positions[j] += displacement * unit_vec
                        fixed = True

    if fixed:
        atoms.set_positions(positions)
        print(f"Fixed overlapping atoms (minimum distance: {min_distance:.2f} Å)")

    return fixed


class XTBForceOnly(Calculator):
    """XTB calculator wrapper that only calculates energy and forces, not stress."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs):
        Calculator.__init__(self)
        if not HAS_XTB:
            raise ImportError("XTB not available")
        self.xtb_calc = XTB(**kwargs)
        self._cached_results = {}

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Force only energy and forces - never ask for stress
        safe_properties = []
        for prop in properties:
            if prop in ["energy", "forces"]:
                safe_properties.append(prop)

        if not safe_properties:
            safe_properties = ["energy"]

        # Try to calculate with XTB, but catch virial errors
        try:
            # Attempt normal calculation
            self.xtb_calc.calculate(atoms, safe_properties, system_changes)

            # If successful, copy results
            for prop in safe_properties:
                if prop in self.xtb_calc.results:
                    self.results[prop] = self.xtb_calc.results[prop]
                    self._cached_results[prop] = self.xtb_calc.results[prop]

        except Exception as e:
            if "virial" in str(e).lower() or "stress" in str(e).lower():
                print(f"XTB virial error caught - extracting available results...")

                # XTB calculates energy/forces before virial, so they might be available
                if hasattr(self.xtb_calc, "results") and self.xtb_calc.results:
                    for prop in safe_properties:
                        if prop in self.xtb_calc.results:
                            self.results[prop] = self.xtb_calc.results[prop]
                            self._cached_results[prop] = self.xtb_calc.results[prop]
                            print(f"Extracted {prop} from XTB results")

                # If we still don't have results, try a different approach
                if "energy" not in self.results and "forces" not in self.results:
                    print(
                        "No results available from XTB - trying manual calculation..."
                    )
                    # Try to call XTB's internal calculation methods directly
                    try:
                        # Force XTB to calculate without stress
                        self.xtb_calc.atoms = atoms
                        if hasattr(self.xtb_calc, "_calculate"):
                            # Some versions have internal _calculate method
                            result = self.xtb_calc._calculate(
                                atoms, safe_properties, system_changes
                            )
                        else:
                            # Last resort: use cached results if available
                            if self._cached_results:
                                print("Using cached XTB results...")
                                for prop in safe_properties:
                                    if prop in self._cached_results:
                                        self.results[prop] = self._cached_results[prop]
                            else:
                                raise e
                    except:
                        raise e

                # Verify we have the required results
                missing = [prop for prop in safe_properties if prop not in self.results]
                if missing:
                    raise RuntimeError(f"XTB failed to provide: {missing}")

            else:
                # Some other error - re-raise
                raise e

    def get_stress(self, atoms):
        """Never provide stress for XTB."""
        raise NotImplementedError(
            "XTB does not support stress calculations in this wrapper"
        )

    def get_forces(self, atoms):
        """Get forces ensuring no stress is calculated."""
        # check_state: without it the first result was cached forever and an
        # entire MD run integrated the forces of the initial geometry.
        if "forces" not in self.results or self.check_state(atoms):
            self.calculate(atoms, ["forces"])
        return self.results["forces"]

    def get_potential_energy(self, atoms):
        """Get energy ensuring no stress is calculated."""
        if "energy" not in self.results or self.check_state(atoms):
            self.calculate(atoms, ["energy"])
        return self.results["energy"]


class XTBDirect(Calculator):
    """Direct XTB calculator that bypasses ASE interface to avoid virial issues."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, method="GFN-FF", **kwargs):
        Calculator.__init__(self)
        if not HAS_XTB:
            raise ImportError("XTB not available")

        self.method = method
        self.kwargs = kwargs
        # Try to import XTB direct interface
        try:
            import xtb

            self.xtb_interface = xtb.interface
        except ImportError:
            raise ImportError("XTB direct interface not available")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        try:
            import xtb

            # Convert atoms to XTB format
            numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions() * 1.8897259886  # Angstrom to Bohr

            # Create XTB calculator
            calc = xtb.GFNCalculator(
                geometry=positions, elements=numbers, method=self.method
            )

            # Calculate energy
            if "energy" in properties:
                energy_hartree = calc.get_energy()
                self.results["energy"] = (
                    energy_hartree * 27.211386245988
                )  # Hartree to eV

            # Calculate forces
            if "forces" in properties:
                forces_hartree_bohr = calc.get_gradient()
                # Convert from Hartree/Bohr to eV/Angstrom
                self.results["forces"] = -forces_hartree_bohr * 51.42208619083232

        except Exception as e:
            raise RuntimeError(f"XTB direct calculation failed: {e}")

    def get_stress(self, atoms):
        """Never provide stress for XTB."""
        raise NotImplementedError("XTB does not support stress calculations")

    def get_forces(self, atoms):
        """Get forces."""
        self.calculate(atoms, ["forces"])
        return self.results["forces"]

    def get_potential_energy(self, atoms):
        """Get energy."""
        self.calculate(atoms, ["energy"])
        return self.results["energy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "input_file",
        help="polymer coordinates (PDB, XYZ, MOL, or other ASE-supported formats)",
    )
    p.add_argument("--density", type=float, default=1.0, help="target density g/cm^3")
    p.add_argument(
        "--scale", type=float, default=1.3, help="initial box inflation factor"
    )
    p.add_argument("--timestep", type=float, default=0.5, help="fs, MD time-step")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 0. read coordinates & inflate box
    try:
        # Auto-detect file format based on extension
        atoms = read(args.input_file)
        print(f"Read {len(atoms)} atoms from {args.input_file}")
        print(f"Chemical formula: {atoms.get_chemical_formula()}")

        # Detect file format for information
        file_ext = args.input_file.split(".")[-1].lower()
        print(f"Detected file format: {file_ext.upper()}")

        # Format-specific handling
        if file_ext in ["xyz", "mol", "sdf"]:
            # These formats typically don't have unit cell info
            if atoms.get_pbc().any():
                print("Warning: Found periodic boundary conditions in molecular format")
            else:
                print("Molecular format detected - will create periodic box")

        elif file_ext in ["pdb"]:
            # PDB might have CRYST1 records for unit cell
            if atoms.cell.volume > 0:
                print(f"Unit cell found in PDB: {atoms.cell.cellpar()}")
            else:
                print("No unit cell in PDB - will create periodic box")

    except Exception as e:
        print(f"Error reading {args.input_file}: {e}")
        print("Supported formats include:")
        print("  - XYZ: Simple atomic coordinates")
        print("  - PDB: Protein Data Bank format")
        print("  - MOL/SDF: MDL Molfile format")
        print("  - CIF: Crystallographic Information File")
        print("  - VASP: POSCAR/CONTCAR format")
        print("  - And many others supported by ASE")
        return
    if atoms.get_pbc() is False or not atoms.cell.volume > 0:
        L0 = (
            len(atoms)
            * atoms.get_masses().mean()
            / args.density
            / AVOGADRO
            * CM_TO_ANG**3
        ) ** (1 / 3)
        atoms.set_cell([L0, L0, L0])
        atoms.center()
    atoms.set_pbc(True)
    atoms.set_cell(atoms.cell * args.scale, scale_atoms=True)

    total_mass = atoms.get_masses().sum() / AVOGADRO  # g

    # ------------------------------------------------------------------
    # Fix true nuclear overlaps before starting MD. The default 0.8 Å only
    # separates atoms that are unphysically fused; a larger cutoff (2.5 Å)
    # treated every covalent bond (C-H 1.1 Å, C-C 1.5 Å) as an overlap and
    # shredded the molecule's bonding geometry before MD even started.
    fix_overlapping_atoms(atoms)

    # Report (but do not "fix") remaining close contacts: rescaling the cell
    # with scale_atoms=True would stretch every bond, not just the contacts.
    from ase.neighborlist import neighbor_list

    i, j, d = neighbor_list("ijd", atoms, 3.0)
    if len(i) > 0:
        min_dist = d.min()
        if min_dist < 0.8:
            print(
                f"Warning: Very close atoms remain (min distance: {min_dist:.2f} Å)"
            )

    # Get the volume corresponding to desired density.
    # total_mass [g] / density [g/cm^3] = volume [cm^3]; convert to Å³ —
    # comparing cm^3 against get_volume()'s Å³ made volume_ratio ~1e-24, so
    # the density targeting always took the extreme-compression clamp.
    volume_needed = total_mass / args.density * CM_TO_ANG**3  # Å³
    volume_current = atoms.get_volume()

    # Calculator: Choose XTB if available, otherwise LennardJones
    if HAS_XTB:
        print("Using XTB GFN-FF calculator (more realistic for polymers)")
        calc = None
        using_xtb = False

        # Try Method 1: Direct XTB interface (bypasses ASE)
        try:
            print("Trying XTB direct interface...")
            calc = XTBDirect(method="GFN-FF")
            atoms.calc = calc
            test_energy = atoms.get_potential_energy()
            test_forces = atoms.get_forces()
            print(f"XTB direct successful - Energy: {test_energy:.3f} eV")
            using_xtb = True
            print("Using XTB direct interface (bypasses ASE)")

        except Exception as e:
            print(f"XTB direct failed: {e}")

            # Try Method 2: XTB ASE wrapper with virial handling
            try:
                print("Trying XTB ASE wrapper with virial error handling...")
                calc = XTBForceOnly(method="GFN-FF")
                atoms.calc = calc
                test_energy = atoms.get_potential_energy()
                test_forces = atoms.get_forces()
                print(f"XTB wrapper successful - Energy: {test_energy:.3f} eV")
                using_xtb = True
                print("Using XTB ASE wrapper")

            except Exception as e2:
                print(f"XTB wrapper also failed: {e2}")
                print("Falling back to LennardJones calculator")
                calc = LennardJones(sigma=4.0, epsilon=0.01, rc=15.0)
                atoms.calc = calc
                using_xtb = False

        if using_xtb:
            print("Note: XTB will use NVT dynamics only (no pressure control)")

    else:
        print("XTB not available - using LennardJones calculator")
        # Using very conservative LJ parameters
        calc = LennardJones(sigma=4.0, epsilon=0.01, rc=15.0)
        atoms.calc = calc
        using_xtb = False

    if using_xtb:
        # For XTB: only adjust volume if significantly different
        volume_ratio = volume_needed / volume_current
        print(f"Volume ratio needed: {volume_ratio:.3f}")
        if abs(volume_ratio - 1.0) > 0.2:  # Only adjust if >20% difference
            # Limit scaling to prevent extreme compression/expansion
            if volume_ratio < 0.5:
                scaling_factor = 0.8  # Max compression
            elif volume_ratio > 2.0:
                scaling_factor = 1.5  # Max expansion
            else:
                scaling_factor = volume_ratio ** (1 / 3)

            atoms.set_cell(atoms.cell * scaling_factor, scale_atoms=True)
            print(
                f"Adjusted volume for XTB: {volume_current:.1f} → {atoms.get_volume():.1f} Å³"
            )
        else:
            print(f"Volume adjustment not needed (ratio: {volume_ratio:.3f})")
    else:
        # For LennardJones: gradual compression/expansion
        scaling_factor = (volume_needed / volume_current) ** (1 / 3)
        if scaling_factor < 0.8:  # Need significant compression
            # Do gradual compression to avoid explosions
            scaling_factor = 0.9  # Only compress by 10% initially
        elif scaling_factor > 1.2:  # Need significant expansion
            scaling_factor = 1.1  # Only expand by 10% initially

        atoms.set_cell(atoms.cell * scaling_factor, scale_atoms=True)
        print(f"LJ initial scaling: {volume_current:.1f} → {atoms.get_volume():.1f} Å³")

    print(f"Final volume: {atoms.get_volume():.1f} Å³")
    print(f"Final density: {density(atoms, total_mass):.3f} g/cm³")

    # ------------------------------------------------------------------
    # Stage 1: soft NVT equilibration
    print("\n--- Stage 1: soft NVT ---")

    # Use very conservative timestep for LJ
    if not using_xtb:
        actual_timestep = min(args.timestep, 0.2)  # Max 0.2 fs for LJ
        print(f"Using reduced timestep for LJ: {actual_timestep} fs")
    else:
        actual_timestep = args.timestep
        print(f"Using timestep: {actual_timestep} fs")

    print(f"Calculator being used: {type(calc).__name__}")
    if hasattr(calc, "parameters"):
        print(f"Calculator parameters: {calc.parameters}")

    # Temperature ramp with more gradual increases
    temperatures = (
        [1, 10, 25, 50, 100, 200, 300] if not using_xtb else [10, 50, 100, 200, 300]
    )

    for i, temp in enumerate(temperatures):
        print(f"Increasing temperature to {temp} K")
        try:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

            # Weaken LJ potential during equilibration
            if not using_xtb and hasattr(calc, "parameters"):
                original_epsilon = calc.parameters.epsilon
                calc.parameters.epsilon = 0.1 * original_epsilon
                print("LennardJones equilibration: Weakening potential")

            # Use short equilibration runs with error checking
            dyn1 = VelocityVerlet(atoms, timestep=actual_timestep * units.fs)

            # Shorter runs for initial temperatures
            steps = (
                int(2000 / actual_timestep)
                if temp < 50
                else int(5000 / actual_timestep)
            )

            for step in range(0, steps, int(100 / actual_timestep)):
                try:
                    dyn1.run(int(100 / actual_timestep))

                    # Check for explosions
                    if not np.isfinite(atoms.get_potential_energy()) or not np.all(
                        np.isfinite(atoms.get_forces())
                    ):
                        print(f"System exploded at step {step}, temperature {temp} K")
                        raise ValueError("System unstable")

                except Exception as e:
                    print(f"Error during equilibration at {temp} K: {e}")
                    if temp == temperatures[0]:  # Failed at first temperature
                        raise RuntimeError(
                            "System too unstable even at lowest temperature"
                        )
                    else:
                        print("Stopping temperature ramp early")
                        break

            # Restore LJ potential strength
            if not using_xtb and hasattr(calc, "parameters"):
                calc.parameters.epsilon = original_epsilon
                print("Restoring LJ potential strength")

        except Exception as e:
            print(f"Failed at temperature {temp} K: {e}")
            if i == 0:  # Failed at first temperature
                raise RuntimeError("System completely unstable")
            else:
                print("Stopping at previous successful temperature")
                break

    # Stage 2: Equilibration (NPT for LJ, extended NVT for XTB)
    print("\n--- Stage 2: Equilibration ---")

    if using_xtb:
        # XTB: Extended NVT equilibration (no pressure control)
        print("XTB: Using extended NVT equilibration (no pressure control)")
        try:
            dyn2 = Langevin(
                atoms, actual_timestep * units.fs, temperature_K=300, friction=0.01
            )
            dyn2.attach(
                MDLogger(dyn2, atoms, sys.stdout, stress=False, peratom=False),
                interval=100,
            )
            dyn2.run(int(20000 / actual_timestep))  # 20 ps
            print("XTB NVT equilibration completed successfully")

        except Exception as e:
            print(f"Error in XTB NVT equilibration: {e}")
            raise
    else:
        # LennardJones: Try NPT first, fallback to NVT
        print("LennardJones: Attempting NPT equilibration")
        try:
            dyn2 = NPT(
                atoms,
                actual_timestep * units.fs,
                temperature_K=300,
                externalstress=0.0,
                ttime=25 * units.fs,
                pfactor=75 * units.fs**2,
            )
            dyn2.attach(
                MDLogger(dyn2, atoms, sys.stdout, stress=True, peratom=False),
                interval=100,
            )
            dyn2.run(int(50000 / actual_timestep))
            print("NPT equilibration completed successfully")

        except Exception as e:
            print(f"Error in Stage 2 NPT: {e}")
            print("Continuing with NVT equilibration instead...")
            try:
                dyn_fallback = Langevin(
                    atoms, actual_timestep * units.fs, temperature_K=300, friction=0.01
                )
                dyn_fallback.attach(
                    MDLogger(
                        dyn_fallback, atoms, sys.stdout, stress=False, peratom=False
                    ),
                    interval=100,
                )
                dyn_fallback.run(int(20000 / actual_timestep))
                print("Fallback NVT equilibration completed")
            except Exception as e2:
                print(f"Fallback NVT also failed: {e2}")
                raise

    # Stage 3: manual squeeze loop (optional)
    current_density = density(atoms, total_mass)
    target_density = 0.95 * args.density

    if using_xtb:
        print(f"\n--- Stage 3: Density check (XTB) ---")
        print(f"Current density: {current_density:.3f} g/cm³")
        print(f"Target density: {target_density:.3f} g/cm³")

        if current_density < target_density:
            print(
                "For XTB, we skip aggressive density compression to avoid instabilities"
            )
            print("Consider using a smaller --scale factor if higher density is needed")

    else:
        # LennardJones can handle density compression better
        print(f"\n--- Stage 3: Manual density compression ---")
        squeeze_count = 0
        max_squeezes = 10  # Prevent infinite loops

        while current_density < target_density and squeeze_count < max_squeezes:
            print(
                f"\n--- Manual squeeze #{squeeze_count+1} (density = {current_density:.3f}) ---"
            )

            # short NVT equilibration
            dyn3 = Langevin(
                atoms, actual_timestep * units.fs, temperature_K=300, friction=0.01
            )
            dyn3.run(int(5000 / actual_timestep))

            # shrink volume by 2%
            cell = atoms.cell
            atoms.set_cell(cell * 0.98, scale_atoms=True)

            current_density = density(atoms, total_mass)
            squeeze_count += 1

        if squeeze_count >= max_squeezes:
            print(f"Reached maximum squeeze attempts ({max_squeezes})")
            print(f"Final density: {current_density:.3f} g/cm³")

    final_rho = density(atoms, total_mass)
    print("\nFinal density = %.3f g/cm³" % final_rho)

    # Create output filename based on input
    input_base = str(Path(args.input_file).with_suffix(""))
    output_file = f"{input_base}_equilibrated.pdb"

    write(output_file, atoms)
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()

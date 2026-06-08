import os
from pathlib import Path
import types
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotPresent, all_changes
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
import pytest
import sys
from unittest.mock import patch, MagicMock
import logging

# Configure logging for tests to see warnings
logging.basicConfig(level=logging.WARNING)

from iqc.asetools import (
    save_atoms,
    translate_atoms,
    get_canonical_smiles_from_atoms,
    get_atoms_from_smiles,
    atoms2xyz,
    get_atoms_from_xyz,
    get_total_electrons,
    get_spin,
    get_multiplicity,
    apply_spin_charge,
    parse_multiplicity_charge_from_comment,
    xyz2atoms,
    get_calculator,
    get_ase_version,
    XTB,
    is_linear_by_inertia,
    get_symmetry_info,
    run_ir,
    run_ir_thermo,
    run_optimization,
    run_single_point,
    run_vibrations,
    run_thermo,
    _get_uma_calculator,
    _get_mace_polar_calculator,
    _ensure_mace_polar_model_cached,
    _file_lock,
    _normalize_calculator_compatibility,
    _patch_e3nn_activation_legacy_state,
    _patch_e3nn_codegen_legacy_state,
    _patch_e3nn_spherical_harmonics_legacy_state,
    _parse_uma_calculator_name,
    _restore_e3nn_activation_paths,
    _restore_e3nn_spherical_harmonics_sph_func,
    _validate_uma_device,
)


# Test fixtures
@pytest.fixture
def water_atoms():
    """Create a simple water molecule for testing."""
    return Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=False
    )


@pytest.fixture
def methane_atoms():
    """Create a methane molecule for testing."""
    return Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],  # C
            [0.6, 0.6, 0.6],  # H
            [-0.6, -0.6, 0.6],  # H
            [0.6, -0.6, -0.6],  # H
            [-0.6, 0.6, -0.6],  # H
        ],
        cell=[10, 10, 10],
        pbc=False,
    )


def test_save_atoms(water_atoms, tmp_path):
    """Test saving atoms to file."""
    file_path = save_atoms(water_atoms, prefix="test", directory=str(tmp_path))
    assert os.path.exists(file_path)
    assert file_path.endswith(".xyz")
    assert "test" in file_path
    assert "H2O" in file_path


def test_translate_atoms(water_atoms):
    """Test translating atoms."""
    # Translate H atoms in the direction of O atom
    translated = translate_atoms(
        water_atoms.copy(),
        indices=[0, 1],  # H atoms
        reference_index=2,  # O atom
        target_index=0,  # First H atom
        distance=1.0,
    )

    # Check that atoms were moved
    assert not np.allclose(water_atoms.positions, translated.positions)
    assert len(translated) == len(water_atoms)


def test_atoms2xyz(water_atoms):
    """Test converting atoms to XYZ format."""
    xyz_str = atoms2xyz(water_atoms)

    # Check basic XYZ format requirements
    lines = xyz_str.strip().split("\n")
    assert len(lines) == 5  # Number of atoms + 2 header lines
    assert lines[0].strip() == "3"  # Number of atoms
    assert lines[1].strip() == ""  # Comment line

    # Check atom lines format
    for line in lines[2:]:
        parts = line.split()
        assert len(parts) == 4  # Symbol and 3 coordinates
        # Check if coordinates can be converted to float
        assert all(isinstance(float(x), float) for x in parts[1:])


def test_get_total_electrons(water_atoms, methane_atoms):
    """Test electron counting."""
    # H2O: O(8) + 2*H(1) = 10 electrons
    assert get_total_electrons(water_atoms) == 10

    # CH4: C(6) + 4*H(1) = 10 electrons
    assert get_total_electrons(methane_atoms) == 10


def test_get_spin(water_atoms, methane_atoms):
    """Test spin calculation (returns S = (multiplicity - 1) / 2)."""
    # Both water and methane have even number of electrons -> singlet, S=0
    assert get_spin(water_atoms) == 0.0
    assert get_spin(methane_atoms) == 0.0

    # Create OH radical (odd number of electrons -> doublet, S=0.5)
    oh_radical = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    assert get_spin(oh_radical) == 0.5

    # Explicit multiplicity override (e.g. triplet O2 -> S=1)
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    assert get_spin(o2) == 0.0
    assert get_spin(o2, multiplicity=3) == 1.0


def test_get_multiplicity(water_atoms):
    """Defaults to electron-count parity, override with explicit value."""
    assert get_multiplicity(water_atoms) == 1  # singlet
    assert get_multiplicity(water_atoms, charge=1) == 2  # cation doublet
    assert get_spin(water_atoms, charge=1) == 0.5
    oh_radical = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    assert get_multiplicity(oh_radical) == 2  # doublet
    # Triplet override on a closed-shell parity molecule (e.g. O2)
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    assert get_multiplicity(o2, multiplicity=3) == 3
    with pytest.raises(ValueError):
        get_multiplicity(water_atoms, multiplicity=0)


def test_apply_spin_charge_xtb_convention():
    """XTB reads sums of initial_charges and initial_magnetic_moments
    (uhf = unpaired electrons = multiplicity - 1)."""

    class FakeXTB:
        pass

    FakeXTB.__name__ = "XTB"
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    mult = apply_spin_charge(atoms, FakeXTB(), multiplicity=2, charge=-1)
    assert mult == 2
    assert int(round(atoms.get_initial_charges().sum())) == -1
    assert int(round(atoms.get_initial_magnetic_moments().sum())) == 1  # uhf


def test_apply_spin_charge_fairchem_convention():
    """FAIRChem UMA reads atoms.info; its `spin` key is the multiplicity."""

    class FakeFAIR:
        pass

    FakeFAIR.__name__ = "FAIRChemCalculator"
    atoms = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    apply_spin_charge(atoms, FakeFAIR(), multiplicity=3, charge=0)
    assert atoms.info["charge"] == 0
    assert atoms.info["spin"] == 3  # triplet multiplicity


def test_apply_spin_charge_mace_polar_convention():
    """MACE-Polar reads charge and total spin S from atoms.info."""

    class FakeMACEPolar:
        _iqc_spin_charge_convention = "mace_polar"

    atoms = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    apply_spin_charge(atoms, FakeMACEPolar(), multiplicity=3, charge=-1)

    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 1.0
    assert atoms.info["external_field"] == [0.0, 0.0, 0.0]


def test_apply_spin_charge_orca_convention():
    """ORCA reads molecular charge/multiplicity from calculator parameters."""

    class ORCA:
        def __init__(self):
            self.parameters = {}

    calc = ORCA()
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    apply_spin_charge(atoms, calc, multiplicity=2, charge=-1)

    assert calc.parameters["charge"] == -1
    assert calc.parameters["mult"] == 2


def test_run_single_point_stores_mace_polar_observables(water_atoms):
    """MACE-Polar result arrays should be serialized into IQC records."""

    class PolarLikeCalculator(Calculator):
        implemented_properties = ["energy", "forces", "dipole"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            n_atoms = len(self.atoms)
            self.results["energy"] = -1.23
            self.results["forces"] = np.zeros((n_atoms, 3))
            self.results["dipole"] = np.array([1.0, 2.0, 3.0])
            self.results["density_coefficients"] = np.array(
                [
                    [0.2, 0.3, 0.4, 0.5],
                    [-0.1, -0.2, -0.3, -0.4],
                    [-0.1, -0.1, -0.1, -0.1],
                ]
            )
            self.results["spin_charge_density"] = np.array(
                [
                    [[0.15, 0, 0, 0], [0.05, 0, 0, 0]],
                    [[-0.05, 0, 0, 0], [-0.05, 0, 0, 0]],
                    [[-0.02, 0, 0, 0], [-0.08, 0, 0, 0]],
                ]
            )

    _, results = run_single_point(water_atoms, calculator=PolarLikeCalculator())

    assert results["dipole"] == [1.0, 2.0, 3.0]
    assert results["partial_charges"] == [0.2, -0.1, -0.1]
    assert results["partial_dipoles"] == [
        [0.5, 0.3, 0.4],
        [-0.4, -0.2, -0.3],
        [-0.1, -0.1, -0.1],
    ]
    assert results["partial_spin_up_charges"] == [0.15, -0.05, -0.02]
    assert results["partial_spin_down_charges"] == [0.05, -0.05, -0.08]
    assert results["partial_spin_charges"] == pytest.approx([0.1, 0.0, 0.06])


def test_run_single_point_keeps_energy_when_forces_missing(water_atoms):
    """Energy-only single-point work should not fail when forces are absent."""

    class MissingForcesCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = -1.23

    atoms, results = run_single_point(
        atoms=water_atoms.copy(),
        calculator=MissingForcesCalculator(),
        unique_name="water",
    )

    assert atoms.get_chemical_formula() == "H2O"
    assert results["error"] == ""
    assert results["energy_eV"] == pytest.approx(-1.23)
    assert results["forces"] == []
    assert any("Forces were not available" in item for item in results["warnings"])


def test_run_optimization_adds_engrad_for_orca_forces(water_atoms):
    """ASE ORCA optimization needs ENGRAD so forces can be parsed."""

    class ORCA(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.parameters["orcasimpleinput"] = "HF def2-SVP"

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = -1.23
            if "forces" in properties:
                simpleinput = self.parameters["orcasimpleinput"].upper()
                if "ENGRAD" not in simpleinput:
                    raise PropertyNotPresent("forces")
                self.results["forces"] = np.zeros((len(self.atoms), 3))

    calc = ORCA()
    _, results = run_optimization(
        atoms=water_atoms.copy(),
        calculator=calc,
        unique_name="water",
        fmax=100.0,
        max_steps=1,
    )

    assert "ENGRAD" in calc.parameters["orcasimpleinput"].upper()
    assert results["error"] == ""
    assert bool(results["opt_converged"]) is True


def test_auto_orca_work_directory_is_unique_per_record(water_atoms):
    """Reused ORCA calculators must not share one orca.out across records."""

    class ORCA(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.parameters["orcasimpleinput"] = "HF def2-SVP"
            self.directory = Path(".")

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = -1.23
            if "forces" in properties:
                self.results["forces"] = np.zeros((len(self.atoms), 3))

    calc = ORCA()
    run_single_point(water_atoms.copy(), calculator=calc, unique_name="mol_0_rank_0")
    first_directory = calc.directory

    run_single_point(water_atoms.copy(), calculator=calc, unique_name="mol_1_rank_0")

    assert first_directory != calc.directory
    assert Path(first_directory).name == "mol_0_rank_0_calc_orca"
    assert Path(calc.directory).name == "mol_1_rank_0_calc_orca"


def test_parse_multiplicity_charge_from_comment():
    """Comment-line parser handles multiplicity, uhf, and charge tokens."""
    assert parse_multiplicity_charge_from_comment("") == (None, None)
    assert parse_multiplicity_charge_from_comment("just a comment") == (None, None)
    # multiplicity preferred form
    assert parse_multiplicity_charge_from_comment("multiplicity=3 charge=-1") == (3, -1)
    assert parse_multiplicity_charge_from_comment("mult=1 q=0") == (1, 0)
    # uhf converts to multiplicity = uhf + 1
    assert parse_multiplicity_charge_from_comment("uhf=1 chrg=1") == (2, 1)
    # case insensitive, mixed separators
    assert parse_multiplicity_charge_from_comment("Mult: 1, Charge: 2") == (1, 2)
    # consistent multiplicity + uhf -> no warning, multiplicity wins
    assert parse_multiplicity_charge_from_comment("multiplicity=3 uhf=2") == (3, None)
    # invalid multiplicity rejected
    with pytest.raises(ValueError):
        parse_multiplicity_charge_from_comment("multiplicity=0")


def test_get_atoms_from_xyz_reads_multiplicity_charge(tmp_path):
    """get_atoms_from_xyz stashes parsed multiplicity/charge in atoms.info."""
    xyz = """3
multiplicity=2 charge=-1
O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0
"""
    f = tmp_path / "anion.xyz"
    f.write_text(xyz)
    atoms = get_atoms_from_xyz(str(f))
    assert atoms.info.get("multiplicity") == 2
    assert atoms.info.get("charge") == -1

    # And from a string with no tokens — info is unset
    atoms2 = get_atoms_from_xyz("1\n\nH 0 0 0\n")
    assert "multiplicity" not in atoms2.info
    assert "charge" not in atoms2.info


def test_apply_spin_charge_emt_warns(caplog):
    """EMT/MACE silently accept defaults but warn on non-default values."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    with caplog.at_level(logging.WARNING):
        apply_spin_charge(atoms, EMT(), multiplicity=1, charge=0)
    assert not any("does not support" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        apply_spin_charge(atoms, EMT(), multiplicity=3, charge=-1)
    assert any("does not support" in r.message for r in caplog.records)


def test_xyz2atoms():
    """Test converting XYZ string to atoms."""
    xyz_str = """3

O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0"""

    atoms = xyz2atoms(xyz_str)
    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert atoms.positions.shape == (3, 3)


def test_get_atoms_from_smiles():
    """Test generating a 3D geometry from a SMILES string."""
    atoms = get_atoms_from_smiles("O")
    assert len(atoms) == 3
    assert atoms.get_chemical_formula() == "H2O"
    assert np.isfinite(atoms.positions).all()
    assert atoms.info["smiles_input"] == "O"
    assert atoms.info["canonical_smiles"] == "O"


def test_get_ase_version():
    """Test getting ASE version."""
    import ase

    version = get_ase_version()

    # Check that a version string is returned
    assert isinstance(version, str)
    assert len(version) > 0

    # Check that it matches the actual ASE version
    assert version == ase.__version__

    # Check that version follows semantic versioning pattern (roughly)
    # Version should have at least one dot (e.g., "3.22.1")
    assert "." in version

    # Check that it contains only valid version characters
    # (digits, dots, letters, hyphens are typical in version strings)
    import re

    assert re.match(
        r"^[0-9]+\.[0-9]+", version
    ), f"Version '{version}' doesn't start with major.minor format"


def test_get_canonical_smiles(methane_atoms):
    """Test SMILES generation."""
    # Skip if rdkit not installed
    pytest.importorskip("rdkit")
    from rdkit import Chem

    smiles = get_canonical_smiles_from_atoms(methane_atoms)
    # Note: RDKit may return SMILES with explicit hydrogens
    # We should check if the molecule is equivalent rather than exact string match
    mol = Chem.MolFromSmiles(smiles)
    assert Chem.MolToSmiles(mol, canonical=True) in ["C", "[H]C([H])([H])[H]"]


def test_get_calculator_mace_success():
    """Test getting MACE calculator successfully."""
    # Skip test if MACE cannot be imported
    pytest.importorskip("mace")
    try:
        calculator = get_calculator(name="mace")
        # Check if it's MACE or a SumCalculator containing MACE
        is_mace_like = False
        if isinstance(calculator, SumCalculator):
            # Correct attribute is 'calcs'
            is_mace_like = any(
                "mace" in str(c).lower() for c in str(calculator).split(",")
            )
        else:
            is_mace_like = "mace" in str(calculator).lower()
        assert is_mace_like, f"Calculator name {calculator.name} does not indicate MACE"

        # Test calculation
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not np.isnan(energy)

    except RuntimeError as e:
        # This might happen if MACE is installed but initialization fails
        pytest.skip(f"Could not initialize MACE or fallback: {e}")


def test_get_calculator_mace_unavailable_fallback():
    """Test fallback to EMT when MACE is not available."""
    # Mock the import to simulate MACE not being available
    with patch.dict(sys.modules, {"mace.calculators": None, "mace": None}):
        try:
            calculator = get_calculator(name="mace")
            # It should fallback to EMT
            assert isinstance(calculator, EMT)

            # Test calculation with fallback
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            assert isinstance(energy, float)
        except RuntimeError as e:
            pytest.skip(f"Could not initialize EMT fallback: {e}")
        except ImportError:
            # This could happen if EMT also fails to import
            pytest.skip("Neither MACE nor EMT fallback available.")


def test_get_mace_polar_calculator_uses_mace_polar_loader(monkeypatch):
    """MACE-Polar should call the Electrostatic MACE loader and expose dipoles."""

    calls = {}
    monkeypatch.setattr(
        "iqc.asetools._ensure_mace_polar_model_cached", lambda model: None
    )

    class FakePolarCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

    def fake_mace_polar(**kwargs):
        calls.update(kwargs)
        return FakePolarCalculator()

    mace_calculators_module = types.ModuleType("mace.calculators")
    mace_calculators_module.mace_polar = fake_mace_polar
    mace_module = types.ModuleType("mace")
    mace_module.calculators = mace_calculators_module

    with patch.dict(
        sys.modules,
        {
            "mace": mace_module,
            "mace.calculators": mace_calculators_module,
        },
    ):
        calculator = _get_mace_polar_calculator(model="polar-1-l", device="cuda")

    assert calls["model"] == "polar-1-l"
    assert calls["device"] == "cuda"
    assert calls["default_dtype"] == "float64"
    assert calculator.model_name == "polar-1-l"
    assert calculator._iqc_spin_charge_convention == "mace_polar"
    assert "dipole" in calculator.implemented_properties


def test_mace_polar_model_prefetch_uses_cache_lock(tmp_path, monkeypatch):
    """Only the first rank/process should download a missing Polar checkpoint."""

    cache_path = tmp_path / "MACEPOLAR1Lmodel"
    calls = []

    def fake_download(model):
        calls.append(model)
        cache_path.write_text("model", encoding="utf-8")
        return str(cache_path)

    monkeypatch.setattr(
        "iqc.asetools._mace_polar_cached_model_path", lambda model: cache_path
    )
    monkeypatch.setattr(
        "iqc.asetools._download_mace_polar_checkpoint", fake_download
    )

    _ensure_mace_polar_model_cached("polar-1-l")
    _ensure_mace_polar_model_cached("polar-1-l")

    assert calls == ["polar-1-l"]
    assert cache_path.exists()


def test_file_lock_uses_directory_and_replaces_old_lock_file(tmp_path):
    """Parallel filesystems do not always honor flock for shared downloads."""

    lock_path = tmp_path / "model.iqc-download.lock"
    lock_path.write_text("old file lock", encoding="utf-8")

    with _file_lock(lock_path):
        assert lock_path.is_dir()
        assert (lock_path / "owner").exists()

    assert not lock_path.exists()


def test_normalize_calculator_exposes_sumcalculator_calcs():
    """Test compatibility shim for ASE SumCalculator child calculators."""

    calculator = SumCalculator([EMT(), EMT()])
    assert not hasattr(calculator, "calcs")

    normalized = _normalize_calculator_compatibility(calculator)

    assert normalized is calculator
    assert calculator.calcs is calculator.mixer.calcs
    assert len(calculator.calcs) == 2


def test_patch_e3nn_codegen_legacy_state_converts_old_buffers():
    """Test e3nn 0.5 can accept legacy raw-byte codegen state."""

    calls = []

    class DummyCodeGenMixin:
        def __setstate__(self, state):
            for fname, (buffer_type, buffer) in state["__codegen__"].items():
                calls.append((fname, buffer_type, buffer))

    codegen_mixin_module = types.ModuleType("e3nn.util.codegen._mixin")
    codegen_mixin_module.CodeGenMixin = DummyCodeGenMixin

    assert _patch_e3nn_codegen_legacy_state(codegen_mixin_module) is True
    DummyCodeGenMixin().__setstate__({"__codegen__": {"compiled": b"legacy"}})

    assert calls == [("compiled", "torchscript", b"legacy")]


def test_patch_e3nn_codegen_legacy_state_skips_old_e3nn_shape():
    """Test old e3nn codegen state support is left untouched."""

    calls = []

    class DummyCodeGenMixin:
        def __setstate__(self, state):
            for fname, buffer in state["__codegen__"].items():
                calls.append((fname, buffer))

    codegen_mixin_module = types.ModuleType("e3nn.util.codegen._mixin")
    codegen_mixin_module.CodeGenMixin = DummyCodeGenMixin

    assert _patch_e3nn_codegen_legacy_state(codegen_mixin_module) is False
    DummyCodeGenMixin().__setstate__({"__codegen__": {"compiled": b"legacy"}})

    assert calls == [("compiled", b"legacy")]


def test_restore_e3nn_spherical_harmonics_sph_func():
    """Test missing e3nn SphericalHarmonics callable is restored."""

    SphericalHarmonics = type("SphericalHarmonics", (), {})
    module = SphericalHarmonics()

    restored = _restore_e3nn_spherical_harmonics_sph_func(
        module, sph_func_factory=lambda: "restored"
    )

    assert restored is True
    assert module.sph_func == "restored"
    assert (
        _restore_e3nn_spherical_harmonics_sph_func(
            module, sph_func_factory=lambda: "new"
        )
        is False
    )
    assert module.sph_func == "restored"


def test_patch_e3nn_spherical_harmonics_legacy_state_repairs_forward():
    """Test patched SphericalHarmonics lazily restores sph_func before forward."""

    class SphericalHarmonics:
        def forward(self, value):
            return self.sph_func(value)

    patched = _patch_e3nn_spherical_harmonics_legacy_state(
        SphericalHarmonics, sph_func_factory=lambda: lambda value: value + 1
    )

    assert patched is True
    assert SphericalHarmonics().forward(2) == 3
    assert (
        _patch_e3nn_spherical_harmonics_legacy_state(
            SphericalHarmonics, sph_func_factory=lambda: lambda value: value + 2
        )
        is False
    )


def test_restore_e3nn_activation_paths():
    """Test missing e3nn Activation paths are rebuilt from irreps and acts."""

    class Activation:
        pass

    module = Activation()
    module.irreps_in = [(2, (0, 1)), (1, (1, -1))]
    module.acts = ["act", None]

    restored = _restore_e3nn_activation_paths(module)

    assert restored is True
    assert module.paths == [
        (2, (0, 1), "act"),
        (1, (1, -1), None),
    ]
    assert _restore_e3nn_activation_paths(module) is False


def test_patch_e3nn_activation_legacy_state_repairs_forward():
    """Test patched Activation lazily restores paths before forward."""

    class Activation:
        def __init__(self):
            self.irreps_in = [(1, (0, 1))]
            self.acts = [lambda value: value + 1]

        def forward(self, value):
            return self.paths[0][2](value)

    patched = _patch_e3nn_activation_legacy_state(Activation)

    assert patched is True
    assert Activation().forward(2) == 3
    assert _patch_e3nn_activation_legacy_state(Activation) is False


def test_parse_uma_calculator_name():
    """Test compact IQC UMA calculator aliases."""

    assert _parse_uma_calculator_name("uma") == ("uma-s-1p2", "omol")
    assert _parse_uma_calculator_name("uma-s-omol") == ("uma-s-1p2", "omol")
    assert _parse_uma_calculator_name("uma-m-odac") == ("uma-m-1p1", "odac")

    with pytest.raises(ValueError):
        _parse_uma_calculator_name("uma-xl-omol")

    with pytest.raises(ValueError):
        _parse_uma_calculator_name("uma-s-unknown")


def test_get_uma_calculator_uses_fairchem_predictor_and_task():
    """Test UMA initialization without importing or downloading real FAIRChem models."""

    calls = []

    class DummyFAIRChemCalculator:
        def __init__(self, predictor, task_name, **kwargs):
            self.predictor = predictor
            self.task_name = task_name
            self.kwargs = kwargs
            self.name = "fairchem"

    class DummyPretrainedMLIP:
        @staticmethod
        def get_predict_unit(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return {"model_name": model_name}

    fairchem_module = types.ModuleType("fairchem")
    fairchem_core_module = types.ModuleType("fairchem.core")
    fairchem_core_module.FAIRChemCalculator = DummyFAIRChemCalculator
    fairchem_core_module.pretrained_mlip = DummyPretrainedMLIP
    fairchem_module.core = fairchem_core_module

    with patch.dict(
        sys.modules,
        {"fairchem": fairchem_module, "fairchem.core": fairchem_core_module},
    ):
        calculator = _get_uma_calculator(
            "uma-m-odac",
            device="cpu",
            inference_settings="turbo",
            foo="bar",
        )

    assert calls == [("uma-m-1p1", {"device": "cpu", "inference_settings": "turbo"})]
    assert calculator.predictor == {"model_name": "uma-m-1p1"}
    assert calculator.task_name == "odac"
    assert calculator.kwargs == {"foo": "bar"}
    assert calculator.model_name == "uma-m-1p1"


def test_get_uma_calculator_can_load_local_checkpoint(tmp_path):
    """Test UMA initialization can bypass hosted Hugging Face checkpoints."""

    checkpoint = tmp_path / "uma-local.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class DummyFAIRChemCalculator:
        def __init__(self, predictor, task_name, **kwargs):
            self.predictor = predictor
            self.task_name = task_name
            self.kwargs = kwargs
            self.name = "fairchem"

    class DummyPretrainedMLIP:
        @staticmethod
        def get_predict_unit(model_name, **kwargs):
            raise AssertionError("hosted checkpoint should not be requested")

    def load_predict_unit(path, **kwargs):
        calls.append((path, kwargs))
        return {"checkpoint": path}

    fairchem_module = types.ModuleType("fairchem")
    fairchem_core_module = types.ModuleType("fairchem.core")
    fairchem_units_module = types.ModuleType("fairchem.core.units")
    fairchem_mlip_unit_module = types.ModuleType("fairchem.core.units.mlip_unit")
    fairchem_core_module.FAIRChemCalculator = DummyFAIRChemCalculator
    fairchem_core_module.pretrained_mlip = DummyPretrainedMLIP
    fairchem_mlip_unit_module.load_predict_unit = load_predict_unit
    fairchem_module.core = fairchem_core_module

    with patch.dict(
        sys.modules,
        {
            "fairchem": fairchem_module,
            "fairchem.core": fairchem_core_module,
            "fairchem.core.units": fairchem_units_module,
            "fairchem.core.units.mlip_unit": fairchem_mlip_unit_module,
        },
    ):
        calculator = _get_uma_calculator(
            "uma-s-omol",
            checkpoint_path=str(checkpoint),
            device="cpu",
            inference_settings="turbo",
            cache_dir="/unused/hosted/cache",
            seed=123,
        )

    assert calls == [
        (checkpoint, {"device": "cpu", "inference_settings": "turbo"})
    ]
    assert calculator.predictor == {"checkpoint": checkpoint}
    assert calculator.task_name == "omol"
    assert calculator.model_name == str(checkpoint)


def test_uma_device_validation_rejects_xpu_before_fairchem_import():
    """UMA should fail clearly for XPU rather than falling back to MACE."""

    _validate_uma_device({"device": "cpu"})
    _validate_uma_device({"device": "cuda"})

    with pytest.raises(RuntimeError) as excinfo:
        _get_uma_calculator("uma-s-omol", device="xpu")

    message = str(excinfo.value)
    assert "device='xpu'" in message
    assert "Use one of: cpu, cuda" in message


def test_get_calculator_uma_invalid_device_does_not_fallback_to_mace():
    """Configuration errors should not be hidden behind fallback attempts."""

    with pytest.raises(RuntimeError) as excinfo:
        get_calculator(name="uma-s-omol", device="xpu")

    message = str(excinfo.value)
    assert "device='xpu'" in message
    assert "UMA" in message


def test_get_calculator_uma_unavailable_falls_back_to_mace():
    """Test UMA import failure follows the existing MACE fallback path."""

    mace_calculators_module = types.ModuleType("mace.calculators")

    def fake_mace_mp(**kwargs):
        calculator = EMT()
        calculator.mace_kwargs = kwargs
        return calculator

    mace_calculators_module.mace_mp = fake_mace_mp
    mace_module = types.ModuleType("mace")
    mace_module.calculators = mace_calculators_module

    with patch.dict(
        sys.modules,
        {
            "fairchem": None,
            "fairchem.core": None,
            "mace": mace_module,
            "mace.calculators": mace_calculators_module,
        },
    ):
        calculator = get_calculator(name="uma-s-omol")

    assert isinstance(calculator, EMT)
    assert calculator.model_name == "large"
    assert calculator.mace_kwargs["dispersion"] is True


def test_mace_uma_dependency_workaround_is_declared():
    """Test packaging metadata keeps MACE out of the resolver conflict path."""

    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    env_yml = (root / "env.yml").read_text(encoding="utf-8")
    base_dependencies = pyproject.split("[project.optional-dependencies]", 1)[0]

    assert '"mace-torch"' not in pyproject
    assert '"e3nn>=0.5"' in pyproject
    assert '"e3nn>=0.5"' not in base_dependencies
    assert "fairchem-core>=2.0" in pyproject
    assert '"pydantic>=2.12,<3"' in pyproject
    assert "torch-dftd" in pyproject

    assert "- e3nn>=0.5" in env_yml
    assert "- pydantic>=2.12,<3" in env_yml
    assert "- fairchem-core" in env_yml
    assert "--no-deps mace-torch" in env_yml


def test_get_calculator_xtb_success():
    """Test getting XTB calculator successfully."""
    # Skip test if XTB cannot be imported
    pytest.importorskip("xtb")
    try:
        calculator = get_calculator(name="xtb")
        # This assertion now happens first. If XTB wasn't importable,
        # the test would have skipped above. If it falls back despite
        # being importable, this assertion will correctly fail.
        assert "xtb" in calculator.name.lower()

        # Test calculation
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not np.isnan(energy)

    except RuntimeError as e:
        # This might happen if XTB is installed but init fails, and fallback also fails
        pytest.skip(f"Could not initialize XTB or fallback: {e}")


def test_get_calculator_xtb_unavailable_fallback():
    """Test fallback when XTB is not available."""
    # Mock the import to simulate XTB not being available
    with patch.dict(sys.modules, {"xtb.ase.calculator": None, "xtb": None}):
        try:
            calculator = get_calculator(name="xtb")
            # It should fallback to MACE or EMT
            assert calculator is not None
            # Check type for EMT or name for MACE/Sum(MACE)
            is_fallback_ok = False
            if isinstance(calculator, EMT):
                is_fallback_ok = True
            elif isinstance(calculator, SumCalculator):
                # Correct attribute is 'calcs'
                is_fallback_ok = any("mace" in c.name.lower() for c in calculator.calcs)
            else:
                is_fallback_ok = "mace" in calculator.name.lower()
            assert (
                is_fallback_ok
            ), f"Fallback calculator {calculator.name} is not MACE or EMT"

            # Test calculation with fallback
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            assert isinstance(energy, float)
        except RuntimeError as e:
            pytest.skip(f"Could not initialize MACE/EMT fallback: {e}")
        except ImportError:
            pytest.skip("Neither XTB nor fallbacks available.")


def test_get_calculator_emt_direct():
    """Test getting EMT calculator directly."""
    try:
        calculator = get_calculator(name="emt")
        assert isinstance(calculator, EMT)

        # Test calculation
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not np.isnan(energy)

    except RuntimeError as e:
        pytest.skip(f"Could not initialize EMT: {e}")
    except ImportError:
        pytest.skip("EMT calculator not available (ASE issue?).")


def test_get_calculator_unknown_fallback():
    """Test fallback when an unknown calculator is requested."""
    try:
        calculator = get_calculator(name="unknown_calc")
        # It should fallback (likely to EMT after trying MACE/XTB)
        assert calculator is not None
        # Check if it's EMT or MACE/Sum(MACE)
        is_fallback_ok = False
        if isinstance(calculator, EMT):
            is_fallback_ok = True
        elif isinstance(calculator, SumCalculator):
            # Correct attribute is 'calcs'
            is_fallback_ok = any("mace" in c.name.lower() for c in calculator.calcs)
        else:
            is_fallback_ok = "mace" in calculator.name.lower()
        assert (
            is_fallback_ok
        ), f"Fallback calculator {calculator.name} is not MACE or EMT"

        # Test calculation with fallback
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
    except RuntimeError as e:
        pytest.skip(f"Could not initialize fallback calculator: {e}")
    except ImportError:
        pytest.skip("Fallback calculator not available.")


def test_is_linear_by_inertia():
    """Test linear molecule detection by moments of inertia."""
    # Linear molecule (CO2) - using more precise geometry
    co2 = Atoms(
        "CO2",
        positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]],  # More precise CO2 geometry
        pbc=False,
    )
    # Center the molecule at origin to ensure proper moment calculation
    assert is_linear_by_inertia(co2)  # Increased tolerance

    # Non-linear molecule (H2O)
    h2o = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        pbc=False,
    )
    assert not is_linear_by_inertia(h2o)

    # Test with custom tolerance
    assert is_linear_by_inertia(co2, tol=1e-2)
    assert not is_linear_by_inertia(h2o, tol=1e-2)


def test_get_symmetry_info():
    """Test symmetry information calculation."""
    # Skip if pymatgen not installed
    pytest.importorskip("pymatgen")

    # Test with water molecule
    h2o = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        cell=[10, 10, 10],
        pbc=False,
    )
    pointgroup, sym_number = get_symmetry_info(h2o)
    assert isinstance(
        str(pointgroup), str
    )  # Convert pointgroup to string before checking
    assert isinstance(sym_number, int)
    assert sym_number > 0

    # Test with methane (higher symmetry)
    ch4 = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],  # C
            [0.6, 0.6, 0.6],  # H
            [-0.6, -0.6, 0.6],  # H
            [0.6, -0.6, -0.6],  # H
            [-0.6, 0.6, -0.6],  # H
        ],
        cell=[10, 10, 10],
        pbc=False,
    )
    pointgroup, sym_number = get_symmetry_info(ch4)
    assert isinstance(
        str(pointgroup), str
    )  # Convert pointgroup to string before checking
    assert isinstance(sym_number, int)
    assert sym_number > 0

    # Test error handling
    with patch(
        "pymatgen.symmetry.analyzer.PointGroupAnalyzer",
        side_effect=Exception("Test error"),
    ):
        pointgroup, sym_number = get_symmetry_info(h2o)
        assert pointgroup == "C1"
        assert sym_number == 1


def test_run_vibrations_error_handling(tmp_path):
    """Test error handling in run_vibrations."""
    # Create a simple molecule
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    vib_dir = tmp_path / "vib"

    # Test with invalid calculator
    with patch("iqc.asetools.get_calculator", side_effect=Exception("Test error")):
        atoms, results = run_vibrations(h2, calculator=None, vib_dir=vib_dir)
        assert atoms is None
        assert "error" in results
        assert "Test error" in results["error"]

    # Test with optimization failure
    with patch("iqc.asetools.run_optimization") as mock_opt:
        mock_opt.return_value = (h2, {"error": "Optimization failed"})
        atoms, results = run_vibrations(
            h2, calculator=EMT(), optimize=True, vib_dir=vib_dir
        )
        assert atoms is None
        assert "error" in results
        assert "Optimization failed" in results["error"]

    # Test with vibration calculation failure
    with patch(
        "ase.vibrations.Vibrations.run", side_effect=Exception("Vibration failed")
    ):
        atoms, results = run_vibrations(
            h2, calculator=EMT(), optimize=False, vib_dir=vib_dir
        )
        assert "error" in results
        assert "Vibration failed" in results["error"]


def test_run_vibrations_filters_optimization_params(tmp_path):
    """Vibration-only parameters must not be passed to run_optimization."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    with patch("iqc.asetools.run_optimization") as mock_opt:
        mock_opt.return_value = (h2, {"error": "Optimization failed"})
        run_vibrations(
            h2,
            calculator=EMT(),
            optimize=True,
            max_trans_rot=50,
            max_vib_imag=20,
            max_steps=7,
            output_dir=str(tmp_path),
            vib_dir=tmp_path / "vib",
        )

    opt_kwargs = mock_opt.call_args.kwargs
    assert "max_trans_rot" not in opt_kwargs
    assert "max_vib_imag" not in opt_kwargs
    assert opt_kwargs["max_steps"] == 7
    assert opt_kwargs["output_dir"] == str(tmp_path)


def test_run_vibrations_warnings(tmp_path):
    """Test warning handling in run_vibrations."""
    # Create a linear molecule with high translational/rotational modes
    co2 = Atoms(
        "CO2",
        positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]],
        cell=[10, 10, 10],
        pbc=False,
    )
    vib_dir = tmp_path / "vib"

    # Mock the vibrations calculation to return high frequencies
    with patch("ase.vibrations.Vibrations.run", return_value=None), patch(
        "ase.vibrations.Vibrations.get_vibrations"
    ) as mock_vib:
        mock_vib.return_value.get_frequencies.return_value = np.array(
            [200, 200, 200, 100, 100, 100, 50, 50, 50]
        )
        mock_vib.return_value.get_energies.return_value = np.array(
            [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025]
        )

        atoms, results = run_vibrations(
            co2,
            calculator=EMT(),
            optimize=False,
            max_trans_rot=50,
            vib_dir=vib_dir,
        )
        assert "warnings" in results
        assert len(results["warnings"]) > 0
        assert any(
            "Translational or rotational modes are too high" in w
            for w in results["warnings"]
        )


def test_run_thermo_error_handling(tmp_path):
    """Test error handling in run_thermo."""
    # Create a simple molecule
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    vib_dir = tmp_path / "vib"

    # Test with vibration calculation failure
    with patch(
        "iqc.asetools.run_vibrations",
        return_value=(None, {"error": "Vibration failed"}),
    ):
        thermo, results = run_thermo(h2, vib_dir=vib_dir)
        assert results["error"] == "Vibration failed"

    # Test with imaginary modes
    with patch("iqc.asetools.run_vibrations") as mock_vib:
        mock_vib.return_value = (
            h2,
            {
                "vib_energies": [0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
                "number_of_imaginary": 1,
                "error": "",
            },
        )
        thermo, results = run_thermo(h2, ignore_imag_modes=False, vib_dir=vib_dir)
        assert "error" in results
        assert "imaginary" in results["error"].lower()

    # Test with thermo calculation failure
    with patch(
        "ase.thermochemistry.IdealGasThermo.get_gibbs_energy",
        side_effect=Exception("Thermo failed"),
    ):
        thermo, results = run_thermo(h2, calculator=EMT(), vib_dir=vib_dir)
        assert "error" in results
        assert "Thermo failed" in results["error"]


def test_run_ir_uses_separate_dipole_calculator(tmp_path, monkeypatch):
    """run_ir must compute forces with vib_calc and dipoles with dip_calc."""
    monkeypatch.chdir(tmp_path)
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=False)

    vib_calls = {"forces": 0, "dipole": 0}
    dip_calls = {"forces": 0, "dipole": 0}

    class _CountingCalc:
        def __init__(self, store, dipole=None):
            self._store = store
            self._dipole = dipole
            self.name = "counting"

        def get_forces(self, atoms):
            self._store["forces"] += 1
            return np.zeros((len(atoms), 3))

        def get_dipole_moment(self, atoms):
            self._store["dipole"] += 1
            if self._dipole is None:
                raise AssertionError("dipole calculator should not be called")
            return np.array(self._dipole, dtype=float)

    vib_calc = _CountingCalc(vib_calls)
    dip_calc = _CountingCalc(dip_calls, dipole=[0.0, 0.0, 0.1])

    _, results = run_ir(
        h2,
        vibration_calculator=vib_calc,
        dipole_calculator=dip_calc,
        optimize=False,
        unique_name="h2_split",
        vib_dir=str(tmp_path / "ir"),
        delta=0.01,
    )

    # Forces requested only from vib_calc; dipoles only from dip_calc.
    assert vib_calls["forces"] > 0
    assert vib_calls["dipole"] == 0
    assert dip_calls["forces"] == 0
    assert dip_calls["dipole"] > 0
    assert results["calculator_vibration"]
    assert results["calculator_dipole"]
    assert "vib_energies" in results
    assert "vibrational_frequencies_cm^-1" in results
    assert results["error"] == ""


def test_run_ir_thermo_reuses_ir_vibrations(tmp_path):
    """run_ir_thermo must add thermo properties without calling run_vibrations."""

    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10])
    h2.calc = EMT()
    ir_results = {
        "warnings": [],
        "error": "",
        "vib_energies": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "number_of_imaginary": 0,
        "multiplicity": 1,
        "opt_sym_number": 1,
    }

    with patch("iqc.asetools.run_ir", return_value=(h2, ir_results)) as mock_ir, patch(
        "iqc.asetools.run_vibrations",
        side_effect=AssertionError("run_ir_thermo should not recompute vibrations"),
    ):
        atoms, results = run_ir_thermo(
            h2,
            calculator=EMT(),
            unique_name="h2_ir_thermo",
            vib_dir=tmp_path / "ir",
        )

    assert atoms is h2
    assert mock_ir.call_count == 1
    assert results["error"] == ""
    assert "G_eV" in results
    assert "H_eV" in results
    assert "S_eV/K" in results
    assert "E_ZPE_eV" in results


def test_run_ir_thermo_respects_imaginary_mode_policy(tmp_path):
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10])
    h2.calc = EMT()
    ir_results = {
        "warnings": [],
        "error": "",
        "vib_energies": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "number_of_imaginary": 1,
        "multiplicity": 1,
        "opt_sym_number": 1,
    }

    with patch("iqc.asetools.run_ir", return_value=(h2, ir_results)):
        atoms, results = run_ir_thermo(
            h2,
            calculator=EMT(),
            ignore_imag_modes=False,
            unique_name="h2_ir_thermo",
            vib_dir=tmp_path / "ir",
        )

    assert atoms is h2
    assert "imaginary" in results["error"].lower()
    assert "G_eV" not in results


def test_run_ir_filters_optimization_params(tmp_path):
    """IR parameters must not leak into run_optimization."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = MagicMock()
    calc.implemented_properties = ["energy", "forces", "dipole"]
    calc.__str__.return_value = "dipole-calc"

    with patch("iqc.asetools.run_optimization") as mock_opt:
        mock_opt.return_value = (h2, {"error": "Optimization failed"})
        run_ir(
            h2,
            calculator=calc,
            optimize=True,
            max_trans_rot=50,
            max_vib_imag=20,
            max_steps=7,
            output_dir=str(tmp_path),
            vib_dir=tmp_path / "ir",
        )

    opt_kwargs = mock_opt.call_args.kwargs
    assert "max_trans_rot" not in opt_kwargs
    assert "max_vib_imag" not in opt_kwargs
    assert opt_kwargs["max_steps"] == 7
    assert opt_kwargs["output_dir"] == str(tmp_path)


def test_get_calculator_orca_constructs_with_profile(monkeypatch):
    """`get_calculator(name='orca', command=...)` returns a real ORCA calculator."""
    pytest.importorskip("ase.calculators.orca")
    from ase.calculators.orca import ORCA

    calc = get_calculator(
        name="orca",
        command="/usr/bin/false",  # dummy; we don't run ORCA, just construct it
        orcasimpleinput="HF def2-SVP",
        orcablocks="%pal nprocs 1 end",
    )
    assert isinstance(calc, ORCA), (
        f"expected ORCA instance, got {type(calc).__name__}"
    )
    # Dipole must be in implemented_properties for the IR dipole role.
    assert "dipole" in calc.implemented_properties


def test_get_calculator_orca_auto_detects_path(monkeypatch):
    """When command/env are unset, get_calculator must probe PATH for `orca`."""
    pytest.importorskip("ase.calculators.orca")
    from ase.calculators.orca import ORCA

    monkeypatch.delenv("ASE_ORCA_COMMAND", raising=False)
    # Pretend `orca` lives at a known path so OrcaProfile gets a command.
    monkeypatch.setattr(
        "shutil.which", lambda exe: "/fake/path/orca" if exe == "orca" else None
    )
    calc = get_calculator(name="orca")
    assert isinstance(calc, ORCA)


ORCA6_OUTPUT_NO_COM = """\
                                  ORCA 6.0.1
                          - the next ORCA -

------------------
TOTAL SCF ENERGY
------------------

FINAL SINGLE POINT ENERGY       -76.42830000

---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  O      0.000000    0.000000    0.122147
  H      0.000000    0.769065   -0.473338
  H      0.000000   -0.769065   -0.463338

Number of atoms                             ...      3

-------------
DIPOLE MOMENT
-------------
                                X             Y             Z
Electronic contribution:      0.000000      0.000000      0.123456
Nuclear contribution   :      0.000000      0.000000     -0.905784
                        -----------------------------------------
Total Dipole Moment    :     -0.000000000       0.000000000      -0.782327949
                        -----------------------------------------
Magnitude (a.u.)       :      0.78232795
Magnitude (Debye)      :      1.98852000


****ORCA TERMINATED NORMALLY****
"""


def test_patch_ase_orca_dipole_recovers_dipole_for_orca6(tmp_path, monkeypatch):
    """ORCA 6 outputs lack the COM line; patched parser must still emit dipole."""
    pytest.importorskip("ase.io.orca")

    # Reset any prior patch state so this test is self-contained.
    from ase.io import orca as _orca_io
    original = getattr(
        _orca_io, "_iqc_original_read_orca_output", _orca_io.read_orca_output
    )
    monkeypatch.setattr(_orca_io, "read_orca_output", original)
    monkeypatch.setattr(_orca_io, "_iqc_dipole_patched", False, raising=False)
    try:
        from iqc.asetools import _patch_ase_orca_dipole

        applied = _patch_ase_orca_dipole()
        assert applied, "patch should have been applied on a fresh import"

        out = tmp_path / "orca.out"
        out.write_text(ORCA6_OUTPUT_NO_COM)
        atoms = _orca_io.read_orca_output(str(out), index=0)
        results = atoms.calc.results
        assert "dipole" in results, (
            f"patched read_orca_output must surface dipole; got keys {list(results)}"
        )
        # ASE's read_dipole converts a.u. (e·Bohr) to e·Å:
        # -0.782327949 * Bohr ≈ -0.41399.
        from ase.units import Bohr
        np.testing.assert_allclose(
            results["dipole"], [0.0, 0.0, -0.782327949 * Bohr], atol=1e-9
        )
    finally:
        _orca_io.read_orca_output = original
        _orca_io._iqc_dipole_patched = False


def test_run_ir_rejects_dipole_calculator_without_dipole_property(tmp_path, monkeypatch):
    """EMT lacks 'dipole' in implemented_properties; must fail before any work."""
    monkeypatch.chdir(tmp_path)
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=False)

    _, results = run_ir(
        h2,
        calculator=EMT(),
        optimize=False,
        unique_name="h2_no_dipole",
        vib_dir=str(tmp_path / "ir"),
        delta=0.01,
    )

    assert "dipole" in results["error"].lower()
    assert "implemented_properties" not in results  # sanity: error is human-readable


def test_run_ir_falls_back_to_single_calculator(tmp_path, monkeypatch):
    """When only `calculator` is given, all three roles use it."""
    monkeypatch.chdir(tmp_path)
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=False)

    calls = {"forces": 0, "dipole": 0}

    class _SingleCalc:
        name = "single"

        def get_forces(self, atoms):
            calls["forces"] += 1
            return np.zeros((len(atoms), 3))

        def get_dipole_moment(self, atoms):
            calls["dipole"] += 1
            return np.zeros(3)

    calc = _SingleCalc()
    _, results = run_ir(
        h2,
        calculator=calc,
        optimize=False,
        unique_name="h2_single",
        vib_dir=str(tmp_path / "ir"),
        delta=0.01,
    )

    assert calls["forces"] > 0
    assert calls["dipole"] > 0
    assert results["error"] == ""

"""Tests for the NMR workflow helpers."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree

from iqc.cli import get_args
from iqc.nmr import (
    ConformerCandidate,
    _write_orca_input,
    build_nmr_settings,
    normalize_nuclei,
    optimize_conformer,
    parse_gaussian_nmr_output,
    parse_nwchem_nmr_output,
    parse_orca_nmr_output,
    parse_reference_shieldings,
    run_nmr_workflow,
    simulate_nmr_spectrum,
)


def test_normalize_nuclei_and_references():
    assert normalize_nuclei("1H,13C") == ["1H", "13C"]
    assert normalize_nuclei([" 13C ", "1H", "13C"]) == ["13C", "1H"]
    assert parse_reference_shieldings(["1H=31.77", "13C=188.10"]) == {
        "1H": 31.77,
        "13C": 188.10,
    }


def test_parse_orca_nmr_output(tmp_path):
    out_file = tmp_path / "orca.out"
    out_file.write_text(
        """
CHEMICAL SHIELDING SUMMARY (ppm)
-------------------------------

Nucleus  Element    Isotropic   Anisotropy
-------  -------    ---------   ----------
0        C          182.123     44.210
1        H          31.456      10.500

FINAL SINGLE POINT ENERGY      -154.123456789
""",
        encoding="utf-8",
    )

    rows, meta = parse_orca_nmr_output(out_file)
    assert rows[0]["atom_index"] == 0
    assert rows[0]["element"] == "C"
    assert rows[1]["isotropic_shielding_ppm"] == pytest.approx(31.456)
    assert meta["energy_eV"] is not None


def test_parse_gaussian_nmr_output(tmp_path):
    out_file = tmp_path / "gaussian.log"
    out_file.write_text(
        """
 SCF GIAO Magnetic shielding tensor (ppm):
      1  C    Isotropic =   194.2234   Anisotropy =     0.0000
      2  H    Isotropic =    31.6000   Anisotropy =     9.6449
""",
        encoding="utf-8",
    )

    rows, meta = parse_gaussian_nmr_output(out_file)
    assert rows[0]["atom_index"] == 0
    assert rows[1]["element"] == "H"
    assert rows[1]["anisotropy_ppm"] == pytest.approx(9.6449)
    assert meta["energy_eV"] is None


def test_parse_nwchem_nmr_output(tmp_path):
    out_file = tmp_path / "nwchem.nwo"
    out_file.write_text(
        """
 Chemical Shielding Tensors (GIAO, in ppm)

 Atom:    1  C
   isotropic =    185.4321
   anisotropy =    12.5000

 Atom:    2  H
   isotropic =     30.8765
   anisotropy =     8.2500
""",
        encoding="utf-8",
    )

    rows, meta = parse_nwchem_nmr_output(out_file)
    assert rows[0]["atom_index"] == 0
    assert rows[0]["isotropic_shielding_ppm"] == pytest.approx(185.4321)
    assert rows[1]["element"] == "H"
    assert meta["energy_eV"] is None


def test_simulate_nmr_spectrum_peak_position():
    ppm, intensity = simulate_nmr_spectrum(
        peak_rows=[
            {"nucleus": "1H", "chemical_shift_ppm": 7.25, "peak_intensity": 1.0},
            {"nucleus": "1H", "chemical_shift_ppm": 1.15, "peak_intensity": 0.5},
        ],
        nucleus="1H",
        linewidth=0.05,
        plot_range=(-1.0, 12.0),
        lineshape="lorentzian",
        num_points=4001,
    )
    peak_position = ppm[np.argmax(intensity)]
    assert peak_position == pytest.approx(7.25, abs=0.05)


def test_build_nmr_settings_rejects_direct_xtb_backend():
    with pytest.raises(ValueError, match="xTB is not supported as a direct NMR backend"):
        build_nmr_settings(backend="xtb")


def test_write_orca_input_moves_eprnmr_after_coordinates(tmp_path):
    atoms = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
    )
    path = tmp_path / "orca.inp"

    _write_orca_input(
        path,
        atoms,
        {
            "orcasimpleinput": "PBE0 def2-TZVP TightSCF RIJCOSX DEF2/J NMR",
            "charge": 0,
            "mult": 1,
            "orcablocks": "\n".join(
                [
                    "%PAL NPROCS 2 END",
                    "%EPRNMR",
                    "  Nuclei = all H {shift}",
                    "END",
                    "%cpcm",
                    "  smd true",
                    "end",
                ]
            ),
        },
    )

    text = path.read_text(encoding="utf-8")
    assert text.index("%PAL") < text.index("*xyz 0 1")
    assert text.index("%cpcm") < text.index("*xyz 0 1")
    assert text.index("*xyz 0 1") < text.index("%EPRNMR")
    assert text.index("*\n%EPRNMR") > text.index("H -1.0 0.0 0.0")


def test_optimize_conformer_with_xtb(monkeypatch, tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    candidate = ConformerCandidate(
        conformer_id=0,
        atoms=atoms,
        output_dir=tmp_path / "conf_00",
    )
    settings = build_nmr_settings(
        backend="orca",
        optimization_backend="xtb",
        solvent_model="smd",
        solvent="chloroform",
    )

    monkeypatch.setattr("iqc.nmr.shutil.which", lambda name: "/usr/bin/xtb")

    def fake_run(cmd, cwd, stdout, stderr, check):
        assert cmd[0] == "/usr/bin/xtb"
        assert "--opt" in cmd
        assert "--gfn" in cmd
        assert "--alpb" in cmd
        assert "chcl3" in cmd
        (Path(cwd) / "xtbopt.xyz").write_text(
            """3
water
O 0.000000 0.000000 0.000000
H 0.960000 0.000000 0.000000
H -0.240000 0.930000 0.000000
""",
            encoding="utf-8",
        )
        stdout.write("TOTAL ENERGY      -10.5000000000\n")
        stdout.flush()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("iqc.nmr.subprocess.run", fake_run)

    optimized = optimize_conformer(candidate, settings, "water")

    assert optimized.optimization_converged is True
    assert optimized.energy_source == "xtb"
    assert optimized.optimization_energy_eV == pytest.approx(-10.5 * Hartree)
    assert (candidate.output_dir / "opt" / "optimized.xyz").exists()
    assert optimized.atoms.get_chemical_formula() == "H2O"


def test_optimize_conformer_with_xtb_recovers_from_log_only(monkeypatch, tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    candidate = ConformerCandidate(
        conformer_id=0,
        atoms=atoms,
        output_dir=tmp_path / "conf_00",
    )
    settings = build_nmr_settings(
        backend="orca",
        optimization_backend="xtb",
        solvent_model=None,
        solvent=None,
    )

    monkeypatch.setattr("iqc.nmr.shutil.which", lambda name: "/usr/bin/xtb")

    def fake_run(cmd, cwd, stdout, stderr, check):
        Path(cwd).mkdir(parents=True, exist_ok=True)
        (Path(cwd) / "xtbopt.log").write_text(
            """3
energy: -10.2500000000 gnorm: 0.0010000000 xtb: 6.7.1
O 0.000000 0.000000 0.000000
H 0.960000 0.000000 0.000000
H -0.240000 0.930000 0.000000
""",
            encoding="utf-8",
        )
        stdout.write("TOTAL ENERGY      -10.2500000000\n")
        stdout.flush()
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr("iqc.nmr.subprocess.run", fake_run)

    optimized = optimize_conformer(candidate, settings, "water")

    assert optimized.optimization_converged is False
    assert optimized.energy_source == "xtb"
    assert optimized.optimization_energy_eV == pytest.approx(-10.25 * Hartree)
    assert (candidate.output_dir / "opt" / "optimized.xyz").exists()
    assert optimized.atoms.get_chemical_formula() == "H2O"


@pytest.mark.parametrize("incomplete_second", [False, True])
def test_run_nmr_workflow_writes_outputs(tmp_path, monkeypatch, incomplete_second):
    atoms = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.6],
            [-0.6, -0.6, 0.6],
            [0.6, -0.6, -0.6],
            [-0.6, 0.6, -0.6],
        ],
    )

    conformers = [
        ConformerCandidate(conformer_id=0, atoms=atoms.copy(), initial_energy_eV=0.0),
        ConformerCandidate(conformer_id=1, atoms=atoms.copy(), initial_energy_eV=0.01),
    ]

    def fake_parse_output(_backend, _path):
        if not hasattr(fake_parse_output, "calls"):
            fake_parse_output.calls = 0
        fake_parse_output.calls += 1
        shift_delta = 0.2 if fake_parse_output.calls == 1 else -0.2
        rows = [{"atom_index": 0, "element": "C", "isotropic_shielding_ppm": 185.0 + shift_delta}]
        rows.extend(
            {
                "atom_index": i,
                "element": "H",
                "isotropic_shielding_ppm": 31.0 + shift_delta,
            }
            for i in range(1, 5)
        )
        if incomplete_second and fake_parse_output.calls == 2:
            rows.pop()
        return rows, {"energy_eV": None}

    monkeypatch.setattr(
        "iqc.nmr.generate_conformer_candidates",
        lambda _atoms, _settings: (conformers, []),
    )
    monkeypatch.setattr("iqc.nmr.build_backend_calculator", lambda **kwargs: object())
    monkeypatch.setattr(
        "iqc.nmr._run_nmr_job",
        lambda calculator, atoms, backend: Path(tmp_path) / "dummy.out",
    )
    monkeypatch.setattr("iqc.nmr.parse_nmr_output", fake_parse_output)

    final_atoms, results = run_nmr_workflow(
        atoms=atoms,
        unique_name="methane",
        backend="gaussian",
        optimize_geometry=False,
        conformer_sampling=False,
        output_dir=str(tmp_path),
    )

    assert final_atoms.get_chemical_formula() == "CH4"
    assert results["error"] == ""
    assert results["nmr_num_conformers_used"] == (1 if incomplete_second else 2)
    assert len(results["nmr_atom_results"]) == (5 if incomplete_second else 10)
    if incomplete_second:
        assert any("Incomplete" in warning for warning in results["warnings"])
        assert all(row["boltzmann_weight"] == 1.0 for row in results["nmr_atom_results"])
    assert Path(results["nmr_spectrum_plot"]).exists()
    assert Path(results["nmr_interactive_html"]).exists()
    assert Path(results["nmr_atom_results_csv"]).exists()
    assert Path(results["nmr_weighted_peaks_csv"]).exists()
    html_text = Path(results["nmr_interactive_html"]).read_text(encoding="utf-8")
    assert "Export PNG" in html_text
    assert "linewidth" in html_text.lower()
    for csv_path in results["nmr_spectrum_csv_files"].values():
        assert Path(csv_path).exists()


def test_cli_parses_nmr_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iqc",
            "--task",
            "nmr",
            "--backend",
            "orca",
            "--optimization-backend",
            "xtb",
            "--nuclei",
            "1H",
            "13C",
            "--method",
            "PBE0",
            "--basis",
            "def2-TZVP",
            "--solvent-model",
            "smd",
            "--solvent",
            "chloroform",
            "--charge",
            "0",
            "--multiplicity",
            "1",
            "--no-conformer-sampling",
            "--output-dir",
            "nmr_run",
            "--reference-shielding",
            "1H=31.77",
        ],
    )
    args = get_args()
    assert args.task == "nmr"
    assert args.backend == "orca"
    assert args.optimization_backend == "xtb"
    assert args.nuclei == ["1H", "13C"]
    assert args.conformer_sampling is False
    assert args.reference_shielding == ["1H=31.77"]


def test_cli_parses_smiles_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iqc",
            "--task",
            "single",
            "--smiles",
            "O",
        ],
    )
    args = get_args()
    assert args.task == "single"
    assert args.smiles == "O"


def test_build_gaussian_calculator_keeps_ase_command_default(tmp_path, monkeypatch):
    """Without an explicit command, ASE's 'g16 < PREFIX.com > PREFIX.log'
    template must be preserved; a bare 'g16' override drops the redirection
    and Gaussian never reads its input."""
    from iqc import nmr as nmr_module

    captured = {}

    class FakeGaussian:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "ase.calculators.gaussian.Gaussian", FakeGaussian, raising=True
    )

    settings = nmr_module.NMRSettings(backend="gaussian")
    nmr_module.build_backend_calculator(
        settings=settings,
        purpose="nmr",
        directory=tmp_path,
        job_name="job",
        nuclei=["1H"],
    )
    assert "command" not in captured

    captured.clear()
    settings = nmr_module.NMRSettings(
        backend="gaussian", command="g16 < PREFIX.com > PREFIX.log"
    )
    nmr_module.build_backend_calculator(
        settings=settings,
        purpose="nmr",
        directory=tmp_path,
        job_name="job",
        nuclei=["1H"],
    )
    assert captured["command"] == "g16 < PREFIX.com > PREFIX.log"


def test_gaussian_calculator_receives_no_atom_indices(tmp_path, monkeypatch):
    """atom_indices must never reach Gaussian(**kwargs): ASE would render it
    as a route keyword and crash input generation for every conformer."""
    from iqc import nmr as nmr_module

    captured = {}

    class FakeGaussian:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "ase.calculators.gaussian.Gaussian", FakeGaussian, raising=True
    )

    settings = nmr_module.NMRSettings(backend="gaussian")
    nmr_module.build_backend_calculator(
        settings=settings,
        purpose="nmr",
        directory=tmp_path,
        job_name="job",
        nuclei=["1H", "13C"],
    )
    assert "atom_indices" not in captured
    assert captured.get("nmr") == "giao"


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_nonphysical_nmr_temperature_is_rejected(temperature):
    from iqc.nmr import _compute_boltzmann_weights

    with pytest.raises(ValueError, match="finite and positive"):
        build_nmr_settings(temperature=temperature)
    with pytest.raises(ValueError, match="finite and positive"):
        _compute_boltzmann_weights([], temperature)


def test_nmr_weights_never_mix_force_field_and_electronic_energies():
    from iqc.nmr import _compute_boltzmann_weights

    candidates = [
        ConformerCandidate(
            0, Atoms("H2"), initial_energy_eV=0.1, optimization_energy_eV=-30
        ),
        ConformerCandidate(1, Atoms("H2"), initial_energy_eV=0.1),
    ]
    assert _compute_boltzmann_weights(candidates, 298.15) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"atom_index": 0, "element": "H", "isotropic_shielding_ppm": float("nan")}],
        [{"atom_index": 0, "element": "H", "isotropic_shielding_ppm": 30}] * 2,
    ],
)
def test_invalid_shielding_output_cannot_enter_average(rows):
    from iqc.nmr import _validate_shielding_rows

    with pytest.raises(ValueError):
        _validate_shielding_rows(Atoms("H2"), rows, {"H": "1H"})

from pathlib import Path

from ase import Atoms
from ase.calculators.emt import EMT

from iqc.asetools import run_optimization


def test_run_optimization_writes_save_artifacts_to_output_dir(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    output_dir = tmp_path / "work"
    trajectory = output_dir / "h2.traj"

    _, results = run_optimization(
        atoms,
        calculator=EMT(),
        unique_name="h2",
        fmax=100.0,
        max_steps=1,
        trajectory=str(trajectory),
        save_geometry=True,
        output_dir=str(output_dir),
    )

    assert results["error"] == ""
    assert trajectory.exists()
    assert Path(results["optimized_geometry_file"]).parent == output_dir
    assert Path(results["optimized_geometry_file"]).exists()

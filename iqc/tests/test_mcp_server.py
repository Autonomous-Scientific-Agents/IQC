"""Exercise the public MCP protocol without optional quantum engines or models."""

import asyncio
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from iqc import mcp_server as server

ROOT = Path(__file__).resolve().parents[2]
COPPER = "1\ncopper atom\nCu 0 0 0\n"


def payload(result):
    assert not result.isError, result
    return json.loads(next(item.text for item in result.content if item.type == "text"))


def test_stdio_workflow(tmp_path):
    """Handshake, discovery, real chemistry/data, and recoverable input errors."""
    data = tmp_path / "molecules.csv"
    data.write_text("smiles,label\nO,water\nCCO,ethanol\n")
    env = dict(
        os.environ,
        IQC_DISABLE_MPI="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
    )

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "iqc.mcp_server"],
            env=env,
            cwd=str(tmp_path),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "run_single_point" in {tool.name for tool in tools.tools}
                catalog = payload(await session.call_tool("list_calculators", {}))
                assert catalog["default_for_mcp"] == "xtb"
                mol = payload(await session.call_tool("smiles_to_xyz", {"smiles": "O"}))
                assert mol["n_atoms"] == 3
                restored = payload(
                    await session.call_tool("xyz_to_smiles", {"xyz": mol["xyz"]})
                )
                from rdkit import Chem

                assert (
                    Chem.MolToSmiles(
                        Chem.RemoveHs(Chem.MolFromSmiles(restored["smiles"]))
                    )
                    == "O"
                )
                energy = payload(
                    await session.call_tool(
                        "run_single_point",
                        {
                            "xyz": COPPER,
                            "calculator": "emt",
                            "keep_forces": True,
                        },
                    )
                )
                assert not energy.get("error")
                assert math.isfinite(energy["energy_eV"])
                assert len(energy["forces"]) == 1
                summary = payload(
                    await session.call_tool("inspect_data_file", {"path": str(data)})
                )
                assert summary["rows"] == 2
                bad = await session.call_tool(
                    "run_single_point",
                    {
                        "xyz": COPPER,
                        "smiles": "O",
                        "calculator": "emt",
                    },
                )
                assert bad.isError
                # An input error must not kill the long-lived connection.
                assert not (await session.call_tool("list_calculators", {})).isError

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


def test_backend_output_cannot_corrupt_stdio(tmp_path):
    """A noisy Python/native backend must leave JSON-RPC parseable."""
    script = """
import os
from iqc import mcp_server as server
from iqc import asetools

def noisy(*args, **kwargs):
    print("python backend diagnostic", flush=True)
    os.write(1, b"native backend diagnostic\\n")
    # Child processes inherit fd 0; it must not be the JSON-RPC stream.
    assert os.read(0, 1) == b""
    return args[0], {"energy_eV": 1.0, "error": "", "warnings": []}

asetools.run_single_point = noisy
server.main()
"""

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", script],
            cwd=str(tmp_path),
            env=dict(
                os.environ,
                PYTHONPATH=str(ROOT),
                IQC_DISABLE_MPI="1",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            ),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = payload(
                    await session.call_tool(
                        "run_single_point",
                        {
                            "xyz": COPPER,
                            "calculator": "emt",
                        },
                    )
                )
                assert result["energy_eV"] == 1.0

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


@pytest.mark.parametrize(
    "tool",
    [
        "run_single_point",
        "run_optimization",
        "run_vibrations",
        "run_thermo",
        "run_ir",
    ],
)
def test_omitted_charge_preserves_structure_metadata(monkeypatch, tool):
    from iqc import asetools

    seen = []

    def calculate(atoms, **kwargs):
        seen.append((atoms.info["charge"], kwargs["charge"]))
        return atoms, {"error": "", "warnings": []}

    monkeypatch.setattr(asetools, tool, calculate)
    getattr(server, tool)(smiles="[NH4+]", calculator="emt")
    getattr(server, tool)(smiles="[NH4+]", calculator="emt", charge=0)
    assert seen == [(1, None), (1, 0)]


def test_file_calculators_are_not_shared(monkeypatch):
    from iqc import asetools

    monkeypatch.setattr(asetools, "get_calculator", lambda **kw: object())
    assert server._get_calculator("orca") is not server._get_calculator("orca")


def test_model_calculator_reused_for_identical_parameters(monkeypatch):
    from iqc import asetools

    monkeypatch.setattr(asetools, "get_calculator", lambda **kw: object())
    monkeypatch.setattr(server, "_CALC_CACHE", {})
    assert server._get_calculator("mace", {"model": "small"}) is server._get_calculator(
        "mace", {"model": "small"}
    )
    assert server._get_calculator(
        "mace", {"model": "large"}
    ) is not server._get_calculator("mace", {"model": "small"})
    assert server._get_calculator("MACE", {"model": "small"}) is server._get_calculator(
        "mace", {"model": "small"}
    )
    assert server._get_calculator("uma") is server._get_calculator("UMA")


def test_result_arrays_and_scalars_remain_numeric():
    import numpy as np

    result = server._trim_results(
        {
            "number_of_atoms": np.int64(3),
            "frequencies": np.array([100.0, 200.0]),
            "forces": np.ones((3, 3)),
        }
    )
    assert result == {"number_of_atoms": 3, "frequencies": [100.0, 200.0]}
    json.dumps(result, allow_nan=False)


def test_complex_and_non_finite_values_stay_typed():
    import numpy as np

    result = server._trim_results(
        {
            "frequencies_cm^-1": np.array([0.0 + 12.5j, 1595.2 + 0j]).tolist(),
            "vib_energies": np.array([0.0 + 0.002j, 0.2 + 0j]),
            "spectrum_intensities": [1.0, float("nan"), 2.0],
            "energy_eV": np.float64("inf"),
            "error": "",
        }
    )
    assert result["frequencies_cm^-1"] == [-12.5, 1595.2]
    assert result["vib_energies"] == [-0.002, 0.2]
    assert result["spectrum_intensities"] == [1.0, None, 2.0]
    assert result["energy_eV"] is None
    assert sorted(result["non_finite_fields"]) == ["energy_eV", "spectrum_intensities"]
    json.dumps(result, allow_nan=False)


@pytest.fixture
def ir_calls(monkeypatch):
    """Record get_calculator calls made by run_ir without running chemistry."""
    from iqc import asetools

    made = []

    def fake_get_calculator(name, **params):
        made.append((name, params))
        return object()

    monkeypatch.setattr(asetools, "get_calculator", fake_get_calculator)
    monkeypatch.setattr(server, "_CALC_CACHE", {})
    monkeypatch.setattr(
        asetools, "run_ir", lambda atoms, **kw: (atoms, {"error": "", "warnings": []})
    )
    return made


def test_run_ir_per_role_calculator_params(ir_calls):
    server.run_ir(
        xyz=COPPER,
        calculator="emt",
        vibration_calculator="emt",
        dipole_calculator="orca",
        dipole_calculator_params={"orcasimpleinput": "HF def2-SVP"},
    )
    assert sorted(ir_calls) == [
        ("emt", {}),
        ("emt", {}),
        ("orca", {"orcasimpleinput": "HF def2-SVP"}),
    ]


def test_run_ir_params_only_inherited_by_same_calculator(ir_calls):
    orca = {"orcasimpleinput": "B3LYP def2-SVP"}
    server.run_ir(
        xyz=COPPER,
        calculator="orca",
        calculator_params=orca,
        vibration_calculator="xtb",
        dipole_calculator="ORCA",
    )
    assert sorted(ir_calls) == [("orca", orca), ("orca", orca), ("xtb", {})]


def test_run_ir_skips_unused_fallback(ir_calls):
    server.run_ir(
        xyz=COPPER,
        optimization_calculator="mace",
        vibration_calculator="mace",
        dipole_calculator="orca",
    )
    assert "xtb" not in [name for name, _ in ir_calls]


def test_orca_scratch_is_temporary(monkeypatch, tmp_path):
    from iqc import asetools

    class ORCA:  # matched by class name, like ase.calculators.orca.ORCA
        directory = "."

    seen = []

    def calculate(atoms, calculator, **kwargs):
        seen.append(Path(calculator.directory))
        assert seen[-1].is_dir()
        return atoms, {"error": "", "warnings": []}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asetools, "get_calculator", lambda **kw: ORCA())
    for tool in ("run_single_point", "run_optimization"):
        monkeypatch.setattr(asetools, tool, calculate)
        getattr(server, tool)(xyz=COPPER, calculator="orca")
    assert len(seen) == 2 and all(not d.exists() for d in seen)
    assert list(tmp_path.iterdir()) == []

    explicit = ORCA()
    explicit.directory = str(tmp_path / "keep")
    with server._orca_scratch(explicit):
        assert explicit.directory == str(tmp_path / "keep")


def test_launcher_skips_incompatible_mcp_environment(tmp_path):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("Launcher requires a POSIX shell")

    # Isolate the checkout's .venv candidate, and provide a working PATH
    # fallback after an interpreter with a discoverable but incompatible MCP.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "run_mcp_server.sh"
    shutil.copyfile(ROOT / "scripts/run_mcp_server.sh", launcher)
    incompatible = tmp_path / "incompatible"
    (incompatible / "mcp").mkdir(parents=True)
    (incompatible / "mcp" / "__init__.py").write_text("")
    candidate = tmp_path / "incompatible-python"
    candidate.write_text(
        f"#!{shell}\n"
        f"export PYTHONPATH={shlex.quote(str(incompatible) + os.pathsep + str(ROOT))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    candidate.chmod(0o755)
    executables = tmp_path / "bin"
    executables.mkdir()
    fallback = executables / "python3"
    fallback.write_text(f'#!{shell}\nexec {shlex.quote(sys.executable)} "$@"\n')
    fallback.chmod(0o755)
    result = subprocess.run(
        [shell, str(launcher), "--help"],
        cwd=tmp_path,
        env=dict(
            os.environ,
            IQC_PYTHON=str(candidate),
            PYTHONPATH=str(ROOT),
            PATH=str(executables) + os.pathsep + os.environ.get("PATH", ""),
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Run the IQC MCP server" in result.stdout


def test_mixed_complex_values_keep_both_parts():
    result = server._trim_results({"x": [3 + 12j, 0 + 2j, 5 + 0j]})
    assert result["x"] == [{"real": 3.0, "imag": 12.0}, -2.0, 5.0]


def test_temporary_paths_and_large_lists_are_omitted():
    results = {
        "ir_plot": "/tmp/iqc_mcp_ir_x/spectrum.png",
        "normal_mode_data": "Normal modes saved as .traj files with prefix /tmp/x",
        "spectrum_frequencies": list(range(3700)),
        "vibrational_frequencies_cm^-1": [100.0, 200.0],
    }
    trimmed = server._trim_results(results)
    assert trimmed == {
        "vibrational_frequencies_cm^-1": [100.0, 200.0],
        "omitted_fields": {"spectrum_frequencies": 3700},
    }
    kept = server._trim_results(results, keep_arrays=True)
    assert len(kept["spectrum_frequencies"]) == 3700
    assert "ir_plot" not in kept and "normal_mode_data" not in kept


def test_launcher_rejects_interpreter_with_only_mcp(tmp_path):
    """A Python that imports mcp but not the scientific stack is skipped,
    and a skipped IQC_PYTHON is reported rather than silently replaced."""
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("Launcher requires a POSIX shell")

    scripts = tmp_path / "scripts"  # isolate the checkout's .venv candidate
    scripts.mkdir()
    launcher = scripts / "run_mcp_server.sh"
    shutil.copyfile(ROOT / "scripts/run_mcp_server.sh", launcher)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python3", "python"):
        fake = bin_dir / name
        fake.write_text(
            f"#!{shell}\n"
            'case "$2" in *rdkit*) echo "No module named rdkit" >&2; exit 1;; esac\n'
            "echo selected\n"
        )
        fake.chmod(0o755)
    result = subprocess.run(
        [shell, str(launcher)],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(tmp_path),
            "IQC_PYTHON": str(bin_dir / "python3"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "selected" not in result.stdout
    assert "skipping IQC_PYTHON" in result.stderr and "rdkit" in result.stderr

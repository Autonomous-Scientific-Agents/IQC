"""Exercise the public MCP protocol without optional quantum engines or models."""

import asyncio
import json
import math
import os
from pathlib import Path
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

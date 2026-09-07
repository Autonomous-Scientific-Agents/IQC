"""Driver failure tests without starting cluster processes."""

import json
import logging
import sys
import types
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from iqc import ensemble_launcher_dispatch as dispatch
from iqc.cli import get_args


@pytest.mark.parametrize("failure", ["launcher_start", "client_start", "submit", "stop", "result"])
def test_cluster_failures_persist_rows_and_exit_nonzero(tmp_path, monkeypatch, caplog, failure):
    monkeypatch.chdir(tmp_path)
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0\n")
    args = get_args(["--xyz", str(xyz), "-t", "single", "--calculator", "emt"])
    for key, value in dict(el_local=True, el_nodes_per_mol=1, el_ppn=1,
                           el_ranks_per_mol=None, el_nlevels=None,
                           el_cpus_per_node=1, el_report_interval=1).items():
        setattr(args, key, value)
    monkeypatch.setattr(dispatch, "_parse_args", lambda: args)
    monkeypatch.setattr(dispatch, "_build_xyz_input_set", lambda *a: ([str(xyz)] * 3, "xyz", 3, 3))
    monkeypatch.setattr(dispatch, "convert_jsonl_results_to_parquet", lambda *a: None)
    launcher = Mock()
    client = Mock()
    if failure == "launcher_start":
        launcher.start.side_effect = RuntimeError("launcher start failed")
    elif failure == "client_start":
        client.start.side_effect = RuntimeError("client start failed")
    elif failure == "stop":
        launcher.stop.side_effect = RuntimeError("stop failed")

    def submit(task):
        if failure == "submit" and task.task_id == "row-0000001":
            raise RuntimeError("submit failed")
        future = Future()
        if failure == "result":
            future.set_exception(RuntimeError("result failed"))
        else:
            future.set_result({"task": "single", "energy_eV": -1.0})
        return future

    client.submit.side_effect = submit
    modules = {
        "ensemble_launcher": dict(EnsembleLauncher=lambda **kw: launcher),
        "ensemble_launcher.config": {name: types.SimpleNamespace for name in
                                     ("LauncherConfig", "MPIConfig", "PolicyConfig", "SystemConfig")},
        "ensemble_launcher.ensemble": dict(Task=types.SimpleNamespace),
        "ensemble_launcher.orchestrator": dict(ClusterClient=lambda **kw: client),
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    with caplog.at_level(logging.INFO):
        assert dispatch.main() == 2
    results = list(tmp_path.glob("iqc_*_results_*.jsonl"))
    assert len(results) == 1
    rows = [json.loads(line) for line in results[0].read_text().splitlines()]
    assert len(rows) == 3
    failures = [row for row in rows if row.get("single_error")]
    expected = 0 if failure == "stop" else 2 if failure == "submit" else 3
    assert len(failures) == expected
    assert f"{expected} failed." in caplog.text
    launcher.stop.assert_called_once()
    if failure != "launcher_start":
        client.teardown.assert_called_once()

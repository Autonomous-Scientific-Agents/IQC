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


def _install_el_mocks(monkeypatch, client):
    """Register stub ensemble_launcher modules pointing at a mock client."""
    launcher = Mock()
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
    return launcher


def test_incremental_append_writes_each_outcome(tmp_path, monkeypatch, caplog):
    """Happy-path collection: each future's outcome is classified and the
    per-job JSONL is written incrementally (one file, correct rows/counts)."""
    monkeypatch.chdir(tmp_path)
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0\n")
    args = get_args(["--xyz", str(xyz), "-t", "single", "--calculator", "emt"])
    for key, value in dict(el_local=True, el_nodes_per_mol=1, el_ppn=1,
                           el_ranks_per_mol=None, el_nlevels=None,
                           el_cpus_per_node=1, el_report_interval=1,
                           el_keep_partials=False).items():
        setattr(args, key, value)
    monkeypatch.setattr(dispatch, "_parse_args", lambda: args)
    # Four rows, four distinct outcomes.
    monkeypatch.setattr(dispatch, "_build_xyz_input_set", lambda *a: ([str(xyz)] * 4, "xyz", 4, 4))
    monkeypatch.setattr(dispatch, "convert_jsonl_results_to_parquet", lambda *a: None)

    outcomes = {
        "row-0000000": {  # success dict: private keys must be stripped, base kept
            "task": "single", "energy_eV": -1.0,
            "unique_name": "water_0_0_20260101_000000",
            "unique_name_base": "water",
            "_unique_name": "water_0_0_20260101_000000",
            "_record_stamp": "20260101_000000",
            "_work_dir_used": "/tmp/x",
        },
        "row-0000001": dispatch.SKIPPED_EXISTING,  # skipped: no row written
        "row-0000002": None,                        # bad input: no row written
        "row-0000003": RuntimeError("boom"),        # failure: failure row written
    }

    def submit(task):
        future = Future()
        result = outcomes[task.task_id]
        if isinstance(result, BaseException):
            future.set_exception(result)
        else:
            future.set_result(result)
        return future

    client = Mock()
    client.submit.side_effect = submit
    _install_el_mocks(monkeypatch, client)

    with caplog.at_level(logging.INFO):
        rc = dispatch.main()

    # failed>0 => exit code 2 even though the cluster itself was healthy.
    assert rc == 2
    results = list(tmp_path.glob("iqc_*_results_*.jsonl"))
    assert len(results) == 1
    rows = [json.loads(line) for line in results[0].read_text().splitlines()]
    # Only the success row and the failure row are persisted; skip/bad-input are not.
    assert len(rows) == 2
    success = [r for r in rows if r.get("energy_eV") == -1.0]
    failures = [r for r in rows if r.get("single_error")]
    assert len(success) == 1 and len(failures) == 1
    # Private helper keys stripped; the stable identity column survives.
    assert "_unique_name" not in success[0]
    assert "_record_stamp" not in success[0]
    assert "_work_dir_used" not in success[0]
    assert success[0]["unique_name_base"] == "water"
    assert "1 completed" in caplog.text
    assert "1 skipped-existing" in caplog.text
    assert "1 bad-input" in caplog.text
    assert "1 failed" in caplog.text
    # No per-mol partials when the flag is off.
    assert not (tmp_path / "results_partials").exists()

"""Artifact CLI flags must reach ExaChem calc_params in every dispatcher."""

from types import SimpleNamespace

from iqc.main import apply_exachem_artifact_args


def _args(**kw):
    base = dict(keep_artifacts=False, artifact_retention="off", artifact_root=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_exachem_retention_on_populates_dict():
    params = {}
    apply_exachem_artifact_args(
        params, "exachem", _args(artifact_retention="on", artifact_root="/a")
    )
    assert params["artifact_retention"] == {
        "enabled": True,
        "compress": True,
        "destination_root": "/a",
    }


def test_yaml_values_win_over_cli():
    params = {"artifact_retention": {"compress": False, "destination_root": "/y"}}
    apply_exachem_artifact_args(
        params, "ExaChem", _args(artifact_retention="on", artifact_root="/a")
    )
    assert params["artifact_retention"]["compress"] is False
    assert params["artifact_retention"]["destination_root"] == "/y"
    assert params["artifact_retention"]["enabled"] is True


def test_keep_artifacts_and_off_by_default():
    params = {}
    apply_exachem_artifact_args(params, "exachem", _args(keep_artifacts=True))
    assert params == {"keep_artifacts": True}


def test_non_exachem_untouched():
    params = {}
    apply_exachem_artifact_args(
        params, "mace", _args(keep_artifacts=True, artifact_retention="on")
    )
    assert params == {}


def test_dispatchers_use_shared_helper():
    import inspect

    import iqc.ensemble_launcher_dispatch as el

    assert "apply_exachem_artifact_args" in inspect.getsource(el.main)

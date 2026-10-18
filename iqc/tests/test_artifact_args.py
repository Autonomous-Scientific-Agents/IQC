"""Artifact CLI flags must reach ExaChem calc_params in every dispatcher."""

from types import SimpleNamespace

import pytest

from iqc.main import (
    PRIVATE_RESULT_KEYS,
    apply_exachem_artifact_args,
    strip_private_keys,
)


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


def test_non_mapping_artifact_retention_raises_clear_error():
    with pytest.raises(ValueError, match="must be a mapping"):
        apply_exachem_artifact_args(
            {"artifact_retention": True}, "exachem", _args(artifact_retention="on")
        )


def test_explicit_enabled_false_in_yaml_wins():
    params = {"artifact_retention": {"enabled": False}}
    apply_exachem_artifact_args(params, "exachem", _args(artifact_retention="on"))
    assert params["artifact_retention"]["enabled"] is False


def test_strip_private_keys_returns_copy_without_helpers():
    result = {"energy": 1.0, "unique_name": "u", **{k: 1 for k in PRIVATE_RESULT_KEYS}}
    stripped = strip_private_keys(result)
    assert stripped == {"energy": 1.0, "unique_name": "u"}
    assert all(k in result for k in PRIVATE_RESULT_KEYS)  # input untouched


def test_el_dispatcher_uses_shared_helpers():
    import inspect

    import iqc.ensemble_launcher_dispatch as el

    src = inspect.getsource(el.main)
    assert "apply_exachem_artifact_args" in src
    assert "strip_private_keys" in inspect.getsource(el)


def test_parsl_dispatcher_uses_shared_helpers():
    pytest.importorskip("parsl")
    import inspect

    import iqc.parsl_dispatch as pd

    assert "apply_exachem_artifact_args" in inspect.getsource(pd.main)
    assert "strip_private_keys" in inspect.getsource(pd.main)

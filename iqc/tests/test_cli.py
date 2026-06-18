from iqc.cli import get_args


def test_argcomplete_hook_is_registered(monkeypatch):
    calls = []

    monkeypatch.setattr("iqc.cli.argcomplete.autocomplete", calls.append)

    get_args([])

    assert calls


def test_calculator_defaults_to_none_so_params_file_can_select_calculator():
    args = get_args([])

    assert args.calculator is None


def test_calculator_cli_option_overrides_params_file():
    args = get_args(["--calculator", "uma-s-omol"])

    assert args.calculator == "uma-s-omol"


def test_mace_polar_calculator_option_is_supported():
    args = get_args(["--calculator", "mace-polar"])

    assert args.calculator == "mace-polar"


def test_skip_existing_options_are_parsed():
    args = get_args(["--skip-existing", "--skip-existing-from", "old.jsonl"])

    assert args.skip_existing is True
    assert args.skip_existing_from == ["old.jsonl"]


def test_ir_thermo_task_is_supported():
    args = get_args(["--task", "ir-thermo"])

    assert args.task == "ir-thermo"


def test_artifact_retention_defaults_on():
    args = get_args([])

    assert args.artifact_retention == "on"


def test_artifact_retention_can_be_disabled():
    args = get_args(["--artifact-retention", "off"])

    assert args.artifact_retention == "off"


def test_artifact_root_honors_env_var(monkeypatch):
    monkeypatch.setenv("IQC_ARTIFACT_ROOT", "/scratch/custom")
    # Reload the module so argparse picks up the patched env default.
    import importlib

    import iqc.cli as cli_module

    importlib.reload(cli_module)
    try:
        args = cli_module.get_args([])
        assert args.artifact_root == "/scratch/custom"
    finally:
        monkeypatch.delenv("IQC_ARTIFACT_ROOT", raising=False)
        importlib.reload(cli_module)

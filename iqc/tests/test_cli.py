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


def test_skip_existing_options_are_parsed():
    args = get_args(["--skip-existing", "--skip-existing-from", "old.jsonl"])

    assert args.skip_existing is True
    assert args.skip_existing_from == ["old.jsonl"]


def test_ir_thermo_task_is_supported():
    args = get_args(["--task", "ir-thermo"])

    assert args.task == "ir-thermo"

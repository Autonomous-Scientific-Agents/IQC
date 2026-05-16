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

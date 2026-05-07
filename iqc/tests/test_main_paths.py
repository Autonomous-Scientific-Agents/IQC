from iqc.main import _make_run_id, _unique_child_path


def test_unique_child_path_returns_absolute_unused_path(tmp_path):
    path = _unique_child_path(tmp_path, "tmp_single_0_run")

    assert path == str(tmp_path / "tmp_single_0_run")


def test_unique_child_path_adds_suffix_for_existing_path(tmp_path):
    existing = tmp_path / "tmp_single_0_run"
    existing.mkdir()

    path = _unique_child_path(tmp_path, "tmp_single_0_run")

    assert path == str(tmp_path / "tmp_single_0_run_1")


def test_make_run_id_has_timestamp_and_random_suffix():
    run_id = _make_run_id()
    stamp, token = run_id.rsplit("_", 1)

    assert len(stamp) == len("YYYYMMDD_HHMMSS")
    assert stamp.replace("_", "").isdigit()
    assert len(token) == 8
    assert all(char in "0123456789abcdef" for char in token)

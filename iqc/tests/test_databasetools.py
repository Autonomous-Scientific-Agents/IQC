import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from iqc.databasetools import (
    calculation_exists,
    calculation_key,
    calculation_key_from_record,
    database_to_data,
    database_to_dataframe,
    get_number_of_molecules,
    hash_string,
    insert_entries,
    insert_entry,
    merge_databases,
)


def calculation_record(index, **overrides):
    record = {
        "initial_xyz": f"1\nmolecule {index}\nH {float(index):.8f} 0.0 0.0\n",
        "calculator": "mace",
        "model": "large",
        "task": "single",
        "params": {"cutoff": index % 3},
        "energy_eV": -float(index),
    }
    record.update(overrides)
    return record


def test_insert_entry_ignores_duplicate_calculation(tmp_path):
    db_path = tmp_path / "calculations.db"
    line = json.dumps(calculation_record(1))

    assert insert_entry(line, db_path) is True
    assert insert_entry(line, db_path) is False

    df = database_to_dataframe(db_path)
    data = database_to_data(db_path)

    assert len(df) == 1
    assert get_number_of_molecules(db_path) == 1
    assert data.loc[0, "calculator"] == "mace"


def test_calculation_exists_uses_database_identity(tmp_path):
    db_path = tmp_path / "calculations.db"
    record = calculation_record(1)

    assert not calculation_exists(
        db_path,
        record["initial_xyz"],
        record["params"],
        record["calculator"],
        record["model"],
        record["task"],
    )

    insert_entry(json.dumps(record), db_path)

    assert calculation_exists(
        db_path,
        record["initial_xyz"],
        record["params"],
        record["calculator"],
        record["model"],
        record["task"],
    )
    assert not calculation_exists(
        db_path,
        record["initial_xyz"],
        {"cutoff": "different"},
        record["calculator"],
        record["model"],
        record["task"],
    )


def test_calculation_exists_include_errors_false_skips_failed_rows(tmp_path):
    db_path = tmp_path / "calculations.db"
    failed_record = calculation_record(7, single_error="SCF diverged")
    insert_entry(json.dumps(failed_record), db_path)

    # Default include_errors=True: row is present.
    assert calculation_exists(
        db_path,
        failed_record["initial_xyz"],
        failed_record["params"],
        failed_record["calculator"],
        failed_record["model"],
        failed_record["task"],
    )
    # F5 contract: include_errors=False ignores failed rows so they re-run.
    assert not calculation_exists(
        db_path,
        failed_record["initial_xyz"],
        failed_record["params"],
        failed_record["calculator"],
        failed_record["model"],
        failed_record["task"],
        include_errors=False,
    )

    # A successful row with a distinct key still counts under include_errors=False.
    ok_record = calculation_record(8)
    insert_entry(json.dumps(ok_record), db_path)
    assert calculation_exists(
        db_path,
        ok_record["initial_xyz"],
        ok_record["params"],
        ok_record["calculator"],
        ok_record["model"],
        ok_record["task"],
        include_errors=False,
    )


def test_calculation_key_from_record_matches_explicit_key():
    record = calculation_record(1)

    assert calculation_key_from_record(record) == calculation_key(
        record["initial_xyz"],
        record["params"],
        record["calculator"],
        record["model"],
        record["task"],
    )


def test_insert_entries_batches_and_reports_duplicates(tmp_path):
    db_path = tmp_path / "calculations.db"
    lines = [json.dumps(calculation_record(index)) for index in range(5)]
    lines.extend(["\n", json.dumps(calculation_record(2))])

    summary = insert_entries(lines, db_path, batch_size=2)

    assert summary == {"processed": 6, "inserted": 5, "duplicates": 1}
    assert len(database_to_dataframe(db_path)) == 5


def test_insert_entry_raises_for_invalid_records(tmp_path):
    db_path = tmp_path / "calculations.db"

    with pytest.raises(ValueError, match="Missing required keys"):
        insert_entry(json.dumps({"initial_xyz": "1\n\nH 0 0 0\n"}), db_path)


def test_insert_entry_treats_missing_model_as_empty_string(tmp_path):
    db_path = tmp_path / "calculations.db"
    record = calculation_record(1)
    record.pop("model")
    line = json.dumps(record)

    assert insert_entry(line, db_path) is True
    assert insert_entry(line, db_path) is False

    df = database_to_dataframe(db_path)

    assert len(df) == 1
    assert df.loc[0, "model"] == ""


def test_concurrent_insert_entry_calls_are_not_lost(tmp_path):
    db_path = tmp_path / "calculations.db"

    def insert(index):
        return insert_entry(json.dumps(calculation_record(index)), db_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        inserted = list(executor.map(insert, range(50)))

    assert all(inserted)
    assert len(database_to_dataframe(db_path)) == 50
    assert get_number_of_molecules(db_path) == 50


def test_merge_databases_handles_source_paths_with_quotes(tmp_path):
    source_db = tmp_path / "source's calculations.db"
    target_db = tmp_path / "target.db"

    insert_entry(json.dumps(calculation_record(1)), source_db)
    merge_databases(target_db, source_db)

    assert len(database_to_dataframe(target_db)) == 1


def test_hash_string_accepts_structured_params():
    assert hash_string({"b": 2, "a": 1}) == hash_string({"a": 1, "b": 2})

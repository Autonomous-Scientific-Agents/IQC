"""Tests for iqc.status_query (F6: iqc-status sweep monitoring)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from iqc import status_query


def _write_jsonl(path, records):
    """Write a list of dict records to ``path`` as JSONL."""
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_db(path, records):
    """Create an iqc-style SQLite db with one row per record (blob_data only)."""
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE calculations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            geometry_hash TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            calculator TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL,
            blob_data TEXT NOT NULL
        )
        """
    )
    for i, r in enumerate(records):
        con.execute(
            "INSERT INTO calculations(geometry_hash, params_hash, calculator,"
            " model, task, blob_data) VALUES(?,?,?,?,?,?)",
            (
                f"g{i}",
                f"p{i}",
                r.get("calculator", ""),
                r.get("model", ""),
                r.get("task", ""),
                json.dumps(r),
            ),
        )
    con.commit()
    con.close()


def test_summarize_counts_ok_and_error_across_multiple_jsonl(tmp_path):
    """summarize aggregates counts across several JSONL files."""
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    _write_jsonl(
        f1,
        [
            {"task": "energy", "calculator": "exachem", "energy_eV": -1.0},
            {
                "task": "energy",
                "calculator": "exachem",
                "energy_error": "OOM at SCF",
            },
        ],
    )
    _write_jsonl(
        f2,
        [
            {"task": "opt", "calculator": "ase", "energy_eV": -2.0},
            {"task": "opt", "calculator": "ase", "opt_error": "convergence failed"},
            {"task": "opt", "calculator": "ase", "opt_error": "OOM at SCF"},
        ],
    )

    summary = status_query.summarize([f1, f2])

    assert summary["counts"] == {"ok": 2, "error": 3, "total": 5}
    assert summary["by_task"]["energy"] == {"ok": 1, "error": 1}
    assert summary["by_task"]["opt"] == {"ok": 1, "error": 2}
    assert summary["by_calculator"]["exachem"] == {"ok": 1, "error": 1}
    assert summary["by_calculator"]["ase"] == {"ok": 1, "error": 2}
    # OOM appeared twice -> top error kind
    assert "OOM at SCF" in summary["by_error_kind"]
    assert summary["by_error_kind"]["OOM at SCF"] == 2


def test_summarize_reads_sqlite_db(tmp_path):
    """SQLite ``calculations`` rows are decoded via blob_data."""
    db = tmp_path / "results.db"
    _make_db(
        db,
        [
            {"task": "energy", "calculator": "exachem", "energy_eV": -1.0},
            {
                "task": "energy",
                "calculator": "exachem",
                "energy_error": "segfault",
            },
        ],
    )
    summary = status_query.summarize([db])
    assert summary["counts"] == {"ok": 1, "error": 1, "total": 2}


def test_status_ok_filter_excludes_errors(tmp_path):
    """``--status=ok`` filter drops error rows from all counters."""
    f = tmp_path / "x.jsonl"
    _write_jsonl(
        f,
        [
            {"task": "energy", "calculator": "exachem"},
            {"task": "energy", "calculator": "exachem", "energy_error": "boom"},
        ],
    )
    summary = status_query.summarize([f], filters={"status": "ok"})
    assert summary["counts"] == {"ok": 1, "error": 0, "total": 1}
    assert summary["by_error_kind"] == {}


def test_status_error_filter_keeps_only_errors(tmp_path):
    """``--status=error`` keeps only error rows."""
    f = tmp_path / "x.jsonl"
    _write_jsonl(
        f,
        [
            {"task": "energy", "calculator": "exachem"},
            {"task": "energy", "calculator": "exachem", "energy_error": "boom"},
        ],
    )
    summary = status_query.summarize([f], filters={"status": "error"})
    assert summary["counts"] == {"ok": 0, "error": 1, "total": 1}


def test_task_and_calculator_filters(tmp_path):
    """task/calculator filters narrow the row set before counting."""
    f = tmp_path / "x.jsonl"
    _write_jsonl(
        f,
        [
            {"task": "energy", "calculator": "exachem"},
            {"task": "opt", "calculator": "ase"},
            {"task": "energy", "calculator": "ase"},
        ],
    )
    summary = status_query.summarize([f], filters={"task": "energy"})
    assert summary["counts"]["total"] == 2
    summary = status_query.summarize([f], filters={"calculator": "ase"})
    assert summary["counts"]["total"] == 2


def test_cli_table_format_produces_output(tmp_path, capsys):
    """The CLI in --table mode prints a Total: line and a by-task block."""
    f = tmp_path / "x.jsonl"
    _write_jsonl(
        f,
        [
            {"task": "energy", "calculator": "exachem"},
            {"task": "energy", "calculator": "exachem", "energy_error": "boom"},
        ],
    )
    rc = status_query.main(["--jsonl", str(f), "--table"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Total: 2" in out
    assert "OK: 1" in out
    assert "ERROR: 1" in out
    assert "By task:" in out
    assert "energy" in out
    assert "Top error kinds:" in out
    assert "boom" in out


def test_cli_json_format(tmp_path, capsys):
    """The CLI in --json mode emits a parseable JSON document."""
    f = tmp_path / "x.jsonl"
    _write_jsonl(f, [{"task": "energy", "calculator": "exachem"}])
    rc = status_query.main(["--jsonl", str(f), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["counts"]["ok"] == 1
    assert payload["counts"]["total"] == 1


def test_cli_requires_at_least_one_source(capsys):
    """argparse should error out when no --db or --jsonl is given."""
    with pytest.raises(SystemExit):
        status_query.main([])


def test_malformed_jsonl_lines_are_skipped(tmp_path):
    """A bad JSON line shouldn't poison the whole summary."""
    f = tmp_path / "x.jsonl"
    with open(f, "w") as fh:
        fh.write(json.dumps({"task": "energy", "calculator": "exachem"}) + "\n")
        fh.write("not-json\n")
        fh.write(
            json.dumps(
                {"task": "energy", "calculator": "exachem", "energy_error": "x"}
            )
            + "\n"
        )
    summary = status_query.summarize([f])
    assert summary["counts"] == {"ok": 1, "error": 1, "total": 2}


def test_plain_error_field_counts_as_error(tmp_path):
    """asetools soft failures live in 'error' with no {task}_error key."""
    f = tmp_path / "soft.jsonl"
    _write_jsonl(
        f,
        [
            {"task": "thermo", "calculator": "mace", "error": "Missing vibs.\n"},
            {"task": "thermo", "calculator": "mace", "error": ""},
        ],
    )

    summary = status_query.summarize([f])

    assert summary["counts"] == {"ok": 1, "error": 1, "total": 2}
    assert "Missing vibs." in list(summary["by_error_kind"])[0]

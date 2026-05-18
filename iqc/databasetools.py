import sqlite3
import json
import hashlib
import time
from pathlib import Path
import pandas as pd


SQLITE_TIMEOUT_SECONDS = 120
SQLITE_BUSY_TIMEOUT_MS = SQLITE_TIMEOUT_SECONDS * 1000

INSERT_CALCULATION_SQL = """
    INSERT OR IGNORE INTO calculations (
        geometry_hash, params_hash, calculator, model, task, blob_data
    ) VALUES (?, ?, ?, ?, ?, ?)
"""


def _connect(db_path):
    """Open a SQLite connection tuned for many small calculation inserts."""

    path = Path(db_path).expanduser()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=SQLITE_TIMEOUT_SECONDS)
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    _execute_with_busy_retry(conn, "PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _execute_with_busy_retry(conn, sql, parameters=()):
    deadline = time.monotonic() + SQLITE_TIMEOUT_SECONDS
    while True:
        try:
            return conn.execute(sql, parameters)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _ensure_schema(conn):
    _execute_with_busy_retry(
        conn,
        """
        CREATE TABLE IF NOT EXISTS calculations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            geometry_hash TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            calculator TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL,
            blob_data TEXT NOT NULL,
            UNIQUE(geometry_hash, params_hash, calculator, model, task)
        )
        """
    )


def calculation_key(initial_xyz, params, calculator="", model="", task=""):
    """Return the natural key used to identify an IQC calculation."""

    return (
        hash_string(initial_xyz),
        hash_string(params),
        calculator or "",
        model or "",
        task or "",
    )


def calculation_key_from_record(data, debug=False):
    """Return the calculation key for a result record after validation."""

    is_valid, missing_keys, available_keys = validate_data_structure(data, debug)

    if not is_valid:
        raise ValueError(
            f"Missing required keys in input data: {missing_keys}. "
            f"Available keys: {available_keys}"
        )

    return calculation_key(
        data["initial_xyz"],
        data["params"],
        data.get("calculator") or "",
        data.get("model") or "",
        data.get("task") or "",
    )


def create_database(db_path):
    """
    Create a SQLite database with a 'calculations' table if it does not exist.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        conn.commit()


def _hash_input(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_string(string):
    """
    Generate a SHA-256 hash for a given string, ignoring leading/trailing whitespace and empty lines.

    Parameters
    ----------
    string : str
        The input string to hash.

    Returns
    -------
    str
        The SHA-256 hash of the normalized string.
    """

    # generate a hash for a string to simplify the unique key
    value = _hash_input(string)
    string_hash = "\n".join(
        line.strip() for line in value.strip().splitlines() if line.strip()
    )

    return hashlib.sha256(string_hash.encode()).hexdigest()


def validate_data_structure(data, debug=False):
    """
    Validate that the data contains all required keys and provide debugging information.

    Parameters
    ----------
    data : dict
        The data dictionary to validate
    debug : bool, optional
        If True, print detailed information about the data structure

    Returns
    -------
    tuple
        (is_valid, missing_keys, available_keys)
    """
    required_keys = ["initial_xyz", "calculator", "task", "params"]
    missing_keys = [key for key in required_keys if key not in data]
    available_keys = list(data.keys())

    if debug:
        print(f"Required keys: {required_keys}")
        print(f"Available keys: {available_keys}")
        print(f"Missing keys: {missing_keys}")
        if data:
            print(f"Data preview: {dict(list(data.items())[:3])}...")

    return len(missing_keys) == 0, missing_keys, available_keys


def inspect_json_data(json_line, max_length=500):
    """
    Inspect JSON data to help debug structure issues.

    Parameters
    ----------
    json_line : str
        The JSON string to inspect
    max_length : int, optional
        Maximum length of the preview to show

    Returns
    -------
    dict
        Information about the JSON structure
    """
    try:
        data = json.loads(json_line)
        is_valid, missing_keys, available_keys = validate_data_structure(
            data, debug=True
        )

        # Create a preview of the data
        preview = {}
        for key, value in data.items():
            if isinstance(value, str):
                preview[key] = value[:100] + "..." if len(value) > 100 else value
            elif isinstance(value, dict):
                preview[key] = f"dict with {len(value)} keys"
            elif isinstance(value, list):
                preview[key] = f"list with {len(value)} items"
            else:
                preview[key] = (
                    str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                )

        return {
            "is_valid": is_valid,
            "missing_keys": missing_keys,
            "available_keys": available_keys,
            "data_preview": preview,
            "total_keys": len(data),
        }

    except json.JSONDecodeError as e:
        return {
            "error": "JSON decode error",
            "message": str(e),
            "json_preview": (
                json_line[:max_length] + "..."
                if len(json_line) > max_length
                else json_line
            ),
        }
    except Exception as e:
        return {"error": "Unexpected error", "message": str(e)}


def _prepare_calculation_row(json_line, debug=False):
    data = json.loads(json_line)

    geometry_hash, params_hash, calculator, model, task = calculation_key_from_record(
        data, debug=debug
    )
    blob_data = json.dumps(data, separators=(",", ":"), default=str)

    return (geometry_hash, params_hash, calculator, model, task, blob_data)


def calculation_exists(db_path, initial_xyz, params, calculator="", model="", task=""):
    """Return True if the calculation key already exists in the database."""

    path = Path(db_path).expanduser()
    if str(path) != ":memory:" and not path.exists():
        return False

    key = calculation_key(initial_xyz, params, calculator, model, task)

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            """
            SELECT 1
            FROM calculations
            WHERE geometry_hash = ?
              AND params_hash = ?
              AND calculator = ?
              AND model = ?
              AND task = ?
            LIMIT 1
            """,
            key,
        ).fetchone()
    return row is not None


def _insert_rows(conn, rows):
    before = conn.total_changes
    conn.executemany(INSERT_CALCULATION_SQL, rows)
    conn.commit()
    return conn.total_changes - before


def insert_entry(json_line, db_path, debug=False):
    """
    Insert a calculation entry into the database from a JSON string.

    Parameters
    ----------
    json_line : str
        JSON string containing calculation data.
    db_path : str
        Path to the SQLite database file.
    debug : bool, optional
        If True, print debug information during validation and insertion.
    """

    row = _prepare_calculation_row(json_line, debug=debug)
    with _connect(db_path) as conn:
        _ensure_schema(conn)
        return bool(_insert_rows(conn, [row]))


def insert_entries(json_lines, db_path, debug=False, batch_size=1000):
    """
    Insert calculation entries from an iterable of JSON strings.

    Returns a summary dictionary with processed, inserted, and duplicate counts.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    processed = 0
    inserted = 0
    batch = []

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        for json_line in json_lines:
            if not json_line.strip():
                continue
            batch.append(_prepare_calculation_row(json_line, debug=debug))
            processed += 1
            if len(batch) >= batch_size:
                inserted += _insert_rows(conn, batch)
                batch.clear()

        if batch:
            inserted += _insert_rows(conn, batch)

    return {
        "processed": processed,
        "inserted": inserted,
        "duplicates": processed - inserted,
    }


def merge_databases(target_db_path, source_db_path):
    """
    Merge calculation entries from a source database into a target database.

    Parameters
    ----------
    target_db_path : str
        Path to the target SQLite database file.
    source_db_path : str
        Path to the source SQLite database file.
    """

    create_database(target_db_path)
    with _connect(target_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("ATTACH DATABASE ? AS source_db", (str(source_db_path),))

        cursor.execute(
            """
            INSERT OR IGNORE INTO calculations (
                geometry_hash, params_hash, calculator, model, task, blob_data
            )
            SELECT geometry_hash, params_hash, calculator, model, task, blob_data
            FROM source_db.calculations
            """
        )

        conn.commit()
        cursor.execute("DETACH DATABASE source_db")


def database_to_dataframe(db):
    """
    Load the 'calculations' table from the database into a pandas DataFrame.

    Parameters
    ----------
    db : str
        Path to the SQLite database file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all rows from the 'calculations' table.
    """

    with _connect(db) as conn:
        _ensure_schema(conn)
        db_dataframe = pd.read_sql("SELECT * FROM calculations", conn)

    return db_dataframe


def database_to_data(db):
    """
    Convert the 'blob_data' column from the database into a pandas DataFrame of calculation data.

    Parameters
    ----------
    db : str
        Path to the SQLite database file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the parsed calculation data from 'blob_data'.
    """

    db_dataframe = database_to_dataframe(db)
    blob_data_list = [json.loads(data) for data in db_dataframe["blob_data"]]
    db_data = pd.DataFrame(blob_data_list)

    return db_data


def get_number_of_molecules(db):
    """
    Get the number of unique molecules in the database based on 'geometry_hash'.

    Parameters
    ----------
    db : str
        Path to the SQLite database file.

    Returns
    -------
    int
        Number of unique molecules in the database.
    """

    with _connect(db) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT COUNT(DISTINCT geometry_hash) FROM calculations"
        ).fetchone()

    return int(row[0] or 0)

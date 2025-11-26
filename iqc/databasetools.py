import sqlite3
import json
import hashlib
import pandas as pd

def create_database(db_path):
    """
    Create a SQLite database with a 'calculations' table if it does not exist.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    with sqlite3.connect(db_path) as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                geometry_hash TEXT,
                params_hash TEXT,
                calculator TEXT,
                model TEXT,
                task TEXT,
                blob_data TEXT,
                UNIQUE(geometry_hash, params_hash, calculator, model, task)
            )
        """
        )

        conn.commit()


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
    string_hash = "\n".join(
        line.strip() for line in string.strip().splitlines() if line.strip()
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
    required_keys = ["initial_xyz", "calculator", "model", "task", "params"]
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

    try:
        data = json.loads(json_line)

        is_valid, missing_keys, available_keys = validate_data_structure(data, debug)

        if not is_valid:
            raise ValueError(
                f"Missing required keys in input data: {missing_keys}. Available keys: {available_keys}"
            )

        geometry_hash = hash_string(data["initial_xyz"])
        params_hash = hash_string(data["params"])
        calculator = data.get("calculator") # mace, xtb, emt
        model = data.get("model") # small, medium, large
        task = data.get("task") # single, opt, vib, thermo
        blob_data = json.dumps(data)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR IGNORE INTO calculations (
                    geometry_hash, params_hash, calculator, model, task, blob_data
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (geometry_hash, params_hash, calculator, model, task, blob_data),
            )

            conn.commit()

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")

    except Exception as e:
        print(f"Error inserting entry: {e}")

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

    try:
        with sqlite3.connect(target_db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(f"ATTACH DATABASE '{source_db_path}' AS source_db")

            cursor.execute("""
                INSERT OR IGNORE INTO calculations (
                    geometry_hash, params_hash, calculator, model, task, blob_data
                )
                SELECT geometry_hash, params_hash, calculator, model, task, blob_data
                FROM source_db.calculations
            """)

            cursor.execute("DETACH DATABASE source_db")
            conn.commit()

    except sqlite3.DatabaseError as e:
        print(f"Database error during merge: {e}")

    except Exception as e:
        print(f"Unexpected error during merge: {e}")

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

    conn = sqlite3.connect(db)
    db_dataframe = pd.read_sql("SELECT * FROM calculations", conn)
    conn.close()

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
    blob_data_list = [json.loads(data) for data in db_dataframe['blob_data']]
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

    db_dataframe = database_to_dataframe(db)

    if db_dataframe.empty:
        return 0

    num_molecules = db_dataframe['geometry_hash'].nunique()

    return num_molecules
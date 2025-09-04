import sqlite3
import json
import hashlib
import pandas as pd

def create_database(db_path):

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


def process_json_file(json_file_path, db_path, debug=False, skip_errors=True):
    """
    Process a JSON file (one JSON object per line) and insert entries into the database.

    Parameters
    ----------
    json_file_path : str
        Path to the JSON file to process
    db_path : str
        Path to the SQLite database
    debug : bool, optional
        If True, print detailed debugging information
    skip_errors : bool, optional
        If True, skip lines with errors and continue processing

    Returns
    -------
    dict
        Summary of processing results
    """
    results = {"total_lines": 0, "successful_inserts": 0, "errors": [], "skipped": 0}

    try:
        with open(json_file_path, "r") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                results["total_lines"] += 1

                try:
                    insert_entry(line, db_path, debug=debug)
                    results["successful_inserts"] += 1

                except Exception as e:
                    error_info = {
                        "line_number": line_num,
                        "error": str(e),
                        "line_preview": line[:200] + "..." if len(line) > 200 else line,
                    }
                    results["errors"].append(error_info)

                    if debug:
                        print(f"Error on line {line_num}: {e}")
                        print(f"Line preview: {line[:200]}...")

                    if not skip_errors:
                        raise e
                    else:
                        results["skipped"] += 1

    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return results
    except Exception as e:
        print(f"Error processing file: {e}")
        return results

    # Print summary
    print(f"Processing complete:")
    print(f"  Total lines: {results['total_lines']}")
    print(f"  Successful inserts: {results['successful_inserts']}")
    print(f"  Errors: {len(results['errors'])}")
    print(f"  Skipped: {results['skipped']}")

    if results["errors"] and debug:
        print("\nFirst few errors:")
        for error in results["errors"][:5]:
            print(f"  Line {error['line_number']}: {error['error']}")

    return results

def merge_databases(target_db_path, source_db_path):

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

    conn = sqlite3.connect(db)
    db_dataframe = pd.read_sql("SELECT * FROM calculations", conn)
    conn.close()

    return db_dataframe

def database_to_data(db):

    db_dataframe = database_to_dataframe(db)
    blob_data_list = [json.loads(data) for data in db_dataframe['blob_data']]
    db_data = pd.DataFrame(blob_data_list)

    return db_data
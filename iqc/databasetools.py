import sqlite3
import json
import hashlib

def create_database(db_path):

    with sqlite3.connect(db_path) as conn:

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                geometry_hash TEXT,
                calculator TEXT,
                model TEXT,
                task TEXT,
                blob_data TEXT,
                UNIQUE(geometry_hash, calculator, model, task)
            )
        """)

        conn.commit()

def hash_geometry(xyz_string): 

    # generate a hash for initial_xyz to simplify the unique key
    xyz_hash = "\n".join(line.strip() for line in xyz_string.strip().splitlines() if line.strip())

    return hashlib.sha256(xyz_hash.encode()).hexdigest()

def insert_entry(json_line, db_path):

    try:
        data = json.loads(json_line)

        if not all(k in data for k in ["initial_xyz", "calculator", "model", "task"]):
            raise ValueError("Missing required keys in input data.")

        geometry_hash = hash_geometry(data["initial_xyz"])
        calculator = data.get("calculator")
        model = data.get("model")
        task = data.get("task")
        blob_data = json.dumps(data)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO calculations (
                    geometry_hash, calculator, model, task, blob_data
                ) VALUES (?, ?, ?, ?, ?)
            """, (geometry_hash, calculator, model, task, blob_data))
        
            conn.commit()
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")

    except Exception as e:
        print(f"Error inserting entry: {e}")
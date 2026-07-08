"""sqltools.py – helper utilities for storing and retrieving
 compressed quantum‑chemistry log blobs in a single SQLite database.

Designed for the single‑writer (rank‑0 fan‑in) pattern demonstrated in
`mpi_sqlite_compress.py`, but also safe for concurrent *readers*.

Dependencies
------------
* sqlite3      – stdlib
* zstandard    – `pip install zstandard`
* pathlib      – stdlib

Schema (created by ``init_db``)
------------------------------
CREATE TABLE blobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_rank INTEGER,
    seq         INTEGER,
    zstd_blob   BLOB
);

``source_rank`` and ``seq`` form the compound natural key that tells you
which rank produced which of its N JSON records.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import zstandard as zstd

# ---------------------------------------------------------------
# Low‑level connection helpers
# ---------------------------------------------------------------

def _open(db_path: Path | str) -> sqlite3.Connection:
    """Return a *writable* connection with tuned pragmas.

    Caller is responsible for closing the connection.
    """
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con


# ---------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------

def init_db(db_path: Path | str) -> None:
    """Create the database file (if missing) and its schema."""
    db_path = Path(db_path)
    first_time = not db_path.exists()
    con = _open(db_path)
    if first_time:
        con.execute(
            """
            CREATE TABLE blobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_rank INTEGER,
                seq         INTEGER,
                zstd_blob   BLOB
            );
            """
        )
    con.close()


# ---------------------------------------------------------------
# Writing utilities
# ---------------------------------------------------------------

def insert_blobs(
    db_path: Path | str,
    rows: Iterable[Tuple[int, int, bytes]],
) -> None:
    """Insert (source_rank, seq, zstd_blob) tuples in one transaction.

    Parameters
    ----------
    db_path : str or Path
        Database location. Created on‑the‑fly if needed.
    rows : iterable of tuples
        Each tuple must be ``(rank:int, seq:int, blob:bytes)`` *already
        compressed* with zstd. ``blob`` will be wrapped with
        :class:`sqlite3.Binary` automatically.
    """
    con = _open(db_path)
    cur = con.cursor()
    cur.executemany(
        "INSERT INTO blobs(source_rank, seq, zstd_blob) VALUES (?,?,?)",
        ((r, s, sqlite3.Binary(b)) for r, s, b in rows),
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------
# Reading utilities
# ---------------------------------------------------------------
_cctx = zstd.ZstdDecompressor()


def fetch_blob(db_path: Path | str, rank: int, seq: int) -> bytes | None:
    """Return raw *compressed* blob for (rank, seq) or *None* if absent."""
    con = sqlite3.connect(str(db_path))
    cur = con.execute(
        "SELECT zstd_blob FROM blobs WHERE source_rank=? AND seq=?",
        (rank, seq),
    )
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


def fetch_json(db_path: Path | str, rank: int, seq: int) -> dict | None:
    """Return *decompressed* JSON object for (rank, seq)."""
    blob = fetch_blob(db_path, rank, seq)
    if blob is None:
        return None
    return json.loads(_cctx.decompress(blob))


def iter_json(db_path: Path | str, ranks: List[int] | None = None):
    """Yield (rank, seq, obj) for all or selected ranks.

    Reads are streaming and *do not* load the entire table into RAM.
    """
    con = sqlite3.connect(str(db_path))
    sql = "SELECT source_rank, seq, zstd_blob FROM blobs"
    params: Tuple = ()
    if ranks is not None:
        sql += " WHERE source_rank IN ({})".format(",".join("?" * len(ranks))
        )
        params = tuple(ranks)
    cur = con.execute(sql, params)
    for r, seq, blob in cur:
        yield r, seq, json.loads(_cctx.decompress(blob))
    con.close()


# ---------------------------------------------------------------
# Convenience helpers for writers
# ---------------------------------------------------------------
_ccompressor = zstd.ZstdCompressor(level=19)


def compress_json(obj: dict) -> bytes:
    """Return zstd‑compressed bytes for a JSON‑serialisable object."""
    return _ccompressor.compress(json.dumps(obj).encode())


__all__ = [
    "init_db",
    "insert_blobs",
    "fetch_blob",
    "fetch_json",
    "iter_json",
    "compress_json",
]

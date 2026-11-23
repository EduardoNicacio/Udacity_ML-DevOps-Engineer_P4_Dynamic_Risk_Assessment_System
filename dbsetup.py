"""
SQLite bookkeeping for the Dynamic Risk Assessment System (standout feature).

Uses only the standard library. ``init_db()`` is idempotent, so it is safe to
call from ingestion, scoring, diagnostics and from the command line.

Tables
------
ingested_files  (id, filename, ingested_at)
model_scores    (id, f1_score, scored_at, model_path)
diagnostics_log (id, run_at, ingestion_time, training_time, missing_data_json)
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'risk_assessment.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS model_scores (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    f1_score  REAL    NOT NULL,
    scored_at TEXT    NOT NULL,
    model_path TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnostics_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,
    ingestion_time  REAL NOT NULL,
    training_time   REAL NOT NULL,
    missing_data_json TEXT NOT NULL
);
"""


def get_connection():
    """Open a connection to the project database."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the database file and all tables when missing."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)
    return DB_PATH


if __name__ == '__main__':
    path = init_db()
    with get_connection() as connection:
        tables = [row['name'] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]

    print(f"Database ready: {path}")
    print(f"Tables: {', '.join(tables)}")

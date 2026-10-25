"""
build_db.py — ingest experiment_results_full.json into SQLite.
Run: python src/build_db.py
"""
import json
import sqlite3
from pathlib import Path

RESULTS_JSON = Path(__file__).parent / "results" / "experiment_results_full.json"
DB_PATH      = Path(__file__).parent / "results" / "experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode        TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    round       INTEGER NOT NULL,
    fed_mae     REAL NOT NULL,
    fed_rmse    REAL NOT NULL,
    cen_mae     REAL NOT NULL,
    cen_rmse    REAL NOT NULL,
    comm_bytes  INTEGER NOT NULL,
    client_maes TEXT NOT NULL,
    UNIQUE(mode, seed, round)
);
"""

def main():
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)

    rows = data["rounds"]
    conn.executemany("""
        INSERT OR REPLACE INTO rounds
        (mode, seed, round, fed_mae, fed_rmse, cen_mae, cen_rmse, comm_bytes, client_maes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (r["mode"], r["seed"], r["round"], r["fed_mae"], r["fed_rmse"],
         r["cen_mae"], r["cen_rmse"], r["comm_bytes"], json.dumps(r["client_maes"]))
        for r in rows
    ])

    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
    print(f"Ingested {len(rows)} rows. Total rows in DB: {n}")
    conn.close()

if __name__ == "__main__":
    main()
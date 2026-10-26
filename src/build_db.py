"""
build_db.py — ingest experiment_results_full.json and
experiment_results_allclients.json into SQLite.
Run: python src/build_db.py
"""
import json
import sqlite3
from pathlib import Path

RESULTS_DIR     = Path(__file__).parent / "results"
RESULTS_FULL    = RESULTS_DIR / "experiment_results_full.json"       # 3 clients, old sweep
RESULTS_ALLCLI  = RESULTS_DIR / "experiment_results_allclients.json" # 3/4/5 clients, new sweep
DB_PATH         = RESULTS_DIR / "experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode        TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    num_clients INTEGER NOT NULL DEFAULT 3,
    round       INTEGER NOT NULL,
    fed_mae     REAL NOT NULL,
    fed_rmse    REAL NOT NULL,
    cen_mae     REAL NOT NULL,
    cen_rmse    REAL NOT NULL,
    comm_bytes  INTEGER NOT NULL,
    client_maes TEXT NOT NULL,
    UNIQUE(mode, seed, num_clients, round)
);
"""

def load_rows(path, default_clients=3):
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    rows = data["rounds"]
    for r in rows:
        r.setdefault("num_clients", default_clients)
    return rows

RESULTS_20ROUNDS = RESULTS_DIR / "experiment_results_20rounds.json"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS rounds")
    conn.execute(SCHEMA)

    all_rows = load_rows(RESULTS_20ROUNDS)   # single source of truth now, replaces old 10-round files

    conn.executemany("""
        INSERT OR REPLACE INTO rounds
        (mode, seed, num_clients, round, fed_mae, fed_rmse, cen_mae, cen_rmse, comm_bytes, client_maes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (r["mode"], r["seed"], r["num_clients"], r["round"], r["fed_mae"], r["fed_rmse"],
         r["cen_mae"], r["cen_rmse"], r["comm_bytes"], json.dumps(r["client_maes"]))
        for r in all_rows
    ])
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
    print(f"Ingested {len(all_rows)} rows. Total rows in DB: {n}")
    conn.close()

if __name__ == "__main__":
    main()
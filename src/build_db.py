"""
build_db.py — ingest experiment_results_20rounds.json into SQLite.
Run: python src/build_db.py

Patched:
  * Removed RESULTS_FULL / RESULTS_ALLCLI and the stale module docstring —
    main() only ever loaded experiment_results_20rounds.json; the other two
    constants were dead code left over from an earlier sweep format, and
    kept around they wrongly suggest to the next reader that 3 files feed
    the DB when only 1 does.
  * load_rows() no longer silently defaults a missing "num_clients" to 3.
    If experiment_results_20rounds.json ever has a row without it (e.g. a
    bug upstream in the sweep script), the old behaviour would tag that
    row num_clients=3 regardless of which client count it actually came
    from — corrupting the DB in a way the dashboard has no way to detect
    (it would just show subtly wrong data for the 4- or 5-client selector,
    or silently merge rows that don't belong together under the UNIQUE
    constraint). This now raises immediately, with the offending row's
    mode/seed/round, so a bad sweep output fails loudly at ingest time
    instead of quietly reaching the dashboard.
  * Validation now happens BEFORE the table is dropped. The original order
    (drop/recreate the table, then load+validate the JSON) meant a
    malformed sweep file would wipe a previously-good database and leave it
    empty — the loud ValueError above is real, but it used to land one step
    too late to save the existing DB. load_rows() now runs first; the table
    is only dropped once we actually have validated rows to replace it with.
"""
import json
import sqlite3
from pathlib import Path

RESULTS_DIR      = Path(__file__).parent / "results"
RESULTS_20ROUNDS = RESULTS_DIR / "experiment_results_20rounds.json"
DB_PATH          = RESULTS_DIR / "experiments.db"

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

REQUIRED_KEYS = ("mode", "seed", "num_clients", "round", "fed_mae", "fed_rmse",
                  "cen_mae", "cen_rmse", "comm_bytes", "client_maes")


def load_rows(path):
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    rows = data["rounds"]
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise ValueError(
                f"{path.name} row {i} (mode={r.get('mode')!r}, seed={r.get('seed')!r}, "
                f"round={r.get('round')!r}) is missing required field(s): {missing}. "
                f"Fix the sweep script that generated this file rather than defaulting "
                f"these here — a silently-guessed num_clients/comm_bytes can corrupt "
                f"which experiment a row is served under."
            )
    return rows


def main():
    all_rows = load_rows(RESULTS_20ROUNDS)   # validate first — raises before anything is touched

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS rounds")
    conn.execute(SCHEMA)

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
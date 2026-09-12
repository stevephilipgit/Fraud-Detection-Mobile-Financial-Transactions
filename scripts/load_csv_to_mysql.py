"""Load Fraud_Analysis_Dataset.csv into MySQL.

Flow:
    Fraud_Analysis_Dataset.csv -> Python -> MySQL (bia_fraud_detection.transactions)

Credentials are read ONLY from environment variables / .env (see .env.example).
Never hardcode credentials in this file.

Idempotent by default: if `transactions` already holds rows the load is
skipped (a duplicate dataset is never auto-inserted). Pass --force to clear
and reload explicitly.

Usage:
    python scripts/load_csv_to_mysql.py [--rows N] [--force]

    --rows N : optional, load only the first N rows (useful for smoke tests)
    --force  : truncate and reload even though the table already has rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# Allow running as `python scripts/load_csv_to_mysql.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, database  # noqa: E402

CSV_PATH = config.PROJECT_ROOT / "Fraud_Analysis_Dataset.csv"


def load_csv(path: Path, rows: int = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=rows)
    expected = {"step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
                "isFraud"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df


def _count_rows() -> int:
    with database.get_engine().connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load fraud CSV into MySQL")
    parser.add_argument("--rows", type=int, default=None,
                        help="Load only the first N rows (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Truncate and reload even if transactions already has rows")
    args = parser.parse_args()

    print("Initializing database (bia_fraud_detection + tables)...")
    try:
        database.init_database()
    except database.DatabaseUnavailableError as exc:
        print(f"ERROR: MySQL is not reachable: {exc}")
        return 1
    print(f"Ready: {config.DB_USER}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")

    df = load_csv(CSV_PATH, rows=args.rows)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    existing = _count_rows()
    if existing and not args.force:
        print(f"transactions already holds {existing} rows — skipping load (idempotent).")
        print("Use --force to truncate and reload.")
    else:
        if existing and args.force:
            print(f"--force: clearing existing {existing} rows before reload.")
            with database.get_engine().begin() as conn:
                conn.execute(text("TRUNCATE TABLE transactions"))

        # isFraud is kept nullable; the CSV always carries the label, but future
        # application-generated transactions will not.
        df = df.copy()
        df["isFraud"] = df["isFraud"].astype("Int64")  # nullable integer

        inserted = database.insert_transactions(df.to_dict(orient="records"))
        print(f"Inserted {inserted} rows into transactions")

    count = _count_rows()
    print(f"Verification: SELECT COUNT(*) FROM transactions -> {count}")
    print(f"CSV rows read: {len(df)}")
    if count != len(df):
        print(f"WARNING: row-count mismatch (DB {count} vs CSV {len(df)})")
        return 2
    print("Row count matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

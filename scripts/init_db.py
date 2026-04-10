"""
Initialize the lakehouse database.

Creates all tables from data_layer/schema.sql into data/db/oracle.duckdb.
Safe to run multiple times (all CREATE statements use IF NOT EXISTS).

Usage:
    python scripts/init_db.py              # initialize or verify
    python scripts/init_db.py --stats      # show row counts
    python scripts/init_db.py --force      # DROP and recreate (dev only!)
    python scripts/init_db.py --export-parquet   # export tables to Parquet

The database file is at: data/db/oracle.duckdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_init(force: bool = False) -> None:
    from data_layer.db import init_schema, get_db_path, schema_stats

    if force:
        print("WARNING: --force will DROP all tables and recreate them.")
        ans = input("Type 'yes' to confirm: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return

    print(f"Database: {get_db_path()}")
    init_schema(force=force)

    stats = schema_stats()
    print("\nTable row counts after initialization:")
    for table, count in stats.items():
        status = f"{count:>8,d}" if count is not None else "   (missing)"
        print(f"  {table:<40} {status}")


def cmd_stats() -> None:
    from data_layer.db import schema_stats, get_db_path

    print(f"Database: {get_db_path()}")
    stats = schema_stats()

    total_rows = sum(v for v in stats.values() if v is not None)
    print(f"\nTotal rows: {total_rows:,}")
    print("\nPer-table:")
    for table, count in stats.items():
        if count is None:
            print(f"  {table:<40}  (table missing)")
        else:
            print(f"  {table:<40}  {count:>10,d}")


def cmd_export_parquet() -> None:
    from data_layer.db import get_db, get_db_path

    export_dir = get_db_path().parent / "parquet_export"
    export_dir.mkdir(exist_ok=True)
    db = get_db()

    tables_to_export = [
        "raw_documents", "canonical_documents", "canonical_events",
        "feature_snapshots", "questions", "question_resolutions",
        "predictions", "data_lineage",
    ]

    for table in tables_to_export:
        out_path = export_dir / f"{table}.parquet"
        try:
            db.execute(
                f"COPY (SELECT * FROM {table}) TO '{out_path}' (FORMAT PARQUET)"
            )
            count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  Exported {table}: {count:,} rows → {out_path}")
        except Exception as e:
            print(f"  SKIP {table}: {e}")

    print(f"\nParquet files written to: {export_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize geopolitical oracle lakehouse DB")
    parser.add_argument("--force", action="store_true",
                        help="DROP all tables and recreate (DESTRUCTIVE)")
    parser.add_argument("--stats", action="store_true",
                        help="Show current row counts and exit")
    parser.add_argument("--export-parquet", action="store_true",
                        help="Export all tables to Parquet files")
    args = parser.parse_args()

    if args.stats:
        cmd_stats()
    elif args.export_parquet:
        cmd_export_parquet()
    else:
        cmd_init(force=args.force)


if __name__ == "__main__":
    main()

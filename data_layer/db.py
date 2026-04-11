"""
DuckDB connection manager for the geopolitical oracle lakehouse.

Provides:
  - get_db()          : returns an open DuckDB connection (singleton per process)
  - init_schema()     : creates all tables from schema.sql if not present
  - close_db()        : closes the connection
  - get_db_path()     : path to the .duckdb file

The database is a single .duckdb file at data/db/oracle.duckdb.
For read-only access (e.g. in tests), use get_db(read_only=True).

For training data export, use:
    db.execute("COPY feature_snapshots TO 'data/db/snapshots.parquet' (FORMAT PARQUET)")
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "oracle.duckdb"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# ── Thread-safe singleton connection ─────────────────────────────────────────
_conn_lock = threading.Lock()
_conn: Optional[object] = None   # duckdb.DuckDBPyConnection


def get_db_path() -> Path:
    return _DB_PATH


def get_db(read_only: bool = False):
    """
    Return the open DuckDB connection.

    Thread-safe singleton: one write connection per process.
    For parallel reads, call duckdb.connect(str(_DB_PATH), read_only=True)
    directly — DuckDB supports multiple concurrent readers.
    """
    global _conn
    if _conn is not None:
        return _conn

    with _conn_lock:
        if _conn is not None:
            return _conn
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb not installed. Run: pip install duckdb"
            )

        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Opening DuckDB at %s (read_only=%s)", _DB_PATH, read_only)
        _conn = duckdb.connect(str(_DB_PATH), read_only=read_only)
        return _conn


def init_schema(force: bool = False) -> None:
    """
    Run schema.sql against the DuckDB database.

    All CREATE TABLE statements use IF NOT EXISTS, so this is idempotent.
    Call this once at startup or from scripts/init_db.py.

    Args:
        force: if True, drop and recreate everything (DESTRUCTIVE — dev only)
    """
    db = get_db()
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")

    if force:
        logger.warning("force=True: dropping all lakehouse tables")
        _drop_all(db)

    logger.info("Initializing lakehouse schema from %s", _SCHEMA_PATH)
    try:
        db.execute(schema_sql)
        logger.info("Schema initialized successfully")
    except Exception as e:
        logger.error("Schema initialization failed: %s", e)
        raise

    # Incremental column migrations (safe to run on existing DBs)
    _migrate_predictions_blend_columns(db)


def _drop_all(db) -> None:
    """Drop all lakehouse tables in reverse dependency order. DESTRUCTIVE."""
    tables = [
        "data_lineage", "predictions", "question_resolutions", "questions",
        "question_templates", "embedding_registry", "feature_snapshots",
        "entity_relations_temporal", "event_arguments", "canonical_events",
        "document_entities", "entity_aliases", "entities",
        "canonical_documents", "raw_documents",
    ]
    for t in tables:
        db.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    views = [
        "active_relations", "overdue_questions",
        "training_ready_snapshots", "calibration_audit", "entity_recent_events",
    ]
    for v in views:
        db.execute(f"DROP VIEW IF EXISTS {v}")


def _migrate_predictions_blend_columns(db) -> None:
    """
    Add market-blend audit columns to the predictions table if they don't exist.

    Safe to call on any existing database — uses ADD COLUMN IF NOT EXISTS.
    This is the Phase B migration: allows the blend calibrator to query
    p_model_raw and p_market_raw for all past predictions.
    """
    blend_columns = [
        ("p_model_raw",             "FLOAT"),
        ("p_market_raw",            "FLOAT"),
        ("market_weight",           "FLOAT"),
        ("market_sources",          "VARCHAR[]"),
        ("market_match_score",      "FLOAT"),
        ("blend_strategy",          "VARCHAR"),
        ("blend_strategy_version",  "VARCHAR"),
        ("n_market_signals",        "SMALLINT"),
        ("market_gate_passed",      "BOOLEAN"),
        ("market_gate_reason",      "VARCHAR"),
    ]
    for col_name, col_type in blend_columns:
        try:
            db.execute(
                f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
            )
        except Exception as e:
            # Table may not exist yet (first init) — silently skip; schema.sql creates it
            logger.debug("blend column migration skip (%s %s): %s", col_name, col_type, e)


def close_db() -> None:
    """Close the singleton connection. Call at process exit if needed."""
    global _conn
    if _conn is not None:
        with _conn_lock:
            if _conn is not None:
                _conn.close()
                _conn = None
                logger.info("DuckDB connection closed")


def table_exists(table_name: str) -> bool:
    """Check whether a table exists in the current database."""
    db = get_db()
    result = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = ?",
        [table_name]
    ).fetchone()
    return result[0] > 0


def row_count(table_name: str) -> int:
    """Return the number of rows in a table."""
    db = get_db()
    return db.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def schema_stats() -> dict:
    """Return row counts for all lakehouse tables. Useful for health checks."""
    tables = [
        "raw_documents", "canonical_documents", "entities", "entity_aliases",
        "document_entities", "canonical_events", "event_arguments",
        "entity_relations_temporal", "feature_snapshots", "embedding_registry",
        "question_templates", "questions", "question_resolutions",
        "predictions", "data_lineage",
    ]
    stats = {}
    db = get_db()
    for t in tables:
        try:
            stats[t] = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            stats[t] = None  # table doesn't exist yet
    return stats

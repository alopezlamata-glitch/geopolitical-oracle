"""
Geopolitical Oracle — Lakehouse temporal data layer.

Modules:
  db       : DuckDB connection manager + schema initialization
  schema   : SQL DDL (schema.sql loaded by db.init_schema())
  lineage  : Data lineage recorder (non-fatal, observability only)
  writers  : High-level insert helpers for each layer

Quick start:
    from data_layer.db import init_schema, schema_stats
    init_schema()   # idempotent — safe to call multiple times
    print(schema_stats())
"""
from data_layer.db import get_db, init_schema, close_db, schema_stats, table_exists

__all__ = ["get_db", "init_schema", "close_db", "schema_stats", "table_exists"]

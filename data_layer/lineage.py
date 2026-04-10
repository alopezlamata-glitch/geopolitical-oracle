"""
Data lineage recorder.

Every transformation — raw doc → canonical, doc → event, events → features,
features → prediction — should call record_lineage() so that any prediction
can be traced back to the exact source documents that produced it.

Usage:
    from data_layer.lineage import record_lineage

    record_lineage(
        output_type="canonical_document",
        output_id=doc_id,
        input_type="raw_document",
        input_ids=[raw_doc_id],
        processor_name="normalizer.canonical",
        processor_version="v2",
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _new_lineage_id(output_type: str, output_id: str) -> str:
    raw = f"{output_type}:{output_id}:{datetime.now(timezone.utc).isoformat()}"
    return "lin_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def record_lineage(
    output_type: str,
    output_id: str,
    input_type: str,
    input_ids: list[str],
    processor_name: str,
    processor_version: str,
    processor_config: Optional[dict] = None,
    confidence: Optional[float] = None,
    duration_ms: Optional[int] = None,
    warnings: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
    is_deterministic: bool = True,
) -> str:
    """
    Record a data transformation in the lineage table.

    Returns the lineage_id of the created record.
    Silently skips if DuckDB is not initialized (non-fatal).
    """
    from data_layer.db import get_db, table_exists

    lineage_id = _new_lineage_id(output_type, output_id)
    now = datetime.now(timezone.utc)

    try:
        if not table_exists("data_lineage"):
            logger.debug("data_lineage table not yet initialized — skipping lineage record")
            return lineage_id

        db = get_db()
        db.execute(
            """
            INSERT INTO data_lineage (
                lineage_id, output_type, output_id,
                input_type, input_ids,
                processor_name, processor_version, processor_config,
                processed_at, duration_ms, confidence,
                warnings, errors, is_deterministic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (lineage_id) DO NOTHING
            """,
            [
                lineage_id, output_type, output_id,
                input_type, input_ids,
                processor_name, processor_version,
                json.dumps(processor_config) if processor_config else None,
                now, duration_ms, confidence,
                warnings or [], errors or [],
                is_deterministic,
            ]
        )
    except Exception as e:
        # Lineage is observability, not correctness — never raise
        logger.debug("lineage record failed (non-fatal): %s", e)

    return lineage_id


def get_lineage_chain(output_id: str, output_type: str, max_depth: int = 10) -> list[dict]:
    """
    Trace the full lineage chain backwards from an artifact.
    Returns list of lineage records from newest (the artifact) to oldest (raw source).
    """
    from data_layer.db import get_db, table_exists

    if not table_exists("data_lineage"):
        return []

    db = get_db()
    chain = []
    current_ids = [output_id]
    current_type = output_type
    visited = set()

    for _ in range(max_depth):
        if not current_ids:
            break

        rows = db.execute(
            "SELECT * FROM data_lineage WHERE output_id = ANY(?) AND output_type = ?",
            [current_ids, current_type]
        ).fetchall()

        if not rows:
            break

        cols = [d[0] for d in db.description]
        for row in rows:
            record = dict(zip(cols, row))
            key = record["lineage_id"]
            if key not in visited:
                chain.append(record)
                visited.add(key)

        # Step back: next level uses the inputs of this level
        next_ids = []
        next_type = None
        for record in rows:
            if record.get("input_ids"):
                next_ids.extend(record["input_ids"])
            if next_type is None and record.get("input_type"):
                next_type = record["input_type"]

        current_ids = list(set(next_ids))
        current_type = next_type or ""

    return chain

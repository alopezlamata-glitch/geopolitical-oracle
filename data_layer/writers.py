"""
High-level write helpers for inserting into each lakehouse layer.

These are thin wrappers that validate, assign IDs, and insert.
They also call record_lineage() automatically.

Usage:
    from data_layer.writers import write_raw_document, write_canonical_event

    doc_id = write_raw_document(
        source="rss",
        source_url="https://...",
        body="text...",
        published_at=datetime(...),
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_WRITER_VERSION = "1.0"


# ── ID generators ─────────────────────────────────────────────────────────────

def _raw_doc_id(source: str, url: Optional[str], body: str, ingested_at: datetime) -> str:
    raw = f"{source}|{url or ''}|{body[:256]}|{ingested_at.isoformat()}"
    return "raw_" + hashlib.sha256(raw.encode()).hexdigest()[:32]

def _content_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()

def _doc_id(raw_doc_id: str) -> str:
    return "doc_" + raw_doc_id[4:]  # replace 'raw_' prefix with 'doc_'

def _event_id(doc_id: str, event_type: str, event_time: datetime, idx: int = 0) -> str:
    raw = f"{doc_id}|{event_type}|{event_time.isoformat()}|{idx}"
    return "evt_" + hashlib.sha256(raw.encode()).hexdigest()[:32]

def _entity_id(canonical_name: str, entity_type: str) -> str:
    raw = f"{entity_type}|{canonical_name.lower().strip()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]

def _snapshot_id(subject_entity_id: Optional[str], as_of_time: datetime, predicate: Optional[str]) -> str:
    raw = f"{subject_entity_id or ''}|{as_of_time.isoformat()}|{predicate or ''}"
    return "snap_" + hashlib.sha256(raw.encode()).hexdigest()[:32]

def _question_id(raw_text: str, created_at: datetime) -> str:
    raw = f"{raw_text}|{created_at.isoformat()}"
    return "q_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Raw document ──────────────────────────────────────────────────────────────

def write_raw_document(
    source: str,
    body: str,
    published_at: Optional[datetime] = None,
    title: Optional[str] = None,
    source_url: Optional[str] = None,
    source_feed: Optional[str] = None,
    language: str = "en",
    source_quality: float = 0.5,
    is_official_source: bool = False,
    collector_version: str = _WRITER_VERSION,
    http_status: Optional[int] = None,
) -> Optional[str]:
    """
    Insert a raw document. Returns raw_doc_id.
    Returns None if insert fails (non-fatal — callers may continue).
    """
    from data_layer.db import get_db, table_exists
    from data_layer.lineage import record_lineage

    if not table_exists("raw_documents"):
        logger.debug("raw_documents not initialized — skipping write")
        return None

    ingested_at = datetime.now(timezone.utc)
    raw_doc_id = _raw_doc_id(source, source_url, body, ingested_at)
    content_hash = _content_hash(body)

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO raw_documents (
                raw_doc_id, content_hash, source, source_url, source_feed,
                published_at, ingested_at, title, body, language,
                source_quality, is_official_source, http_status, collector_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (content_hash, source) DO NOTHING
            """,
            [
                raw_doc_id, content_hash, source, source_url, source_feed,
                published_at, ingested_at, title, body, language,
                source_quality, is_official_source, http_status, collector_version,
            ]
        )
        record_lineage(
            output_type="raw_document", output_id=raw_doc_id,
            input_type="external_source", input_ids=[source_url or source],
            processor_name="collector", processor_version=collector_version,
        )
        return raw_doc_id
    except Exception as e:
        logger.warning("write_raw_document failed: %s", e)
        return None


# ── Canonical event ───────────────────────────────────────────────────────────

def write_canonical_event(
    doc_id: str,
    event_type: str,
    event_time: datetime,
    actor_entity_ids: Optional[list[str]] = None,
    target_entity_ids: Optional[list[str]] = None,
    location_entity_id: Optional[str] = None,
    intensity: Optional[float] = None,
    polarity: Optional[float] = None,
    certainty: Optional[float] = None,
    is_official: bool = False,
    fatalities: int = 0,
    source_quality: Optional[float] = None,
    independent_sources: int = 1,
    contradiction_score: float = 0.0,
    extractor_version: str = _WRITER_VERSION,
    extraction_method: str = "rule",
    confidence: float = 1.0,
    idx: int = 0,
) -> Optional[str]:
    """
    Insert a canonical event. Returns event_id or None on failure.
    """
    from data_layer.db import get_db, table_exists
    from data_layer.lineage import record_lineage

    if not table_exists("canonical_events"):
        logger.debug("canonical_events not initialized — skipping write")
        return None

    event_id = _event_id(doc_id, event_type, event_time, idx)

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO canonical_events (
                event_id, doc_id, event_time, event_type,
                actor_entity_ids, target_entity_ids, location_entity_id,
                intensity, polarity, certainty, is_official, fatalities,
                source_quality, independent_sources, contradiction_score,
                extractor_version, extraction_method, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (event_id) DO NOTHING
            """,
            [
                event_id, doc_id, event_time, event_type,
                actor_entity_ids or [], target_entity_ids or [], location_entity_id,
                intensity, polarity, certainty, is_official, fatalities,
                source_quality, independent_sources, contradiction_score,
                extractor_version, extraction_method, confidence,
            ]
        )
        record_lineage(
            output_type="canonical_event", output_id=event_id,
            input_type="canonical_document", input_ids=[doc_id],
            processor_name="event_extractor", processor_version=extractor_version,
        )
        return event_id
    except Exception as e:
        logger.warning("write_canonical_event failed: %s", e)
        return None


# ── Feature snapshot ──────────────────────────────────────────────────────────

def write_feature_snapshot(
    as_of_time: datetime,
    explicit_features: dict,
    feature_schema_ver: str,
    builder_version: str,
    subject_entity_id: Optional[str] = None,
    question_id: Optional[str] = None,
    question_predicate: Optional[str] = None,
    question_deadline: Optional[datetime] = None,
    doc_ids_used: Optional[list[str]] = None,
    event_ids_used: Optional[list[str]] = None,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    outcome: Optional[int] = None,
) -> Optional[str]:
    """
    Persist a feature snapshot. Returns snapshot_id or None on failure.

    This is the most critical write for training reproducibility: it captures
    exactly what the model saw at prediction time, with as_of_time as the
    strict data cutoff.
    """
    from data_layer.db import get_db, table_exists
    from data_layer.lineage import record_lineage

    if not table_exists("feature_snapshots"):
        logger.debug("feature_snapshots not initialized — skipping write")
        return None

    snapshot_id = _snapshot_id(subject_entity_id, as_of_time, question_predicate)
    built_at = datetime.now(timezone.utc)

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO feature_snapshots (
                snapshot_id, question_id, subject_entity_id, as_of_time,
                question_predicate, question_deadline,
                explicit_features, feature_schema_ver,
                doc_ids_used, event_ids_used, window_start, window_end,
                builder_version, built_at, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (snapshot_id) DO NOTHING
            """,
            [
                snapshot_id, question_id, subject_entity_id, as_of_time,
                question_predicate, question_deadline,
                json.dumps(explicit_features), feature_schema_ver,
                doc_ids_used or [], event_ids_used or [],
                window_start, window_end,
                builder_version, built_at, outcome,
            ]
        )
        input_ids = (doc_ids_used or []) + (event_ids_used or [])
        record_lineage(
            output_type="feature_snapshot", output_id=snapshot_id,
            input_type="canonical_event",
            input_ids=input_ids or ["(no events)"],
            processor_name="features.builder", processor_version=builder_version,
        )
        return snapshot_id
    except Exception as e:
        logger.warning("write_feature_snapshot failed: %s", e)
        return None


# ── Question ──────────────────────────────────────────────────────────────────

def write_question(
    raw_text: str,
    predicate: str,
    event_family: str,
    subject: str,
    deadline,           # date or datetime
    resolution_rule: str,
    as_of_time: datetime,
    subject_entity_id: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    is_negated: bool = False,
    ood_score: Optional[float] = None,
    matched_model: Optional[str] = None,
    parse_confidence: Optional[float] = None,
    source: str = "user",
    status: str = "open",
) -> Optional[str]:
    """Write a parsed question to the registry. Returns question_id."""
    from data_layer.db import get_db, table_exists

    if not table_exists("questions"):
        return None

    created_at = datetime.now(timezone.utc)
    question_id = _question_id(raw_text, created_at)

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO questions (
                question_id, raw_text, predicate, event_family, subject,
                subject_entity_id, jurisdiction, is_negated,
                deadline, resolution_rule, status,
                ood_score, matched_model, parse_confidence,
                created_at, as_of_time, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (question_id) DO NOTHING
            """,
            [
                question_id, raw_text, predicate, event_family, subject,
                subject_entity_id, jurisdiction, is_negated,
                deadline, resolution_rule, status,
                ood_score, matched_model, parse_confidence,
                created_at, as_of_time, source,
            ]
        )
        return question_id
    except Exception as e:
        logger.warning("write_question failed: %s", e)
        return None


# ── Entity ────────────────────────────────────────────────────────────────────

def write_entity(
    canonical_name: str,
    entity_type: str,
    country: Optional[str] = None,
    description: Optional[str] = None,
    wikidata_id: Optional[str] = None,
    conflict_baserate: Optional[float] = None,
    polity_norm: Optional[float] = None,
    mil_spending_norm: Optional[float] = None,
) -> Optional[str]:
    """
    Upsert a canonical entity. Returns entity_id.

    Uses ON CONFLICT DO NOTHING — the same entity may be inserted from
    multiple prediction runs; only the first write creates the record.
    """
    from data_layer.db import get_db, table_exists
    from datetime import datetime, timezone

    if not table_exists("entities"):
        return None
    if not canonical_name.strip():
        return None

    entity_id = _entity_id(canonical_name, entity_type)
    now = datetime.now(timezone.utc)

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO entities (
                entity_id, canonical_name, entity_type,
                wikidata_id, country, description,
                conflict_baserate, polity_norm, mil_spending_norm,
                first_seen_at, last_updated_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            ON CONFLICT (entity_id) DO NOTHING
            """,
            [
                entity_id, canonical_name.strip(), entity_type,
                wikidata_id, country, description,
                conflict_baserate, polity_norm, mil_spending_norm,
                now, now,
            ]
        )
        return entity_id
    except Exception as e:
        logger.warning("write_entity failed (%s): %s", canonical_name, e)
        return None


def write_entity_alias(
    entity_id: str,
    alias: str,
    source: str = "extracted",
    confidence: float = 0.9,
    language: str = "en",
) -> Optional[str]:
    """
    Write an alias for a known entity. Returns alias_id or None.

    The (alias, alias_language) pair is unique — ON CONFLICT DO NOTHING
    if the alias is already registered (possibly to a different entity).
    """
    from data_layer.db import get_db, table_exists
    import hashlib

    if not table_exists("entity_aliases"):
        return None
    if not alias.strip():
        return None

    alias_id = "ali_" + hashlib.sha256(
        f"{alias.lower().strip()}|{language}".encode()
    ).hexdigest()[:24]

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO entity_aliases (
                alias_id, entity_id, alias, alias_language, source, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (alias, alias_language) DO NOTHING
            """,
            [alias_id, entity_id, alias.strip(), language, source, confidence]
        )
        return alias_id
    except Exception as e:
        logger.debug("write_entity_alias failed (%s → %s): %s", alias, entity_id, e)
        return None


# ── Prediction ────────────────────────────────────────────────────────────────

def write_prediction(
    prediction: dict,
    question_id: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    model_id: str = "xgb_conflict_v3",
    model_version: Optional[str] = None,
    schema_version: Optional[str] = None,
    top_features: Optional[dict] = None,
    flip_set_size: Optional[int] = None,
    as_of_time: Optional[datetime] = None,
) -> Optional[str]:
    """
    Write a prediction record including full market-blend audit trail.
    Returns prediction_id.

    The blend audit columns (p_model_raw, p_market_raw, etc.) are written when
    present in the prediction dict. They are used by blend_calibrator.py for
    Phase B learning.
    """
    from data_layer.db import get_db, table_exists
    import uuid

    if not table_exists("predictions"):
        return None

    prediction_id = str(uuid.uuid4())
    predicted_at = datetime.now(timezone.utc)
    if as_of_time is None:
        as_of_time = predicted_at

    # Market sources is a list — store as VARCHAR[] compatible with DuckDB
    market_sources = prediction.get("market_sources", []) or []

    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO predictions (
                prediction_id, question_id, snapshot_id,
                raw_prob, calibrated_prob, ci_lo, ci_hi, ci_method, answer,
                model_id, model_version, schema_version,
                market_override, market_prob_used,
                p_model_raw, p_market_raw, market_weight, market_sources,
                market_match_score, blend_strategy, blend_strategy_version,
                n_market_signals, market_gate_passed, market_gate_reason,
                top_features, flip_set_size,
                predicted_at, as_of_time
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?
            )
            """,
            [
                prediction_id, question_id, snapshot_id,
                prediction.get("raw_prob"), prediction.get("calibrated_prob"),
                prediction.get("ci_lo"), prediction.get("ci_hi"),
                prediction.get("ci_method"), prediction.get("answer"),
                model_id, model_version, schema_version,
                prediction.get("market_override", False),
                prediction.get("p_market_raw"),   # alias for legacy market_prob_used
                # Blend audit trail
                prediction.get("p_model_raw"),
                prediction.get("p_market_raw"),
                prediction.get("market_weight", 0.0),
                market_sources,
                prediction.get("market_match_score"),
                prediction.get("blend_strategy", "model_only"),
                prediction.get("blend_strategy_version", "logodds_v1"),
                prediction.get("n_market_signals", 0),
                prediction.get("market_gate_passed", False),
                prediction.get("market_gate_reason"),
                json.dumps(top_features) if top_features else None,
                flip_set_size,
                predicted_at, as_of_time,
            ]
        )
        return prediction_id
    except Exception as e:
        logger.warning("write_prediction failed: %s", e)
        return None

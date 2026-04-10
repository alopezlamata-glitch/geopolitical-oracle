"""
Pipeline hooks: connect the predict pipeline to the lakehouse.

Every call to `cmd_predict` passes through four write points:

  1. persist_raw_events()      → raw_documents (one row per RawEvent)
  2. persist_canonical_events() → canonical_documents + canonical_events
  3. persist_question()        → questions (with parse + OOD metadata)
  4. persist_prediction()      → feature_snapshots + predictions

All writes are non-fatal: if DuckDB is unavailable or a write fails,
the pipeline continues and logs a warning. The user always gets their
prediction; the lakehouse just doesn't have that run's record.

Design rule: every artifact written here carries as_of_time = the
moment collection ended, which is the strict data cutoff for that run.
Anything after that timestamp was not visible to the model.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from collector.base import RawEvent
    from normalizer.canonical import CanonicalEvent
    from question.parser import ParsedQuestion
    from question.ood import OODAssessment

logger = logging.getLogger(__name__)

_HOOKS_VERSION = "1.0"


# ── 1. Raw events → raw_documents ────────────────────────────────────────────

def persist_raw_events(
    raw_events: list["RawEvent"],
    as_of_time: datetime,
) -> list[str]:
    """
    Write each RawEvent to raw_documents.
    Returns list of raw_doc_ids written (empty if DB unavailable).
    Non-fatal.
    """
    from data_layer.writers import write_raw_document

    raw_doc_ids = []
    for ev in raw_events:
        body = json.dumps({
            "title": ev.title,
            "url": ev.url,
            "tone": ev.tone,
            "country": ev.country,
            "event_type": ev.event_type,
            "actors": ev.actors,
            "notes": ev.notes,
            "themes": ev.themes,
        }, ensure_ascii=False)

        raw_doc_id = write_raw_document(
            source=ev.source,
            body=body,
            published_at=ev.published_at,
            title=ev.title or "",
            source_url=ev.url or None,
            language="en",
            source_quality=_source_quality(ev.source),
            is_official_source=False,
            collector_version=_HOOKS_VERSION,
        )
        if raw_doc_id:
            raw_doc_ids.append(raw_doc_id)

    if raw_doc_ids:
        logger.debug("persist_raw_events: wrote %d raw_documents", len(raw_doc_ids))

    return raw_doc_ids


def _source_quality(source: str) -> float:
    return {
        "gdelt": 0.55,
        "rss": 0.60,
        "acled": 0.80,
        "metaculus": 0.90,
        "polymarket": 0.85,
    }.get(source, 0.50)


# ── 2. Canonical events → canonical_documents + canonical_events ──────────────

def persist_canonical_events(
    canonical_events: list["CanonicalEvent"],
    raw_doc_ids: list[str],
    as_of_time: datetime,
) -> list[str]:
    """
    Write canonical events to both canonical_documents and canonical_events.
    Returns list of event_ids written. Non-fatal.
    """
    from data_layer.db import get_db, table_exists
    from data_layer.writers import write_canonical_event
    from data_layer.lineage import record_lineage

    if not table_exists("canonical_documents"):
        return []

    db = get_db()
    event_ids = []

    for i, ev in enumerate(canonical_events):
        # canonical_documents: one row per CanonicalEvent (event is the doc)
        doc_id = f"doc_{ev.event_id}"
        source_domain = ev.source_domains[0] if ev.source_domains else ev.source

        try:
            db.execute(
                """
                INSERT INTO canonical_documents (
                    doc_id, raw_doc_id, source, published_at, ingested_at,
                    title, source_quality, is_official_source,
                    country_tags, entity_mentions, contradiction_score,
                    normalizer_version, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (doc_id) DO NOTHING
                """,
                [
                    doc_id,
                    raw_doc_ids[i] if i < len(raw_doc_ids) else None,
                    ev.source,
                    ev.occurred_at,
                    as_of_time,
                    ev.title if hasattr(ev, "title") else None,
                    ev.independent_sources / 5.0,   # rough quality proxy
                    False,
                    [ev.country] if hasattr(ev, "country") and ev.country else [],
                    [ev.source],
                    ev.contradiction_score,
                    "canonical_v1",
                    as_of_time,
                ]
            )
            record_lineage(
                output_type="canonical_document", output_id=doc_id,
                input_type="raw_document",
                input_ids=[raw_doc_ids[i]] if i < len(raw_doc_ids) else ["(no raw)"],
                processor_name="normalizer.canonical", processor_version="v1",
            )
        except Exception as e:
            logger.debug("canonical_document insert failed (non-fatal): %s", e)
            continue

        # canonical_events row
        event_id = write_canonical_event(
            doc_id=doc_id,
            event_type=ev.event_type,
            event_time=ev.occurred_at,
            actor_entity_ids=[],   # entity resolution not yet wired
            location_entity_id=None,
            intensity=float(ev.severity),
            polarity=ev.polarity,
            certainty=1.0 - ev.contradiction_score,
            is_official=False,
            fatalities=ev.fatalities,
            source_quality=ev.independent_sources / 5.0,
            independent_sources=ev.independent_sources,
            contradiction_score=ev.contradiction_score,
            extractor_version="v3",
            extraction_method="rule",
            confidence=1.0 - ev.contradiction_score,
            idx=i,
        )
        if event_id:
            event_ids.append(event_id)

    logger.debug("persist_canonical_events: wrote %d events", len(event_ids))
    return event_ids


# ── 3. Question → questions table ────────────────────────────────────────────

def persist_question(
    pq: "ParsedQuestion",
    ood: "OODAssessment",
    as_of_time: datetime,
) -> Optional[str]:
    """
    Write parsed question to the questions registry.
    Returns question_id or None. Non-fatal.
    """
    from data_layer.writers import write_question

    return write_question(
        raw_text=pq.raw,
        predicate=pq.predicate,
        event_family=pq.event_family,
        subject=pq.subject,
        deadline=pq.deadline,
        resolution_rule=pq.resolution_rule,
        as_of_time=as_of_time,
        subject_entity_id=None,   # entity resolution not yet wired
        jurisdiction=pq.jurisdiction,
        is_negated=pq.is_negated,
        ood_score=ood.ood_score,
        matched_model=ood.matched_model,
        parse_confidence=pq.parse_confidence,
        source="user",
        status="open",
    )


# ── 4. Feature snapshot + prediction ─────────────────────────────────────────

def persist_feature_snapshot(
    features: dict,
    as_of_time: datetime,
    question_id: Optional[str],
    pq: "ParsedQuestion",
    event_ids: list[str],
) -> Optional[str]:
    """
    Write the exact feature vector the model will see to feature_snapshots.
    Returns snapshot_id or None. Non-fatal.
    """
    from data_layer.writers import write_feature_snapshot

    return write_feature_snapshot(
        as_of_time=as_of_time,
        explicit_features=features,
        feature_schema_ver="v3",
        builder_version="v3",
        question_id=question_id,
        subject_entity_id=None,
        question_predicate=pq.predicate,
        question_deadline=pq.deadline,
        event_ids_used=event_ids,
        window_start=None,
        window_end=as_of_time,
        outcome=None,   # filled later via cmd_label
    )


def persist_prediction_record(
    prediction: dict,
    question_id: Optional[str],
    snapshot_id: Optional[str],
    as_of_time: datetime,
    attribution: dict,
) -> Optional[str]:
    """
    Write the final prediction to the predictions table.
    Returns prediction_id or None. Non-fatal.
    """
    from data_layer.writers import write_prediction

    # Extract top features for the log (top 5 by absolute SHAP)
    top_feats = None
    shap_vals = attribution.get("feature_shap_values", {})
    if shap_vals:
        sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_feats = {k: round(v, 4) for k, v in sorted_feats}

    flip_set = attribution.get("flip_set", [])

    return write_prediction(
        prediction=prediction,
        question_id=question_id,
        snapshot_id=snapshot_id,
        model_id=attribution.get("model_id", "xgb_conflict_v3"),
        model_version=prediction.get("model_version"),
        schema_version=prediction.get("schema_version", "1.1"),
        top_features=top_feats,
        flip_set_size=len(flip_set) if flip_set else 0,
        as_of_time=as_of_time,
    )


# ── 5. Label outcome → update feature_snapshot + question ────────────────────

def persist_outcome(
    question_raw_text: str,
    outcome: int,
    resolver_source: str = "user",
    resolution_notes: str = "",
) -> bool:
    """
    After resolution, update the matching question and feature_snapshot with outcome.
    Returns True if records were found and updated. Non-fatal.
    """
    from data_layer.db import get_db, table_exists
    from datetime import date

    if not table_exists("questions"):
        return False

    db = get_db()
    try:
        # Find matching question by raw_text similarity
        rows = db.execute(
            "SELECT question_id, deadline FROM questions WHERE raw_text = ? AND status = 'open'",
            [question_raw_text]
        ).fetchall()

        if not rows:
            logger.debug("persist_outcome: no open question found for %r", question_raw_text)
            return False

        question_id, deadline = rows[0]
        now = datetime.now(timezone.utc)

        # Write resolution record
        import uuid
        resolution_id = str(uuid.uuid4())
        db.execute(
            """
            INSERT INTO question_resolutions (
                resolution_id, question_id, outcome, resolved_at,
                deadline_was, resolver_source, resolution_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [resolution_id, question_id, outcome, now,
             deadline, resolver_source, resolution_notes]
        )

        # Update question status
        db.execute(
            "UPDATE questions SET status = 'resolved' WHERE question_id = ?",
            [question_id]
        )

        # Update feature_snapshot outcome
        db.execute(
            """
            UPDATE feature_snapshots
            SET outcome = ?, outcome_resolved_at = ?
            WHERE question_id = ?
            """,
            [outcome, now, question_id]
        )

        logger.info(
            "persist_outcome: question %s resolved → outcome=%d", question_id, outcome
        )
        return True

    except Exception as e:
        logger.debug("persist_outcome failed (non-fatal): %s", e)
        return False

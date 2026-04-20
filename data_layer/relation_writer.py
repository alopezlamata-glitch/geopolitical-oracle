"""
Temporal relation writer — persists extracted relations to entity_relations_temporal.

Design rules (matching schema.sql):
  - Never overwrite. If a relation changes, set valid_to on the old row
    and insert a new one.
  - Dedup: if an identical (subject, object, relation_type) with overlapping
    validity already exists at >= confidence, skip.
  - Retraction: if action='retract', close the most recent open row.

Entity upsert: creates entity rows in `entities` if not present.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _entity_id(canonical_name: str, entity_type: str = "country") -> str:
    raw = f"{entity_type}|{canonical_name.lower()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def _relation_id(subject_id: str, object_id: str, relation_type: str, valid_from: datetime) -> str:
    raw = f"{subject_id}|{object_id}|{relation_type}|{valid_from.isoformat()}"
    return "rel_" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def upsert_entity(
    canonical_name: str,
    entity_type: str,
    description: Optional[str] = None,
    wikidata_id: Optional[str] = None,
    country: Optional[str] = None,
) -> Optional[str]:
    """Ensure entity exists in the entities table. Returns entity_id."""
    try:
        from data_layer.db import get_db
        db = get_db()
        eid = _entity_id(canonical_name, entity_type)
        now = datetime.now(timezone.utc)

        db.execute("""
            INSERT INTO entities (entity_id, canonical_name, entity_type,
                                  description, wikidata_id, country,
                                  first_seen_at, last_updated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            ON CONFLICT (entity_id) DO UPDATE SET
                last_updated_at = excluded.last_updated_at,
                description     = COALESCE(excluded.description, entities.description)
        """, [eid, canonical_name, entity_type, description, wikidata_id, country, now, now])
        return eid
    except Exception as e:
        logger.debug("upsert_entity failed for %s: %s", canonical_name, e)
        return None


def write_relation(
    subject_name: str,
    subject_type: str,
    object_name: str,
    object_type: str,
    relation_type: str,
    valid_from: datetime,
    valid_to: Optional[datetime] = None,
    confidence: float = 1.0,
    source_event_ids: Optional[list[str]] = None,
    source_doc_ids: Optional[list[str]] = None,
    attributes: Optional[dict] = None,
    is_official: bool = False,
    extractor_version: str = "manual",
) -> Optional[str]:
    """
    Write a temporal relation. Returns relation_id or None on failure.
    Skips if an identical open relation already exists at >= confidence.
    """
    try:
        from data_layer.db import get_db
        db = get_db()

        subj_id = upsert_entity(subject_name, subject_type)
        obj_id  = upsert_entity(object_name,  object_type)
        if subj_id is None or obj_id is None:
            return None

        now = datetime.now(timezone.utc)

        # Check for existing open relation with same (subj, obj, type)
        existing = db.execute("""
            SELECT relation_id, confidence, valid_from
            FROM entity_relations_temporal
            WHERE subject_entity_id = ?
              AND object_entity_id  = ?
              AND relation_type     = ?
              AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY valid_from DESC
            LIMIT 1
        """, [subj_id, obj_id, relation_type, valid_from]).fetchone()

        if existing:
            ex_rid, ex_conf, ex_vf = existing
            # Same or better confidence already exists — skip
            if float(ex_conf or 0) >= confidence and valid_to is None:
                logger.debug(
                    "relation_writer: skipping duplicate %s/%s/%s (existing conf=%.2f)",
                    subject_name, relation_type, object_name, ex_conf,
                )
                return ex_rid

            # New info: close existing and write fresh
            if valid_to is None:  # asserting a continued or updated relation
                db.execute("""
                    UPDATE entity_relations_temporal
                    SET valid_to = ?, retracted_at = ?
                    WHERE relation_id = ?
                """, [valid_from, now, ex_rid])

        rid = _relation_id(subj_id, obj_id, relation_type, valid_from)

        db.execute("""
            INSERT INTO entity_relations_temporal (
                relation_id,
                subject_entity_id, object_entity_id,
                relation_type,
                valid_from, valid_to,
                source_doc_ids, source_event_ids,
                confidence, is_official,
                attributes,
                asserted_at, retracted_at,
                extractor_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (relation_id) DO NOTHING
        """, [
            rid,
            subj_id, obj_id,
            relation_type,
            valid_from, valid_to,
            source_doc_ids or [], source_event_ids or [],
            confidence, is_official,
            json.dumps(attributes or {}),
            now, None,
            extractor_version,
        ])

        logger.debug(
            "relation_writer: %s -[%s]-> %s (from=%s conf=%.2f)",
            subject_name, relation_type, object_name,
            valid_from.date(), confidence,
        )
        return rid

    except Exception as e:
        logger.warning("write_relation failed for %s/%s/%s: %s",
                       subject_name, relation_type, object_name, e)
        return None


def retract_relation(
    subject_name: str,
    object_name: str,
    relation_type: str,
    retracted_at: Optional[datetime] = None,
) -> bool:
    """Close the most recent open relation of this type between subject and object."""
    try:
        from data_layer.db import get_db
        db = get_db()
        now = retracted_at or datetime.now(timezone.utc)

        subj_id = _entity_id(subject_name)
        obj_id  = _entity_id(object_name)

        n = db.execute("""
            UPDATE entity_relations_temporal
            SET valid_to = ?, retracted_at = ?
            WHERE subject_entity_id = ?
              AND object_entity_id  = ?
              AND relation_type     = ?
              AND valid_to IS NULL
        """, [now, now, subj_id, obj_id, relation_type]).rowcount

        if n > 0:
            logger.info("relation_writer: retracted %s -[%s]-> %s",
                        subject_name, relation_type, object_name)
        return n > 0
    except Exception as e:
        logger.warning("retract_relation failed: %s", e)
        return False


def seed_from_file(path: str) -> int:
    """
    Load data/relations_seed.json and write all relations to the DB.
    Returns number of relations written.
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    n_ok = 0

    for rel in data.get("relations", []):
        vf_str = rel.get("valid_from")
        vt_str = rel.get("valid_to")

        try:
            vf = datetime.fromisoformat(vf_str).replace(tzinfo=timezone.utc) if vf_str else datetime.now(timezone.utc)
            vt = datetime.fromisoformat(vt_str).replace(tzinfo=timezone.utc) if vt_str else None
        except ValueError:
            logger.warning("seed: bad date in relation %s", rel)
            continue

        rid = write_relation(
            subject_name=rel["subject"],
            subject_type=rel.get("subject_type", "person"),
            object_name=rel["object"],
            object_type=rel.get("object_type", "country"),
            relation_type=rel["relation_type"],
            valid_from=vf,
            valid_to=vt,
            confidence=float(rel.get("confidence", 1.0)),
            attributes=rel.get("attributes"),
            is_official=True,
            extractor_version="seed_v1",
        )
        if rid:
            n_ok += 1

    logger.info("seed_from_file: %d/%d relations written", n_ok, len(data.get("relations", [])))
    return n_ok

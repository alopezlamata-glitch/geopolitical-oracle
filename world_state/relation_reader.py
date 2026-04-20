"""
Entity relation reader — loads temporal relations and derives features.

Key features produced:

  For a COUNTRY entity (e.g., Ukraine):
    leader_tenure_years       : years current head-of-state has been in power
    leader_tenure_norm        : tenure_years / 10, capped at 1.0
    leader_under_investigation: 1.0 if leader has open legal cases
    has_active_conflict       : 1.0 if country has open conflicts_with relation
    n_sanctions               : count of active sanctioned_by relations
    coalition_n_parties       : count of member_of relations for the governing coalition
    has_formal_alliance       : 1.0 if allied_with relation exists

  For a PERSON entity (e.g., Zelensky):
    tenure_years              : years in current role
    tenure_norm               : tenure_years / 10
    under_investigation       : 1.0 if under_investigation relation open
    in_coalition              : 1.0 if member_of coalition relation open

These features are non-event-based (static + temporal), complementing the
EMA-smoothed event counts in world_state.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 300.0  # 5 minutes
_cache: dict[str, tuple[float, list]] = {}


def _entity_id(name: str, entity_type: str = "country") -> str:
    raw = f"{entity_type}|{name.lower()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def get_active_relations(
    entity_name: str,
    entity_type: str = "country",
    as_of: Optional[datetime] = None,
    role: str = "subject",   # 'subject' | 'object' | 'both'
) -> list[dict]:
    """
    Return all currently valid relations where entity is subject or object.
    """
    import time
    cache_key = f"{entity_name}:{entity_type}:{role}"
    now_ts = time.monotonic()
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if now_ts - ts < _CACHE_TTL_S:
            return cached

    as_of = as_of or datetime.now(timezone.utc)

    try:
        from data_layer.db import get_db, table_exists
        if not table_exists("entity_relations_temporal"):
            return []

        db = get_db()
        eid = _entity_id(entity_name, entity_type)

        if role == "subject":
            where = "r.subject_entity_id = ?"
            params = [eid, as_of, as_of]
        elif role == "object":
            where = "r.object_entity_id = ?"
            params = [eid, as_of, as_of]
        else:
            where = "(r.subject_entity_id = ? OR r.object_entity_id = ?)"
            params = [eid, eid, as_of, as_of]

        rows = db.execute(f"""
            SELECT
                r.relation_id,
                r.relation_type,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.attributes,
                r.is_official,
                es.canonical_name AS subject_name,
                es.entity_type    AS subject_type,
                eo.canonical_name AS object_name,
                eo.entity_type    AS object_type
            FROM entity_relations_temporal r
            JOIN entities es ON r.subject_entity_id = es.entity_id
            JOIN entities eo ON r.object_entity_id  = eo.entity_id
            WHERE {where}
              AND r.valid_from <= ?
              AND (r.valid_to IS NULL OR r.valid_to > ?)
            ORDER BY r.valid_from ASC
        """, params).fetchall()

        cols = [
            "relation_id", "relation_type", "valid_from", "valid_to",
            "confidence", "attributes", "is_official",
            "subject_name", "subject_type", "object_name", "object_type",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            try:
                d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
            except Exception:
                d["attributes"] = {}
            result.append(d)

        _cache[cache_key] = (now_ts, result)
        return result

    except Exception as e:
        logger.debug("get_active_relations failed for %s: %s", entity_name, e)
        return []


def _tenure_years(valid_from: datetime, as_of: Optional[datetime] = None) -> float:
    ref = as_of or datetime.now(timezone.utc)
    if valid_from.tzinfo is None:
        valid_from = valid_from.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max(0.0, (ref - valid_from).total_seconds() / (365.25 * 86400))


def get_country_relation_features(
    country_name: str,
    as_of: Optional[datetime] = None,
) -> dict[str, float]:
    """
    Derive relation-based features for a country entity.
    Returns feature dict ready to merge into the prediction feature vector.
    """
    features: dict[str, float] = {
        "leader_tenure_years":        0.0,
        "leader_tenure_norm":         0.0,
        "leader_under_investigation": 0.0,
        "has_active_conflict":        0.0,
        "n_sanctions":                0.0,
        "has_formal_alliance":        0.0,
    }

    # Relations where this country is the object (someone governs it)
    obj_relations = get_active_relations(country_name, "country", as_of, role="object")
    # Relations where this country is the subject (country does something)
    subj_relations = get_active_relations(country_name, "country", as_of, role="subject")

    # ── Leader tenure ─────────────────────────────────────────────────────────
    leadership_rels = [
        r for r in obj_relations
        if r["relation_type"] == "holds_office"
        and r["subject_type"] == "person"
    ]
    if leadership_rels:
        # Most recent leader
        latest = max(leadership_rels, key=lambda r: r["valid_from"])
        tenure_yrs = _tenure_years(latest["valid_from"], as_of)
        features["leader_tenure_years"] = round(tenure_yrs, 2)
        features["leader_tenure_norm"]  = round(min(1.0, tenure_yrs / 10.0), 4)

        # Check if the current leader is under investigation
        leader_name = latest["subject_name"]
        leader_relations = get_active_relations(leader_name, "person", as_of, role="subject")
        is_investigated = any(
            r["relation_type"] == "under_investigation"
            for r in leader_relations
        )
        features["leader_under_investigation"] = 1.0 if is_investigated else 0.0

        logger.debug(
            "relation_reader[%s]: leader=%s tenure=%.1fy investigated=%s",
            country_name, leader_name, tenure_yrs, is_investigated,
        )

    # ── Active conflicts ───────────────────────────────────────────────────────
    active_conflicts = [
        r for r in subj_relations
        if r["relation_type"] in ("conflicts_with", "at_war_with")
    ] + [
        r for r in obj_relations
        if r["relation_type"] in ("conflicts_with", "at_war_with")
    ]
    features["has_active_conflict"] = 1.0 if active_conflicts else 0.0

    # ── Sanctions ─────────────────────────────────────────────────────────────
    sanctions = [
        r for r in obj_relations
        if r["relation_type"] == "sanctioned_by"
    ]
    features["n_sanctions"] = min(1.0, len(sanctions) / 5.0)  # normalize to [0,1]

    # ── Alliances ─────────────────────────────────────────────────────────────
    alliances = [
        r for r in subj_relations
        if r["relation_type"] in ("allied_with", "member_of")
    ] + [
        r for r in obj_relations
        if r["relation_type"] in ("allied_with", "member_of")
    ]
    features["has_formal_alliance"] = 1.0 if alliances else 0.0

    n_nonzero = sum(1 for v in features.values() if v != 0.0)
    if n_nonzero:
        logger.info(
            "relation_reader[%s]: %d relation features (%d nonzero)",
            country_name, len(features), n_nonzero,
        )

    return features


def get_person_relation_features(
    person_name: str,
    as_of: Optional[datetime] = None,
) -> dict[str, float]:
    """Derive relation-based features for a person entity."""
    features: dict[str, float] = {
        "tenure_years":        0.0,
        "tenure_norm":         0.0,
        "under_investigation": 0.0,
        "in_coalition":        0.0,
    }

    subj_relations = get_active_relations(person_name, "person", as_of, role="subject")

    office_rels = [r for r in subj_relations if r["relation_type"] == "holds_office"]
    if office_rels:
        latest = max(office_rels, key=lambda r: r["valid_from"])
        tenure_yrs = _tenure_years(latest["valid_from"], as_of)
        features["tenure_years"] = round(tenure_yrs, 2)
        features["tenure_norm"]  = round(min(1.0, tenure_yrs / 10.0), 4)

    investigated = [r for r in subj_relations if r["relation_type"] == "under_investigation"]
    features["under_investigation"] = 1.0 if investigated else 0.0

    coalition = [r for r in subj_relations if r["relation_type"] == "member_of"]
    features["in_coalition"] = 1.0 if coalition else 0.0

    return features


def get_leader_name(country_name: str, as_of: Optional[datetime] = None) -> Optional[str]:
    """Return the canonical name of the current head-of-state for a country."""
    rels = get_active_relations(country_name, "country", as_of, role="object")
    office_rels = [r for r in rels if r["relation_type"] == "holds_office" and r["subject_type"] == "person"]
    if not office_rels:
        return None
    latest = max(office_rels, key=lambda r: r["valid_from"])
    return latest["subject_name"]


def invalidate_cache(entity_name: Optional[str] = None) -> None:
    """Clear the in-process cache."""
    if entity_name is None:
        _cache.clear()
    else:
        keys = [k for k in _cache if k.startswith(entity_name)]
        for k in keys:
            _cache.pop(k, None)

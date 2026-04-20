"""
World state reader — load pre-computed entity state from DuckDB.

Called by features/builder.py to inject world state features
instead of recomputing them from scratch per question.

If no world state exists for an entity (first run, or entity not
in registry), returns None and the builder falls back to its
existing ad-hoc feature computation.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Cache for 5 minutes to avoid repeated DB reads within a single session
_cache: dict[tuple[str, str], Optional[dict]] = {}
_cache_ts: dict[tuple[str, str], float] = {}
_CACHE_TTL_S = 300.0


def _entity_id(canonical_name: str, entity_type: str = "country") -> str:
    raw = f"{entity_type}|{canonical_name.lower()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def get_world_state(
    entity_name: str,
    as_of_date: Optional[date] = None,
) -> Optional[dict]:
    """
    Load the most recent world_state row for `entity_name` on or before `as_of_date`.

    Returns a flat feature dict matching the world_state schema columns,
    or None if no state exists for this entity.
    """
    import time
    if as_of_date is None:
        as_of_date = date.today()

    cache_key = (entity_name.lower(), as_of_date.isoformat())
    now_ts = time.monotonic()

    if cache_key in _cache and (now_ts - _cache_ts.get(cache_key, 0)) < _CACHE_TTL_S:
        return _cache[cache_key]

    eid = _entity_id(entity_name)
    result = _load_from_db(eid, as_of_date)

    _cache[cache_key] = result
    _cache_ts[cache_key] = now_ts
    return result


def _load_from_db(entity_id: str, as_of_date: date) -> Optional[dict]:
    try:
        from data_layer.db import get_db, table_exists
        if not table_exists("world_state"):
            return None

        db = get_db()
        row = db.execute("""
            SELECT
                as_of_date,
                military_count_7d, military_count_30d,
                protest_count_7d, protest_count_30d,
                diplomatic_count_7d, ceasefire_count_7d, sanction_count_7d,
                military_intensity_7d, protest_intensity_7d, overall_intensity_7d,
                military_accel, protest_accel, overall_accel,
                avg_polarity_7d, avg_polarity_30d, tone_trend,
                source_diversity_7d, avg_independent_sources,
                has_military_7d, has_ceasefire_7d,
                escalation_index, ceasefire_ratio_7d, event_velocity_7d, military_share_7d,
                pol_resignation_signals, pol_approval_pressure,
                pol_coalition_stability, pol_electoral_proximity, pol_judicial_pressure,
                eco_rate_change_prob, eco_gdp_momentum, eco_debt_stress,
                eco_market_volatility, eco_policy_uncertainty,
                country_conflict_baserate, country_polity_norm, country_mil_spending_norm,
                wgi_pol_stability, wgi_gov_effectiveness, wgi_rule_of_law,
                fred_vix, fred_yield_spread,
                n_events_used, data_completeness, updater_version
            FROM world_state
            WHERE entity_id = ?
              AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
        """, [entity_id, as_of_date]).fetchone()

        if row is None:
            return None

        cols = [
            "as_of_date",
            "military_count_7d", "military_count_30d",
            "protest_count_7d", "protest_count_30d",
            "diplomatic_count_7d", "ceasefire_count_7d", "sanction_count_7d",
            "military_intensity_7d", "protest_intensity_7d", "overall_intensity_7d",
            "military_accel", "protest_accel", "overall_accel",
            "avg_polarity_7d", "avg_polarity_30d", "tone_trend",
            "source_diversity_7d", "avg_independent_sources",
            "has_military_7d", "has_ceasefire_7d",
            "escalation_index", "ceasefire_ratio_7d", "event_velocity_7d", "military_share_7d",
            "pol_resignation_signals", "pol_approval_pressure",
            "pol_coalition_stability", "pol_electoral_proximity", "pol_judicial_pressure",
            "eco_rate_change_prob", "eco_gdp_momentum", "eco_debt_stress",
            "eco_market_volatility", "eco_policy_uncertainty",
            "country_conflict_baserate", "country_polity_norm", "country_mil_spending_norm",
            "wgi_pol_stability", "wgi_gov_effectiveness", "wgi_rule_of_law",
            "fred_vix", "fred_yield_spread",
            "n_events_used", "data_completeness", "updater_version",
        ]
        state = dict(zip(cols, row))

        staleness_days = (date.today() - state["as_of_date"]).days
        state["_world_state_staleness_days"] = staleness_days
        state["_world_state_available"] = True

        logger.debug(
            "world_state: loaded for %s (as_of=%s, stale=%dd, events=%d)",
            entity_id, state["as_of_date"], staleness_days,
            state.get("n_events_used", 0),
        )
        return state

    except Exception as e:
        logger.debug("world_state reader failed for %s: %s", entity_id, e)
        return None


def get_staleness_days(entity_name: str) -> Optional[int]:
    """Return how many days old the world state is, or None if not available."""
    state = get_world_state(entity_name)
    if state is None:
        return None
    return state.get("_world_state_staleness_days")


def invalidate_cache(entity_name: Optional[str] = None) -> None:
    """Clear the in-process cache (called after an update run)."""
    if entity_name is None:
        _cache.clear()
        _cache_ts.clear()
    else:
        keys_to_remove = [k for k in _cache if k[0] == entity_name.lower()]
        for k in keys_to_remove:
            _cache.pop(k, None)
            _cache_ts.pop(k, None)

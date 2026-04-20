"""
Daily world state updater — Phase 1 of the world model.

For each entity in data/entity_registry.json:
  1. Collect raw events from GDELT + ACLED (async, per-entity queries)
  2. Load structural data: World Bank, V-Dem, FRED (US entities)
  3. Compute state feature vector from events
  4. Apply EMA smoothing against prior world_state row
  5. Apply cross-entity causal propagation (causal_links.json)
  6. Write to world_state + world_state_history in DuckDB

Usage:
  python scripts/update_world_state.py
  python scripts/update_world_state.py --entity Ukraine Russia
  python scripts/update_world_state.py --dry-run
  python scripts/update_world_state.py --no-causal
  python scripts/update_world_state.py --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp

# Add project root so local imports work when called from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from features.event_aggregations import (
    partition_windows,
    collect_unique_sources,
    mean_attr,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("world_state")
for _noisy in ("aiohttp", "urllib3", "feedparser"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_ROOT = Path(__file__).parent.parent
_REGISTRY_PATH = _ROOT / "data" / "entity_registry.json"
_CAUSAL_PATH   = _ROOT / "data" / "causal_links.json"

UPDATER_VERSION = "v1"

# Event type → feature category mapping
_MILITARY_TYPES  = {"military_action", "armed_attack", "airstrike", "coup_attempt",
                    "nuclear_test", "military_buildup", "armed_clashes"}
_PROTEST_TYPES   = {"protest", "riot", "civil_unrest", "strike", "demonstration"}
_CEASEFIRE_TYPES = {"ceasefire_signal", "peace_deal", "withdrawal_signal"}
_DIPLOMATIC_TYPES = {"diplomatic_meeting", "sanction_lifted", "negotiation",
                     "treaty_signed", "diplomatic_protest"}
_SANCTION_TYPES  = {"sanction_imposed", "embargo", "asset_freeze"}


# ─────────────────────────────────────────────────────────────────────────────
# Entity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entity_id(canonical_name: str, entity_type: str = "country") -> str:
    raw = f"{entity_type}|{canonical_name.lower()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def _state_id(entity_id: str, as_of_date: date) -> str:
    raw = f"{entity_id}|{as_of_date.isoformat()}"
    return "ws_" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _history_id(entity_id: str, computed_at: datetime) -> str:
    raw = f"{entity_id}|{computed_at.isoformat()}"
    return "wsh_" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def load_registry() -> list[dict]:
    data = json.loads(_REGISTRY_PATH.read_text())
    return data["entities"]


def load_smoothing_config() -> dict:
    data = json.loads(_REGISTRY_PATH.read_text())
    return data.get("smoothing", {
        "dynamic_features":    0.35,
        "binary_features":     0.50,
        "structural_features": 0.05,
        "domain_features":     0.30,
    })


def load_causal_links() -> list[dict]:
    if not _CAUSAL_PATH.exists():
        return []
    data = json.loads(_CAUSAL_PATH.read_text())
    return data.get("links", [])


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_entity(
    entity: dict,
    session: aiohttp.ClientSession,
) -> tuple[list, dict]:
    """
    Collect raw events + structural data for one entity.
    Returns (raw_events, structural_dict).
    """
    name    = entity["canonical_name"]
    iso2    = entity.get("iso2", "")
    collectors = entity.get("collectors", [])

    tasks: list = []
    task_names: list[str] = []

    if "gdelt" in collectors:
        from collector.gdelt import collect_gdelt
        tasks.append(collect_gdelt(session, name))
        task_names.append("gdelt")

    if "acled" in collectors:
        from collector.acled import collect_acled
        tasks.append(collect_acled(session, name, country=name))
        task_names.append("acled")

    if "worldbank" in collectors:
        from collector.worldbank import collect_worldbank
        tasks.append(collect_worldbank(session, name))
        task_names.append("worldbank")

    if "fred" in collectors:
        from collector.fred import collect_fred
        tasks.append(collect_fred(session))
        task_names.append("fred")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    raw_events: list = []
    structural: dict = {}

    for task_name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.debug("%s collector failed for %s: %s", task_name, name, result)
            continue
        if task_name in ("gdelt", "acled"):
            if isinstance(result, list):
                raw_events.extend(result)
        elif task_name in ("worldbank", "fred"):
            if isinstance(result, dict):
                structural.update(result)

    # V-Dem (sync, from snapshot)
    try:
        from collector.vdem import get_vdem_features
        vdem = get_vdem_features(name)
        if vdem:
            structural.update(vdem)
    except Exception:
        pass

    return raw_events, structural


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _event_category(event_type: str) -> str:
    et = event_type.lower()
    for t in _MILITARY_TYPES:
        if t in et:
            return "military"
    for t in _PROTEST_TYPES:
        if t in et:
            return "protest"
    for t in _CEASEFIRE_TYPES:
        if t in et:
            return "ceasefire"
    for t in _DIPLOMATIC_TYPES:
        if t in et:
            return "diplomatic"
    for t in _SANCTION_TYPES:
        if t in et:
            return "sanction"
    return "other"


def compute_features(
    raw_events: list,
    structural: dict,
    entity: dict,
) -> dict:
    """
    Aggregate raw events + structural data into the world_state feature vector.
    All features are normalized to [0, 1] or unbounded floats where noted.
    """
    now = datetime.now(timezone.utc)

    # ── Partition events by recency — shared primitive ────────────────────────
    # Uses occurred_at with published_at fallback; handles naive timestamps.
    _windows = partition_windows(
        raw_events, now,
        windows_days=(7, 30),
        get_ts=lambda e: getattr(e, "occurred_at", None) or getattr(e, "published_at", None),
    )
    events_7d  = _windows[7]
    events_30d = _windows[30]

    # ── Count by category ─────────────────────────────────────────────────────
    def _counts(evts: list) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for ev in evts:
            cat = _event_category(getattr(ev, "event_type", ""))
            c[cat] += 1
        return c

    c7  = _counts(events_7d)
    c30 = _counts(events_30d)

    mil_7d  = float(c7.get("military",  0))
    mil_30d = float(c30.get("military", 0))
    pro_7d  = float(c7.get("protest",   0))
    pro_30d = float(c30.get("protest",  0))
    cea_7d  = float(c7.get("ceasefire", 0))
    dip_7d  = float(c7.get("diplomatic", 0))
    san_7d  = float(c7.get("sanction",  0))
    total_7d  = float(max(1, len(events_7d)))
    total_30d = float(max(1, len(events_30d)))

    # ── Intensity (mean of raw tone / 10 to get [-1, 1] direction) ───────────
    def _mean_intensity(evts: list, cat: Optional[str] = None) -> float:
        vals = []
        for ev in evts:
            if cat and _event_category(getattr(ev, "event_type", "")) != cat:
                continue
            # CanonicalEvent has .severity [0,1]; RawEvent has .tone
            severity = getattr(ev, "severity", None)
            if severity is not None:
                vals.append(max(0.0, min(1.0, float(severity))))
            else:
                tone = getattr(ev, "tone", None)
                if tone is not None:
                    vals.append(max(0.0, min(1.0, abs(float(tone)) / 10.0)))
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _mean_polarity(evts: list) -> float:
        vals = []
        for ev in evts:
            # CanonicalEvent has .polarity [-1, 1]; RawEvent has .tone
            pol = getattr(ev, "polarity", None)
            if pol is not None:
                vals.append(max(-1.0, min(1.0, float(pol))))
            else:
                tone = getattr(ev, "tone", None)
                if tone is not None:
                    vals.append(max(-1.0, min(1.0, float(tone) / 10.0)))
        return float(sum(vals) / len(vals)) if vals else 0.0

    # Acceleration: current week vs average weekly rate in last month
    weekly_rate_30d_mil = (mil_30d / 4.0) if mil_30d > 0 else 1.0
    weekly_rate_30d_pro = (pro_30d / 4.0) if pro_30d > 0 else 1.0
    overall_30d_rate    = (total_30d / 4.0) if total_30d > 0 else 1.0

    mil_accel     = mil_7d  / max(0.5, weekly_rate_30d_mil)
    pro_accel     = pro_7d  / max(0.5, weekly_rate_30d_pro)
    overall_accel = total_7d / max(0.5, overall_30d_rate)

    # Source diversity — shared primitives (with .source fallback for raw events)
    sources_7d = collect_unique_sources(events_7d)
    avg_ind    = mean_attr(events_7d, "independent_sources", default=1.0, cap=5.0)

    # ── Escalation index ─────────────────────────────────────────────────────
    # > 0 means escalating, < 0 means de-escalating
    denom = max(1.0, mil_7d + cea_7d + dip_7d)
    escalation_index = (mil_7d - cea_7d) / denom

    # ── Political features ────────────────────────────────────────────────────
    # These come from structural (V-Dem, WB WGI) and domain events
    pol_resign   = structural.get("pol_resignation_signals",  0.0)
    pol_pressure = structural.get("pol_approval_pressure",    0.0)
    pol_coalit   = structural.get("pol_coalition_stability",  0.5)
    pol_elect    = structural.get("pol_electoral_proximity",  0.0)
    pol_judicial = structural.get("pol_judicial_pressure",    0.0)

    # Political events can sharpen these signals
    pol_events_7d = [ev for ev in events_7d
                     if "political" in getattr(ev, "event_type", "").lower()
                     or "resign" in getattr(ev, "event_type", "").lower()
                     or "election" in getattr(ev, "event_type", "").lower()]
    if pol_events_7d:
        pol_resign   = min(1.0, pol_resign + 0.1 * len(pol_events_7d))
        pol_pressure = min(1.0, pol_pressure + 0.05 * len(pol_events_7d))

    # ── Economic features ─────────────────────────────────────────────────────
    eco_rate      = structural.get("eco_rate_change_prob",   0.0)
    eco_gdp       = structural.get("eco_gdp_momentum",       0.0)
    eco_debt      = structural.get("eco_debt_stress",        0.0)
    eco_vol       = structural.get("eco_market_volatility",  0.0)
    eco_pol_unc   = structural.get("eco_policy_uncertainty", 0.0)

    # VIX → global volatility signal
    vix = structural.get("fred_vix", structural.get("vix", None))
    if vix is not None:
        eco_vol = max(eco_vol, min(1.0, float(vix) / 50.0))

    yield_spread = structural.get("fred_yield_spread", structural.get("yield_spread_10y2y", None))
    if yield_spread is not None:
        # Inverted yield curve (spread < 0) → recession signal
        eco_gdp = min(1.0, max(0.0, (float(yield_spread) + 0.5) / 2.0))

    # ── Structural (country baseline) ─────────────────────────────────────────
    from features.country_data import get_country_features
    name = entity["canonical_name"]
    cd   = get_country_features(name)

    baserate          = float(cd.get("country_conflict_baserate", 0.15))
    polity_norm       = float(cd.get("country_polity_norm",       0.0))
    mil_spending_norm = float(cd.get("country_mil_spending_norm", 0.15))

    wgi_pol  = structural.get("wgi_pol_stability", None)
    wgi_gov  = structural.get("wgi_gov_effectiveness", None)
    wgi_rul  = structural.get("wgi_rule_of_law", None)
    fred_vix = structural.get("fred_vix", structural.get("vix", None))
    fred_ysp = structural.get("fred_yield_spread", structural.get("yield_spread_10y2y", None))

    return {
        # Conflict / security
        "military_count_7d":       mil_7d,
        "military_count_30d":      mil_30d,
        "protest_count_7d":        pro_7d,
        "protest_count_30d":       pro_30d,
        "diplomatic_count_7d":     dip_7d,
        "ceasefire_count_7d":      cea_7d,
        "sanction_count_7d":       san_7d,
        "military_intensity_7d":   _mean_intensity(events_7d, "military"),
        "protest_intensity_7d":    _mean_intensity(events_7d, "protest"),
        "overall_intensity_7d":    _mean_intensity(events_7d),
        "military_accel":          round(min(10.0, mil_accel),  4),
        "protest_accel":           round(min(10.0, pro_accel),  4),
        "overall_accel":           round(min(10.0, overall_accel), 4),
        "avg_polarity_7d":         _mean_polarity(events_7d),
        "avg_polarity_30d":        _mean_polarity(events_30d),
        "tone_trend":              _mean_polarity(events_7d) - _mean_polarity(events_30d),
        "source_diversity_7d":     min(1.0, len(sources_7d) / 5.0),
        "avg_independent_sources": min(5.0, avg_ind),
        "has_military_7d":         1.0 if mil_7d > 0 else 0.0,
        "has_ceasefire_7d":        1.0 if cea_7d > 0 else 0.0,
        "escalation_index":        round(max(-1.0, min(1.0, escalation_index)), 4),
        "ceasefire_ratio_7d":      round(cea_7d / max(1.0, mil_7d), 4),
        "event_velocity_7d":       round(len(events_7d) / 7.0, 4),
        "military_share_7d":       round(mil_7d / total_7d, 4),
        # Political
        "pol_resignation_signals": round(min(1.0, pol_resign),   4),
        "pol_approval_pressure":   round(min(1.0, pol_pressure), 4),
        "pol_coalition_stability": round(min(1.0, max(0.0, pol_coalit)), 4),
        "pol_electoral_proximity": round(min(1.0, pol_elect),    4),
        "pol_judicial_pressure":   round(min(1.0, pol_judicial), 4),
        # Economic
        "eco_rate_change_prob":    round(min(1.0, max(0.0, eco_rate)),   4),
        "eco_gdp_momentum":        round(min(1.0, max(0.0, eco_gdp)),    4),
        "eco_debt_stress":         round(min(1.0, max(0.0, eco_debt)),   4),
        "eco_market_volatility":   round(min(1.0, max(0.0, eco_vol)),    4),
        "eco_policy_uncertainty":  round(min(1.0, max(0.0, eco_pol_unc)), 4),
        # Structural
        "country_conflict_baserate": round(baserate,          4),
        "country_polity_norm":       round(polity_norm,       4),
        "country_mil_spending_norm": round(mil_spending_norm, 4),
        "wgi_pol_stability":         round(float(wgi_pol),  4) if wgi_pol  is not None else None,
        "wgi_gov_effectiveness":     round(float(wgi_gov),  4) if wgi_gov  is not None else None,
        "wgi_rule_of_law":           round(float(wgi_rul),  4) if wgi_rul  is not None else None,
        "fred_vix":                  round(float(fred_vix), 4) if fred_vix is not None else None,
        "fred_yield_spread":         round(float(fred_ysp), 4) if fred_ysp is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EMA smoothing
# ─────────────────────────────────────────────────────────────────────────────

# Features that change slowly → use structural_features alpha
_STRUCTURAL_FEATURES = {
    "country_conflict_baserate", "country_polity_norm", "country_mil_spending_norm",
    "wgi_pol_stability", "wgi_gov_effectiveness", "wgi_rule_of_law",
}
# Binary flags → binary_features alpha
_BINARY_FEATURES = {
    "has_military_7d", "has_ceasefire_7d",
}
# Event counts / velocities → dynamic_features alpha
_DYNAMIC_FEATURES = {
    "military_count_7d", "military_count_30d", "protest_count_7d", "protest_count_30d",
    "diplomatic_count_7d", "ceasefire_count_7d", "sanction_count_7d",
    "military_intensity_7d", "protest_intensity_7d", "overall_intensity_7d",
    "military_accel", "protest_accel", "overall_accel",
    "avg_polarity_7d", "avg_polarity_30d", "tone_trend",
    "source_diversity_7d", "avg_independent_sources", "escalation_index",
    "ceasefire_ratio_7d", "event_velocity_7d", "military_share_7d",
}


def apply_ema(
    new_features: dict,
    prior_state: Optional[dict],
    smoothing: dict,
) -> dict:
    """
    Exponential moving average: blends new observation with prior state.
    Features with no prior (first run) are returned as-is.
    """
    if prior_state is None:
        return new_features

    result = dict(new_features)
    for key, new_val in new_features.items():
        if new_val is None:
            continue
        prior_val = prior_state.get(key)
        if prior_val is None:
            continue

        if key in _STRUCTURAL_FEATURES:
            alpha = smoothing.get("structural_features", 0.05)
        elif key in _BINARY_FEATURES:
            alpha = smoothing.get("binary_features", 0.50)
        elif key in _DYNAMIC_FEATURES:
            alpha = smoothing.get("dynamic_features", 0.35)
        else:
            alpha = smoothing.get("domain_features", 0.30)

        result[key] = round(alpha * float(new_val) + (1.0 - alpha) * float(prior_val), 6)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Causal propagation
# ─────────────────────────────────────────────────────────────────────────────

def apply_causal_inflow(
    target_entity: str,
    target_features: dict,
    all_new_states: dict[str, dict],  # name → feature dict
    causal_links: list[dict],
    as_of_date: date,
) -> tuple[dict, dict]:
    """
    Apply causal inflow from source entities to the target entity.

    delta = weight * source_value * exp(-elapsed_days / decay_days)

    Returns (updated_features, inflow_summary).
    """
    import math

    result = dict(target_features)
    inflow_summary: dict[str, float] = {}

    for link in causal_links:
        if link.get("target_entity") != target_entity:
            continue

        source_name    = link["source_entity"]
        source_feature = link["source_feature"]
        target_feature = link["target_feature"]
        weight         = float(link.get("weight", 0.0))
        decay_days     = float(link.get("decay_days", 14))

        source_state = all_new_states.get(source_name, {})
        source_val   = source_state.get(source_feature, 0.0)
        if source_val is None or source_val == 0.0:
            continue

        # Decay is 1.0 at t=0 (same day); this is daily so elapsed=0 → decay=1
        delta = weight * float(source_val) * math.exp(-1.0 / decay_days)

        target_key = target_feature
        current    = float(result.get(target_key, 0.0) or 0.0)
        updated    = max(-2.0, min(2.0, current + delta))
        result[target_key] = round(updated, 6)
        inflow_summary[f"{source_name}.{source_feature}→{target_feature}"] = round(delta, 4)

    return result, inflow_summary


# ─────────────────────────────────────────────────────────────────────────────
# DuckDB persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_prior_state(entity_id: str) -> Optional[dict]:
    """Load the most recent world_state row for this entity."""
    try:
        from data_layer.db import get_db, table_exists
        if not table_exists("world_state"):
            return None
        db = get_db()
        row = db.execute("""
            SELECT
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
                fred_vix, fred_yield_spread
            FROM world_state
            WHERE entity_id = ?
              AND valid_to IS NULL
            ORDER BY as_of_date DESC
            LIMIT 1
        """, [entity_id]).fetchone()

        if row is None:
            return None

        cols = [
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
        ]
        return dict(zip(cols, row))
    except Exception as e:
        logger.debug("load_prior_state failed for %s: %s", entity_id, e)
        return None


def write_world_state(
    entity: dict,
    features: dict,
    inflow_summary: dict,
    n_events: int,
    sources_used: list[str],
    as_of_date: date,
    dry_run: bool = False,
) -> bool:
    """Write world_state + world_state_history rows. Returns True on success."""
    eid         = _entity_id(entity["canonical_name"])
    sid         = _state_id(eid, as_of_date)
    computed_at = datetime.now(timezone.utc)
    valid_from  = datetime.combine(as_of_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    data_completeness = 1.0
    null_count = sum(1 for v in features.values() if v is None)
    if null_count:
        data_completeness = round(1.0 - (null_count / max(1, len(features))), 3)

    if dry_run:
        logger.info(
            "[DRY-RUN] %s  events=%d  mil_7d=%.0f  esc=%.3f  completeness=%.0f%%",
            entity["canonical_name"], n_events,
            features.get("military_count_7d", 0),
            features.get("escalation_index", 0),
            data_completeness * 100,
        )
        return True

    try:
        from data_layer.db import get_db

        db = get_db()

        # Expire the previous row for this entity
        db.execute("""
            UPDATE world_state
            SET valid_to = ?
            WHERE entity_id = ? AND valid_to IS NULL AND as_of_date < ?
        """, [computed_at, eid, as_of_date])

        def f(key, default=0.0):
            v = features.get(key, default)
            return float(v) if v is not None else default

        db.execute("""
            INSERT INTO world_state (
                state_id, entity_id, as_of_date,
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
                smoothing_alpha, causal_inflow,
                n_events_used, data_completeness, sources_used,
                updater_version, computed_at, valid_from, valid_to
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?
            )
            ON CONFLICT (entity_id, as_of_date) DO UPDATE SET
                military_count_7d     = excluded.military_count_7d,
                escalation_index      = excluded.escalation_index,
                n_events_used         = excluded.n_events_used,
                causal_inflow         = excluded.causal_inflow,
                computed_at           = excluded.computed_at,
                valid_from            = excluded.valid_from,
                valid_to              = excluded.valid_to,
                updater_version       = excluded.updater_version
        """, [
            sid, eid, as_of_date,
            f("military_count_7d"),    f("military_count_30d"),
            f("protest_count_7d"),     f("protest_count_30d"),
            f("diplomatic_count_7d"),  f("ceasefire_count_7d"),  f("sanction_count_7d"),
            f("military_intensity_7d"), f("protest_intensity_7d"), f("overall_intensity_7d"),
            f("military_accel", 1.0),   f("protest_accel", 1.0),   f("overall_accel", 1.0),
            f("avg_polarity_7d"),  f("avg_polarity_30d"),  f("tone_trend"),
            f("source_diversity_7d"),  f("avg_independent_sources", 1.0),
            f("has_military_7d"),   f("has_ceasefire_7d"),
            f("escalation_index"),  f("ceasefire_ratio_7d"),
            f("event_velocity_7d"), f("military_share_7d"),
            f("pol_resignation_signals"), f("pol_approval_pressure"),
            f("pol_coalition_stability", 0.5), f("pol_electoral_proximity"),
            f("pol_judicial_pressure"),
            f("eco_rate_change_prob"), f("eco_gdp_momentum"),
            f("eco_debt_stress"),      f("eco_market_volatility"),
            f("eco_policy_uncertainty"),
            f("country_conflict_baserate", 0.15),
            f("country_polity_norm"),
            f("country_mil_spending_norm", 0.15),
            features.get("wgi_pol_stability"),
            features.get("wgi_gov_effectiveness"),
            features.get("wgi_rule_of_law"),
            features.get("fred_vix"),
            features.get("fred_yield_spread"),
            0.3, json.dumps(inflow_summary),
            n_events, data_completeness, sources_used,
            UPDATER_VERSION, computed_at, valid_from, None,
        ])

        # Append to history (always insert, never update)
        hid = _history_id(eid, computed_at)

        # Compute delta vs prior (for time-series diagnostics)
        prior = load_prior_state(eid)
        delta: dict = {}
        if prior:
            for k, v in features.items():
                pv = prior.get(k)
                if v is not None and pv is not None:
                    d = round(float(v) - float(pv), 6)
                    if abs(d) > 1e-6:
                        delta[k] = d

        db.execute("""
            INSERT INTO world_state_history (
                history_id, entity_id, as_of_date, computed_at,
                features, delta_from_prior, prior_state_date,
                causal_inflow_summary, sources_used,
                n_events_used, data_completeness, updater_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            hid, eid, as_of_date, computed_at,
            json.dumps(features), json.dumps(delta), None,
            json.dumps(inflow_summary), sources_used,
            n_events, data_completeness, UPDATER_VERSION,
        ])

        logger.info(
            "world_state: %s  events=%d  mil_7d=%.0f  esc=%.3f  completeness=%.0f%%",
            entity["canonical_name"], n_events,
            features.get("military_count_7d", 0),
            features.get("escalation_index", 0),
            data_completeness * 100,
        )
        return True

    except Exception as e:
        logger.error("write_world_state failed for %s: %s", entity["canonical_name"], e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def _update_all(
    entities: list[dict],
    dry_run: bool,
    apply_causal: bool,
    concurrency: int = 4,
) -> dict[str, dict]:
    """
    Pass 1: collect + compute features for all entities concurrently.
    Returns entity_name → smoothed_feature_dict.
    """
    sem = asyncio.Semaphore(concurrency)
    today = date.today()
    smoothing = load_smoothing_config()

    # Shared session for all requests
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:

        async def _process_one(entity: dict) -> tuple[str, dict, int, list[str]]:
            async with sem:
                name = entity["canonical_name"]
                t0 = time.monotonic()
                try:
                    raw_events, structural = await _collect_entity(entity, session)
                except Exception as e:
                    logger.warning("collect failed for %s: %s", name, e)
                    raw_events, structural = [], {}

                # Normalize raw events → typed CanonicalEvents before feature computation
                canonical_events = []
                if raw_events:
                    try:
                        from normalizer.canonical import normalize_all
                        from normalizer.deduplicator import deduplicate
                        canonical_events = deduplicate(normalize_all(raw_events))
                    except Exception as e:
                        logger.debug("normalization failed for %s: %s", name, e)

                features = compute_features(canonical_events, structural, entity)

                # Relation extraction from headlines (Phase 4, non-fatal)
                if canonical_events:
                    try:
                        from data_layer.relation_extractor import extract_and_write
                        headlines = [
                            getattr(ev, "raw_title", "") or getattr(ev, "title", "")
                            for ev in canonical_events
                            if getattr(ev, "raw_title", "") or getattr(ev, "title", "")
                        ]
                        n_rel = extract_and_write(headlines, name, use_llm=False)
                        if n_rel:
                            logger.debug("relation_extractor: %d new relations for %s", n_rel, name)
                    except Exception as exc:
                        logger.debug("relation extraction skipped for %s: %s", name, exc)

                # Identify sources actually used
                sources = list({getattr(ev, "source", "?") for ev in raw_events})
                if structural:
                    if any(k.startswith("wgi_") or k.startswith("eco_") for k in structural):
                        if "worldbank" not in sources:
                            sources.append("worldbank")
                    if any(k.startswith("fred_") or k == "vix" for k in structural):
                        if "fred" not in sources:
                            sources.append("fred")
                    if any(k.startswith("pol_") for k in structural):
                        if "vdem" not in sources:
                            sources.append("vdem")

                # EMA with prior state
                eid = _entity_id(name)
                prior = load_prior_state(eid)
                smoothed = apply_ema(features, prior, smoothing)

                elapsed = round((time.monotonic() - t0) * 1000)
                logger.debug("%s: %d events, %dms", name, len(raw_events), elapsed)
                return name, smoothed, len(raw_events), sources

        tasks = [asyncio.create_task(_process_one(e)) for e in entities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_states: dict[str, dict] = {}
    n_events_map: dict[str, int] = {}
    sources_map: dict[str, list] = {}

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("entity[%d] failed: %s", i, result)
            continue
        name, feats, n_evts, srcs = result
        all_states[name] = feats
        n_events_map[name] = n_evts
        sources_map[name] = srcs

    # Pass 2: causal propagation (uses all freshly computed states)
    if apply_causal and not dry_run:
        causal_links = load_causal_links()
        if causal_links:
            logger.info("applying causal propagation (%d links)...", len(causal_links))
            for entity in entities:
                name = entity["canonical_name"]
                if name not in all_states:
                    continue
                updated, inflow = apply_causal_inflow(
                    name, all_states[name], all_states, causal_links, today
                )
                all_states[name] = updated
                if inflow:
                    logger.debug("causal inflow for %s: %s", name, inflow)
        else:
            logger.debug("no causal links found, skipping propagation")

    # Pass 3: write to DuckDB
    n_ok = 0
    for entity in entities:
        name = entity["canonical_name"]
        if name not in all_states:
            continue

        # Recompute inflow_summary for logging (already applied in state)
        inflow_log: dict = {}
        if apply_causal:
            causal_links = load_causal_links()
            _, inflow_log = apply_causal_inflow(
                name, all_states[name], all_states, causal_links, today
            )

        ok = write_world_state(
            entity=entity,
            features=all_states[name],
            inflow_summary=inflow_log,
            n_events=n_events_map.get(name, 0),
            sources_used=sources_map.get(name, []),
            as_of_date=today,
            dry_run=dry_run,
        )
        if ok:
            n_ok += 1

    logger.info("world state update complete: %d/%d entities written", n_ok, len(entities))
    return all_states


def run(
    entity_names: Optional[list[str]] = None,
    dry_run: bool = False,
    apply_causal: bool = True,
    priority_only: bool = False,
    concurrency: int = 4,
) -> None:
    from data_layer.db import init_schema
    init_schema()

    registry = load_registry()

    # Filter by name if requested
    if entity_names:
        registry = [e for e in registry if e["canonical_name"] in entity_names]
        if not registry:
            logger.error("No matching entities found for: %s", entity_names)
            return

    if priority_only:
        registry = [e for e in registry if e.get("priority") == "high"]

    logger.info(
        "updating world state for %d entities (dry_run=%s, causal=%s)",
        len(registry), dry_run, apply_causal,
    )

    asyncio.run(_update_all(
        entities=registry,
        dry_run=dry_run,
        apply_causal=apply_causal,
        concurrency=concurrency,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily world state updater")
    parser.add_argument(
        "--entity", "-e", nargs="+", metavar="NAME",
        help="Only update these entities (exact canonical name)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute features but do not write to DuckDB",
    )
    parser.add_argument(
        "--no-causal", action="store_true",
        help="Skip cross-entity causal propagation",
    )
    parser.add_argument(
        "--priority-only", action="store_true",
        help="Only update high-priority entities (faster)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent entity fetches (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("world_state").setLevel(logging.DEBUG)
        logging.getLogger("collector").setLevel(logging.DEBUG)

    run(
        entity_names=args.entity,
        dry_run=args.dry_run,
        apply_causal=not args.no_causal,
        priority_only=args.priority_only,
        concurrency=args.concurrency,
    )

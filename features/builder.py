from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from normalizer.canonical import CanonicalEvent
from features.country_data import get_country_features

logger = logging.getLogger(__name__)

_DECAY_HALFLIFE_DAYS = 7.0

# ── Feature registry ──────────────────────────────────────────────────────────
# Changes from v1 (30 features) → v2 (32 features):
#
#   REMOVED (3): acled_military_90d, acled_fatalities_90d, acled_conflict_active
#     Reason: always 0.0 in training data (ICEWS source has no ACLED events);
#     model never learned the relationship → dead weight at inference too.
#     Re-add when historical ACLED data is integrated into training.
#
#   CHANGED (3): metaculus_p, polymarket_p, market_available
#     sentinel changed from -1.0 → 0.0 when unavailable.
#     Two new binary availability flags added (metaculus_available, polymarket_available)
#     so the model knows to ignore the p value when it's a default 0.5.
#
#   ADDED (3): country_conflict_baserate, country_polity_norm, country_mil_spending_norm
#     Static structural prior per country (see features/country_data.py).
#     These anchor the prediction before reading any news:
#       conflict_baserate — fraction of years with active conflict (UCDP 2000-2023)
#       polity_norm       — Polity5 score normalized to [-1, +1]
#       mil_spending_norm — military % of GDP / 10  (SIPRI 2022)

_FEATURE_NAMES = [
    # ── Event-derived features (24) ──────────────────────────────────────────
    "military_count_7d", "military_count_30d",
    "protest_count_7d", "protest_count_30d",
    "diplomatic_count_7d", "ceasefire_count_7d",
    "sanction_count_7d", "political_crisis_count_7d",
    "military_intensity_7d", "protest_intensity_7d", "overall_intensity_7d",
    "military_accel", "protest_accel", "overall_accel",
    "avg_polarity_7d", "avg_polarity_30d", "tone_trend",
    "source_diversity_7d", "avg_independent_sources", "avg_contradiction_score",
    "fatalities_7d", "has_military_7d", "has_ceasefire_7d",
    "escalation_index",
    # ── Market signals (5) ────────────────────────────────────────────────────
    # metaculus_p / polymarket_p: 0.5 when unavailable (neutral prior), not -1
    # *_available: 1.0 when the market has real data, 0.0 otherwise
    "metaculus_p", "metaculus_available",
    "polymarket_p", "polymarket_available",
    "market_available",
    # ── Structural country features (3) ───────────────────────────────────────
    "country_conflict_baserate",   # UCDP: fraction of years 2000-2023 with conflict
    "country_polity_norm",         # Polity5 / 10: -1 (autocracy) to +1 (democracy)
    "country_mil_spending_norm",   # SIPRI mil%GDP / 10: 0 to 1
]


def _decay_weight(event: CanonicalEvent, now: datetime) -> float:
    delta_days = (now - event.occurred_at).total_seconds() / 86400
    return math.exp(-delta_days / _DECAY_HALFLIFE_DAYS)


def _quality_weight(event: CanonicalEvent) -> float:
    return event.severity * math.sqrt(max(1, event.independent_sources)) * (1.0 - event.contradiction_score)


def build_features(
    events: list[CanonicalEvent],
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
    country: Optional[str] = None,
    now: Optional[datetime] = None,
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """
    Returns (feature_vector, provenance).
    provenance[feature_name] = [{"event_id": str, "weight": float}, ...]

    Args:
        events      : normalized, deduplicated events for the question
        metaculus_p : Metaculus community probability (None if unavailable)
        polymarket_p: Polymarket YES price (None if unavailable)
        country     : country name for structural features lookup
        now         : reference time (defaults to UTC now)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    window_7d = now - timedelta(days=7)
    window_30d = now - timedelta(days=30)
    window_3d = now - timedelta(days=3)

    events_7d = [e for e in events if e.occurred_at >= window_7d]
    events_30d = [e for e in events if e.occurred_at >= window_30d]
    events_last3d = [e for e in events if e.occurred_at >= window_3d]
    events_prior4d = [e for e in events if window_7d <= e.occurred_at < window_3d]

    feat: dict[str, float] = {k: 0.0 for k in _FEATURE_NAMES}
    prov: dict[str, list[dict]] = defaultdict(list)

    def _add_prov(fname: str, event_id: str, weight: float) -> None:
        prov[fname].append({"event_id": event_id, "weight": round(weight, 6)})

    # ── Count features ──────────────────────────────────────────────────────

    for ev in events_7d:
        if ev.event_type == "military_action":
            feat["military_count_7d"] += 1
            _add_prov("military_count_7d", ev.event_id, 1.0)
        elif ev.event_type == "protest":
            feat["protest_count_7d"] += 1
            _add_prov("protest_count_7d", ev.event_id, 1.0)
        elif ev.event_type == "diplomatic_statement":
            feat["diplomatic_count_7d"] += 1
            _add_prov("diplomatic_count_7d", ev.event_id, 1.0)
        elif ev.event_type == "ceasefire_signal":
            feat["ceasefire_count_7d"] += 1
            _add_prov("ceasefire_count_7d", ev.event_id, 1.0)
        elif ev.event_type == "sanction":
            feat["sanction_count_7d"] += 1
            _add_prov("sanction_count_7d", ev.event_id, 1.0)
        elif ev.event_type == "political_crisis":
            feat["political_crisis_count_7d"] += 1
            _add_prov("political_crisis_count_7d", ev.event_id, 1.0)

    for ev in events_30d:
        if ev.event_type == "military_action":
            feat["military_count_30d"] += 1
            _add_prov("military_count_30d", ev.event_id, 1.0)
        elif ev.event_type == "protest":
            feat["protest_count_30d"] += 1
            _add_prov("protest_count_30d", ev.event_id, 1.0)

    # ── Intensity (decay-weighted) ───────────────────────────────────────────

    for ev in events_7d:
        w = _decay_weight(ev, now) * _quality_weight(ev)
        fname_map = {
            "military_action": "military_intensity_7d",
            "protest": "protest_intensity_7d",
        }
        specific = fname_map.get(ev.event_type)
        if specific:
            feat[specific] += w
            _add_prov(specific, ev.event_id, w)
        feat["overall_intensity_7d"] += w
        _add_prov("overall_intensity_7d", ev.event_id, w)

    # ── Acceleration (last 3d vs prior 4d) ───────────────────────────────────

    def _count_type(evlist: list[CanonicalEvent], etype: str) -> int:
        return sum(1 for e in evlist if e.event_type == etype)

    mil_last3 = _count_type(events_last3d, "military_action")
    mil_prior4 = _count_type(events_prior4d, "military_action")
    feat["military_accel"] = mil_last3 / (mil_prior4 + 0.1)

    pro_last3 = _count_type(events_last3d, "protest")
    pro_prior4 = _count_type(events_prior4d, "protest")
    feat["protest_accel"] = pro_last3 / (pro_prior4 + 0.1)

    all_last3 = len(events_last3d)
    all_prior4 = len(events_prior4d)
    feat["overall_accel"] = all_last3 / (all_prior4 + 0.1)

    for ev in events_last3d + events_prior4d:
        w = _decay_weight(ev, now)
        if ev.event_type == "military_action":
            _add_prov("military_accel", ev.event_id, w)
        elif ev.event_type == "protest":
            _add_prov("protest_accel", ev.event_id, w)
        _add_prov("overall_accel", ev.event_id, w)

    # ── Polarity / tone ──────────────────────────────────────────────────────

    def _weighted_polarity(evlist: list[CanonicalEvent]) -> tuple[float, list]:
        total_w = 0.0
        total_pol = 0.0
        contributors = []
        for ev in evlist:
            w = _decay_weight(ev, now) * max(0.01, _quality_weight(ev))
            total_pol += ev.polarity * w
            total_w += w
            contributors.append((ev.event_id, w))
        if total_w == 0:
            return 0.0, []
        return total_pol / total_w, contributors

    pol_7d, pol_7d_contrib = _weighted_polarity(events_7d)
    feat["avg_polarity_7d"] = pol_7d
    for eid, w in pol_7d_contrib:
        _add_prov("avg_polarity_7d", eid, w)

    pol_30d, pol_30d_contrib = _weighted_polarity(events_30d)
    feat["avg_polarity_30d"] = pol_30d
    for eid, w in pol_30d_contrib:
        _add_prov("avg_polarity_30d", eid, w)

    pol_last3, _ = _weighted_polarity(events_last3d)
    pol_prior4, _ = _weighted_polarity(events_prior4d)
    feat["tone_trend"] = pol_last3 - pol_prior4
    for ev in events_last3d + events_prior4d:
        _add_prov("tone_trend", ev.event_id, _decay_weight(ev, now))

    # ── Source diversity (domain-level) ─────────────────────────────────────
    # Count unique root domains, not coarse collector labels.

    if events_7d:
        all_domains_7d: set[str] = set()
        for e in events_7d:
            all_domains_7d.update(d for d in e.source_domains if d)
        n_domains = len(all_domains_7d) if all_domains_7d else len({e.source for e in events_7d})
        feat["source_diversity_7d"] = min(1.0, n_domains / max(len(events_7d), 1))
        feat["avg_independent_sources"] = sum(e.independent_sources for e in events_7d) / len(events_7d)
        feat["avg_contradiction_score"] = sum(e.contradiction_score for e in events_7d) / len(events_7d)
        for ev in events_7d:
            _add_prov("source_diversity_7d", ev.event_id, 1.0)
            _add_prov("avg_independent_sources", ev.event_id, 1.0)
            _add_prov("avg_contradiction_score", ev.event_id, 1.0)

    # ── Escalation signals ───────────────────────────────────────────────────

    for ev in events_7d:
        feat["fatalities_7d"] += ev.fatalities
        _add_prov("fatalities_7d", ev.event_id, float(ev.fatalities) if ev.fatalities > 0 else 0.1)

    has_mil = any(e.event_type == "military_action" for e in events_7d)
    has_cease = any(e.event_type == "ceasefire_signal" for e in events_7d)
    feat["has_military_7d"] = float(has_mil)
    feat["has_ceasefire_7d"] = float(has_cease)

    for ev in events_7d:
        if ev.event_type == "military_action":
            _add_prov("has_military_7d", ev.event_id, 1.0)
        elif ev.event_type == "ceasefire_signal":
            _add_prov("has_ceasefire_7d", ev.event_id, 1.0)

    feat["escalation_index"] = feat["military_intensity_7d"] - feat["ceasefire_count_7d"] * 0.3
    for ev in events_7d:
        w = _decay_weight(ev, now) * _quality_weight(ev)
        _add_prov("escalation_index", ev.event_id, w)

    # ── Market signals ───────────────────────────────────────────────────────
    # Sentinel is 0.5 (neutral prior) when unavailable, NOT -1.
    # Separate *_available flags tell the model whether to trust the p value.

    meta_avail = metaculus_p is not None and 0.0 < metaculus_p < 1.0
    poly_avail = polymarket_p is not None and 0.0 < polymarket_p < 1.0

    feat["metaculus_p"] = float(metaculus_p) if meta_avail else 0.5
    feat["metaculus_available"] = 1.0 if meta_avail else 0.0
    feat["polymarket_p"] = float(polymarket_p) if poly_avail else 0.5
    feat["polymarket_available"] = 1.0 if poly_avail else 0.0
    feat["market_available"] = float(meta_avail or poly_avail)

    prov["metaculus_p"] = []
    prov["polymarket_p"] = []
    prov["market_available"] = []
    prov["metaculus_available"] = []
    prov["polymarket_available"] = []

    # ── Structural country features (no events, static lookup) ───────────────
    # These are the same regardless of the query window.
    # country is passed from main.py; falls back to world medians if unknown.

    struct = get_country_features(country or "")
    feat.update(struct)

    # Provenance: no events feed these (they're static), but record the country
    prov["country_conflict_baserate"] = []
    prov["country_polity_norm"] = []
    prov["country_mil_spending_norm"] = []

    return dict(feat), dict(prov)


def get_feature_names() -> list[str]:
    return list(_FEATURE_NAMES)

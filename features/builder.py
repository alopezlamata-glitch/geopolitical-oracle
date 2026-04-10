from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from normalizer.canonical import CanonicalEvent

logger = logging.getLogger(__name__)

_DECAY_HALFLIFE_DAYS = 7.0
_FEATURE_NAMES = [
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
    "metaculus_p", "polymarket_p", "market_available",
    # ACLED structural features — computed relative to ACLED's own timeline,
    # not current date, so they remain non-zero despite the ~13-month data lag.
    "acled_military_90d", "acled_fatalities_90d", "acled_conflict_active",
]


def _decay_weight(event: CanonicalEvent, now: datetime) -> float:
    delta_days = (now - event.occurred_at).total_seconds() / 86400
    return math.exp(-delta_days / _DECAY_HALFLIFE_DAYS)


def _quality_weight(event: CanonicalEvent) -> float:
    return event.severity * math.sqrt(max(1, event.independent_sources)) * (1.0 - event.contradiction_score)


def _build_acled_structural(
    events: list[CanonicalEvent],
    prov: dict[str, list[dict]],
) -> dict[str, float]:
    """
    Compute ACLED-specific features relative to the latest ACLED event date.

    ACLED data lags ~13 months behind real time. Using the current date as
    the reference would place all ACLED events outside the 7d/30d windows,
    contributing nothing. Instead, we anchor to ACLED's own most recent event
    and compute a 90-day structural window from there.

    This captures the conflict baseline at the last known point in ACLED's
    dataset — valid as a structural prior, but explicitly NOT a live signal.
    Features are prefixed acled_* to make the semantic clear.
    """
    acled = [e for e in events if e.source == "acled"]
    feat: dict[str, float] = {
        "acled_military_90d": 0.0,
        "acled_fatalities_90d": 0.0,
        "acled_conflict_active": 0.0,
    }
    if not acled:
        return feat

    # Anchor: most recent ACLED event date
    latest = max(e.occurred_at for e in acled)
    window_start = latest - timedelta(days=90)
    recent_acled = [e for e in acled if e.occurred_at >= window_start]

    mil_count = sum(1 for e in recent_acled if e.event_type == "military_action")
    fatalities = sum(e.fatalities for e in recent_acled)

    feat["acled_military_90d"] = float(mil_count)
    feat["acled_fatalities_90d"] = float(fatalities)
    feat["acled_conflict_active"] = 1.0 if mil_count >= 3 else 0.0

    for ev in recent_acled:
        if ev.event_type == "military_action":
            prov["acled_military_90d"].append({"event_id": ev.event_id, "weight": 1.0})
        prov["acled_fatalities_90d"].append(
            {"event_id": ev.event_id, "weight": float(ev.fatalities) if ev.fatalities else 0.1}
        )
        if feat["acled_conflict_active"]:
            prov["acled_conflict_active"].append({"event_id": ev.event_id, "weight": 1.0})

    return feat


def build_features(
    events: list[CanonicalEvent],
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
    now: Optional[datetime] = None,
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """
    Returns (feature_vector, provenance).
    provenance[feature_name] = [{"event_id": str, "weight": float}, ...]
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

    # ── Intensity (weighted) ────────────────────────────────────────────────

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

    # ── Acceleration ────────────────────────────────────────────────────────

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

    # ── Polarity ────────────────────────────────────────────────────────────

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
    # Use root domains tracked in source_domains, not coarse collector labels.
    # This avoids counting 10 GDELT rows from reuters.com as 10 independent
    # observations — they are one.

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

    # ── Escalation signals ──────────────────────────────────────────────────

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

    # ── ACLED structural features (source-relative timeline) ─────────────────

    acled_feat = _build_acled_structural(events, prov)
    feat.update(acled_feat)

    # ── Market signals ──────────────────────────────────────────────────────

    feat["metaculus_p"] = metaculus_p if metaculus_p is not None else -1.0
    feat["polymarket_p"] = polymarket_p if polymarket_p is not None else -1.0
    feat["market_available"] = float(metaculus_p is not None or polymarket_p is not None)

    prov["metaculus_p"] = []
    prov["polymarket_p"] = []
    prov["market_available"] = []

    return dict(feat), dict(prov)


def get_feature_names() -> list[str]:
    return list(_FEATURE_NAMES)

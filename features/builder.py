from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from normalizer.canonical import CanonicalEvent
from features.country_data import get_country_features

logger = logging.getLogger(__name__)

# ── LLM feature names (v4 additions) ─────────────────────────────────────────
# These are NOT in _FEATURE_NAMES (v3 model never sees them).
# Stored in feature_snapshots for future v4 model training.
_LLM_FEATURE_NAMES = [
    "llm_threat_level",
    "llm_escalation",
    "llm_deescalation",
    "llm_event_certainty",
    "llm_actor_hostility",
    "llm_available",
    # Semantic embedding feature (v4)
    "llm_query_event_similarity",  # cosine sim between question embedding and mean event embedding
]

# ── Domain-specific LLM-extracted features ───────────────────────────────────
# Used by base_rate_predictor for non-conflict domains.
# LLM extracts these as numbers from text; statistical engine uses them for probability.
_DOMAIN_FEATURE_NAMES = [
    # Political
    "pol_resignation_signals", "pol_approval_pressure", "pol_coalition_stability",
    "pol_electoral_proximity", "pol_judicial_pressure",
    # Economic
    "eco_rate_change_prob", "eco_gdp_momentum", "eco_debt_stress",
    "eco_market_volatility", "eco_policy_uncertainty",
    # Legal
    "leg_arrest_probability", "leg_extradition_risk", "leg_evidence_strength",
    "leg_jurisdictional_support", "leg_precedent_match",
]

_DECAY_HALFLIFE_DAYS = 7.0

# ── Feature registry ──────────────────────────────────────────────────────────
# Changes from v2 (32 features) → v3 (27 features):
#
#   REMOVED (8): features with 0.0 XGBoost importance (never used by model)
#     political_crisis_count_7d  — too rare in ICEWS training data
#     avg_contradiction_score    — always ~0, ICEWS has no cross-source contradiction
#     fatalities_7d              — ICEWS events lack casualty data (always 0)
#     metaculus_p, metaculus_available   — 0% of training has real market data
#     polymarket_p, polymarket_available — same
#     market_available                   — same
#     Market signals are now applied as a post-model override in predictor/inference.py
#
#   ADDED (3): informative derived features replacing the removed ones
#     ceasefire_ratio_7d  — ceasefire_count_7d / (military_count_7d + 0.1)
#                           captures de-escalation pressure relative to conflict level
#     event_velocity_7d   — events_7d / (events_30d/4.3 + 0.1)
#                           detects acceleration vs rolling baseline (>1 = accelerating)
#     military_share_7d   — military_count_7d / (total_events_7d + 0.1)
#                           conflict concentration: how military-dominated is the news?

_FEATURE_NAMES = [
    # ── Event-derived features (21) ──────────────────────────────────────────
    "military_count_7d", "military_count_30d",
    "protest_count_7d", "protest_count_30d",
    "diplomatic_count_7d", "ceasefire_count_7d",
    "sanction_count_7d",
    "military_intensity_7d", "protest_intensity_7d", "overall_intensity_7d",
    "military_accel", "protest_accel", "overall_accel",
    "avg_polarity_7d", "avg_polarity_30d", "tone_trend",
    "source_diversity_7d", "avg_independent_sources",
    "has_military_7d", "has_ceasefire_7d",
    "escalation_index",
    # ── Derived ratio features (3) ────────────────────────────────────────────
    "ceasefire_ratio_7d",   # ceasefire_count_7d / (military_count_7d + 0.1)
    "event_velocity_7d",    # events_7d / (events_30d/4.3 + 0.1)  — >1 means accelerating
    "military_share_7d",    # military_count_7d / (total_events_7d + 0.1)
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
    metaculus_p: Optional[float] = None,   # passed through for caller use; NOT in feature vector
    polymarket_p: Optional[float] = None,  # passed through for caller use; NOT in feature vector
    country: Optional[str] = None,
    now: Optional[datetime] = None,
    question: Optional[str] = None,        # used for LLM feature extraction (non-fatal)
    use_llm: bool = True,                  # set False to skip LLM even if Ollama available
    wiki_context: str = "",                # Wikipedia background text for LLM prompt enrichment
    event_family: Optional[str] = None,    # if set, extract domain-specific features (pol/eco/leg)
    economic_context: Optional[dict] = None,  # pre-fetched WB/FRED/V-Dem features
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """
    Returns (feature_vector, provenance).
    provenance[feature_name] = [{"event_id": str, "weight": float}, ...]

    Args:
        events      : normalized, deduplicated events for the question
        metaculus_p : passed through for market override — NOT added to feature vector
        polymarket_p: passed through for market override — NOT added to feature vector
        country     : country name for structural features lookup
        now         : reference time (defaults to UTC now)
        question     : original question text (used for LLM feature extraction)
        use_llm      : if True, attempt Ollama text feature extraction (non-fatal)
        wiki_context : Wikipedia background text injected into LLM prompt (optional)

    Market signals are applied as a post-model override in predictor/inference.py,
    not as XGBoost input features, because 0% of ICEWS training examples have market data.

    LLM features (llm_*): added to the returned dict but NOT in _FEATURE_NAMES.
    The v3 XGBoost model ignores them. They are stored in feature_snapshots for v4.
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
        for ev in events_7d:
            _add_prov("source_diversity_7d", ev.event_id, 1.0)
            _add_prov("avg_independent_sources", ev.event_id, 1.0)

    # ── Escalation signals ───────────────────────────────────────────────────

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

    # ── Derived ratio features ────────────────────────────────────────────────
    # These capture relative dynamics that raw counts miss.

    total_events_7d = len(events_7d)
    total_events_30d = len(events_30d)

    feat["ceasefire_ratio_7d"] = feat["ceasefire_count_7d"] / (feat["military_count_7d"] + 0.1)
    feat["event_velocity_7d"] = total_events_7d / (total_events_30d / 4.3 + 0.1)
    feat["military_share_7d"] = feat["military_count_7d"] / (total_events_7d + 0.1)

    for ev in events_7d:
        _add_prov("ceasefire_ratio_7d", ev.event_id, 1.0)
        _add_prov("event_velocity_7d", ev.event_id, 1.0)
        _add_prov("military_share_7d", ev.event_id, 1.0)
    for ev in events_30d:
        _add_prov("event_velocity_7d", ev.event_id, 0.5)

    # ── Structural country features (no events, static lookup) ───────────────
    # These are the same regardless of the query window.
    # country is passed from main.py; falls back to world medians if unknown.

    struct = get_country_features(country or "")
    feat.update(struct)

    # Provenance: no events feed these (they're static), but record the country
    prov["country_conflict_baserate"] = []
    prov["country_polity_norm"] = []
    prov["country_mil_spending_norm"] = []

    # ── World state enrichment (Phase 1 world model) ─────────────────────────
    # If a pre-computed world state exists for this country (from the daily
    # update_world_state.py run), use it to enrich features:
    #   - Features with no signal from current events → filled from world state
    #   - Features with signal from current events → blended (fresh=0.7, ws=0.3)
    # World state captures EMA-smoothed, causally-propagated history not visible
    # in the per-question event window.
    if country:
        try:
            from world_state.reader import get_world_state
            ws = get_world_state(country)
            if ws is not None:
                staleness = ws.get("_world_state_staleness_days", 99)
                # Only use if not too stale (≤2 days for dynamic features)
                ws_weight = 0.3 if staleness <= 2 else 0.15
                n_enriched = 0
                for fname in _FEATURE_NAMES:
                    ws_val = ws.get(fname)
                    if ws_val is None:
                        continue
                    current = feat.get(fname, 0.0)
                    if current == 0.0:
                        # No fresh signal — use world state directly
                        feat[fname] = float(ws_val)
                        n_enriched += 1
                    else:
                        # Fresh signal exists — blend, prioritizing fresh events
                        feat[fname] = round((1.0 - ws_weight) * current + ws_weight * float(ws_val), 6)
                if n_enriched:
                    logger.info(
                        "world_state: enriched %d zero-features from %s state (stale=%dd, ws_weight=%.2f)",
                        n_enriched, country, staleness, ws_weight,
                    )
        except Exception as e:
            logger.debug("world_state enrichment skipped: %s", e)

    # ── Neighbor context injection (Phase 3 world model) ─────────────────────
    # For each causal neighbor of this entity, inject their world state
    # source features as prefixed neighbor_* keys. These are informational
    # (not in _FEATURE_NAMES) and used by base_rate_predictor + coherence.
    if country:
        try:
            from world_state.entity_graph import get_neighbor_context
            neighbor_ctx = get_neighbor_context(country)
            if neighbor_ctx:
                # Store non-private keys as features (base_rate_predictor reads them)
                for k, v in neighbor_ctx.items():
                    if not k.startswith("_"):
                        feat[k] = v
        except Exception as e:
            logger.debug("neighbor context injection skipped: %s", e)

    # ── LLM feature extraction (v4, non-fatal) ────────────────────────────────
    # These 6 features are NOT in _FEATURE_NAMES so the v3 XGBoost ignores them.
    # They are stored alongside v3 features in feature_snapshots for v4 training.

    llm_feats: dict[str, float] = {k: 0.0 for k in _LLM_FEATURE_NAMES}

    if use_llm and question and events:
        # Collect headlines once — shared by both text extractor and embedder
        headlines = [
            ev.raw_title
            for ev in sorted(events, key=lambda e: e.occurred_at, reverse=True)
            if ev.raw_title.strip()
        ]

        # ── Text feature extraction ───────────────────────────────────────────
        if headlines:
            try:
                from llm.text_features import extract_llm_features
                llm_result = extract_llm_features(
                    question=question,
                    headlines=headlines,
                    wiki_context=wiki_context,
                )
                llm_feats.update(llm_result.to_feature_dict())
                if llm_result.available:
                    logger.info(
                        "llm text features: threat=%.2f esc=%.2f deesc=%.2f "
                        "cert=%.2f host=%.2f (%.0fms)",
                        llm_result.threat_level, llm_result.escalation,
                        llm_result.deescalation, llm_result.event_certainty,
                        llm_result.actor_hostility, llm_result.latency_ms or 0,
                    )
            except Exception as e:
                logger.debug("llm text feature extraction skipped: %s", e)

        # ── Semantic similarity embedding ─────────────────────────────────────
        if headlines:
            try:
                from llm.embedder import embed_events_and_question
                embed_result = embed_events_and_question(
                    question=question,
                    headlines=headlines,
                )
                if embed_result.available:
                    llm_feats["llm_query_event_similarity"] = embed_result.query_similarity
                    logger.info(
                        "llm embedding: query_sim=%.3f  n=%d  (%.0fms)",
                        embed_result.query_similarity,
                        embed_result.n_embedded,
                        embed_result.latency_ms or 0,
                    )
            except Exception as e:
                logger.debug("llm embedding skipped: %s", e)

    feat.update(llm_feats)

    # ── Structural economic context (WB / FRED / V-Dem) ──────────────────────
    # Pre-fetched features from World Bank, FRED, and V-Dem snapshot.
    # These are structural/slow-moving signals; LLM domain features may override
    # with question-specific dynamic signals.
    if economic_context:
        for k, v in economic_context.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                feat[k] = float(v)
        n_applied = sum(1 for k, v in economic_context.items()
                        if not k.startswith("_") and isinstance(v, (int, float)) and v != 0.0)
        if n_applied:
            logger.info("builder: applied %d structural features (WB/FRED/V-Dem)", n_applied)

    # ── Domain-specific feature extraction (non-conflict) ─────────────────────
    # For political/economic/legal domains, LLM extracts structured domain features.
    # These are used by base_rate_predictor (not by XGBoost) and stored in feature_snapshots.
    if use_llm and event_family and event_family not in ("conflict", "unknown"):
        domain_feats: dict[str, float] = {k: 0.0 for k in _DOMAIN_FEATURE_NAMES}
        try:
            from llm.domain_features import extract_domain_features
            # Collect headlines for domain extractor
            domain_headlines = [
                ev.raw_title
                for ev in sorted(events, key=lambda e: e.occurred_at, reverse=True)
                if ev.raw_title.strip()
            ] if events else []
            extracted = extract_domain_features(
                event_family=event_family,
                question=question or "",
                headlines=domain_headlines,
            )
            domain_feats.update(extracted)
            n_nonzero = sum(1 for v in extracted.values() if abs(v) > 0.01)
            logger.info(
                "domain features[%s]: %d/%d nonzero extracted",
                event_family, n_nonzero, len(extracted),
            )
        except Exception as e:
            logger.debug("domain feature extraction skipped: %s", e)
        # Only apply LLM domain features when nonzero — don't overwrite WB/FRED with zeros
        for k, v in domain_feats.items():
            if abs(v) > 0.01 or feat.get(k, 0.0) == 0.0:
                feat[k] = v

    return dict(feat), dict(prov)


def get_feature_names() -> list[str]:
    """Return v3 feature names (used by the trained XGBoost model)."""
    return list(_FEATURE_NAMES)


def get_feature_names_v4() -> list[str]:
    """Return all feature names including LLM features (for future v4 training)."""
    return list(_FEATURE_NAMES) + list(_LLM_FEATURE_NAMES)


def get_domain_feature_names() -> list[str]:
    """Return domain-specific feature names for non-conflict domains."""
    return list(_DOMAIN_FEATURE_NAMES)

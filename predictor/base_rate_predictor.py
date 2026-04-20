"""
Statistical base-rate predictor for non-conflict domains.

Architecture:
  [LLM-extracted features]  → never touch probability calculation
  [Base rate lookup]        → P(event) given predicate + horizon
  [Log-odds accumulation]   → each feature shifts log-odds by weight
  [Interaction terms]       → cross-domain non-linear relationships
  [Isotonic calibration]    → correct systematic bias from domain data
  [Market blend]            → same blend layer as XGBoost path
  [Conformal CI]            → same CI layer as XGBoost path

LLMs are used ONLY upstream (feature extraction). No LLM reasoning here.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import scipy.stats

from predictor.market_prior import (
    BlendResult,
    blend_market_prior,
    resolve_market_signal,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_BASE_RATES_PATH = _DATA_DIR / "base_rates.json"
_FEATURE_WEIGHTS_PATH = _DATA_DIR / "feature_weights.json"
_REFERENCE_PERIOD = 180  # days; base rates calibrated to this window

_base_rates_cache: Optional[dict] = None
_feature_weights_cache: Optional[dict] = None


def _load_base_rates() -> dict:
    global _base_rates_cache
    if _base_rates_cache is None:
        _base_rates_cache = json.loads(_BASE_RATES_PATH.read_text())
    return _base_rates_cache


def _load_feature_weights() -> dict:
    global _feature_weights_cache
    if _feature_weights_cache is None:
        _feature_weights_cache = json.loads(_FEATURE_WEIGHTS_PATH.read_text())
    return _feature_weights_cache


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _scale_rate_to_horizon(rate_per_ref: float, horizon_days: int) -> float:
    """
    Convert a rate calibrated to _REFERENCE_PERIOD days to a given horizon.
    Uses survival / constant-hazard model:
      P(event in N days) = 1 - (1 - rate_per_ref)^(N / ref)
    """
    if horizon_days <= 0:
        return 0.0
    exponent = horizon_days / _REFERENCE_PERIOD
    return 1.0 - (1.0 - min(rate_per_ref, 0.999)) ** exponent


def _lookup_base_rate(
    predicate: str,
    horizon_days: Optional[int],
    entity_features: Optional[dict] = None,
) -> float:
    """
    Return P(event) given predicate and horizon, optionally conditioned on
    entity-specific structural features.

    Conditioning multipliers (applied to the raw base rate before horizon scaling):
      - country_polity_norm < 0.3  → coup / resign / removal predicates × 2.0
        (autocratic regimes have structurally higher leadership-turnover risk)
      - has_active_conflict ≥ 1.0  → military_escalation × 1.5
        (ongoing conflicts dramatically increase escalation base rate)
      - eco_debt_stress > 0.7      → economic default / crisis predicates × 2.5
        (high debt stress → near-term default risk well above historical mean)
      - country_conflict_baserate > 0.5 → military_escalation × 1.2
        (countries with chronic conflict history regress to higher base rate)

    All multipliers are capped so the conditioned rate never exceeds 0.95.
    """
    data = _load_base_rates()
    predicates = data.get("predicates", {})

    # Normalize predicate: try exact, then strip suffix, then 'unknown'
    entry = predicates.get(predicate) or predicates.get(predicate.split("_")[0]) or predicates.get("unknown", {})

    # Pick best rate key for the given horizon
    if horizon_days is None:
        horizon_days = _REFERENCE_PERIOD

    # Try rate_per_Nd keys in order of proximity to horizon
    ref = data.get("reference_period_days", _REFERENCE_PERIOD)
    rate = entry.get(f"rate_per_{ref}d")
    if rate is None:
        rate = entry.get("rate_per_180d") or entry.get("rate_per_365d") or entry.get("rate_per_30d") or 0.15
    rate = float(rate)

    # ── Conditional multipliers from entity structural features ───────────────
    if entity_features:
        polity      = float(entity_features.get("country_polity_norm", 0.5))
        active_conf = float(entity_features.get("has_active_conflict", 0.0))
        debt_stress = float(entity_features.get("eco_debt_stress", 0.0))
        conf_base   = float(entity_features.get("country_conflict_baserate", 0.0))

        # Autocratic regimes → higher leadership-turnover base rates
        if polity < 0.3 and any(kw in predicate for kw in ("coup", "resign", "removal", "election")):
            rate = min(0.95, rate * 2.0)
            logger.debug("base_rate: polity_norm=%.2f → %s multiplier 2.0 → rate=%.3f", polity, predicate, rate)

        # Active conflict → military escalation more likely
        if active_conf >= 1.0 and "military" in predicate:
            rate = min(0.95, rate * 1.5)
            logger.debug("base_rate: active_conflict → military multiplier 1.5 → rate=%.3f", rate)

        # Chronic conflict → military escalation somewhat more likely
        if conf_base > 0.5 and "military" in predicate:
            rate = min(0.95, rate * 1.2)
            logger.debug("base_rate: conf_baserate=%.2f → military multiplier 1.2 → rate=%.3f", conf_base, rate)

        # High debt stress → economic crisis more likely
        if debt_stress > 0.7 and any(kw in predicate for kw in ("default", "crisis", "recession", "economic")):
            rate = min(0.95, rate * 2.5)
            logger.debug("base_rate: debt_stress=%.2f → economic multiplier 2.5 → rate=%.3f", debt_stress, rate)

    return float(_scale_rate_to_horizon(rate, horizon_days))


def _compute_interactions(features: dict, event_family: str) -> float:
    """
    Non-linear cross-feature interaction terms for hidden relationships.

    Interaction formula: w_ij * f_i * f_j
    Positive: co-occurrence of both signals amplifies log-odds.
    Negative: one signal dampens the other.
    """
    delta = 0.0
    weights = _load_feature_weights()
    interactions = weights.get("interactions", {}).get(event_family, {})

    for key, w in interactions.items():
        if "__" not in key:
            continue
        fa, fb = key.split("__", 1)
        va = features.get(fa, 0.0)
        vb = features.get(fb, 0.0)
        delta += w * va * vb

    return delta


def predict_base_rate(
    features: dict,
    predicate: str,
    event_family: str,
    horizon_days: Optional[int],
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
    metaculus_forecasters: Optional[int] = None,
    polymarket_volume: Optional[float] = None,
    polymarket_match_score: Optional[float] = None,
    as_of_time: Optional[datetime] = None,
) -> dict:
    """
    Full prediction pipeline for non-conflict domains.

    1. Base rate anchor from predicate + horizon (outside view)
    2. Log-odds accumulation via domain feature weights (inside view)
    3. Interaction terms for cross-feature non-linearity
    4. Domain isotonic calibration (if fitted data exists)
    5. Market prior blend (same layer as XGBoost path)
    6. Beta-CI scaled by feature richness

    Returns dict with same schema as inference.predict() for full compatibility.
    """
    t0 = time.monotonic()
    if as_of_time is None:
        as_of_time = datetime.now(timezone.utc)

    # ── 1. Base rate (outside view, entity-conditioned) ──────────────────────
    base_rate = _lookup_base_rate(predicate, horizon_days, entity_features=features)
    log_odds = _logit(base_rate)

    logger.debug(
        "base_rate_predictor[%s/%s]: base_rate=%.3f log_odds=%.3f horizon=%s",
        event_family, predicate, base_rate, log_odds, horizon_days,
    )

    # ── 2. Feature weight accumulation (inside view) ──────────────────────────
    weights_data = _load_feature_weights()
    domain_weights: dict = weights_data.get(event_family, {})

    feature_contributions: dict[str, float] = {}
    for fname, w in domain_weights.items():
        if fname.startswith("_"):
            continue
        value = features.get(fname, 0.0)
        if not isinstance(value, (int, float)):
            continue
        if value != 0.0:
            contrib = float(w) * float(value)
            log_odds += contrib
            feature_contributions[fname] = round(contrib, 4)

    # ── 3. Interaction terms ──────────────────────────────────────────────────
    interaction_delta = _compute_interactions(features, event_family)
    log_odds += interaction_delta

    # ── 4. Raw probability ────────────────────────────────────────────────────
    raw_prob = _sigmoid(log_odds)
    raw_prob = max(0.01, min(0.99, raw_prob))

    # ── 5. Domain isotonic calibration ───────────────────────────────────────
    from model.domain_calibrator import apply_domain_calibrator
    calibrated_prob = apply_domain_calibrator(raw_prob, event_family)
    calibrated_prob = max(0.01, min(0.99, calibrated_prob))

    # ── 6. Market prior blend ─────────────────────────────────────────────────
    signals = resolve_market_signal(
        metaculus_p=metaculus_p,
        metaculus_forecasters=metaculus_forecasters,
        polymarket_p=polymarket_p,
        polymarket_volume=polymarket_volume,
        polymarket_match_score=polymarket_match_score,
        as_of_time=as_of_time,
    )
    blend: BlendResult = blend_market_prior(calibrated_prob, signals, as_of_time)
    final_prob = max(0.01, min(0.99, blend.p_final))

    # ── 7. Confidence interval (beta, width scales with feature richness) ─────
    n_active = sum(
        1 for v in features.values()
        if isinstance(v, (int, float)) and abs(float(v)) > 0.01
    )
    eff_n = max(5, min(50, n_active * 3))
    alpha_p = final_prob * eff_n
    beta_p = (1.0 - final_prob) * eff_n
    lo, hi = scipy.stats.beta.interval(0.80, max(0.1, alpha_p), max(0.1, beta_p))

    answer = "YES" if final_prob >= 0.5 else "NO"
    latency_ms = round((time.monotonic() - t0) * 1000, 0)

    logger.info(
        "base_rate_predictor[%s/%s]: base=%.3f raw=%.3f cal=%.3f final=%.3f answer=%s latency=%.0fms",
        event_family, predicate, base_rate, raw_prob, calibrated_prob, final_prob, answer, latency_ms,
    )

    return {
        # Core prediction (same keys as inference.predict())
        "raw_prob": round(raw_prob, 4),
        "calibrated_prob": round(final_prob, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "ci_method": "beta_feature_scaled",
        "answer": answer,
        "untrained": False,
        # Market blend audit trail
        "p_model_raw": round(blend.p_model, 4),
        "p_market_raw": blend.p_market,
        "market_weight": blend.market_weight,
        "market_sources": blend.market_sources,
        "market_match_score": blend.best_match_score,
        "blend_strategy": blend.blend_strategy,
        "blend_strategy_version": blend.blend_strategy_version,
        "market_gate_passed": blend.gate_passed,
        "market_gate_reason": blend.gate_reason,
        "n_market_signals": blend.n_signals,
        "market_override": blend.gate_passed and blend.n_signals > 0,
        # Base-rate predictor specific
        "predictor": "base_rate_v1",
        "base_rate": round(base_rate, 4),
        "log_odds_final": round(log_odds, 4),
        "interaction_delta": round(interaction_delta, 4),
        "feature_contributions": feature_contributions,
        "n_active_features": n_active,
        "latency_ms": latency_ms,
    }

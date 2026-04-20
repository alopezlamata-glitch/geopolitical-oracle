from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats

from model.trainer import load_model
from model.calibrator import load_calibrator
from features.builder import get_feature_names
from predictor.market_prior import (
    BlendResult,
    MarketSignal,
    blend_market_prior,
    resolve_market_signal,
)

logger = logging.getLogger(__name__)

# Models that use XGBoost (trained, feature-vector based)
_XGB_MODEL_IDS = {"xgb_conflict_v3"}
# Models that use the statistical base-rate predictor (non-conflict domains)
_BASE_RATE_MODEL_IDS = {"ollama_reasoning_v1", "base_rate_v1"}


def _quantile_at(scores: list[float], coverage: float) -> float:
    """Split-conformal quantile: ceil((n+1)*(1-alpha))-th order statistic."""
    n = len(scores)
    k = min(math.ceil((n + 1) * (1.0 - (1.0 - coverage))), n)
    return sorted(scores)[k - 1]


def _conformal_ci(calibrated_prob: float, coverage: float = 0.80) -> tuple[float, float, str]:
    """
    Asymmetric split-conformal prediction interval. See CONFORMAL.md for guarantee.

    Uses per-class nonconformity scores saved at calibration time:
      scores_pos: s_i = 1 - p_cal for y=1 calibration examples
      scores_neg: s_i = p_cal     for y=0 calibration examples

    q_pos (upper margin) = quantile of scores_pos at (1-alpha) level
    q_neg (lower margin) = quantile of scores_neg at (1-alpha) level

    CI = [p - q_neg, p + q_pos]

    Marginal coverage P(Y in C(X)) >= 1-alpha holds under exchangeability.
    Falls back to symmetric conformal, then beta heuristic, when n < 5.
    """
    conformal_path = Path(__file__).parent.parent / "data" / "model" / "conformal_scores.json"
    if conformal_path.exists():
        try:
            data = json.loads(conformal_path.read_text())
            scores_pos = data.get("scores_pos", [])
            scores_neg = data.get("scores_neg", [])
            p = calibrated_prob

            if len(scores_pos) >= 5 and len(scores_neg) >= 5:
                q_pos = _quantile_at(scores_pos, coverage)
                q_neg = _quantile_at(scores_neg, coverage)
                lo = max(0.01, p - q_neg)
                hi = min(0.99, p + q_pos)
                return lo, hi, "conformal"

            all_scores = scores_pos + scores_neg
            if len(all_scores) >= 5:
                margin = _quantile_at(all_scores, coverage)
                lo = max(0.01, p - margin)
                hi = min(0.99, p + margin)
                return lo, hi, "conformal-sym"
        except Exception:
            pass

    # Final fallback: beta heuristic (honest about small-N uncertainty)
    eff_n = 20
    alpha_p = calibrated_prob * eff_n
    beta_p = (1 - calibrated_prob) * eff_n
    lo, hi = scipy.stats.beta.interval(coverage, max(0.1, alpha_p), max(0.1, beta_p))
    return float(lo), float(hi), "heuristic"


def predict(
    features: dict[str, float],
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
    *,
    # Quality metadata — optional, improves blend fidelity when provided.
    # Pass these from the collectors for a calibrated, gated blend.
    metaculus_forecasters: Optional[int] = None,
    polymarket_volume: Optional[float] = None,
    polymarket_match_score: Optional[float] = None,
    as_of_time: Optional[datetime] = None,
) -> dict:
    """
    Returns prediction dict with full audit trail.

    Market signals (metaculus_p, polymarket_p) enter ONLY through the post-
    calibration blend layer — they never touch the XGBoost feature vector,
    preventing signal duplication with the market_available / market_consensus
    features already encoded in the v3 feature schema.

    Blend behaviour:
      - No market signal available → model_only.
      - Gate fails (low match, stale) → model_only.
      - Gate passes → log-odds blend with dynamic weight in [0, 0.70].

    All blend metadata is included in the returned dict for full auditability.
    """
    if as_of_time is None:
        as_of_time = datetime.now(timezone.utc)

    model, feature_names = load_model()
    calibrator = load_calibrator()
    untrained = model is None

    feat_names = feature_names or get_feature_names()
    x = np.array([[float(features.get(f, 0.0)) for f in feat_names]], dtype=np.float32)

    # ── 1. Raw model score ────────────────────────────────────────────────────
    if untrained:
        # No model: use market signals as prior, else uniform
        valid = [p for p in [metaculus_p, polymarket_p] if p is not None and 0 < p < 1]
        raw_prob = float(np.mean(valid)) if valid else 0.5
    else:
        raw_prob = float(model.predict_proba(x)[0, 1])

    # ── 2. Isotonic recalibration ─────────────────────────────────────────────
    if calibrator is not None and not untrained:
        calibrated_prob = float(calibrator.predict([raw_prob])[0])
    else:
        calibrated_prob = raw_prob

    calibrated_prob = max(0.01, min(0.99, calibrated_prob))

    # ── 3. Calibrated market prior blend (post-model, non-fatal) ─────────────
    blend: BlendResult
    if untrained:
        # Untrained: skip blend, raw_prob already incorporates markets if present
        blend = BlendResult(
            p_final=calibrated_prob,
            p_model=calibrated_prob,
            p_market=None,
            market_weight=0.0,
            market_sources=[],
            best_match_score=None,
            blend_strategy="model_only",
            blend_strategy_version="logodds_v1",
            gate_passed=False,
            gate_reason="model untrained — market prior skipped",
            n_signals=0,
        )
    else:
        signals = resolve_market_signal(
            metaculus_p=metaculus_p,
            metaculus_forecasters=metaculus_forecasters,
            polymarket_p=polymarket_p,
            polymarket_volume=polymarket_volume,
            polymarket_match_score=polymarket_match_score,
            as_of_time=as_of_time,
        )
        blend = blend_market_prior(calibrated_prob, signals, as_of_time)

    final_prob = max(0.01, min(0.99, blend.p_final))

    # ── 4. Conformal CI (on final blended probability) ────────────────────────
    lo, hi, ci_method = _conformal_ci(final_prob)

    answer = "YES" if final_prob >= 0.5 else "NO"

    return {
        # Core prediction
        "raw_prob": round(raw_prob, 4),
        "calibrated_prob": round(final_prob, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "ci_method": ci_method,
        "answer": answer,
        "untrained": untrained,
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
        # Backward-compat alias (old code checked this key)
        "market_override": blend.gate_passed and blend.n_signals > 0,
        "predictor": "xgb_conflict_v3",
    }


def predict_for_domain(
    matched_model: str,
    features: dict[str, float],
    event_family: str,
    question: str,
    headlines: list[str],
    deadline: Optional[datetime] = None,
    wiki_context: str = "",
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
    metaculus_forecasters: Optional[int] = None,
    polymarket_volume: Optional[float] = None,
    polymarket_match_score: Optional[float] = None,
    as_of_time: Optional[datetime] = None,
    country: Optional[str] = None,
) -> dict:
    """
    Model router: dispatches to the right predictor based on matched_model.

    - xgb_conflict_v3    → XGBoost predict() (existing, trained)
    - ollama_reasoning_v1 → Ollama reasoning predictor (new domains)
    - unknown            → XGBoost fallback with untrained warning

    All paths return the same dict schema for downstream compatibility.
    """
    if matched_model in _BASE_RATE_MODEL_IDS:
        from predictor.base_rate_predictor import predict_base_rate
        # Extract predicate and horizon from features dict (set by caller)
        predicate = features.get("_predicate", "unknown")
        if isinstance(predicate, float):
            predicate = "unknown"
        horizon_days: Optional[int] = None
        if deadline is not None and as_of_time is not None:
            ref_date = as_of_time if isinstance(as_of_time, datetime) else datetime.now(timezone.utc)
            delta = deadline - ref_date
            horizon_days = max(0, delta.days)
        elif deadline is not None:
            from datetime import datetime as _dt, timezone as _tz
            delta = deadline - _dt.now(_tz.utc)
            horizon_days = max(0, delta.days)
        return predict_base_rate(
            features=features,
            predicate=str(predicate),
            event_family=event_family,
            horizon_days=horizon_days,
            metaculus_p=metaculus_p,
            polymarket_p=polymarket_p,
            metaculus_forecasters=metaculus_forecasters,
            polymarket_volume=polymarket_volume,
            polymarket_match_score=polymarket_match_score,
            as_of_time=as_of_time,
        )

    # Default: XGBoost path (conflict domain or unknown model)
    if matched_model not in _XGB_MODEL_IDS:
        logger.warning("predict_for_domain: unknown model_id '%s' — using XGBoost fallback", matched_model)

    result = predict(
        features=features,
        metaculus_p=metaculus_p,
        polymarket_p=polymarket_p,
        metaculus_forecasters=metaculus_forecasters,
        polymarket_volume=polymarket_volume,
        polymarket_match_score=polymarket_match_score,
        as_of_time=as_of_time,
    )

    # ── World model trajectory enrichment (Phase 2) ────────────────────────────
    # If a world state + VAR trajectory is available for this country, compute
    # trajectory-based probability and use it to refine the CI bounds.
    # The point estimate (calibrated_prob) stays model-driven; trajectory shifts CI.
    if country and deadline is not None:
        _enrich_with_trajectory(result, country, event_family, features, deadline, as_of_time)

    return result


def _enrich_with_trajectory(
    result: dict,
    country: str,
    event_family: str,
    features: dict,
    deadline: datetime,
    as_of_time: Optional[datetime],
) -> None:
    """
    Non-fatal: enriches result dict in-place with trajectory metadata.
    Does NOT change calibrated_prob — only adds trajectory context.
    """
    try:
        from world_state.trajectory import marginalize_with_uncertainty, predict_trajectory

        ref = as_of_time or datetime.now(timezone.utc)
        horizon_days = max(1, (deadline - ref).days)
        result["_horizon_days"] = horizon_days

        predicate = str(features.get("_predicate", "unknown"))

        traj_result = marginalize_with_uncertainty(
            entity_name=country,
            predicate=predicate,
            event_family=event_family,
            horizon_days=horizon_days,
            n_samples=100,
        )

        if not traj_result:
            return

        p_traj = traj_result.get("p_trajectory")
        if p_traj is None:
            return

        p_current = result.get("calibrated_prob", 0.5)

        # Blend CI: trajectory narrows/widens based on consistency with current estimate
        consistency = 1.0 - abs(p_traj - p_current)   # 1.0 = perfectly aligned

        result["trajectory_p"]          = p_traj
        result["trajectory_ci_lo"]      = traj_result.get("p_lo")
        result["trajectory_ci_hi"]      = traj_result.get("p_hi")
        result["trajectory_ci_method"]  = traj_result.get("trajectory_ci_method", "var")
        result["trajectory_risk_profile"] = traj_result.get("risk_profile")
        result["trajectory_consistency"] = round(consistency, 3)
        result["trajectory_n_samples"]  = traj_result.get("n_samples", 0)

        # ── Multi-horizon sub-forecasts ────────────────────────────────────────
        for short_h in (7, 30):
            if horizon_days > short_h:
                sub = marginalize_with_uncertainty(
                    entity_name=country,
                    predicate=predicate,
                    event_family=event_family,
                    horizon_days=short_h,
                    n_samples=60,
                )
                if sub and sub.get("p_trajectory") is not None:
                    result[f"trajectory_p_{short_h}d"] = sub["p_trajectory"]

        logger.info(
            "trajectory[%s/%s]: p_traj=%.3f  p_current=%.3f  consistency=%.2f  "
            "horizon=%dd  risk=%s",
            country, event_family,
            p_traj, p_current, consistency,
            horizon_days, traj_result.get("risk_profile", "?"),
        )
    except Exception as e:
        logger.debug("trajectory enrichment skipped: %s", e)

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats

from model.trainer import load_model
from model.calibrator import load_calibrator
from features.builder import get_feature_names

logger = logging.getLogger(__name__)


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


def _apply_market_override(
    model_prob: float,
    metaculus_p: Optional[float],
    polymarket_p: Optional[float],
) -> tuple[float, bool]:
    """
    Post-model market override: blend calibrated model output with live market signals.

    When Metaculus or Polymarket data is available for this specific question,
    the market aggregates thousands of forecasters who have already processed
    more context than the ICEWS-trained model sees. We trust it more.

    Blend weights (log-odds space, geometric pooling):
      60% model, 40% market  (when one market available)
      50% model, 50% market  (when both markets available)

    The model is still the anchor because markets may be for slightly different
    questions, and thin markets can be manipulated.

    Returns: (blended_prob, market_was_used)
    """
    valid: list[float] = []
    if metaculus_p is not None and 0.01 < metaculus_p < 0.99:
        valid.append(metaculus_p)
    if polymarket_p is not None and 0.01 < polymarket_p < 0.99:
        valid.append(polymarket_p)

    if not valid:
        return model_prob, False

    # Geometric mean of market signals in log-odds space
    market_lo = sum(math.log(p / (1 - p)) for p in valid) / len(valid)
    market_prob = 1.0 / (1.0 + math.exp(-market_lo))

    # Blend weights: model gets 60%, market gets 40% (one source) or 50% (two)
    w_market = 0.5 if len(valid) >= 2 else 0.4
    w_model = 1.0 - w_market

    model_lo = math.log(model_prob / (1.0 - model_prob))
    blended_lo = w_model * model_lo + w_market * market_lo
    blended = 1.0 / (1.0 + math.exp(-blended_lo))

    logger.info(
        "Market override: model=%.3f market=%.3f (n=%d) → blended=%.3f",
        model_prob, market_prob, len(valid), blended,
    )
    return float(blended), True


def predict(
    features: dict[str, float],
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
) -> dict:
    """
    Returns prediction dict with: raw_prob, calibrated_prob, ci_lo, ci_hi, answer, untrained.

    Market signals (metaculus_p, polymarket_p) are applied as a post-calibration
    blend — they do NOT enter the feature vector, because ICEWS training data has
    no real market observations. Passing them here triggers the override layer.
    """
    model, feature_names = load_model()
    calibrator = load_calibrator()
    untrained = model is None

    feat_names = feature_names or get_feature_names()
    x = np.array([[float(features.get(f, 0.0)) for f in feat_names]], dtype=np.float32)

    if untrained:
        # No model: use market signals as prior, else 0.5
        valid = [p for p in [metaculus_p, polymarket_p] if p is not None and 0 < p < 1]
        raw_prob = float(np.mean(valid)) if valid else 0.5
    else:
        raw_prob = float(model.predict_proba(x)[0, 1])

    if calibrator is not None and not untrained:
        calibrated_prob = float(calibrator.predict([raw_prob])[0])
    else:
        calibrated_prob = raw_prob

    # Clamp before market override so override math is in valid range
    calibrated_prob = max(0.01, min(0.99, calibrated_prob))

    # ── Post-model market override ────────────────────────────────────────────
    market_override_used = False
    if not untrained:
        calibrated_prob, market_override_used = _apply_market_override(
            calibrated_prob, metaculus_p, polymarket_p
        )
        calibrated_prob = max(0.01, min(0.99, calibrated_prob))

    # Conformal CI (falls back to beta heuristic if scores unavailable)
    lo, hi, ci_method = _conformal_ci(calibrated_prob)

    answer = "YES" if calibrated_prob >= 0.5 else "NO"

    return {
        "raw_prob": round(raw_prob, 4),
        "calibrated_prob": round(calibrated_prob, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "ci_method": ci_method,
        "answer": answer,
        "untrained": untrained,
        "market_override": market_override_used,
    }

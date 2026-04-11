from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats

from features.builder import get_feature_names
from model.calibrator import load_calibrator
from model.trainer import load_model

logger = logging.getLogger(__name__)

BASELINE_VERSION = "baseline_v1"
BASELINE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / f"{BASELINE_VERSION}.json"
BASELINE_VARIANTS = frozenset({"model_only", "market_only", "blended"})
DEFAULT_BASELINE_VARIANT = "blended"


def _load_baseline_config() -> dict:
    """Best-effort config loader for baseline metadata and freeze manifest."""
    if not BASELINE_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(BASELINE_CONFIG_PATH.read_text())
    except Exception:
        logger.warning("failed to parse baseline config: %s", BASELINE_CONFIG_PATH)
        return {}


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

    Blend weights (log-odds space, geometric pooling):
      60% model, 40% market  (when one market available)
      50% model, 50% market  (when both markets available)

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


def _extract_market_signal(market_signal: Optional[dict | float]) -> tuple[Optional[float], Optional[float], str]:
    metaculus_p: Optional[float] = None
    polymarket_p: Optional[float] = None
    variant = DEFAULT_BASELINE_VARIANT

    if isinstance(market_signal, (int, float)):
        metaculus_p = float(market_signal)
    elif isinstance(market_signal, dict):
        metaculus_p = market_signal.get("metaculus_p")
        polymarket_p = market_signal.get("polymarket_p")
        if "variant" in market_signal:
            variant = str(market_signal["variant"])

    if variant not in BASELINE_VARIANTS:
        logger.warning("unknown baseline variant '%s'; using %s", variant, DEFAULT_BASELINE_VARIANT)
        variant = DEFAULT_BASELINE_VARIANT

    return metaculus_p, polymarket_p, variant


def _extract_features(snapshot_or_features: dict) -> dict[str, float]:
    if not isinstance(snapshot_or_features, dict):
        raise TypeError("snapshot_or_features must be a dict")
    raw = snapshot_or_features.get("features") if "features" in snapshot_or_features else snapshot_or_features
    if not isinstance(raw, dict):
        raise TypeError("snapshot_or_features['features'] must be a dict when provided")
    return {str(k): float(v) for k, v in raw.items()}


def run_baseline_v1(snapshot_or_features, market_signal=None) -> dict:
    """
    Frozen inference baseline for reproducible evaluation.

    Frozen in `baseline_v1`:
      - Feature schema `v3` ordering/lookup via `get_feature_names()`.
      - Model artifact family (`data/model/xgb_model.json`) and calibrator usage.
      - Conformal CI logic and fallback heuristic.
      - Market-prior policy and blend weights used by inference.
      - Output contract keys used by downstream components.

    Out of scope for baseline_v1:
      - Training changes, feature redesign, new calibration methods.
      - Any new market connectors or alternate fusion logic.
      - Altering output semantics expected by existing callers.
    """
    _ = _load_baseline_config()  # metadata manifest; behavior remains code-frozen for v1.

    features = _extract_features(snapshot_or_features)
    metaculus_p, polymarket_p, variant = _extract_market_signal(market_signal)

    model, feature_names = load_model()
    calibrator = load_calibrator()
    untrained = model is None

    feat_names = feature_names or get_feature_names()
    x = np.array([[float(features.get(f, 0.0)) for f in feat_names]], dtype=np.float32)

    valid = [p for p in [metaculus_p, polymarket_p] if p is not None and 0 < p < 1]
    market_prior = float(np.mean(valid)) if valid else 0.5

    if untrained or variant == "market_only":
        raw_prob = market_prior
    else:
        raw_prob = float(model.predict_proba(x)[0, 1])

    if calibrator is not None and (not untrained) and variant != "market_only":
        calibrated_prob = float(calibrator.predict([raw_prob])[0])
    else:
        calibrated_prob = raw_prob

    calibrated_prob = max(0.01, min(0.99, calibrated_prob))

    market_override_used = False
    if not untrained and variant == "blended":
        calibrated_prob, market_override_used = _apply_market_override(
            calibrated_prob, metaculus_p, polymarket_p
        )
        calibrated_prob = max(0.01, min(0.99, calibrated_prob))

    lo, hi, ci_method = _conformal_ci(calibrated_prob)
    answer = "YES" if calibrated_prob >= 0.5 else "NO"

    return {
        "baseline_version": BASELINE_VERSION,
        "variant": variant,
        "raw_prob": round(raw_prob, 4),
        "calibrated_prob": round(calibrated_prob, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "ci_method": ci_method,
        "answer": answer,
        "untrained": untrained,
        "market_override": market_override_used,
    }

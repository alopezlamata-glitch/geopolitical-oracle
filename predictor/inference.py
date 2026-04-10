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


def _conformal_ci(calibrated_prob: float, coverage: float = 0.80) -> tuple[float, float, str]:
    """
    Marginal conformal prediction interval.
    Uses saved nonconformity scores from calibration validation set.
    Falls back to beta heuristic if scores unavailable.
    Returns (lo, hi, method_name).
    """
    conformal_path = Path(__file__).parent.parent / "data" / "model" / "conformal_scores.json"
    if conformal_path.exists():
        try:
            data = json.loads(conformal_path.read_text())
            scores = data["scores"]
            n = len(scores)
            if n >= 5:
                # Conformal quantile: ceil((n+1)*(1-alpha))/n
                alpha = 1.0 - coverage
                q_level = math.ceil((n + 1) * (1 - alpha)) / n
                q_level = min(q_level, 1.0)
                sorted_scores = sorted(scores)
                idx = min(int(q_level * n), n - 1)
                margin = sorted_scores[idx]
                lo = max(0.01, calibrated_prob - margin)
                hi = min(0.99, calibrated_prob + margin)
                return lo, hi, "conformal"
        except Exception:
            pass
    # Fallback: beta heuristic with honest effective N
    eff_n = 20  # conservative
    alpha_p = calibrated_prob * eff_n
    beta_p = (1 - calibrated_prob) * eff_n
    lo, hi = scipy.stats.beta.interval(coverage, max(0.1, alpha_p), max(0.1, beta_p))
    return float(lo), float(hi), "heuristic"


def predict(features: dict[str, float]) -> dict:
    """
    Returns prediction dict with: raw_prob, calibrated_prob, ci_lo, ci_hi, answer, untrained.
    """
    model, feature_names = load_model()
    calibrator = load_calibrator()
    untrained = model is None

    feat_names = feature_names or get_feature_names()
    x = np.array([[float(features.get(f, 0.0)) for f in feat_names]], dtype=np.float32)

    if untrained:
        # No model: use market signals as prior, else 0.5
        meta_p = features.get("metaculus_p", -1.0)
        poly_p = features.get("polymarket_p", -1.0)
        valid = [p for p in [meta_p, poly_p] if 0 < p < 1]
        raw_prob = float(np.mean(valid)) if valid else 0.5
    else:
        raw_prob = float(model.predict_proba(x)[0, 1])

    if calibrator is not None and not untrained:
        calibrated_prob = float(calibrator.predict([raw_prob])[0])
    else:
        calibrated_prob = raw_prob

    # Clamp
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
    }

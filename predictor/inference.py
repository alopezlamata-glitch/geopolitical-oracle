from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.stats

from model.trainer import load_model
from model.calibrator import load_calibrator
from features.builder import get_feature_names

logger = logging.getLogger(__name__)


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

    # Beta CI
    alpha_p = calibrated_prob * 20
    beta_p = (1 - calibrated_prob) * 20
    lo, hi = scipy.stats.beta.interval(0.80, alpha_p, beta_p)

    answer = "YES" if calibrated_prob >= 0.5 else "NO"

    return {
        "raw_prob": round(raw_prob, 4),
        "calibrated_prob": round(calibrated_prob, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "answer": answer,
        "untrained": untrained,
    }

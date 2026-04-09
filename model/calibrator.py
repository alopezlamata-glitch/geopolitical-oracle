from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "model"
_CALIBRATOR_PATH = _MODEL_DIR / "calibrator.pkl"
_CALIB_LOG_PATH = _MODEL_DIR / "calibration_log.json"


class PlattWrapper:
    """Logistic regression calibrator accepting raw probabilities as input."""
    def __init__(self, lr_model):
        self.lr = lr_model

    def predict(self, probs):
        eps = 1e-6
        p = np.clip(np.array(probs), eps, 1 - eps)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return self.lr.predict_proba(logits)[:, 1]


def calibrate(model, X_val: np.ndarray, y_val: np.ndarray, feature_names: list[str]) -> None:
    """
    Platt scaling (logistic regression on raw scores) — stable on small N.
    Falls back to isotonic only when N >= 100.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        logger.error("calibrator: scikit-learn not installed")
        return

    raw_probs = model.predict_proba(X_val)[:, 1]
    n = len(raw_probs)

    if n >= 100:
        # Isotonic is more powerful but needs data
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, y_val)
        calibrator = iso
        method = "isotonic"
    else:
        # Platt scaling: logistic regression on logit(raw_prob)
        eps = 1e-6
        logits = np.log(np.clip(raw_probs, eps, 1 - eps) /
                        (1 - np.clip(raw_probs, eps, 1 - eps))).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(logits, y_val)

        calibrator = PlattWrapper(lr)
        method = "platt"

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CALIBRATOR_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info("calibrator: saved (%s) to %s", method, _CALIBRATOR_PATH)

    # Compute ECE over 10 bins
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bucket_log = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (raw_probs >= lo) & (raw_probs < hi)
        if mask.sum() == 0:
            continue
        bucket_p = raw_probs[mask].mean()
        bucket_outcome = y_val[mask].mean()
        bucket_n = mask.sum()
        ece += (bucket_n / n) * abs(bucket_p - bucket_outcome)
        bucket_log.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "count": int(bucket_n),
            "mean_pred": round(float(bucket_p), 4),
            "mean_actual": round(float(bucket_outcome), 4),
        })

    log = {"ece": round(ece, 4), "n_val": n, "buckets": bucket_log}
    _CALIB_LOG_PATH.write_text(json.dumps(log, indent=2))
    logger.info("calibrator: ECE = %.4f on %d validation examples", ece, n)


def load_calibrator():
    """Load saved isotonic calibrator. Returns None if not available."""
    if not _CALIBRATOR_PATH.exists():
        return None
    try:
        with open(_CALIBRATOR_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("calibrator: load failed: %s", e)
        return None

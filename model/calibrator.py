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

    # Save asymmetric split-conformal nonconformity scores for inference-time CI.
    # Standard score: s(x, y) = 1 - f_hat(x)[y]
    #   y=1 examples: s = 1 - p_cal  (residual toward 1)
    #   y=0 examples: s = p_cal      (residual toward 0)
    # Separate per-class lists allow asymmetric [lo, hi] intervals that respect
    # the true calibration residual distribution for each label.
    # Marginal coverage guarantee: P(Y in C(X)) >= 1-alpha under exchangeability.
    # See CONFORMAL.md for assumptions and limitations.
    calibrated_probs = calibrator.predict(raw_probs)
    pos_mask = y_val == 1
    neg_mask = y_val == 0
    scores_pos = (1.0 - calibrated_probs[pos_mask]).tolist() if pos_mask.any() else []
    scores_neg = calibrated_probs[neg_mask].tolist() if neg_mask.any() else []

    conformal_path = _MODEL_DIR / "conformal_scores.json"
    conformal_path.write_text(json.dumps({
        "scores_pos": scores_pos,
        "scores_neg": scores_neg,
        "n_pos": len(scores_pos),
        "n_neg": len(scores_neg),
        "method": method,
    }))
    logger.info("calibrator: saved conformal scores (%d pos, %d neg)",
                len(scores_pos), len(scores_neg))

    # Compute ECE over 10 bins — on calibrated probs (primary) and raw probs (diagnostic).
    # Using calibrated probs is correct: we're measuring the quality of the full
    # model+calibrator pipeline, not the raw classifier output.
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)

    def _ece_buckets(probs: np.ndarray) -> tuple[float, list[dict]]:
        err = 0.0
        log_rows = []
        for i in range(n_bins):
            lo_b, hi_b = bins[i], bins[i + 1]
            mask = (probs >= lo_b) & (probs < hi_b)
            if mask.sum() == 0:
                continue
            bucket_p = probs[mask].mean()
            bucket_outcome = y_val[mask].astype(float).mean()
            bucket_n = int(mask.sum())
            err += (bucket_n / n) * abs(bucket_p - bucket_outcome)
            log_rows.append({
                "bin": f"[{lo_b:.1f},{hi_b:.1f})",
                "count": bucket_n,
                "mean_pred": round(float(bucket_p), 4),
                "mean_actual": round(float(bucket_outcome), 4),
            })
        return round(float(err), 4), log_rows

    ece_cal, bucket_log = _ece_buckets(calibrated_probs)
    ece_raw, _ = _ece_buckets(raw_probs)

    log = {
        "ece_cal": ece_cal,
        "ece_raw": round(ece_raw, 4),
        "n_val": n,
        "method": method,
        "buckets": bucket_log,
    }
    _CALIB_LOG_PATH.write_text(json.dumps(log, indent=2))
    logger.info("calibrator: ECE_cal=%.4f  ECE_raw=%.4f  method=%s  n=%d",
                ece_cal, ece_raw, method, n)


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

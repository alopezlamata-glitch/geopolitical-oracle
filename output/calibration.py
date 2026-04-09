from __future__ import annotations

import glob
import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MIN_RESOLVED = 30
_PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"


def load_resolved_predictions(data_dir: Path = _PREDICTIONS_DIR) -> tuple[list[float], list[int]]:
    """Load all predictions with known outcomes."""
    probs: list[float] = []
    outcomes: list[int] = []

    for path in sorted(data_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("calibration: could not read %s: %s", path.name, e)
            continue

        if not record.get("resolved"):
            continue
        outcome = record.get("outcome")
        if outcome is None:
            continue

        p = record.get("final_probability")
        if p is None:
            continue

        try:
            probs.append(float(p))
            outcomes.append(int(outcome))
        except (TypeError, ValueError):
            continue

    return probs, outcomes


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error between probabilities and binary outcomes."""
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def run_calibration(data_dir: Path = _PREDICTIONS_DIR) -> None:
    """
    Show calibration statistics on resolved predictions.
    Fits Platt scaling (logistic regression) when >= 30 resolved predictions exist.
    """
    probs, outcomes = load_resolved_predictions(data_dir)
    n = len(probs)

    print(f"\n{'─'*50}")
    print(f"  CALIBRATION REPORT")
    print(f"{'─'*50}")
    print(f"  Resolved predictions: {n}")

    if n == 0:
        print("  No resolved predictions found.")
        print(f"{'─'*50}\n")
        return

    bs = brier_score(probs, outcomes)
    print(f"  Brier score:          {bs:.4f}  (lower = better, 0.25 = random)")

    # Calibration buckets
    print(f"\n  Calibration buckets (predicted vs actual):")
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print(f"  {'Bucket':<14} {'N':>4} {'Pred':>8} {'Actual':>8}")
    print(f"  {'─'*38}")
    for lo, hi in buckets:
        bucket_probs = [p for p, o in zip(probs, outcomes) if lo <= p < hi]
        bucket_outcomes = [o for p, o in zip(probs, outcomes) if lo <= p < hi]
        if not bucket_probs:
            continue
        mean_pred = sum(bucket_probs) / len(bucket_probs)
        mean_actual = sum(bucket_outcomes) / len(bucket_outcomes)
        label = f"[{lo:.0%}, {hi:.0%})"
        print(f"  {label:<14} {len(bucket_probs):>4} {mean_pred:>7.1%} {mean_actual:>7.1%}")

    if n < _MIN_RESOLVED:
        print(f"\n  Platt scaling requires >= {_MIN_RESOLVED} resolved predictions.")
        print(f"  ({_MIN_RESOLVED - n} more needed to activate calibration correction.)")
        print(f"{'─'*50}\n")
        return

    # Platt scaling via logistic regression
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        X = np.array([math.log(p / (1 - p)) for p in probs]).reshape(-1, 1)
        y = np.array(outcomes)

        clf = LogisticRegression()
        clf.fit(X, y)

        coef = clf.coef_[0][0]
        intercept = clf.intercept_[0]
        print(f"\n  Platt scaling fit (logit space):")
        print(f"    slope:     {coef:.4f}  (1.0 = perfectly calibrated)")
        print(f"    intercept: {intercept:.4f}  (0.0 = perfectly calibrated)")

        if abs(coef - 1.0) < 0.1 and abs(intercept) < 0.1:
            print("    Status: well-calibrated ✓")
        elif coef < 1.0:
            print("    Status: overconfident (predictions too extreme)")
        else:
            print("    Status: underconfident (predictions too conservative)")

    except ImportError:
        print("\n  scikit-learn not available — skipping Platt scaling.")

    print(f"{'─'*50}\n")


def apply_calibration(p: float, data_dir: Path = _PREDICTIONS_DIR) -> tuple[float, bool]:
    """
    Apply Platt scaling correction if >= 30 resolved predictions exist.
    Returns (corrected_p, was_applied).
    """
    probs, outcomes = load_resolved_predictions(data_dir)
    if len(probs) < _MIN_RESOLVED:
        return p, False

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        X = np.array([math.log(q / (1 - q)) for q in probs]).reshape(-1, 1)
        y = np.array(outcomes)

        clf = LogisticRegression()
        clf.fit(X, y)

        p_clamped = max(0.001, min(0.999, p))
        logit_p = math.log(p_clamped / (1 - p_clamped))
        corrected = clf.predict_proba([[logit_p]])[0][1]
        return float(corrected), True

    except (ImportError, Exception) as e:
        logger.warning("calibration: could not apply correction: %s", e)
        return p, False

"""
Per-domain isotonic calibrators for the base-rate predictor.

Files: data/model/calibrator_{event_family}.pkl
Format: sklearn IsotonicRegression pickled with joblib.

If no calibrator exists for a domain, identity function is returned
(the model is self-calibrating via the base rate + weight choice).

Calibrators are fitted by scripts/fit_domain_calibrator.py using
labeled examples from training_ready_snapshots filtered by event_family.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "model"


@lru_cache(maxsize=8)
def _load_calibrator(event_family: str):
    """Load per-domain calibrator, cached. Returns None if not found."""
    path = _MODEL_DIR / f"calibrator_{event_family}.pkl"
    if not path.exists():
        return None
    try:
        import joblib
        cal = joblib.load(path)
        logger.info("domain_calibrator: loaded calibrator for '%s' from %s", event_family, path)
        return cal
    except Exception as e:
        logger.warning("domain_calibrator: failed to load '%s': %s", event_family, e)
        return None


def apply_domain_calibrator(raw_prob: float, event_family: str) -> float:
    """
    Apply domain-specific isotonic calibration.
    Returns raw_prob unchanged if no calibrator exists for this domain.
    """
    cal = _load_calibrator(event_family)
    if cal is None:
        return raw_prob
    try:
        result = float(cal.predict([raw_prob])[0])
        return max(0.01, min(0.99, result))
    except Exception as e:
        logger.warning("domain_calibrator: prediction failed for '%s': %s", event_family, e)
        return raw_prob


def save_domain_calibrator(calibrator, event_family: str) -> Path:
    """Save a fitted calibrator to disk and invalidate cache."""
    import joblib
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / f"calibrator_{event_family}.pkl"
    joblib.dump(calibrator, path)
    _load_calibrator.cache_clear()
    logger.info("domain_calibrator: saved calibrator for '%s' to %s", event_family, path)
    return path


def list_fitted_domains() -> list[str]:
    """Return list of domains that have a fitted calibrator on disk."""
    if not _MODEL_DIR.exists():
        return []
    return [
        p.stem.replace("calibrator_", "")
        for p in _MODEL_DIR.glob("calibrator_*.pkl")
    ]

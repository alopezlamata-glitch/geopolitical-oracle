from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "model"
_BASELINE_PATH = _MODEL_DIR / "feature_baseline.json"
_DRIFT_LOG_PATH = _MODEL_DIR / "drift_log.json"

_PSI_MONITOR = 0.10
_PSI_DRIFT = 0.25
_N_BINS = 5


def _psi_continuous(actual_val: float, baseline: dict) -> float:
    """Compute PSI for a single continuous feature value vs baseline distribution."""
    mean = baseline.get("mean", 0.0)
    std = baseline.get("std", 1.0) or 1.0
    values = baseline.get("values", [])
    if not values:
        return 0.0

    # Build baseline histogram
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return 0.0
    bin_edges = [lo + i * (hi - lo) / _N_BINS for i in range(_N_BINS + 1)]

    def _bucket_idx(v: float) -> int:
        for i in range(_N_BINS):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                return i
        return _N_BINS - 1

    expected_counts = [0] * _N_BINS
    for v in values:
        expected_counts[_bucket_idx(v)] += 1
    n_exp = len(values)
    expected_pct = [c / n_exp for c in expected_counts]

    actual_counts = [0] * _N_BINS
    actual_counts[_bucket_idx(actual_val)] += 1
    actual_pct = [c / 1.0 for c in actual_counts]

    psi = 0.0
    for ep, ap in zip(expected_pct, actual_pct):
        ep = max(ep, 1e-6)
        ap = max(ap, 1e-6)
        psi += (ap - ep) * math.log(ap / ep)
    return psi


def detect_drift(features: dict[str, float]) -> list[str]:
    """
    Compare current feature values to training baseline.
    Returns list of drifted feature names (PSI > 0.25).
    """
    if not _BASELINE_PATH.exists():
        return []

    try:
        baseline = json.loads(_BASELINE_PATH.read_text())
    except Exception:
        return []

    drifted = []
    log_entries = []

    for fname, val in features.items():
        if fname not in baseline:
            continue
        psi = _psi_continuous(float(val), baseline[fname])
        status = "stable"
        if psi > _PSI_DRIFT:
            status = "DRIFTED"
            drifted.append(fname)
        elif psi > _PSI_MONITOR:
            status = "monitor"
        log_entries.append({"feature": fname, "psi": round(psi, 4), "status": status})

    # Append to drift log
    log = []
    if _DRIFT_LOG_PATH.exists():
        try:
            log = json.loads(_DRIFT_LOG_PATH.read_text())
        except Exception:
            pass
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drifted_count": len(drifted),
        "features": log_entries,
    })
    _DRIFT_LOG_PATH.write_text(json.dumps(log[-50:], indent=2))  # keep last 50 runs

    if drifted:
        logger.warning("drift: %d features drifted: %s", len(drifted), drifted)

    return drifted

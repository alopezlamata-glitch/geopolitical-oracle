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
_HISTORY_PATH = _MODEL_DIR / "feature_history.json"

_MIN_HISTORY = 10      # need this many predictions before drift is meaningful
_WINDOW = 10           # compare last N predictions vs baseline
_Z_THRESHOLD = 2.0     # flag if batch mean drifts > 2 std from train mean


def _load_history() -> list[dict]:
    if not _HISTORY_PATH.exists():
        return []
    try:
        return json.loads(_HISTORY_PATH.read_text())
    except Exception:
        return []


def _save_history(history: list[dict]) -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # keep last 200 to bound file size
    _HISTORY_PATH.write_text(json.dumps(history[-200:], indent=2))


def detect_drift(features: dict[str, float]) -> list[str]:
    """
    Append current features to history. When >= MIN_HISTORY entries exist,
    compare last-WINDOW batch mean vs training baseline using z-score.
    Returns list of feature names that have drifted (|z| > Z_THRESHOLD).
    """
    if not _BASELINE_PATH.exists():
        return []

    try:
        baseline = json.loads(_BASELINE_PATH.read_text())
    except Exception:
        return []

    # Append to history
    history = _load_history()
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {k: float(v) for k, v in features.items()},
    })
    _save_history(history)

    if len(history) < _MIN_HISTORY:
        logger.info("drift: insufficient history (%d/%d predictions)", len(history), _MIN_HISTORY)
        return []

    # Use last WINDOW predictions as current batch
    batch = history[-_WINDOW:]
    drifted = []
    log_entries = []

    for fname, base_stats in baseline.items():
        train_mean = base_stats.get("mean", 0.0)
        train_std = base_stats.get("std", 1.0) or 1.0

        batch_vals = [e["features"].get(fname, 0.0) for e in batch if fname in e.get("features", {})]
        if not batch_vals:
            continue

        batch_mean = sum(batch_vals) / len(batch_vals)
        z = abs(batch_mean - train_mean) / (train_std + 1e-6)

        status = "DRIFTED" if z > _Z_THRESHOLD else "stable"
        if status == "DRIFTED":
            drifted.append(fname)

        log_entries.append({
            "feature": fname,
            "z_score": round(z, 3),
            "batch_mean": round(batch_mean, 4),
            "train_mean": round(train_mean, 4),
            "status": status,
        })

    # Append to drift log (schema_version ensures forward-compatible parsing)
    drift_log = []
    if _DRIFT_LOG_PATH.exists():
        try:
            drift_log = json.loads(_DRIFT_LOG_PATH.read_text())
        except Exception:
            pass
    drift_log.append({
        "schema_version": "1.0",
        "method": "z_score",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_history": len(history),
        "drifted_count": len(drifted),
        "features": log_entries,
    })
    _DRIFT_LOG_PATH.write_text(json.dumps(drift_log[-50:], indent=2))

    if drifted:
        logger.warning("drift: %d features drifted (z>%.1f): %s", len(drifted), _Z_THRESHOLD, drifted[:5])

    return drifted

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "model"
_TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
_MIN_EXAMPLES = 30

_FEATURE_NAMES_PATH = _MODEL_DIR / "feature_names.json"
_MODEL_PATH = _MODEL_DIR / "xgb_model.json"
_CALIBRATOR_PATH = _MODEL_DIR / "calibrator.pkl"
_BASELINE_PATH = _MODEL_DIR / "feature_baseline.json"


def load_training_data_from_lakehouse() -> tuple[list[dict], list[int]]:
    """
    Primary training path (DuckDB lakehouse).

    Reads snapshots that are already resolved, using persisted explicit_features.
    Returns (X_dicts, y).
    """
    from data_layer.db import get_db, init_schema

    init_schema()
    db = get_db()

    rows = db.execute(
        """
        SELECT
            trs.explicit_features,
            CAST(trs.outcome AS INTEGER) AS outcome
        FROM training_ready_snapshots trs
        ORDER BY trs.as_of_time ASC
        """
    ).fetchall()

    X, y = [], []
    for raw_features, outcome in rows:
        try:
            features = raw_features
            if isinstance(raw_features, str):
                features = json.loads(raw_features)
            if not isinstance(features, dict):
                continue
            X.append(features)
            y.append(int(outcome))
        except Exception as e:
            logger.warning("trainer: skipping malformed lakehouse row: %s", e)
    return X, y


def load_training_data_legacy() -> tuple[list[dict], list[int]]:
    """Legacy fallback: load labeled examples from local JSON files."""
    X, y = [], []
    if not _TRAINING_DIR.exists():
        return X, y
    for f in sorted(_TRAINING_DIR.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
            if "outcome" not in rec or "features" not in rec:
                continue
            X.append(rec["features"])
            y.append(int(rec["outcome"]))
        except Exception as e:
            logger.warning("trainer: skipping legacy %s: %s", f.name, e)
    return X, y


def load_training_data() -> tuple[list[dict], list[int]]:
    """
    Load labeled training examples.

    Primary source: DuckDB lakehouse (`training_ready_snapshots`).
    Fallback: `data/training/*.json` when lakehouse has < MIN_EXAMPLES
              (covers ICEWS historical examples generated offline).
    """
    try:
        X, y = load_training_data_from_lakehouse()
        logger.info("trainer: lakehouse has %d examples", len(X))
        if len(X) >= _MIN_EXAMPLES:
            return X, y
        # Fall through to legacy when lakehouse is not yet populated
        logger.info(
            "trainer: lakehouse has %d examples (< %d minimum) — "
            "merging with legacy ICEWS training set",
            len(X), _MIN_EXAMPLES,
        )
        legacy_X, legacy_y = load_training_data_legacy()
        merged_X = legacy_X + X
        merged_y = legacy_y + y
        logger.info(
            "trainer: total %d examples (%d legacy + %d lakehouse)",
            len(merged_X), len(legacy_X), len(X),
        )
        return merged_X, merged_y
    except Exception as e:
        logger.warning("trainer: lakehouse unavailable, falling back to legacy JSON: %s", e)
        return load_training_data_legacy()


def _to_matrix(X_dicts: list[dict], feature_names: list[str]) -> np.ndarray:
    rows = []
    for d in X_dicts:
        rows.append([float(d.get(f, 0.0)) for f in feature_names])
    return np.array(rows, dtype=np.float32)


def train() -> bool:
    """Train XGBoost model. Returns True if training succeeded."""
    from features.builder import get_feature_names

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("trainer: xgboost not installed. Run: pip install xgboost")
        return False

    X_dicts, y = load_training_data()
    n = len(X_dicts)
    logger.info("trainer: loaded %d labeled examples", n)

    if n < _MIN_EXAMPLES:
        logger.warning("trainer: need %d examples, have %d — skipping training", _MIN_EXAMPLES, n)
        return False

    feature_names = get_feature_names()
    X = _to_matrix(X_dicts, feature_names)
    y_arr = np.array(y, dtype=np.int32)

    # Temporal split: oldest 80% train, newest 20% validate
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_arr[:split], y_arr[split:]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))
    logger.info("trainer: class balance — %d YES, %d NO, scale_pos_weight=%.2f",
                n_pos, n_neg, scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(_MODEL_PATH))
    _FEATURE_NAMES_PATH.write_text(json.dumps(feature_names))
    logger.info("trainer: model saved to %s", _MODEL_PATH)

    baseline = {}
    for i, fname in enumerate(feature_names):
        col = X_train[:, i]
        baseline[fname] = {"mean": float(col.mean()), "std": float(col.std()), "values": col.tolist()}
    _BASELINE_PATH.write_text(json.dumps(baseline))

    from model.calibrator import calibrate
    calibrate(model, X_val, y_val, feature_names)

    return True


def load_model():
    """Load trained XGBoost model and feature names. Returns (model, feature_names) or (None, [])."""
    try:
        import xgboost as xgb
    except ImportError:
        return None, []

    if not _MODEL_PATH.exists():
        return None, []

    model = xgb.XGBClassifier()
    model.load_model(str(_MODEL_PATH))

    feature_names = json.loads(_FEATURE_NAMES_PATH.read_text()) if _FEATURE_NAMES_PATH.exists() else []
    return model, feature_names

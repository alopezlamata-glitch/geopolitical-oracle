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


def load_training_data() -> tuple[list[dict], list[int]]:
    """Load all labeled training examples from data/training/."""
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
            logger.warning("trainer: skipping %s: %s", f.name, e)
    return X, y


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

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
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

    # Save feature baseline (mean + std per feature) for drift detection
    baseline = {}
    for i, fname in enumerate(feature_names):
        col = X_train[:, i]
        baseline[fname] = {"mean": float(col.mean()), "std": float(col.std()), "values": col.tolist()}
    _BASELINE_PATH.write_text(json.dumps(baseline))

    # Calibrate
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

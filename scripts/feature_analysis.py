"""
SHAP-based feature importance analysis.

Loads the trained XGBoost model and a representative sample of inputs
(training data + recent predictions) then computes SHAP values to identify:
  - Top drivers: features with highest mean |SHAP|
  - Dead features: features with near-zero importance (candidates for pruning)
  - Direction: whether each feature moves probability in the expected direction
  - Stability: variance of SHAP values across examples

Usage:
  python scripts/feature_analysis.py
  python scripts/feature_analysis.py --top 20 --threshold 0.001
  python main.py feature-analysis

Outputs: console table + data/model/feature_analysis.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_analysis")

_OUT_PATH = Path(__file__).parent.parent / "data" / "model" / "feature_analysis.json"


def _load_samples() -> tuple[list[dict], list[int]]:
    """Load labeled feature vectors from lakehouse (primary) + legacy files (fallback)."""
    X, y = [], []

    # Primary: lakehouse training snapshots
    try:
        from data_layer.db import get_db, init_schema, table_exists
        init_schema()
        if table_exists("training_ready_snapshots"):
            db = get_db()
            rows = db.execute(
                "SELECT explicit_features, outcome FROM training_ready_snapshots"
            ).fetchall()
            for raw_feats, outcome in rows:
                feats = raw_feats if isinstance(raw_feats, dict) else json.loads(raw_feats or "{}")
                if feats:
                    X.append(feats)
                    y.append(int(outcome))
            logger.info("loaded %d labeled samples from lakehouse", len(X))
    except Exception as e:
        logger.debug("lakehouse load failed: %s", e)

    # Fallback: legacy training JSON files
    training_dir = Path(__file__).parent.parent / "data" / "training"
    if training_dir.exists():
        for p in sorted(training_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text())
                feats = rec.get("features", {})
                outcome = rec.get("outcome")
                if feats and outcome is not None:
                    X.append(feats)
                    y.append(int(outcome))
            except Exception:
                pass
        logger.info("loaded %d total samples (including legacy)", len(X))

    return X, y


def run(top_n: int = 25, dead_threshold: float = 0.001, save: bool = True) -> dict:
    try:
        import numpy as np
    except ImportError:
        print("numpy required: pip install numpy")
        sys.exit(1)

    # Load model
    try:
        from model.trainer import load_model
        model, feature_names = load_model()
    except Exception as e:
        print(f"Could not load model: {e}")
        return {}

    if model is None:
        print("Model is untrained. Run: python main.py seed-training-data && python main.py train")
        return {"status": "untrained"}

    feature_names = feature_names or []
    logger.info("model loaded, %d features", len(feature_names))

    # Load samples
    X_dicts, y = _load_samples()
    if not X_dicts:
        print("No labeled samples found. Run: python main.py seed-training-data")
        return {"status": "no_data"}

    # Build feature matrix
    X = np.array([
        [float(d.get(f, 0.0)) for f in feature_names]
        for d in X_dicts
    ], dtype=np.float32)
    y_arr = np.array(y)
    logger.info("feature matrix: %d × %d", X.shape[0], X.shape[1])

    # Compute SHAP values
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (positive)
        logger.info("SHAP computed: %s", shap_values.shape)
    except ImportError:
        # Fallback: use XGBoost's built-in feature importance
        logger.warning("shap not installed — using XGBoost gain importance")
        import_dict = model.get_score(importance_type="gain")
        shap_values = None
    except Exception as e:
        logger.warning("SHAP failed (%s) — using XGBoost gain importance", e)
        import_dict = model.get_score(importance_type="gain")
        shap_values = None

    # Build per-feature stats
    stats = []
    if shap_values is not None:
        import numpy as np
        for i, fname in enumerate(feature_names):
            col = shap_values[:, i]
            mean_abs = float(np.mean(np.abs(col)))
            mean_shap = float(np.mean(col))
            std_shap  = float(np.std(col))
            # Pearson correlation of feature value with its SHAP value
            # Positive = feature pushes prob up when high (expected direction)
            feat_col = X[:, i]
            corr = float(np.corrcoef(feat_col, col)[0, 1]) if feat_col.std() > 0 else 0.0
            stats.append({
                "feature":   fname,
                "mean_abs_shap": round(mean_abs, 6),
                "mean_shap":     round(mean_shap, 6),
                "std_shap":      round(std_shap, 6),
                "direction":     "+" if corr >= 0 else "-",
                "corr_feat_shap": round(corr, 4),
                "dead": mean_abs < dead_threshold,
            })
    else:
        # XGBoost fallback
        for fname in feature_names:
            gain = float(import_dict.get(f"f{feature_names.index(fname)}", 0.0))
            stats.append({
                "feature":   fname,
                "mean_abs_shap": round(gain, 6),
                "mean_shap":     0.0,
                "std_shap":      0.0,
                "direction":     "?",
                "corr_feat_shap": 0.0,
                "dead": gain < dead_threshold,
            })

    # Sort by importance
    stats.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    dead = [s for s in stats if s["dead"]]
    alive = [s for s in stats if not s["dead"]]

    result = {
        "n_features":  len(feature_names),
        "n_samples":   len(X_dicts),
        "n_dead":      len(dead),
        "dead_threshold": dead_threshold,
        "top_features": stats[:top_n],
        "dead_features": [s["feature"] for s in dead],
        "all_features":  stats,
    }

    if save:
        _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OUT_PATH.write_text(json.dumps(result, indent=2))
        logger.info("saved feature analysis to %s", _OUT_PATH)

    _print_report(result, top_n)
    return result


def _print_report(r: dict, top_n: int) -> None:
    W = 70

    def row(s): return f"│ {s:<{W}} │"
    def sep(): return "├─" + "─" * W + "─┤"

    lines = [
        "┌─" + "─" * W + "─┐",
        row("  FEATURE IMPORTANCE ANALYSIS (SHAP)"),
        row(f"  {r['n_features']} features | {r['n_samples']} labeled samples "
            f"| {r['n_dead']} dead (< {r['dead_threshold']})"),
        sep(),
        row(f"  {'feature':<32} {'mean|SHAP|':>10}  {'std':>8}  {'dir':>4}  {'corr':>6}"),
        sep(),
    ]

    for s in r["top_features"]:
        flag = " DEAD" if s["dead"] else ""
        lines.append(row(
            f"  {s['feature']:<32} "
            f"{s['mean_abs_shap']:>10.5f}  "
            f"{s['std_shap']:>8.4f}  "
            f"{s['direction']:>4}  "
            f"{s['corr_feat_shap']:>6.3f}"
            f"{flag}"
        ))

    if r["dead_features"]:
        lines.append(sep())
        lines.append(row(f"  DEAD features (prune candidates): {len(r['dead_features'])}"))
        for i in range(0, len(r["dead_features"]), 3):
            chunk = "  ".join(r["dead_features"][i:i+3])
            lines.append(row(f"    {chunk}"))

    lines.append("└─" + "─" * W + "─┘")
    print("\n".join(lines))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SHAP feature importance analysis")
    parser.add_argument("--top",       type=int,   default=25)
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Mean |SHAP| below this → dead feature")
    parser.add_argument("--no-save",   action="store_true")
    args = parser.parse_args()
    run(top_n=args.top, dead_threshold=args.threshold, save=not args.no_save)


if __name__ == "__main__":
    main()

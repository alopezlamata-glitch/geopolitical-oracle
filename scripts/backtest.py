"""
Walk-forward temporal backtest with proper scoring rules.

Evaluates whether predicted probabilities are calibrated across time by running
held-out prediction on rolling temporal folds — the correct evaluation strategy
for time-series data where future outcomes cannot inform past predictions.

Two fold strategies:
  within-era  — rolling 3-month folds within 2012-2013 (train ≥100 examples)
  cross-era   — train on each accumulated era, test on next era

Metrics per fold and aggregate:
  Brier score / Brier skill score  (proper scoring rule)
  ROC-AUC
  ECE (10 bins, weighted)
  Conformal CI coverage (asymmetric, from held-out nonconformity scores)
  Reliability diagram (ASCII)

Output: data/model/backtest_results.json

Run: python scripts/backtest.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

_TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "model" / "backtest_results.json"
_MIN_TRAIN = 100
_CALIB_FRAC = 0.20   # fraction of training fold used for Platt calibration


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_examples() -> list[dict]:
    examples = []
    from features.builder import get_feature_names
    feat_names = get_feature_names()
    for f in sorted(_TRAINING_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if "outcome" not in d or "features" not in d or "timestamp" not in d:
                continue
            feats = [float(d["features"].get(fn, 0.0)) for fn in feat_names]
            examples.append({
                "ts": d["timestamp"][:10],
                "outcome": int(d["outcome"]),
                "features": np.array(feats, dtype=np.float32),
                "source": d.get("source", "unknown"),
            })
        except Exception:
            continue
    examples.sort(key=lambda e: e["ts"])
    return examples


# ─── Model training (mini, per-fold) ─────────────────────────────────────────

def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple:
    """Train XGBoost + Platt calibrator on fold. Returns (model, calibrator)."""
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = max(1.0, n_neg / max(n_pos, 1))

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    # Platt calibration on calibration split (stable for small N)
    raw_cal = model.predict_proba(X_cal)[:, 1]
    eps = 1e-6
    logits = np.log(np.clip(raw_cal, eps, 1 - eps) /
                    (1 - np.clip(raw_cal, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, y_cal)

    return model, lr


def _predict_fold(
    model,
    calibrator,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_probs, calibrated_probs) for test examples."""
    raw = model.predict_proba(X_test)[:, 1]
    eps = 1e-6
    logits = np.log(np.clip(raw, eps, 1 - eps) /
                    (1 - np.clip(raw, eps, 1 - eps))).reshape(-1, 1)
    cal = calibrator.predict_proba(logits)[:, 1]
    return raw, cal


# ─── Conformal coverage ───────────────────────────────────────────────────────

def _conformal_coverage(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    coverage: float = 0.80,
) -> dict:
    """
    Compute asymmetric conformal coverage on test fold.
    Nonconformity scores from calibration split — correct split-conformal setup.
    Covered = true label is included in the conformal prediction set.
    """
    pos_mask = y_cal == 1
    neg_mask = y_cal == 0
    scores_pos = (1.0 - p_cal[pos_mask]).tolist() if pos_mask.any() else []
    scores_neg = p_cal[neg_mask].tolist() if neg_mask.any() else []

    if len(scores_pos) < 3 or len(scores_neg) < 3:
        return {"coverage": None, "n_test": int(len(y_test))}

    def _q(scores: list[float]) -> float:
        n = len(scores)
        k = min(math.ceil((n + 1) * (1.0 - (1.0 - coverage))), n)
        return sorted(scores)[k - 1]

    q_pos = _q(scores_pos)  # upper margin: 1-p ≤ q_pos → p ≥ 1-q_pos
    q_neg = _q(scores_neg)  # lower margin: p ≤ q_neg

    covered = sum(
        (1 - p <= q_pos) if y == 1 else (p <= q_neg)
        for p, y in zip(p_test, y_test)
    )
    return {
        "coverage": round(covered / len(y_test), 4) if len(y_test) > 0 else None,
        "n_test": int(len(y_test)),
        "q_pos": round(q_pos, 4),
        "q_neg": round(q_neg, 4),
    }


# ─── Scoring metrics ──────────────────────────────────────────────────────────

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y.astype(float)) ** 2))


def _brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    base_rate = float(y.mean())
    b_clim = base_rate * (1 - base_rate)   # climatological Brier score
    if b_clim == 0:
        return 0.0
    return round(1.0 - _brier(y, p) / b_clim, 4)


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return round(float(roc_auc_score(y, p)), 4)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].astype(float).mean())
    return round(float(ece), 4)


def _reliability_diagram(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        rows.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": int(mask.sum()),
            "mean_pred": round(float(p[mask].mean()), 3),
            "mean_actual": round(float(y[mask].astype(float).mean()), 3),
        })
    return rows


# ─── Fold builders ────────────────────────────────────────────────────────────

def _within_era_folds(
    examples: list[dict],
    fold_months: int = 3,
) -> list[tuple[str, list[dict], list[dict]]]:
    """
    Rolling folds within 2012-2013 data.
    Each fold: train = all examples before fold_start, test = fold_start..fold_end.
    """
    era = [e for e in examples if e["ts"] < "2014-01-01"]
    if not era:
        return []

    start = datetime(2012, 1, 1)
    end = datetime(2014, 1, 1)
    folds = []
    t = start + timedelta(days=fold_months * 30)
    while t < end:
        fold_end = t + timedelta(days=fold_months * 30)
        train = [e for e in era if e["ts"] < t.strftime("%Y-%m-%d")]
        test = [e for e in era
                if t.strftime("%Y-%m-%d") <= e["ts"] < fold_end.strftime("%Y-%m-%d")]
        if len(train) >= _MIN_TRAIN and len(test) >= 5:
            label = f"within {t.strftime('%Y-%m')}..{fold_end.strftime('%Y-%m')}"
            folds.append((label, train, test))
        t = fold_end
    return folds


def _cross_era_folds(examples: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    """
    Cross-era folds: train on accumulated eras, test on the next era.
    Reveals temporal generalisation across the 7-year data gap.
    """
    eras = {
        "2012-2013": [e for e in examples if e["ts"] < "2014-01-01"],
        "2021":      [e for e in examples if "2021-01-01" <= e["ts"] < "2022-01-01"],
        "2022":      [e for e in examples if "2022-01-01" <= e["ts"] < "2023-01-01"],
    }
    folds = []
    # Fold 1: train 2012-2013 → test 2021
    train1 = eras["2012-2013"]
    test1 = eras["2021"]
    if len(train1) >= _MIN_TRAIN and len(test1) >= 5:
        folds.append(("cross: 2012-2013 → 2021", train1, test1))

    # Fold 2: train 2012-2013+2021 → test 2022
    train2 = eras["2012-2013"] + eras["2021"]
    test2 = eras["2022"]
    if len(train2) >= _MIN_TRAIN and len(test2) >= 5:
        folds.append(("cross: 2012-2021 → 2022", train2, test2))

    return folds


# ─── Run backtest ─────────────────────────────────────────────────────────────

def _run_folds(folds: list[tuple[str, list, list]]) -> list[dict]:
    results = []
    for label, train_ex, test_ex in folds:
        # Split training into fit + calibration
        n_cal = max(10, int(len(train_ex) * _CALIB_FRAC))
        fit_ex = train_ex[:-n_cal]
        cal_ex = train_ex[-n_cal:]

        X_fit = np.vstack([e["features"] for e in fit_ex])
        y_fit = np.array([e["outcome"] for e in fit_ex], dtype=np.int32)
        X_cal = np.vstack([e["features"] for e in cal_ex])
        y_cal = np.array([e["outcome"] for e in cal_ex], dtype=np.int32)
        X_test = np.vstack([e["features"] for e in test_ex])
        y_test = np.array([e["outcome"] for e in test_ex], dtype=np.int32)

        try:
            model, calibrator = _train_fold(X_fit, y_fit, X_cal, y_cal)
        except Exception as exc:
            logger.warning("fold '%s' training failed: %s", label, exc)
            continue

        _, p_cal = _predict_fold(model, calibrator, X_cal)
        _, p_test = _predict_fold(model, calibrator, X_test)

        base_rate = float(y_test.mean())
        conf = _conformal_coverage(y_cal, p_cal, y_test, p_test)

        result = {
            "fold": label,
            "n_train": len(train_ex),
            "n_cal": len(cal_ex),
            "n_test": len(test_ex),
            "base_rate": round(base_rate, 3),
            "brier": round(_brier(y_test, p_test), 4),
            "brier_skill": _brier_skill(y_test, p_test),
            "roc_auc": _roc_auc(y_test, p_test),
            "ece": _ece(y_test, p_test),
            "conformal_coverage_80": conf,
            "reliability": _reliability_diagram(y_test, p_test),
        }
        results.append(result)
        logger.info(
            "%-40s  n=%3d  brier=%.3f  bss=%.3f  auc=%.3f  ece=%.3f  cov=%s",
            label, len(test_ex),
            result["brier"], result["brier_skill"], result["roc_auc"], result["ece"],
            f'{conf["coverage"]:.0%}' if conf["coverage"] is not None else "n/a",
        )

    return results


# ─── Aggregate metrics ────────────────────────────────────────────────────────

def _aggregate(fold_results: list[dict]) -> dict:
    def _wmean(key: str) -> float | None:
        vals = [(r[key], r["n_test"]) for r in fold_results
                if r.get(key) is not None and not math.isnan(r[key])]
        if not vals:
            return None
        total_n = sum(n for _, n in vals)
        return round(sum(v * n for v, n in vals) / total_n, 4)

    all_y, all_p = [], []
    for r in fold_results:
        all_y.extend([r["base_rate"]] * r["n_test"])  # placeholder
    # Collect per-fold reliability rows for aggregate diagram
    bin_stats: dict[str, list] = defaultdict(list)
    for r in fold_results:
        for row in r.get("reliability", []):
            bin_stats[row["bin"]].append((row["n"], row["mean_pred"], row["mean_actual"]))

    agg_reliability = []
    for bname, rows in sorted(bin_stats.items()):
        total_n = sum(n for n, _, _ in rows)
        avg_pred = sum(n * p for n, p, _ in rows) / total_n
        avg_actual = sum(n * a for n, _, a in rows) / total_n
        agg_reliability.append({
            "bin": bname,
            "n": total_n,
            "mean_pred": round(avg_pred, 3),
            "mean_actual": round(avg_actual, 3),
        })

    conformal_coverages = [
        r["conformal_coverage_80"]["coverage"]
        for r in fold_results
        if r.get("conformal_coverage_80", {}).get("coverage") is not None
    ]
    avg_coverage = round(sum(conformal_coverages) / len(conformal_coverages), 4) \
        if conformal_coverages else None

    return {
        "n_folds": len(fold_results),
        "brier_weighted": _wmean("brier"),
        "brier_skill_weighted": _wmean("brier_skill"),
        "roc_auc_weighted": _wmean("roc_auc"),
        "ece_weighted": _wmean("ece"),
        "conformal_coverage_80_avg": avg_coverage,
        "conformal_target": 0.80,
        "reliability_diagram": agg_reliability,
    }


# ─── ASCII reliability diagram ────────────────────────────────────────────────

def _print_reliability(rows: list[dict], title: str = "Reliability diagram") -> None:
    W = 40
    print(f"\n  {title}")
    print(f"  {'Bin':>12}  {'N':>5}  {'Pred':>6}  {'Actual':>6}  {'Err':>5}  Chart")
    print("  " + "-" * 70)
    for row in rows:
        pred = row["mean_pred"]
        actual = row["mean_actual"]
        err = actual - pred
        bar_p = int(pred * W)
        bar_a = int(actual * W)
        lo, hi = min(bar_p, bar_a), max(bar_p, bar_a)
        bar = " " * lo + "=" * (hi - lo + 1)
        marker = "+" if err > 0 else "-"
        print(f"  {row['bin']:>12}  {row['n']:>5}  {pred:>6.3f}  {actual:>6.3f}  "
              f"{err:>+5.3f}  {bar}{marker}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading training examples...")
    examples = _load_examples()
    logger.info("Loaded %d examples (%s – %s)",
                len(examples), examples[0]["ts"], examples[-1]["ts"])

    logger.info("\n=== Within-era folds (2012-2013, 3-month rolling) ===")
    within = _within_era_folds(examples, fold_months=3)
    within_results = _run_folds(within)

    logger.info("\n=== Cross-era folds (temporal generalisation) ===")
    cross = _cross_era_folds(examples)
    cross_results = _run_folds(cross)

    all_results = within_results + cross_results
    if not all_results:
        logger.error("No folds produced results — check training data.")
        return

    agg = _aggregate(all_results)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": agg,
        "within_era": within_results,
        "cross_era": cross_results,
    }
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=True))
    logger.info("Results saved to %s", _OUTPUT_PATH)

    # Print summary
    print("\n" + "=" * 60)
    print("  BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Folds:              {agg['n_folds']}")
    print(f"  Brier score:        {agg['brier_weighted']:.4f}  "
          f"(skill: {agg['brier_skill_weighted']:+.4f})")
    print(f"  ROC-AUC:            {agg['roc_auc_weighted']:.4f}")
    print(f"  ECE:                {agg['ece_weighted']:.4f}")
    cov = agg['conformal_coverage_80_avg']
    cov_str = f"{cov:.1%}" if cov is not None else "n/a"
    print(f"  80% CI coverage:    {cov_str}  (target: 80.0%)")
    print()

    _print_reliability(agg["reliability_diagram"], "Aggregate reliability diagram")

    print("  Interpretation:")
    bss = agg["brier_skill_weighted"]
    if bss is not None:
        if bss > 0.1:
            print("  - Brier skill > 0.1: model beats base rate meaningfully")
        elif bss > 0:
            print("  - Brier skill marginally positive: slight improvement over base rate")
        else:
            print("  - Brier skill <= 0: model does not beat base rate — needs more data")
    auc = agg["roc_auc_weighted"]
    if auc is not None and not math.isnan(auc):
        if auc > 0.7:
            print("  - AUC > 0.70: reasonable discrimination")
        elif auc > 0.6:
            print("  - AUC 0.60-0.70: weak discrimination")
        else:
            print("  - AUC <= 0.60: poor discrimination")
    if cov is not None:
        gap = abs(cov - 0.80)
        if gap < 0.05:
            print("  - Conformal coverage within 5pp of target: intervals are valid")
        else:
            print(f"  - Conformal coverage off by {gap:.0%}: distribution shift detected")
    print()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "model" / "eval_baseline_v1.json"


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise ValueError(f"Unsupported timestamp type: {type(value)!r}")


def _as_numpy_probs(raw: Any, n_expected: int) -> np.ndarray:
    probs = np.asarray(raw, dtype=float).reshape(-1)
    if probs.shape[0] != n_expected:
        raise ValueError(
            f"predictor_fn returned {probs.shape[0]} probabilities, expected {n_expected}"
        )
    if np.any(~np.isfinite(probs)):
        raise ValueError("predictor_fn returned non-finite probabilities")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("predictor_fn returned probabilities outside [0, 1]")
    return probs


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_true))
    error = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += (float(mask.sum()) / total) * abs(confidence - accuracy)
    return float(error)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, ece_bins: int) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "n_examples": int(len(y_true)),
        "brier": float(np.mean((y_prob - y_true.astype(float)) ** 2)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": _ece(y_true, y_prob, n_bins=ece_bins),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
    return out


def _weighted_aggregate(metrics: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    rows = list(metrics)
    total_n = int(sum(int(r["n_examples"]) for r in rows))
    if total_n == 0:
        return {
            "n_examples": 0,
            "brier": None,
            "log_loss": None,
            "roc_auc": None,
            "ece": None,
        }

    def _wmean(key: str) -> float | None:
        vals = [(float(r[key]), int(r["n_examples"])) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        w = sum(n for _, n in vals)
        return float(sum(v * n for v, n in vals) / w)

    return {
        "n_examples": total_n,
        "brier": _wmean("brier"),
        "log_loss": _wmean("log_loss"),
        "roc_auc": _wmean("roc_auc"),
        "ece": _wmean("ece"),
    }


def _breakdown(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    field: str,
    min_support: int,
    ece_bins: int,
) -> dict[str, dict[str, float | int | None]]:
    by_key: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        value = row.get(field)
        if value is None:
            continue
        key = str(value)
        by_key.setdefault(key, []).append(i)

    out: dict[str, dict[str, float | int | None]] = {}
    for key in sorted(by_key):
        idx = by_key[key]
        if len(idx) < min_support:
            continue
        y = np.asarray([int(rows[i]["outcome"]) for i in idx], dtype=int)
        p = probs[np.asarray(idx, dtype=int)]
        out[key] = _compute_metrics(y, p, ece_bins=ece_bins)
    return out


def evaluate_walkforward(
    dataset: list[dict[str, Any]],
    predictor_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
    *,
    time_field: str = "timestamp",
    outcome_field: str = "outcome",
    min_train_size: int = 100,
    test_size: int = 50,
    step_size: int | None = None,
    ece_bins: int = 10,
    min_breakdown_support: int = 20,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    version: str = "baseline_v1",
    schema_version: str = "walkforward_eval.v1",
) -> dict[str, Any]:
    """Run temporal walk-forward evaluation and persist reproducible JSON output."""
    if min_train_size <= 0 or test_size <= 0:
        raise ValueError("min_train_size and test_size must be > 0")

    data = list(dataset)
    if not data:
        raise ValueError("dataset is empty")

    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be > 0")

    for row in data:
        if time_field not in row:
            raise ValueError(f"Row missing required time field '{time_field}'")
        if outcome_field not in row:
            raise ValueError(f"Row missing required outcome field '{outcome_field}'")

    ordered = sorted(data, key=lambda r: _to_datetime(r[time_field]))
    n = len(ordered)

    fold_metrics: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_probs: list[float] = []

    fold_id = 1
    for train_end_idx in range(min_train_size, n - test_size + 1, step):
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size, n)

        train_rows = ordered[:train_end_idx]
        test_rows = ordered[test_start_idx:test_end_idx]
        if not test_rows:
            continue

        train_end_dt = _to_datetime(train_rows[-1][time_field])
        test_start_dt = _to_datetime(test_rows[0][time_field])
        if not (train_end_dt < test_start_dt):
            raise ValueError(
                "Temporal leakage detected: train_end must be strictly before test_start "
                f"(got {train_end_dt.isoformat()} >= {test_start_dt.isoformat()})"
            )

        raw_probs = predictor_fn(train_rows, test_rows)
        probs = _as_numpy_probs(raw_probs, len(test_rows))
        y_true = np.asarray([int(r[outcome_field]) for r in test_rows], dtype=int)

        metrics = _compute_metrics(y_true, probs, ece_bins=ece_bins)
        metrics.update(
            {
                "fold": fold_id,
                "train_start": _to_datetime(train_rows[0][time_field]).isoformat(),
                "train_end": train_end_dt.isoformat(),
                "test_start": test_start_dt.isoformat(),
                "test_end": _to_datetime(test_rows[-1][time_field]).isoformat(),
            }
        )
        fold_metrics.append(metrics)

        all_rows.extend(test_rows)
        all_probs.extend(probs.tolist())
        fold_id += 1

    if not fold_metrics:
        raise ValueError("No folds generated. Check min_train_size/test_size against dataset length.")

    all_probs_np = np.asarray(all_probs, dtype=float)
    y_all = np.asarray([int(r[outcome_field]) for r in all_rows], dtype=int)

    result: dict[str, Any] = {
        "metadata": {
            "version": version,
            "schema_version": schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "split_params": {
                "strategy": "temporal_walkforward",
                "time_field": time_field,
                "outcome_field": outcome_field,
                "min_train_size": min_train_size,
                "test_size": test_size,
                "step_size": step,
                "ece_bins": ece_bins,
                "min_breakdown_support": min_breakdown_support,
                "anti_leakage_rule": "train_end < test_start",
            },
        },
        "folds": fold_metrics,
        "aggregate": _weighted_aggregate(fold_metrics),
        "breakdown": {
            "horizon": _breakdown(all_rows, all_probs_np, "horizon", min_breakdown_support, ece_bins),
            "predicate": _breakdown(all_rows, all_probs_np, "predicate", min_breakdown_support, ece_bins),
            "event_family": _breakdown(all_rows, all_probs_np, "event_family", min_breakdown_support, ece_bins),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result

"""Build a deterministic evaluation dataset from lakehouse snapshots.

Usage:
    python -m training.build_eval_dataset --min-resolved 100
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from data_layer.db import get_db

# Candidate market fields commonly present in explicit_features.
_MARKET_FEATURE_KEYS = [
    "metaculus_p",
    "metaculus_available",
    "polymarket_p",
    "polymarket_available",
    "market_available",
]

# Candidate market/prediction columns available from predictions table.
_MARKET_PREDICTION_COLUMNS = [
    "raw_prob",
    "calibrated_prob",
    "market_override",
    "market_prob_used",
]

_BASE_COLUMNS = [
    "question_id",
    "snapshot_id",
    "as_of_time",
    "deadline",
    "predicate",
    "event_family",
    "explicit_features",
    "outcome",
]


def _normalize_explicit_features(value: Any) -> dict[str, Any]:
    """Normalize explicit_features payload into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def build_eval_dataset(min_resolved: int = 0) -> pd.DataFrame:
    """Build resolved, unambiguous evaluation rows without recomputing features.

    Data source is the lakehouse DB via ``data_layer.db.get_db(read_only=True)``
    and uses the ``training_ready_snapshots`` view with a fallback query when
    the view is unavailable.
    """
    db = get_db(read_only=True)

    base_query = """
    SELECT
        trs.question_id,
        trs.snapshot_id,
        trs.as_of_time,
        trs.deadline,
        trs.predicate,
        trs.event_family,
        trs.explicit_features,
        trs.outcome,
        p.raw_prob,
        p.calibrated_prob,
        p.market_override,
        p.market_prob_used
    FROM training_ready_snapshots trs
    LEFT JOIN (
        SELECT
            snapshot_id,
            raw_prob,
            calibrated_prob,
            market_override,
            market_prob_used,
            ROW_NUMBER() OVER (
                PARTITION BY snapshot_id
                ORDER BY predicted_at DESC, prediction_id DESC
            ) AS rn
        FROM predictions
    ) p
      ON p.snapshot_id = trs.snapshot_id
     AND p.rn = 1
    ORDER BY trs.as_of_time ASC, trs.snapshot_id ASC
    """

    fallback_query = """
    SELECT
        fs.question_id,
        fs.snapshot_id,
        fs.as_of_time,
        q.deadline,
        q.predicate,
        q.event_family,
        fs.explicit_features,
        qr.outcome,
        p.raw_prob,
        p.calibrated_prob,
        p.market_override,
        p.market_prob_used
    FROM feature_snapshots fs
    JOIN questions q ON q.question_id = fs.question_id
    JOIN question_resolutions qr ON qr.question_id = q.question_id
    LEFT JOIN (
        SELECT
            snapshot_id,
            raw_prob,
            calibrated_prob,
            market_override,
            market_prob_used,
            ROW_NUMBER() OVER (
                PARTITION BY snapshot_id
                ORDER BY predicted_at DESC, prediction_id DESC
            ) AS rn
        FROM predictions
    ) p
      ON p.snapshot_id = fs.snapshot_id
     AND p.rn = 1
    WHERE fs.outcome IS NOT NULL
      AND fs.explicit_features IS NOT NULL
      AND qr.is_ambiguous = FALSE
    ORDER BY fs.as_of_time ASC, fs.snapshot_id ASC
    """

    try:
        df = db.execute(base_query).fetchdf()
    except Exception:
        df = db.execute(fallback_query).fetchdf()

    if df.empty:
        return df

    # Defensive normalization in case source query changes in future.
    df = df[df["outcome"].isin([0, 1])].copy()

    # Normalize explicit_features payload and expose known market feature keys.
    normalized_features = df["explicit_features"].apply(_normalize_explicit_features)
    df["explicit_features"] = normalized_features
    for key in _MARKET_FEATURE_KEYS:
        df[key] = normalized_features.apply(lambda x: x.get(key))

    # Deterministic ordering and columns.
    df = df.sort_values(["as_of_time", "snapshot_id"], kind="mergesort").reset_index(drop=True)
    ordered_columns = _BASE_COLUMNS + _MARKET_FEATURE_KEYS + _MARKET_PREDICTION_COLUMNS
    df = df[ordered_columns]

    if len(df) < min_resolved:
        raise ValueError(
            f"Resolved examples ({len(df)}) below minimum required ({min_resolved})."
        )

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic eval dataset")
    parser.add_argument(
        "--min-resolved",
        type=int,
        default=0,
        help="Minimum resolved rows required in the output dataset.",
    )
    args = parser.parse_args()

    df = build_eval_dataset(min_resolved=args.min_resolved)
    print(f"rows={len(df)}")
    print(",".join(df.columns.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

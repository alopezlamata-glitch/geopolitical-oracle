"""Baseline v1 evaluation helpers.

Provides a stable public interface `run_baseline_v1` that wraps predictor.inference
and evaluates only variants that are actually computable from inputs.
"""
from __future__ import annotations

from typing import Optional

from predictor.inference import predict


_PREDICT_KEYS = {
    "raw_prob",
    "calibrated_prob",
    "ci_lo",
    "ci_hi",
    "ci_method",
    "answer",
    "untrained",
    "market_override",
}


def _valid_market(p: Optional[float]) -> bool:
    return p is not None and 0.0 < p < 1.0


def run_baseline_v1(
    features: dict[str, float],
    *,
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
) -> dict:
    """Run baseline variants with a stable output contract.

    Always evaluates `model_only`. Evaluates `market_blend` only when at least one
    usable market probability is provided.
    """
    variants: dict[str, dict] = {
        "model_only": predict(features, metaculus_p=None, polymarket_p=None)
    }

    has_market = _valid_market(metaculus_p) or _valid_market(polymarket_p)
    if has_market:
        variants["market_blend"] = predict(
            features,
            metaculus_p=metaculus_p,
            polymarket_p=polymarket_p,
        )

    # Contract check at runtime to keep interface drift visible.
    for name, pred in variants.items():
        missing = _PREDICT_KEYS - set(pred.keys())
        if missing:
            raise ValueError(f"variant '{name}' missing predict keys: {sorted(missing)}")

    return {
        "baseline": "v1",
        "prediction": variants["model_only"],
        "variants": variants,
        "market_coverage": 1.0 if has_market else 0.0,
        "omitted_variants": [] if has_market else ["market_blend"],
    }

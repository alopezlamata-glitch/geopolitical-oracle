from __future__ import annotations

from typing import Optional

from predictor.baseline import run_baseline_v1


def predict(
    features: dict[str, float],
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
) -> dict:
    """
    Compatibility facade for historical callers.

    Delegates to `run_baseline_v1(..., variant="blended")` to preserve
    the existing prediction semantics (model + optional market override).
    """
    out = run_baseline_v1(
        features,
        market_signal={
            "metaculus_p": metaculus_p,
            "polymarket_p": polymarket_p,
            "variant": "blended",
        },
    )

    # Keep legacy output shape for existing integrations.
    out.pop("baseline_version", None)
    out.pop("variant", None)
    return out

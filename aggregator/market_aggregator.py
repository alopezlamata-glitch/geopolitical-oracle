from __future__ import annotations

import logging
import math
from typing import Optional

from collector.base import EvidenceBlock

logger = logging.getLogger(__name__)

_MARKET_SOURCES = {"metaculus", "polymarket"}
_CLAMP_LOW = 0.001
_CLAMP_HIGH = 0.999


def _clamp(p: float) -> float:
    return max(_CLAMP_LOW, min(_CLAMP_HIGH, p))


def aggregate_markets(evidence: list[EvidenceBlock]) -> Optional[float]:
    """
    Logarithmic pooling (geometric mean in log-odds space) of valid market probabilities.
    Returns None if no valid market signals found.
    """
    valid_probs: list[float] = []

    for block in evidence:
        if block.source not in _MARKET_SOURCES:
            continue
        if block.quality == "insufficient":
            continue
        p = block.metadata.get("probability")
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        valid_probs.append(_clamp(p))

    if not valid_probs:
        logger.debug("market_aggregator: no valid market signals")
        return None

    log_odds = [math.log(p / (1 - p)) for p in valid_probs]
    avg_log_odds = sum(log_odds) / len(log_odds)
    combined = 1 / (1 + math.exp(-avg_log_odds))

    logger.debug(
        "market_aggregator: pooled %d source(s) → %.3f (inputs: %s)",
        len(valid_probs),
        combined,
        [f"{p:.3f}" for p in valid_probs],
    )
    return combined

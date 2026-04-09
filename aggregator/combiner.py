from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

_W_LLM = 0.4
_W_MARKET = 0.6
_ALPHA = 1.4  # extremization factor
_CLAMP_LOW = 0.005
_CLAMP_HIGH = 0.995


def _clamp(p: float) -> float:
    return max(_CLAMP_LOW, min(_CLAMP_HIGH, p))


def _to_log_odds(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1 - p))


def _from_log_odds(lo: float) -> float:
    return 1 / (1 + math.exp(-lo))


def combine(p_llm: float, p_market: Optional[float]) -> float:
    """
    Combine LLM estimate with market aggregate using weighted log-odds pooling,
    then extremize by alpha=1.4 to correct for shared information.

    If p_market is None (no market coverage), extremize LLM estimate only.
    """
    p_llm = _clamp(p_llm)

    if p_market is not None:
        p_market = _clamp(p_market)
        # Weighted combination in log-odds space
        lo_combined = _W_LLM * _to_log_odds(p_llm) + _W_MARKET * _to_log_odds(p_market)
        p_weighted = _from_log_odds(lo_combined)
        logger.debug(
            "combiner: LLM=%.3f  market=%.3f  weighted=%.3f",
            p_llm, p_market, p_weighted,
        )
    else:
        p_weighted = p_llm
        logger.debug("combiner: no market signal — LLM only: %.3f", p_llm)

    # Extremize: pull away from 50% to correct for shared information between sources
    lo_extremized = _to_log_odds(p_weighted) * _ALPHA
    p_final = _from_log_odds(lo_extremized)

    logger.debug("combiner: extremized %.3f → %.3f", p_weighted, p_final)
    return p_final

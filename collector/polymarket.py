from __future__ import annotations

import logging
import re
from typing import Optional

import aiohttp

from .base import graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://gamma-api.polymarket.com/markets"
_TIMEOUT = aiohttp.ClientTimeout(total=10)
_MIN_VOLUME = 10000
_MIN_OVERLAP = 0.35


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def _keywords(text: str) -> set[str]:
    stop = {"will", "the", "a", "an", "be", "is", "are", "was", "in", "on",
            "at", "to", "for", "of", "and", "or", "by", "with", "there", "their"}
    return {w for w in _normalize(text).split() if len(w) >= 3 and w not in stop}


def _score(query: str, market_title: str) -> float:
    q_kws = _keywords(query)
    m_kws = _keywords(market_title)
    if not q_kws:
        return 0.0
    overlap = len(q_kws & m_kws) / max(len(q_kws), 1)
    return overlap


@graceful_collector("polymarket")
async def collect_polymarket(session: aiohttp.ClientSession, query: str) -> tuple[Optional[float], float]:
    """
    Returns (yes_probability, volume) or (None, 0) if no suitable market found.
    """
    params = {"active": "true", "limit": "100"}
    async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
        if resp.status in (403, 429):
            logger.warning("polymarket: HTTP %s", resp.status)
            return None, 0.0
        resp.raise_for_status()
        markets = await resp.json()

    if not isinstance(markets, list):
        return None, 0.0

    best_score = 0.0
    best_p: Optional[float] = None
    best_vol = 0.0

    for m in markets:
        title = m.get("question", "") or m.get("title", "") or ""
        if not title:
            continue
        score = _score(query, title)
        if score < _MIN_OVERLAP:
            continue
        volume = float(m.get("volume", 0) or 0)
        if volume < _MIN_VOLUME:
            continue
        if score <= best_score:
            continue

        # Extract YES price
        outcome_prices = m.get("outcomePrices", [])
        outcomes = m.get("outcomes", [])
        p = None
        if isinstance(outcome_prices, list) and outcome_prices:
            try:
                p = float(outcome_prices[0])
            except (ValueError, TypeError):
                pass
        if p is None or not (0.01 < p < 0.99):
            continue

        best_score = score
        best_p = p
        best_vol = volume

    if best_p is not None:
        logger.info("polymarket: found p=%.3f (vol=%.0f, score=%.2f)", best_p, best_vol, best_score)
    return best_p, best_vol, best_score

"""
Manifold Markets prediction market collector — public API, no auth required.

Used as fallback when Metaculus API token is not configured.
Returns (community_prediction: float, num_forecasters: int) for the
best-matching open binary market, or (None, 0) when none is found.

API: https://api.manifold.markets/v0/search-markets
  Params: term=..., limit=20, sort=liquidity, filter=open, contractType=BINARY

Market selection criteria:
  - Keyword overlap score >= 0.30
  - uniqueBettorCount >= 10 (min liquidity of crowd wisdom)
  - probability in (0.01, 0.99)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import aiohttp

from .base import graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.manifold.markets/v0/search-markets"
_TIMEOUT = aiohttp.ClientTimeout(total=10)
_HEADERS = {
    "User-Agent": "geopolitical-oracle/1.0 (research; github.com/alopezlamata-glitch)",
    "Accept": "application/json",
}
_MIN_BETTORS = 10
_MIN_OVERLAP = 0.30

_STOP = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
    "from", "before", "after", "during", "about", "have", "has", "had",
    "do", "does", "did", "not", "no", "its", "it", "there", "their",
    "happen", "occur", "take", "place", "until", "between",
}


def _keywords(text: str) -> set[str]:
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {t for t in tokens if len(t) >= 3 and t not in _STOP}


def _score(query: str, market_question: str) -> float:
    q_kws = _keywords(query)
    m_kws = _keywords(market_question)
    if not q_kws:
        return 0.0
    return round(len(q_kws & m_kws) / max(len(q_kws), 1), 4)


@graceful_collector("manifold")
async def collect_manifold(
    session: aiohttp.ClientSession, query: str
) -> tuple[Optional[float], int]:
    """
    Returns (community_prediction, num_bettors) or (None, 0) if no suitable market.
    """
    keywords = " ".join(list(_keywords(query))[:6])
    if not keywords:
        return None, 0

    params = {
        "term": keywords,
        "limit": "20",
        "sort": "liquidity",
        "filter": "open",
        "contractType": "BINARY",
    }
    async with session.get(
        _BASE_URL, params=params, timeout=_TIMEOUT, headers=_HEADERS
    ) as resp:
        if resp.status in (403, 429):
            logger.warning("manifold: HTTP %d", resp.status)
            return None, 0
        resp.raise_for_status()
        markets = await resp.json(content_type=None)

    if not isinstance(markets, list):
        return None, 0

    for m in markets:
        question = m.get("question", "")
        score = _score(query, question)
        if score < _MIN_OVERLAP:
            continue

        bettors = m.get("uniqueBettorCount", 0) or 0
        if bettors < _MIN_BETTORS:
            continue

        p = m.get("probability")
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue

        if not (0.01 < p < 0.99):
            continue

        logger.info(
            "manifold: p=%.3f (%d bettors, score=%.2f) — %r",
            p, bettors, score, question[:70],
        )
        return p, bettors

    logger.debug("manifold: no market matched for query: %r", keywords[:60])
    return None, 0

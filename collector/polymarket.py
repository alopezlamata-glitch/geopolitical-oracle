"""
Polymarket prediction market collector.

Uses the Polymarket Gamma API (public, no auth required).
Returns (yes_probability, volume_usd, match_score) for the best-matching
active binary market, or (None, 0.0, 0.0) when none found.

Endpoint: https://gamma-api.polymarket.com/markets
  Params: active=true, closed=false, limit=150

Market selection criteria:
  - Keyword overlap score >= _MIN_OVERLAP (0.25, relaxed for broader coverage)
  - Volume >= _MIN_VOLUME ($5k, relaxed for broader coverage)
  - YES outcome price in (0.01, 0.99)

The match_score is used by market_prior.py to gate and weight the market blend.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import aiohttp

from .base import graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://gamma-api.polymarket.com/markets"
_TIMEOUT = aiohttp.ClientTimeout(total=12)
_HEADERS = {
    "User-Agent": "geopolitical-oracle/1.0 (research; github.com/alopezlamata-glitch)",
    "Accept": "application/json",
}
_MIN_VOLUME = 5_000    # USD — relaxed for broader coverage
_MIN_OVERLAP = 0.25    # keyword overlap — relaxed for broader coverage


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


_STOP = {
    "will", "the", "a", "an", "be", "is", "are", "was", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "there", "their",
    "this", "that", "from", "before", "after", "during", "about",
}


def _keywords(text: str) -> set[str]:
    return {w for w in _normalize(text).split() if len(w) >= 3 and w not in _STOP}


def _score(query: str, market_title: str) -> float:
    q_kws = _keywords(query)
    m_kws = _keywords(market_title)
    if not q_kws:
        return 0.0
    overlap = len(q_kws & m_kws) / max(len(q_kws), 1)
    return round(overlap, 4)


def _extract_yes_price(m: dict) -> Optional[float]:
    """
    Extract the YES outcome price from a Gamma API market object.

    Gamma API returns outcomePrices as either:
      - A JSON string: '["0.75", "0.25"]'
      - A list: ["0.75", "0.25"]
      - A list of floats: [0.75, 0.25]

    outcomes is a list of outcome names like ["Yes", "No"].
    YES is typically index 0 but we verify against the outcomes list.
    """
    outcomes = m.get("outcomes", [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []

    prices_raw = m.get("outcomePrices", [])
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except Exception:
            return None

    if not isinstance(prices_raw, list) or not prices_raw:
        return None

    # Find YES index
    yes_idx = 0
    if isinstance(outcomes, list):
        for i, o in enumerate(outcomes):
            if isinstance(o, str) and o.strip().lower() in ("yes", "true", "1"):
                yes_idx = i
                break

    try:
        if yes_idx < len(prices_raw):
            return float(prices_raw[yes_idx])
    except (TypeError, ValueError):
        pass

    return None


def _extract_volume(m: dict) -> float:
    """Extract USD volume from various field names the API may use."""
    for field in ("volume", "volumeNum", "volume24hr", "liquidityNum", "liquidity"):
        v = m.get(field)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


@graceful_collector("polymarket")
async def collect_polymarket(
    session: aiohttp.ClientSession, query: str
) -> tuple[Optional[float], float, float]:
    """
    Returns (yes_probability, volume_usd, match_score) or (None, 0.0, 0.0).

    match_score is the keyword overlap in [0, 1] — used by market_prior.py
    to weight the market signal quality.
    """
    params = {
        "active": "true",
        "closed": "false",
        "limit": "150",
    }
    async with session.get(
        _BASE_URL, params=params, timeout=_TIMEOUT, headers=_HEADERS
    ) as resp:
        if resp.status in (403, 429):
            logger.warning("polymarket: HTTP %d", resp.status)
            return None, 0.0, 0.0
        resp.raise_for_status()
        markets = await resp.json(content_type=None)

    if not isinstance(markets, list):
        logger.warning("polymarket: unexpected response type: %s", type(markets))
        return None, 0.0, 0.0

    logger.debug("polymarket: evaluating %d active markets", len(markets))

    best_score = 0.0
    best_p: Optional[float] = None
    best_vol = 0.0

    for m in markets:
        title = m.get("question", "") or m.get("title", "") or m.get("slug", "") or ""
        if not title:
            continue

        score = _score(query, title)
        if score < _MIN_OVERLAP:
            continue

        volume = _extract_volume(m)
        if volume < _MIN_VOLUME:
            continue

        # Prefer higher overlap; break ties with volume
        if score < best_score or (score == best_score and volume <= best_vol):
            continue

        p = _extract_yes_price(m)
        if p is None or not (0.01 < p < 0.99):
            continue

        best_score = score
        best_p = p
        best_vol = volume

    if best_p is not None:
        logger.info(
            "polymarket: p=%.3f (vol=$%.0f, score=%.2f)",
            best_p, best_vol, best_score,
        )
    else:
        logger.debug(
            "polymarket: no market matched (checked %d, threshold: overlap>=%.2f vol>=$%.0f)",
            len(markets), _MIN_OVERLAP, _MIN_VOLUME,
        )

    return best_p, best_vol, best_score

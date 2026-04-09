from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

import aiohttp

from .base import EvidenceBlock, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://gamma-api.polymarket.com/markets"
_MIN_VOLUME = 10_000
_MIN_SIMILARITY = 0.3
_TIMEOUT = aiohttp.ClientTimeout(total=8)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _parse_outcome_price(raw) -> float | None:
    """Parse Polymarket outcomePrices field (may be double-encoded JSON string)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(raw, list) and len(raw) == 2:
        try:
            return float(raw[0])
        except (TypeError, ValueError):
            return None

    return None


@graceful_collector("polymarket")
async def collect_polymarket(session: aiohttp.ClientSession, query: str) -> EvidenceBlock | None:
    params = {"active": "true", "limit": 100}

    async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        markets = await resp.json()

    if not isinstance(markets, list):
        # Some responses nest under a key
        markets = markets.get("markets", markets.get("data", []))

    if not markets:
        return EvidenceBlock(
            source="polymarket",
            content="No Polymarket markets returned.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    # Find best fuzzy match
    best_market = None
    best_score = 0.0
    for market in markets:
        title = market.get("question") or market.get("title") or ""
        description = market.get("description") or ""
        score = max(
            _similarity(query, title),
            _similarity(query, description),
        )
        if score > best_score:
            best_score = score
            best_market = market

    if best_market is None or best_score < _MIN_SIMILARITY:
        return EvidenceBlock(
            source="polymarket",
            content=f"No sufficiently similar Polymarket market found (best score: {best_score:.2f}).",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    title = best_market.get("question") or best_market.get("title") or "Unknown"
    volume = float(best_market.get("volume") or 0)
    liquidity = float(best_market.get("liquidity") or 0)
    outcome_prices_raw = best_market.get("outcomePrices")
    yes_probability = _parse_outcome_price(outcome_prices_raw)

    if volume < _MIN_VOLUME or yes_probability is None:
        reason = f"volume ${volume:,.0f} < ${_MIN_VOLUME:,}" if volume < _MIN_VOLUME else "could not parse price"
        return EvidenceBlock(
            source="polymarket",
            content=f"Polymarket match '{title}' (score={best_score:.2f}) but insufficient: {reason}.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={"probability": None, "volume": volume, "liquidity": liquidity},
        )

    content = (
        f"Polymarket: '{title}' (match={best_score:.2f})\n"
        f"YES price: {yes_probability:.1%} | Volume: ${volume:,.0f} | Liquidity: ${liquidity:,.0f}"
    )

    return EvidenceBlock(
        source="polymarket",
        content=content,
        quality="high" if volume >= 100_000 else "medium",
        timestamp=datetime.now(timezone.utc),
        metadata={"probability": yes_probability, "volume": volume, "liquidity": liquidity},
    )

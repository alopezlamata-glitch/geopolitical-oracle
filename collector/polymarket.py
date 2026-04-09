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
_MIN_KEYWORD_OVERLAP = 0.35  # fraction of query keywords that must appear in market title
_MIN_SEQUENCE_SCORE = 0.55   # fallback: high bar for sequence similarity
_TIMEOUT = aiohttp.ClientTimeout(total=8)

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
    "from", "before", "after", "during", "about", "have", "has", "had",
    "do", "does", "did", "not", "no", "its", "it", "there", "their",
    "happen", "occur", "take", "place", "by", "until",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _keywords(text: str) -> set[str]:
    return {w for w in _normalize(text).split() if len(w) >= 3 and w not in _STOP_WORDS}


def _score(query: str, market_title: str, description: str = "") -> float:
    """
    Hybrid score: keyword overlap (primary) + sequence similarity (secondary).
    A market must share meaningful keywords with the query to qualify.
    """
    q_kws = _keywords(query)
    if not q_kws:
        return 0.0

    title_kws = _keywords(market_title)
    desc_kws = _keywords(description)
    all_market_kws = title_kws | desc_kws

    overlap = len(q_kws & all_market_kws) / len(q_kws)

    # Sequence similarity on normalized strings (structural match)
    seq = SequenceMatcher(None, _normalize(query), _normalize(market_title)).ratio()

    # Overlap is the gating signal; sequence adds a small boost for identical phrasing
    return overlap * 0.75 + seq * 0.25


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
        markets = markets.get("markets", markets.get("data", []))

    if not markets:
        return EvidenceBlock(
            source="polymarket",
            content="No Polymarket markets returned.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    # Score all markets; require keyword overlap as primary gate
    best_market = None
    best_score = 0.0
    for market in markets:
        title = market.get("question") or market.get("title") or ""
        description = market.get("description") or ""
        s = _score(query, title, description)
        if s > best_score:
            best_score = s
            best_market = market

    # Require meaningful keyword overlap (not just structural string similarity)
    q_kws = _keywords(query)
    if best_market:
        t = best_market.get("question") or best_market.get("title") or ""
        overlap_fraction = len(q_kws & _keywords(t)) / max(len(q_kws), 1)
    else:
        overlap_fraction = 0.0

    if best_market is None or overlap_fraction < _MIN_KEYWORD_OVERLAP:
        return EvidenceBlock(
            source="polymarket",
            content=f"No relevant Polymarket market found (best overlap: {overlap_fraction:.0%}).",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    title = best_market.get("question") or best_market.get("title") or "Unknown"
    volume = float(best_market.get("volume") or 0)
    liquidity = float(best_market.get("liquidity") or 0)
    yes_probability = _parse_outcome_price(best_market.get("outcomePrices"))

    if volume < _MIN_VOLUME or yes_probability is None:
        reason = f"volume ${volume:,.0f} < ${_MIN_VOLUME:,}" if volume < _MIN_VOLUME else "could not parse price"
        return EvidenceBlock(
            source="polymarket",
            content=f"Polymarket: '{title}' — insufficient: {reason}.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={"probability": None, "volume": volume, "liquidity": liquidity},
        )

    content = (
        f"Polymarket: '{title}' (overlap={overlap_fraction:.0%})\n"
        f"YES price: {yes_probability:.1%} | Volume: ${volume:,.0f} | Liquidity: ${liquidity:,.0f}"
    )

    return EvidenceBlock(
        source="polymarket",
        content=content,
        quality="high" if volume >= 100_000 else "medium",
        timestamp=datetime.now(timezone.utc),
        metadata={"probability": yes_probability, "volume": volume, "liquidity": liquidity},
    )

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


def _entities(text: str) -> set[str]:
    """Extract proper nouns (capitalized tokens) as entity signals."""
    tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    return {t.lower() for t in tokens if t.lower() not in _STOP}


def _score(query: str, market_question: str) -> float:
    """
    Entity-aware overlap score.
    Proper nouns in the query (countries, people, orgs) count 3× vs generic keywords.
    """
    q_kws = _keywords(query)
    m_kws = _keywords(market_question)
    if not q_kws:
        return 0.0

    q_ents = _entities(query)
    if q_ents:
        regular_kws = q_kws - q_ents
        ent_matches     = len(q_ents & m_kws)
        regular_matches = len(regular_kws & m_kws)
        total_weight = 3 * len(q_ents) + max(len(regular_kws), 1)
        return round((3 * ent_matches + regular_matches) / total_weight, 4)

    return round(len(q_kws & m_kws) / max(len(q_kws), 1), 4)


async def _fetch_markets(
    session: aiohttp.ClientSession, term: str
) -> list[dict]:
    params = {
        "term": term,
        "limit": "20",
        "sort": "liquidity",
        "filter": "open",
        "contractType": "BINARY",
    }
    async with session.get(
        _BASE_URL, params=params, timeout=_TIMEOUT, headers=_HEADERS
    ) as resp:
        if resp.status in (403, 429):
            logger.warning("manifold: HTTP %d on term=%r", resp.status, term[:40])
            return []
        resp.raise_for_status()
        data = await resp.json(content_type=None)
    return data if isinstance(data, list) else []


@graceful_collector("manifold")
async def collect_manifold(
    session: aiohttp.ClientSession, query: str
) -> tuple[Optional[float], int]:
    """
    Returns (community_prediction, num_bettors) or (None, 0) if no suitable market.

    Tries two search strategies:
      1. Full keyword set (existing behaviour)
      2. Entity-only terms (country/org names) — better recall for geopolitical queries
    Results are merged; best entity-aware score wins.
    """
    kws = _keywords(query)
    if not kws:
        return None, 0

    # Build search term lists
    full_term   = " ".join(list(kws)[:6])
    entity_term = " ".join(list(_entities(query))[:4])

    # Fetch from both strategies; deduplicate by market id
    seen: dict[str, dict] = {}
    for term in dict.fromkeys([full_term, entity_term] if entity_term else [full_term]):
        for m in await _fetch_markets(session, term):
            mid = m.get("id") or m.get("slug") or m.get("question", "")[:60]
            seen.setdefault(mid, m)

    # Score all candidates against original query, pick best
    best_score = 0.0
    best_p: Optional[float] = None
    best_bettors = 0
    best_question = ""

    for m in seen.values():
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

        # Prefer higher score; break ties with bettor count
        if score > best_score or (score == best_score and bettors > best_bettors):
            best_score    = score
            best_p        = p
            best_bettors  = bettors
            best_question = question

    if best_p is not None:
        logger.info(
            "manifold: p=%.3f (%d bettors, score=%.2f) — %r",
            best_p, best_bettors, best_score, best_question[:70],
        )
    else:
        logger.debug("manifold: no market matched for query: %r", full_term[:60])

    return best_p, best_bettors

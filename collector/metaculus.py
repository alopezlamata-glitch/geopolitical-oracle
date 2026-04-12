"""
Metaculus question finder — search for related questions and report forecaster count.

IMPORTANT: Metaculus intentionally hides community predictions via API — you only
see the CP after making your own prediction (anti-anchoring design). As a result
this collector returns (None, nr_forecasters) — the probability is always None,
but nr_forecasters signals how much crowd attention the question has attracted.

Use Manifold Markets (collector/manifold.py) for free crowd probability.
Use Polymarket (collector/polymarket.py) for real-money probability.

Token required: set METACULUS_API_TOKEN in .env (free at metaculus.com).
Without a token, the API returns HTTP 403.

API used: GET /api/posts/?search=...&forecast_type=binary&limit=20
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import aiohttp

from .base import graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.metaculus.com/api/posts/"
_TIMEOUT = aiohttp.ClientTimeout(total=12)
_MIN_FORECASTERS = 10   # min crowd size to consider the question relevant


def _build_headers() -> dict:
    import os
    h = {
        "User-Agent": "geopolitical-oracle/1.0 (research; github.com/alopezlamata-glitch)",
        "Accept": "application/json",
    }
    token = os.getenv("METACULUS_API_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Token {token}"
    return h


def _extract_keywords(query: str, n: int = 6) -> str:
    stop = {
        "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
        "from", "before", "after", "during", "about", "have", "has", "had",
        "do", "does", "did", "not", "no", "its", "it", "there", "their",
        "happen", "occur", "take", "place", "until", "between",
    }
    tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
    filtered = [t for t in tokens if len(t) >= 3 and t not in stop]
    return " ".join(filtered[:n])


@graceful_collector("metaculus")
async def collect_metaculus(
    session: aiohttp.ClientSession, query: str
) -> tuple[Optional[float], int]:
    """
    Returns (None, nr_forecasters) — probability is never returned because
    Metaculus intentionally hides CP via API (only visible after you forecast).

    nr_forecasters > 0 signals that a relevant, active question exists on
    Metaculus, providing a quality/attention signal even without a probability.

    The pipeline uses Manifold Markets as fallback probability source.
    """
    import os
    if not os.getenv("METACULUS_API_TOKEN", "").strip():
        logger.debug(
            "metaculus: no METACULUS_API_TOKEN in .env — skipping "
            "(get a free token at metaculus.com)"
        )
        return None, 0

    keywords = _extract_keywords(query)
    if not keywords:
        return None, 0

    headers = _build_headers()
    params = {
        "search": keywords,
        "forecast_type": "binary",
        "limit": "20",
        "order_by": "-activity",
    }

    try:
        async with session.get(
            _BASE_URL, params=params, timeout=_TIMEOUT, headers=headers
        ) as resp:
            if resp.status == 403:
                logger.warning(
                    "metaculus: HTTP 403 — check your METACULUS_API_TOKEN in .env"
                )
                return None, 0
            if resp.status in (404, 429):
                logger.warning("metaculus: HTTP %d", resp.status)
                return None, 0
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except Exception as e:
        logger.debug("metaculus: request failed: %s", e)
        return None, 0

    results = data.get("results", [])
    if not results:
        logger.debug("metaculus: no results for keywords: %r", keywords)
        return None, 0

    # Find the most-forecasted relevant question
    best_forecasters = 0
    best_title = ""
    for post in results:
        nf = post.get("nr_forecasters") or 0
        if nf < _MIN_FORECASTERS:
            continue
        if nf > best_forecasters:
            best_forecasters = nf
            best_title = post.get("title", "")[:60]

    if best_forecasters > 0:
        logger.info(
            "metaculus: found related question (%d forecasters) — %r "
            "[note: CP hidden by API design; using Manifold for probability]",
            best_forecasters, best_title,
        )
        # Return None probability — CP is intentionally hidden
        # nr_forecasters is still a useful attention/quality signal
        return None, best_forecasters

    logger.debug("metaculus: no question met ≥%d forecasters threshold", _MIN_FORECASTERS)
    return None, 0

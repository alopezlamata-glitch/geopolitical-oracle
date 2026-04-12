"""
Metaculus community prediction collector.

Uses the Metaculus API v3 (public, no auth required for read-only access).
Returns (community_prediction: float, num_forecasters: int) for the best-matching
active binary question, or (None, 0) when no suitable question is found.

API v3 endpoints used:
  GET /api/v3/questions/?search=...&status=open&type=binary&limit=20

Community prediction extraction from v3 response:
  q["aggregations"]["recency_weighted"]["history"][-1]["centers"][0]
  Fallback: q["community_prediction"]  (v2-style field still present on some)

Min forecasters: 30 (questions with fewer are too noisy to use as market signal).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import aiohttp

from .base import graceful_collector

logger = logging.getLogger(__name__)

# v3 API — requires auth token (free account at metaculus.com)
_BASE_URL_V3 = "https://www.metaculus.com/api/v3/questions/"
_BASE_URL_V2 = "https://www.metaculus.com/api2/questions/"
_TIMEOUT = aiohttp.ClientTimeout(total=12)
_MIN_FORECASTERS = 30


def _build_headers() -> dict:
    """Build request headers, injecting Metaculus token from env if present."""
    import os
    h = {
        "User-Agent": "geopolitical-oracle/1.0 (research; github.com/alopezlamata-glitch)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    token = os.getenv("METACULUS_API_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Token {token}"
    return h


def _extract_keywords(query: str, n: int = 8) -> str:
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


def _extract_prob_v3(q: dict) -> Optional[float]:
    """Extract community prediction from API v3 question object."""
    try:
        agg = q.get("aggregations", {})
        rw = agg.get("recency_weighted", {})
        history = rw.get("history", [])
        if history:
            centers = history[-1].get("centers") or []
            if centers:
                return float(centers[0])
    except Exception:
        pass

    # Fallback: try v2-style field sometimes present in v3 responses
    try:
        cp = q.get("community_prediction")
        if isinstance(cp, dict):
            v = cp.get("full", {}).get("q2") or cp.get("q2")
            if v is not None:
                return float(v)
        if isinstance(cp, (int, float)):
            return float(cp)
    except Exception:
        pass

    # Last resort: prediction field
    try:
        pred = q.get("prediction")
        if pred is not None:
            return float(pred)
    except Exception:
        pass

    return None


def _extract_forecasters(q: dict) -> int:
    for key in ("nr_forecasters", "number_of_forecasters", "forecasters_count"):
        v = q.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 0


async def _try_fetch(
    session: aiohttp.ClientSession, url: str, params: dict
) -> Optional[list]:
    """Attempt a single GET; return results list or None on HTTP error."""
    headers = _build_headers()
    try:
        async with session.get(url, params=params, timeout=_TIMEOUT, headers=headers) as resp:
            if resp.status == 403:
                import os
                if not os.getenv("METACULUS_API_TOKEN", "").strip():
                    logger.warning(
                        "metaculus: HTTP 403 — Metaculus now requires authentication. "
                        "Get a free token at metaculus.com and set METACULUS_API_TOKEN in .env"
                    )
                else:
                    logger.warning("metaculus: HTTP 403 — check your METACULUS_API_TOKEN")
                return None
            if resp.status in (404, 429):
                logger.warning("metaculus: HTTP %d from %s", resp.status, url)
                return None
            resp.raise_for_status()
            data = await resp.json(content_type=None)
            return data.get("results", [])
    except Exception as e:
        logger.debug("metaculus: fetch error from %s: %s", url, e)
        return None


@graceful_collector("metaculus")
async def collect_metaculus(
    session: aiohttp.ClientSession, query: str
) -> tuple[Optional[float], int]:
    """
    Returns (community_prediction, num_forecasters) or (None, 0) if unavailable.
    Tries API v3 first, falls back to v2 on 403/404.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return None, 0

    # ── API v3 attempt ─────────────────────────────────────────────────────
    params_v3 = {
        "search": keywords,
        "status": "open",
        "type": "binary",
        "limit": "20",
    }
    results = await _try_fetch(session, _BASE_URL_V3, params_v3)

    # ── API v2 fallback ─────────────────────────────────────────────────────
    if results is None:
        params_v2 = {
            "search": keywords,
            "status": "active",
            "type": "forecast",
            "limit": "20",
        }
        results = await _try_fetch(session, _BASE_URL_V2, params_v2)

    if not results:
        logger.debug("metaculus: no results for query: %r", keywords[:60])
        return None, 0

    # ── Find best question ───────────────────────────────────────────────────
    for q in results:
        forecasters = _extract_forecasters(q)
        if forecasters < _MIN_FORECASTERS:
            continue

        p = _extract_prob_v3(q)
        if p is None or not (0.01 < p < 0.99):
            continue

        title = q.get("title", "")[:80]
        logger.info(
            "metaculus: p=%.3f (%d forecasters) — %r",
            p, forecasters, title,
        )
        return p, forecasters

    logger.debug("metaculus: no question met quality threshold (>=30 forecasters)")
    return None, 0

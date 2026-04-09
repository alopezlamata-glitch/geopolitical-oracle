from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import aiohttp

from .base import EvidenceBlock, TTLCache, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = aiohttp.ClientTimeout(total=20)  # GDELT can be slow, give it more time
_CACHE = TTLCache(ttl_minutes=30)


def _cache_key(query: str) -> str:
    tokens = re.sub(r"[^\w\s]", "", query.lower()).split()
    return "gdelt:" + "_".join(tokens[:3])


def _extract_keywords(query: str) -> str:
    """Return top keywords from query for GDELT search."""
    tokens = re.sub(r"[^\w\s]", "", query.lower()).split()
    # Remove common stop words
    stop = {"will", "the", "a", "an", "be", "is", "are", "was", "were", "in",
            "on", "at", "to", "for", "of", "and", "or", "by", "with", "this",
            "that", "from", "before", "after", "during", "about", "have",
            "has", "had", "do", "does", "did", "not", "no", "its", "it"}
    filtered = [t for t in tokens if t not in stop and len(t) >= 3]
    return " ".join(filtered[:5])  # GDELT works best with 3-5 key terms


@graceful_collector("gdelt")
async def collect_gdelt(session: aiohttp.ClientSession, query: str) -> EvidenceBlock | None:
    key = _cache_key(query)
    cached = _CACHE.get(key)
    if cached is not None:
        logger.debug("gdelt: cache hit")
        return cached

    keywords = _extract_keywords(query)
    params = {
        "query": keywords,
        "mode": "artlist",
        "maxrecords": "20",
        "format": "json",
    }

    async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        raw = await resp.text()
        # GDELT sometimes returns empty body or HTML on overload
        if not raw or raw.lstrip().startswith("<"):
            raise ValueError("GDELT returned empty or HTML response (API overloaded)")
        import json as _json
        data = _json.loads(raw)

    articles = data.get("articles", [])
    if not articles:
        block = EvidenceBlock(
            source="gdelt",
            content="No GDELT articles found.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={"avg_tone": 0.0, "article_count": 0, "source_diversity": 0},
        )
        _CACHE.set(key, block)
        return block

    tones = []
    domains = set()
    for article in articles:
        tone = article.get("tone")
        if tone is not None:
            try:
                tones.append(float(tone))
            except (ValueError, TypeError):
                pass
        domain = article.get("domain") or article.get("url", "").split("/")[2] if article.get("url") else ""
        if domain:
            domains.add(domain)

    avg_tone = sum(tones) / len(tones) if tones else 0.0
    article_count = len(articles)
    source_diversity = len(domains)

    content = (
        f"GDELT analysis of '{keywords}':\n"
        f"Articles found: {article_count} | Avg tone: {avg_tone:.2f} | Sources: {source_diversity}"
    )

    block = EvidenceBlock(
        source="gdelt",
        content=content,
        quality="medium" if article_count >= 5 else "low",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "avg_tone": round(avg_tone, 4),
            "article_count": article_count,
            "source_diversity": source_diversity,
        },
    )
    _CACHE.set(key, block)
    return block

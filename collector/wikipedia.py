"""
Wikipedia background context collector.

Uses the Wikipedia REST API (no key required) to fetch a summary of the most
relevant article for the forecast question. The content is passed to the LLM
text feature extractor as background context, improving threat/escalation
scores without adding it as a synthetic event.

API used:
  - OpenSearch: /w/api.php?action=opensearch  (find best-matching article title)
  - REST summary: /api/rest_v1/page/summary/{title}  (1-2 paragraph extract)

Design:
  - No TTL cache — summary content rarely changes within a session
  - Falls back gracefully: returns WikipediaResult(found=False) on any error
  - Truncates to MAX_CONTENT_CHARS to keep LLM prompts manageable
  - Skips disambiguation/list pages (they have no useful extract)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import aiohttp

from .base import graceful_collector

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=10)
_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_HEADERS = {"User-Agent": "geopolitical-oracle/1.0 (research tool)"}

MAX_CONTENT_CHARS = 1_500   # keep LLM prompt short

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
    "from", "before", "after", "during", "about", "have", "has", "had",
    "do", "does", "did", "not", "no", "its", "it", "there", "their",
    "happen", "occur", "take", "place", "until", "between",
}


@dataclass
class WikipediaResult:
    """Output of collect_wikipedia()."""
    content: str = ""          # truncated article extract
    page_title: str = ""       # canonical Wikipedia title
    page_url: str = ""         # full article URL
    found: bool = False        # False when API failed or no article found


def _keywords(query: str, n: int = 5) -> str:
    """Extract the N most meaningful tokens from the query string."""
    tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
    filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
    return " ".join(filtered[:n])


async def _search_title(session: aiohttp.ClientSession, query: str) -> Optional[str]:
    """Return the best-matching Wikipedia article title, or None."""
    keywords = _keywords(query)
    if not keywords:
        return None

    params = {
        "action": "opensearch",
        "search": keywords,
        "limit": "3",
        "namespace": "0",
        "format": "json",
    }
    try:
        async with session.get(
            _SEARCH_URL, params=params, timeout=_TIMEOUT, headers=_HEADERS
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
            # OpenSearch format: [query, [titles], [descs], [urls]]
            titles = data[1] if len(data) > 1 else []
            return titles[0] if titles else None
    except Exception as e:
        logger.debug("wikipedia: opensearch failed: %s", e)
        return None


async def _fetch_summary(session: aiohttp.ClientSession, title: str) -> Optional[dict]:
    """Fetch the REST summary JSON for a given Wikipedia title."""
    url = _SUMMARY_URL.format(title=title.replace(" ", "_"))
    try:
        async with session.get(url, timeout=_TIMEOUT, headers=_HEADERS) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except Exception as e:
        logger.debug("wikipedia: summary fetch failed for %r: %s", title, e)
        return None


@graceful_collector("wikipedia")
async def collect_wikipedia(
    session: aiohttp.ClientSession,
    question: str,
) -> WikipediaResult:
    """
    Fetch Wikipedia background context for a forecast question.

    Returns WikipediaResult with `found=False` on any failure.
    Never raises (graceful_collector wraps all exceptions).

    Args:
        session  : shared aiohttp session
        question : the binary question being forecast
    """
    title = await _search_title(session, question)
    if not title:
        logger.debug("wikipedia: no article found for question: %r", question[:80])
        return WikipediaResult(found=False)

    summary = await _fetch_summary(session, title)
    if not summary:
        return WikipediaResult(found=False)

    # Skip disambiguation and list pages — they have no useful extract
    page_type = summary.get("type", "")
    if page_type in ("disambiguation", "no-extract"):
        logger.debug("wikipedia: skipping %r (%s)", title, page_type)
        return WikipediaResult(found=False)

    extract = summary.get("extract", "").strip()
    if not extract:
        return WikipediaResult(found=False)

    content = extract[:MAX_CONTENT_CHARS]
    page_url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
    canonical_title = summary.get("title", title)

    logger.info(
        "wikipedia: fetched %d chars for %r (article: %r)",
        len(content), question[:60], canonical_title,
    )
    return WikipediaResult(
        content=content,
        page_title=canonical_title,
        page_url=page_url,
        found=True,
    )

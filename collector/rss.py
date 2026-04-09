from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import aiohttp
import feedparser

from .base import EvidenceBlock, TTLCache, graceful_collector

logger = logging.getLogger(__name__)

_FEEDS = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
]
_WINDOW_HOURS = 72
_CACHE = TTLCache(ttl_minutes=30)


def _cache_key(query: str) -> str:
    tokens = re.sub(r"[^\w\s]", "", query.lower()).split()
    return "rss:" + "_".join(tokens[:3])


def _entry_time(entry) -> datetime | None:
    t = getattr(entry, "published_parsed", None)
    if t is None:
        return None
    import calendar
    return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)


def _parse_feed(url: str):
    return feedparser.parse(url)


def _extract_keywords(query: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s]", "", query.lower()).split() if len(w) >= 3]


@graceful_collector("rss")
async def collect_rss(session: aiohttp.ClientSession, query: str) -> EvidenceBlock | None:
    key = _cache_key(query)
    cached = _CACHE.get(key)
    if cached is not None:
        logger.debug("rss: cache hit")
        return cached

    keywords = _extract_keywords(query)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_WINDOW_HOURS)

    loop = asyncio.get_event_loop()
    # Parse feeds concurrently using executor (feedparser is synchronous)
    tasks = [loop.run_in_executor(None, _parse_feed, url) for url in _FEEDS]
    feeds = await asyncio.gather(*tasks, return_exceptions=True)

    matched: list[tuple[str, str]] = []
    for feed in feeds:
        if isinstance(feed, Exception):
            logger.warning("rss: feed parse error: %s", feed)
            continue
        for entry in getattr(feed, "entries", []):
            pub = _entry_time(entry)
            if pub is None or pub < cutoff:
                continue
            title = (getattr(entry, "title", "") or "").encode("ascii", "ignore").decode()
            summary = (getattr(entry, "summary", "") or "").encode("ascii", "ignore").decode()
            text = (title + " " + summary).lower()
            if any(kw in text for kw in keywords):
                matched.append((title, summary[:200]))

    articles_matched = len(matched)
    headlines = [f"• {t}: {s}" for t, s in matched[:10]]  # cap at 10

    if articles_matched == 0:
        quality = "insufficient"
    elif articles_matched < 2:
        quality = "low"
    elif articles_matched <= 5:
        quality = "medium"
    else:
        quality = "high"

    content = "\n".join(headlines) if headlines else "No matching headlines found."

    block = EvidenceBlock(
        source="rss",
        content=content,
        quality=quality,
        timestamp=datetime.now(timezone.utc),
        metadata={"headlines": headlines, "articles_matched": articles_matched},
    )
    _CACHE.set(key, block)
    return block

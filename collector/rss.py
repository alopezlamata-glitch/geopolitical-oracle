from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import aiohttp
import feedparser

from .base import RawEvent, new_event_id, graceful_collector, load_cached, save_cached, save_raw_events

logger = logging.getLogger(__name__)

_FEEDS = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.france24.com/en/rss",
]
_WINDOW_HOURS = 72

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
    "from", "before", "after", "during", "about", "have", "has", "had",
}


def _extract_keywords(query: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
    return [w for w in tokens if len(w) >= 3 and w not in _STOP_WORDS]


def _entry_time(entry) -> datetime | None:
    t = getattr(entry, "published_parsed", None)
    if t is None:
        return None
    import calendar
    return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)


def _safe_str(s) -> str:
    return (s or "").encode("ascii", "ignore").decode()


def _parse_feed(url: str):
    return feedparser.parse(url)


@graceful_collector("rss")
async def collect_rss(session: aiohttp.ClientSession, query: str) -> list[RawEvent]:
    keywords = _extract_keywords(query)
    cache_key = "rss_" + "_".join(keywords[:3])
    cached = load_cached(cache_key)
    if cached is not None:
        logger.debug("rss: cache hit (%d events)", len(cached))
        return cached

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_WINDOW_HOURS)
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _parse_feed, url) for url in _FEEDS]
    feeds = await asyncio.gather(*tasks, return_exceptions=True)

    events: list[RawEvent] = []
    for feed in feeds:
        if isinstance(feed, Exception):
            logger.warning("rss: feed error: %s", feed)
            continue
        for entry in getattr(feed, "entries", []):
            pub = _entry_time(entry)
            if pub is None or pub < cutoff:
                continue
            title = _safe_str(getattr(entry, "title", ""))
            summary = _safe_str(getattr(entry, "summary", ""))[:300]
            text = (title + " " + summary).lower()
            if not any(kw in text for kw in keywords):
                continue
            events.append(RawEvent(
                event_id=new_event_id(),
                source="rss",
                published_at=pub,
                title=title,
                url=getattr(entry, "link", "") or "",
                tone=0.0,
                country="",
                event_type="",
                sub_event_type="",
                actors=[],
                fatalities=0,
                themes=[],
                notes=summary,
                location={},
                raw_metadata={"feed": getattr(feed, "feed", {}).get("title", "")},
            ))

    logger.info("rss: collected %d matching events", len(events))
    if events:
        save_raw_events("rss", events)
        save_cached(cache_key, events)
    return events

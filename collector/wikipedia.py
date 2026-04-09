from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp

from .base import EvidenceBlock, graceful_collector

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_EXTRACT_MAX_CHARS = 1500
_TIMEOUT = aiohttp.ClientTimeout(total=8)


@graceful_collector("wikipedia")
async def collect_wikipedia(session: aiohttp.ClientSession, query: str) -> EvidenceBlock | None:
    # Step 1: search for top page
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    async with session.get(_SEARCH_URL, params=search_params, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        search_data = await resp.json()

    search_results = search_data.get("query", {}).get("search", [])
    if not search_results:
        return EvidenceBlock(
            source="wikipedia",
            content="",
            quality="low",
            timestamp=datetime.now(timezone.utc),
            metadata={"page_title": "", "page_url": ""},
        )

    page_title = search_results[0]["title"]

    # Step 2: fetch introductory extract
    extract_params = {
        "action": "query",
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "titles": page_title,
        "format": "json",
    }
    async with session.get(_SEARCH_URL, params=extract_params, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        extract_data = await resp.json()

    pages = extract_data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = page.get("extract", "").strip()

    if not extract:
        return EvidenceBlock(
            source="wikipedia",
            content="",
            quality="low",
            timestamp=datetime.now(timezone.utc),
            metadata={"page_title": page_title, "page_url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"},
        )

    truncated = extract[:_EXTRACT_MAX_CHARS]
    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"

    return EvidenceBlock(
        source="wikipedia",
        content=truncated,
        quality="medium",
        timestamp=datetime.now(timezone.utc),
        metadata={"page_title": page_title, "page_url": page_url},
    )

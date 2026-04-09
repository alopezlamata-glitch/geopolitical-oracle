from __future__ import annotations

import logging
import re
from typing import Optional

import aiohttp

from .base import RawEvent, new_event_id, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.metaculus.com/api2/questions/"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


@graceful_collector("metaculus")
async def collect_metaculus(session: aiohttp.ClientSession, query: str) -> tuple[Optional[float], int]:
    """
    Returns (community_prediction, num_forecasters) or (None, 0) if unavailable.
    Not returning RawEvents — Metaculus provides market probability, not events.
    """
    keywords = " ".join(
        w for w in re.sub(r"[^\w\s]", " ", query.lower()).split()
        if len(w) >= 3
    )[:80]

    params = {
        "search": keywords,
        "status": "active",
        "type": "forecast",
        "limit": "20",
    }

    async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
        if resp.status in (403, 429):
            logger.warning("metaculus: HTTP %s", resp.status)
            return None, 0
        resp.raise_for_status()
        data = await resp.json()

    results = data.get("results", [])
    for q in results:
        forecasters = q.get("number_of_forecasters") or q.get("forecasters_count") or 0
        if forecasters < 30:
            continue
        # Try multiple prediction paths
        for path in [
            ("community_prediction", "full", "q2"),
            ("community_prediction", "q2"),
            ("prediction",),
        ]:
            obj = q
            for key in path:
                obj = obj.get(key) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if isinstance(obj, (int, float)):
                p = float(obj)
                if 0.0 < p < 1.0:
                    logger.info("metaculus: found p=%.3f (%d forecasters)", p, forecasters)
                    return p, forecasters

    return None, 0

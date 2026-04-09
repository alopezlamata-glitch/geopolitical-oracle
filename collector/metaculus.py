from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp

from .base import EvidenceBlock, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.metaculus.com/api2/questions/"
_MIN_FORECASTERS = 30
_TIMEOUT = aiohttp.ClientTimeout(total=8)


@graceful_collector("metaculus")
async def collect_metaculus(session: aiohttp.ClientSession, query: str) -> EvidenceBlock | None:
    params = {
        "search": query,
        "status": "active",
        "limit": 20,
    }
    headers = {"Accept": "application/json"}

    async with session.get(_BASE_URL, params=params, headers=headers, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        data = await resp.json()

    questions = data.get("results", [])
    if not questions:
        return EvidenceBlock(
            source="metaculus",
            content="No Metaculus questions found for this query.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    # Pick best question: most forecasters
    best = max(questions, key=lambda q: q.get("number_of_forecasters") or 0)
    num_forecasters = best.get("number_of_forecasters") or 0
    title = best.get("title") or best.get("question", {}).get("title", "Unknown")
    close_time = best.get("close_time") or best.get("question", {}).get("close_time", "")

    # Extract community prediction (median)
    cp = _extract_prediction(best)

    if num_forecasters < _MIN_FORECASTERS or cp is None:
        return EvidenceBlock(
            source="metaculus",
            content=f"Metaculus question found ('{title}') but insufficient forecasters ({num_forecasters}) or no prediction.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={"probability": None, "forecasters": num_forecasters, "close_time": close_time},
        )

    content = (
        f"Metaculus: '{title}'\n"
        f"Community prediction: {cp:.1%} | Forecasters: {num_forecasters} | Closes: {close_time}"
    )

    return EvidenceBlock(
        source="metaculus",
        content=content,
        quality="high" if num_forecasters > 100 else "medium",
        timestamp=datetime.now(timezone.utc),
        metadata={"probability": cp, "forecasters": num_forecasters, "close_time": close_time},
    )


def _extract_prediction(question: dict) -> float | None:
    """Safely extract the median community prediction from a Metaculus question."""
    # Try multiple known field paths
    cp = question.get("community_prediction")
    if cp is None:
        cp = question.get("question", {}).get("community_prediction")

    if cp is None:
        return None

    if isinstance(cp, (int, float)):
        return float(cp)

    if isinstance(cp, dict):
        # Try nested paths
        for path in [("full", "q2"), ("q2",)]:
            val = cp
            for key in path:
                if not isinstance(val, dict):
                    val = None
                    break
                val = val.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue

    return None

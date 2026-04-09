from __future__ import annotations

import logging
import functools
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class EvidenceBlock:
    source: str
    content: str
    quality: Literal["high", "medium", "low", "insufficient"]
    timestamp: datetime
    metadata: dict = field(default_factory=dict)


class TTLCache:
    """Simple in-process TTL cache. Not thread-safe, not needed for single-threaded asyncio."""

    def __init__(self, ttl_minutes: int = 30):
        self._store: dict[str, tuple[Any, datetime]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if datetime.utcnow() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, datetime.utcnow() + self._ttl)


def graceful_collector(collector_name: str):
    """Decorator: catch all exceptions from a collector, log, return None."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except asyncio.TimeoutError:
                logger.warning("%s: timed out after 8s", collector_name)
                return None
            except aiohttp.ClientError as e:
                logger.warning("%s: HTTP error: %s", collector_name, e)
                return None
            except Exception as e:
                logger.warning("%s: unexpected error: %s", collector_name, e)
                return None
        return wrapper
    return decorator

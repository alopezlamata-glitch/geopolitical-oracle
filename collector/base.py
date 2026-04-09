from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RAW_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "cache"


@dataclass
class RawEvent:
    event_id: str
    source: str              # "gdelt" | "acled" | "rss"
    published_at: datetime
    title: str
    url: str = ""
    tone: float = 0.0
    country: str = ""
    event_type: str = ""
    sub_event_type: str = ""
    actors: list[str] = field(default_factory=list)
    fatalities: int = 0
    themes: list[str] = field(default_factory=list)
    notes: str = ""
    location: dict = field(default_factory=dict)
    raw_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RawEvent":
        d = dict(d)
        if isinstance(d.get("published_at"), str):
            d["published_at"] = datetime.fromisoformat(d["published_at"])
        return cls(**d)


def new_event_id() -> str:
    return str(uuid.uuid4())


def save_raw_events(source: str, events: list[RawEvent]) -> None:
    """Save raw events to immutable per-source storage."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = Path(__file__).parent.parent / "data" / "raw" / source / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    for ev in events:
        path = out_dir / f"{ev.event_id}.json"
        if not path.exists():
            path.write_text(json.dumps(ev.to_dict(), ensure_ascii=False, indent=2))


def load_cached(cache_key: str) -> Optional[list[RawEvent]]:
    path = _RAW_CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        ts = datetime.fromisoformat(data["timestamp"])
        age_minutes = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age_minutes > 30:
            return None
        return [RawEvent.from_dict(e) for e in data["events"]]
    except Exception:
        return None


def save_cached(cache_key: str, events: list[RawEvent]) -> None:
    _RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _RAW_CACHE_DIR / f"{cache_key}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events": [e.to_dict() for e in events],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def graceful_collector(name: str):
    """Decorator: catch all exceptions in a collector, log, return []."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except asyncio.TimeoutError:
                logger.warning("%s: timed out", name)
                return []
            except Exception as e:
                logger.warning("%s: error: %s", name, e)
                return []
        return wrapper
    return decorator

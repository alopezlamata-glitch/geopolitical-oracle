from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from collector.base import RawEvent

# ─── Taxonomy ────────────────────────────────────────────────────────────────

EVENT_TYPES = {
    "military_action",
    "protest",
    "diplomatic_statement",
    "sanction",
    "ceasefire_signal",
    "political_crisis",
    "economic_shock",
    "humanitarian",
    "cyber",
    "other",
}

# GDELT theme prefixes → canonical event type
_GDELT_THEME_MAP: list[tuple[str, str]] = [
    ("MILITARY", "military_action"),
    ("TERROR", "military_action"),
    ("ATTACK", "military_action"),
    ("WAR", "military_action"),
    ("PROTEST", "protest"),
    ("UNREST", "protest"),
    ("RIOT", "protest"),
    ("STRIKE", "protest"),
    ("CEASEFIRE", "ceasefire_signal"),
    ("PEACE", "ceasefire_signal"),
    ("TRUCE", "ceasefire_signal"),
    ("NEGOTIAT", "ceasefire_signal"),
    ("SANCTION", "sanction"),
    ("EMBARGO", "sanction"),
    ("DIPLOMAT", "diplomatic_statement"),
    ("MEET", "diplomatic_statement"),
    ("SUMMIT", "diplomatic_statement"),
    ("ELECTION", "political_crisis"),
    ("COUP", "political_crisis"),
    ("CRISIS", "political_crisis"),
    ("ECONOM", "economic_shock"),
    ("INFLATION", "economic_shock"),
    ("RECESSION", "economic_shock"),
    ("HUMANITAR", "humanitarian"),
    ("REFUGEE", "humanitarian"),
    ("FAMINE", "humanitarian"),
    ("CYBER", "cyber"),
    ("HACK", "cyber"),
]

# ACLED event_type → canonical
_ACLED_TYPE_MAP: dict[str, str] = {
    "Battles": "military_action",
    "Explosions/Remote violence": "military_action",
    "Violence against civilians": "military_action",
    "Protests": "protest",
    "Riots": "protest",
    "Strategic developments": "diplomatic_statement",
    "Non-violent actions": "diplomatic_statement",
}

# Polarity ranges per event type: (base_polarity)
_BASE_POLARITY: dict[str, float] = {
    "military_action": -0.8,
    "protest": -0.4,
    "diplomatic_statement": 0.1,
    "sanction": -0.6,
    "ceasefire_signal": 0.6,
    "political_crisis": -0.5,
    "economic_shock": -0.5,
    "humanitarian": -0.3,
    "cyber": -0.6,
    "other": 0.0,
}

# ─── Dataclass ───────────────────────────────────────────────────────────────


@dataclass
class CanonicalEvent:
    event_id: str
    doc_ids: list[str]
    source: str
    occurred_at: datetime
    event_type: str
    sub_event_type: str
    actors: list[str]
    country: str
    severity: float             # 0.0 to 1.0
    polarity: float             # -1.0 to +1.0
    fatalities: int
    independent_sources: int    # how many sources reported this
    contradiction_score: float  # 0.0 consistent, 1.0 contradictory
    raw_title: str = ""

    def to_dict(self) -> dict:
        d = {
            "event_id": self.event_id,
            "doc_ids": self.doc_ids,
            "source": self.source,
            "occurred_at": self.occurred_at.isoformat(),
            "event_type": self.event_type,
            "sub_event_type": self.sub_event_type,
            "actors": self.actors,
            "country": self.country,
            "severity": self.severity,
            "polarity": self.polarity,
            "fatalities": self.fatalities,
            "independent_sources": self.independent_sources,
            "contradiction_score": self.contradiction_score,
            "raw_title": self.raw_title,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalEvent":
        d = dict(d)
        if isinstance(d.get("occurred_at"), str):
            d["occurred_at"] = datetime.fromisoformat(d["occurred_at"])
        return cls(**d)


# ─── Type inference ──────────────────────────────────────────────────────────


def _infer_type_from_gdelt(ev: RawEvent) -> str:
    for theme in ev.themes:
        theme_upper = theme.upper()
        for prefix, canonical in _GDELT_THEME_MAP:
            if theme_upper.startswith(prefix):
                return canonical
    # fallback: scan title
    title_lower = ev.title.lower()
    for prefix, canonical in _GDELT_THEME_MAP:
        if prefix.lower() in title_lower:
            return canonical
    return "other"


def _infer_type_from_acled(ev: RawEvent) -> str:
    return _ACLED_TYPE_MAP.get(ev.event_type, "other")


def _infer_type_from_rss(ev: RawEvent) -> str:
    title_lower = ev.title.lower()
    kw_map = [
        (["attack", "missile", "bomb", "airstrike", "strike", "troops", "military", "soldier"], "military_action"),
        (["protest", "march", "demonstrat", "riot", "unrest"], "protest"),
        (["ceasefire", "peace talks", "truce", "negotiat", "deal"], "ceasefire_signal"),
        (["sanction", "embargo", "restrict"], "sanction"),
        (["diplomat", "summit", "meeting", "talk", "foreign minister"], "diplomatic_statement"),
        (["coup", "election", "political crisis", "resign"], "political_crisis"),
        (["economy", "inflation", "recession", "trade war", "tariff"], "economic_shock"),
        (["refugee", "humanitarian", "famine", "aid"], "humanitarian"),
        (["hack", "cyber", "ransomware"], "cyber"),
    ]
    for keywords, canonical in kw_map:
        if any(kw in title_lower for kw in keywords):
            return canonical
    return "other"


# ─── Severity ────────────────────────────────────────────────────────────────


def _compute_severity(ev: RawEvent, event_type: str) -> float:
    if ev.source == "acled":
        if ev.fatalities > 100:
            return 1.0
        elif ev.fatalities > 10:
            return 0.7
        elif ev.fatalities > 0:
            return 0.4
        else:
            return 0.2
    elif ev.source == "gdelt":
        return min(1.0, abs(ev.tone) / 100.0)
    else:
        # RSS: use event type heuristic
        type_severity = {
            "military_action": 0.6,
            "political_crisis": 0.5,
            "sanction": 0.4,
            "protest": 0.3,
            "ceasefire_signal": 0.3,
            "diplomatic_statement": 0.2,
            "economic_shock": 0.4,
            "humanitarian": 0.4,
            "cyber": 0.5,
            "other": 0.2,
        }
        return type_severity.get(event_type, 0.2)


# ─── Polarity ────────────────────────────────────────────────────────────────


def _compute_polarity(ev: RawEvent, event_type: str) -> float:
    base = _BASE_POLARITY.get(event_type, 0.0)
    # Modulate with GDELT tone (-100 to +100)
    if ev.source == "gdelt" and ev.tone != 0.0:
        tone_normalized = max(-1.0, min(1.0, ev.tone / 50.0))
        # blend: 60% base, 40% tone signal
        return max(-1.0, min(1.0, 0.6 * base + 0.4 * tone_normalized))
    return base


# ─── Main normalizer ─────────────────────────────────────────────────────────


def normalize(raw_event: RawEvent) -> CanonicalEvent:
    """Convert a RawEvent to a CanonicalEvent."""
    if raw_event.source == "gdelt":
        event_type = _infer_type_from_gdelt(raw_event)
    elif raw_event.source == "acled":
        event_type = _infer_type_from_acled(raw_event)
    else:
        event_type = _infer_type_from_rss(raw_event)

    severity = _compute_severity(raw_event, event_type)
    polarity = _compute_polarity(raw_event, event_type)

    return CanonicalEvent(
        event_id=raw_event.event_id,
        doc_ids=[raw_event.event_id],
        source=raw_event.source,
        occurred_at=raw_event.published_at,
        event_type=event_type,
        sub_event_type=raw_event.sub_event_type,
        actors=raw_event.actors,
        country=raw_event.country,
        severity=severity,
        polarity=polarity,
        fatalities=raw_event.fatalities,
        independent_sources=1,
        contradiction_score=0.0,
        raw_title=raw_event.title,
    )


def normalize_all(raw_events: list[RawEvent]) -> list[CanonicalEvent]:
    return [normalize(ev) for ev in raw_events]

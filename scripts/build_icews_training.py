"""
Build training data from ICEWS (Integrated Crisis Early Warning System) historical events.
ICEWS is a public dataset of 20M+ coded geopolitical events from 1995-present.
Source: Harvard Dataverse doi:10.7910/DVN/28075 (free, no auth required)

This script:
  1. Downloads ICEWS 2021 + 2022 event files (~50MB total)
  2. Parses events into our feature schema
  3. Slides a 30-day window across each country/month
  4. Labels each window: did a CAMEO "military attack" (code 19*) happen in the next 30 days?
  5. Saves labeled examples to data/training/

Run: python scripts/build_icews_training.py
Expected output: ~500-2000 labeled training examples
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("icews_builder")

_TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "icews"

# ICEWS file IDs on Harvard Dataverse (2020, 2021, 2022)
_ICEWS_FILES = [
    ("events.2020.20210329.tab.zip", 6091024),
    ("events.2021.20220623.tab.zip", 6352620),
    ("events.2022.20230106.tab.zip", 6880739),
]

# CAMEO event codes → our taxonomy
# Full CAMEO codebook: https://parusanalytics.com/eventdata/cameo.dir/CAMEO.Manual.1.1b3.pdf
_CAMEO_TO_TYPE = {
    # Military action: 13x (threats), 14x (protest/demo — skip), 17x (coerce), 18x (assault), 19x (fight/attack)
    "13": "military_action",   # Threaten
    "17": "military_action",   # Coerce
    "18": "military_action",   # Assault
    "19": "military_action",   # Fight / Use conventional military force
    # Protest: 14x
    "14": "protest",
    # Diplomatic: 02x, 03x, 04x, 05x (verbal cooperation/statement)
    "02": "diplomatic_statement",
    "03": "diplomatic_statement",
    "04": "diplomatic_statement",
    "05": "diplomatic_statement",
    # Ceasefire / de-escalation: 06x (engage in material cooperation), 07x (provide aid), 12x (yield)
    "06": "ceasefire_signal",
    "12": "ceasefire_signal",
    # Sanction: 16x (reduce relations)
    "16": "sanction",
    # Economic: 15x (exhibit force posture) — treated as political_crisis when paired with economic context
    "15": "political_crisis",
    # Humanitarian: 07x (provide humanitarian aid)
    "07": "humanitarian",
}

# Countries to sample — mix of high-conflict AND stable countries for balance
_TARGET_COUNTRIES = {
    # High-conflict (likely YES)
    "Syria", "Iraq", "Yemen", "Afghanistan", "Ukraine",
    "Israel", "Libya", "Sudan", "Somalia", "Mali", "Myanmar",
    # Medium (mixed signal)
    "Ethiopia", "Pakistan", "Nigeria", "Turkey", "Iran",
    # Stable (likely NO — needed for class balance)
    "Germany", "France", "Japan", "Canada", "Australia",
    "Sweden", "Norway", "Netherlands", "New Zealand", "Portugal",
    "Denmark", "Finland", "Switzerland", "Austria", "Belgium",
}

_WINDOW_DAYS = 30       # feature window
_FORECAST_DAYS = 30     # outcome window: did escalation happen in next N days?
_DECAY_HALF = 7.0       # days for time-decay weight
_MIN_EVENTS = 5         # skip windows with too few events (not enough signal)
_MAX_PER_COUNTRY_YEAR = 12   # cap samples per country/year to balance dataset


def _download(file_id: int, filename: str) -> Path:
    """Download ICEWS zip from Harvard Dataverse if not already cached."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tab_name = filename.replace(".zip", "")
    tab_path = _CACHE_DIR / tab_name
    if tab_path.exists():
        logger.info("cache hit: %s", tab_name)
        return tab_path

    zip_path = _CACHE_DIR / filename
    if not zip_path.exists():
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        logger.info("downloading %s from Harvard Dataverse (~25MB)...", filename)
        req = urllib.request.Request(url, headers={"User-Agent": "geopolitical-oracle/1.0 (research)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        zip_path.write_bytes(data)
        logger.info("saved %s (%.1fMB)", filename, len(data) / 1e6)

    logger.info("extracting %s...", filename)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(tab_name, _CACHE_DIR)
    return tab_path


def _cameo_to_type(code: str) -> str:
    """Map CAMEO event code to our taxonomy. Code is e.g. '190' or '18'."""
    if not code:
        return "other"
    # Try 2-digit prefix first
    prefix2 = code[:2]
    if prefix2 in _CAMEO_TO_TYPE:
        return _CAMEO_TO_TYPE[prefix2]
    prefix1 = code[:1]
    if prefix1 in _CAMEO_TO_TYPE:
        return _CAMEO_TO_TYPE[prefix1]
    return "other"


def _parse_tab(tab_path: Path) -> list[dict]:
    """
    Parse ICEWS .tab file into list of event dicts.
    ICEWS columns (tab-separated):
      Event ID, Event Date, Source Name, Source Sectors, Source Country,
      Event Text, CAMEO Code, Intensity, Target Name, Target Sectors,
      Target Country, Story ID, Sentence Number, Publisher, City,
      District, Province, Country, Latitude, Longitude
    """
    events = []
    logger.info("parsing %s...", tab_path.name)
    with open(tab_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            country = (row.get("Country") or row.get("Target Country") or "").strip()
            if country not in _TARGET_COUNTRIES:
                continue
            date_str = (row.get("Event Date") or "").strip()
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            cameo = (row.get("CAMEO Code") or "").strip()
            intensity = 0.0
            try:
                intensity = float(row.get("Intensity") or 0)
            except (ValueError, TypeError):
                pass
            events.append({
                "date": dt,
                "country": country,
                "event_type": _cameo_to_type(cameo),
                "cameo": cameo,
                "intensity": intensity,
                "source_country": (row.get("Source Country") or "").strip(),
                "target_country": (row.get("Target Country") or "").strip(),
            })
    logger.info("parsed %d events for target countries", len(events))
    return events


def _decay(delta_days: float) -> float:
    return math.exp(-delta_days / _DECAY_HALF)


def _build_features(events: list[dict], window_end: datetime) -> dict[str, float]:
    """Build feature vector from events in [window_end - 30d, window_end]."""
    w30 = window_end - timedelta(days=30)
    w7  = window_end - timedelta(days=7)
    w3  = window_end - timedelta(days=3)
    w_prior4 = window_end - timedelta(days=7)

    evs30 = [e for e in events if w30 <= e["date"] < window_end]
    evs7  = [e for e in evs30 if e["date"] >= w7]
    evs3  = [e for e in evs30 if e["date"] >= w3]
    evs_p4 = [e for e in evs30 if w_prior4 <= e["date"] < w3]

    def cnt(evlist, etype):
        return sum(1 for e in evlist if e["event_type"] == etype)

    def intensity(evlist, etype=None):
        total = 0.0
        for e in evlist:
            if etype and e["event_type"] != etype:
                continue
            delta = (window_end - e["date"]).total_seconds() / 86400
            w = _decay(delta) * (1.0 + abs(e["intensity"]) / 10.0)
            total += w
        return total

    def polarity(evlist):
        pol_map = {
            "military_action": -0.8, "protest": -0.4, "sanction": -0.6,
            "political_crisis": -0.5, "ceasefire_signal": 0.6,
            "diplomatic_statement": 0.1, "humanitarian": -0.2, "other": 0.0,
        }
        total_w, total_pol = 0.0, 0.0
        for e in evlist:
            delta = (window_end - e["date"]).total_seconds() / 86400
            w = _decay(delta)
            total_pol += pol_map.get(e["event_type"], 0.0) * w
            total_w += w
        return total_pol / (total_w + 1e-9)

    mil7 = cnt(evs7, "military_action")
    mil30 = cnt(evs30, "military_action")
    pro7 = cnt(evs7, "protest")
    pro30 = cnt(evs30, "protest")
    cease7 = cnt(evs7, "ceasefire_signal")
    mil_accel = cnt(evs3, "military_action") / (cnt(evs_p4, "military_action") + 0.1)
    pro_accel = cnt(evs3, "protest") / (cnt(evs_p4, "protest") + 0.1)
    all_accel = len(evs3) / (len(evs_p4) + 0.1)
    mil_int7 = intensity(evs7, "military_action")
    pro_int7 = intensity(evs7, "protest")
    overall_int7 = intensity(evs7)
    pol7 = polarity(evs7)
    pol30 = polarity(evs30)
    pol_trend = polarity(evs3) - polarity(evs_p4)
    sources = {e.get("source_country", "?") for e in evs7}
    src_div = min(1.0, len(sources) / max(len(evs7), 1))

    return {
        "military_count_7d": float(mil7),
        "military_count_30d": float(mil30),
        "protest_count_7d": float(pro7),
        "protest_count_30d": float(pro30),
        "diplomatic_count_7d": float(cnt(evs7, "diplomatic_statement")),
        "ceasefire_count_7d": float(cease7),
        "sanction_count_7d": float(cnt(evs7, "sanction")),
        "political_crisis_count_7d": float(cnt(evs7, "political_crisis")),
        "military_intensity_7d": mil_int7,
        "protest_intensity_7d": pro_int7,
        "overall_intensity_7d": overall_int7,
        "military_accel": mil_accel,
        "protest_accel": pro_accel,
        "overall_accel": all_accel,
        "avg_polarity_7d": pol7,
        "avg_polarity_30d": pol30,
        "tone_trend": pol_trend,
        "source_diversity_7d": src_div,
        "avg_independent_sources": min(3.0, len(sources) / max(len(evs7), 1) * 3),
        "avg_contradiction_score": 0.1,
        "fatalities_7d": 0.0,   # ICEWS doesn't include fatalities
        "has_military_7d": float(mil7 > 0),
        "has_ceasefire_7d": float(cease7 > 0),
        "escalation_index": mil_int7 - cease7 * 0.3,
        "metaculus_p": -1.0,
        "polymarket_p": -1.0,
        "market_available": 0.0,
    }


def _is_escalation(events: list[dict], window_start: datetime, window_end: datetime) -> int:
    """
    Outcome: major military escalation in [window_start, window_end].
    Defined as: cumulative intensity of military events >= 30.0
    (CAMEO max intensity ~10 per event; 30 = ~3 major armed attacks).
    This threshold gives ~35-45% base rate across conflict/stable country mix.
    """
    total_intensity = sum(
        abs(e["intensity"])
        for e in events
        if window_start <= e["date"] < window_end
        and e["event_type"] == "military_action"
    )
    return int(total_intensity >= 30.0)


def _build_question(country: str, window_end: datetime) -> str:
    month = window_end.strftime("%B %Y")
    return f"Will there be a military escalation in {country} before {month}?"


def main():
    _TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    all_events_by_country: dict[str, list[dict]] = defaultdict(list)

    # Download and parse ICEWS files
    for filename, file_id in _ICEWS_FILES:
        try:
            tab_path = _download(file_id, filename)
            events = _parse_tab(tab_path)
            for e in events:
                all_events_by_country[e["country"]].append(e)
        except Exception as ex:
            logger.error("failed to process %s: %s", filename, ex)
            continue

    if not all_events_by_country:
        logger.error("No events loaded. Check network connectivity.")
        return

    total_countries = len(all_events_by_country)
    logger.info("loaded events for %d countries", total_countries)

    # Slide window across each country
    examples = []
    for country, events in all_events_by_country.items():
        events.sort(key=lambda e: e["date"])
        if not events:
            continue

        first_date = events[0]["date"]
        last_date = events[-1]["date"]

        # Sample one window per month
        count_this_country = 0
        current = first_date + timedelta(days=30)
        while current < last_date - timedelta(days=30):
            window_events = [e for e in events if (current - timedelta(days=30)) <= e["date"] < current]
            if len(window_events) < _MIN_EVENTS:
                current += timedelta(days=30)
                continue

            features = _build_features(window_events, current)
            outcome = _is_escalation(
                events,
                window_start=current,
                window_end=current + timedelta(days=_FORECAST_DAYS),
            )
            question = _build_question(country, current + timedelta(days=_FORECAST_DAYS))
            examples.append({
                "question": question,
                "timestamp": current.isoformat(),
                "country": country,
                "features": features,
                "outcome": outcome,
                "source": "icews_historical",
            })
            count_this_country += 1
            if count_this_country >= _MAX_PER_COUNTRY_YEAR * len(_ICEWS_FILES):
                break
            current += timedelta(days=30)

    logger.info("generated %d labeled examples (before balancing)", len(examples))

    # Balance: undersample YES to 2x the number of NO examples
    import random as _random
    _random.seed(42)
    no_examples  = [e for e in examples if e["outcome"] == 0]
    yes_examples = [e for e in examples if e["outcome"] == 1]
    target_yes = min(len(yes_examples), max(len(no_examples) * 2, len(no_examples)))
    if len(yes_examples) > target_yes:
        yes_examples = _random.sample(yes_examples, target_yes)
    examples = sorted(yes_examples + no_examples, key=lambda e: e["timestamp"])
    logger.info("after balancing: %d YES + %d NO = %d total",
                len(yes_examples), len(no_examples), len(examples))

    # Save
    saved = 0
    outcomes = {0: 0, 1: 0}
    for ex in examples:
        ts = datetime.fromisoformat(ex["timestamp"]).strftime("%Y%m%dT%H%M%S")
        country_slug = ex["country"].lower().replace(" ", "_")
        path = _TRAINING_DIR / f"{ts}_{country_slug}_icews.json"
        if not path.exists():
            path.write_text(json.dumps(ex, ensure_ascii=False, indent=2))
            saved += 1
        outcomes[ex["outcome"]] += 1

    total = len(examples)
    yes_pct = outcomes[1] / total if total else 0
    print(f"\n✓ Saved {saved} new training examples to {_TRAINING_DIR}")
    print(f"  YES (escalation): {outcomes[1]} ({yes_pct:.0%})")
    print(f"  NO  (stable):     {outcomes[0]} ({1-yes_pct:.0%})")
    print(f"  Countries: {len(all_events_by_country)}")
    print(f"\nNow run: python main.py train")


if __name__ == "__main__":
    main()

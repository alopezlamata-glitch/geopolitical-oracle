from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"


def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text.lower())[:50].strip("_")


def save_features(
    question: str,
    features: dict[str, float],
    provenance: dict[str, list[dict]],
    event_ids_used: list[str],
) -> Path:
    slug = _slug(question)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = _FEATURES_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ts}.json"
    payload = {
        "question": question,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "provenance": provenance,
        "event_ids_used": event_ids_used,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def load_latest_features(question: str) -> Optional[dict[str, Any]]:
    slug = _slug(question)
    out_dir = _FEATURES_DIR / slug
    if not out_dir.exists():
        return None
    files = sorted(out_dir.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())

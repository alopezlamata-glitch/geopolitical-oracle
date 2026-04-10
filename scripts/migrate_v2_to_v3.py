"""
Migrate training data from v2 (32 features) → v3 (27 features).

Changes:
  REMOVE (8): political_crisis_count_7d, avg_contradiction_score, fatalities_7d,
              metaculus_p, metaculus_available, polymarket_p, polymarket_available,
              market_available
  ADD (3):    ceasefire_ratio_7d, event_velocity_7d, military_share_7d
              (computed from already-present raw counts)

Run: python scripts/migrate_v2_to_v3.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"

_REMOVE = {
    "political_crisis_count_7d",
    "avg_contradiction_score",
    "fatalities_7d",
    "metaculus_p",
    "metaculus_available",
    "polymarket_p",
    "polymarket_available",
    "market_available",
}


def _add_derived(features: dict) -> dict:
    """Compute the 3 new derived features from existing raw counts."""
    mil7  = features.get("military_count_7d", 0.0)
    total7 = (
        features.get("military_count_7d", 0.0)
        + features.get("protest_count_7d", 0.0)
        + features.get("diplomatic_count_7d", 0.0)
        + features.get("ceasefire_count_7d", 0.0)
        + features.get("sanction_count_7d", 0.0)
        + features.get("political_crisis_count_7d", 0.0)
    )
    cease7 = features.get("ceasefire_count_7d", 0.0)
    # event_velocity_7d: approximate 7d count / (30d count / 4.3)
    # We don't have total 30d directly, use mil30+pro30 as proxy
    total30 = features.get("military_count_30d", 0.0) + features.get("protest_count_30d", 0.0)
    # Estimate total_30d: scale up if we have more than mil+pro in 7d
    # (rough but correct order-of-magnitude)
    total30_est = max(total30, total7)  # 30d must be >= 7d counts

    features["ceasefire_ratio_7d"] = cease7 / (mil7 + 0.1)
    features["event_velocity_7d"]  = total7 / (total30_est / 4.3 + 0.1)
    features["military_share_7d"]  = mil7   / (total7 + 0.1)
    return features


def migrate_file(path: Path, dry_run: bool = False) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", {})

    # Check which version we're working with
    n_before = len(features)

    # Skip if already migrated (no old keys present)
    if not any(k in features for k in _REMOVE):
        return "skip"

    # Remove dead features
    for k in _REMOVE:
        features.pop(k, None)

    # Add derived features (idempotent)
    _add_derived(features)

    data["features"] = features
    data["feature_schema_version"] = "v3"

    if not dry_run:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return f"migrated ({n_before} → {len(features)} features)"


def main():
    files = sorted(_TRAINING_DIR.glob("*.json"))
    if not files:
        print(f"No training files found in {_TRAINING_DIR}")
        return

    migrated = skipped = errors = 0
    for path in files:
        try:
            result = migrate_file(path)
            if result == "skip":
                skipped += 1
            else:
                migrated += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR {path.name}: {e}")

    print(f"Done: {migrated} migrated, {skipped} already v3, {errors} errors")
    print(f"Total files: {len(files)}")


if __name__ == "__main__":
    main()

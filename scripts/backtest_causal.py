"""
Causal propagation backtest — validate whether causal links in data/causal_links.json
actually improve predictions vs ignoring them.

Method:
  For each causal link (source_entity, target_entity, feature, weight):
    1. Find resolved predictions for the TARGET entity
    2. Split by whether the SOURCE entity's feature was above median at prediction time
    3. Compare actual outcome rates: high_source vs low_source
    4. Compute lift = P(outcome=1 | high_source) / P(outcome=1 | low_source)
    5. Links with lift < 0.8 or > 1.2 are considered informative; others neutral

Outputs:
  Console table + data/model/causal_backtest.json

Usage:
  python scripts/backtest_causal.py
  python main.py backtest-causal
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_causal")

_CAUSAL_PATH  = Path(__file__).parent.parent / "data" / "causal_links.json"
_OUT_PATH     = Path(__file__).parent.parent / "data" / "model" / "causal_backtest.json"
_MIN_SAMPLES  = 5   # minimum resolved predictions per link for backtest


def _load_causal_links() -> list[dict]:
    if not _CAUSAL_PATH.exists():
        logger.warning("causal_links.json not found at %s", _CAUSAL_PATH)
        return []
    try:
        data = json.loads(_CAUSAL_PATH.read_text())
        if isinstance(data, list):
            return data
        return data.get("links", [])
    except Exception as e:
        logger.error("failed to load causal_links.json: %s", e)
        return []


def _load_resolved_with_features() -> list[dict]:
    """
    Load resolved predictions with their feature snapshots from the lakehouse.

    Returns list of dicts with: features (dict), outcome (int), entity/country, domain.
    """
    try:
        from data_layer.db import get_db, init_schema, table_exists
        init_schema()
        if not table_exists("training_ready_snapshots"):
            return []

        db = get_db()
        rows = db.execute(
            """
            SELECT
                trs.explicit_features,
                CAST(trs.outcome AS INTEGER) AS outcome,
                COALESCE(trs.predicate, '') AS predicate,
                COALESCE(trs.event_family, 'conflict') AS domain,
                COALESCE(q.jurisdiction, '') AS country
            FROM training_ready_snapshots trs
            LEFT JOIN questions q ON trs.question_id = q.question_id
            """
        ).fetchall()

        results = []
        for raw_feats, outcome, predicate, domain, country in rows:
            feats = raw_feats if isinstance(raw_feats, dict) else json.loads(raw_feats or "{}")
            results.append({
                "features": feats,
                "outcome":  outcome,
                "predicate": predicate,
                "domain":    domain,
                "country":   country.lower(),
            })

        logger.info("loaded %d resolved snapshots for backtest", len(results))
        return results
    except Exception as e:
        logger.warning("_load_resolved_with_features failed: %s", e)
        return []


def _entity_matches(record: dict, entity: str) -> bool:
    """Fuzzy country/entity match."""
    entity_lower = entity.lower()
    country = record.get("country", "")
    return entity_lower in country or country in entity_lower


def _conditional_lift(
    records: list[dict],
    source_feature: str,
    target_entity: str,
    split_quantile: float = 0.5,
) -> Optional[dict]:
    """
    For records of the target entity, split on whether source_feature > median.
    Returns dict with lift, P(Y=1|high), P(Y=1|low), n_high, n_low.
    """
    target_records = [r for r in records if _entity_matches(r, target_entity)]
    if len(target_records) < _MIN_SAMPLES:
        return None

    values = [r["features"].get(source_feature, 0.0) for r in target_records]
    sorted_vals = sorted(values)
    threshold = sorted_vals[int(len(sorted_vals) * split_quantile)]

    high = [r for r, v in zip(target_records, values) if v >= threshold]
    low  = [r for r, v in zip(target_records, values) if v <  threshold]

    if len(high) < 2 or len(low) < 2:
        return None

    p_high = sum(r["outcome"] for r in high) / len(high)
    p_low  = sum(r["outcome"] for r in low)  / len(low)

    lift = (p_high / p_low) if p_low > 0.001 else None

    return {
        "p_high": round(p_high, 4),
        "p_low":  round(p_low,  4),
        "lift":   round(lift, 3) if lift is not None else None,
        "n_high": len(high),
        "n_low":  len(low),
        "threshold_feature_value": round(threshold, 4),
    }


def run(save: bool = True) -> dict:
    links = _load_causal_links()
    if not links:
        print("No causal links found. Expected: data/causal_links.json")
        return {"status": "no_links"}

    records = _load_resolved_with_features()

    if len(records) < 10:
        msg = (
            f"Only {len(records)} resolved snapshots — not enough for backtest "
            f"(need ≥10). Run:\n"
            f"  python main.py seed-training-data\n"
            f"  python main.py seed-markets\n"
            f"  python main.py train\n"
        )
        print(msg)
        return {"status": "insufficient_data", "n_records": len(records)}

    logger.info("backtesting %d causal links on %d records", len(links), len(records))

    results = []
    for link in links:
        source = link.get("source_entity", "")
        target = link.get("target_entity", "")
        feature = link.get("feature", "")
        declared_weight = float(link.get("weight", 0.0))
        decay_days = float(link.get("decay_days", 30.0))

        if not (source and target and feature):
            continue

        stat = _conditional_lift(records, feature, target)
        if stat is None:
            result = {
                "source": source,
                "target": target,
                "feature": feature,
                "declared_weight": declared_weight,
                "decay_days": decay_days,
                "status": "insufficient_data",
                "lift": None,
                "verdict": "skip",
            }
        else:
            lift = stat.get("lift")
            if lift is None:
                verdict = "skip"
            elif lift > 1.3:
                verdict = "validated_positive"
            elif lift < 0.77:
                verdict = "validated_inverse"
            elif 0.9 <= lift <= 1.1:
                verdict = "neutral"
            else:
                verdict = "weak"

            # Check if declared weight direction matches data
            if lift is not None and declared_weight != 0:
                data_direction = "+" if lift > 1.0 else "-"
                declared_direction = "+" if declared_weight > 0 else "-"
                direction_match = data_direction == declared_direction
            else:
                direction_match = None

            result = {
                "source": source,
                "target": target,
                "feature": feature,
                "declared_weight": declared_weight,
                "decay_days": decay_days,
                **stat,
                "verdict": verdict,
                "direction_match": direction_match,
            }
        results.append(result)

    n_validated = sum(1 for r in results if r["verdict"] in ("validated_positive", "validated_inverse"))
    n_neutral   = sum(1 for r in results if r["verdict"] == "neutral")
    n_skip      = sum(1 for r in results if r["verdict"] in ("skip", "insufficient_data"))

    report = {
        "n_links":     len(links),
        "n_backtested": len(results),
        "n_validated": n_validated,
        "n_neutral":   n_neutral,
        "n_skip":      n_skip,
        "n_records":   len(records),
        "links":       results,
    }

    if save:
        _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OUT_PATH.write_text(json.dumps(report, indent=2))
        logger.info("saved causal backtest to %s", _OUT_PATH)

    _print_report(report)
    return report


def _fmt_lift(v) -> str:
    if v is None:
        return "   -  "
    return f"{v:>6.3f}"


def _print_report(r: dict) -> None:
    W = 78

    def row(s): return f"│ {s:<{W}} │"
    def sep(): return "├─" + "─" * W + "─┤"

    lines = [
        "┌─" + "─" * W + "─┐",
        row("  CAUSAL LINK BACKTEST"),
        row(f"  {r['n_links']} declared links | {r['n_records']} resolved snapshots"),
        row(f"  validated={r['n_validated']}  neutral={r['n_neutral']}  skipped={r['n_skip']}"),
        sep(),
        row(f"  {'source→target':<26} {'feature':<24} {'lift':>6}  {'pHi':>5}  {'pLo':>5}  {'verdict'}"),
        sep(),
    ]

    for lnk in r["links"]:
        pair = f"{lnk['source'][:12]}→{lnk['target'][:12]}"
        verdict = lnk.get("verdict", "-")
        dm = "" if lnk.get("direction_match") is None else ("OK" if lnk["direction_match"] else "FLIP")
        flag = f"[{dm}]" if dm else ""
        lines.append(row(
            f"  {pair:<26} "
            f"{lnk['feature'][:24]:<24} "
            f"{_fmt_lift(lnk.get('lift'))}  "
            f"{lnk.get('p_high', 0):>5.3f}  "
            f"{lnk.get('p_low', 0):>5.3f}  "
            f"{verdict} {flag}"
        ))

    lines.append(sep())
    lines.append(row(
        "  Verdicts: validated_positive (lift>1.3) | validated_inverse (lift<0.77) | "
        "neutral (0.9-1.1) | weak"
    ))
    lines.append("└─" + "─" * W + "─┘")
    print("\n".join(lines))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Backtest causal propagation links")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run(save=not args.no_save)


if __name__ == "__main__":
    main()

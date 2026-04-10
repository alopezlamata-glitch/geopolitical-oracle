#!/usr/bin/env python3
"""
Geopolitical Oracle — XGBoost + SHAP binary prediction pipeline.
No LLM in the critical path.

Usage:
  python main.py predict --question "..." --country "Iran" --resolution 2026-06-01
  python main.py train
  python main.py label --prediction-id <id> --outcome yes
  python main.py calibration
  python main.py drift
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# Windows asyncio fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oracle")

# Suppress noisy sub-loggers
for _log in ("aiohttp", "urllib3", "feedparser", "shap"):
    logging.getLogger(_log).setLevel(logging.WARNING)

async def _collector_healthcheck(question: str, country: str | None) -> dict:
    """Run each collector and return a structured health report."""
    from collector import collect_gdelt, collect_rss, collect_metaculus, collect_polymarket, collect_acled

    report: dict[str, dict] = {}
    async with aiohttp.ClientSession() as session:
        collectors = {
            "gdelt": collect_gdelt(session, question),
            "rss": collect_rss(session, question),
            "metaculus": collect_metaculus(session, question),
            "polymarket": collect_polymarket(session, question),
            "acled": collect_acled(session, question, country=country),
        }
        names = list(collectors.keys())
        results = await asyncio.gather(*collectors.values(), return_exceptions=True)

    for name, result in zip(names, results):
        if isinstance(result, Exception):
            report[name] = {"ok": False, "error": str(result)}
            continue

        if name in ("gdelt", "rss", "acled"):
            count = len(result or [])
            report[name] = {"ok": True, "events": count}
        elif name == "metaculus":
            p, n = result if isinstance(result, tuple) else (None, 0)
            report[name] = {"ok": True, "probability": p, "forecasters": n}
        elif name == "polymarket":
            p, vol = result if isinstance(result, tuple) else (None, 0.0)
            report[name] = {"ok": True, "probability": p, "volume": vol}

    return report


def cmd_doctor(args) -> None:
    """Run a quick diagnostic over connectivity, credentials and collector outputs."""
    cases = [
        ("Will there be armed clashes in Sudan before July 1, 2026?", "Sudan"),
        ("Will Russia launch a major offensive in Ukraine before July 1, 2026?", "Ukraine"),
        ("Will Iran conduct a missile test before July 1, 2026?", "Iran"),
    ]

    if args.question:
        cases = [(args.question, args.country)]

    print("\n=== Geopolitical Oracle doctor ===")
    has_token = bool(os.environ.get("ACLED_ACCESS_TOKEN"))
    has_legacy = bool(os.environ.get("ACLED_API_KEY") and os.environ.get("ACLED_EMAIL"))
    if has_token or has_legacy:
        mode = "bearer token" if has_token else "legacy key+email"
        print(f"ACLED credentials: OK ({mode})")
    else:
        print("ACLED credentials: MISSING (set ACLED_ACCESS_TOKEN or ACLED_API_KEY+ACLED_EMAIL)")

    for idx, (question, country) in enumerate(cases, start=1):
        print(f"\n[{idx}/{len(cases)}] {question}")
        report = asyncio.run(_collector_healthcheck(question, country))

        total_events = 0
        for name in ("gdelt", "rss", "acled"):
            item = report.get(name, {})
            if not item.get("ok"):
                print(f"  - {name:10}: ERROR -> {item.get('error', 'unknown error')}")
                continue
            events = item.get("events", 0)
            total_events += events
            print(f"  - {name:10}: {events:4d} events")

        meta = report.get("metaculus", {})
        if meta.get("ok"):
            print(f"  - metaculus : p={meta.get('probability')} (forecasters={meta.get('forecasters', 0)})")
        else:
            print(f"  - metaculus : ERROR -> {meta.get('error', 'unknown error')}")

        poly = report.get("polymarket", {})
        if poly.get("ok"):
            print(f"  - polymarket: p={poly.get('probability')} (volume={poly.get('volume', 0.0)})")
        else:
            print(f"  - polymarket: ERROR -> {poly.get('error', 'unknown error')}")

        if total_events == 0:
            print("  => Status: INSUFFICIENT EVIDENCE (no raw events from GDELT/RSS/ACLED)")
        else:
            print(f"  => Status: OK ({total_events} raw events available)")


async def _collect_all(question: str, country: str | None) -> tuple[list, float | None, float | None]:
    """Run all collectors concurrently. Returns (raw_events, metaculus_p, polymarket_p)."""
    from collector import collect_gdelt, collect_rss, collect_metaculus, collect_polymarket, collect_acled

    # No session-level timeout — each collector manages its own timeout
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            collect_gdelt(session, question),
            collect_rss(session, question),
            collect_metaculus(session, question),
            collect_polymarket(session, question),
            collect_acled(session, question, country=country),
            return_exceptions=True,
        )

    gdelt_events = results[0] if not isinstance(results[0], Exception) else []
    rss_events = results[1] if not isinstance(results[1], Exception) else []
    meta_result = results[2] if not isinstance(results[2], Exception) else (None, 0)
    poly_result = results[3] if not isinstance(results[3], Exception) else (None, 0.0)
    acled_events = results[4] if not isinstance(results[4], Exception) else []

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("collector[%d] raised: %s", i, r)

    all_events = (gdelt_events or []) + (rss_events or []) + (acled_events or [])
    metaculus_p = meta_result[0] if isinstance(meta_result, tuple) else None
    polymarket_p = poly_result[0] if isinstance(poly_result, tuple) else None

    return all_events, metaculus_p, polymarket_p


def cmd_predict(args) -> None:
    from normalizer.canonical import normalize_all
    from normalizer.deduplicator import deduplicate
    from features.builder import build_features
    from features.store import save_features
    from predictor.inference import predict
    from predictor.attribution import compute_shap_attribution
    from predictor.output import format_output, save_prediction
    from monitor.drift import detect_drift

    question = args.question
    country = getattr(args, "country", None)

    print(f"\nGathering evidence for: {question!r}")
    print("Collecting from GDELT, RSS, Metaculus, Polymarket, ACLED...")

    raw_events, metaculus_p, polymarket_p = asyncio.run(_collect_all(question, country))

    print(f"  Raw events collected: {len(raw_events)}")

    # Normalize
    canonical = normalize_all(raw_events)
    # Deduplicate
    deduped = deduplicate(canonical)
    print(f"  After dedup: {len(deduped)} events")

    if not deduped:
        print("\nINSUFFICIENT EVIDENCE — no events found for this question.")
        print("Try a different phrasing, verify collector network access, and configure ACLED credentials/token in .env")
        return

    # Build features
    features, provenance = build_features(
        deduped,
        metaculus_p=metaculus_p,
        polymarket_p=polymarket_p,
    )

    # Save features
    event_ids = [e.event_id for e in deduped]
    save_features(question, features, provenance, event_ids)

    # Predict
    prediction = predict(features)

    # Build event lookup
    events_by_id = {e.event_id: e for e in deduped}

    # SHAP attribution
    attribution = compute_shap_attribution(features, provenance, events_by_id)

    # Drift detection
    drift_flags = detect_drift(features)

    # Format and print
    output = format_output(question, prediction, attribution, features, len(deduped), drift_flags)
    print("\n" + output)

    # Save prediction
    path = save_prediction(question, prediction, attribution, features, provenance, len(deduped))
    print(f"\nSaved: {path}")

    if prediction.get("untrained"):
        print("\n⚠  Model is UNTRAINED (need ≥30 labeled examples).")
        print("   Run 'python main.py train' after labeling predictions.")


def cmd_train(args) -> None:
    from model.trainer import train, load_training_data
    X, y = load_training_data()
    print(f"Training data: {len(X)} labeled examples")
    if len(X) < 30:
        print(f"Need at least 30 examples (have {len(X)}). Label predictions with:")
        print("  python main.py label --prediction-id <id> --outcome yes|no")
        return
    ok = train()
    if ok:
        print("Model trained and saved.")
    else:
        print("Training failed. Check logs.")


def cmd_label(args) -> None:
    """Add a resolved outcome to a prediction file."""
    pred_dir = Path("data/predictions")
    if not pred_dir.exists():
        print("No predictions directory found.")
        return

    # Find prediction by ID prefix
    matches = list(pred_dir.glob(f"*{args.prediction_id}*.json"))
    if not matches:
        matches = list(pred_dir.glob("*.json"))
        # Try to match by timestamp prefix
        matches = [f for f in matches if args.prediction_id in f.stem]

    if not matches:
        print(f"No prediction found matching ID: {args.prediction_id}")
        return

    path = matches[0]
    data = json.loads(path.read_text())
    outcome = 1 if args.outcome.lower() in ("yes", "1", "true") else 0
    data["resolved"] = True
    data["outcome"] = outcome
    path.write_text(json.dumps(data, indent=2))
    print(f"Labeled {path.name}: outcome={outcome}")

    # Copy to training data
    training_dir = Path("data/training")
    training_dir.mkdir(parents=True, exist_ok=True)
    training_record = {
        "question": data["question"],
        "timestamp": data["timestamp"],
        "features": data["features"],
        "outcome": outcome,
    }
    train_path = training_dir / path.name
    train_path.write_text(json.dumps(training_record, indent=2))
    print(f"Added to training data: {train_path.name}")


def cmd_calibration(args) -> None:
    from pathlib import Path
    log_path = Path("data/model/calibration_log.json")
    if not log_path.exists():
        print("No calibration data yet. Train the model first.")
        return
    data = json.loads(log_path.read_text())
    print(f"ECE: {data['ece']:.4f}  (n_val={data['n_val']})")
    print("\nCalibration buckets:")
    for b in data.get("buckets", []):
        bar = "█" * int(b["mean_actual"] * 20)
        print(f"  {b['bin']:12}  n={b['count']:4d}  pred={b['mean_pred']:.3f}  actual={b['mean_actual']:.3f}  {bar}")


def cmd_drift(args) -> None:
    from pathlib import Path
    log_path = Path("data/model/drift_log.json")
    if not log_path.exists():
        print("No drift data yet. Run a prediction first.")
        return
    entries = json.loads(log_path.read_text())
    last = entries[-1] if entries else {}
    print(f"Last drift check: {last.get('timestamp', '?')}")
    print(f"Drifted features: {last.get('drifted_count', 0)}")
    for f in last.get("features", []):
        if f["status"] != "stable":
            print(f"  {f['feature']:35}  PSI={f['psi']:.4f}  [{f['status']}]")


def main():
    parser = argparse.ArgumentParser(description="Geopolitical Oracle")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Run full prediction pipeline")
    p_predict.add_argument("--question", "-q", required=True, help="Binary question to answer")
    p_predict.add_argument("--country", "-c", default=None, help="Target country (for ACLED)")
    p_predict.add_argument("--resolution", "-r", default=None, help="Resolution date YYYY-MM-DD")
    p_predict.set_defaults(func=cmd_predict)

    p_train = sub.add_parser("train", help="Train XGBoost on labeled data")
    p_train.set_defaults(func=cmd_train)

    p_label = sub.add_parser("label", help="Label a prediction outcome")
    p_label.add_argument("--prediction-id", required=True)
    p_label.add_argument("--outcome", required=True, choices=["yes", "no", "0", "1"])
    p_label.set_defaults(func=cmd_label)

    p_calib = sub.add_parser("calibration", help="Show calibration stats")
    p_calib.set_defaults(func=cmd_calibration)

    p_drift = sub.add_parser("drift", help="Show drift report")
    p_drift.set_defaults(func=cmd_drift)

    p_doctor = sub.add_parser("doctor", help="Run connectivity and collector diagnostics")
    p_doctor.add_argument("--question", default=None, help="Optional single diagnostic question")
    p_doctor.add_argument("--country", default=None, help="Country for ACLED when --question is provided")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

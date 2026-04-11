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


def _print_parse_summary(pq, ood) -> None:
    """Print a compact parse + OOD summary before prediction."""
    import sys
    # Use UTF-8 for output on all platforms
    out = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout

    def p(line: str) -> None:
        out.write((line + "\n").encode("utf-8", errors="replace"))
        out.flush()

    status = "[IN DOMAIN]" if ood.in_domain else "[OUT OF DOMAIN]"
    conf_pct = int(pq.parse_confidence * 100)
    conf_bar = "#" * int(pq.parse_confidence * 10) + "." * (10 - int(pq.parse_confidence * 10))

    p("")
    p("+-- Question parse " + "-" * 42 + "+")
    p(f"|  Subject      : {pq.subject} ({pq.subject_type})")
    p(f"|  Predicate    : {pq.predicate}")
    p(f"|  Event family : {pq.event_family}")
    p(f"|  Deadline     : {pq.deadline or '(none detected)'}")
    p(f"|  Jurisdiction : {pq.jurisdiction or '(none)'}")
    p(f"|  Negated      : {'yes' if pq.is_negated else 'no'}")
    p(f"|  Parse conf   : [{conf_bar}] {conf_pct}%")
    p(f"|  Model status : {status}")
    if ood.in_domain:
        p(f"|  Model        : {ood.matched_model}")
    rule_short = pq.resolution_rule[:65] + ("..." if len(pq.resolution_rule) > 65 else "")
    p(f"|  Resolution   : {rule_short}")
    p("+" + "-" * 60 + "+")


def cmd_parse(args) -> None:
    """Detailed parse + OOD analysis without running the full pipeline."""
    from question.parser import parse_question
    from question.ood import assess_ood

    pq = parse_question(args.question)
    ood = assess_ood(pq)

    _print_parse_summary(pq, ood)

    if ood.domain_gap:
        print("\nDomain gaps:")
        for gap in ood.domain_gap:
            print(f"  • {gap}")

    if not ood.in_domain:
        print(f"\nSuggested path to build a model for this domain:")
        print(f"  {ood.suggested_action}")
    else:
        print(f"\nAll checks passed. Run with 'predict' to get a probability.")


def cmd_predict(args) -> None:
    from normalizer.canonical import normalize_all
    from normalizer.deduplicator import deduplicate
    from features.builder import build_features
    from features.store import save_features
    from predictor.inference import predict
    from predictor.attribution import compute_shap_attribution
    from predictor.output import format_output, save_prediction
    from monitor.drift import detect_drift
    from question.parser import parse_question
    from question.ood import assess_ood
    from data_layer.db import init_schema
    from data_layer.pipeline_hooks import (
        persist_raw_events,
        persist_canonical_events,
        persist_question,
        persist_feature_snapshot,
        persist_prediction_record,
    )

    question = args.question
    country = getattr(args, "country", None)

    # ── Step 1: Parse and OOD check ──────────────────────────────────────────
    pq = parse_question(question)
    ood = assess_ood(pq)

    _print_parse_summary(pq, ood)

    if not ood.in_domain:
        out = sys.stdout.buffer
        def _pb(line):
            out.write((line + "\n").encode("utf-8", errors="replace")); out.flush()
        _pb("\n" + "-" * 60)
        _pb("OUT OF DOMAIN -- prediction not issued.")
        _pb(f"\nReason: {ood.reason}")
        _pb(f"\nTo build a model for this domain:")
        _pb(f"  {ood.suggested_action}")
        _pb("-" * 60)
        return

    print(f"\nGathering evidence for: {question!r}")
    print("Collecting from GDELT, RSS, Metaculus, Polymarket, ACLED...")

    # ── Step 2: Collect ───────────────────────────────────────────────────────
    raw_events, metaculus_p, polymarket_p = asyncio.run(_collect_all(question, country))
    as_of_time = datetime.now(timezone.utc)   # strict data cutoff: nothing after this

    print(f"  Raw events collected: {len(raw_events)}")

    # ── Step 3: Normalize + deduplicate ───────────────────────────────────────
    canonical = normalize_all(raw_events)
    deduped = deduplicate(canonical)
    print(f"  After dedup: {len(deduped)} events")

    if not deduped:
        print("\nINSUFFICIENT EVIDENCE — no events found for this question.")
        print("Try a different phrasing or add ACLED credentials in .env")
        return

    # ── Step 4: Build features ────────────────────────────────────────────────
    features, provenance = build_features(
        deduped,
        metaculus_p=metaculus_p,
        polymarket_p=polymarket_p,
        country=country,
    )
    event_ids = [e.event_id for e in deduped]
    # Legacy artifact for local debugging/export; not used by train/label/eval primary flows
    save_features(question, features, provenance, event_ids)

    # ── Step 5: Predict ───────────────────────────────────────────────────────
    prediction = predict(features, metaculus_p=metaculus_p, polymarket_p=polymarket_p)
    events_by_id = {e.event_id: e for e in deduped}
    attribution = compute_shap_attribution(features, provenance, events_by_id)
    drift_flags = detect_drift(features)

    # ── Step 6: Format + print ────────────────────────────────────────────────
    output = format_output(question, prediction, attribution, features, len(deduped), drift_flags)
    sys.stdout.buffer.write(("\n" + output + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    # ── Step 7: Save legacy JSON artifacts (secondary/debug only) ───────────────
    path = save_prediction(question, prediction, attribution, features, provenance, len(deduped))
    print(f"\nSaved: {path}")

    # ── Step 8: Write to lakehouse (non-fatal) ────────────────────────────────
    try:
        init_schema()   # idempotent — no-op if tables already exist

        # 8a. Raw events → raw_documents
        raw_doc_ids = persist_raw_events(raw_events, as_of_time)

        # 8b. Canonical events → canonical_documents + canonical_events
        db_event_ids = persist_canonical_events(deduped, raw_doc_ids, as_of_time)

        # 8c. Parsed question → questions
        question_id = persist_question(pq, ood, as_of_time)

        # 8d. Feature vector → feature_snapshots
        snapshot_id = persist_feature_snapshot(
            features, as_of_time, question_id, pq, db_event_ids
        )

        # 8e. Prediction → predictions
        persist_prediction_record(prediction, question_id, snapshot_id, as_of_time, attribution)

        logger.info(
            "lakehouse: q=%s snap=%s events=%d",
            question_id, snapshot_id, len(db_event_ids),
        )
    except Exception as e:
        logger.warning("lakehouse write failed (non-fatal, prediction still saved): %s", e)

    if prediction.get("untrained"):
        print("\n  Model is UNTRAINED (need >=30 labeled examples).")
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
    """Resolve an outcome by prediction_id (primary: lakehouse, fallback: legacy JSON)."""
    from data_layer.resolution import resolve_outcome_by_prediction_id, resolve_outcome_legacy_json

    outcome = 1 if args.outcome.lower() in ("yes", "1", "true") else 0

    # Primary source of truth: DuckDB lakehouse
    try:
        updated = resolve_outcome_by_prediction_id(
            prediction_id=args.prediction_id,
            outcome=outcome,
            resolver_source="user",
            resolution_notes=f"Labeled via CLI: prediction_id={args.prediction_id}",
        )
        if updated:
            print(f"Labeled prediction {args.prediction_id} in lakehouse: outcome={outcome}")
            return
    except Exception as e:
        logger.warning("label: lakehouse path failed, trying legacy fallback: %s", e)

    # Legacy/deprecated fallback path (temporary compatibility)
    if resolve_outcome_legacy_json(args.prediction_id, outcome):
        print(
            f"Labeled {args.prediction_id} via legacy JSON fallback (deprecated): outcome={outcome}. "
            "DuckDB remains the primary source of truth."
        )
        return

    print(f"No prediction found matching ID: {args.prediction_id}")


def cmd_calibration(args) -> None:
    from pathlib import Path
    log_path = Path("data/model/calibration_log.json")
    if not log_path.exists():
        print("No calibration data yet. Train the model first.")
        return
    data = json.loads(log_path.read_text())
    ece_cal = data.get("ece_cal", data.get("ece", "?"))
    ece_raw = data.get("ece_raw", "?")
    method = data.get("method", "?")
    print(f"Calibrator       : {method}")
    print(f"ECE (calibrated) : {ece_cal}  (n_val={data['n_val']})")
    print(f"ECE (raw)        : {ece_raw}")
    print("\nCalibration buckets (calibrated probs vs actual):")
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
    method = last.get("method", "z_score")
    print(f"Last drift check : {last.get('timestamp', '?')}")
    print(f"Method           : {method}")
    print(f"Drifted features : {last.get('drifted_count', 0)}")
    for f in last.get("features", []):
        if f["status"] != "stable":
            z = f.get("z_score", "?")
            z_str = f"{z:.4f}" if isinstance(z, float) else str(z)
            print(f"  {f['feature']:35}  z={z_str}  [{f['status']}]")


def cmd_db(args) -> None:
    """Show lakehouse statistics and recent predictions."""
    from data_layer.db import init_schema, schema_stats, get_db_path

    init_schema()
    print(f"Database : {get_db_path()}")

    stats = schema_stats()
    total = sum(v for v in stats.values() if v)
    print(f"Total rows: {total:,}\n")

    # Show only tables with data
    has_data = {k: v for k, v in stats.items() if v}
    if not has_data:
        print("No data yet. Run: python main.py predict --question '...'")
        return

    print(f"{'Table':<40} {'Rows':>10}")
    print("-" * 52)
    for table, count in stats.items():
        marker = " *" if count else ""
        print(f"  {table:<38} {(count or 0):>10,}{marker}")

    # Recent predictions
    from data_layer.db import get_db, table_exists
    if table_exists("predictions"):
        db = get_db()
        rows = db.execute("""
            SELECT predicted_at, calibrated_prob, answer, model_id
            FROM predictions
            ORDER BY predicted_at DESC
            LIMIT 5
        """).fetchall()
        if rows:
            print(f"\nLast {len(rows)} predictions:")
            for r in rows:
                ts = str(r[0])[:16] if r[0] else "?"
                print(f"  {ts}  p={r[1]:.3f}  {r[2]}  [{r[3]}]")

    # Unresolved questions
    if table_exists("questions"):
        db = get_db()
        n_open = db.execute(
            "SELECT COUNT(*) FROM questions WHERE status = 'open'"
        ).fetchone()[0]
        n_resolved = db.execute(
            "SELECT COUNT(*) FROM questions WHERE status = 'resolved'"
        ).fetchone()[0]
        print(f"\nQuestions: {n_open} open, {n_resolved} resolved")
        if n_resolved > 0:
            avg_brier = db.execute(
                "SELECT AVG(brier_component) FROM predictions WHERE brier_component IS NOT NULL"
            ).fetchone()[0]
            if avg_brier is not None:
                print(f"Mean Brier (resolved): {avg_brier:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Geopolitical Oracle")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse question and check OOD — no prediction")
    p_parse.add_argument("--question", "-q", required=True, help="Binary question to analyse")
    p_parse.set_defaults(func=cmd_parse)

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

    p_db = sub.add_parser("db", help="Show lakehouse database stats and recent predictions")
    p_db.set_defaults(func=cmd_db)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

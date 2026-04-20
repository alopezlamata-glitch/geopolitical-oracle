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


async def _collect_all(question: str, country: str | None) -> tuple[list, dict, str, dict]:
    """
    Run all collectors concurrently (9 in parallel).

    Returns:
        (raw_events, market_meta, wiki_context, economic_context) where:
          - raw_events        : list[RawEvent] from GDELT + RSS + ACLED
          - market_meta       : dict with forecasting market quality signals
          - wiki_context      : str  Wikipedia article extract (may be "")
          - economic_context  : dict with eco_* and pol_* features from WB + FRED + V-Dem

    Market priority:
      1. Polymarket (real-money, highest signal quality)
      2. Metaculus (requires METACULUS_API_TOKEN in .env)
      3. Manifold Markets (free, no auth, fallback when Metaculus unavailable)
    """
    from collector import (
        collect_gdelt, collect_rss, collect_metaculus,
        collect_polymarket, collect_acled, collect_wikipedia,
        collect_manifold,
    )
    from collector.worldbank import collect_worldbank
    from collector.fred import collect_fred
    from collector.acled import _extract_country as _acled_country

    # Resolve country for structural data fetchers (WB, ACLED, V-Dem)
    resolved_country = country or _acled_country(question) or ""

    # No session-level timeout — each collector manages its own timeout
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            collect_gdelt(session, question),                            # 0
            collect_rss(session, question),                              # 1
            collect_metaculus(session, question),                        # 2
            collect_polymarket(session, question),                       # 3
            collect_acled(session, question, country=resolved_country),  # 4
            collect_wikipedia(session, question),                        # 5
            collect_manifold(session, question),                         # 6
            collect_worldbank(session, resolved_country),                # 7
            collect_fred(session),                                       # 8
            return_exceptions=True,
        )

    gdelt_events   = results[0] if not isinstance(results[0], Exception) else []
    rss_events     = results[1] if not isinstance(results[1], Exception) else []
    meta_result    = results[2] if not isinstance(results[2], Exception) else (None, 0)
    poly_result    = results[3] if not isinstance(results[3], Exception) else (None, 0.0, 0.0)
    acled_events   = results[4] if not isinstance(results[4], Exception) else []
    wiki_result    = results[5] if not isinstance(results[5], Exception) else None
    manifold_result= results[6] if not isinstance(results[6], Exception) else (None, 0)
    wb_result      = results[7] if not isinstance(results[7], Exception) else {}
    fred_result    = results[8] if not isinstance(results[8], Exception) else {}

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("collector[%d] raised: %s", i, r)

    all_events = (gdelt_events or []) + (rss_events or []) + (acled_events or [])

    # Wikipedia context (plain string, not an event)
    from collector.wikipedia import WikipediaResult
    if isinstance(wiki_result, WikipediaResult) and wiki_result.found:
        wiki_context = wiki_result.content
        logger.info("wikipedia: %d chars from %r", len(wiki_context), wiki_result.page_title)
    else:
        wiki_context = ""

    # Unpack market quality metadata
    # Metaculus: returns (None, nr_forecasters) — API intentionally hides CP
    if isinstance(meta_result, tuple):
        metaculus_p = meta_result[0]              # always None by design
        metaculus_forecasters = meta_result[1] if len(meta_result) > 1 else None
    else:
        metaculus_p, metaculus_forecasters = None, None

    # Manifold: always try as probability source (Metaculus never provides p)
    # Keep Metaculus forecaster count as attention signal even when using Manifold p
    if isinstance(manifold_result, tuple) and manifold_result[0] is not None:
        manifold_p, manifold_bettors = manifold_result[0], manifold_result[1]
        metaculus_p = manifold_p
        # Prefer Metaculus forecaster count (larger, more calibrated crowd) if available
        if not metaculus_forecasters:
            metaculus_forecasters = manifold_bettors
        logger.info(
            "manifold: p=%.3f (%d bettors) — used as crowd probability source",
            manifold_p, manifold_bettors,
        )

    # Polymarket
    if isinstance(poly_result, tuple):
        polymarket_p      = poly_result[0]
        polymarket_vol    = poly_result[1] if len(poly_result) > 1 else None
        polymarket_mscore = poly_result[2] if len(poly_result) > 2 else None
    else:
        polymarket_p, polymarket_vol, polymarket_mscore = None, None, None

    market_meta = {
        "metaculus_p":          metaculus_p,
        "metaculus_forecasters": int(metaculus_forecasters) if metaculus_forecasters else None,
        "polymarket_p":         polymarket_p,
        "polymarket_volume":    float(polymarket_vol) if polymarket_vol else None,
        "polymarket_match_score": float(polymarket_mscore) if polymarket_mscore else None,
    }

    # Merge economic context: World Bank + FRED (FRED overwrites WB for overlapping keys)
    economic_context: dict[str, float] = {}
    if isinstance(wb_result, dict):
        economic_context.update(wb_result)
    if isinstance(fred_result, dict):
        economic_context.update(fred_result)

    # V-Dem political indicators (static, from snapshot if available)
    if resolved_country:
        try:
            from collector.vdem import get_vdem_features
            vdem_feats = get_vdem_features(resolved_country)
            if vdem_feats:
                economic_context.update(vdem_feats)
                logger.info("vdem: loaded %d features for '%s'", len(vdem_feats), resolved_country)
        except Exception as e:
            logger.debug("vdem: skipped: %s", e)

    n_eco = sum(1 for k, v in economic_context.items() if not k.startswith("_") and v != 0.0)
    if n_eco:
        logger.info("economic_context: %d non-zero features from WB/FRED/V-Dem", n_eco)

    return all_events, market_meta, wiki_context, economic_context


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
    from predictor.inference import predict_for_domain
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
    # Use parsed jurisdiction as country if not explicitly provided
    if not country and pq.jurisdiction:
        country = pq.jurisdiction
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
    raw_events, market_meta, wiki_context, economic_context = asyncio.run(_collect_all(question, country))
    as_of_time = datetime.now(timezone.utc)   # strict data cutoff: nothing after this

    metaculus_p   = market_meta["metaculus_p"]
    polymarket_p  = market_meta["polymarket_p"]

    print(f"  Raw events collected: {len(raw_events)}")
    if metaculus_p is not None:
        print(f"  Metaculus signal    : p={metaculus_p:.3f}  "
              f"(n={market_meta['metaculus_forecasters'] or '?'})")
    if polymarket_p is not None:
        print(f"  Polymarket signal   : p={polymarket_p:.3f}  "
              f"(vol=${market_meta['polymarket_volume'] or 0:,.0f}  "
              f"match={market_meta['polymarket_match_score'] or 0:.2f})")

    # ── Step 3: Normalize + deduplicate ───────────────────────────────────────
    canonical = normalize_all(raw_events)
    deduped = deduplicate(canonical)
    print(f"  After dedup: {len(deduped)} events")

    if not deduped:
        print("\nINSUFFICIENT EVIDENCE — no events found for this question.")
        print("Try a different phrasing or add ACLED credentials in .env")
        return

    # ── Step 4: Build features (+ LLM enrichment if Ollama available) ────────
    if wiki_context:
        print(f"  Wikipedia       : {len(wiki_context)} chars of background context")
    features, provenance = build_features(
        deduped,
        metaculus_p=metaculus_p,
        polymarket_p=polymarket_p,
        country=country,
        question=question,
        use_llm=True,
        wiki_context=wiki_context,
        event_family=pq.event_family,
        economic_context=economic_context,
    )
    # Pass predicate as a sentinel so base_rate_predictor can look up the base rate.
    # Stored with underscore prefix so XGBoost skips it (not in _FEATURE_NAMES).
    features["_predicate"] = pq.predicate or "unknown"
    if features.get("llm_available", 0.0) > 0:
        print(f"  LLM features    : threat={features['llm_threat_level']:.2f}  "
              f"esc={features['llm_escalation']:.2f}  "
              f"deesc={features['llm_deescalation']:.2f}  "
              f"host={features['llm_actor_hostility']:.2f}")
    event_ids = [e.event_id for e in deduped]
    # Legacy artifact for local debugging/export; not used by train/label/eval primary flows
    save_features(question, features, provenance, event_ids)

    # ── Step 5: Predict (routed by matched model) ─────────────────────────────
    headlines = [e.raw_title for e in deduped if e.raw_title.strip()]
    deadline_dt = (
        datetime.combine(pq.deadline, datetime.min.time()).replace(tzinfo=timezone.utc)
        if pq.deadline else None
    )
    prediction = predict_for_domain(
        matched_model=ood.matched_model or "xgb_conflict_v3",
        features=features,
        event_family=pq.event_family,
        question=question,
        headlines=headlines,
        deadline=deadline_dt,
        wiki_context=wiki_context,
        metaculus_p=metaculus_p,
        polymarket_p=polymarket_p,
        metaculus_forecasters=market_meta["metaculus_forecasters"],
        polymarket_volume=market_meta["polymarket_volume"],
        polymarket_match_score=market_meta["polymarket_match_score"],
        as_of_time=as_of_time,
        country=country,
    )
    if prediction.get("predictor") == "ollama_reasoning_v1":
        attr = prediction.get("attribution", {})
        if attr.get("reasoning"):
            reasoning_line = attr["reasoning"][:120]
            sys.stdout.buffer.write(
                (f"  Ollama reasoning: {reasoning_line}...\n").encode("utf-8", errors="replace")
            )
            sys.stdout.buffer.flush()
    events_by_id = {e.event_id: e for e in deduped}
    attribution = compute_shap_attribution(features, provenance, events_by_id)
    drift_flags = detect_drift(features)

    # ── Step 5b: Coherence check + auto-correction (Phase 3 world model) ───────
    coherence_output = ""
    try:
        from world_state.coherence import check, register, correct_probability, format_coherence_output
        final_p = prediction.get("calibrated_prob", 0.5)
        coherence_report = check(
            entity_name=country or pq.subject or "",
            predicate=pq.predicate or "unknown",
            probability=final_p,
            deadline=pq.deadline,
            event_family=pq.event_family,
        )
        # Apply soft auto-correction when violations exist
        if coherence_report.violations:
            corrected_p = correct_probability(final_p, coherence_report)
            if corrected_p != final_p:
                prediction["p_before_coherence"] = final_p
                prediction["calibrated_prob"]    = corrected_p
                prediction["answer"] = "YES" if corrected_p >= 0.5 else "NO"
                logger.info(
                    "coherence: auto-corrected %.3f → %.3f (%d violation(s))",
                    final_p, corrected_p, len(coherence_report.violations),
                )
                final_p = corrected_p
        # Register so future predictions in this session can check against this
        register(
            entity_name=country or pq.subject or "",
            predicate=pq.predicate or "unknown",
            probability=final_p,
            deadline=pq.deadline,
        )
        coherence_output = format_coherence_output(coherence_report)
        prediction["coherence_score"] = coherence_report.consistency_score
        prediction["coherence_violations"] = len(coherence_report.violations)
        if coherence_report.implications:
            prediction["causal_implications"] = coherence_report.implications[:5]
    except Exception as e:
        logger.debug("coherence check skipped: %s", e)

    # ── Step 5c: Scenario generation (escalation / baseline / de-escalation) ──
    scenarios = {}
    if country and deadline_dt is not None:
        try:
            from world_state.scenario_engine import generate_scenarios
            horizon_days_sc = max(1, (deadline_dt - as_of_time).days)
            scenarios = generate_scenarios(
                entity_name=country,
                predicate=pq.predicate or "unknown",
                event_family=pq.event_family,
                horizon_days=horizon_days_sc,
                n_samples=80,
            )
        except Exception as e:
            logger.debug("scenario generation skipped: %s", e)

    # ── Step 6: Format + print ────────────────────────────────────────────────
    output = format_output(question, prediction, attribution, features, len(deduped), drift_flags, scenarios=scenarios)
    sys.stdout.buffer.write(("\n" + output + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    if coherence_output:
        sys.stdout.buffer.write(coherence_output.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
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
        bar = "#" * int(b["mean_actual"] * 20)
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


def cmd_blend_calibrate(args) -> None:
    """
    Phase B: learn blend coefficients from resolved prediction history.

    Queries DuckDB for all predictions with known outcomes, compares
    model_only vs fixed_blend vs learned_blend Brier scores, and saves
    alpha/beta/bias to data/model/blend_weights.json if enough data exists.
    """
    from predictor.blend_calibrator import run_blend_calibration, MIN_RESOLVED_WITH_MARKET
    from data_layer.db import init_schema

    init_schema()  # ensure tables exist (and blend columns migrated)
    run_blend_calibration(verbose=True)


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


def cmd_auto_resolve(args) -> None:
    from scripts.auto_resolve import run as auto_resolve_run
    auto_resolve_run(
        dry_run=args.dry_run,
        limit=args.limit,
        min_confidence=args.min_confidence,
    )


def cmd_global_risk(args) -> None:
    """Compute and display the global geopolitical risk index across all entities."""
    from world_state.global_risk import run as global_risk_run
    global_risk_run()


def cmd_backfill_world_state(args) -> None:
    """Seed synthetic world_state_history for all entities so VAR models can be fitted."""
    from scripts.backfill_world_state import run as backfill_run
    results = backfill_run(
        n_weeks=args.weeks,
        skip_existing=not args.no_skip,
        fit_after=not args.no_fit,
        dry_run=args.dry_run,
        entity_names=args.entity or None,
    )
    print(f"\nBackfill complete: {sum(results.values())} rows across {len(results)} entities")


def cmd_update_world_state(args) -> None:
    from scripts.update_world_state import run as uws_run
    uws_run(
        entity_names=args.entity or None,
        dry_run=args.dry_run,
        apply_causal=not args.no_causal,
        priority_only=args.priority_only,
        concurrency=args.concurrency,
    )


def cmd_world_state(args) -> None:
    """Show current world state for one or all entities."""
    from data_layer.db import init_schema, get_db, table_exists

    init_schema()
    if not table_exists("world_state"):
        print("No world state yet. Run: python main.py update-world-state")
        return

    db = get_db()

    if args.entity:
        import hashlib
        eid = "ent_" + hashlib.sha256(f"country|{args.entity.lower()}".encode()).hexdigest()[:24]
        rows = db.execute("""
            SELECT as_of_date, military_count_7d, escalation_index,
                   event_velocity_7d, pol_approval_pressure,
                   eco_market_volatility, n_events_used, data_completeness
            FROM world_state
            WHERE entity_id = ?
            ORDER BY as_of_date DESC
            LIMIT 10
        """, [eid]).fetchall()

        if not rows:
            print(f"No world state found for '{args.entity}'")
            return

        print(f"\nWorld state history — {args.entity}")
        print(f"{'Date':<12} {'Mil_7d':>7} {'Esc':>7} {'Vel':>7} {'Pol':>7} {'EcoVol':>7} {'Events':>7} {'Compl':>6}")
        print("-" * 66)
        for r in rows:
            print(f"{str(r[0]):<12} {r[1]:>7.1f} {r[2]:>7.3f} {r[3]:>7.3f} {r[4]:>7.3f} {r[5]:>7.3f} {r[6]:>7d} {r[7]:>5.0%}")
    else:
        # Summary across all entities
        rows = db.execute("""
            SELECT e_id, name, as_of_date, mil_7d, esc, vel, n_ev
            FROM (
                SELECT ws.entity_id AS e_id,
                       ws.entity_id AS name,
                       ws.as_of_date,
                       ws.military_count_7d AS mil_7d,
                       ws.escalation_index  AS esc,
                       ws.event_velocity_7d AS vel,
                       ws.n_events_used     AS n_ev,
                       ROW_NUMBER() OVER (PARTITION BY ws.entity_id ORDER BY ws.as_of_date DESC) AS rn
                FROM world_state ws
                WHERE ws.valid_to IS NULL
            ) sub
            WHERE rn = 1
            ORDER BY esc DESC
            LIMIT 20
        """).fetchall()

        if not rows:
            print("No world state rows yet.")
            return

        total = db.execute("SELECT COUNT(DISTINCT entity_id) FROM world_state").fetchone()[0]
        latest = db.execute("SELECT MAX(as_of_date) FROM world_state").fetchone()[0]
        print(f"\nCurrent world state — {total} entities, last update: {latest}")
        print(f"{'Entity ID':<30} {'Date':<12} {'Mil_7d':>7} {'Esc':>7} {'Vel':>7} {'Events':>7}")
        print("-" * 74)
        for r in rows:
            print(f"{str(r[0])[:28]:<30} {str(r[2]):<12} {r[3]:>7.1f} {r[4]:>7.3f} {r[5]:>7.3f} {r[6]:>7d}")


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

    p_blend = sub.add_parser(
        "blend-calibrate",
        help="Phase B: learn market blend weights from resolved history",
    )
    p_blend.set_defaults(func=cmd_blend_calibrate)

    p_autores = sub.add_parser(
        "auto-resolve",
        help="Auto-resolve overdue questions via Ollama LLM (flywheel labeling)",
    )
    p_autores.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    p_autores.add_argument("--limit", type=int, default=20, help="Max questions to process")
    p_autores.add_argument("--min-confidence", type=float, default=0.85)
    p_autores.set_defaults(func=cmd_auto_resolve)

    p_uws = sub.add_parser(
        "update-world-state",
        help="Run daily world state update for all entities",
    )
    p_uws.add_argument("--entity", "-e", nargs="+", metavar="NAME",
                       help="Only update these entities")
    p_uws.add_argument("--dry-run", action="store_true")
    p_uws.add_argument("--no-causal", action="store_true",
                       help="Skip cross-entity causal propagation")
    p_uws.add_argument("--priority-only", action="store_true",
                       help="Only high-priority entities")
    p_uws.add_argument("--concurrency", type=int, default=4)
    p_uws.set_defaults(func=cmd_update_world_state)

    p_ws = sub.add_parser(
        "world-state",
        help="Show current world state snapshot",
    )
    p_ws.add_argument("--entity", "-e", metavar="NAME",
                      help="Show history for a specific entity")
    p_ws.set_defaults(func=cmd_world_state)

    p_fit = sub.add_parser(
        "fit-transition-model",
        help="Fit VAR transition models from world state history",
    )
    p_fit.add_argument("--entity", "-e", nargs="+", metavar="NAME",
                       help="Only fit for these entities")
    p_fit.set_defaults(func=lambda a: __import__(
        "scripts.fit_transition_model", fromlist=["run"]
    ).run(entity_names=a.entity))

    p_seed_rel = sub.add_parser(
        "seed-relations",
        help="Seed entity relations DB from data/relations_seed.json",
    )
    p_seed_rel.set_defaults(func=lambda a: __import__(
        "scripts.seed_relations", fromlist=["run"]
    ).run())

    p_seed_mkt = sub.add_parser(
        "seed-markets",
        help="Seed resolved market predictions from Manifold (blend calibration bootstrap)",
    )
    p_seed_mkt.add_argument("--limit", type=int, default=500)
    p_seed_mkt.add_argument("--dry-run", action="store_true")
    p_seed_mkt.set_defaults(func=lambda a: __import__(
        "scripts.seed_from_markets", fromlist=["run"]
    ).run(limit=a.limit, dry_run=a.dry_run))

    p_seed_td = sub.add_parser(
        "seed-training-data",
        help="Generate synthetic training examples to bootstrap XGBoost",
    )
    p_seed_td.set_defaults(func=lambda a: __import__(
        "scripts.seed_training_data", fromlist=["main"]
    ).main())

    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate model: Brier, AUC, calibration ECE on resolved predictions",
    )
    p_eval.add_argument("--since", metavar="YYYY-MM-DD", help="Only include predictions after this date")
    p_eval.add_argument("--no-save", action="store_true")
    p_eval.set_defaults(func=lambda a: __import__(
        "scripts.evaluate", fromlist=["run"]
    ).run(
        since=__import__("datetime", fromlist=["datetime"]).datetime.fromisoformat(a.since).replace(
            tzinfo=__import__("datetime", fromlist=["timezone"]).timezone.utc
        ) if a.since else None,
        save=not a.no_save,
    ))

    p_nightly = sub.add_parser(
        "nightly",
        help="Run full nightly pipeline: update-world-state → fit → resolve → blend → evaluate",
    )
    p_nightly.add_argument("--skip", nargs="+",
                           choices=["update", "fit", "resolve", "blend", "evaluate"],
                           default=[], metavar="STEP")
    p_nightly.add_argument("--dry-run", action="store_true")
    p_nightly.set_defaults(func=lambda a: __import__(
        "scripts.nightly_pipeline", fromlist=["run"]
    ).run(skip=set(a.skip), dry_run=a.dry_run))

    p_feat = sub.add_parser(
        "feature-analysis",
        help="SHAP feature importance analysis — identify dead/key features",
    )
    p_feat.add_argument("--top",       type=int,   default=25)
    p_feat.add_argument("--threshold", type=float, default=0.001)
    p_feat.add_argument("--no-save",   action="store_true")
    p_feat.set_defaults(func=lambda a: __import__(
        "scripts.feature_analysis", fromlist=["run"]
    ).run(top_n=a.top, dead_threshold=a.threshold, save=not a.no_save))

    p_bt = sub.add_parser(
        "backtest-causal",
        help="Validate causal propagation links against resolved prediction history",
    )
    p_bt.add_argument("--no-save", action="store_true")
    p_bt.set_defaults(func=lambda a: __import__(
        "scripts.backtest_causal", fromlist=["run"]
    ).run(save=not a.no_save))

    p_gr = sub.add_parser(
        "global-risk",
        help="Compute global geopolitical risk index across all entities",
    )
    p_gr.set_defaults(func=cmd_global_risk)

    p_bfws = sub.add_parser(
        "backfill-world-state",
        help="Seed synthetic world_state_history for all entities (enables VAR fitting)",
    )
    p_bfws.add_argument("--weeks",   type=int,  default=26, help="Weeks of history to generate (default: 26)")
    p_bfws.add_argument("--no-skip", action="store_true",   help="Re-insert even if history already exists")
    p_bfws.add_argument("--no-fit",  action="store_true",   help="Skip VAR model refitting after insert")
    p_bfws.add_argument("--dry-run", action="store_true",   help="Simulate without writing")
    p_bfws.add_argument("--entity",  nargs="+", metavar="NAME", help="Limit to these entities")
    p_bfws.set_defaults(func=cmd_backfill_world_state)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

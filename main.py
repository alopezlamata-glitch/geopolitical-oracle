#!/usr/bin/env python3
"""
Geopolitical Oracle — CLI entry point.

Full pipeline:
  1. Evidence collection  — Wikipedia, RSS, GDELT (snapshot + 30-day series), ACLED,
                            Metaculus, Polymarket  [all parallel, 8-second timeouts]
  2. Pipeline layer       — Temporal analysis, semantic clustering, risk scoring
  3. LLM reasoning        — 3-scenario probabilistic forecast (Ollama / Llama 3.1)
  4. Market aggregation   — logarithmic pooling of Metaculus + Polymarket signals
  5. Combiner             — weighted log-odds blend + extremization
  6. Output               — box-formatted terminal + JSON log for calibration

Requires Ollama:
    ollama pull llama3.1:8b   # recommended
    ollama pull llama3.3:70b  # more powerful (~48GB RAM)
    ollama serve

Usage:
    python main.py "Will France hold snap elections before July 2026?"
    python main.py --model llama3.3:70b "Will Iran sign a nuclear deal this year?"
    python main.py --calibrate
    python main.py --verbose "Will the Fed cut rates in June 2026?"

Optional env vars (see .env.example):
    OLLAMA_URL      Ollama server URL    (default: http://localhost:11434)
    OLLAMA_MODEL    Model tag            (default: llama3.1:8b)
    ACLED_API_KEY   ACLED API key        (enables conflict event data)
    ACLED_EMAIL     ACLED account email
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from collector import (
    EvidenceBlock,
    collect_metaculus,
    collect_polymarket,
    collect_gdelt,
    collect_gdelt_timeseries,
    collect_wikipedia,
    collect_rss,
    collect_acled,
)
from aggregator import aggregate_markets, estimate, combine
from aggregator.llm_estimator import check_ollama
from pipeline import analyze_temporal, embed_texts, cluster_events, compute_risk_score
from output import format_output, save_prediction, run_calibration, apply_calibration

logger = logging.getLogger(__name__)


async def run_oracle(question: str, model_override: str | None = None) -> None:
    """Full oracle pipeline for a single question."""

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = model_override or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ["OLLAMA_MODEL"] = model

    # ── 1. Health check ───────────────────────────────────────────────────────
    print(f"\n  Checking Ollama ({model})...", end=" ", flush=True)
    try:
        await check_ollama(ollama_url, model)
        print("OK")
    except RuntimeError as e:
        print(f"\n\n  ERROR: {e}\n")
        sys.exit(1)

    # ── 2. Normalize question ─────────────────────────────────────────────────
    question = question.strip()
    if not question.endswith("?"):
        question += "?"

    print(f"  Question: {question}")
    print("  Collecting evidence (7 sources in parallel)...", end=" ", flush=True)

    # ── 3. Run ALL collectors concurrently ────────────────────────────────────
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            collect_metaculus(session, question),
            collect_polymarket(session, question),
            collect_gdelt(session, question),
            collect_gdelt_timeseries(session, question),
            collect_wikipedia(session, question),
            collect_rss(session, question),
            collect_acled(session, question),
            return_exceptions=True,
        )

    collector_names = [
        "metaculus", "polymarket", "gdelt", "gdelt_timeseries",
        "wikipedia", "rss", "acled",
    ]
    evidence: list[EvidenceBlock] = []
    for name, result in zip(collector_names, results):
        if isinstance(result, Exception):
            logger.warning("Collector '%s' failed: %s", name, result)
        elif result is not None:
            evidence.append(result)
        else:
            logger.warning("Collector '%s' returned None", name)

    good = sum(1 for b in evidence if b.quality != "insufficient")
    print(f"done ({good}/{len(collector_names)} with data)")

    if not evidence:
        print("  CANNOT ANSWER: all collectors failed.")
        sys.exit(1)

    # ── 4. Pipeline layer ─────────────────────────────────────────────────────
    print("  Running pipeline (temporal + clustering + scoring)...", end=" ", flush=True)

    # 4a. Temporal analysis from GDELT 30-day series
    temporal = None
    ts_block = next((b for b in evidence if b.source == "gdelt_timeseries"), None)
    if ts_block and ts_block.quality != "insufficient":
        vol_series = ts_block.metadata.get("volume_series", [])
        tone_series = ts_block.metadata.get("tone_series", [])
        if vol_series:
            temporal = analyze_temporal(vol_series, tone_series)

    # 4b. Semantic clustering of RSS headlines + Wikipedia extract
    cluster_result = None
    texts_to_cluster: list[str] = []
    rss_block = next((b for b in evidence if b.source == "rss"), None)
    wiki_block = next((b for b in evidence if b.source == "wikipedia"), None)
    if rss_block and rss_block.quality != "insufficient":
        # Split headlines into individual texts
        texts_to_cluster.extend([
            line.strip() for line in rss_block.content.split("\n")
            if line.strip() and len(line.strip()) > 20
        ])
    if wiki_block and wiki_block.content:
        texts_to_cluster.append(wiki_block.content[:500])

    if len(texts_to_cluster) >= 3:
        embeddings = embed_texts(texts_to_cluster)
        cluster_result = cluster_events(texts_to_cluster, embeddings)

    # 4c. Risk score
    gdelt_block = next((b for b in evidence if b.source == "gdelt"), None)
    acled_block = next((b for b in evidence if b.source == "acled"), None)
    risk_score = compute_risk_score(
        temporal=temporal,
        gdelt_tone=gdelt_block.metadata.get("avg_tone") if gdelt_block and gdelt_block.quality != "insufficient" else None,
        gdelt_article_count=gdelt_block.metadata.get("article_count") if gdelt_block and gdelt_block.quality != "insufficient" else None,
        rss_article_count=rss_block.metadata.get("articles_matched") if rss_block and rss_block.quality != "insufficient" else None,
        acled_fatalities=acled_block.metadata.get("fatalities") if acled_block and acled_block.quality != "insufficient" else None,
        acled_event_count=acled_block.metadata.get("event_count") if acled_block and acled_block.quality != "insufficient" else None,
    )
    print(f"done (risk={risk_score.score:.0f}/100 [{risk_score.label}])")

    # ── 5. Market aggregation ─────────────────────────────────────────────────
    p_market = aggregate_markets(evidence)

    # ── 6. LLM estimation (3 scenarios) ──────────────────────────────────────
    print(f"  Reasoning with {model} (generating scenarios)...", end=" ", flush=True)
    llm_result = await estimate(
        question=question,
        evidence=evidence,
        temporal=temporal,
        risk_score=risk_score,
        cluster_result=cluster_result,
    )
    print("done\n")

    if llm_result.get("abstain"):
        print("  INSUFFICIENT EVIDENCE: the model could not generate a calibrated forecast.\n")
        sys.exit(0)

    p_llm = llm_result.get("probability")
    if p_llm is None:
        print("  CANNOT ANSWER: LLM returned no probability.")
        sys.exit(1)

    # ── 7. Combine LLM + market ───────────────────────────────────────────────
    p_final = combine(p_llm, p_market)

    # ── 8. Platt calibration (if >= 30 resolved predictions exist) ───────────
    p_corrected, was_calibrated = apply_calibration(p_final)
    if was_calibrated:
        logger.debug("Platt calibration: %.3f → %.3f", p_final, p_corrected)
        p_final = p_corrected

    # ── 9. Format and print ───────────────────────────────────────────────────
    output = format_output(
        question=question,
        p_final=p_final,
        evidence=evidence,
        llm_result=llm_result,
        p_llm=p_llm,
        p_market=p_market,
        temporal=temporal,
        risk_score=risk_score,
    )
    print(output)

    # ── 10. Save prediction log ───────────────────────────────────────────────
    saved_path = save_prediction(
        question=question,
        p_final=p_final,
        p_llm=p_llm,
        p_market=p_market,
        llm_result=llm_result,
        evidence=evidence,
        temporal=temporal,
        risk_score=risk_score,
    )
    print(f"\n  Saved → {saved_path.relative_to(Path(__file__).parent)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geopolitical Oracle — calibrated yes/no probability forecasts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "Will France hold snap elections before July 2026?"\n'
            '  python main.py --model llama3.3:70b "Will China invade Taiwan this year?"\n'
            "  python main.py --calibrate\n"
        ),
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="A yes/no question resolvable within 2 years.")
    parser.add_argument("--model", "-m", default=None, metavar="TAG",
                        help="Ollama model tag (e.g. llama3.1:8b, llama3.3:70b).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Show calibration stats on resolved predictions.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    if args.calibrate:
        run_calibration()
        return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    asyncio.run(run_oracle(args.question, model_override=args.model))


if __name__ == "__main__":
    main()

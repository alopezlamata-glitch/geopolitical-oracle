#!/usr/bin/env python3
"""
Geopolitical Oracle — CLI entry point.

Requires Ollama running locally:
    ollama pull llama3.1:8b   # recommended
    ollama pull llama3.3:70b  # more powerful (needs ~48GB RAM)
    ollama serve

Usage:
    python main.py "Will France hold snap elections before July 2026?"
    python main.py "Will the Fed cut rates in June 2026?"
    python main.py --model llama3.3:70b "Will China invade Taiwan this year?"
    python main.py --calibrate
    python main.py --verbose "Will Iran sign a nuclear deal before October?"

Environment variables (optional, see .env.example):
    OLLAMA_URL    Base URL of Ollama server  (default: http://localhost:11434)
    OLLAMA_MODEL  Model tag                  (default: llama3.1:8b)
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
    collect_metaculus,
    collect_polymarket,
    collect_gdelt,
    collect_wikipedia,
    collect_rss,
    EvidenceBlock,
)
from aggregator import aggregate_markets, estimate, combine
from aggregator.llm_estimator import check_ollama
from output import format_output, save_prediction, run_calibration, apply_calibration

logger = logging.getLogger(__name__)


async def run_oracle(question: str, model_override: str | None = None) -> None:
    """Full oracle pipeline for a single question."""

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = model_override or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    # Override env so llm_estimator picks it up
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ["OLLAMA_MODEL"] = model

    # ── 1. Check Ollama is running before wasting time on collectors ──────────
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

    print(f"  Gathering evidence for: {question}")
    print("  (Metaculus · Polymarket · GDELT · Wikipedia · RSS running in parallel...)\n")

    # ── 3. Run all collectors concurrently ────────────────────────────────────
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            collect_metaculus(session, question),
            collect_polymarket(session, question),
            collect_gdelt(session, question),
            collect_wikipedia(session, question),
            collect_rss(session, question),
            return_exceptions=True,
        )

    # ── 4. Separate failures from evidence blocks ─────────────────────────────
    collector_names = ["metaculus", "polymarket", "gdelt", "wikipedia", "rss"]
    evidence: list[EvidenceBlock] = []
    for name, result in zip(collector_names, results):
        if isinstance(result, Exception):
            logger.warning("Collector '%s' failed: %s", name, result)
        elif result is not None:
            evidence.append(result)
        else:
            logger.warning("Collector '%s' returned no data", name)

    # ── 5. All collectors failed ──────────────────────────────────────────────
    if not evidence:
        print("CANNOT ANSWER: insufficient evidence — all collectors failed.")
        sys.exit(1)

    # ── 6. Market aggregation ─────────────────────────────────────────────────
    p_market = aggregate_markets(evidence)

    # ── 7. LLM estimation ────────────────────────────────────────────────────
    print(f"  Reasoning with {model}...", end=" ", flush=True)
    llm_result = await estimate(question, evidence)
    print("done\n")

    if llm_result.get("abstain"):
        print("  INSUFFICIENT EVIDENCE: the model determined there is not enough")
        print("  information to make a calibrated forecast for this question.\n")
        sys.exit(0)

    p_llm = llm_result.get("probability")
    if p_llm is None:
        print("CANNOT ANSWER: model returned no probability.")
        sys.exit(1)

    # ── 8. Combine LLM + market signals ──────────────────────────────────────
    p_final = combine(p_llm, p_market)

    # ── 9. Apply Platt calibration if available ───────────────────────────────
    p_corrected, was_calibrated = apply_calibration(p_final)
    if was_calibrated:
        logger.debug("Platt calibration: %.3f → %.3f", p_final, p_corrected)
        p_final = p_corrected

    # ── 10. Format and print ──────────────────────────────────────────────────
    output = format_output(
        question=question,
        p_final=p_final,
        evidence=evidence,
        llm_result=llm_result,
        p_llm=p_llm,
        p_market=p_market,
    )
    print(output)

    # ── 11. Save prediction log ───────────────────────────────────────────────
    saved_path = save_prediction(
        question=question,
        p_final=p_final,
        p_llm=p_llm,
        p_market=p_market,
        llm_result=llm_result,
        evidence=evidence,
    )
    print(f"\n  Saved to: {saved_path.relative_to(Path(__file__).parent)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geopolitical Oracle — calibrated yes/no probability forecasts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "Will France hold snap elections before July 2026?"\n'
            '  python main.py "Will the Fed cut rates in June 2026?"\n'
            '  python main.py --model llama3.3:70b "Will China invade Taiwan this year?"\n'
            "  python main.py --calibrate\n"
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="A yes/no question resolvable within 2 years.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="TAG",
        help="Ollama model tag to use (e.g. llama3.1:8b, llama3.3:70b, mistral). "
             "Overrides OLLAMA_MODEL env var.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Show calibration statistics on resolved predictions.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
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

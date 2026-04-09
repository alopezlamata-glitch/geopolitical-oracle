#!/usr/bin/env python3
"""
Geopolitical Oracle — CLI entry point.

Usage:
    python main.py "Will France hold snap elections before July 2026?"
    python main.py "Will the Fed cut rates in June 2026?"
    python main.py --calibrate
    python main.py --verbose "Will China launch military exercises near Taiwan this month?"
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

# Ensure project root is on the path so subpackages resolve correctly
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
from output import format_output, save_prediction, run_calibration, apply_calibration

logger = logging.getLogger(__name__)


async def run_oracle(question: str) -> None:
    """Full oracle pipeline for a single question."""

    # ── 1. Validate API key early ────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    # ── 2. Normalize question ────────────────────────────────────────────────
    question = question.strip()
    if not question.endswith("?"):
        question += "?"

    print(f"\n  Gathering evidence for: {question}")
    print("  Please wait...\n")

    # ── 3. Run all collectors concurrently ───────────────────────────────────
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            collect_metaculus(session, question),
            collect_polymarket(session, question),
            collect_gdelt(session, question),
            collect_wikipedia(session, question),
            collect_rss(session, question),
            return_exceptions=True,
        )

    # ── 4. Separate failures from evidence blocks ────────────────────────────
    collector_names = ["metaculus", "polymarket", "gdelt", "wikipedia", "rss"]
    evidence: list[EvidenceBlock] = []
    for name, result in zip(collector_names, results):
        if isinstance(result, Exception):
            logger.warning("Collector '%s' failed: %s", name, result)
        elif result is not None:
            evidence.append(result)
        else:
            logger.warning("Collector '%s' returned no data", name)

    # ── 5. Check if we have anything at all ──────────────────────────────────
    if not evidence:
        print("CANNOT ANSWER: insufficient evidence — all collectors failed.")
        sys.exit(1)

    # ── 6. Market aggregation (synchronous, fast) ────────────────────────────
    p_market = aggregate_markets(evidence)

    # ── 7. LLM estimation (async, longest step) ──────────────────────────────
    llm_result = await estimate(question, evidence)

    if llm_result.get("abstain"):
        print("INSUFFICIENT EVIDENCE: the LLM determined there is not enough")
        print("information to make a calibrated forecast for this question.")
        sys.exit(0)

    p_llm = llm_result.get("probability")
    if p_llm is None:
        print("CANNOT ANSWER: LLM returned no probability.")
        sys.exit(1)

    # ── 8. Combine LLM + market estimates ────────────────────────────────────
    p_final = combine(p_llm, p_market)

    # ── 9. Apply Platt calibration correction if available ───────────────────
    p_corrected, was_calibrated = apply_calibration(p_final)
    if was_calibrated:
        logger.debug("Platt calibration applied: %.3f → %.3f", p_final, p_corrected)
        p_final = p_corrected

    # ── 10. Format and print ─────────────────────────────────────────────────
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
        description="Geopolitical Oracle — answers yes/no questions with calibrated probabilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "Will France hold snap elections before July 2026?"\n'
            '  python main.py "Will the Fed cut rates in June 2026?"\n'
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
        "--calibrate",
        action="store_true",
        help="Show calibration statistics on resolved predictions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    # Load .env from the project root
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    if args.calibrate:
        run_calibration()
        return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    # Windows Python 3.10+ uses ProactorEventLoop by default, compatible with aiohttp
    # For Python 3.8/3.9 on Windows, uncomment the line below:
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_oracle(args.question))


if __name__ == "__main__":
    main()

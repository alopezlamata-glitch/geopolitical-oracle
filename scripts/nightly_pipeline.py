"""
Nightly pipeline orchestrator.

Runs the full daily update sequence:
  1. update_world_state    — collect fresh events, update EMA-smoothed state
  2. fit_transition_model  — refit VAR models on new history
  3. auto_resolve          — close overdue questions with high-confidence Ollama
  4. blend_calibrate       — update blend weights from resolved predictions
  5. evaluate              — emit eval report (logged, not printed)

Designed to be run by cron at ~02:00 UTC:
  0 2 * * * cd /path/to/oracle && python scripts/nightly_pipeline.py >> logs/nightly.log 2>&1

Run manually:
  python scripts/nightly_pipeline.py [--skip update] [--skip fit]
  python scripts/nightly_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nightly")


def _step(name: str, fn, skip: set[str], dry_run: bool) -> bool:
    if name in skip:
        logger.info("SKIP  %s", name)
        return True
    if dry_run:
        logger.info("DRY   %s", name)
        return True
    logger.info("START %s", name)
    t0 = time.monotonic()
    try:
        fn()
        logger.info("OK    %s (%.1fs)", name, time.monotonic() - t0)
        return True
    except Exception as e:
        logger.error("FAIL  %s: %s", name, e)
        return False


def run(skip: set[str] | None = None, dry_run: bool = False) -> None:
    skip = skip or set()
    results: dict[str, bool] = {}

    # ── 1. World state update ─────────────────────────────────────────────────
    def _update_world_state():
        from scripts.update_world_state import run as _run
        asyncio.run(_run())

    results["update"] = _step("update_world_state", _update_world_state, skip, dry_run)

    # ── 2. Fit transition models ──────────────────────────────────────────────
    def _fit_var():
        from scripts.fit_transition_model import run as _run
        _run()

    results["fit"] = _step("fit_transition_model", _fit_var, skip, dry_run)

    # ── 3. Auto-resolve overdue questions ─────────────────────────────────────
    def _auto_resolve():
        from scripts.auto_resolve import run as _run
        _run(limit=30, min_confidence=0.85)

    results["resolve"] = _step("auto_resolve", _auto_resolve, skip, dry_run)

    # ── 4. Blend calibration ──────────────────────────────────────────────────
    def _blend_calibrate():
        from predictor.blend_calibrator import run_blend_calibration
        run_blend_calibration(verbose=False)

    results["blend"] = _step("blend_calibrate", _blend_calibrate, skip, dry_run)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    def _evaluate():
        from scripts.evaluate import run as _run
        _run(save=True)

    results["evaluate"] = _step("evaluate", _evaluate, skip, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok   = [k for k, v in results.items() if v]
    fail = [k for k, v in results.items() if not v]
    logger.info(
        "nightly done: %d ok (%s) | %d failed (%s)",
        len(ok),   ", ".join(ok)   or "-",
        len(fail), ", ".join(fail) or "-",
    )
    if fail:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly oracle pipeline")
    parser.add_argument("--skip", nargs="+", metavar="STEP",
                        choices=["update", "fit", "resolve", "blend", "evaluate"],
                        default=[], help="Steps to skip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log steps but don't execute")
    args = parser.parse_args()
    run(skip=set(args.skip), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

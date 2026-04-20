"""
Auto-resolve overdue predictions using Ollama LLM.

The flywheel:
  1. Find open questions past their deadline (overdue_questions view)
  2. Collect fresh news headlines for each question
  3. Ask Ollama: "Did this event happen? YES / NO / UNCERTAIN + confidence"
  4. If confidence >= MIN_CONFIDENCE: write resolution to question_resolutions
     → This creates a labeled training example for future model retraining

Resolution quality:
  - Only resolves when Ollama confidence >= 0.85 (avoid noisy labels)
  - Marks is_ambiguous=True for uncertain cases (excluded from training view)
  - Logs all resolution attempts for human audit

Run:
  python scripts/auto_resolve.py [--dry-run] [--limit 20] [--min-confidence 0.85]

Add to cron for nightly auto-resolution:
  0 2 * * * cd /path/to/oracle && python scripts/auto_resolve.py >> logs/auto_resolve.log 2>&1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_resolve")

_MIN_CONFIDENCE_DEFAULT = 0.85
_LIMIT_DEFAULT = 20
_MAX_HEADLINES = 20

_RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["YES", "NO", "UNCERTAIN"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["outcome", "confidence", "reasoning"],
}

_RESOLUTION_SYSTEM = """\
You are a resolution judge for binary prediction questions.
Your task: determine whether a specific event occurred before its deadline.
Be strict: only say YES or NO if the evidence clearly confirms or denies the event.
Say UNCERTAIN if the evidence is insufficient or ambiguous.
Output only valid JSON."""

_RESOLUTION_PROMPT = """\
Binary prediction question: {question}
Resolution deadline: {deadline}

Recent news headlines ({n} headlines):
{headlines}

Did this event occur before the deadline?
- YES: The event clearly happened before {deadline}
- NO: The event clearly did NOT happen before {deadline}
- UNCERTAIN: Evidence is insufficient or ambiguous

Respond with JSON:
{{
  "outcome": "YES" | "NO" | "UNCERTAIN",
  "confidence": <float 0.0-1.0, how confident are you?>,
  "reasoning": "<1-2 sentence explanation>",
  "key_evidence": ["<headline that supports your conclusion>"]
}}"""


async def _collect_headlines(question: str, country: str = "") -> list[str]:
    """Collect fresh headlines for resolution check."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from collector.gdelt import GDELTCollector
        from collector.rss import RSSCollector

        gdelt = GDELTCollector()
        rss = RSSCollector()

        results = await asyncio.gather(
            gdelt.collect(question, country=country or None),
            rss.collect(question, country=country or None),
            return_exceptions=True,
        )
        events = []
        for r in results:
            if isinstance(r, list):
                events.extend(r)

        return [e.title for e in events if hasattr(e, "title") and e.title.strip()][:_MAX_HEADLINES]
    except Exception as e:
        logger.warning("auto_resolve: headline collection failed: %s", e)
        return []


async def _resolve_question_async(
    question_text: str,
    deadline: str,
    headlines: list[str],
    client,
) -> dict:
    """Ask Ollama to resolve a question."""
    if not headlines:
        return {"outcome": "UNCERTAIN", "confidence": 0.0, "reasoning": "No evidence collected."}

    block = "\n".join(f"- {h[:150]}" for h in headlines[:_MAX_HEADLINES])
    n = len(headlines)

    prompt = _RESOLUTION_PROMPT.format(
        question=question_text[:300],
        deadline=deadline,
        n=n,
        headlines=block,
    )

    raw = await client.generate_json(
        prompt=prompt,
        system=_RESOLUTION_SYSTEM,
        temperature=0.0,
        schema=_RESOLUTION_SCHEMA,
    )

    if not isinstance(raw, dict):
        return {"outcome": "UNCERTAIN", "confidence": 0.0, "reasoning": "LLM returned invalid response."}

    outcome = raw.get("outcome", "UNCERTAIN")
    if outcome not in ("YES", "NO", "UNCERTAIN"):
        outcome = "UNCERTAIN"

    return {
        "outcome": outcome,
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
        "reasoning": raw.get("reasoning", ""),
        "key_evidence": raw.get("key_evidence", []),
    }


def run(dry_run: bool = False, limit: int = _LIMIT_DEFAULT, min_confidence: float = _MIN_CONFIDENCE_DEFAULT) -> None:
    from data_layer.db import get_db, init_schema
    from llm.client import get_client

    init_schema()
    db = get_db()

    # Find overdue open questions
    overdue = db.execute(
        """
        SELECT q.question_id, q.raw_text, q.deadline, q.jurisdiction
        FROM overdue_questions q
        LIMIT ?
        """,
        [limit],
    ).fetchall()

    if not overdue:
        logger.info("auto_resolve: no overdue questions found")
        return

    logger.info("auto_resolve: found %d overdue questions (limit=%d)", len(overdue), limit)

    client = get_client()

    resolved_count = 0
    skipped_count = 0
    uncertain_count = 0

    for question_id, question_text, deadline, jurisdiction in overdue:
        deadline_str = str(deadline)
        country = jurisdiction or ""

        logger.info("auto_resolve: processing '%s...' (deadline %s)", question_text[:60], deadline_str)

        # Collect fresh headlines
        headlines = asyncio.run(_collect_headlines(question_text, country))
        logger.info("  collected %d headlines", len(headlines))

        # Ask Ollama to resolve
        try:
            result = asyncio.run(_resolve_question_async(question_text, deadline_str, headlines, client))
        except Exception as e:
            logger.warning("  resolution failed: %s", e)
            skipped_count += 1
            continue

        outcome_str = result["outcome"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]

        logger.info(
            "  outcome=%s confidence=%.2f reasoning=%s",
            outcome_str, confidence, reasoning[:80],
        )

        if outcome_str == "UNCERTAIN":
            uncertain_count += 1
            # Still write to DB as ambiguous (for audit, excluded from training view)
            if not dry_run:
                resolution_id = str(uuid.uuid4()).replace("-", "")[:32]
                try:
                    db.execute(
                        """
                        INSERT OR IGNORE INTO question_resolutions (
                            resolution_id, question_id, outcome,
                            resolved_at, deadline_was,
                            resolver_source, resolution_notes,
                            resolution_confidence, is_ambiguous, resolved_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            resolution_id, question_id, -1,
                            datetime.now(timezone.utc), deadline,
                            "ollama_auto", f"UNCERTAIN: {reasoning[:200]}",
                            confidence, True, "auto_resolve",
                        ],
                    )
                except Exception as e:
                    logger.debug("  uncertain write failed: %s", e)
            continue

        if confidence < min_confidence:
            logger.info("  skipping: confidence %.2f < %.2f threshold", confidence, min_confidence)
            skipped_count += 1
            continue

        outcome_int = 1 if outcome_str == "YES" else 0

        if dry_run:
            logger.info("  [DRY RUN] Would write outcome=%d confidence=%.2f", outcome_int, confidence)
            resolved_count += 1
            continue

        # Write resolution to lakehouse
        resolution_id = str(uuid.uuid4()).replace("-", "")[:32]
        try:
            db.execute(
                """
                INSERT OR IGNORE INTO question_resolutions (
                    resolution_id, question_id, outcome,
                    resolved_at, deadline_was,
                    resolver_source, resolution_notes,
                    resolution_confidence, is_ambiguous, resolved_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    resolution_id, question_id, outcome_int,
                    datetime.now(timezone.utc), deadline,
                    "ollama_auto",
                    json.dumps({"reasoning": reasoning, "key_evidence": result.get("key_evidence", [])}),
                    confidence, False, "auto_resolve",
                ],
            )
            # Update question status to resolved
            db.execute(
                "UPDATE questions SET status = 'resolved' WHERE question_id = ?",
                [question_id],
            )
            # Update feature_snapshots outcome
            db.execute(
                """
                UPDATE feature_snapshots
                SET outcome = ?, outcome_resolved_at = ?
                WHERE question_id = ? AND outcome IS NULL
                """,
                [outcome_int, datetime.now(timezone.utc), question_id],
            )
            logger.info("  RESOLVED: outcome=%d (confidence=%.2f)", outcome_int, confidence)
            resolved_count += 1
        except Exception as e:
            logger.warning("  DB write failed: %s", e)
            skipped_count += 1

    total = len(overdue)
    logger.info(
        "auto_resolve: done. %d resolved, %d uncertain, %d skipped / %d total",
        resolved_count, uncertain_count, skipped_count, total,
    )

    if resolved_count > 0 and not dry_run:
        # Check if we now have enough for retraining
        n = db.execute("SELECT COUNT(*) FROM training_ready_snapshots").fetchone()[0]
        logger.info("training_ready_snapshots now has %d rows", n)
        if n >= 30:
            logger.info("Tip: run 'python main.py train' to retrain on new labeled data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-resolve overdue predictions via Ollama")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be resolved without writing")
    parser.add_argument("--limit", type=int, default=_LIMIT_DEFAULT, help="Max questions to process")
    parser.add_argument(
        "--min-confidence", type=float, default=_MIN_CONFIDENCE_DEFAULT,
        help="Minimum Ollama confidence to write a resolution (default: 0.85)",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run, limit=args.limit, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()

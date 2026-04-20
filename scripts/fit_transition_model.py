"""
Fit (or retrain) the VAR transition models after a world state update.

Run daily after update_world_state.py:
  python scripts/fit_transition_model.py

Options:
  --entity NAME [NAME ...]  Only fit for these entities
  --verbose                 Debug output
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_var")


def run(entity_names: list[str] | None = None) -> None:
    import hashlib
    from data_layer.db import init_schema, table_exists, get_db
    from model.transition_var import fit, ALL_DOMAINS, list_fitted_models

    init_schema()

    if not table_exists("world_state_history"):
        logger.error("world_state_history table missing — run update_world_state first")
        return

    db = get_db()
    if entity_names:
        entity_ids = [
            "ent_" + hashlib.sha256(f"country|{n.lower()}".encode()).hexdigest()[:24]
            for n in entity_names
        ]
    else:
        rows = db.execute(
            "SELECT DISTINCT entity_id FROM world_state_history"
        ).fetchall()
        entity_ids = [r[0] for r in rows]

    if not entity_ids:
        logger.warning("no entities in world_state_history — nothing to fit")
        return

    logger.info("fitting VAR models for %d entities × %d domains...",
                len(entity_ids), len(ALL_DOMAINS))

    n_ok = 0
    for eid in entity_ids:
        for domain in ALL_DOMAINS:
            m = fit(eid, domain)
            if m:
                n_ok += 1

    models = list_fitted_models()
    logger.info("done: %d models fitted, %d total on disk", n_ok, len(models))

    # Summary
    if models:
        print(f"\n{'Entity':<30} {'Domain':<12} {'Type':<5} {'N_obs':>6} {'Lag':>4}")
        print("-" * 60)
        for m in sorted(models, key=lambda x: (x.get("entity_id",""), x.get("domain",""))):
            print(
                f"  {m.get('entity_id','?')[:26]:<28} "
                f"{m.get('domain','?'):<12} "
                f"{m.get('type','?'):<5} "
                f"{m.get('n_obs', 0):>6} "
                f"{m.get('lag_order', '-'):>4}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit VAR transition models")
    parser.add_argument("--entity", nargs="+", metavar="NAME")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(entity_names=args.entity)

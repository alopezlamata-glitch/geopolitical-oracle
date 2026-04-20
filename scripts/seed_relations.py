"""
Seed the entity_relations_temporal table from data/relations_seed.json.

Safe to run multiple times — write_relation skips duplicates.

Usage:
  python scripts/seed_relations.py
  python scripts/seed_relations.py --verbose
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
logger = logging.getLogger("seed_relations")


def run() -> None:
    from data_layer.db import init_schema, get_db, table_exists
    from data_layer.relation_writer import seed_from_file

    init_schema()

    seed_path = Path(__file__).parent.parent / "data" / "relations_seed.json"
    if not seed_path.exists():
        logger.error("seed file not found: %s", seed_path)
        return

    logger.info("seeding entity relations from %s", seed_path)
    n = seed_from_file(str(seed_path))
    logger.info("done: %d relations written", n)

    # Summary
    db = get_db()
    n_rels  = db.execute("SELECT COUNT(*) FROM entity_relations_temporal").fetchone()[0]
    n_ents  = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_open  = db.execute("SELECT COUNT(*) FROM entity_relations_temporal WHERE valid_to IS NULL").fetchone()[0]

    print(f"\nEntity relations: {n_rels} total, {n_open} currently valid")
    print(f"Entities:         {n_ents}")

    # Show leader tenure for key countries
    from world_state.relation_reader import get_country_relation_features, get_leader_name
    print(f"\n{'Country':<20} {'Leader':<25} {'Tenure':<12} {'Conflict':>8} {'Sanctions':>9}")
    print("-" * 76)
    for country in ["Ukraine", "Russia", "United States", "Israel", "China", "Germany", "France"]:
        feats = get_country_relation_features(country)
        leader = get_leader_name(country) or "?"
        tenure = feats.get("leader_tenure_years", 0)
        conflict = "YES" if feats.get("has_active_conflict") else "no"
        sanctions = f"{feats.get('n_sanctions', 0):.1f}"
        invest = " *" if feats.get("leader_under_investigation") else ""
        print(f"  {country:<18} {(leader + invest):<25} {tenure:<12.1f} {conflict:>8} {sanctions:>9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed entity relations")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run()

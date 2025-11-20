#!/usr/bin/env python3
"""
Utility: Update lottery CSVs for all games.

Usage:
  - CSV-first (reads CSV; fetches only if behind):
      python update_all_games.py
  - Force web fetch and update regardless of CSV freshness:
      python update_all_games.py --force

This wraps ProductionMichiganScraper with clear progress output and
JSON summary for quick diagnostics.
"""

import argparse
import json
import sys
import time
from datetime import datetime

try:
    from fixed_scraper import ProductionMichiganScraper
except Exception as e:
    print(f"ERROR: failed to import fixed_scraper. {e}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force web fetch for all games (ignore CSV cache)"
    )
    args = parser.parse_args()

    scraper = ProductionMichiganScraper()

    start = time.time()
    print("==== Updating Lottery CSVs ====")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode   : {'FORCE FETCH' if args.force else 'CSV-FIRST'}")
    print("================================")

    try:
        if args.force:
            results = scraper.force_fetch_today()
        else:
            results = scraper.fetch_all_games()
    except Exception as e:
        print(f"ERROR during update: {e}", file=sys.stderr)
        return 2

    # Print compact summary and machine-readable JSON
    print("\n---- Summary ----")
    failures = 0
    for game, res in results.items():
        ok = res.get("success") is True
        if not ok:
            failures += 1
        src = res.get("source", "?")
        date = res.get("draw_date", "?")
        nums = res.get("numbers", [])
        print(f"{game:14} | {'OK' if ok else 'FAIL'} | {src:10} | {date} | {nums}")

    print("\nJSON:")
    try:
        print(json.dumps(results, ensure_ascii=False))
    except TypeError:
        # Fallback serialization
        safe = {k: {**v} for k, v in results.items()}
        print(json.dumps(safe, ensure_ascii=False))

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Standalone script to rebuild unified lottery CSVs.
Usage (PowerShell):
  .\.conda\Scripts\python.exe rebuild_all_csvs.py
"""
from fixed_scraper import ProductionMichiganScraper

def main():
    scraper = ProductionMichiganScraper()
    summary = scraper.rebuild_all_unified_csvs()
    print("\n✅ Rebuild complete. Row counts:")
    for game, count in sorted(summary.items()):
        print(f"  - {game}: {count} rows")

if __name__ == "__main__":
    main()
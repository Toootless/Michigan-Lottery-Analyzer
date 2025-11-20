# Documentation Update — November 10, 2025

This note summarizes the changes introduced today to improve the Learning Dashboard’s reliability and transparency around CSV data sources.

## What’s new (v3.0.4)

- Learning Dashboard auto-repair when CSV history is empty:
  - Ensure unified CSV exists, re-read history
  - Attempt a lightweight update, re-read again
- Inline quick actions when no data is available:
  - "🌐 Fetch Latest Now" forces a web fetch + CSV update and refreshes
  - "✍️ Add Manual Draw" prepares the manual insertion flow (hooks in place)
- Shared scraper instance (`get_csv_scraper`) to avoid redundant instantiation and state skew
- CSV diagnostics helpers in `fixed_scraper.py`:
  - `get_csv_info(game)` returns `{ path, exists, rows, latest_date, headers }`
  - `get_unified_csv_path(game)` resolves the CSV path for a game
- Automatic CSV creation on write:
  - `ProductionMichiganScraper._update_csv_file()` creates missing CSVs on first update
  - `append_manual_draw()` creates missing CSVs with appropriate schema per game

## UI enhancements

- Learning Dashboard now shows a CSV status caption (path • exists • rows • latest date) when no data is available
- Recent Winning Numbers uses structured fields (`main_numbers`, `bonus_number`, `bonus_label`) for bonus games to label Mega Ball / Powerball / Lucky Ball consistently

## How to launch

- Windows (recommended):
  ```powershell
  ./run_MLA4.ps1 -Address localhost -Port 8520
  ```
- Cross-platform:
  ```bash
  python -m streamlit run MLA4.py --server.address localhost --server.port 8520
  ```

## Troubleshooting tips

- "OpenAI not available" warning: Set `OPENAI_API_KEY` if you want chat features; not required for CSV-based dashboards
- Powerball/Mega Millions not showing: Click "🌐 Fetch Latest Now" in the Learning Dashboard or use the sidebar refresh options
- Still empty? Check `logs/rebuild_summary.csv` and the CSV diagnostics caption for the selected game

## Next steps (optional)

- Add inline manual-entry form to append a draw directly from the dashboard
- Extend structured fields to the sidebar’s recent results widget
- Add tests for CSV creation, duplicate-date prevention, and bonus parsing

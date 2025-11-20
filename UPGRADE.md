# Upgrade Guide to v3.1.0

## Overview
Version 3.1.0 introduces a structured LotteryUSA draw parser and full Powerball Double Play support (numbers, bonus, multipliers). This guide provides safe, incremental steps to upgrade existing installations.

## Who Should Use This Guide
- You have a previous version (≤ v3.0.5) checked out locally.
- Your Powerball CSV does NOT yet include Double Play or Power Play columns.
- You experienced wrong Mega Millions numbers due to legacy parsing.

## Key Changes in v3.1.0
- Structured parser `_parse_lotteryusa_structured` (resilient to DOM changes).
- Expanded Powerball schema:
  `Date, Result, Powerball, Double Play, Double Play Powerball, Power Play, Double Play Multiplier`
- Metadata propagation: `main_numbers`, `bonus_number`, `double_play_numbers`, `double_play_bonus`, `multiplier`, `double_play_multiplier`.
- Date normalization / duplicate removal for Powerball & Mega Millions.
- Import resilience (zstd shim) preventing `requests` failure.

## Prerequisites
1. Python 3.11 environment active.
2. Ensure `beautifulsoup4` and `requests` installed:
```bash
pip install requests beautifulsoup4
```
3. Back up existing CSVs (optional but recommended):
```powershell
Copy-Item final_integrated_data\Powerball_Complete.csv backup_Powerball_Complete.csv
Copy-Item final_integrated_data\Mega_Millions_Complete.csv backup_Mega_Millions_Complete.csv
```

## Step 1: Pull / Sync Code
If using git:
```bash
git pull origin main
```
(If not under git, copy new `fixed_scraper.py`, `README.md`, `CHANGELOG_UPDATED.md`, `LOTTERY_DATA_SYSTEM_GUIDE.md` and this `UPGRADE.md`.)

## Step 2: Regenerate Powerball & Mega Millions Rows (Optional Cleanup)
Run the application or a quick Python snippet to force fetch latest draws:
```powershell
python update_all_games.py --force
```
This ensures the expanded schema is applied and new metadata captured.

## Step 3: Deduplicate Existing Rows
Already performed by integrated normalization logic. To re-run manually:
```powershell
python - <<'PY'
from pathlib import Path
import csv, re
for name in ['final_integrated_data/Mega_Millions_Complete.csv','final_integrated_data/Powerball_Complete.csv']:
    p=Path(name); rows=list(csv.DictReader(p.open()))
    seen=set(); out=[]
    for r in rows:
        d=re.sub(r'\s*,\s*', ', ', (r.get('Date','').strip('"')))
        if d in seen: continue
        seen.add(d); r['Date']=d; out.append(r)
    p.open('w',newline='').write('')
    w=csv.DictWriter(p.open('w',newline=''), fieldnames=out[0].keys()); w.writeheader(); [w.writerow(r) for r in out]
    print('Cleaned', name, 'rows=', len(out))
PY
```

## Step 4: Verify Structured Parser Works
Quick check:
```powershell
python - <<'PY'
from fixed_scraper import ProductionMichiganScraper
s=ProductionMichiganScraper()
print(s._fetch_from_lotteryusa('Powerball'))
print(s._fetch_from_lotteryusa('Mega Millions'))
PY
```
Expect to see `double_play_numbers` and `double_play_bonus` for Powerball when available.

## Step 5: Integrate With Streamlit UI
Nothing breaking—existing code reading `numbers` continues working. To display new metadata, extend UI components using keys:
- `double_play_numbers`
- `double_play_bonus`
- `multiplier`

## Step 6: Manual Entry (Powerball)
If creating a new Powerball CSV via manual entry, schema is auto-expanded. No additional action required.

## Rollback
Restore backed up CSVs and revert to previous commit/tag:
```bash
git checkout v3.0.5
```

## Post-Upgrade Validation Checklist
| Item | Expected |
|------|----------|
| Powerball CSV header | Contains Double Play & multiplier columns |
| Mega Millions latest row | Structured date formatting (e.g., "Tuesday, Nov 11, 2025") |
| Fetch returns metadata | `main_numbers` + `bonus_number` present |
| Double play appearing | Present on draws where offered |
| No duplicated date rows | Only one row per draw date |

## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Missing Double Play columns | Old CSV not rewritten | Run force fetch or append manual row to trigger rewrite |
| Import error `ZstdError` | Env compression shim absent | Ensure updated `fixed_scraper.py` present |
| Wrong Mega Millions numbers | Legacy unstructured parse | Force fetch to regenerate latest row |

## Next Steps
- Add UI display for Double Play metrics.
- Extend structured parser to Lotto 47 once source stabilizes.
- Consider adding tests for multiplier parsing invariants.

## Release Tagging (Optional)
If using git:
```bash
git add .
git commit -m "Release v3.1.0: structured parser + double play support"
git tag v3.1.0
git push origin main --tags
```

---
For further enhancements, see `CHANGELOG_UPDATED.md` and `LOTTERY_DATA_SYSTEM_GUIDE.md`.

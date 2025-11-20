# Release Notes v3.1.0 (2025-11-13)

## Highlights
- Structured LotteryUSA parser for stable extraction despite DOM shifts.
- Full Powerball Double Play capture: numbers, bonus, Power Play, Double Play Multiplier.
- Mega Millions corrected (legacy mismatch fixed via structured parse).
- Environment resilience (zstd shim) prevents `requests` import failures.

## Data & Schema
- Powerball CSV header expanded: `Date, Result, Powerball, Double Play, Double Play Powerball, Power Play, Double Play Multiplier`.
- Metadata fields available across fetch responses and CSV rows.
- Duplicate date rows normalized (spacing and comma formatting).

## Reliability
- Lotto 47 pages 404 on LotteryUSA; retain historical CSV until alternative source integrated.
- Lucky for Life structured page parsed; multiple rows accessible.

## Upgrade Steps
See `UPGRADE.md` for detailed procedure. Core actions:
1. Pull new code.
2. Force fetch games (`python update_all_games.py --force`).
3. Optional dedup script for Powerball & Mega Millions.

## Backward Compatibility
- Existing consumers of `numbers` unaffected.
- New metadata keys optional; absent for games without double play.

## Verification Checklist
- Powerball latest row includes Double Play columns.
- Mega Millions latest row uses structured date (e.g., `Tuesday, Nov 11, 2025`).
- Fetch returns `main_numbers` + `bonus_number`.
- No duplicate date rows post-cleanup.

## Known Limitations
- Lotto 47 structured parser pending stable source.
- Fallback sites may not expose Double Play/multiplier consistently.

## Next Roadmap Targets
- Integrate Lotto 47 structured alternative.
- UI enhancement for displaying Double Play multipliers.
- Automated test suite for parser regressions.

---
Refer to `CHANGELOG_UPDATED.md` and `LOTTERY_DATA_SYSTEM_GUIDE.md` for deeper technical details.

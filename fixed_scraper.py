#!/usr/bin/env python3
"""
Production-Ready Michigan Lottery Scraper
CSV-based fetch system with daily data updates
"""

"""Production-Ready Michigan Lottery Scraper
CSV-based fetch system with daily data updates.
This module now supports running in a degraded 'CSV-only' mode if the
`requests`/`urllib3` stack fails to import due to environment compression
module issues (e.g. missing `compression.zstd.ZstdError`).
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import json
import time
import csv
import os
from pathlib import Path

# Attempt to import networking stack; fall back gracefully if it fails.
REQUESTS_AVAILABLE = True
# Workaround: Some environments have a third-party 'compression' module that
# shadows urllib3's expectation of zstandard. Provide a minimal shim that
# exposes compression.zstd.ZstdError to satisfy urllib3 import-time checks.
try:
    import zstandard as _zstd  # noqa: F401
except Exception:
    try:
        import sys, types  # noqa: F401
        comp = sys.modules.get('compression')
        if comp is None:
            comp = types.ModuleType('compression')
            sys.modules['compression'] = comp
        zstd_mod = sys.modules.get('compression.zstd')
        if zstd_mod is None:
            zstd_mod = types.ModuleType('compression.zstd')
            sys.modules['compression.zstd'] = zstd_mod
        if not hasattr(zstd_mod, 'ZstdError'):
            class ZstdError(Exception):
                pass
            zstd_mod.ZstdError = ZstdError  # type: ignore[attr-defined]
    except Exception:
        # If shim injection fails, we'll still try to import requests and handle failure below
        pass
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception as _req_err:  # Broad except intentional: env compression issues
    REQUESTS_AVAILABLE = False
    BeautifulSoup = None  # type: ignore
    def _requests_warning():
        print(f"⚠️ Network fetch disabled (requests import failed: {_req_err}). Operating in CSV-only mode.")
    _requests_warning()

# --- Safe print monkey patch to prevent Windows cp1252 UnicodeEncodeError with emojis ---
try:
    import builtins as _builtins
    _orig_print = _builtins.print
    def _safe_print_patch(*args, **kwargs):
        try:
            _orig_print(*args, **kwargs)
        except UnicodeEncodeError:
            safe_args = []
            for a in args:
                try:
                    s = str(a)
                    safe_args.append(s.encode('ascii', 'ignore').decode())
                except Exception:
                    safe_args.append('')
            _orig_print(*safe_args, **kwargs)
    _builtins.print = _safe_print_patch
except Exception:
    # If anything goes wrong, leave print unchanged
    pass

class ProductionMichiganScraper:
    """CSV-based lottery scraper with daily data updates"""
    
    def __init__(self, data_folder: str = "past_games"):
        self.data_folder = Path(data_folder)
        # Initialize HTTP session only if requests stack is available
        self.session = None
        if REQUESTS_AVAILABLE:
            try:
                self.session = requests.Session()
                self.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'DNT': '1',
                    'Connection': 'keep-alive'
                })
            except Exception as e:
                print(f"⚠️ Failed to initialize requests session: {e}. Continuing CSV-only.")
        
        # CSV file mapping for each game (using complete integrated data)
        self.csv_files = {
            'Daily 4 Evening': 'final_integrated_data/Daily_4_Evening_Complete.csv',
            'Daily 4 Midday': 'final_integrated_data/Daily_4_Midday_Complete.csv', 
            'Daily 4': 'final_integrated_data/Daily_4_Evening_Complete.csv',  # Default to evening
            'Daily 3 Evening': 'final_integrated_data/Daily_3_Evening_Complete.csv',
            'Daily 3 Midday': 'final_integrated_data/Daily_3_Midday_Complete.csv',
            'Daily 3': 'final_integrated_data/Daily_3_Evening_Complete.csv',  # Default to evening
            'Lucky for Life': 'final_integrated_data/Lucky_for_Life_Complete.csv',
            'Lotto 47': 'final_integrated_data/Lotto_47_Complete.csv',
            'Fantasy 5': 'past_games/MI Fantasy 5 numbers from LotteryUSA.csv',
            'Fantasy 5 Double Play': 'past_games/MI Fantasy 5 numbers from LotteryUSA.csv',
            'Powerball': 'final_integrated_data/Powerball_Complete.csv',
            'Mega Millions': 'final_integrated_data/Mega_Millions_Complete.csv'
        }
        
        # LotteryUSA.com scraping endpoints (for new data)
        self.sources = {
            'lotteryusa': {
                # Try base page first; we'll also try with a trailing 'results' as a fallback per game
                'base_url': 'https://www.lotteryusa.com/michigan/{}/',
                'games': {
                    'Daily 4': 'daily-4',
                    'Daily 3': 'daily-3',
                    'Lucky for Life': 'lucky-for-life',
                    'Lotto 47': 'lotto-47',
                    'Powerball': 'powerball',
                    'Mega Millions': 'mega-millions'
                }
            }
        }
        
        # Date format tracking
        self.last_fetch_date = self._get_last_fetch_date()
    
    def _get_last_fetch_date(self) -> Optional[str]:
        """Get the date of the last fetch attempt"""
        try:
            log_file = self.data_folder / "last_fetch.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        return None
    
    def _update_last_fetch_date(self, date_str: str):
        """Update the last fetch date"""
        try:
            log_file = self.data_folder / "last_fetch.log"
            with open(log_file, 'w') as f:
                f.write(date_str)
        except Exception:
            pass
    
    def get_today_numbers(self, game: str) -> Dict[str, Any]:
        """
        Get today's winning numbers for a specific game from CSV or fetch if needed
        
        Args:
            game: Game name (e.g., 'Daily 4', 'Fantasy 5')
            
        Returns:
            Dict with winning numbers and metadata
        """
        print(f"🎯 Getting today's numbers for {game}...")
        
        result = {
            'game': game,
            'numbers': [],
            'draw_date': None,
            'success': False,
            'source': 'csv',
            'error': None
        }
        
        # First, try to get data from CSV
        csv_result = self._get_from_csv(game)
        if csv_result['success']:
            result.update(csv_result)
            return result
        
        # If CSV doesn't have today's data, check if we need to fetch
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_fetch_date != today:
            print(f"   📡 CSV data not current, fetching new data...")
            fetch_result = self._fetch_and_update_csv(game)
            if fetch_result['success']:
                result.update(fetch_result)
                self._update_last_fetch_date(today)
                return result
        
        result['error'] = "No current data available"
        return result
    
    def _get_from_csv(self, game: str) -> Dict[str, Any]:
        """Get the most recent numbers for a game from its CSV file"""
        if game not in self.csv_files:
            return {'success': False, 'error': f'No CSV file for game {game}'}
        
        # Handle both absolute and relative paths
        csv_file = self.csv_files[game]
        if csv_file.startswith('final_integrated_data/'):
            # Use the path as-is for the new integrated data
            csv_path = Path(csv_file)
        elif csv_file.startswith('past_games/'):
            # For Fantasy 5 files that already include the past_games path
            csv_path = Path(csv_file)
        else:
            # Use the old logic for backward compatibility
            csv_path = self.data_folder / csv_file
            
        if not csv_path.exists():
            return {'success': False, 'error': f'CSV file not found: {csv_path}'}
        
        try:
            # Reuse get_recent_history to consistently select the latest draw across all games
            latest = self.get_recent_history(game, count=1)
            if latest:
                rec = latest[0]
                result = {
                    'success': True,
                    'numbers': rec.get('numbers', []),
                    'draw_date': rec.get('draw_date'),
                    'source': 'csv'
                }
                for key in (
                    'main_numbers',
                    'bonus_number',
                    'bonus_label',
                    'multiplier',
                    'double_play_numbers',
                    'double_play_bonus',
                    'double_play_multiplier'
                ):
                    if key in rec and rec[key] not in (None, ''):
                        result[key] = rec[key]
                return result
            return {'success': False, 'error': 'No data in CSV'}
        except Exception as e:
            return {'success': False, 'error': f'Error reading CSV: {str(e)}'}
    
    def _parse_csv_numbers(self, row: Dict[str, str], game: str) -> List[int]:
        """Parse numbers from CSV row based on game type."""

        import re as _re

        def _clean(val: Optional[str]) -> str:
            if val is None:
                return ""
            return str(val).strip().strip('"')

        def _extract_numbers(val: Optional[str]) -> List[int]:
            cleaned = _clean(val)
            if not cleaned:
                return []
            numbers: List[int] = []
            for match in _re.findall(r"\d+", cleaned):
                try:
                    numbers.append(int(match))
                except ValueError:
                    continue
            return numbers

        def _safe_int(val: Optional[str]) -> Optional[int]:
            cleaned = _clean(val)
            if not cleaned:
                return None
            try:
                return int(cleaned)
            except ValueError:
                try:
                    return int(float(cleaned))
                except Exception:
                    return None

        # Handle Fantasy 5 games specially - they have Regular and Double Play columns
        if game == "Fantasy 5":
            numbers = _extract_numbers(row.get('Regular'))
        elif game == "Fantasy 5 Double Play":
            numbers = _extract_numbers(row.get('Double Play'))
        else:
            # Find the correct column for main numbers
            result_str = None
            for key in ('Result', 'Main draw', 'Winning Numbers'):
                if key in row and row[key]:
                    result_str = row[key]
                    break
            if result_str is None:
                for value in row.values():
                    if isinstance(value, str) and any(char.isdigit() for char in value):
                        result_str = value
                        break
            numbers = _extract_numbers(result_str)

        # Add bonus number if applicable - handle special cases
        if game == "Lucky for Life":
            lucky = _safe_int(row.get('Lucky Ball'))
            if lucky is not None:
                numbers.append(lucky)
        elif game == "Powerball":
            for key in ('Powerball', 'Power Ball', 'PB'):
                powerball_val = _safe_int(row.get(key))
                if powerball_val is not None:
                    numbers.append(powerball_val)
                    break
        elif game == "Mega Millions":
            mega_ball = _safe_int(row.get('Mega Ball'))
            if mega_ball is not None:
                numbers.append(mega_ball)

        return numbers
    
    def _fetch_and_update_csv(self, game: str) -> Dict[str, Any]:
        """Fetch new data and update the CSV file.
        Supports Daily 3/Daily 4 variants (Midday/Evening) by normalizing to base game
        and selecting the correct session on the source page when possible.
        """
        print(f"   📡 Fetching new data for {game}...")

        # Normalize Daily game variants
        session = None
        base_game = game
        if game.startswith("Daily 3"):
            base_game = "Daily 3"
            if "Midday" in game:
                session = "Midday"
            elif "Evening" in game:
                session = "Evening"
        elif game.startswith("Daily 4"):
            base_game = "Daily 4"
            if "Midday" in game:
                session = "Midday"
            elif "Evening" in game:
                session = "Evening"
        
        # Try to fetch from LotteryUSA
        if REQUESTS_AVAILABLE:
            fetch_result = self._fetch_from_lotteryusa(base_game, session=session)
            if fetch_result['success']:
                # Update the CSV file with new data
                self._update_csv_file(game, fetch_result)
                fetch_result['source'] = 'web_fetch'
                return fetch_result
        else:
            # Explicit degraded-mode notice
            return {'success': False, 'error': 'Network fetch unavailable (CSV-only mode)'}

        # Fallback: try backup sources if primary failed
        backup_result = self._try_backup_sources(base_game)
        if backup_result.get('success'):
            self._update_csv_file(game, backup_result)
            backup_result['source'] = 'web_fetch_backup'
            return backup_result
        
        return {'success': False, 'error': 'Failed to fetch new data'}
    
    def _update_csv_file(self, game: str, data: Dict[str, Any]):
        """Update CSV file with new lottery data, using unified per-game schema.
        Ensures fieldnames are a union of existing + new to avoid KeyErrors and
        keeps values unquoted (csv module handles quoting).
        """
        if game not in self.csv_files:
            return
        
        # Resolve path similarly to readers: support integrated absolute-ish paths
        csv_file = self.csv_files[game]
        if csv_file.startswith('final_integrated_data/'):
            csv_path = Path(csv_file)
        elif csv_file.startswith('past_games/'):
            csv_path = Path(csv_file)
        else:
            csv_path = self.data_folder / csv_file
        # If the target CSV does not exist yet, create it with a unified schema
        if not csv_path.exists():
            try:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                new_row = self._create_csv_row(data, game)
                fieldnames = list(new_row.keys())
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(new_row)
                print(f"   🆕 Created new CSV file {csv_path.name} with initial row")
            except Exception as e:
                print(f"   ❌ Failed to create new CSV for {game}: {e}")
            return
        
        try:
            # Read existing data
            existing_rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
            
            # Create new row
            new_row = self._create_csv_row(data, game)
            
            # Check if this date already exists
            new_date = str(new_row.get('Date', '')).strip().strip('"')
            date_exists = any(str(row.get('Date','')).strip().strip('"') == new_date for row in existing_rows)
            
            if not date_exists:
                # Write updated data
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    # Union of headers, preserving desired unified schemas
                    existing_headers = list(existing_rows[0].keys()) if existing_rows else []
                    fieldnames = list(dict.fromkeys(existing_headers + list(new_row.keys())))
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    # New data goes first
                    writer.writerow({k: new_row.get(k, '') for k in fieldnames})
                    # Then write previous rows normalized to these headers
                    for r in existing_rows:
                        writer.writerow({k: r.get(k, '') for k in fieldnames})
                
                print(f"   ✅ Updated {csv_path.name} with new data")
            else:
                print(f"   ℹ️  Data for {new_date} already exists in CSV")
                
        except Exception as e:
            print(f"   ❌ Error updating CSV: {str(e)}")

    def append_manual_draw(self, game: str, date_str: str, numbers: List[int], top_prize: Optional[str] = None, source: str = "Manual_Entry") -> Dict[str, Any]:
        """Append a manual draw row at the top of the game's CSV.
        - game: game key (e.g., 'Daily 4 Midday')
        - date_str: draw date string (e.g., '2025-11-03' or 'Monday, Nov 03, 2025')
        - numbers: list of ints for the main numbers in draw order
        - top_prize: optional top prize string; defaults inferred from existing rows when possible
        - source: value for Source column
        Returns dict success/error like other methods.
        """
        if game not in self.csv_files:
            return {'success': False, 'error': f'No CSV file for game {game}'}

        # Resolve CSV path as other methods do
        csv_file = self.csv_files[game]
        if csv_file.startswith('final_integrated_data/') or csv_file.startswith('past_games/'):
            csv_path = Path(csv_file)
        else:
            csv_path = self.data_folder / csv_file

        if not csv_path.exists():
            # Create a new file with a minimal schema appropriate for the game
            try:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                # Decide schema based on game
                if game == 'Mega Millions':
                    headers = ['Date', 'Result', 'Mega Ball']
                elif game == 'Powerball':
                    headers = ['Date', 'Result', 'Powerball', 'Double Play', 'Double Play Powerball', 'Power Play', 'Double Play Multiplier']
                elif game == 'Lucky for Life':
                    headers = ['Date', 'Result', 'Lucky Ball']
                elif game in ['Fantasy 5', 'Fantasy 5 Double Play']:
                    headers = ['Date', 'Regular', 'Double Play']
                else:
                    headers = ['Date', 'Result']
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                print(f"   🆕 Created new CSV file {csv_path.name} for manual entry")
            except Exception as e:
                return {'success': False, 'error': f'Failed creating new CSV: {e}'}

        try:
            # Read existing rows and headers
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

            # Deduplicate by date
            def _normalize_date_value(s: str) -> str:
                s = (s or '').strip().strip('"')
                return s
            new_date_norm = _normalize_date_value(date_str)
            for row in rows:
                if _normalize_date_value(row.get('Date', '')) == new_date_norm:
                    return {'success': False, 'error': f'Date {new_date_norm} already exists in CSV'}

            # Infer top prize
            if not top_prize:
                if rows and 'Top prize' in rows[0]:
                    top_prize = rows[0]['Top prize']
                else:
                    # Sensible defaults
                    if game.startswith('Daily 4'):
                        top_prize = '"$5,000 "'
                    elif game.startswith('Daily 3'):
                        top_prize = '$500 '
                    else:
                        top_prize = ''

            # Build new row structure matching existing headers when possible
            def _has_cols(prefix: str, n: int) -> bool:
                return all((f'{prefix}{i}' in headers) for i in range(1, n+1))

            new_row: Dict[str, str] = {}
            # Always include these when present
            if 'Date' in headers:
                new_row['Date'] = new_date_norm if new_date_norm.count('-') == 2 else f'"{new_date_norm}"'
            if 'Result' in headers:
                new_row['Result'] = f'"{", ".join(map(str, numbers))}"'
            if 'Top prize' in headers:
                new_row['Top prize'] = str(top_prize or '')
            if 'Source' in headers:
                new_row['Source'] = source

            # Fill Number1..NumberN if present
            if _has_cols('Number', len(numbers)):
                for i, n in enumerate(numbers, start=1):
                    new_row[f'Number{i}'] = str(n)

            # For any missing headers that might be required by writer, ensure we provide keys
            if not headers:
                # If headers missing for any reason, build a minimal schema
                headers = ['Date', 'Result', 'Top prize', 'Source'] + [f'Number{i}' for i in range(1, len(numbers)+1)]

            # Write file back with new row on top
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerow(new_row)
                for r in rows:
                    writer.writerow(r)

            return {'success': True, 'numbers': numbers, 'draw_date': new_date_norm, 'source': 'manual'}
        except Exception as e:
            return {'success': False, 'error': f'Error appending manual row: {e}'}
    
    def _create_csv_row(self, data: Dict[str, Any], game: str) -> Dict[str, str]:
        """Create a CSV row from fetched data, aligned to unified CSV schemas.
        - Dates are plain strings (no manual quoting)
        - Result is a comma-separated string for main numbers
        - Bonus columns are added per game (Mega Ball, Powerball, Lucky Ball)
        """
        numbers: List[int] = data.get('numbers') or []
        draw_date = data.get('draw_date') or datetime.now().strftime("%Y-%m-%d")
        # Normalize draw_date: ensure clean spacing and no quotes
        try:
            import re as _re
            s = str(draw_date).strip().strip('"')
            s = _re.sub(r"\s*,\s*", ", ", s)
            s = _re.sub(r",(?=\S)", ", ")
            s = _re.sub(r"\s{2,}", " ")
            draw_date = s
        except Exception:
            pass

        main = data.get('main_numbers') or (numbers[:5] if len(numbers) >= 5 else numbers)
        bonus = data.get('bonus_number')
        if bonus is None and len(numbers) > 5:
            bonus = numbers[5]

        double_play_numbers = data.get('double_play_numbers') or []
        double_play_bonus = data.get('double_play_bonus')
        multiplier = data.get('multiplier')
        double_play_multiplier = data.get('double_play_multiplier')

        # Game-specific schemas
        if game == 'Mega Millions':
            row = {'Date': draw_date, 'Result': ", ".join(map(str, main))}
            if bonus is not None:
                row['Mega Ball'] = str(bonus)
            return row
        if game == 'Powerball':
            row = {'Date': draw_date, 'Result': ", ".join(map(str, main))}
            if bonus is not None:
                row['Powerball'] = str(bonus)
            if double_play_numbers:
                row['Double Play'] = ", ".join(map(str, double_play_numbers[:5]))
            if double_play_bonus is not None:
                row['Double Play Powerball'] = str(double_play_bonus)
            if multiplier:
                row['Power Play'] = multiplier
            if double_play_multiplier:
                row['Double Play Multiplier'] = double_play_multiplier
            return row
        if game == 'Lucky for Life':
            row = {'Date': draw_date, 'Result': ", ".join(map(str, main))}
            if bonus is not None:
                row['Lucky Ball'] = str(bonus)
            return row
        if game in ['Fantasy 5', 'Fantasy 5 Double Play']:
            # For unified Fantasy 5, we store both Regular and Double Play; when fetching new, put in Regular
            return {'Date': draw_date, 'Regular': ", ".join(map(str, main)), 'Double Play': ''}
        # Default: Daily 3, Daily 4, Lotto 47, etc. just use Date + Result
        return {'Date': draw_date, 'Result': ", ".join(map(str, main))}

    def _parse_lotteryusa_structured(self, soup: 'BeautifulSoup', game: str, session: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse the modern LotteryUSA results table for structured draw data."""
        rows = soup.select('tr.c-results-table__item')
        if not rows:
            return None

        session_lc = session.lower() if session else None
        selected_row = None
        if session_lc:
            for row in rows:
                text = row.get_text(" ", strip=True).lower()
                if session_lc in text:
                    selected_row = row
                    break
        if selected_row is None:
            selected_row = rows[0]

        date_dow = selected_row.select_one('.c-draw-card__draw-date-dow')
        date_sub = selected_row.select_one('.c-draw-card__draw-date-sub')
        if date_dow and date_sub:
            draw_date = f"{date_dow.get_text(strip=True)} {date_sub.get_text(strip=True)}".replace(' ,', ',').strip()
        else:
            # Fallback to any time element text
            date_el = selected_row.select_one('time')
            draw_date = date_el.get_text(strip=True) if date_el else None

        result: Dict[str, Any] = {'draw_date': draw_date}

        def _capture_box(box, is_primary: bool = False):
            title_el = box.select_one('.c-draw-card__ball-title')
            title_text = title_el.get_text(strip=True).lower() if title_el else ''
            balls = [int(li.get_text(strip=True)) for li in box.select('li.c-ball') if li.get_text(strip=True).isdigit()]
            bonus_el = box.select_one('li.c-result__bonus .c-ball')
            bonus_val = int(bonus_el.get_text(strip=True)) if bonus_el and bonus_el.get_text(strip=True).isdigit() else None
            multiplier_el = box.select_one('li.c-result__multiplier')
            multiplier_text = multiplier_el.get_text(strip=True) if multiplier_el else None

            return title_text, balls, bonus_val, multiplier_text

        boxes = selected_row.select('div.c-draw-card__ball-box')
        for box in boxes:
            title_text, balls, bonus_val, multiplier_text = _capture_box(box)

            # Determine box type
            is_double_play = 'double' in title_text and 'play' in title_text
            is_main = False
            if not title_text:
                is_main = True
            elif session_lc and session_lc in title_text:
                is_main = True
            elif any(token in title_text for token in ['main', 'regular', 'winning numbers', 'results']):
                is_main = True
            elif game in ('Powerball', 'Mega Millions') and not is_double_play:
                # Powerball/Mega typically list main draw first even if title missing
                if 'double' not in title_text:
                    is_main = True

            if is_double_play:
                result['double_play_numbers'] = balls[:5]
                if bonus_val is not None:
                    result['double_play_bonus'] = bonus_val
                if multiplier_text:
                    result['double_play_multiplier'] = multiplier_text
                continue

            if is_main:
                result['main_numbers'] = balls[:5]
                if bonus_val is not None:
                    result['bonus_number'] = bonus_val
                if multiplier_text:
                    result['multiplier'] = multiplier_text

        # Capture generic multiplier info outside of boxes (e.g., Power Play / Megaplier)
        extra_mult = selected_row.select_one('li.c-result__multiplier')
        if extra_mult and 'multiplier' not in result:
            result['multiplier'] = extra_mult.get_text(strip=True)

        main_numbers = result.get('main_numbers')
        if not main_numbers:
            return None

        numbers = list(main_numbers)
        bonus_number = result.get('bonus_number')
        if bonus_number is not None:
            numbers.append(bonus_number)

        result['numbers'] = numbers
        return result
    
    def _fetch_from_lotteryusa(self, game: str, session: Optional[str] = None) -> Dict[str, Any]:
        """Fetch from LotteryUSA.com (primary source).
        If session is provided (e.g., "Midday"/"Evening"), try to pick the matching
        result on the page for Daily games.
        """
        print(f"   📡 Trying LotteryUSA.com...")
        
        if game not in self.sources['lotteryusa']['games']:
            return {'success': False, 'error': f'Game {game} not supported'}
        
        game_slug = self.sources['lotteryusa']['games'][game]
        base = self.sources['lotteryusa']['base_url']
        candidates: List[str] = []
        # Base game pages
        candidates.append(base.format(game_slug))
        candidates.append(base.format(game_slug) + 'results')
        # Session-specific pages (common pattern uses -midday / -evening)
        if session and game in ("Daily 3", "Daily 4"):
            sess_slug = session.lower()
            ses_game_slug = f"{game_slug}-{sess_slug}"
            candidates.append(base.format(ses_game_slug))
            candidates.append(base.format(ses_game_slug) + 'results')
        
        if not self.session:
            return {'success': False, 'error': 'HTTP session unavailable'}

        last_error = None
        for url in candidates:
            try:
                response = self.session.get(url, timeout=20, headers={
                    'User-Agent': self.session.headers.get('User-Agent', ''),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                })
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract numbers using structured parser first
                numbers, draw_date, meta = self._extract_lotteryusa_data(soup, game, session=session)
                if not numbers:
                    numbers, draw_date, meta = self._extract_generic_data(soup, game, session=session)
                if numbers:
                    print(f"   ✅ Found numbers: {numbers} ({url})")
                    result = {
                        'success': True,
                        'numbers': numbers,
                        'draw_date': draw_date,
                        'source': 'lotteryusa.com',
                        'url': url
                    }
                    if meta:
                        result.update(meta)
                    return result
                else:
                    print(f"   ⚠️  No valid numbers found on {url}")
            except Exception as e:
                last_error = str(e)
                print(f"   ❌ LotteryUSA error on {url}: {last_error}")
                continue
        return {'success': False, 'error': last_error or 'No valid numbers extracted'}
    
    def _extract_lotteryusa_data(self, soup: 'BeautifulSoup', game: str, session: Optional[str] = None) -> Tuple[List[int], Optional[str], Dict[str, Any]]:
        """Extract lottery data from LotteryUSA HTML.
        Attempts to honor 'session' (Midday/Evening) for Daily 3/4 when provided.
        """
        # Prefer the structured draw table when available
        structured = self._parse_lotteryusa_structured(soup, game, session=session)
        if structured:
            return structured.get('numbers', []), structured.get('draw_date'), structured

        numbers = []
        draw_date = None
        session_regex = None
        if session:
            try:
                import re as _re
                session_regex = _re.compile(rf"\b{session}\b", _re.IGNORECASE)
            except Exception:
                session_regex = None

        # Strategy 1: Look for result containers
        result_selectors = [
            '.result-numbers',
            '.winning-numbers', 
            '.numbers',
            '.draw-result',
            '[class*="number"]',
            '[class*="ball"]',
            '.ball-container'
        ]
        
        def _elements_with_optional_session(selector: str):
            elems = soup.select(selector)
            if session_regex:
                # try to keep only elements near a session label
                filtered = []
                for e in elems:
                    container_text = e.get_text(" ", strip=True)
                    if session_regex.search(container_text):
                        filtered.append(e)
                    else:
                        # also inspect a couple of ancestors for the session token
                        p = e
                        found = False
                        for _ in range(2):
                            p = p.parent
                            if not p:
                                break
                            if session_regex.search(p.get_text(" ", strip=True)):
                                found = True
                                break
                        if found:
                            filtered.append(e)
                if filtered:
                    return filtered
            return elems

        for selector in result_selectors:
            elements = _elements_with_optional_session(selector)
            for elem in elements:
                # Extract numbers from element text
                text = elem.get_text(strip=True)
                found_nums = re.findall(r'\b(\d+)\b', text)
                if found_nums:
                    try:
                        candidate_numbers = [int(n) for n in found_nums]
                        if self._validate_numbers(candidate_numbers, game):
                            numbers = candidate_numbers
                            break
                    except ValueError:
                        continue
            if numbers:
                break
        
        # Strategy 2: Look for individual ball elements
        if not numbers:
            ball_selectors = ['.ball', '.number-ball', '[class*="ball-"]']
            for selector in ball_selectors:
                balls = soup.select(selector)
                if balls:
                    try:
                        candidate_numbers = []
                        for ball in balls:
                            num_text = ball.get_text(strip=True)
                            if num_text.isdigit():
                                candidate_numbers.append(int(num_text))
                        
                        if self._validate_numbers(candidate_numbers, game):
                            numbers = candidate_numbers
                            break
                    except ValueError:
                        continue
        
        # Strategy 3: Pattern matching on full page text
        if not numbers:
            full_text = soup.get_text()
            patterns = self._get_game_patterns(game)
            
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            candidate_numbers = [int(d) for d in match if d.isdigit()]
                        else:
                            candidate_numbers = [int(d) for d in str(match) if d.isdigit()]
                        
                        if self._validate_numbers(candidate_numbers, game):
                            numbers = candidate_numbers
                            break
                    except (ValueError, TypeError):
                        continue
                
                if numbers:
                    break
        
        # Extract draw date (prefer a date element near a session match)
        date_selectors = ['.draw-date', '.date', '[class*="date"]']
        date_elem = None
        if session_regex:
            # try to find a date near a session label
            try:
                session_nodes = soup.find_all(string=session_regex)
                for node in session_nodes:
                    # look in parent for a date
                    parent = node.parent
                    if parent:
                        for sel in date_selectors:
                            cand = parent.select_one(sel)
                            if cand:
                                date_elem = cand
                                break
                    if date_elem:
                        break
            except Exception:
                pass
        if not date_elem:
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    break
        if date_elem:
            draw_date = date_elem.get_text(strip=True)
        
        return numbers, draw_date, {}
    
    def _get_game_patterns(self, game: str) -> List[str]:
        """Get regex patterns for specific games"""
        if game == "Daily 4":
            return [
                r'\\b(\\d)\\s+(\\d)\\s+(\\d)\\s+(\\d)\\b',  # Spaced digits
                r'\\b(\\d{4})\\b',  # Consecutive digits
                r'\\b(\\d)-(\\d)-(\\d)-(\\d)\\b',  # Hyphenated
            ]
        elif game == "Daily 3":
            return [
                r'\\b(\\d)\\s+(\\d)\\s+(\\d)\\b',
                r'\\b(\\d{3})\\b',
                r'\\b(\\d)-(\\d)-(\\d)\\b',
            ]
        elif game == "Fantasy 5":
            return [
                r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\b',
            ]
        elif game == "Lotto 47":
            return [
                r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\b',
            ]
        elif game == "Lucky for Life":
            return [
                r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+.*?(\\d{1,2})\\b',  # 5 main + lucky ball
                r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\b',  # Just main numbers
            ]
        else:
            return [r'\\b(\\d{1,2})\\b']
    
    def _validate_numbers(self, numbers: List[int], game: str) -> bool:
        """Validate numbers for specific games"""
        if not numbers:
            return False
        
        validations = {
            "Daily 4": lambda nums: len(nums) == 4 and all(0 <= n <= 9 for n in nums),
            "Daily 3": lambda nums: len(nums) == 3 and all(0 <= n <= 9 for n in nums),
            "Fantasy 5": lambda nums: len(nums) == 5 and all(1 <= n <= 39 for n in nums),
            "Lotto 47": lambda nums: len(nums) == 6 and all(1 <= n <= 47 for n in nums),
            "Lucky for Life": lambda nums: (len(nums) == 5 and all(1 <= n <= 48 for n in nums)) or 
                                          (len(nums) == 6 and all(1 <= n <= 48 for n in nums[:5]) and 1 <= nums[5] <= 18),
            "Powerball": lambda nums: len(nums) >= 5 and all(1 <= n <= 69 for n in nums[:5]),
            "Mega Millions": lambda nums: len(nums) >= 5 and all(1 <= n <= 70 for n in nums[:5])
        }
        
        validator = validations.get(game)
        if validator:
            return validator(numbers)
        
        return True  # Default validation
    
    def _try_backup_sources(self, game: str) -> Dict[str, Any]:
        """Try backup data sources"""
        print(f"   🔄 Trying backup sources...")
        
        # Could add more backup sources here
        backup_sites = [
            'https://www.lottery.net/michigan-{}/results',
            'https://lottery.com/results/michigan/{}',
        ]
        
        game_slug = game.lower().replace(' ', '-')
        
        if not self.session:
            return {'success': False, 'error': 'HTTP session unavailable'}

        for site_pattern in backup_sites:
            url = site_pattern.format(game_slug)
            try:
                print(f"   📡 Backup: {url}")
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Use a generic extractor for backup sites with different DOMs
                    numbers, draw_date, meta = self._extract_generic_data(soup, game)

                    if numbers:
                        print(f"   ✅ Backup success: {numbers}")
                        result = {
                            'success': True,
                            'numbers': numbers,
                            'draw_date': draw_date,
                            'source': url,
                            'url': url
                        }
                        if meta:
                            result.update(meta)
                        return result
            except Exception as e:
                print(f"   ❌ Backup failed: {str(e)}")
                continue
        
        return {'success': False, 'error': 'All backup sources failed'}

    def _extract_generic_data(self, soup: 'BeautifulSoup', game: str, session: Optional[str] = None) -> Tuple[List[int], Optional[str], Dict[str, Any]]:
        """Generic extraction strategy for varied sites (lottery.net / lottery.com etc.).
        Uses regex over text and lightweight heuristics. Honors session if provided.
        """
        try:
            text = soup.get_text("\n", strip=True)
        except Exception:
            text = soup.get_text(" ", strip=True)

        # If session specified, try to focus near that word
        if session:
            try:
                import re as _re
                sess_pat = _re.compile(rf"\b{session}\b", _re.IGNORECASE)
                m = sess_pat.search(text)
                if m:
                    # Narrow window around the session label
                    start = max(0, m.start() - 400)
                    end = min(len(text), m.end() + 400)
                    text = text[start:end]
            except Exception:
                pass

        # Extract a date-like token (many formats)
        draw_date = None
        try:
            import re as _re
            date_patterns = [
                r"\b\d{4}-\d{2}-\d{2}\b",
                r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,\s+[A-Z][a-z]{2}\s+\d{2},\s+\d{4}\b",
                r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b",
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
            ]
            for pat in date_patterns:
                dm = _re.search(pat, text)
                if dm:
                    draw_date = dm.group(0)
                    break
        except Exception:
            pass

        # Extract numbers based on game
        def _pick_first_valid(groups: List[List[int]]) -> List[int]:
            for g in groups:
                if self._validate_numbers(g, game):
                    return g
            return []

        import re
        candidates: List[List[int]] = []
        if game == "Daily 4":
            # Look for 4 single digits separated by non-digits
            for m in re.finditer(r"\b(\d)[^\d]+(\d)[^\d]+(\d)[^\d]+(\d)\b", text):
                grp = [int(m.group(i)) for i in range(1,5)]
                candidates.append(grp)
            # Comma-separated fallback
            for m in re.finditer(r"\b(\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\b", text):
                grp = [int(m.group(i)) for i in range(1,5)]
                candidates.append(grp)
        elif game == "Daily 3":
            for m in re.finditer(r"\b(\d)[^\d]+(\d)[^\d]+(\d)\b", text):
                grp = [int(m.group(i)) for i in range(1,4)]
                candidates.append(grp)
            for m in re.finditer(r"\b(\d)\s*,\s*(\d)\s*,\s*(\d)\b", text):
                grp = [int(m.group(i)) for i in range(1,4)]
                candidates.append(grp)
        elif game == "Lotto 47":
            for m in re.finditer(r"\b(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})\b", text):
                grp = [int(x) for x in m.groups()]
                candidates.append(grp)
        elif game == "Fantasy 5":
            for m in re.finditer(r"\b(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})\b", text):
                grp = [int(x) for x in m.groups()]
                candidates.append(grp)
        elif game == "Lucky for Life":
            # 5 + lucky ball
            for m in re.finditer(r"\b(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})(?:\D+)(\d{1,2})\b", text):
                grp = [int(x) for x in m.groups()]
                candidates.append(grp)
        else:
            # Generic: collect up to 6 numbers
            nums = [int(n) for n in re.findall(r"\b\d{1,2}\b", text)[:6]]
            if nums:
                candidates.append(nums)

        numbers = _pick_first_valid(candidates)
        meta: Dict[str, Any] = {}
        if numbers and game in ('Powerball', 'Mega Millions'):
            meta['main_numbers'] = numbers[:5]
            if len(numbers) > 5:
                meta['bonus_number'] = numbers[5]
        return numbers, draw_date, meta
        
    
    def fetch_all_games(self) -> Dict[str, Any]:
        """Get today's numbers for all supported games (from CSV or fetch if needed)"""
        print("🎰 Getting today's numbers for all games...")
        print("=" * 50)
        
        results = {}
        games = ["Daily 4", "Daily 3", "Lucky for Life", "Lotto 47", "Powerball", "Mega Millions"]
        
        for game in games:
            print(f"\\n🎯 {game}")
            print("-" * 30)
            result = self.get_today_numbers(game)
            results[game] = result
            
            if result['success']:
                source_icon = "📁" if result['source'] == 'csv' else "🌐"
                print(f"✅ Success ({source_icon}): {result['numbers']}")
                if result.get('draw_date'):
                    print(f"   📅 Date: {result['draw_date']}")
            else:
                print(f"❌ Failed: {result['error']}")
            
            # Small delay between requests if fetching
            if result.get('source') == 'web_fetch':
                time.sleep(1)
        
        return results
    
    def force_fetch_today(self) -> Dict[str, Any]:
        """Force fetch today's data for all games (ignores CSV cache)"""
        print("🔄 Force fetching today's data...")
        print("=" * 50)
        
        results = {}
        games = ["Daily 4", "Daily 3", "Lucky for Life", "Lotto 47", "Powerball", "Mega Millions"]
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        for game in games:
            print(f"\\n🎯 {game}")
            print("-" * 30)
            result = self._fetch_and_update_csv(game)
            results[game] = result
            
            if result['success']:
                print(f"✅ Success: {result['numbers']}")
                if result.get('draw_date'):
                    print(f"   📅 Date: {result['draw_date']}")
            else:
                print(f"❌ Failed: {result['error']}")
            
            # Small delay between requests
            time.sleep(1)
        
        # Update last fetch date
        self._update_last_fetch_date(today)
        return results

    def force_fetch_game(self, game: str) -> Dict[str, Any]:
        """Force fetch latest data for a single game and update its CSV, ignoring cache.
        Returns a dict with success flag, numbers, draw_date, and source info.
        """
        try:
            result = self._fetch_and_update_csv(game)
            # Update last fetch date for bookkeeping
            self._update_last_fetch_date(datetime.now().strftime("%Y-%m-%d"))
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_latest_numbers(self, game: str) -> Dict[str, Any]:
        """
        Simple method to get the latest available numbers for a game.
        Always returns CSV data if available, regardless of date.
        Perfect for integration with the main application.
        """
        # Use the corrected _get_from_csv method which handles special cases like Powerball
        result = self._get_from_csv(game)
        
        if result['success']:
            return {
                'success': True,
                'numbers': result['numbers'],
                'draw_date': result['draw_date'],
                'source': 'csv',
                'game': game
            }
        else:
            return {
                'success': False, 
                'error': result.get('error', 'Failed to read CSV'),
                'numbers': [],
                'draw_date': None
            }
    
    def get_recent_history(self, game: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent history for a game from CSV, sorted by draw date descending."""
        if game not in self.csv_files:
            return []
        
        # Handle both absolute and relative paths
        csv_file = self.csv_files[game]
        if csv_file.startswith('final_integrated_data/'):
            # Use the path as-is for the new integrated data
            csv_path = Path(csv_file)
        elif csv_file.startswith('past_games/'):
            # For Fantasy 5 files that already include the past_games path
            csv_path = Path(csv_file)
        else:
            # Use the old logic for backward compatibility
            csv_path = self.data_folder / csv_file
            
        # Ensure we have a unified per-game CSV with as much history as available
        try:
            csv_path = self.ensure_unified_game_csv(game, csv_path)
        except Exception:
            pass

        if not csv_path.exists():
            return []
        
        try:
            history: List[Dict[str, Any]] = []
            import re as _re

            def _clean_field(val: Optional[str]) -> str:
                if val is None:
                    return ""
                return str(val).strip().strip('"')

            def _safe_int(val: Optional[str]) -> Optional[int]:
                cleaned = _clean_field(val)
                if not cleaned:
                    return None
                try:
                    return int(cleaned)
                except ValueError:
                    try:
                        return int(float(cleaned))
                    except Exception:
                        return None

            def _parse_number_list(val: Optional[str]) -> List[int]:
                cleaned = _clean_field(val)
                if not cleaned:
                    return []
                try:
                    return [int(match) for match in _re.findall(r"\d+", cleaned)]
                except Exception:
                    return []

            def _augment_powerball_record(record: Dict[str, Any], source_row: Dict[str, str]) -> None:
                if game != 'Powerball':
                    return

                dp_candidates = [
                    source_row.get('Double Play'),
                    source_row.get('DoublePlay'),
                    source_row.get('Double Play Numbers')
                ]
                for candidate in dp_candidates:
                    dp_numbers = _parse_number_list(candidate)
                    if dp_numbers:
                        record['double_play_numbers'] = dp_numbers[:5]
                        break

                dp_bonus_candidates = [
                    source_row.get('Double Play Powerball'),
                    source_row.get('Double Play Power Ball'),
                    source_row.get('Double Play PB')
                ]
                for candidate in dp_bonus_candidates:
                    dp_bonus = _safe_int(candidate)
                    if dp_bonus is not None:
                        record['double_play_bonus'] = dp_bonus
                        break

                dp_multiplier_candidates = [
                    source_row.get('Double Play Multiplier'),
                    source_row.get('Double Play Power Play')
                ]
                for candidate in dp_multiplier_candidates:
                    cleaned = _clean_field(candidate)
                    if cleaned:
                        record['double_play_multiplier'] = cleaned
                        break

                multiplier_candidates = [
                    source_row.get('Power Play'),
                    source_row.get('Multiplier')
                ]
                for candidate in multiplier_candidates:
                    cleaned = _clean_field(candidate)
                    if cleaned:
                        record['multiplier'] = cleaned
                        break
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Handle Fantasy 5 games with special CSV format
                if game in ["Fantasy 5", "Fantasy 5 Double Play"]:
                    lines = f.readlines()
                    if len(lines) < 2:
                        return []
                    # Parse header
                    header_line = lines[0].strip()
                    import csv as csvmodule
                    header_parts = list(csvmodule.reader([header_line]))[0]
                    # Find column indices
                    date_idx = header_parts.index('Date') if 'Date' in header_parts else 0
                    regular_idx = header_parts.index('Regular') if 'Regular' in header_parts else 1
                    double_play_idx = header_parts.index('Double Play') if 'Double Play' in header_parts else 2
                    # Process all data lines
                    for line in lines[1:]:
                        data_parts = list(csvmodule.reader([line.strip()]))[0]
                        if len(data_parts) <= max(date_idx, regular_idx, double_play_idx):
                            continue
                        draw_date = data_parts[date_idx].strip().strip('"')
                        numbers_str = data_parts[regular_idx] if game == "Fantasy 5" else data_parts[double_play_idx]
                        try:
                            numbers = [int(x.strip()) for x in numbers_str.split(',')]
                            # Only accept valid 5-number rows
                            if len(numbers) == 5:
                                history.append({'numbers': numbers, 'draw_date': draw_date, 'game': game})
                        except ValueError:
                            continue
                else:
                    # Use normal DictReader for other games; read all rows
                    reader = csv.DictReader(f)
                    for row in reader:
                        numbers = self._parse_csv_numbers(row, game)
                        draw_date = row.get('Date', '').strip('"').strip()
                        record: Dict[str, Any] = {'game': game, 'draw_date': draw_date}
                        bonus_label = None
                        bonus_number = None
                        # Determine bonus per game from CSV columns directly
                        if game == 'Mega Millions':
                            bonus_label = 'Mega Ball'
                            bonus_number = _safe_int(row.get('Mega Ball'))
                        elif game == 'Powerball':
                            bonus_label = 'Powerball'
                            for key in ('Powerball', 'Power Ball', 'PB'):
                                bonus_number = _safe_int(row.get(key))
                                if bonus_number is not None:
                                    break
                        elif game == 'Lucky for Life':
                            bonus_label = 'Lucky Ball'
                            bonus_number = _safe_int(row.get('Lucky Ball'))

                        # Derive main vs bonus and compose 'numbers' with bonus at end when known
                        if bonus_label and bonus_number is not None:
                            main_numbers = numbers[:5] if len(numbers) >= 5 else numbers
                            numbers_with_bonus = list(main_numbers)
                            # ensure bonus appended if not already
                            if len(numbers_with_bonus) == 5:
                                numbers_with_bonus.append(bonus_number)
                            record.update({
                                'numbers': numbers_with_bonus,
                                'main_numbers': main_numbers,
                                'bonus_number': bonus_number,
                                'bonus_label': bonus_label,
                            })
                        else:
                            record['numbers'] = numbers
                        _augment_powerball_record(record, row)
                        history.append(record)

            # If too few rows found in the primary CSV, try known fallbacks to use all available data
            if len(history) < count:
                fallback_paths: List[Path] = []
                if game == 'Mega Millions':
                    fallback_paths = [
                        Path('enhanced_data') / 'Mega_Millions_Enhanced.csv',
                        Path('past_games') / 'Mega Millions numbers from LotteryUSA.csv'
                    ]
                elif game == 'Powerball':
                    fallback_paths = [
                        Path('enhanced_data') / 'Powerball_Enhanced.csv',
                        Path('past_games') / 'Powerball numbers from LotteryUSA.csv'
                    ]
                # Load the first fallback that provides more rows
                for fb in fallback_paths:
                    if not fb.exists():
                        continue
                    try:
                        fb_hist: List[Dict[str, Any]] = []
                        with open(fb, 'r', encoding='utf-8') as ff:
                            reader = csv.DictReader(ff)
                            for row in reader:
                                numbers = self._parse_csv_numbers(row, game)
                                draw_date = (row.get('Date') or '').strip('"').strip()
                                record: Dict[str, Any] = {'game': game, 'draw_date': draw_date}
                                bonus_label = None
                                bonus_number = None
                                if game == 'Mega Millions':
                                    bonus_label = 'Mega Ball'
                                    bonus_number = _safe_int(row.get('Mega Ball'))
                                elif game == 'Powerball':
                                    bonus_label = 'Powerball'
                                    for key in ('Powerball', 'Power Ball', 'PB'):
                                        bonus_number = _safe_int(row.get(key))
                                        if bonus_number is not None:
                                            break
                                elif game == 'Lucky for Life':
                                    bonus_label = 'Lucky Ball'
                                    bonus_number = _safe_int(row.get('Lucky Ball'))

                                if bonus_label and bonus_number is not None:
                                    main_numbers = numbers[:5] if len(numbers) >= 5 else numbers
                                    numbers_with_bonus = list(main_numbers)
                                    if len(numbers_with_bonus) == 5:
                                        numbers_with_bonus.append(bonus_number)
                                    record.update({
                                        'numbers': numbers_with_bonus,
                                        'main_numbers': main_numbers,
                                        'bonus_number': bonus_number,
                                        'bonus_label': bonus_label,
                                    })
                                else:
                                    record['numbers'] = numbers
                                _augment_powerball_record(record, row)
                                fb_hist.append(record)
                        if len(fb_hist) > len(history):
                            history = fb_hist
                            break
                    except Exception:
                        continue

            # Sort by date descending using multiple known formats
            def _parse_dt(s: str):
                s = (s or "").strip().strip('"')
                # Remove common non-date tokens like draw session or time strings
                try:
                    import re as _re
                    # normalize comma spacing (e.g., "Sunday,Nov 2, 2025" -> "Sunday, Nov 2, 2025")
                    s = _re.sub(r"\s*,\s*", ", ", s)
                    s = _re.sub(r",(?=\S)", ", ", s)
                    s = _re.sub(r"\b(Evening|Midday|Mid-day|Night|Morning)\b", "", s, flags=_re.IGNORECASE)
                    s = _re.sub(r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?", "", s, flags=_re.IGNORECASE)
                    s = s.replace("  ", " ").strip().strip(',')
                except Exception:
                    pass
                for fmt in (
                    "%Y-%m-%d",
                    "%m/%d/%Y", "%m/%d/%y",
                    "%A, %b %d, %Y", "%a, %b %d, %Y",
                    "%A, %B %d, %Y", "%a, %B %d, %Y",
                    "%B %d, %Y", "%b %d, %Y",
                    "%B %d %Y", "%b %d %Y"
                ):
                    try:
                        return datetime.strptime(s, fmt)
                    except Exception:
                        continue
                return None

            history.sort(key=lambda r: _parse_dt(str(r.get('draw_date', ''))) or datetime.min, reverse=True)
            return history[:count]
        except Exception:
            return []

    # --- New unified CSV management helpers ---
    def ensure_unified_game_csv(self, game: str, target_path: Path) -> Path:
        """Ensure a single, unified CSV exists per game with the richest available history.
        If the target_path has too few rows or doesn't exist, consolidate from fallback sources
        and write a normalized CSV schema suitable for parsing.
        """
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # If file exists and appears populated, keep it
        if target_path.exists():
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    total = sum(1 for _ in f)
                if total > 20:  # heuristic: already populated
                    return target_path
            except Exception:
                pass

        # Build from known richer sources
        sources: List[Path] = []
        if game == 'Mega Millions':
            sources = [
                Path('enhanced_data') / 'Mega_Millions_Enhanced.csv',
                Path('past_games') / 'Mega Millions numbers from LotteryUSA.csv'
            ]
            out_header = ['Date', 'Result', 'Mega Ball']
        elif game == 'Powerball':
            sources = [
                Path('enhanced_data') / 'Powerball_Enhanced.csv',
                Path('past_games') / 'Powerball numbers from LotteryUSA.csv'
            ]
            out_header = ['Date', 'Result', 'Powerball', 'Double Play', 'Double Play Powerball', 'Power Play', 'Double Play Multiplier']
        elif game == 'Lucky for Life':
            sources = [Path('final_integrated_data') / 'Lucky_for_Life_Complete.csv',
                       Path('past_games') / 'Lucky for Life numbers from LotteryUSA.csv']
            out_header = ['Date', 'Result', 'Lucky Ball']
        elif game in ['Fantasy 5', 'Fantasy 5 Double Play']:
            sources = [Path('past_games') / 'MI Fantasy 5 numbers from LotteryUSA.csv']
            # Keep both columns to let parser pick correct one
            out_header = ['Date', 'Regular', 'Double Play']
        else:
            # Daily 3 / Daily 4 / Lotto 47: consolidate multiple variant sources if present
            if game == 'Daily 3':
                sources = [
                    Path('final_integrated_data') / 'Daily_3_Evening_Complete.csv',
                    Path('final_integrated_data') / 'Daily_3_Midday_Complete.csv',
                    Path('past_games') / 'MI Daily 3 numbers from LotteryUSA.csv'
                ]
            elif game == 'Daily 4':
                sources = [
                    Path('final_integrated_data') / 'Daily_4_Evening_Complete.csv',
                    Path('final_integrated_data') / 'Daily_4_Midday_Complete.csv',
                    Path('past_games') / 'MI Daily 4 numbers from LotteryUSA.csv'
                ]
            elif game == 'Lotto 47':
                sources = [
                    Path('enhanced_data') / 'Lotto_47_Enhanced.csv',
                    Path('final_integrated_data') / 'Lotto_47_Complete.csv',
                    Path('past_games') / 'MI Lotto 47 numbers from LotteryUSA.csv'
                ]
            else:
                sources = [Path('final_integrated_data') / f'{game.replace(" ", "_")}_Complete.csv']
            out_header = ['Date', 'Result']

        def _clean_value(val: Optional[str]) -> str:
            if val is None:
                return ''
            return str(val).strip().strip('"')

        def _normalize_intish(val: Optional[str]) -> str:
            cleaned = _clean_value(val)
            if not cleaned:
                return ''
            try:
                return str(int(float(cleaned)))
            except Exception:
                return cleaned

        rows: List[Dict[str, str]] = []
        seen = set()

        def _row_result_from_dict(row: Dict[str, str]) -> str:
            # Prefer standardized 'Result', else common alternatives
            for key in ['Result', 'Regular', 'Winning Numbers', 'Main draw']:
                if key in row and row[key]:
                    return row[key].strip('"')
            # Fallback: join any numeric comma list we can find
            for key, val in row.items():
                if isinstance(val, str) and ',' in val and any(ch.isdigit() for ch in val):
                    return val.strip('"')
            return ''

        for src in sources:
            if not src.exists():
                continue
            try:
                import csv as _csv
                with open(src, 'r', encoding='utf-8') as f:
                    reader = _csv.DictReader(f)
                    for r in reader:
                        date_val = (r.get('Date') or '').strip('"').strip()
                        if not date_val:
                            continue
                        result_val = _row_result_from_dict(r)
                        if game == 'Mega Millions':
                            mb = r.get('Mega Ball') or r.get('Megaball') or r.get('MegaBall') or ''
                            try:
                                mb = str(int(float(mb))) if mb else ''
                            except Exception:
                                mb = str(mb).strip()
                            key = (date_val, result_val, 'MB:'+mb)
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({'Date': date_val, 'Result': result_val, 'Mega Ball': mb})
                        elif game == 'Powerball':
                            pb = _normalize_intish(r.get('Powerball') or r.get('Power Ball') or r.get('PB'))
                            double_play = _clean_value(r.get('Double Play') or r.get('DoublePlay') or r.get('Double Play Numbers'))
                            if double_play:
                                dp_matches = re.findall(r"\d+", double_play)
                                if dp_matches:
                                    double_play = ", ".join(dp_matches)
                            double_play_pb = _normalize_intish(
                                r.get('Double Play Powerball')
                                or r.get('Double Play Power Ball')
                                or r.get('Double Play PB')
                            )
                            power_play = _clean_value(r.get('Power Play') or r.get('Multiplier'))
                            double_play_multiplier = _clean_value(
                                r.get('Double Play Multiplier') or r.get('Double Play Power Play')
                            )
                            key = (
                                date_val,
                                result_val,
                                'PB:' + pb,
                                'DP:' + double_play,
                                'DPB:' + double_play_pb,
                                'PP:' + power_play,
                                'DPM:' + double_play_multiplier
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({
                                'Date': date_val,
                                'Result': result_val,
                                'Powerball': pb,
                                'Double Play': double_play,
                                'Double Play Powerball': double_play_pb,
                                'Power Play': power_play,
                                'Double Play Multiplier': double_play_multiplier
                            })
                        elif game in ['Fantasy 5', 'Fantasy 5 Double Play']:
                            reg = (r.get('Regular') or '').strip('"')
                            dbl = (r.get('Double Play') or '').strip('"')
                            key = (date_val, reg, dbl)
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({'Date': date_val, 'Regular': reg, 'Double Play': dbl})
                        elif game == 'Lucky for Life':
                            lb = r.get('Lucky Ball') or ''
                            try:
                                lb = str(int(float(lb))) if lb else ''
                            except Exception:
                                lb = str(lb).strip()
                            key = (date_val, result_val, 'LB:'+lb)
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({'Date': date_val, 'Result': result_val, 'Lucky Ball': lb})
                        else:
                            key = (date_val, result_val)
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({'Date': date_val, 'Result': result_val})
            except Exception:
                continue

        # If we still have nothing, just return target as-is (will fail later gracefully)
        if not rows:
            return target_path

        # Sort by parsed date desc
        def _parse_dt2(s: str):
            s = (s or '').strip().strip('"')
            try:
                import re as _re
                s = _re.sub(r"\s*,\s*", ", ", s)
                s = _re.sub(r",(?=\S)", ", ", s)
                s = _re.sub(r"\b(Evening|Midday|Mid-day|Night|Morning)\b", "", s, flags=_re.IGNORECASE)
                s = _re.sub(r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?", "", s, flags=_re.IGNORECASE)
                s = s.replace("  ", " ").strip().strip(',')
            except Exception:
                pass
            for fmt in (
                "%Y-%m-%d",
                "%m/%d/%Y", "%m/%d/%y",
                "%A, %b %d, %Y", "%a, %b %d, %Y",
                "%A, %B %d, %Y", "%a, %B %d, %Y",
                "%B %d, %Y", "%b %d, %Y",
                "%B %d %Y", "%b %d %Y"
            ):
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    continue
            return datetime.min

        rows.sort(key=lambda rr: _parse_dt2(rr.get('Date', '')), reverse=True)

        # Write unified CSV
        try:
            import csv as _csv
            with open(target_path, 'w', encoding='utf-8', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=out_header)
                writer.writeheader()
                for r in rows:
                    writer.writerow({h: r.get(h, '') for h in out_header})
        except Exception:
            pass

        return target_path

    def update_game_csv(self, game: str) -> bool:
        """Fetch the latest numbers and append to the game's unified CSV if it's a new draw."""
        latest = self.get_today_numbers(game)
        if not latest.get('success'):
            return False
        csv_file = self.csv_files.get(game)
        if not csv_file:
            return False
        # Resolve path similar to readers
        if csv_file.startswith('final_integrated_data/'):
            csv_path = Path(csv_file)
        elif csv_file.startswith('past_games/'):
            csv_path = Path(csv_file)
        else:
            csv_path = self.data_folder / csv_file

        csv_path = self.ensure_unified_game_csv(game, csv_path)

        # Build row for this game
        date_val = latest.get('draw_date') or ''
        try:
            # keep as string
            date_val = str(date_val)
        except Exception:
            pass
        nums: List[int] = latest.get('numbers') or []
        result_str = ', '.join(str(n) for n in (nums[:5] if len(nums) >= 5 else nums))
        bonus = nums[5] if len(nums) > 5 else ''

        # Read existing to check duplication
        try:
            import csv as _csv
            existing_keys = set()
            rows: List[Dict[str, str]] = []
            if csv_path.exists():
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = _csv.DictReader(f)
                    headers = reader.fieldnames or []
                    for r in reader:
                        rows.append(r)
                        key = (r.get('Date',''), r.get('Result','') or r.get('Regular','') or '')
                        existing_keys.add(key)
            else:
                headers = []

            # Compose new row depending on game
            if game == 'Mega Millions':
                new_row = {'Date': date_val, 'Result': result_str, 'Mega Ball': str(bonus) if bonus != '' else ''}
                desired_headers = ['Date', 'Result', 'Mega Ball']
            elif game == 'Powerball':
                new_row = {'Date': date_val, 'Result': result_str, 'Powerball': str(bonus) if bonus != '' else ''}
                desired_headers = ['Date', 'Result', 'Powerball']
            elif game in ['Fantasy 5', 'Fantasy 5 Double Play']:
                # Append to Regular for base game
                new_row = {'Date': date_val, 'Regular': result_str, 'Double Play': ''}
                desired_headers = ['Date', 'Regular', 'Double Play']
            elif game == 'Lucky for Life':
                new_row = {'Date': date_val, 'Result': result_str, 'Lucky Ball': str(bonus) if bonus != '' else ''}
                desired_headers = ['Date', 'Result', 'Lucky Ball']
            else:
                new_row = {'Date': date_val, 'Result': result_str}
                desired_headers = ['Date', 'Result']

            key = (new_row.get('Date',''), new_row.get('Result','') or new_row.get('Regular','') or '')
            if key in existing_keys:
                return True  # already present

            # Rewrite file with header union and appended row
            header_set = set(headers) | set(desired_headers)
            ordered = [h for h in desired_headers] + [h for h in headers or [] if h not in desired_headers]
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=ordered)
                writer.writeheader()
                for r in rows:
                    writer.writerow({h: r.get(h, '') for h in ordered})
                writer.writerow({h: new_row.get(h, '') for h in ordered})
            return True
        except Exception:
            return False

    def rebuild_all_unified_csvs(self) -> Dict[str, int]:
        """Consolidate and (re)write unified CSVs for all known games.
        Returns a mapping of game -> row count after rebuild."""
        result: Dict[str, int] = {}
        for game, csv_file in self.csv_files.items():
            # Resolve default path
            if csv_file.startswith('final_integrated_data/'):
                csv_path = Path(csv_file)
            elif csv_file.startswith('past_games/'):
                csv_path = Path(csv_file)
            else:
                csv_path = self.data_folder / csv_file

            try:
                # Force consolidation even if file exists
                csv_path = self.ensure_unified_game_csv(game, csv_path)
                # Count rows after write
                import csv as _csv
                count = 0
                if csv_path.exists():
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = _csv.reader(f)
                        # subtract header
                        count = max(sum(1 for _ in reader) - 1, 0)
                result[game] = count
            except Exception:
                result[game] = 0
        # Append a timestamped summary log with per-game row counts
        try:
            from datetime import datetime as _dt
            import csv as _csv
            log_dir = Path('logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'rebuild_summary.csv'
            write_header = not log_path.exists()
            timestamp = _dt.now().isoformat(timespec='seconds')
            total_rows = sum(result.values())
            with open(log_path, 'a', encoding='utf-8', newline='') as lf:
                writer = _csv.writer(lf)
                if write_header:
                    writer.writerow(['timestamp', 'game', 'rows', 'run_total_rows'])
                for game, rows in sorted(result.items()):
                    writer.writerow([timestamp, game, rows, total_rows])
                # Also write a TOTAL row for the run
                writer.writerow([timestamp, 'TOTAL', total_rows, total_rows])
        except Exception:
            # Logging failures shouldn't break the rebuild pathway
            pass
        return result

    # --- Maintenance utility: correct latest row ---
    def correct_latest_draw(self, game: str, main_numbers: List[int], bonus_number: Optional[int] = None, date_str: Optional[str] = None) -> bool:
        """Correct the latest draw row in a game's unified CSV.
        If date_str is provided, correct the first row with matching Date; otherwise, correct the first data row (latest).
        Returns True on success.
        """
        csv_file = self.csv_files.get(game)
        if not csv_file:
            return False
        # Resolve path similarly to readers
        if csv_file.startswith('final_integrated_data/'):
            csv_path = Path(csv_file)
        elif csv_file.startswith('past_games/'):
            csv_path = Path(csv_file)
        else:
            csv_path = self.data_folder / csv_file

        if not csv_path.exists():
            return False

        try:
            import csv as _csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = _csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)
            if not rows:
                return False

            # Locate target row index
            idx = 0
            if date_str:
                ds = str(date_str).strip().strip('"')
                for i, r in enumerate(rows):
                    rds = str(r.get('Date','')).strip().strip('"')
                    if rds == ds:
                        idx = i
                        break

            # Compose corrected fields depending on game
            corrected = dict(rows[idx])
            result_str = ', '.join(str(n) for n in (main_numbers[:5] if len(main_numbers) >= 5 else main_numbers))
            if game == 'Mega Millions':
                corrected['Result'] = result_str
                if 'Mega Ball' in headers:
                    corrected['Mega Ball'] = str(bonus_number or '')
            elif game == 'Powerball':
                corrected['Result'] = result_str
                if 'Powerball' in headers:
                    corrected['Powerball'] = str(bonus_number or '')
            elif game == 'Lucky for Life':
                corrected['Result'] = result_str
                if 'Lucky Ball' in headers:
                    corrected['Lucky Ball'] = str(bonus_number or '')
            elif game in ['Fantasy 5', 'Fantasy 5 Double Play']:
                corrected['Regular'] = result_str
            else:
                corrected['Result'] = result_str

            rows[idx] = corrected

            # Write back preserving header order
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in rows:
                    writer.writerow({h: r.get(h, '') for h in headers})
            return True
        except Exception:
            return False

    # --- Introspection helpers for UI diagnostics ---
    def get_unified_csv_path(self, game: str) -> Path:
        """Return the resolved CSV path for a game, ensuring directories are consistent."""
        if game not in self.csv_files:
            return self.data_folder / f"{game.replace(' ', '_')}.csv"
        csv_file = self.csv_files[game]
        if csv_file.startswith('final_integrated_data/') or csv_file.startswith('past_games/'):
            return Path(csv_file)
        return self.data_folder / csv_file

    def get_csv_info(self, game: str) -> Dict[str, Any]:
        """Return CSV diagnostics for a game: path, exists, row count, latest date, headers.
        Attempts to ensure a unified CSV exists before reporting.
        """
        info: Dict[str, Any] = {
            'game': game,
            'path': None,
            'exists': False,
            'rows': 0,
            'latest_date': None,
            'headers': []
        }
        try:
            path = self.get_unified_csv_path(game)
            # Try to ensure unified CSV if possible (no-op if already good)
            try:
                path = self.ensure_unified_game_csv(game, path)
            except Exception:
                pass
            info['path'] = str(path)
            if not path.exists():
                return info
            info['exists'] = True
            import csv as _csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = _csv.reader(f)
                all_rows = list(reader)
            if not all_rows:
                return info
            headers = all_rows[0]
            info['headers'] = headers
            data_rows = all_rows[1:]
            info['rows'] = len(data_rows)
            # latest is first row after header in our unified CSVs
            if data_rows and 'Date' in headers:
                try:
                    date_idx = headers.index('Date')
                    latest_date = str(data_rows[0][date_idx]).strip().strip('"')
                    info['latest_date'] = latest_date
                except Exception:
                    pass
        except Exception:
            # Keep best-effort info
            pass
        return info
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lottery_results_{timestamp}.json"
        
        # Add metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_games': len(results),
                'successful': sum(1 for r in results.values() if r['success']),
                'failed': sum(1 for r in results.values() if not r['success'])
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\\n💾 Results saved to: {filename}")
        return filename

def main():
    """Main function to test the CSV-based lottery scraper"""
    print("🚀 CSV-Based Michigan Lottery Scraper")
    print("=" * 60)
    
    scraper = ProductionMichiganScraper()
    
    # Test individual game
    print("\\n🧪 Testing Daily 4:")
    daily4_result = scraper.get_today_numbers("Daily 4")
    print(f"Result: {daily4_result}")
    
    print("\\n" + "="*60)
    
    # Test all games (CSV first, fetch if needed)
    all_results = scraper.fetch_all_games()
    
    # Option to force fetch fresh data
    print("\\n" + "="*60)
    choice = input("\\n🔄 Force fetch fresh data? (y/n): ").lower()
    if choice == 'y':
        fresh_results = scraper.force_fetch_today()
        print("\\n📊 FRESH DATA SUMMARY:")
        print("=" * 30)
        successful = 0
        for game, result in fresh_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"{game:15} {status}")
            if result['success']:
                successful += 1
                print(f"   Numbers: {result['numbers']}")
            else:
                print(f"   Error: {result['error']}")
        print(f"\\nFresh Fetch Success Rate: {successful}/{len(fresh_results)} games")
    
    # Save results
    filename = scraper.save_results(all_results)
    
    # Print summary
    print("\\n📊 CSV/CACHED DATA SUMMARY:")
    print("=" * 30)
    successful = 0
    csv_count = 0
    fetch_count = 0
    
    for game, result in all_results.items():
        status = "✅" if result['success'] else "❌"
        source_icon = ""
        if result['success']:
            if result['source'] == 'csv':
                source_icon = " 📁"
                csv_count += 1
            elif result['source'] == 'web_fetch':
                source_icon = " 🌐"
                fetch_count += 1
        
        print(f"{game:15} {status}{source_icon}")
        if result['success']:
            successful += 1
            print(f"   Numbers: {result['numbers']}")
            if result.get('draw_date'):
                print(f"   Date: {result['draw_date']}")
        else:
            print(f"   Error: {result['error']}")
    
    print(f"\\nSuccess Rate: {successful}/{len(all_results)} games")
    print(f"From CSV: {csv_count} | Fresh Fetch: {fetch_count}")
    
    if successful == 0:
        print("\\n💡 Troubleshooting:")
        print("- Check CSV files in past_games folder")
        print("- Check internet connection for fresh data")
        print("- Data sources may be temporarily unavailable")
    else:
        print(f"\\n🎉 Successfully retrieved {successful} games!")
        print("💡 CSV data is used when available, web fetch only when needed")

if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
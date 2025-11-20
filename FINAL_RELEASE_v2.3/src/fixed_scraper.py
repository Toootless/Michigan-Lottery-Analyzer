#!/usr/bin/env python3
"""
Production-Ready Michigan Lottery Scraper
CSV-based fetch system with daily data updates
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import json
import time
import csv
import os
from pathlib import Path

class ProductionMichiganScraper:
    """CSV-based lottery scraper with daily data updates"""
    
    def __init__(self, data_folder: str = "past_games"):
        self.data_folder = Path(data_folder)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Connection': 'keep-alive'
        })
        
        # CSV file mapping for each game (using complete integrated data)
        # Build paths relative to the data_folder
        final_data_dir = self.data_folder / "final_integrated_data"
        past_games_dir = self.data_folder / "past_games"
        
        self.csv_files = {
            'Daily 4 Evening': str(final_data_dir / 'Daily_4_Evening_Complete.csv'),
            'Daily 4 Midday': str(final_data_dir / 'Daily_4_Midday_Complete.csv'), 
            'Daily 4': str(final_data_dir / 'Daily_4_Evening_Complete.csv'),  # Default to evening
            'Daily 3 Evening': str(final_data_dir / 'Daily_3_Evening_Complete.csv'),
            'Daily 3 Midday': str(final_data_dir / 'Daily_3_Midday_Complete.csv'),
            'Daily 3': str(final_data_dir / 'Daily_3_Evening_Complete.csv'),  # Default to evening
            'Lucky for Life': str(final_data_dir / 'Lucky_for_Life_Complete.csv'),
            'Lotto 47': str(final_data_dir / 'Lotto_47_Complete.csv'),
            'Fantasy 5': str(past_games_dir / 'MI Fantasy 5 numbers from LotteryUSA.csv'),
            'Fantasy 5 Double Play': str(past_games_dir / 'MI Fantasy 5 numbers from LotteryUSA.csv'),
            'Powerball': str(final_data_dir / 'Powerball_Complete.csv'),
            'Mega Millions': str(final_data_dir / 'Mega_Millions_Complete.csv')
        }
        
        # LotteryUSA.com scraping endpoints (for new data)
        self.sources = {
            'lotteryusa': {
                'base_url': 'https://www.lotteryusa.com/michigan/{}/results',
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
            with open(csv_path, 'r', encoding='utf-8') as f:
                # For Powerball, handle duplicate column names by reading raw lines
                if game == "Powerball":
                    lines = f.readlines()
                    if len(lines) < 2:
                        return {'success': False, 'error': 'No data in CSV'}
                    
                    # Parse header and first data line manually
                    header_line = lines[0].strip()
                    data_line = lines[1].strip()
                    
                    # Split by comma, handling quoted values
                    import csv as csvmodule
                    header_parts = list(csvmodule.reader([header_line]))[0]
                    data_parts = list(csvmodule.reader([data_line]))[0]
                    
                    # Find positions
                    date_idx = header_parts.index('Date') if 'Date' in header_parts else 0
                    main_draw_idx = header_parts.index('Main draw') if 'Main draw' in header_parts else 1
                    powerball_idx = None
                    
                    # Find FIRST occurrence of Powerball
                    for i, col in enumerate(header_parts):
                        if col == 'Powerball':
                            powerball_idx = i
                            break
                            
                # Handle Fantasy 5 games with special CSV format
                elif game in ["Fantasy 5", "Fantasy 5 Double Play"]:
                    lines = f.readlines()
                    if len(lines) < 2:
                        return {'success': False, 'error': 'No data in CSV'}
                    
                    # Parse header and first data line manually for Fantasy 5
                    header_line = lines[0].strip()
                    data_line = lines[1].strip()
                    
                    # Split by comma, handling quoted values properly
                    import csv as csvmodule
                    header_parts = list(csvmodule.reader([header_line]))[0]
                    data_parts = list(csvmodule.reader([data_line]))[0]
                    
                    # Find positions
                    date_idx = header_parts.index('Date') if 'Date' in header_parts else 0
                    regular_idx = header_parts.index('Regular') if 'Regular' in header_parts else 1
                    double_play_idx = header_parts.index('Double Play') if 'Double Play' in header_parts else 2
                    
                    if len(data_parts) <= max(date_idx, regular_idx, double_play_idx):
                        return {'success': False, 'error': 'Could not parse Fantasy 5 CSV structure'}
                    
                    # Extract the appropriate column based on game type
                    draw_date = data_parts[date_idx]
                    if game == "Fantasy 5":
                        numbers_str = data_parts[regular_idx]
                    else:   # Fantasy 5 Double Play
                        numbers_str = data_parts[double_play_idx]
                    
                    # Parse numbers from string
                    try:
                        numbers = [int(x.strip()) for x in numbers_str.split(',')]
                    except ValueError:
                        return {'success': False, 'error': f'Could not parse numbers from: {numbers_str}'}
                    
                    return {
                        'success': True,
                        'numbers': numbers,
                        'draw_date': draw_date,
                        'source': 'csv'
                    }
                    
                    if main_draw_idx < len(data_parts) and powerball_idx is not None and powerball_idx < len(data_parts):
                        # Parse main numbers
                        main_numbers_str = data_parts[main_draw_idx].strip('"')
                        numbers = [int(n.strip()) for n in main_numbers_str.split(', ')]
                        
                        # Add Powerball
                        powerball_num = int(data_parts[powerball_idx])
                        numbers.append(powerball_num)
                        
                        draw_date = data_parts[date_idx].strip('"')
                        
                        return {
                            'success': True,
                            'numbers': numbers,
                            'draw_date': draw_date,
                            'source': 'csv',
                            'is_current': False  # We'll check this later if needed
                        }
                    else:
                        return {'success': False, 'error': 'Could not parse Powerball CSV structure'}
                else:
                    # Use normal DictReader for other games
                    f.seek(0)  # Reset file pointer
                    reader = csv.DictReader(f)
                    first_row = next(reader)  # Get the most recent entry (first data row)
                    
                    # Parse the data based on game type
                    numbers = self._parse_csv_numbers(first_row, game)
                    draw_date = first_row['Date'].strip('"')
                    
                    # Check if this is today's data
                    today = datetime.now()
                    today_formats = [
                        today.strftime("%A, %b %d, %Y"),     # Tuesday, Oct 29, 2025
                        today.strftime("%A, %b %-d, %Y") if os.name != 'nt' else today.strftime("%A, %b %#d, %Y"),  # Without leading zero
                        today.strftime("%a, %b %d, %Y"),     # Tue, Oct 29, 2025
                        today.strftime("%a, %b %-d, %Y") if os.name != 'nt' else today.strftime("%a, %b %#d, %Y")   # Short without leading zero
                    ]
                    
                    is_today = any(fmt in draw_date for fmt in today_formats)
                    
                    return {
                        'success': True,
                        'numbers': numbers,
                        'draw_date': draw_date,
                        'source': 'csv',
                        'is_current': is_today
                    }
                
        except Exception as e:
            return {'success': False, 'error': f'Error reading CSV: {str(e)}'}
    
    def _parse_csv_numbers(self, row: Dict[str, str], game: str) -> List[int]:
        """Parse numbers from CSV row based on game type"""
        # Handle Fantasy 5 games specially - they have Regular and Double Play columns
        if game == "Fantasy 5":
            if 'Regular' in row:
                result_str = row['Regular'].strip('"')
            else:
                result_str = ""
        elif game == "Fantasy 5 Double Play":
            if 'Double Play' in row:
                result_str = row['Double Play'].strip('"')
            else:
                result_str = ""
        else:
            # Find the correct column for main numbers
            result_str = ""
            if 'Result' in row:
                result_str = row['Result'].strip('"')
            elif 'Main draw' in row:
                result_str = row['Main draw'].strip('"')
            elif 'Winning Numbers' in row:
                result_str = row['Winning Numbers'].strip('"')
            else:
                # Try to find any column with comma-separated numbers
                for key, value in row.items():
                    if ',' in value and any(char.isdigit() for char in value):
                        result_str = value.strip('"')
                        break
        
        # Extract main numbers
        numbers = []
        for num_str in result_str.split(', '):
            try:
                numbers.append(int(num_str.strip()))
            except ValueError:
                continue
        
        # Add bonus number if applicable - handle special cases
        if game == "Lucky for Life" and 'Lucky Ball' in row:
            try:
                lucky_ball = int(row['Lucky Ball'])
                numbers.append(lucky_ball)
            except ValueError:
                pass
        elif game == "Powerball":
            # For Powerball CSV with duplicate column names, we need to read raw values
            # The issue is that DictReader overwrites duplicate keys
            # For now, let's manually extract from the known position
            if 'Main draw' in row:
                # We know from the CSV structure that the first Powerball number comes after Main draw
                # Let's use a different approach - parse the raw line if needed
                try:
                    # Try to get the Powerball value - but DictReader gives us the last occurrence
                    # For a proper fix, we should read the raw CSV line, but for now:
                    # The main draw is position 2, and first Powerball should be position 3
                    if hasattr(self, '_current_raw_line'):
                        parts = self._current_raw_line.split(',')
                        if len(parts) >= 3:
                            powerball_value = int(parts[2])
                            numbers.append(powerball_value)
                    else:
                        # Fallback to the DictReader value (which might be wrong due to duplicate columns)
                        powerball_value = int(row.get('Powerball', 0))
                        numbers.append(powerball_value)
                except (ValueError, IndexError):
                    pass
        elif game == "Mega Millions" and 'Mega Ball' in row:
            try:
                mega_ball = int(row['Mega Ball'])
                numbers.append(mega_ball)
            except ValueError:
                pass
        
        return numbers
    
    def _fetch_and_update_csv(self, game: str) -> Dict[str, Any]:
        """Fetch new data and update the CSV file"""
        print(f"   📡 Fetching new data for {game}...")
        
        # Try to fetch from LotteryUSA
        fetch_result = self._fetch_from_lotteryusa(game)
        if fetch_result['success']:
            # Update the CSV file with new data
            self._update_csv_file(game, fetch_result)
            fetch_result['source'] = 'web_fetch'
            return fetch_result
        
        return {'success': False, 'error': 'Failed to fetch new data'}
    
    def _update_csv_file(self, game: str, data: Dict[str, Any]):
        """Update CSV file with new lottery data"""
        if game not in self.csv_files:
            return
        
        csv_path = self.data_folder / self.csv_files[game]
        if not csv_path.exists():
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
            new_date = new_row['Date']
            date_exists = any(row['Date'] == new_date for row in existing_rows)
            
            if not date_exists:
                # Write updated data
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = existing_rows[0].keys() if existing_rows else new_row.keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(new_row)  # New data goes first
                    writer.writerows(existing_rows)
                
                print(f"   ✅ Updated {csv_path.name} with new data")
            else:
                print(f"   ℹ️  Data for {new_date} already exists in CSV")
                
        except Exception as e:
            print(f"   ❌ Error updating CSV: {str(e)}")
    
    def _create_csv_row(self, data: Dict[str, Any], game: str) -> Dict[str, str]:
        """Create a CSV row from fetched data"""
        numbers = data['numbers']
        draw_date = data.get('draw_date', datetime.now().strftime("%A, %b %d, %Y"))
        
        if game == "Lucky for Life" and len(numbers) >= 6:
            return {
                'Date': f'"{draw_date}"',
                'Result': f'"{", ".join(map(str, numbers[:5]))}"',
                'Lucky Ball': str(numbers[5]),
                'Top prize': '"$1,000 Per day for life"'
            }
        else:
            return {
                'Date': f'"{draw_date}"',
                'Result': f'"{", ".join(map(str, numbers))}"',
                'Top prize': '"$5,000 "'  # Default for Daily games
            }
    
    def _fetch_from_lotteryusa(self, game: str) -> Dict[str, Any]:
        """Fetch from LotteryUSA.com (primary source)"""
        print(f"   📡 Trying LotteryUSA.com...")
        
        if game not in self.sources['lotteryusa']['games']:
            return {'success': False, 'error': f'Game {game} not supported'}
        
        game_slug = self.sources['lotteryusa']['games'][game]
        url = self.sources['lotteryusa']['base_url'].format(game_slug)
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract numbers using multiple strategies
            numbers, draw_date = self._extract_lotteryusa_data(soup, game)
            
            if numbers:
                print(f"   ✅ Found numbers: {numbers}")
                return {
                    'success': True,
                    'numbers': numbers,
                    'draw_date': draw_date,
                    'source': 'lotteryusa.com',
                    'url': url
                }
            else:
                print(f"   ⚠️  No valid numbers found")
                return {'success': False, 'error': 'No valid numbers extracted'}
                
        except Exception as e:
            print(f"   ❌ LotteryUSA error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _extract_lotteryusa_data(self, soup: BeautifulSoup, game: str) -> Tuple[List[int], Optional[str]]:
        """Extract lottery data from LotteryUSA HTML"""
        numbers = []
        draw_date = None
        
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
        
        for selector in result_selectors:
            elements = soup.select(selector)
            for elem in elements:
                # Extract numbers from element text
                text = elem.get_text(strip=True)
                found_nums = re.findall(r'\\b(\\d+)\\b', text)
                
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
        
        # Extract draw date
        date_selectors = ['.draw-date', '.date', '[class*="date"]']
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                draw_date = date_elem.get_text(strip=True)
                break
        
        return numbers, draw_date
    
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
        
        for site_pattern in backup_sites:
            url = site_pattern.format(game_slug)
            try:
                print(f"   📡 Backup: {url}")
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    numbers, draw_date = self._extract_lotteryusa_data(soup, game)
                    
                    if numbers:
                        print(f"   ✅ Backup success: {numbers}")
                        return {
                            'success': True,
                            'numbers': numbers,
                            'draw_date': draw_date,
                            'source': url,
                            'url': url
                        }
            except Exception as e:
                print(f"   ❌ Backup failed: {str(e)}")
                continue
        
        return {'success': False, 'error': 'All backup sources failed'}
    
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
        """Get recent history for a game from CSV"""
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
            
        if not csv_path.exists():
            return []
        
        try:
            history = []
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
                    
                    # Process data lines
                    for i, line in enumerate(lines[1:]):
                        if i >= count:
                            break
                        
                        data_parts = list(csvmodule.reader([line.strip()]))[0]
                        if len(data_parts) <= max(date_idx, regular_idx, double_play_idx):
                            continue
                        
                        # Extract appropriate column based on game type
                        draw_date = data_parts[date_idx]
                        if game == "Fantasy 5":
                            numbers_str = data_parts[regular_idx]
                        else:  # Fantasy 5 Double Play
                            numbers_str = data_parts[double_play_idx]
                        
                        # Parse numbers
                        try:
                            numbers = [int(x.strip()) for x in numbers_str.split(',')]
                            history.append({
                                'numbers': numbers,
                                'draw_date': draw_date,
                                'game': game
                            })
                        except ValueError:
                            continue
                else:
                    # Use normal DictReader for other games
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= count:
                            break
                        
                        numbers = self._parse_csv_numbers(row, game)
                        draw_date = row['Date'].strip('"')
                        
                        history.append({
                            'numbers': numbers,
                            'draw_date': draw_date,
                            'game': game
                        })
            
            return history
            
        except Exception:
            return []
    
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
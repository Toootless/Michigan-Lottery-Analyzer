#!/usr/bin/env python3
"""
Complete Michigan Lottery Data System
Combines OCR historical data (2000-2021) with LotteryUSA.com current data (2021+)
Includes daily updates and local database management
"""

import sqlite3
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lottery_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteLotterySystem:
    """Complete lottery data management system"""
    
    def __init__(self, db_path: str = "complete_lottery_database.sqlite"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Game configurations
        self.games_config = {
            'daily-4': {
                'url': 'https://www.lotteryusa.com/michigan/daily-4',
                'validation': lambda nums: len(nums) == 4 and all(0 <= n <= 9 for n in nums),
                'draws_per_day': 2  # Midday and evening
            },
            'daily-3': {
                'url': 'https://www.lotteryusa.com/michigan/daily-3', 
                'validation': lambda nums: len(nums) == 3 and all(0 <= n <= 9 for n in nums),
                'draws_per_day': 2
            },
            'fantasy-5': {
                'url': 'https://www.lotteryusa.com/michigan/fantasy-5',
                'validation': lambda nums: len(nums) == 5 and all(1 <= n <= 39 for n in nums),
                'draws_per_day': 1
            }
        }
        
        self.init_database()
    
    def init_database(self):
        """Initialize comprehensive SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main lottery results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lottery_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game TEXT NOT NULL,
                draw_date DATE NOT NULL,
                draw_time TEXT,
                draw_type TEXT, -- 'midday', 'evening', 'daily'
                numbers TEXT NOT NULL, -- JSON array of numbers
                bonus_numbers TEXT, -- JSON array for bonus numbers
                jackpot_amount REAL,
                source TEXT NOT NULL, -- 'ocr', 'lotteryusa', 'manual'
                source_file TEXT, -- Original file/URL
                confidence REAL DEFAULT 1.0, -- OCR confidence score
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game, draw_date, draw_type, numbers)
            )
        ''')
        
        # Data sources tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL, -- 'ocr', 'web', 'api'
                base_url TEXT,
                date_range_start DATE,
                date_range_end DATE,
                last_updated TIMESTAMP,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        ''')
        
        # Daily update log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_date DATE NOT NULL,
                game TEXT NOT NULL,
                source TEXT NOT NULL,
                results_added INTEGER DEFAULT 0,
                results_updated INTEGER DEFAULT 0,
                status TEXT, -- 'success', 'partial', 'failed'
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert known data sources
        sources = [
            ('ocr_pdfs', 'ocr', None, '2000-01-01', '2021-12-31', None, 'active', '22 years of PDF OCR data'),
            ('lotteryusa', 'web', 'https://www.lotteryusa.com/michigan/', '2021-01-01', None, None, 'active', 'Current data from LotteryUSA.com')
        ]
        
        for source in sources:
            cursor.execute('''
                INSERT OR IGNORE INTO data_sources 
                (source_name, source_type, base_url, date_range_start, date_range_end, last_updated, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', source)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def scrape_current_results(self, game: str) -> List[Dict[str, Any]]:
        """Scrape current results from LotteryUSA.com"""
        if game not in self.games_config:
            logger.error(f"Game {game} not supported")
            return []
        
        config = self.games_config[game]
        url = config['url']
        
        try:
            logger.info(f"Scraping {game} from {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = self._extract_lottery_results(soup, game)
            
            logger.info(f"Extracted {len(results)} results for {game}")
            return results
            
        except Exception as e:
            logger.error(f"Error scraping {game}: {e}")
            return []
    
    def _extract_lottery_results(self, soup: BeautifulSoup, game: str) -> List[Dict[str, Any]]:
        """Extract lottery results from LotteryUSA.com HTML"""
        results = []
        
        # Find the main results table (first table with Date, Result columns)
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Check if this looks like a results table
            header_row = rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            header_text = [cell.get_text(strip=True).lower() for cell in header_cells]
            
            if 'date' in header_text and ('result' in header_text or 'numbers' in header_text):
                logger.info(f"Found results table for {game}")
                
                # Process data rows
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        result = self._parse_result_row(cells, game)
                        if result:
                            results.append(result)
                
                break  # Found the main results table
        
        return results
    
    def _parse_result_row(self, cells: List, game: str) -> Optional[Dict[str, Any]]:
        """Parse a single result row from LotteryUSA.com"""
        try:
            if len(cells) < 2:
                return None
            
            # Extract date
            date_text = cells[0].get_text(strip=True)
            draw_date = self._parse_date(date_text)
            if not draw_date:
                return None
            
            # Extract numbers
            numbers_text = cells[1].get_text(strip=True)
            numbers = self._extract_numbers_from_text(numbers_text)
            
            # Validate numbers for the game
            config = self.games_config[game]
            if not config['validation'](numbers):
                return None
            
            # Determine draw type (midday/evening) from date text
            draw_type = 'evening'  # Default
            if 'midday' in date_text.lower() or 'mid' in date_text.lower():
                draw_type = 'midday'
            
            # Extract jackpot amount if available
            jackpot_amount = None
            if len(cells) > 2:
                prize_text = cells[2].get_text(strip=True)
                jackpot_match = re.search(r'\\$(\\d{1,3},?\\d+)', prize_text)
                if jackpot_match:
                    jackpot_amount = float(jackpot_match.group(1).replace(',', ''))
            
            return {
                'game': game,
                'draw_date': draw_date,
                'draw_type': draw_type,
                'numbers': numbers,
                'jackpot_amount': jackpot_amount,
                'source': 'lotteryusa',
                'raw_date': date_text,
                'raw_numbers': numbers_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing result row: {e}")
            return None
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """Parse date from various formats to YYYY-MM-DD"""
        if not date_text:
            return None
        
        logger.debug(f"Parsing date: '{date_text}'")
        
        try:
            # Clean up date text but preserve basic punctuation
            date_text_clean = re.sub(r'[^a-zA-Z0-9\\s,/\\-]', ' ', date_text.strip())
            
            # Try different date patterns - fix escaping
            patterns = [
                # Tuesday,Oct 28, 2025 or Tuesday, Oct 28, 2025
                (r'(\\w+),?\\s*(\\w+)\\s+(\\d{1,2}),?\\s*(\\d{4})', 'weekday_month_day_year'),
                # Oct 28, 2025
                (r'(\\w+)\\s+(\\d{1,2}),\\s*(\\d{4})', 'month_day_year'),
                # 10/28/2025
                (r'(\\d{1,2})/(\\d{1,2})/(\\d{4})', 'mdy_slash'),
                # 2025-10-28
                (r'(\\d{4})-(\\d{2})-(\\d{2})', 'ymd_dash'),
            ]
            
            for pattern, pattern_type in patterns:
                match = re.search(pattern, date_text_clean)
                if match:
                    logger.debug(f"Matched pattern: {pattern_type}")
                    groups = match.groups()
                    
                    if pattern_type == 'weekday_month_day_year':
                        # Skip weekday, use month, day, year
                        _, month, day, year = groups
                        date_str = f"{month} {day} {year}"
                        try:
                            date_obj = datetime.strptime(date_str, '%b %d %Y')
                        except ValueError:
                            # Try full month name
                            date_obj = datetime.strptime(date_str, '%B %d %Y')
                    
                    elif pattern_type == 'month_day_year':
                        month, day, year = groups
                        date_str = f"{month} {day} {year}"
                        try:
                            date_obj = datetime.strptime(date_str, '%b %d %Y')
                        except ValueError:
                            date_obj = datetime.strptime(date_str, '%B %d %Y')
                    
                    elif pattern_type == 'mdy_slash':
                        month, day, year = groups
                        date_obj = datetime.strptime(f"{month}/{day}/{year}", '%m/%d/%Y')
                    
                    elif pattern_type == 'ymd_dash':
                        year, month, day = groups
                        date_obj = datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
                    
                    result = date_obj.strftime('%Y-%m-%d')
                    logger.debug(f"Parsed date: {result}")
                    return result
            
            logger.warning(f"Could not parse date: '{date_text}'")
            
        except Exception as e:
            logger.error(f"Error parsing date '{date_text}': {e}")
        
        return None
    
    def _extract_numbers_from_text(self, text: str) -> List[int]:
        """Extract lottery numbers from text"""
        if not text:
            return []
        
        # Extract all digit sequences
        number_matches = re.findall(r'\\d+', text)
        
        try:
            # Try to parse as individual digits first (for Daily 3/4)
            if len(text.replace(' ', '')) <= 6:  # Short string, likely individual digits
                digits = [int(d) for d in text if d.isdigit()]
                if digits:
                    return digits
            
            # Parse as separate numbers
            numbers = [int(n) for n in number_matches if n.isdigit()]
            return numbers
            
        except ValueError:
            return []
    
    def store_results(self, results: List[Dict[str, Any]], update_duplicates: bool = True):
        """Store results in database"""
        if not results:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        added = 0
        updated = 0
        
        for result in results:
            try:
                # Check if result already exists
                cursor.execute('''
                    SELECT id FROM lottery_results 
                    WHERE game = ? AND draw_date = ? AND draw_type = ? AND numbers = ?
                ''', (result['game'], result['draw_date'], result['draw_type'], json.dumps(result['numbers'])))
                
                existing = cursor.fetchone()
                
                if existing and update_duplicates:
                    # Update existing record
                    cursor.execute('''
                        UPDATE lottery_results SET
                            jackpot_amount = ?, source = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (result.get('jackpot_amount'), result['source'], existing[0]))
                    updated += 1
                    
                elif not existing:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO lottery_results 
                        (game, draw_date, draw_type, numbers, jackpot_amount, source, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result['game'],
                        result['draw_date'], 
                        result['draw_type'],
                        json.dumps(result['numbers']),
                        result.get('jackpot_amount'),
                        result['source'],
                        result.get('confidence', 1.0)
                    ))
                    added += 1
                    
            except Exception as e:
                logger.error(f"Error storing result: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored results: {added} added, {updated} updated")
        return {'added': added, 'updated': updated}
    
    def daily_update(self) -> Dict[str, Any]:
        """Perform daily update of all games"""
        logger.info("Starting daily lottery update")
        
        update_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'games': {},
            'total_added': 0,
            'total_updated': 0,
            'status': 'success'
        }
        
        for game in self.games_config.keys():
            try:
                logger.info(f"Updating {game}")
                
                # Scrape current results
                results = self.scrape_current_results(game)
                
                if results:
                    # Store in database
                    store_result = self.store_results(results)
                    
                    update_summary['games'][game] = {
                        'scraped': len(results),
                        'added': store_result['added'],
                        'updated': store_result['updated'],
                        'status': 'success'
                    }
                    
                    update_summary['total_added'] += store_result['added']
                    update_summary['total_updated'] += store_result['updated']
                    
                    # Log update
                    self._log_update(game, 'lotteryusa', store_result['added'], store_result['updated'], 'success')
                    
                else:
                    update_summary['games'][game] = {
                        'scraped': 0,
                        'status': 'no_data'
                    }
                    self._log_update(game, 'lotteryusa', 0, 0, 'no_data')
                
                # Respectful delay between games
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error updating {game}: {e}")
                update_summary['games'][game] = {
                    'status': 'error',
                    'error': str(e)
                }
                update_summary['status'] = 'partial'
                self._log_update(game, 'lotteryusa', 0, 0, 'failed', str(e))
        
        logger.info(f"Daily update completed: {update_summary['total_added']} added, {update_summary['total_updated']} updated")
        return update_summary
    
    def _log_update(self, game: str, source: str, added: int, updated: int, status: str, error: str = None):
        """Log update activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO update_log (update_date, game, source, results_added, results_updated, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime('%Y-%m-%d'), game, source, added, updated, status, error))
        
        conn.commit()
        conn.close()
    
    def get_data_coverage(self) -> Dict[str, Any]:
        """Get overview of data coverage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        coverage = {}
        
        for game in self.games_config.keys():
            cursor.execute('''
                SELECT 
                    MIN(draw_date) as earliest,
                    MAX(draw_date) as latest,
                    COUNT(*) as total_draws,
                    COUNT(DISTINCT draw_date) as unique_dates,
                    source
                FROM lottery_results 
                WHERE game = ?
                GROUP BY source
            ''', (game,))
            
            results = cursor.fetchall()
            
            coverage[game] = {
                'sources': {},
                'total_draws': 0,
                'date_range': {'earliest': None, 'latest': None}
            }
            
            for row in results:
                earliest, latest, total, unique_dates, source = row
                coverage[game]['sources'][source] = {
                    'earliest_date': earliest,
                    'latest_date': latest,
                    'total_draws': total,
                    'unique_dates': unique_dates
                }
                coverage[game]['total_draws'] += total
                
                # Update overall date range
                if not coverage[game]['date_range']['earliest'] or earliest < coverage[game]['date_range']['earliest']:
                    coverage[game]['date_range']['earliest'] = earliest
                if not coverage[game]['date_range']['latest'] or latest > coverage[game]['date_range']['latest']:
                    coverage[game]['date_range']['latest'] = latest
        
        conn.close()
        return coverage
    
    def export_data(self, game: str = None, start_date: str = None, end_date: str = None, format: str = 'json') -> str:
        """Export lottery data"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM lottery_results WHERE 1=1"
        params = []
        
        if game:
            query += " AND game = ?"
            params.append(game)
        
        if start_date:
            query += " AND draw_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND draw_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY draw_date DESC, game"
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        # Create export data
        export_data = []
        for row in rows:
            record = dict(zip(columns, row))
            # Parse JSON fields
            record['numbers'] = json.loads(record['numbers'])
            if record['bonus_numbers']:
                record['bonus_numbers'] = json.loads(record['bonus_numbers'])
            export_data.append(record)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lottery_export_{game or 'all'}_{timestamp}.{format}"
        
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'export_date': datetime.now().isoformat(),
                    'total_records': len(export_data),
                    'filters': {'game': game, 'start_date': start_date, 'end_date': end_date},
                    'data': export_data
                }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} records to {filename}")
        return filename

def main():
    """Main function to demonstrate the complete system"""
    print("🎯 Complete Michigan Lottery Data System")
    print("=" * 60)
    
    system = CompleteLotterySystem()
    
    # Perform daily update
    print("\\n🔄 Performing daily update...")
    update_result = system.daily_update()
    
    print(f"\\n📊 Update Summary:")
    print(f"Date: {update_result['date']}")
    print(f"Status: {update_result['status']}")
    print(f"Total Added: {update_result['total_added']}")
    print(f"Total Updated: {update_result['total_updated']}")
    
    for game, stats in update_result['games'].items():
        print(f"\\n{game}:")
        print(f"  Status: {stats['status']}")
        if 'scraped' in stats:
            print(f"  Scraped: {stats['scraped']}")
        if 'added' in stats:
            print(f"  Added: {stats['added']}")
        if 'updated' in stats:
            print(f"  Updated: {stats['updated']}")
        if 'error' in stats:
            print(f"  Error: {stats['error']}")
    
    # Show data coverage
    print("\\n📊 Data Coverage:")
    coverage = system.get_data_coverage()
    
    for game, info in coverage.items():
        print(f"\\n{game.upper()}:")
        print(f"  Total Draws: {info['total_draws']}")
        print(f"  Date Range: {info['date_range']['earliest']} to {info['date_range']['latest']}")
        for source, stats in info['sources'].items():
            print(f"  {source}: {stats['total_draws']} draws ({stats['earliest_date']} to {stats['latest_date']})")
    
    # Export sample data
    print("\\n💾 Exporting sample data...")
    export_file = system.export_data('daily-4')
    print(f"Sample export saved to: {export_file}")

if __name__ == "__main__":
    main()
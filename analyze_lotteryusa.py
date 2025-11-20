#!/usr/bin/env python3
"""
Analyze Current LotteryUSA.com Page Content
Extract what lottery data is actually available
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json

def analyze_current_page():
    """Analyze the current LotteryUSA.com page for Michigan Daily 4"""
    print("🔍 Analyzing Current LotteryUSA.com Content")
    print("=" * 50)
    
    url = "https://www.lotteryusa.com/michigan/daily-4"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        print(f"✅ Successfully loaded page")
        print(f"📊 Content length: {len(response.text)} characters")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract page title
        title = soup.find('title')
        if title:
            print(f"📝 Page title: {title.get_text()}")
        
        # Look for main content areas
        print(f"\n🔍 Analyzing page structure...")
        
        # Find all tables (lottery results often in tables)
        tables = soup.find_all('table')
        print(f"📊 Found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            print(f"\n📊 Table {i+1}:")
            rows = table.find_all('tr')
            print(f"   Rows: {len(rows)}")
            
            # Show first few rows
            for j, row in enumerate(rows[:5]):
                cells = row.find_all(['td', 'th'])
                if cells:
                    cell_text = [cell.get_text(strip=True) for cell in cells]
                    print(f"   Row {j+1}: {cell_text}")
        
        # Look for result containers
        result_selectors = [
            '.result', '.draw', '.winning', '[class*="number"]', 
            '[class*="ball"]', '.lottery-result', '.draw-result'
        ]
        
        print(f"\n🎯 Looking for lottery result containers...")
        for selector in result_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"Found {len(elements)} elements with selector '{selector}':")
                for j, elem in enumerate(elements[:3]):
                    text = elem.get_text(strip=True)[:100]  # First 100 chars
                    print(f"   {j+1}: {text}")
        
        # Extract all dates from the page
        print(f"\n📅 Looking for dates...")
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-01
            r'\d{1,2}/\d{1,2}/\d{4}',  # 1/1/2024
            r'\w+ \d{1,2}, \d{4}',  # January 1, 2024
            r'\d{1,2} \w+ \d{4}',  # 1 January 2024
        ]
        
        all_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, response.text)
            all_dates.extend(matches)
        
        if all_dates:
            print(f"📅 Found {len(all_dates)} date-like strings:")
            unique_dates = list(set(all_dates))[:10]  # Show first 10 unique
            for date in unique_dates:
                print(f"   {date}")
        
        # Extract all numbers (potential lottery numbers)
        print(f"\n🎲 Looking for lottery numbers...")
        all_numbers = re.findall(r'\b\d+\b', response.text)
        if all_numbers:
            # Filter for reasonable lottery numbers
            lottery_candidates = []
            for num_str in all_numbers:
                if num_str.isdigit():
                    num = int(num_str)
                    if 0 <= num <= 50:  # Reasonable range for lottery numbers
                        lottery_candidates.append(num)
            
            if lottery_candidates:
                print(f"🎲 Found {len(lottery_candidates)} potential lottery numbers:")
                # Show unique numbers up to 20
                unique_nums = list(dict.fromkeys(lottery_candidates))[:20]
                print(f"   Sample: {unique_nums}")
                
                # Look for 4-digit sequences (Daily 4 pattern)
                four_digit_sequences = []
                for i in range(len(lottery_candidates) - 3):
                    sequence = lottery_candidates[i:i+4]
                    if all(0 <= n <= 9 for n in sequence):
                        four_digit_sequences.append(sequence)
                
                if four_digit_sequences:
                    print(f"🎯 Found {len(four_digit_sequences)} potential Daily 4 sequences:")
                    for seq in four_digit_sequences[:5]:
                        print(f"   {seq}")
        
        # Look for JavaScript or dynamic content indicators
        print(f"\n⚡ Checking for dynamic content...")
        scripts = soup.find_all('script')
        print(f"📜 Found {len(scripts)} script tags")
        
        # Check if content might be loaded via AJAX
        ajax_indicators = ['ajax', 'fetch', 'xhr', 'api', 'json']
        dynamic_content = False
        
        for script in scripts:
            script_text = script.get_text().lower()
            for indicator in ajax_indicators:
                if indicator in script_text:
                    dynamic_content = True
                    print(f"   ⚡ Found '{indicator}' in JavaScript - content might be dynamic")
                    break
        
        if not dynamic_content:
            print("   📋 No obvious dynamic content indicators")
        
        # Save HTML for manual inspection
        with open('lotteryusa_daily4_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"\n💾 Saved full HTML to: lotteryusa_daily4_page.html")
        
        return {
            'success': True,
            'url': url,
            'content_length': len(response.text),
            'tables_found': len(tables),
            'dates_found': len(all_dates),
            'numbers_found': len(lottery_candidates) if 'lottery_candidates' in locals() else 0,
            'dynamic_content': dynamic_content
        }
    
    except Exception as e:
        print(f"❌ Error analyzing page: {e}")
        return {'success': False, 'error': str(e)}

def test_other_games():
    """Test other Michigan lottery games on LotteryUSA.com"""
    print(f"\n🎯 Testing Other Michigan Games")
    print("=" * 40)
    
    games = [
        'daily-3',
        'fantasy-5', 
        'lotto-47',
        'mega-millions',
        'powerball'
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    working_games = []
    
    for game in games:
        url = f"https://www.lotteryusa.com/michigan/{game}"
        try:
            print(f"📡 Testing: {game}")
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                content_length = len(response.text)
                print(f"   ✅ Working - Length: {content_length}")
                working_games.append(game)
                
                # Quick check for numbers
                soup = BeautifulSoup(response.text, 'html.parser')
                numbers = re.findall(r'\b\d+\b', response.text)
                lottery_nums = [int(n) for n in numbers if n.isdigit() and 0 <= int(n) <= 50]
                
                if lottery_nums:
                    print(f"   🎲 Sample numbers: {lottery_nums[:10]}")
            else:
                print(f"   ❌ Status: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Working games: {working_games}")
    return working_games

def main():
    """Main analysis function"""
    print("🎯 LotteryUSA.com Content Analysis")
    print("=" * 50)
    
    # Analyze Daily 4 page
    daily4_analysis = analyze_current_page()
    
    # Test other games
    working_games = test_other_games()
    
    # Summary and recommendations
    print(f"\n🎯 ANALYSIS SUMMARY")
    print("=" * 30)
    
    if daily4_analysis['success']:
        print(f"✅ Daily 4 page accessible")
        print(f"📊 Content: {daily4_analysis['content_length']} chars")
        print(f"📊 Tables: {daily4_analysis['tables_found']}")
        print(f"📅 Dates: {daily4_analysis['dates_found']}")
        print(f"🎲 Numbers: {daily4_analysis['numbers_found']}")
        print(f"⚡ Dynamic: {daily4_analysis['dynamic_content']}")
    
    print(f"\n🎮 Working games: {len(working_games)}")
    for game in working_games:
        print(f"   ✅ {game}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. LotteryUSA.com has current data but limited historical access")
    print("2. Content appears to be mostly static (good for scraping)")
    print("3. For 2021+ data: Use LotteryUSA.com with pagination scraping")
    print("4. For 2000-2021 data: Continue using your OCR system")
    print("5. Set up daily scraping of current results")
    
    print(f"\n🔧 IMPLEMENTATION STRATEGY:")
    print("- OCR System: Handle 2000-2021 (22 years)")
    print("- LotteryUSA scraping: Handle 2021+ (current)")
    print("- Daily updates: Scrape recent results daily")
    print("- Database: Combine both data sources")

if __name__ == "__main__":
    main()
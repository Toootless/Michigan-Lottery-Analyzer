#!/usr/bin/env python3
"""
Test script for lottery web scraping functionality - IMPROVED VERSION
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from typing import Dict, Any, Optional

def test_scraping_for_game(game_name: str) -> bool:
    """Test scraping functionality for a specific game"""
    print(f"\n🎯 Testing scraping for {game_name}...")
    
    # Updated Michigan Lottery URLs
    urls = {
        "Fantasy 5": "https://www.michiganlottery.com/games/draw-games/fantasy-5",
        "Daily 4": "https://www.michiganlottery.com/games/draw-games/daily-4",
        "Daily 3": "https://www.michiganlottery.com/games/draw-games/daily-3",
        "Powerball": "https://www.michiganlottery.com/games/draw-games/powerball",
        "Mega Millions": "https://www.michiganlottery.com/games/draw-games/mega-millions",
    }
    
    if game_name not in urls:
        print(f"❌ Game '{game_name}' not supported for testing")
        return False
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        print(f"📡 Fetching data from: {urls[game_name]}")
        response = requests.get(urls[game_name], headers=headers, timeout=15)
        response.raise_for_status()
        
        print(f"✅ Successfully fetched page (Status: {response.status_code})")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        page_text = soup.get_text()
        
        print(f"📄 Page length: {len(page_text)} characters")
        
        # Look for common lottery number patterns
        found_patterns = []
        
        if game_name in ["Daily 3", "Daily 4"]:
            digit_count = 3 if game_name == "Daily 3" else 4
            patterns = [
                fr'\b(\d{{{digit_count}}})\b',  # Consecutive digits
                fr'\b(\d)' + r'\s+(\d)' * (digit_count-1) + r'\b',  # Spaced digits
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    found_patterns.append(f"{pattern}: {matches[:3]}...")  # Show first 3 matches
        
        elif game_name == "Fantasy 5":
            pattern = r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b'
            matches = re.findall(pattern, page_text)
            if matches:
                found_patterns.append(f"5-number sequences: {matches[:3]}...")
        
        if found_patterns:
            print("🔍 Found potential number patterns:")
            for pattern in found_patterns:
                print(f"   {pattern}")
        else:
            print("⚠️ No obvious number patterns found")
        
        # Show a sample of the page content for debugging
        print("\n📝 Sample page content (first 300 characters):")
        print(page_text[:300] + "...")
        
        return True
        
    except requests.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_daily_4_scraping():
    """Legacy function - now uses improved test_scraping_for_game"""
    return test_scraping_for_game("Daily 4")

def main():
    """Test scraping for all supported games"""
    print("🎰 Michigan Lottery Scraping Test")
    print("=" * 40)
    
    # Test required libraries
    try:
        import requests
        import bs4
        print("✅ Required libraries (requests, beautifulsoup4) are installed")
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("💡 Install with: pip install requests beautifulsoup4")
        return
    
    games_to_test = ["Fantasy 5", "Daily 4", "Daily 3", "Powerball", "Mega Millions"]
    
    results = {}
    for game in games_to_test:
        results[game] = test_scraping_for_game(game)
    
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    for game, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{game:15} {status}")
    
    successful_tests = sum(results.values())
    print(f"\nOverall: {successful_tests}/{len(games_to_test)} tests passed")
    
    if successful_tests == 0:
        print("\n💡 Troubleshooting tips:")
        print("1. Check your internet connection")  
        print("2. The Michigan Lottery website might be down")
        print("3. Website structure may have changed")
        print("4. Try running the script at a different time")

def test_basic_scraping():
    """Legacy function - now uses main()"""
    return main()

def test_advanced_daily_4_scraping():
    """Advanced Daily 4 scraping with multiple detection methods"""
    print("\n🎯 Advanced Daily 4 Scraping Test")
    print("=" * 40)
    
    url = "https://www.michiganlottery.com/games/draw-games/daily-4"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        print(f"📡 Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        print(f"✅ Page fetched successfully (Status: {response.status_code})")
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("\n--- Looking for winning numbers ---")
        
        # Method 1: Look for common winning number selectors
        selectors = [
            '.winning-numbers',
            '.numbers', 
            '.draw-results',
            '[class*="number"]',
            '[class*="winning"]',
            '.ball',
            '.digit'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                print(f"Found elements with selector '{selector}':")
                for i, elem in enumerate(elements[:5]):  # Show first 5
                    text = elem.get_text(strip=True)
                    if text and len(text) <= 20:  # Reasonable length
                        print(f"  {i+1}: {text}")
        
        # Method 2: Look for 4-digit patterns in text
        all_text = soup.get_text()
        
        # Look for Daily 4 patterns
        patterns = [
            r'\b(\d{4})\b',  # 4 digit sequence like "5739"
            r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b',  # Spaced digits like "5 7 3 9"
            r'\b(\d)-(\d)-(\d)-(\d)\b',  # Hyphenated like "5-7-3-9"
        ]
        
        print("\n--- Pattern matching ---")
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                print(f"Pattern '{pattern}' found {len(matches)} matches:")
                for match in matches[:5]:  # Show first 5
                    if isinstance(match, tuple):
                        digits = [int(d) for d in match if d.isdigit()]
                        if len(digits) == 4 and all(0 <= d <= 9 for d in digits):
                            print(f"  Valid Daily 4: {digits}")
                    elif len(match) == 4 and match.isdigit():
                        digits = [int(d) for d in match]
                        if all(0 <= d <= 9 for d in digits):
                            print(f"  Valid Daily 4: {digits}")
        
        # Method 3: Save HTML for manual inspection
        with open('daily_4_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"\nSaved HTML to daily_4_page.html for manual inspection")
        
        # Print some sample text to see structure
        print(f"\nSample page text (first 500 chars):")
        print(all_text[:500])
        
        return True
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return False

if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    test_advanced_daily_4_scraping()
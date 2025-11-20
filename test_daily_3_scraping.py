#!/usr/bin/env python3
"""
Test script for Daily 3 Midday scraping specifically
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

def test_daily_3_scraping():
    """Test scraping Daily 3 numbers with debugging"""
    print("🧪 Testing Daily 3 Midday scraping...")
    
    url = "https://www.michiganlottery.com/draw-games/daily-3"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"📡 Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Save HTML for inspection
            with open('daily_3_debug.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("💾 Saved HTML to daily_3_debug.html")
            
            # Look for winning numbers
            print("\n🔍 Looking for winning numbers...")
            
            # Strategy 1: Common CSS selectors
            selectors = [
                '.winning-numbers',
                '.numbers', 
                '.draw-results',
                '[class*="number"]',
                '[class*="winning"]',
                '.ball',
                '.digit',
                '.recent-results',
                '.latest-draw'
            ]
            
            found_numbers = []
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    print(f"✅ Found elements with selector '{selector}':")
                    for i, elem in enumerate(elements[:3]):  # Show first 3
                        text = elem.get_text(strip=True)
                        if text and len(text) <= 30:
                            print(f"   {i+1}: '{text}'")
                            # Look for 3-digit patterns
                            if re.match(r'^\d{3}$', text):
                                found_numbers.append(text)
                                print(f"   🎯 Potential Daily 3 number: {text}")
            
            # Strategy 2: Text pattern matching
            print("\n🔎 Pattern matching in full text...")
            all_text = soup.get_text()
            
            # Look for Daily 3 patterns
            patterns = [
                r'\b(\d{3})\b',  # 3 digit sequence like "573"
                r'\b(\d)\s+(\d)\s+(\d)\b',  # Spaced digits like "5 7 3"
                r'\b(\d)-(\d)-(\d)\b',  # Hyphenated like "5-7-3"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    print(f"📋 Pattern '{pattern}' found {len(matches)} matches:")
                    for match in matches[:5]:  # Show first 5
                        if isinstance(match, tuple):
                            digits = [int(d) for d in match if d.isdigit()]
                            if len(digits) == 3 and all(0 <= d <= 9 for d in digits):
                                print(f"   ✅ Valid Daily 3: {digits}")
                                found_numbers.append(''.join(map(str, digits)))
                        elif len(match) == 3 and match.isdigit():
                            digits = [int(d) for d in match]
                            if all(0 <= d <= 9 for d in digits):
                                print(f"   ✅ Valid Daily 3: {digits}")
                                found_numbers.append(match)
            
            # Look for draw time indicators
            print("\n⏰ Looking for draw time indicators...")
            time_keywords = ['midday', 'noon', 'evening', 'night', '12:', '7:', 'pm', 'am']
            for keyword in time_keywords:
                if keyword.lower() in all_text.lower():
                    print(f"   🕐 Found time keyword: '{keyword}'")
            
            # Summary
            print(f"\n📊 Summary:")
            print(f"   Response successful: ✅")
            print(f"   HTML length: {len(response.text)} characters")
            print(f"   Potential numbers found: {len(set(found_numbers))}")
            if found_numbers:
                print(f"   Numbers: {list(set(found_numbers))}")
            
            # Show a sample of the page text
            print(f"\n📖 Sample page text (first 300 chars):")
            print(f"'{all_text[:300]}...'")
            
        else:
            print(f"❌ Failed to fetch page: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"💥 Error during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_daily_3_scraping()
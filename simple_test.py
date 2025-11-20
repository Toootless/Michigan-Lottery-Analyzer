#!/usr/bin/env python3
"""
Simple Daily 3 fetching test without BeautifulSoup dependency issues
"""

import requests
import re
from datetime import datetime

def simple_daily_3_test():
    """Simple test of Daily 3 fetching"""
    print("🧪 Testing Daily 3 Midday web access...")
    
    url = "https://www.michiganlottery.com/draw-games/daily-3"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"📡 Requesting: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📏 Content Length: {len(response.text)} characters")
        
        if response.status_code == 200:
            # Look for 3-digit patterns in the raw text
            text = response.text.lower()
            
            # Simple pattern search
            patterns = re.findall(r'\\b\\d{3}\\b', text)
            print(f"🔍 Found {len(patterns)} 3-digit patterns")
            
            # Show first few patterns that could be Daily 3 numbers
            valid_patterns = []
            for pattern in patterns[:20]:  # Check first 20 patterns
                if len(pattern) == 3 and all(c.isdigit() for c in pattern):
                    valid_patterns.append(pattern)
            
            print(f"✅ Valid 3-digit patterns: {valid_patterns[:10]}")  # Show first 10
            
            # Check for draw time indicators
            time_indicators = []
            for indicator in ['midday', 'noon', 'evening', 'night', 'am', 'pm']:
                if indicator in text:
                    time_indicators.append(indicator)
            
            print(f"⏰ Time indicators found: {time_indicators}")
            
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"🌐 Network Error: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = simple_daily_3_test()
    if success:
        print("\\n✅ Basic web access successful!")
        print("The scraping URLs are working, the issue might be in HTML parsing logic.")
    else:
        print("\\n❌ Web access failed.")
        print("This could be a network issue or website blocking.")
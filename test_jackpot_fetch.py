#!/usr/bin/env python3
"""
Test script to verify real jackpot fetching functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

def test_jackpot_fetch(game_name: str):
    """Test fetching jackpot for a specific game"""
    print(f"\n🎰 Testing jackpot fetch for {game_name}...")
    
    urls = {
        'Powerball': 'https://www.powerball.com/games/powerball',
        'Mega Millions': 'https://www.megamillions.com/',
        'Lotto 47': 'https://www.michiganlottery.com/draw-games/lotto-47',
        'Fantasy 5': 'https://www.michiganlottery.com/draw-games/fantasy-5',
    }
    
    if game_name not in urls:
        print(f"❌ No URL configured for {game_name}")
        return
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"📡 Fetching from: {urls[game_name]}")
        response = requests.get(urls[game_name], headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code} - Failed to fetch page")
            return
            
        print(f"✅ Page fetched successfully ({len(response.content)} bytes)")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Search for jackpot-related text
        jackpot_patterns = [
            r'\\$[\\d,]+\\s*million',
            r'\\$[\\d,]+,?[\\d,]*',
            r'jackpot.*\\$[\\d,]+',
            r'\\$[\\d,]+.*jackpot'
        ]
        
        found_amounts = []
        page_text = soup.get_text()
        
        for pattern in jackpot_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            found_amounts.extend(matches)
        
        # Remove duplicates and show results
        unique_amounts = list(set(found_amounts))
        
        if unique_amounts:
            print(f"🎯 Found potential jackpot amounts:")
            for amount in unique_amounts[:5]:  # Show first 5
                print(f"   • {amount}")
        else:
            print(f"⚠️  No jackpot amounts found in page text")
            
        # Look for specific elements that might contain jackpot info
        potential_elements = soup.find_all(text=re.compile(r'\\$[\\d,]+', re.IGNORECASE))
        if potential_elements:
            print(f"🔍 Found {len(potential_elements)} elements with dollar amounts")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Test jackpot fetching for all supported games"""
    print("🚀 Testing Real Jackpot Fetching System")
    print("=" * 50)
    
    games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
    
    for game in games:
        test_jackpot_fetch(game)
        print()
    
    print("✅ Testing complete!")

if __name__ == "__main__":
    main()
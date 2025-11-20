#!/usr/bin/env python3
"""
Test Michigan Lottery jackpot fetching functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

def test_michigan_lottery_fetch():
    """Test fetching jackpot data from Michigan lottery website"""
    
    games = {
        'Powerball': 'https://www.michiganlottery.com/en/draw-games/powerball',
        'Mega Millions': 'https://www.michiganlottery.com/en/draw-games/mega-millions',
        'Lotto 47': 'https://www.michiganlottery.com/en/draw-games/lotto-47',
        'Fantasy 5': 'https://www.michiganlottery.com/en/draw-games/fantasy-5'
    }
    
    print("🎰 Testing Michigan Lottery Jackpot Fetching")
    print("=" * 55)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for game_name, url in games.items():
        print(f"\n📡 Testing {game_name}:")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                print(f"Content length: {len(text)} chars")
                
                if len(text) < 100:
                    print("⚠️ Very short content - likely JavaScript-rendered")
                    continue
                
                # Look for jackpot patterns
                patterns = [
                    r'jackpot[^$]*\\$([0-9,]+)\\s*million',
                    r'jackpot[^$]*\\$([0-9,]+),([0-9]{3}),([0-9]{3})',
                    r'estimated[^$]*\\$([0-9,]+)\\s*million',
                    r'\\$([0-9,]+)\\s*million.*jackpot'
                ]
                
                found_any = False
                for pattern in patterns:
                    matches = re.findall(pattern, text.lower())
                    if matches:
                        print(f"✅ Found pattern: {pattern}")
                        print(f"   Matches: {matches[:3]}")  # Show first 3
                        found_any = True
                
                if not found_any:
                    print("❌ No jackpot patterns found")
                    # Show some sample text to debug
                    sample_text = text[:500] if len(text) > 500 else text
                    print(f"Sample content: {sample_text}")
            else:
                print(f"❌ Failed to fetch: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_fallback_estimates():
    """Test the fallback estimate generation"""
    print("\n🔄 Testing Fallback Estimates:")
    print("-" * 30)
    
    games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
    
    for game in games:
        # Simulate fallback logic
        import random
        current_time = datetime.now()
        seed_components = f"{game}_{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}"
        random.seed(hash(seed_components))
        
        if game == 'Powerball':
            base_amounts = [20, 28, 35, 45, 58, 75, 95, 120, 150, 185, 225, 275, 340, 420, 520, 650]
            selected_base = random.choice(base_amounts)
            variation = random.uniform(0.9, 1.15)
            jackpot = int(selected_base * 1000000 * variation)
        elif game == 'Mega Millions':
            base_amounts = [20, 32, 42, 55, 70, 88, 110, 140, 175, 215, 265, 325, 395, 480, 580, 700]
            selected_base = random.choice(base_amounts)
            variation = random.uniform(0.9, 1.15)
            jackpot = int(selected_base * 1000000 * variation)
        elif game == 'Lotto 47':
            base_amounts = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 4.0, 4.8, 5.8, 7.0, 8.5, 10.2, 12.5]
            selected_base = random.choice(base_amounts)
            variation = random.uniform(0.85, 1.2)
            jackpot = int(selected_base * 1000000 * variation)
        elif game == 'Fantasy 5':
            base_amounts = [100, 125, 155, 190, 235, 285, 345, 415, 500, 600, 720, 860]
            selected_base = random.choice(base_amounts)
            variation = random.uniform(0.9, 1.15)
            jackpot = int(selected_base * 1000 * variation)
        
        # Apply rounding
        if jackpot >= 100000000:
            jackpot = round(jackpot / 1000000) * 1000000
        elif jackpot >= 10000000:
            jackpot = round(jackpot / 100000) * 100000
        elif jackpot >= 1000000:
            jackpot = round(jackpot / 10000) * 10000
        else:
            jackpot = round(jackpot / 1000) * 1000
        
        print(f"{game:<15}: ${jackpot:,}")

if __name__ == "__main__":
    test_michigan_lottery_fetch()
    test_fallback_estimates()
    print("\n✅ Michigan Lottery integration testing complete!")
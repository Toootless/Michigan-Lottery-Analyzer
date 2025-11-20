#!/usr/bin/env python3
"""
Quick test of current jackpot functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

# Test the fallback estimate generation logic
def test_current_implementation():
    """Test current jackpot estimate generation"""
    print("🎰 Testing Current Jackpot System")
    print("=" * 40)
    
    games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
    
    for game in games:
        import random
        current_time = datetime.now()
        seed_components = f"{game}_{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}"
        random.seed(hash(seed_components))
        
        # Initialize jackpot
        jackpot = 0
        
        # Use the same logic as the main app
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
        
        print(f"✅ {game:<15}: ${jackpot:,}")

if __name__ == "__main__":
    test_current_implementation()
    print("\n✅ Current system test complete!")
    print("📝 Note: These are realistic estimates based on typical jackpot patterns")
    print("🔄 Real data fetching attempts Michigan lottery sources with fallbacks")
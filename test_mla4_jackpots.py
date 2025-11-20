"""
Test script to verify MLA4 jackpot fetching with PDF scraper
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the function we need
from MichiganLotteryAnalyzer import fetch_real_jackpot_amount

def test_jackpots():
    """Test jackpot fetching for all major games"""
    games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
    
    print("=" * 70)
    print("Testing MLA4 Jackpot Fetching with PDF Scraper Integration")
    print("=" * 70)
    print()
    
    for game in games:
        print(f"🎲 Fetching {game} jackpot...")
        try:
            result = fetch_real_jackpot_amount(game)
            
            if result:
                amount = result.get('amount', 0)
                formatted = result.get('formatted', 'N/A')
                source = result.get('source', 'unknown')
                
                # Visual indicator based on source
                if source == 'pdf_official':
                    icon = "📄 ✅"
                    status = "PDF SCRAPER (REAL-TIME)"
                elif source == 'official':
                    icon = "🌐"
                    status = "OFFICIAL SOURCE"
                else:
                    icon = "⚠️"
                    status = "ESTIMATED"
                
                print(f"{icon} {game}: {formatted}")
                print(f"   Source: {status}")
                print(f"   Raw amount: ${amount:,}")
            else:
                print(f"❌ {game}: No data available")
            
            print()
            
        except Exception as e:
            print(f"❌ Error fetching {game}: {e}")
            print()
    
    print("=" * 70)
    print("Test completed!")
    print("=" * 70)

if __name__ == "__main__":
    test_jackpots()

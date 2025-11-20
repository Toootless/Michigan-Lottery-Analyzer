#!/usr/bin/env python3
"""Test the updated PDF scraper with keyword-based extraction"""

from michigan_lottery_pdf_scraper import get_jackpots_from_michigan_lottery

print("Testing updated PDF scraper with 'Estimated Jackpot:' keywords...")
print("=" * 70)

try:
    jackpots = get_jackpots_from_michigan_lottery()
    
    if jackpots:
        print("\n✅ Successfully extracted jackpots:\n")
        for game, amount in jackpots.items():
            print(f"  {game:20s}: ${amount:,}")
        print("\n" + "=" * 70)
        print(f"Total games found: {len(jackpots)}")
    else:
        print("\n❌ No jackpots extracted")
        print("Check the logs above for details")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Test Billion amount parsing fix"""

import re

def test_billion_parsing():
    """Test if Billion amounts are parsed correctly"""
    
    test_cases = [
        ("Powerball\nEstimated Jackpot: $1.25 Billion", 1.25e9),
        ("Powerball\nEstimated Jackpot: $1 Billion", 1e9),
        ("Powerball\nEstimated Jackpot: $445 Million", 445e6),
        ("Powerball\nEstimated Jackpot: $445.5 Million", 445.5e6),
        ("Powerball\nEstimated Jackpot: $500 Thousand", 500e3),
        ("Mega Millions\nEstimated Jackpot: $2.4 Billion", 2.4e9),
    ]
    
    # Updated patterns
    amount_patterns = [
        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Billion|B)',  # $1.25 Billion
        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Million|M)',  # $445 Million
        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Thousand|K)',  # $500 Thousand
        r'\$\s*(\d{1,3}(?:,\d{3})*)',  # $10,500,000
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Billion|B)(?![A-Z])',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Million|M)(?![A-Z])',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Thousand|K)(?![A-Z])',
    ]
    
    print("Testing Billion Amount Parsing\n" + "="*50)
    
    for test_text, expected_amount in test_cases:
        print(f"\nTest: {test_text}")
        print(f"Expected: ${int(expected_amount):,}")
        
        found = False
        for pattern in amount_patterns:
            match = re.search(pattern, test_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Check if it's in billions, millions, or thousands
                match_text = match.group(0).lower()
                if 'billion' in match_text or 'b' in match_text:
                    amount *= 1_000_000_000
                elif 'million' in match_text or ('m' in match_text and 'billion' not in match_text):
                    amount *= 1_000_000
                elif 'thousand' in match_text or ('k' in match_text and 'billion' not in match_text and 'million' not in match_text):
                    amount *= 1_000
                
                print(f"Matched: {match.group(0)}")
                print(f"Parsed: ${int(amount):,}")
                
                if int(amount) == int(expected_amount):
                    print("✅ CORRECT")
                else:
                    print(f"❌ WRONG (expected ${int(expected_amount):,})")
                found = True
                break
        
        if not found:
            print("❌ NO MATCH FOUND")

if __name__ == "__main__":
    test_billion_parsing()

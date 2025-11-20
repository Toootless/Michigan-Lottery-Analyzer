#!/usr/bin/env python3
"""
Test OCR Powerball Processing
Test the enhanced OCR system for Powerball number extraction
"""

import sys
import os
sys.path.append('src')

from data_collection.pdf_ocr_reader import LotteryPDFOCRReader
import tempfile

def test_powerball_ocr():
    """Test Powerball OCR processing"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        reader = LotteryPDFOCRReader(temp_dir)
        
        print("🎯 OCR System Status:")
        print("✅ Powerball pattern exists:", 'Powerball' in reader.game_patterns)
        print("✅ Mega Millions pattern exists:", 'Mega Millions' in reader.game_patterns)
        print("✅ Lucky for Life pattern exists:", 'Lucky for Life' in reader.game_patterns)
        
        # Test different Powerball text formats
        test_cases = [
            {
                'name': 'Standard Format',
                'text': '''Powerball Drawing Results
Date: 10/28/2025
Numbers: 17 39 43 51 66 20
Jackpot: $50 Million'''
            },
            {
                'name': 'Comma Separated',
                'text': '''POWERBALL RESULTS
17, 39, 43, 51, 66, 20
Drawing Date: October 28, 2025'''
            },
            {
                'name': 'Dash Separated',
                'text': '''Power Ball Numbers
17-39-43-51-66-20
Date: 10/28/25'''
            },
            {
                'name': 'With PB Label',
                'text': '''Powerball
Main: 17 39 43 51 66  PB: 20
10/28/2025'''
            }
        ]
        
        print("\n🔍 Testing Powerball Number Extraction:")
        for test_case in test_cases:
            print(f"\n  📊 {test_case['name']}:")
            print(f"     Text: {repr(test_case['text'][:50])}...")
            
            results = reader.extract_numbers_from_ocr_text(test_case['text'], 'Powerball')
            print(f"     Found: {len(results)} results")
            
            for i, result in enumerate(results):
                main_nums = result.get('numbers', [])
                bonus_num = result.get('bonus_number')
                print(f"       Result {i+1}: Main={main_nums}, Powerball={bonus_num}")
                
                # Validate result format
                if len(main_nums) == 5 and bonus_num is not None:
                    valid_main = all(1 <= n <= 69 for n in main_nums)
                    valid_bonus = 1 <= bonus_num <= 26
                    if valid_main and valid_bonus:
                        print(f"       ✅ Valid Powerball numbers!")
                    else:
                        print(f"       ❌ Invalid ranges - Main: {valid_main}, Bonus: {valid_bonus}")
                else:
                    print(f"       ❌ Invalid format - Expected 5 main + 1 bonus")
        
        # Test Mega Millions
        print("\n🔍 Testing Mega Millions:")
        mega_text = '''Mega Millions Results
Numbers: 12 29 33 41 56 24
Date: 10/25/2025'''
        
        mega_results = reader.extract_numbers_from_ocr_text(mega_text, 'Mega Millions')
        print(f"  Found: {len(mega_results)} Mega Millions results")
        for result in mega_results:
            main_nums = result.get('numbers', [])
            bonus_num = result.get('bonus_number')
            print(f"    Main={main_nums}, Mega Ball={bonus_num}")
        
        # Test Lucky for Life
        print("\n🔍 Testing Lucky for Life:")
        lucky_text = '''Lucky for Life Drawing
Numbers: 14 15 21 24 45 8
Date: 10/24/2025'''
        
        lucky_results = reader.extract_numbers_from_ocr_text(lucky_text, 'Lucky for Life')
        print(f"  Found: {len(lucky_results)} Lucky for Life results")
        for result in lucky_results:
            main_nums = result.get('numbers', [])
            bonus_num = result.get('bonus_number')
            print(f"    Main={main_nums}, Lucky Ball={bonus_num}")
                
    finally:
        os.rmdir(temp_dir)

if __name__ == "__main__":
    test_powerball_ocr()
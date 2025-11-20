#!/usr/bin/env python3
"""
Quick test of current jackpot amounts
"""

import sys
import os
sys.path.append('.')

# Test current jackpot function directly
try:
    from MichiganLotteryAnalyzer import get_current_jackpot, LOTTERY_GAMES
    
    print('🔍 Current Jackpot Function Test')
    print('=' * 40)
    
    # Initialize a mock session state since we're not running in Streamlit
    class MockSessionState:
        def __init__(self):
            self.data = {}
        def get(self, key, default=None):
            return self.data.get(key, default)
        def __setitem__(self, key, value):
            self.data[key] = value
        def __getitem__(self, key):
            return self.data[key]
        def __contains__(self, key):
            return key in self.data
    
    # Create mock streamlit module
    class MockST:
        def __init__(self):
            self.session_state = MockSessionState()
    
    import MichiganLotteryAnalyzer
    MichiganLotteryAnalyzer.st = MockST()
    
    games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
    
    for game in games:
        try:
            jackpot_info = get_current_jackpot(game)
            if jackpot_info and not jackpot_info.get('error'):
                amount = jackpot_info.get('formatted', 'N/A')
                source = jackpot_info.get('source', 'unknown')
                print(f'{game:<15}: {amount} (Source: {source})')
            else:
                error_msg = jackpot_info.get('error', 'Unknown error') if jackpot_info else 'No data'
                print(f'{game:<15}: ERROR - {error_msg}')
        except Exception as e:
            print(f'{game:<15}: EXCEPTION - {e}')
    
    print()
    print('🎯 If amounts still look wrong, please specify what should be corrected.')
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
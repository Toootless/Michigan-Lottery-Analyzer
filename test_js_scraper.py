#!/usr/bin/env python3
"""
Test JavaScript Scraper Integration

This script tests the JavaScript-enabled scraping of the Michigan lottery website
to verify that real jackpot data can be extracted from JavaScript-rendered pages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_js_scraper_import():
    """Test if the JavaScript scraper can be imported"""
    print("🔧 Testing JavaScript Scraper Import")
    print("=" * 45)
    
    try:
        from michigan_lottery_js_scraper import (
            MichiganLotteryJSScraper, 
            fetch_michigan_jackpot_js, 
            fetch_all_michigan_jackpots_js,
            SELENIUM_AVAILABLE
        )
        
        if SELENIUM_AVAILABLE:
            print("✅ Selenium and WebDriver Manager available")
            print("✅ JavaScript scraper module imported successfully")
            return True
        else:
            print("❌ Selenium not available - install with: pip install selenium webdriver-manager")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_individual_game_scraping():
    """Test scraping individual games"""
    print("\n🎰 Testing Individual Game Scraping")
    print("=" * 45)
    
    try:
        from michigan_lottery_js_scraper import fetch_michigan_jackpot_js
        
        games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
        results = {}
        
        for game in games:
            print(f"\n📡 Testing {game}...")
            try:
                result = fetch_michigan_jackpot_js(game, timeout=20)
                if result:
                    amount = result.get('formatted', 'N/A')
                    source = result.get('source', 'unknown')
                    print(f"✅ {game}: {amount} (Source: {source})")
                    results[game] = result
                else:
                    print(f"❌ {game}: No jackpot data retrieved")
                    
            except Exception as e:
                print(f"❌ {game}: Error - {e}")
                
        return results
        
    except Exception as e:
        print(f"❌ Individual game testing failed: {e}")
        return {}

def test_bulk_scraping():
    """Test scraping all games at once"""
    print("\n🚀 Testing Bulk Game Scraping")
    print("=" * 40)
    
    try:
        from michigan_lottery_js_scraper import fetch_all_michigan_jackpots_js
        
        print("📡 Fetching all jackpots...")
        results = fetch_all_michigan_jackpots_js(timeout=30)
        
        if results:
            print("✅ Bulk scraping successful:")
            for game, info in results.items():
                amount = info.get('formatted', 'N/A')
                source = info.get('source', 'unknown')
                print(f"  • {game}: {amount} (Source: {source})")
        else:
            print("❌ Bulk scraping returned no results")
            
        return results
        
    except Exception as e:
        print(f"❌ Bulk scraping failed: {e}")
        return {}

def test_main_app_integration():
    """Test integration with the main application"""
    print("\n🔗 Testing Main App Integration")
    print("=" * 38)
    
    try:
        # Test import of main app components
        from MichiganLotteryAnalyzer import SELENIUM_AVAILABLE
        
        if SELENIUM_AVAILABLE:
            print("✅ JavaScript scraper available in main app")
        else:
            print("❌ JavaScript scraper not available in main app")
            
        return SELENIUM_AVAILABLE
        
    except ImportError as e:
        print(f"❌ Main app integration test failed: {e}")
        return False

def test_connection_only():
    """Test basic connection without full scraping"""
    print("\n🌐 Testing Website Connection")
    print("=" * 35)
    
    try:
        from michigan_lottery_js_scraper import MichiganLotteryJSScraper
        
        print("🔧 Initializing headless browser...")
        with MichiganLotteryJSScraper(headless=True, timeout=15) as scraper:
            print("✅ WebDriver initialized successfully")
            
            if scraper.test_connection():
                print("✅ Successfully connected to Michigan lottery website")
                return True
            else:
                print("❌ Failed to connect to Michigan lottery website")
                return False
                
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎰 Michigan Lottery JavaScript Scraper Test Suite")
    print("=" * 55)
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'import_test': False,
        'connection_test': False,
        'individual_scraping': {},
        'bulk_scraping': {},
        'integration_test': False
    }
    
    # Test 1: Import test
    results['import_test'] = test_js_scraper_import()
    
    if not results['import_test']:
        print("\n❌ Cannot proceed - JavaScript scraper not available")
        return results
    
    # Test 2: Connection test
    results['connection_test'] = test_connection_only()
    
    if not results['connection_test']:
        print("\n⚠️ Connection failed - proceeding with other tests")
    
    # Test 3: Individual game scraping
    results['individual_scraping'] = test_individual_game_scraping()
    
    # Test 4: Bulk scraping
    results['bulk_scraping'] = test_bulk_scraping()
    
    # Test 5: Integration test
    results['integration_test'] = test_main_app_integration()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"✅ Import Test: {'PASS' if results['import_test'] else 'FAIL'}")
    print(f"✅ Connection Test: {'PASS' if results['connection_test'] else 'FAIL'}")
    print(f"✅ Individual Scraping: {len(results['individual_scraping'])} games successful")
    print(f"✅ Bulk Scraping: {len(results['bulk_scraping'])} games successful")
    print(f"✅ Integration Test: {'PASS' if results['integration_test'] else 'FAIL'}")
    
    overall_success = (
        results['import_test'] and
        results['integration_test'] and
        (len(results['individual_scraping']) > 0 or len(results['bulk_scraping']) > 0)
    )
    
    print(f"\n🎯 Overall Status: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
    
    if overall_success:
        print("🚀 JavaScript scraper is ready for production use!")
    else:
        print("🔧 Some issues detected - check the logs above for details")
    
    return results

if __name__ == "__main__":
    main()
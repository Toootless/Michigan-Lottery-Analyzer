#!/usr/bin/env python3
"""
Enhanced Michigan Lottery JavaScript Scraper Debugger

This script provides detailed debugging information about what's actually
on the Michigan lottery pages to help identify the correct selectors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import re
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("❌ Selenium not available")
    sys.exit(1)

def debug_michigan_lottery_page(game_name: str):
    """Debug a specific Michigan lottery page to understand its structure"""
    
    game_urls = {
        'Powerball': 'https://www.michiganlottery.com/en/draw-games/powerball',
        'Mega Millions': 'https://www.michiganlottery.com/en/draw-games/mega-millions',
        'Lotto 47': 'https://www.michiganlottery.com/en/draw-games/lotto-47',
        'Fantasy 5': 'https://www.michiganlottery.com/en/draw-games/fantasy-5'
    }
    
    if game_name not in game_urls:
        print(f"❌ Game '{game_name}' not supported")
        return
    
    url = game_urls[game_name]
    
    print(f"\n🔍 Debugging {game_name} page")
    print(f"URL: {url}")
    print("=" * 60)
    
    # Setup Chrome with debugging options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        print("📡 Loading page...")
        driver.get(url)
        
        # Wait for page to load completely
        print("⏳ Waiting for page to load...")
        time.sleep(10)  # Give more time for JavaScript
        
        # Get page title
        title = driver.title
        print(f"📄 Page Title: {title}")
        
        # Get page source length
        page_source = driver.page_source
        print(f"📏 Page Source Length: {len(page_source)} characters")
        
        # Save full page source for inspection
        with open(f"debug_{game_name.lower().replace(' ', '_')}_page.html", 'w', encoding='utf-8') as f:
            f.write(page_source)
        print(f"💾 Full page source saved to debug_{game_name.lower().replace(' ', '_')}_page.html")
        
        # Look for any dollar signs or numbers
        dollar_matches = re.findall(r'\\$[0-9,]+', page_source)
        print(f"💰 Found {len(dollar_matches)} dollar amounts: {dollar_matches[:10]}")
        
        # Look for the word "jackpot" or "million"
        jackpot_contexts = []
        for match in re.finditer(r'.{0,50}jackpot.{0,50}', page_source.lower()):
            jackpot_contexts.append(match.group())
        print(f"🎰 Found {len(jackpot_contexts)} 'jackpot' mentions:")
        for context in jackpot_contexts[:5]:
            print(f"  • {context.strip()}")
        
        million_contexts = []
        for match in re.finditer(r'.{0,50}million.{0,50}', page_source.lower()):
            million_contexts.append(match.group())
        print(f"💵 Found {len(million_contexts)} 'million' mentions:")
        for context in million_contexts[:5]:
            print(f"  • {context.strip()}")
        
        # Look for common lottery-related classes and IDs
        common_selectors = [
            '[class*="jackpot"]',
            '[class*="prize"]', 
            '[class*="amount"]',
            '[class*="estimated"]',
            '[class*="current"]',
            '[id*="jackpot"]',
            '[id*="prize"]',
            '[id*="amount"]'
        ]
        
        print("\\n🎯 Testing common selectors:")
        for selector in common_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"  ✅ {selector}: Found {len(elements)} elements")
                    for i, elem in enumerate(elements[:3]):
                        text = elem.text.strip()[:100]
                        if text:
                            print(f"    [{i+1}] {text}")
                else:
                    print(f"  ❌ {selector}: No elements found")
            except Exception as e:
                print(f"  ⚠️ {selector}: Error - {e}")
        
        # Look for any visible text containing numbers
        print("\\n🔢 All visible text with numbers:")
        all_text = driver.find_element(By.TAG_NAME, "body").text
        number_lines = []
        for line in all_text.split('\\n'):
            if re.search(r'[0-9,]+', line.strip()) and line.strip():
                number_lines.append(line.strip()[:100])
        
        for line in number_lines[:10]:
            print(f"  • {line}")
        
        # Check for specific lottery terminology
        print("\\n🎲 Lottery terminology check:")
        lottery_terms = ['jackpot', 'prize', 'estimated', 'million', 'next draw', 'current', 'winning']
        for term in lottery_terms:
            count = page_source.lower().count(term)
            print(f"  • '{term}': {count} occurrences")
        
        # Check for JavaScript errors or loading issues
        print("\\n🐛 Checking for errors:")
        logs = driver.get_log('browser')
        if logs:
            print(f"  Found {len(logs)} browser log entries:")
            for log in logs[:5]:
                print(f"    {log['level']}: {log['message'][:100]}")
        else:
            print("  No browser errors detected")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
    
    finally:
        driver.quit()
        print("🔚 Browser closed")

def main():
    """Debug all Michigan lottery game pages"""
    print("🔍 Michigan Lottery Page Structure Debugger")
    print("=" * 50)
    print(f"🕒 Debug started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
    
    for game in games:
        try:
            debug_michigan_lottery_page(game)
        except KeyboardInterrupt:
            print(f"\\n⏹️ Debugging interrupted for {game}")
            break
        except Exception as e:
            print(f"❌ Error debugging {game}: {e}")
    
    print("\\n✅ Debugging complete!")
    print("📁 Check the generated HTML files for detailed page structure")

if __name__ == "__main__":
    main()
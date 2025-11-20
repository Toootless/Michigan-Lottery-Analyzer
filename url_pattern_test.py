#!/usr/bin/env python3
"""
Quick URL Pattern Test for LotteryUSA.com
Find the correct URL structure and test historical data availability
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time

def test_url_patterns():
    """Test various URL patterns to find working ones"""
    print("🔍 Testing LotteryUSA.com URL Patterns")
    print("=" * 50)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    # Base patterns to test
    base_patterns = [
        "https://www.lotteryusa.com/michigan/daily-4",
        "https://www.lotteryusa.com/michigan/daily-4/",
        "https://www.lotteryusa.com/michigan/daily-4/results",
        "https://www.lotteryusa.com/michigan/daily-4/results/",
        "https://www.lotteryusa.com/michigan/daily4",
        "https://www.lotteryusa.com/michigan/daily4/",
        "https://www.lotteryusa.com/michigan/mi-daily-4",
        "https://www.lotteryusa.com/michigan/mi-daily-4/",
        "https://www.lotteryusa.com/states/michigan/daily-4",
        "https://www.lotteryusa.com/states/michigan/daily4",
    ]
    
    working_urls = []
    
    for url in base_patterns:
        try:
            print(f"📡 Testing: {url}")
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                content_length = len(response.text)
                print(f"   ✅ SUCCESS - Status: {response.status_code}, Length: {content_length}")
                
                # Quick check for lottery content
                if any(keyword in response.text.lower() for keyword in ['lottery', 'winning', 'numbers', 'draw']):
                    print(f"   🎯 Contains lottery content!")
                    working_urls.append(url)
                    
                    # Look for historical navigation
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Check for years in links
                    year_links = []
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        if re.search(r'20\d{2}', href) or re.search(r'20\d{2}', text):
                            year_links.append({'href': href, 'text': text})
                    
                    if year_links:
                        print(f"   📅 Found {len(year_links)} year-related links")
                        for link in year_links[:5]:  # Show first 5
                            print(f"      - {link['text']}: {link['href']}")
                    
                    # Look for recent results
                    numbers_found = re.findall(r'\b\d+\b', response.text)
                    if numbers_found:
                        # Filter for reasonable lottery numbers
                        lottery_numbers = [int(n) for n in numbers_found if n.isdigit() and 0 <= int(n) <= 50]
                        if lottery_numbers:
                            print(f"   🎲 Sample numbers found: {lottery_numbers[:10]}")
                
                else:
                    print(f"   ⚠️  No lottery content detected")
            else:
                print(f"   ❌ Failed - Status: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        time.sleep(0.5)  # Be respectful
    
    print(f"\n📊 SUMMARY:")
    print(f"Working URLs found: {len(working_urls)}")
    for url in working_urls:
        print(f"  ✅ {url}")
    
    return working_urls

def test_historical_access(base_url: str):
    """Test historical data access for a working URL"""
    print(f"\n🕰️  Testing Historical Access for: {base_url}")
    print("=" * 60)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Test different historical URL patterns
    historical_patterns = [
        f"{base_url}/2021",
        f"{base_url}/2022", 
        f"{base_url}/2023",
        f"{base_url}?year=2021",
        f"{base_url}/archive/2021",
        f"{base_url}/results/2021",
        f"{base_url}/history/2021",
    ]
    
    successful_years = {}
    
    for pattern in historical_patterns:
        for year in range(2021, 2025):
            url = pattern.replace("2021", str(year))
            try:
                print(f"📡 Testing: {url}")
                response = session.get(url, timeout=10)
                
                if response.status_code == 200:
                    content_length = len(response.text)
                    print(f"   ✅ Found data for {year} - Length: {content_length}")
                    
                    # Quick analysis of content
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for date patterns from that year
                    year_dates = re.findall(rf'{year}-\d{{2}}-\d{{2}}', response.text)
                    if not year_dates:
                        year_dates = re.findall(rf'\d{{1,2}}/\d{{1,2}}/{year}', response.text)
                    
                    if year_dates:
                        print(f"   📅 Found {len(year_dates)} dates from {year}")
                        successful_years[year] = {
                            'url': url,
                            'dates_found': len(year_dates),
                            'sample_dates': year_dates[:3]
                        }
                        break  # Found working pattern for this year
                    else:
                        print(f"   ⚠️  No {year} dates found in content")
                else:
                    print(f"   ❌ Status: {response.status_code}")
            
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
            
            time.sleep(0.3)
    
    print(f"\n📊 Historical Data Summary:")
    if successful_years:
        for year, info in successful_years.items():
            print(f"  {year}: {info['dates_found']} dates found")
            print(f"    URL: {info['url']}")
            if info['sample_dates']:
                print(f"    Sample: {info['sample_dates']}")
    else:
        print("  ❌ No historical data found")
    
    return successful_years

def main():
    """Main investigation"""
    print("🎯 LotteryUSA.com Investigation for Michigan Lottery")
    print("=" * 70)
    
    # Step 1: Find working URLs
    working_urls = test_url_patterns()
    
    if not working_urls:
        print("\n❌ No working URLs found. LotteryUSA.com might not support Michigan lottery or has different URL structure.")
        print("\n💡 Alternative approaches:")
        print("1. Try different lottery result websites")
        print("2. Use official Michigan Lottery API (if available)")  
        print("3. Continue using OCR system for historical data")
        print("4. Web scrape official Michigan Lottery site with JavaScript rendering")
        return
    
    # Step 2: Test historical access for the first working URL
    best_url = working_urls[0]
    historical_data = test_historical_access(best_url)
    
    # Step 3: Provide recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 40)
    
    if historical_data:
        earliest_year = min(historical_data.keys())
        latest_year = max(historical_data.keys())
        
        print(f"✅ LotteryUSA.com has data from {earliest_year} to {latest_year}")
        print(f"📊 Total coverage: {latest_year - earliest_year + 1} years")
        
        if earliest_year <= 2021:
            print("✅ GOOD NEWS: Data goes back to 2021 or earlier!")
            print("🔧 Strategy:")
            print("  1. Use LotteryUSA.com for 2021+ data")
            print("  2. Use your OCR system for 2000-2020 data")
            print("  3. Set up daily updates from LotteryUSA.com")
        else:
            print(f"⚠️  Data only goes back to {earliest_year}")
            print("🔧 Strategy:")
            print("  1. Use OCR system for all historical data (2000-2021)")
            print(f"  2. Use LotteryUSA.com for {earliest_year}+ data")
            print("  3. Set up daily updates from LotteryUSA.com")
    else:
        print("❌ No historical data access found")
        print("🔧 Alternative strategy:")
        print("  1. Use OCR system for all historical data (2000-2021)")
        print("  2. Use current working URL for recent/daily updates")
        print("  3. Consider other data sources for historical data")

if __name__ == "__main__":
    main()
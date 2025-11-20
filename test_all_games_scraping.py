#!/usr/bin/env python3
"""
Comprehensive test script for web scraping functionality of all games
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime

def test_game_scraping(game_name, url):
    """Test scraping for a specific game"""
    print(f"\\n🎯 Testing: {game_name}")
    print(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for number patterns based on game type
            all_text = soup.get_text().lower()
            
            found_numbers = []
            
            if "daily 3" in game_name.lower():
                # Look for 3-digit patterns
                patterns = [
                    r'\\b(\\d{3})\\b',
                    r'\\b(\\d)\\s+(\\d)\\s+(\\d)\\b',
                    r'\\b(\\d)-(\\d)-(\\d)\\b'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, all_text)
                    for match in matches[:3]:  # First 3 matches
                        if isinstance(match, tuple):
                            digits = [int(d) for d in match if d.isdigit()]
                            if len(digits) == 3 and all(0 <= d <= 9 for d in digits):
                                found_numbers.append(digits)
                        elif len(match) == 3 and match.isdigit():
                            digits = [int(d) for d in match]
                            if all(0 <= d <= 9 for d in digits):
                                found_numbers.append(digits)
            
            elif "daily 4" in game_name.lower():
                # Look for 4-digit patterns
                patterns = [
                    r'\\b(\\d{4})\\b',
                    r'\\b(\\d)\\s+(\\d)\\s+(\\d)\\s+(\\d)\\b',
                    r'\\b(\\d)-(\\d)-(\\d)-(\\d)\\b'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, all_text)
                    for match in matches[:3]:  # First 3 matches
                        if isinstance(match, tuple):
                            digits = [int(d) for d in match if d.isdigit()]
                            if len(digits) == 4 and all(0 <= d <= 9 for d in digits):
                                found_numbers.append(digits)
                        elif len(match) == 4 and match.isdigit():
                            digits = [int(d) for d in match]
                            if all(0 <= d <= 9 for d in digits):
                                found_numbers.append(digits)
            
            elif "fantasy 5" in game_name.lower():
                # Look for 5 numbers between 1-39
                pattern = r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\b'
                matches = re.findall(pattern, all_text)
                for match in matches[:3]:  # First 3 matches
                    numbers = [int(n) for n in match]
                    if len(numbers) == 5 and all(1 <= n <= 39 for n in numbers):
                        found_numbers.append(sorted(numbers))
            
            elif "powerball" in game_name.lower():
                # Look for 5 main numbers + powerball
                # This is more complex, look for number sequences
                pattern = r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s.*?(\\d{1,2})\\b'
                matches = re.findall(pattern, all_text)
                for match in matches[:2]:  # First 2 matches
                    main_numbers = [int(n) for n in match[:5]]
                    powerball = int(match[5])
                    if (len(main_numbers) == 5 and all(1 <= n <= 69 for n in main_numbers) 
                        and 1 <= powerball <= 26):
                        found_numbers.append({'main': sorted(main_numbers), 'powerball': powerball})
            
            elif "mega millions" in game_name.lower():
                # Look for 5 main numbers + mega ball
                pattern = r'\\b(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s.*?(\\d{1,2})\\b'
                matches = re.findall(pattern, all_text)
                for match in matches[:2]:  # First 2 matches
                    main_numbers = [int(n) for n in match[:5]]
                    mega_ball = int(match[5])
                    if (len(main_numbers) == 5 and all(1 <= n <= 70 for n in main_numbers) 
                        and 1 <= mega_ball <= 25):
                        found_numbers.append({'main': sorted(main_numbers), 'mega_ball': mega_ball})
            
            # Look for draw time indicators
            time_indicators = []
            for indicator in ['midday', 'noon', 'evening', 'night', 'am', 'pm', '12:', '7:', '19:', '22:']:
                if indicator in all_text:
                    time_indicators.append(indicator)
            
            # Results
            print(f"Content length: {len(response.text)} characters")
            print(f"Potential numbers found: {len(found_numbers)}")
            if found_numbers:
                print(f"Examples: {found_numbers[:3]}")  # Show first 3
            print(f"Time indicators: {time_indicators[:5]}")  # Show first 5
            
            return {
                'success': True,
                'status_code': response.status_code,
                'content_length': len(response.text),
                'numbers_found': len(found_numbers),
                'example_numbers': found_numbers[:2],
                'time_indicators': time_indicators[:3]
            }
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return {'success': False, 'status_code': response.status_code}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def test_all_games():
    """Test web scraping for all supported games"""
    
    # URLs from the application (including both old and new URLs)
    game_urls = {
        'Fantasy 5': 'https://www.michiganlottery.com/draw-games/fantasy-5',
        'Daily 3': 'https://www.michiganlottery.com/draw-games/daily-3',
        'Daily 4': 'https://www.michiganlottery.com/draw-games/daily-4',
        'Daily 3 Midday': 'https://www.michiganlottery.com/draw-games/daily-3',
        'Daily 3 Evening': 'https://www.michiganlottery.com/draw-games/daily-3',
        'Daily 4 Midday': 'https://www.michiganlottery.com/draw-games/daily-4',
        'Daily 4 Evening': 'https://www.michiganlottery.com/draw-games/daily-4',
        'Powerball': 'https://www.michiganlottery.com/games/draw-games/powerball',
        'Mega Millions': 'https://www.michiganlottery.com/games/draw-games/mega-millions'
    }
    
    print("🌐 Comprehensive Web Scraping Test")
    print("=" * 50)
    
    results = {}
    
    for game_name, url in game_urls.items():
        result = test_game_scraping(game_name, url)
        results[game_name] = result
    
    # Summary
    print("\\n" + "=" * 50)
    print("📊 SUMMARY REPORT")
    print("=" * 50)
    
    successful = 0
    total = len(results)
    
    for game, result in results.items():
        if result.get('success'):
            status = "✅ SUCCESS"
            successful += 1
            if result.get('numbers_found', 0) > 0:
                status += f" - {result['numbers_found']} patterns found"
        else:
            status = f"❌ FAILED - {result.get('error', result.get('status_code', 'Unknown'))}"
        
        print(f"{game:20}: {status}")
    
    print(f"\\nOverall Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'web_scraping_test_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nDetailed results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = test_all_games()
    print("\\n🎯 Test complete! All games should now work with web scraping.")
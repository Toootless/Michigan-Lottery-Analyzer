#!/usr/bin/env python3
"""
Improved Michigan Lottery Scraper with multiple strategies
Handles JavaScript-heavy sites and anti-bot protection
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import urllib.parse

class ImprovedMichiganScraper:
    """Enhanced scraper with multiple strategies for lottery data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"'
        })
    
    def test_multiple_approaches(self, game: str = "Daily 4") -> Dict[str, Any]:
        """Test multiple scraping approaches for a game"""
        print(f"\n🔍 Testing multiple approaches for {game}")
        print("=" * 50)
        
        results = {
            'game': game,
            'approaches': {},
            'winning_numbers': None,
            'success': False
        }
        
        # Approach 1: Direct game page
        results['approaches']['direct_page'] = self._test_direct_page(game)
        
        # Approach 2: API endpoints
        results['approaches']['api_search'] = self._test_api_endpoints(game)
        
        # Approach 3: Search for AJAX/JSON endpoints
        results['approaches']['ajax_search'] = self._test_ajax_endpoints(game)
        
        # Approach 4: Alternative lottery sites
        results['approaches']['alternative_sites'] = self._test_alternative_sites(game)
        
        # Determine overall success
        for approach, result in results['approaches'].items():
            if result.get('success', False):
                results['success'] = True
                results['winning_numbers'] = result.get('numbers')
                break
        
        return results
    
    def _test_direct_page(self, game: str) -> Dict[str, Any]:
        """Test direct page scraping"""
        print(f"\n📄 Approach 1: Direct page scraping")
        
        urls = {
            "Daily 4": "https://www.michiganlottery.com/games/draw-games/daily-4",
            "Daily 3": "https://www.michiganlottery.com/games/draw-games/daily-3", 
            "Fantasy 5": "https://www.michiganlottery.com/games/draw-games/fantasy-5"
        }
        
        url = urls.get(game)
        if not url:
            return {'success': False, 'error': f'No URL for game {game}'}
        
        try:
            print(f"   📡 Fetching: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            print(f"   ✅ Status: {response.status_code}, Length: {len(response.text)}")
            
            # Check if content is minimal (potential JavaScript loading)
            if len(response.text) < 1000:
                print(f"   ⚠️  Content too short ({len(response.text)} chars) - likely JavaScript-rendered")
                
                # Save for inspection
                with open(f'{game.lower().replace(" ", "_")}_direct.html', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"   💾 Saved HTML to {game.lower().replace(' ', '_')}_direct.html")
                
                return {'success': False, 'error': 'Content too short, likely JavaScript-rendered'}
            
            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')
            numbers = self._extract_numbers_from_soup(soup, game)
            
            return {
                'success': len(numbers) > 0,
                'numbers': numbers,
                'content_length': len(response.text),
                'url': url
            }
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_api_endpoints(self, game: str) -> Dict[str, Any]:
        """Test potential API endpoints"""
        print(f"\n🔌 Approach 2: API endpoint search")
        
        # Common API patterns for lottery sites
        api_patterns = [
            "https://www.michiganlottery.com/api/games/{}/results",
            "https://www.michiganlottery.com/api/draws/{}/latest",
            "https://www.michiganlottery.com/api/lottery/{}/current",
            "https://api.michiganlottery.com/games/{}/results",
            "https://www.michiganlottery.com/services/results/{}",
        ]
        
        game_slugs = {
            "Daily 4": ["daily-4", "daily4", "d4"],
            "Daily 3": ["daily-3", "daily3", "d3"],
            "Fantasy 5": ["fantasy-5", "fantasy5", "f5"]
        }
        
        slugs = game_slugs.get(game, [game.lower().replace(" ", "-")])
        
        for pattern in api_patterns:
            for slug in slugs:
                url = pattern.format(slug)
                try:
                    print(f"   📡 Testing: {url}")
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        print(f"   ✅ Found API endpoint: {url}")
                        
                        # Try to parse as JSON
                        try:
                            data = response.json()
                            print(f"   📊 JSON Response: {json.dumps(data, indent=2)[:200]}...")
                            numbers = self._extract_numbers_from_json(data, game)
                            if numbers:
                                return {
                                    'success': True,
                                    'numbers': numbers,
                                    'url': url,
                                    'data': data
                                }
                        except:
                            print(f"   ⚠️  Not JSON, content: {response.text[:100]}...")
                    
                except Exception as e:
                    print(f"   ❌ {url}: {str(e)}")
                    continue
        
        return {'success': False, 'error': 'No working API endpoints found'}
    
    def _test_ajax_endpoints(self, game: str) -> Dict[str, Any]:
        """Look for AJAX endpoints by analyzing network requests"""
        print(f"\n⚡ Approach 3: AJAX endpoint discovery")
        
        # Common AJAX patterns
        ajax_patterns = [
            "https://www.michiganlottery.com/ajax/game-results",
            "https://www.michiganlottery.com/wp-json/lottery/v1/games",
            "https://www.michiganlottery.com/services/draw-results",
        ]
        
        for url in ajax_patterns:
            try:
                print(f"   📡 Testing AJAX: {url}")
                
                # Try different request methods and payloads
                methods = [
                    ('GET', {}),
                    ('POST', {'game': game}),
                    ('POST', {'gameId': game.lower().replace(' ', '')}),
                ]
                
                for method, data in methods:
                    try:
                        if method == 'GET':
                            response = self.session.get(url, timeout=10)
                        else:
                            response = self.session.post(url, json=data, timeout=10)
                        
                        if response.status_code == 200 and len(response.text) > 10:
                            print(f"   ✅ AJAX response: {method} {url}")
                            print(f"   📄 Content preview: {response.text[:200]}...")
                            
                            try:
                                json_data = response.json()
                                numbers = self._extract_numbers_from_json(json_data, game)
                                if numbers:
                                    return {
                                        'success': True,
                                        'numbers': numbers,
                                        'url': url,
                                        'method': method
                                    }
                            except:
                                # Try HTML parsing
                                soup = BeautifulSoup(response.text, 'html.parser')
                                numbers = self._extract_numbers_from_soup(soup, game)
                                if numbers:
                                    return {
                                        'success': True,
                                        'numbers': numbers,
                                        'url': url,
                                        'method': method
                                    }
                    except Exception as e:
                        print(f"   ❌ {method} {url}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"   ❌ AJAX test failed: {str(e)}")
                continue
        
        return {'success': False, 'error': 'No working AJAX endpoints found'}
    
    def _test_alternative_sites(self, game: str) -> Dict[str, Any]:
        """Test alternative lottery result sites"""
        print(f"\n🔄 Approach 4: Alternative lottery sites")
        
        # Alternative sites that might have Michigan lottery results
        alt_sites = [
            "https://www.lottery.net/michigan-{}/results",
            "https://www.lotteryusa.com/michigan/{}/",
            "https://lottery.com/results/michigan/{}",
        ]
        
        game_mappings = {
            "Daily 4": ["daily-4", "daily4"],
            "Daily 3": ["daily-3", "daily3"], 
            "Fantasy 5": ["fantasy-5", "fantasy5"]
        }
        
        slugs = game_mappings.get(game, [game.lower().replace(" ", "-")])
        
        for site_pattern in alt_sites:
            for slug in slugs:
                url = site_pattern.format(slug)
                try:
                    print(f"   📡 Testing alternative: {url}")
                    response = self.session.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        print(f"   ✅ Alternative site accessible: {url}")
                        soup = BeautifulSoup(response.text, 'html.parser')
                        numbers = self._extract_numbers_from_soup(soup, game)
                        
                        if numbers:
                            return {
                                'success': True,
                                'numbers': numbers,
                                'url': url,
                                'source': 'alternative_site'
                            }
                        else:
                            print(f"   ⚠️  No numbers found on {url}")
                
                except Exception as e:
                    print(f"   ❌ {url}: {str(e)}")
                    continue
        
        return {'success': False, 'error': 'No working alternative sites found'}
    
    def _extract_numbers_from_soup(self, soup: BeautifulSoup, game: str) -> List[int]:
        """Extract lottery numbers from HTML soup"""
        numbers = []
        
        # Try multiple selectors
        selectors = [
            '.winning-numbers',
            '.numbers',
            '.draw-results', 
            '[class*="number"]',
            '[class*="winning"]',
            '.ball',
            '.digit',
            '.result-number',
            '.lottery-number'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                # Extract numbers from text
                found_numbers = re.findall(r'\b\d+\b', text)
                if found_numbers:
                    try:
                        nums = [int(n) for n in found_numbers]
                        if self._validate_numbers(nums, game):
                            numbers.extend(nums)
                    except ValueError:
                        continue
        
        # Also try regex patterns on full text
        all_text = soup.get_text()
        patterns = self._get_number_patterns(game)
        
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        nums = [int(d) for d in match if d.isdigit()]
                    else:
                        nums = [int(d) for d in match]
                    
                    if self._validate_numbers(nums, game):
                        numbers.extend(nums)
                except (ValueError, TypeError):
                    continue
        
        return list(set(numbers))  # Remove duplicates
    
    def _extract_numbers_from_json(self, data: Dict[str, Any], game: str) -> List[int]:
        """Extract lottery numbers from JSON data"""
        numbers = []
        
        # Recursively search for number arrays in JSON
        def search_json(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if key.lower() in ['numbers', 'winning_numbers', 'results', 'draws']:
                        if isinstance(value, list):
                            try:
                                nums = [int(x) for x in value if str(x).isdigit()]
                                if self._validate_numbers(nums, game):
                                    numbers.extend(nums)
                            except:
                                pass
                    search_json(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_json(item, f"{path}[{i}]")
        
        search_json(data)
        return list(set(numbers))
    
    def _get_number_patterns(self, game: str) -> List[str]:
        """Get regex patterns for different games"""
        if game == "Daily 4":
            return [
                r'\b(\d{4})\b',
                r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b',
                r'\b(\d)-(\d)-(\d)-(\d)\b',
            ]
        elif game == "Daily 3":
            return [
                r'\b(\d{3})\b',
                r'\b(\d)\s+(\d)\s+(\d)\b',
                r'\b(\d)-(\d)-(\d)\b',
            ]
        elif game == "Fantasy 5":
            return [
                r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b',
            ]
        else:
            return [r'\b(\d{1,2})\b']
    
    def _validate_numbers(self, numbers: List[int], game: str) -> bool:
        """Validate numbers for specific games"""
        if not numbers:
            return False
        
        if game == "Daily 4":
            return len(numbers) == 4 and all(0 <= n <= 9 for n in numbers)
        elif game == "Daily 3":
            return len(numbers) == 3 and all(0 <= n <= 9 for n in numbers)
        elif game == "Fantasy 5":
            return len(numbers) == 5 and all(1 <= n <= 39 for n in numbers)
        
        return True  # Default validation

def main():
    """Test the improved scraper"""
    print("🚀 Improved Michigan Lottery Scraper Test")
    print("=" * 60)
    
    scraper = ImprovedMichiganScraper()
    
    games = ["Daily 4", "Daily 3", "Fantasy 5"]
    
    for game in games:
        print(f"\\n{'='*60}")
        print(f"🎯 TESTING: {game}")
        print(f"{'='*60}")
        
        results = scraper.test_multiple_approaches(game)
        
        print(f"\\n📊 RESULTS SUMMARY for {game}:")
        print(f"Overall Success: {'✅ YES' if results['success'] else '❌ NO'}")
        
        if results['winning_numbers']:
            print(f"Winning Numbers: {results['winning_numbers']}")
        
        print("\\nIndividual Approaches:")
        for approach, result in results['approaches'].items():
            status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
            print(f"  {approach:20} {status}")
            if 'error' in result:
                print(f"    Error: {result['error']}")
            if 'numbers' in result and result['numbers']:
                print(f"    Numbers: {result['numbers']}")

if __name__ == "__main__":
    main()
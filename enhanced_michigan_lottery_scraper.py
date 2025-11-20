#!/usr/bin/env python3
"""
Enhanced Michigan Lottery JavaScript Scraper with Advanced Detection

This module provides a more sophisticated approach to scraping the Michigan lottery
website which uses a complex single-page application (SPA) with dynamic content loading.
"""

import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.action_chains import ActionChains
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

class EnhancedMichiganLotteryJSScraper:
    """Advanced JavaScript-enabled scraper for Michigan lottery website"""
    
    def __init__(self, headless: bool = True, timeout: int = 60):
        """
        Initialize the enhanced JavaScript scraper
        
        Args:
            headless: Run browser in headless mode (no GUI)
            timeout: Maximum time to wait for elements
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available. Install with: pip install selenium webdriver-manager")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_browser()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_browser()
    
    def start_browser(self):
        """Initialize and start the Chrome WebDriver with enhanced options"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless=new")  # Use new headless mode
            
            # Essential Chrome options for complex SPA sites
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Allow JavaScript and XHR requests
            chrome_options.add_argument("--enable-javascript")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--disable-web-security")
            
            # User agent to appear as regular browser
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Disable logging to reduce noise
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
            
            # Setup Chrome service with WebDriverManager
            service = Service(ChromeDriverManager().install())
            
            # Initialize WebDriver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            
            # Set page load timeout
            self.driver.set_page_load_timeout(self.timeout)
            
            # Execute script to avoid detection and enable better JavaScript handling
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("Enhanced Chrome WebDriver initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def close_browser(self):
        """Close the browser and cleanup"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("WebDriver closed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None
    
    def wait_for_spa_load(self, expected_content: str = "jackpot", max_wait: int = 45):
        """
        Wait for Single Page Application to load dynamic content
        
        Args:
            expected_content: Text to wait for in page content
            max_wait: Maximum time to wait in seconds
        """
        start_time = time.time()
        last_content_length = 0
        stable_count = 0
        
        while time.time() - start_time < max_wait:
            try:
                # Check if page has stabilized (content stopped changing)
                current_content = self.driver.page_source
                current_length = len(current_content)
                
                if abs(current_length - last_content_length) < 1000:  # Less than 1KB change
                    stable_count += 1
                else:
                    stable_count = 0
                
                # If page is stable for 3 checks, it's likely loaded
                if stable_count >= 3:
                    # Check if expected content is present
                    if expected_content.lower() in current_content.lower():
                        self.logger.info(f"SPA loaded - found expected content: {expected_content}")
                        return True
                
                last_content_length = current_length
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.warning(f"Error during SPA load wait: {e}")
                time.sleep(1)
        
        self.logger.warning(f"SPA load timeout after {max_wait} seconds")
        return False
    
    def fetch_game_jackpot_enhanced(self, game_name: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced fetch for jackpot information with SPA handling
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Dictionary with jackpot information or None if failed
        """
        game_urls = {
            'Powerball': 'https://www.michiganlottery.com/en/draw-games/powerball',
            'Mega Millions': 'https://www.michiganlottery.com/en/draw-games/mega-millions',
            'Lotto 47': 'https://www.michiganlottery.com/en/draw-games/lotto-47',
            'Fantasy 5': 'https://www.michiganlottery.com/en/draw-games/fantasy-5'
        }
        
        if game_name not in game_urls:
            self.logger.warning(f"Game '{game_name}' not supported")
            return None
        
        if not self.driver:
            self.logger.error("WebDriver not initialized")
            return None
        
        try:
            url = game_urls[game_name]
            self.logger.info(f"Enhanced fetching {game_name} from {url}")
            
            # Load the page
            self.driver.get(url)
            
            # Wait for initial page load
            WebDriverWait(self.driver, 30).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Wait for SPA to load dynamic content
            self.wait_for_spa_load("jackpot", max_wait=45)
            
            # Try multiple strategies to find jackpot information
            strategies = [
                self._strategy_find_by_aria_labels,
                self._strategy_find_by_data_attributes,
                self._strategy_find_by_text_patterns,
                self._strategy_find_by_css_selectors,
                self._strategy_find_by_xpath,
                self._strategy_find_in_scripts
            ]
            
            for strategy in strategies:
                try:
                    result = strategy(game_name)
                    if result:
                        result.update({
                            'source': 'michigan_lottery_enhanced_js',
                            'last_updated': datetime.now(),
                            'url': url
                        })
                        self.logger.info(f"Successfully extracted jackpot using {strategy.__name__}: {result.get('formatted', 'N/A')}")
                        return result
                except Exception as e:
                    self.logger.debug(f"{strategy.__name__} failed: {e}")
                    continue
            
            # If all strategies fail, save page for debugging
            self._save_debug_page(game_name)
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error in enhanced fetch for {game_name}: {e}")
            return None
    
    def _strategy_find_by_aria_labels(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Strategy: Look for ARIA labels related to jackpots"""
        aria_selectors = [
            '[aria-label*="jackpot"]',
            '[aria-label*="prize"]',
            '[aria-label*="estimated"]',
            '[aria-labelledby*="jackpot"]'
        ]
        
        for selector in aria_selectors:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip()
                if text and '$' in text:
                    jackpot_info = self._parse_jackpot_text(text, game_name)
                    if jackpot_info:
                        return jackpot_info
        return None
    
    def _strategy_find_by_data_attributes(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Strategy: Look for data attributes that might contain jackpot info"""
        data_selectors = [
            '[data-jackpot]',
            '[data-prize]',
            '[data-amount]',
            '[data-estimated]',
            '[data-value]'
        ]
        
        for selector in data_selectors:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                # Check both text content and data attributes
                text = element.text.strip()
                data_value = element.get_attribute('data-value') or element.get_attribute('data-jackpot')
                
                for content in [text, data_value]:
                    if content and ('$' in content or content.isdigit()):
                        jackpot_info = self._parse_jackpot_text(content, game_name)
                        if jackpot_info:
                            return jackpot_info
        return None
    
    def _strategy_find_by_text_patterns(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Strategy: Look for specific text patterns that indicate jackpots"""
        # Get all visible text on the page
        body_text = self.driver.find_element(By.TAG_NAME, "body").text
        
        # Advanced regex patterns for jackpot detection
        patterns = [
            r'(?:jackpot|estimated|prize)[\s\S]*?\\$\s*([0-9,]+)\s*million',
            r'\\$\s*([0-9,]+)\s*million[\s\S]*?(?:jackpot|estimated|prize)',
            r'\\$\s*([0-9,]+),([0-9]{3}),([0-9]{3})[\s\S]*?(?:jackpot|prize)',
            r'(?:jackpot|prize)[\s\S]*?\\$\s*([0-9,]+),([0-9]{3}),([0-9]{3})',
            r'next\s+draw[\s\S]*?\\$\s*([0-9,]+)\s*million',
            r'estimated\s+at[\s\S]*?\\$\s*([0-9,]+)\s*million'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, body_text.lower())
            for match in matches:
                try:
                    if 'million' in pattern:
                        amount_str = match.group(1).replace(',', '')
                        amount = int(amount_str) * 1000000
                    else:
                        # Full dollar amount
                        amount_parts = [match.group(i+1) for i in range(match.lastindex)]
                        amount_str = ''.join(amount_parts).replace(',', '')
                        amount = int(amount_str)
                    
                    if self._validate_jackpot_amount(amount, game_name):
                        return {
                            'amount': amount,
                            'formatted': f"${amount:,}",
                            'game': game_name,
                            'has_jackpot': True
                        }
                except (ValueError, AttributeError, TypeError):
                    continue
        return None
    
    def _strategy_find_by_css_selectors(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Strategy: Try common CSS selectors for lottery sites"""
        selectors = [
            '.jackpot', '.prize', '.amount', '.estimated',
            '.jackpot-amount', '.prize-amount', '.current-jackpot',
            '.game-jackpot', '.draw-jackpot', '.lottery-jackpot',
            '#jackpot', '#prize', '#amount', '#estimated'
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and '$' in text:
                        jackpot_info = self._parse_jackpot_text(text, game_name)
                        if jackpot_info:
                            return jackpot_info
            except Exception:
                continue
        return None
    
    def _strategy_find_by_xpath(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Strategy: Use XPath to find jackpot information"""
        xpaths = [
            "//text()[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'jackpot')]/parent::*",
            "//text()[contains(., '$') and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'million')]/parent::*",
            "//*[contains(@class, 'jackpot') or contains(@class, 'prize')]",
            "//*[text()[contains(., '$')] and text()[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'jackpot')]]"
        ]
        
        for xpath in xpaths:
            try:
                elements = self.driver.find_elements(By.XPATH, xpath)
                for element in elements:
                    text = element.text.strip()
                    if text and '$' in text:
                        jackpot_info = self._parse_jackpot_text(text, game_name)
                        if jackpot_info:
                            return jackpot_info
            except Exception:
                continue
        return None
    
    def _strategy_find_in_scripts(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Strategy: Look for jackpot data in JavaScript variables or JSON"""
        try:
            scripts = self.driver.find_elements(By.TAG_NAME, "script")
            
            for script in scripts:
                script_content = script.get_attribute("innerHTML") or ""
                
                # Look for JSON data or JavaScript variables
                patterns = [
                    r'"jackpot":\s*"?\$?([0-9,]+)"?',
                    r'"prize":\s*"?\$?([0-9,]+)"?',
                    r'"amount":\s*"?\$?([0-9,]+)"?',
                    r'jackpot[\\s=:]*"?\$?([0-9,]+)"?',
                    r'prize[\\s=:]*"?\$?([0-9,]+)"?'
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, script_content.lower())
                    for match in matches:
                        try:
                            amount_str = match.group(1).replace(',', '')
                            amount = int(amount_str)
                            
                            # Could be in thousands, millions, or full amount
                            if amount < 10000:  # Likely in millions
                                amount *= 1000000
                            elif amount < 100000:  # Likely in thousands  
                                amount *= 1000
                            
                            if self._validate_jackpot_amount(amount, game_name):
                                return {
                                    'amount': amount,
                                    'formatted': f"${amount:,}",
                                    'game': game_name,
                                    'has_jackpot': True
                                }
                        except (ValueError, TypeError):
                            continue
                            
        except Exception:
            pass
        
        return None
    
    def _parse_jackpot_text(self, text: str, game_name: str) -> Optional[Dict[str, Any]]:
        """Parse jackpot amount from text string with enhanced patterns"""
        try:
            # Clean up text
            text = text.strip().replace('\\n', ' ').replace('\\t', ' ')
            
            # Multiple parsing strategies
            patterns = [
                # Million formats
                (r'\\$\\s*([0-9,]+)\\s*million', lambda m: int(m.group(1).replace(',', '')) * 1000000),
                (r'([0-9,]+)\\s*million', lambda m: int(m.group(1).replace(',', '')) * 1000000),
                
                # Full dollar amounts
                (r'\\$\\s*([0-9,]+),([0-9]{3}),([0-9]{3})', lambda m: int(''.join(m.groups()).replace(',', ''))),
                (r'\\$\\s*([0-9,]+)(?:,([0-9]{3}))*', lambda m: int(''.join(filter(None, m.groups())).replace(',', ''))),
                
                # Simple numbers (might need context clues)
                (r'\\$\\s*([0-9,]+)', lambda m: int(m.group(1).replace(',', '')))
            ]
            
            for pattern, converter in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    try:
                        amount = converter(match)
                        if self._validate_jackpot_amount(amount, game_name):
                            return {
                                'amount': amount,
                                'formatted': f"${amount:,}",
                                'game': game_name,
                                'has_jackpot': True
                            }
                    except (ValueError, TypeError):
                        continue
            
            return None
            
        except Exception:
            return None
    
    def _validate_jackpot_amount(self, amount: int, game_name: str) -> bool:
        """Enhanced validation for jackpot amounts"""
        minimums = {
            'Powerball': 20000000,      # $20M minimum
            'Mega Millions': 20000000,  # $20M minimum  
            'Lotto 47': 1000000,        # $1M minimum
            'Fantasy 5': 100000         # $100K minimum
        }
        
        maximums = {
            'Powerball': 5000000000,    # $5B maximum reasonable
            'Mega Millions': 5000000000, # $5B maximum reasonable
            'Lotto 47': 200000000,      # $200M maximum reasonable
            'Fantasy 5': 20000000       # $20M maximum reasonable
        }
        
        min_amount = minimums.get(game_name, 100000)
        max_amount = maximums.get(game_name, 1000000000)
        
        return min_amount <= amount <= max_amount
    
    def _save_debug_page(self, game_name: str):
        """Save page content for debugging when extraction fails"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"enhanced_debug_{game_name.lower().replace(' ', '_')}_{timestamp}.html"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            
            self.logger.info(f"Debug page saved: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug page: {e}")
    
    def fetch_all_jackpots_enhanced(self) -> Dict[str, Any]:
        """Fetch jackpots for all supported games using enhanced methods"""
        games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
        results = {}
        
        for game in games:
            try:
                jackpot_info = self.fetch_game_jackpot_enhanced(game)
                if jackpot_info:
                    results[game] = jackpot_info
                    self.logger.info(f"Enhanced extraction successful for {game}")
                else:
                    self.logger.warning(f"Enhanced extraction failed for {game}")
                
                # Be respectful to the server
                time.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Error in enhanced fetch for {game}: {e}")
                continue
        
        return results

# Convenience functions
def fetch_michigan_jackpot_enhanced_js(game_name: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """
    Convenience function to fetch a single game's jackpot using enhanced JavaScript scraping
    """
    try:
        with EnhancedMichiganLotteryJSScraper(headless=True, timeout=timeout) as scraper:
            return scraper.fetch_game_jackpot_enhanced(game_name)
    except Exception as e:
        logging.error(f"Enhanced JavaScript scraping failed for {game_name}: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Enhanced Michigan Lottery JavaScript Scraper")
    print("=" * 60)
    
    try:
        with EnhancedMichiganLotteryJSScraper(headless=True) as scraper:
            # Test one game thoroughly
            game = 'Powerball'
            print(f"\\n🎰 Enhanced testing {game}...")
            
            result = scraper.fetch_game_jackpot_enhanced(game)
            if result:
                print(f"✅ {game}: {result['formatted']} (Source: {result.get('source', 'unknown')})")
                print(f"📊 Details: {result}")
            else:
                print(f"❌ {game}: Enhanced extraction failed")
                    
    except Exception as e:
        print(f"❌ Error during enhanced testing: {e}")
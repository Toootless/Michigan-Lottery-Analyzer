#!/usr/bin/env python3
"""
Michigan Lottery JavaScript Scraper

This module provides JavaScript-enabled web scraping capabilities
for the Michigan lottery website using Selenium WebDriver.
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
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

class MichiganLotteryJSScraper:
    """JavaScript-enabled scraper for Michigan lottery website"""
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """
        Initialize the JavaScript scraper
        
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
        """Initialize and start the Chrome WebDriver"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Essential Chrome options for stability and security
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # User agent to appear as regular browser
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Disable logging to reduce noise
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            
            # Setup Chrome service with WebDriverManager
            service = Service(ChromeDriverManager().install())
            
            # Initialize WebDriver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            
            # Set page load timeout
            self.driver.set_page_load_timeout(self.timeout)
            
            # Execute script to avoid detection
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("Chrome WebDriver initialized successfully")
            
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
    
    def fetch_game_jackpot(self, game_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch jackpot information for a specific game using JavaScript rendering
        
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
            self.logger.info(f"Fetching {game_name} from {url}")
            
            # Load the page
            self.driver.get(url)
            
            # Wait for page to load and JavaScript to execute
            time.sleep(5)
            
            # Wait for content to be loaded
            try:
                WebDriverWait(self.driver, self.timeout).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                self.logger.warning(f"Page load timeout for {game_name}")
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Get page content after JavaScript execution
            page_source = self.driver.page_source
            
            # Extract jackpot information using multiple strategies
            jackpot_info = self._extract_jackpot_from_html(page_source, game_name)
            
            if jackpot_info:
                jackpot_info.update({
                    'source': 'michigan_lottery_js',
                    'last_updated': datetime.now(),
                    'url': url
                })
                self.logger.info(f"Successfully extracted jackpot for {game_name}: {jackpot_info.get('formatted', 'N/A')}")
                return jackpot_info
            else:
                # Try finding elements directly with Selenium
                return self._extract_jackpot_with_selenium(game_name)
                
        except Exception as e:
            self.logger.error(f"Error fetching {game_name}: {e}")
            return None
    
    def _extract_jackpot_from_html(self, html_content: str, game_name: str) -> Optional[Dict[str, Any]]:
        """Extract jackpot from HTML content using regex patterns"""
        try:
            # Comprehensive regex patterns for jackpot extraction
            patterns = [
                # Dollar amounts with million
                r'(?:jackpot|prize|estimated)[^$]*\$\s*([0-9,]+)\s*million',
                r'\$\s*([0-9,]+)\s*million[^a-z]*(?:jackpot|prize|estimated)',
                
                # Full dollar amounts
                r'(?:jackpot|prize|estimated)[^$]*\$\s*([0-9,]+),([0-9]{3}),([0-9]{3})',
                r'\$\s*([0-9,]+),([0-9]{3}),([0-9]{3})[^a-z]*(?:jackpot|prize)',
                
                # Amounts with various separators
                r'(?:jackpot|prize)[^$]*\$\s*([0-9]+(?:,[0-9]{3})*)',
                r'next\s*draw[^$]*\$\s*([0-9,]+)\s*million',
                
                # Alternative patterns
                r'estimated\s*at\s*\$\s*([0-9,]+)\s*million',
                r'current\s*jackpot[^$]*\$\s*([0-9,]+)',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, html_content.lower())
                for match in matches:
                    try:
                        if 'million' in pattern.lower():
                            # Extract million amount
                            amount_str = match.group(1).replace(',', '')
                            amount = int(amount_str) * 1000000
                        else:
                            # Extract full amount
                            amount_str = match.group(1).replace(',', '')
                            amount = int(amount_str)
                        
                        # Validate amount is reasonable for the game
                        if self._validate_jackpot_amount(amount, game_name):
                            return {
                                'amount': amount,
                                'formatted': f"${amount:,}",
                                'game': game_name,
                                'has_jackpot': True
                            }
                    except (ValueError, IndexError):
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting jackpot from HTML: {e}")
            return None
    
    def _extract_jackpot_with_selenium(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Extract jackpot using Selenium element finding"""
        try:
            # Common selectors for jackpot amounts
            selectors = [
                # Class-based selectors
                '[class*="jackpot"]',
                '[class*="prize"]',
                '[class*="amount"]',
                '[class*="estimated"]',
                
                # Data attribute selectors
                '[data-jackpot]',
                '[data-prize]',
                '[data-amount]',
                
                # Text-based searches
                '//*[contains(text(),"million")]',
                '//*[contains(text(),"$")]',
                '//*[contains(text(),"jackpot")]',
            ]
            
            for selector in selectors:
                try:
                    if selector.startswith('//'):
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
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
            
        except Exception as e:
            self.logger.error(f"Error extracting jackpot with Selenium: {e}")
            return None
    
    def _parse_jackpot_text(self, text: str, game_name: str) -> Optional[Dict[str, Any]]:
        """Parse jackpot amount from text string"""
        try:
            # Extract numbers and format indicators
            if 'million' in text.lower():
                match = re.search(r'\$?\s*([0-9,]+)\s*million', text.lower())
                if match:
                    amount = int(match.group(1).replace(',', '')) * 1000000
            else:
                match = re.search(r'\$?\s*([0-9,]+(?:,[0-9]{3})*)', text)
                if match:
                    amount = int(match.group(1).replace(',', ''))
                else:
                    return None
            
            if self._validate_jackpot_amount(amount, game_name):
                return {
                    'amount': amount,
                    'formatted': f"${amount:,}",
                    'game': game_name,
                    'has_jackpot': True
                }
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    def _validate_jackpot_amount(self, amount: int, game_name: str) -> bool:
        """Validate that jackpot amount is reasonable for the game"""
        minimums = {
            'Powerball': 20000000,      # $20M minimum
            'Mega Millions': 20000000,  # $20M minimum  
            'Lotto 47': 1000000,        # $1M minimum
            'Fantasy 5': 100000         # $100K minimum
        }
        
        maximums = {
            'Powerball': 3000000000,    # $3B maximum reasonable
            'Mega Millions': 3000000000, # $3B maximum reasonable
            'Lotto 47': 100000000,      # $100M maximum reasonable
            'Fantasy 5': 10000000       # $10M maximum reasonable
        }
        
        min_amount = minimums.get(game_name, 100000)
        max_amount = maximums.get(game_name, 1000000000)
        
        return min_amount <= amount <= max_amount
    
    def fetch_all_jackpots(self) -> Dict[str, Any]:
        """Fetch jackpots for all supported games"""
        games = ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']
        results = {}
        
        for game in games:
            try:
                jackpot_info = self.fetch_game_jackpot(game)
                if jackpot_info:
                    results[game] = jackpot_info
                    time.sleep(2)  # Be respectful to the server
                else:
                    self.logger.warning(f"Failed to fetch jackpot for {game}")
            except Exception as e:
                self.logger.error(f"Error fetching {game}: {e}")
                continue
        
        return results
    
    def test_connection(self) -> bool:
        """Test if the scraper can connect to Michigan lottery website"""
        try:
            if not self.driver:
                return False
            
            self.driver.get("https://www.michiganlottery.com/")
            time.sleep(3)
            
            title = self.driver.title
            return "michigan" in title.lower() or "lottery" in title.lower()
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

# Convenience functions for easy integration

def fetch_michigan_jackpot_js(game_name: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Convenience function to fetch a single game's jackpot using JavaScript
    
    Args:
        game_name: Name of the lottery game
        timeout: Maximum time to wait for page load
        
    Returns:
        Dictionary with jackpot information or None if failed
    """
    try:
        with MichiganLotteryJSScraper(headless=True, timeout=timeout) as scraper:
            return scraper.fetch_game_jackpot(game_name)
    except Exception as e:
        logging.error(f"JavaScript scraping failed for {game_name}: {e}")
        return None

def fetch_all_michigan_jackpots_js(timeout: int = 30) -> Dict[str, Any]:
    """
    Convenience function to fetch all game jackpots using JavaScript
    
    Args:
        timeout: Maximum time to wait for page loads
        
    Returns:
        Dictionary with jackpot information for all games
    """
    try:
        with MichiganLotteryJSScraper(headless=True, timeout=timeout) as scraper:
            return scraper.fetch_all_jackpots()
    except Exception as e:
        logging.error(f"JavaScript scraping failed: {e}")
        return {}

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Testing Michigan Lottery JavaScript Scraper")
    print("=" * 50)
    
    try:
        with MichiganLotteryJSScraper(headless=True) as scraper:
            # Test connection
            if scraper.test_connection():
                print("✅ Connection to Michigan lottery website successful")
            else:
                print("❌ Failed to connect to Michigan lottery website")
                exit(1)
            
            # Test individual game fetching
            for game in ['Powerball', 'Mega Millions', 'Lotto 47', 'Fantasy 5']:
                print(f"\n🎰 Testing {game}...")
                result = scraper.fetch_game_jackpot(game)
                if result:
                    print(f"✅ {game}: {result['formatted']} (Source: {result.get('source', 'unknown')})")
                else:
                    print(f"❌ {game}: Failed to fetch jackpot")
                    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        if not SELENIUM_AVAILABLE:
            print("💡 Install Selenium with: pip install selenium webdriver-manager")
#!/usr/bin/env python3
"""
Michigan Lottery PDF Scraper
Visits the lottery website, saves as PDF, and extracts jackpot amounts
"""

import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MichiganLotteryPDFScraper:
    """Scrapes Michigan Lottery website and saves as PDF"""
    
    def __init__(self, save_directory=None):
        """
        Initialize the scraper
        
        Args:
            save_directory: Directory to save PDF files (default: data/Saved_files)
        """
        if save_directory is None:
            save_directory = Path(__file__).parent / 'data' / 'Saved_files'
        
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.driver = None
        
    def _init_driver(self):
        """Initialize Chrome WebDriver with PDF printing capabilities"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Set window size for better rendering
            chrome_options.add_argument('--window-size=1920,1080')
            
            # User agent to avoid detection
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute CDP command to hide webdriver
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            logger.info("✅ Chrome WebDriver initialized for PDF generation")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebDriver: {e}")
            return False
    
    def save_page_as_pdf(self, url, filename=None):
        """
        Visit URL and save page as PDF
        
        Args:
            url: URL to visit
            filename: Optional filename for PDF (auto-generated if not provided)
            
        Returns:
            Path to saved PDF file or None if failed
        """
        if not self.driver:
            if not self._init_driver():
                return None
        
        try:
            logger.info(f"📥 Loading page: {url}")
            self.driver.get(url)
            
            # Wait for page to load
            logger.info("⏳ Waiting for page to load...")
            time.sleep(5)  # Initial wait for JavaScript
            
            # Wait for specific elements that indicate lottery data has loaded
            try:
                wait = WebDriverWait(self.driver, 15)
                # Look for game cards or lottery content
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "game-card")))
                logger.info("✅ Game content loaded")
            except Exception as e:
                logger.warning(f"⚠️ Timeout waiting for game cards, proceeding anyway: {e}")
            
            # Additional wait to ensure all dynamic content loads
            time.sleep(3)
            
            # Scroll to load any lazy-loaded content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"michigan_lottery_draw_games_{timestamp}.pdf"
            
            # Ensure .pdf extension
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            pdf_path = self.save_directory / filename
            
            # Print to PDF using Chrome DevTools Protocol
            logger.info("📄 Generating PDF...")
            
            result = self.driver.execute_cdp_cmd("Page.printToPDF", {
                "landscape": False,
                "displayHeaderFooter": False,
                "printBackground": True,
                "preferCSSPageSize": False,
                "paperWidth": 8.5,
                "paperHeight": 11,
                "marginTop": 0.4,
                "marginBottom": 0.4,
                "marginLeft": 0.4,
                "marginRight": 0.4,
                "scale": 0.9
            })
            
            # Decode and save PDF
            pdf_data = base64.b64decode(result['data'])
            
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)
            
            logger.info(f"✅ PDF saved: {pdf_path}")
            logger.info(f"📊 PDF size: {len(pdf_data) / 1024:.2f} KB")
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"❌ Error saving PDF: {e}")
            return None
    
    def extract_jackpots_from_pdf(self, pdf_path):
        """
        Extract jackpot amounts from saved PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with game names and jackpot amounts
        """
        try:
            import fitz  # PyMuPDF
            
            logger.info(f"📖 Reading PDF: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            jackpots = {}
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                full_text += page.get_text()
            
            doc.close()
            
            logger.info(f"📝 Extracted {len(full_text)} characters from PDF")
            
            # Game patterns with specific "Estimated Jackpot" keyword searches
            games = {
                'Powerball': {
                    'keywords': ['Powerball\nEstimated Jackpot:', 'Powerball', 'POWERBALL'],
                    'jackpot_marker': 'Estimated Jackpot:'
                },
                'Mega Millions': {
                    'keywords': ['Mega Millions\nEstimated Jackpot:', 'Mega Millions', 'MEGA MILLIONS'],
                    'jackpot_marker': 'Estimated Jackpot:'
                },
                'Lotto 47': {
                    'keywords': ['Lotto 47\nEstimated Jackpot', 'Lotto 47', 'LOTTO 47'],
                    'jackpot_marker': 'Estimated Jackpot'
                },
                'Fantasy 5': {
                    'keywords': ['Fantasy 5\nEstimated Jackpot', 'Fantasy 5', 'FANTASY 5'],
                    'jackpot_marker': 'Estimated Jackpot'
                }
            }
            
            # Jackpot amount patterns - looking for amounts after "Estimated Jackpot:"
            amount_patterns = [
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Billion|B)',  # $1.25 Billion or $1 Billion
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Million|M)',  # $445 Million or $445.5 Million
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Thousand|K)',  # $500 Thousand or $500K
                r'\$\s*(\d{1,3}(?:,\d{3})*)',  # $10,500,000 or $610,000
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Billion|B)(?![A-Z])',  # 1.25 Billion
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Million|M)(?![A-Z])',  # 445 Million
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:Thousand|K)(?![A-Z])',  # 500 Thousand
            ]
            
            for game_name, game_config in games.items():
                found = False
                
                # Try to find game name followed by jackpot marker
                for keyword in game_config['keywords']:
                    game_index = full_text.find(keyword)
                    if game_index != -1:
                        # Look for "Estimated Jackpot:" marker after game name
                        jackpot_marker = game_config['jackpot_marker']
                        marker_index = full_text.find(jackpot_marker, game_index)
                        
                        if marker_index != -1:
                            # Extract text after the marker (next 100 chars should contain the amount)
                            # Keep it short to avoid picking up jackpots from other games
                            context_start = marker_index + len(jackpot_marker)
                            context = full_text[context_start:context_start + 100]
                            
                            logger.info(f"🔍 Searching for {game_name} jackpot in context: {context[:100]}...")
                            
                            # Try to find jackpot amount in context
                            for pattern in amount_patterns:
                                match = re.search(pattern, context, re.IGNORECASE)
                                if match:
                                    try:
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
                                        
                                        # Validate amount is reasonable for lottery
                                        min_amount = 100_000 if game_name in ['Fantasy 5'] else 1_000_000
                                        max_amount = 2_000_000 if game_name == 'Fantasy 5' else 3_000_000_000  # Allow up to $3B for high Powerball jackpots
                                        if min_amount <= amount <= max_amount:
                                            formatted = f"${int(amount):,}"
                                            jackpots[game_name] = {
                                                'amount': int(amount),
                                                'formatted': formatted
                                            }
                                            logger.info(f"✅ Found {game_name}: {formatted}")
                                            found = True
                                            break
                                    except (ValueError, IndexError) as e:
                                        logger.warning(f"⚠️ Error parsing amount for {game_name}: {e}")
                                        continue
                            
                            if found:
                                break
                    
                    if found:
                        break
            
            return jackpots
            
        except ImportError:
            logger.error("❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
            return {}
        except Exception as e:
            logger.error(f"❌ Error extracting jackpots: {e}")
            return {}
    
    def scrape_and_extract(self, url=None):
        """
        Complete workflow: visit page, save PDF, extract jackpots
        
        Args:
            url: URL to scrape (default: Michigan Lottery draw games)
            
        Returns:
            Tuple of (pdf_path, jackpots_dict)
        """
        if url is None:
            url = "https://www.michiganlottery.com/games?WHERE_TO_PLAY=ONLINE&GAME_TYPE=DRAW"
        
        # Save page as PDF
        pdf_path = self.save_page_as_pdf(url)
        
        if not pdf_path:
            logger.error("❌ Failed to save PDF")
            return None, {}
        
        # Extract jackpots from PDF
        jackpots = self.extract_jackpots_from_pdf(pdf_path)
        
        return pdf_path, jackpots
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("✅ WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_jackpots_from_michigan_lottery():
    """
    Convenience function to get current jackpots
    
    Returns:
        Dictionary with jackpot amounts
    """
    save_dir = Path(__file__).parent / 'data' / 'Saved_files'
    
    with MichiganLotteryPDFScraper(save_directory=save_dir) as scraper:
        pdf_path, jackpots = scraper.scrape_and_extract()
        
        if jackpots:
            print("\n" + "="*60)
            print("🎰 Michigan Lottery Draw Games - Current Jackpots")
            print("="*60)
            for game, info in jackpots.items():
                print(f"{game:<20}: {info['formatted']}")
            print("="*60)
            print(f"📄 PDF saved to: {pdf_path}")
            print("="*60 + "\n")
            
            # Return simple dict with game:amount for compatibility
            return {game: info['amount'] for game, info in jackpots.items()}
        else:
            print("⚠️ No jackpots found in PDF")
            return {}


if __name__ == '__main__':
    print("🚀 Michigan Lottery PDF Scraper")
    print("=" * 60)
    print()
    
    # Run the scraper
    jackpots = get_jackpots_from_michigan_lottery()
    
    if not jackpots:
        print("\n💡 Note: If no jackpots were found, the PDF might not contain")
        print("   the jackpot amounts in extractable text format.")
        print("   Check the saved PDF manually to verify.")

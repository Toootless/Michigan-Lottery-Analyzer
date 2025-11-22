#!/usr/bin/env python3
"""
Michigan Lottery Winning Numbers Scraper v2
Uses PDF-based extraction with corrected number parsing
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
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MichiganLotteryWinningNumbers:
    """Extracts Michigan Lottery winning numbers from game PDFs"""
    
    # Game URLs
    GAME_URLS = {
        'Powerball': 'https://www.michiganlottery.com/games/powerball',
        'Lotto 47': 'https://www.michiganlottery.com/games/classic-lotto-47',
        'Mega Millions': 'https://www.michiganlottery.com/games/mega-millions',
    }
    
    def __init__(self, save_directory=None):
        """Initialize the scraper"""
        if save_directory is None:
            save_directory = Path(__file__).parent / 'data' / 'Saved_files'
        
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.driver = None
        
    def _init_driver(self):
        """Initialize Chrome WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("✓ Chrome WebDriver initialized")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to initialize WebDriver: {e}")
            return False
    
    def _close_driver(self):
        """Close WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("✓ WebDriver closed")
            except Exception as e:
                logger.warning(f"Warning closing WebDriver: {e}")
    
    def _get_pdf_from_page(self, url, game_name):
        """Generate PDF from webpage using Chrome DevTools Protocol"""
        try:
            logger.info(f"  Visiting {game_name} page...")
            self.driver.get(url)
            
            # Wait for page load
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(2)
            
            # Generate PDF via CDP
            result = self.driver.execute_cdp_cmd('Page.printToPDF', {
                'paperWidth': 8.5,
                'paperHeight': 11,
                'marginTop': 0.25,
                'marginBottom': 0.25,
                'marginLeft': 0.25,
                'marginRight': 0.25,
                'printBackground': True,
            })
            
            # Save PDF
            pdf_data = base64.b64decode(result['data'])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = self.save_directory / f"{game_name.replace(' ', '_')}_{timestamp}.pdf"
            
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)
            
            logger.info(f"  ✓ PDF saved: {pdf_path.name}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"  ✗ Failed to generate PDF for {game_name}: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        try:
            doc = fitz.open(str(pdf_path))
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if isinstance(page_text, str):
                    full_text += page_text + "\n"
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return None
    
    def extract_powerball(self, text):
        """Extract Powerball numbers: 5 white balls + 1 red powerball"""
        results = {'game': 'Powerball', 'draws': {}, 'doubleplay': None}
        
        try:
            # Regular drawing
            regular_match = re.search(r'REGULAR DRAWING\n([\d\n\s]+?)(?:DOUBLE PLAY|Double Play)', text, re.IGNORECASE)
            if regular_match:
                numbers_section = regular_match.group(1)
                all_numbers = re.findall(r'\b(\d{1,2})\b', numbers_section)
                
                if len(all_numbers) >= 6:
                    white_balls = sorted([int(n) for n in all_numbers[:5]])
                    powerball = int(all_numbers[5])
                    
                    if all(1 <= x <= 69 for x in white_balls) and 1 <= powerball <= 26:
                        results['draws']['regular'] = {
                            'main_numbers': white_balls,
                            'bonus_number': powerball,
                        }
                        logger.info(f"  ✓ Regular: {white_balls} + {powerball}")
            
            # Double play drawing
            doubleplay_match = re.search(r'DOUBLE PLAY DRAWING\n([\d\n\s]+?)(?:Game Details|Game Type|WIN CONDITIONS|$)', text, re.IGNORECASE)
            if doubleplay_match:
                numbers_section = doubleplay_match.group(1)
                all_numbers = re.findall(r'\b(\d{1,2})\b', numbers_section)
                
                if len(all_numbers) >= 6:
                    white_balls = sorted([int(n) for n in all_numbers[:5]])
                    powerball = int(all_numbers[5])
                    
                    if all(1 <= x <= 69 for x in white_balls) and 1 <= powerball <= 26:
                        results['doubleplay'] = {
                            'main_numbers': white_balls,
                            'bonus_number': powerball,
                        }
                        logger.info(f"  ✓ Doubleplay: {white_balls} + {powerball}")
        
        except Exception as e:
            logger.error(f"Error extracting Powerball: {e}")
        
        return results if results['draws'] else None
    
    def extract_mega_millions(self, text):
        """Extract Mega Millions numbers: 5 white balls + 1 mega ball"""
        results = {'game': 'Mega Millions', 'draws': {}, 'doubleplay': None}
        
        try:
            # Find numbers between "Showing: Fri" and "Past Drawings"
            # Mega Millions displays numbers after all the prize conditions
            match = re.search(r'Showing:.*?Fri.*?([0-9\n\s]+?)Past Drawings', text, re.IGNORECASE | re.DOTALL)
            if not match:
                # Try alternative pattern if first doesn't work
                match = re.search(r'Next Drawing.*?([0-9\n\s]+?)Past Drawings', text, re.IGNORECASE | re.DOTALL)
            
            if match:
                numbers_section = match.group(1)
                all_numbers = re.findall(r'\b(\d{1,2})\b', numbers_section)
                
                # Filter to get valid lottery numbers (skip dates like 21)
                # Strategy: take the first number that could be a valid white ball (1-70)
                # and continue until we have 6 numbers, but the last one must be <= 25 (mega ball)
                lottery_numbers = []
                for i, num_str in enumerate(all_numbers):
                    num = int(num_str)
                    # If this could be a white ball (1-70) or mega ball (1-25)
                    if 1 <= num <= 70:
                        lottery_numbers.append(num)
                
                # Now find the sequence where we have 5 white balls + 1 mega ball (<=25)
                if len(lottery_numbers) >= 6:
                    # The last number should be <= 25 (mega ball)
                    # Try taking the last 6 numbers
                    test_nums = lottery_numbers[-6:]
                    white_balls = sorted([int(n) for n in test_nums[:-1]])
                    mega_ball = int(test_nums[-1])
                    
                    if all(1 <= x <= 70 for x in white_balls) and 1 <= mega_ball <= 25:
                        results['draws']['regular'] = {
                            'main_numbers': white_balls,
                            'bonus_number': mega_ball,
                        }
                        logger.info(f"  ✓ Regular: {white_balls} + {mega_ball}")
            
            # Look for Double Play drawing
            doubleplay_match = re.search(r'DOUBLE PLAY DRAWING\n([\d\n\s]+?)(?:Game Details|Game Type|WIN CONDITIONS|$)', text, re.IGNORECASE)
            if doubleplay_match:
                numbers_section = doubleplay_match.group(1)
                all_numbers = re.findall(r'\b(\d{1,2})\b', numbers_section)
                
                if len(all_numbers) >= 6:
                    white_balls = sorted([int(n) for n in all_numbers[:5]])
                    mega_ball = int(all_numbers[5])
                    
                    if all(1 <= x <= 70 for x in white_balls) and 1 <= mega_ball <= 25:
                        results['doubleplay'] = {
                            'main_numbers': white_balls,
                            'bonus_number': mega_ball,
                        }
                        logger.info(f"  ✓ Doubleplay: {white_balls} + {mega_ball}")
        
        except Exception as e:
            logger.error(f"Error extracting Mega Millions: {e}")
        
        return results if results['draws'] else None
    
    def extract_lotto_47(self, text):
        """Extract Lotto 47 numbers: 6 numbers from 1-47"""
        results = {'game': 'Lotto 47', 'draws': {}, 'doubleplay': None}
        
        try:
            # Regular drawing
            regular_match = re.search(r'REGULAR DRAWING\n([\d\n\s]+?)(?:DOUBLE PLAY|Double Play)', text, re.IGNORECASE)
            if regular_match:
                numbers_section = regular_match.group(1)
                all_numbers = re.findall(r'\b(\d{1,2})\b', numbers_section)
                
                if len(all_numbers) >= 6:
                    numbers = sorted([int(n) for n in all_numbers[:6]])
                    
                    if all(1 <= x <= 47 for x in numbers):
                        results['draws']['regular'] = {
                            'main_numbers': numbers,
                        }
                        logger.info(f"  ✓ Regular: {numbers}")
            
            # Double play drawing
            doubleplay_match = re.search(r'DOUBLE PLAY DRAWING\n([\d\n\s]+?)(?:Game Details|Game Type|WIN CONDITIONS|$)', text, re.IGNORECASE)
            if doubleplay_match:
                numbers_section = doubleplay_match.group(1)
                all_numbers = re.findall(r'\b(\d{1,2})\b', numbers_section)
                
                if len(all_numbers) >= 6:
                    numbers = sorted([int(n) for n in all_numbers[:6]])
                    
                    if all(1 <= x <= 47 for x in numbers):
                        results['doubleplay'] = {
                            'main_numbers': numbers,
                        }
                        logger.info(f"  ✓ Doubleplay: {numbers}")
        
        except Exception as e:
            logger.error(f"Error extracting Lotto 47: {e}")
        
        return results if results['draws'] else None
    
    def get_all_winning_numbers(self):
        """Scrape all games' winning numbers"""
        results = {}
        
        if not self._init_driver():
            return None
        
        try:
            for game_name, url in self.GAME_URLS.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Scraping {game_name}")
                logger.info(f"{'='*60}")
                
                # Get PDF
                pdf_path = self._get_pdf_from_page(url, game_name)
                if not pdf_path:
                    logger.warning(f"Skipping {game_name} - PDF generation failed")
                    continue
                
                # Extract text
                text = self._extract_text_from_pdf(pdf_path)
                if not text:
                    logger.warning(f"Skipping {game_name} - text extraction failed")
                    continue
                
                # Parse numbers based on game
                if game_name == 'Powerball':
                    numbers = self.extract_powerball(text)
                elif game_name == 'Mega Millions':
                    numbers = self.extract_mega_millions(text)
                elif game_name == 'Lotto 47':
                    numbers = self.extract_lotto_47(text)
                else:
                    continue
                
                if numbers:
                    results[game_name] = numbers
                else:
                    logger.warning(f"No numbers extracted for {game_name}")
            
            return results
            
        finally:
            self._close_driver()


def test_scraper():
    """Test the scraper"""
    logger.info("Starting Winning Numbers Scraper v2")
    logger.info("=" * 60)
    
    scraper = MichiganLotteryWinningNumbers()
    all_numbers = scraper.get_all_winning_numbers()
    
    if all_numbers:
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION RESULTS")
        logger.info("=" * 60)
        
        for game_name, data in all_numbers.items():
            logger.info(f"\n{game_name}:")
            if data.get('draws'):
                for draw_type, draw in data['draws'].items():
                    nums_str = f"{draw['main_numbers']}"
                    if 'bonus_number' in draw:
                        nums_str += f" + {draw['bonus_number']}"
                    logger.info(f"  {draw_type.title()}: {nums_str}")
            
            if data.get('doubleplay'):
                nums_str = f"{data['doubleplay']['main_numbers']}"
                if 'bonus_number' in data['doubleplay']:
                    nums_str += f" + {data['doubleplay']['bonus_number']}"
                logger.info(f"  Doubleplay: {nums_str}")
    else:
        logger.error("Failed to retrieve winning numbers")


if __name__ == '__main__':
    test_scraper()

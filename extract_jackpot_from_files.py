#!/usr/bin/env python3
"""
Extract jackpot amounts from saved HTML or PDF files
"""

import re
import os
from pathlib import Path
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_jackpot_from_html(html_file_path):
    """
    Extract jackpot amount from Michigan Lottery HTML file
    
    Args:
        html_file_path: Path to HTML file
        
    Returns:
        dict with 'amount' (int) and 'formatted' (str), or None if not found
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Strategy 1: Look for jackpot in meta tags or title
        patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:Million|M)',  # $445 Million
            r'jackpot[:\s]*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:Million|M)',  # Jackpot: $445 Million
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Million',  # $445.5 Million
            r'(\d{1,3}(?:,\d{3})*)\s*Million\s*Jackpot',  # 445 Million Jackpot
        ]
        
        # Search in text content
        text_content = soup.get_text()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_content, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    # Try to parse as millions
                    if 'million' in match.group(0).lower():
                        amount = float(amount_str) * 1_000_000
                    else:
                        amount = float(amount_str)
                    
                    # Format nicely
                    if amount >= 1_000_000:
                        formatted = f"${int(amount):,}"
                    else:
                        formatted = f"${int(amount):,}"
                    
                    logger.info(f"Found jackpot: {formatted}")
                    return {
                        'amount': int(amount),
                        'formatted': formatted
                    }
                except (ValueError, IndexError):
                    continue
        
        # Strategy 2: Look for specific HTML elements commonly used for jackpots
        jackpot_selectors = [
            {'class': re.compile(r'jackpot', re.I)},
            {'id': re.compile(r'jackpot', re.I)},
            {'class': re.compile(r'prize', re.I)},
            {'data-jackpot': True}
        ]
        
        for selector in jackpot_selectors:
            elements = soup.find_all(attrs=selector)
            for element in elements:
                text = element.get_text()
                # Look for dollar amounts
                dollar_match = re.search(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
                if dollar_match:
                    amount_str = dollar_match.group(1).replace(',', '')
                    try:
                        amount = float(amount_str)
                        if 'million' in text.lower():
                            amount *= 1_000_000
                        
                        formatted = f"${int(amount):,}"
                        logger.info(f"Found jackpot in element: {formatted}")
                        return {
                            'amount': int(amount),
                            'formatted': formatted
                        }
                    except ValueError:
                        continue
        
        # Strategy 3: Look in JavaScript/JSON data embedded in page
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string:
                # Look for jackpot in JSON data
                json_match = re.search(r'"jackpot"[:\s]*(\d+)', script.string, re.I)
                if json_match:
                    try:
                        amount = int(json_match.group(1))
                        formatted = f"${amount:,}"
                        logger.info(f"Found jackpot in script: {formatted}")
                        return {
                            'amount': amount,
                            'formatted': formatted
                        }
                    except ValueError:
                        continue
        
        logger.warning(f"Could not find jackpot in {html_file_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting from HTML: {e}")
        return None


def extract_jackpot_from_pdf(pdf_file_path):
    """
    Extract jackpot amount from PDF file using OCR or text extraction
    
    Args:
        pdf_file_path: Path to PDF file
        
    Returns:
        dict with 'amount' (int) and 'formatted' (str), or None if not found
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_file_path)
        
        # Extract text from first few pages
        text_content = ""
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text_content += page.get_text()
        
        doc.close()
        
        # Look for jackpot patterns
        patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:Million|M)',
            r'jackpot[:\s]*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*)\s*Million\s*Jackpot',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_content, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    if 'million' in match.group(0).lower():
                        amount = float(amount_str) * 1_000_000
                    else:
                        amount = float(amount_str)
                    
                    formatted = f"${int(amount):,}"
                    logger.info(f"Found jackpot in PDF: {formatted}")
                    return {
                        'amount': int(amount),
                        'formatted': formatted
                    }
                except (ValueError, IndexError):
                    continue
        
        logger.warning(f"Could not find jackpot in PDF {pdf_file_path}")
        return None
        
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        return None
    except Exception as e:
        logger.error(f"Error extracting from PDF: {e}")
        return None


def extract_jackpot_from_file(file_path):
    """
    Auto-detect file type and extract jackpot
    
    Args:
        file_path: Path to HTML or PDF file
        
    Returns:
        dict with 'amount' (int) and 'formatted' (str), or None if not found
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    if file_path.suffix.lower() == '.html':
        return extract_jackpot_from_html(file_path)
    elif file_path.suffix.lower() == '.pdf':
        return extract_jackpot_from_pdf(file_path)
    else:
        logger.error(f"Unsupported file type: {file_path.suffix}")
        return None


def find_latest_debug_file(game_name, directory='.'):
    """
    Find the most recent debug HTML file for a game
    
    Args:
        game_name: Name of the game (e.g., 'powerball', 'mega_millions')
        directory: Directory to search in
        
    Returns:
        Path to most recent file, or None if not found
    """
    directory = Path(directory)
    pattern = f"enhanced_debug_{game_name.lower().replace(' ', '_')}*.html"
    
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if files:
        logger.info(f"Found latest file: {files[0]}")
        return files[0]
    else:
        logger.warning(f"No debug files found for {game_name}")
        return None


def extract_all_game_jackpots(directory='.'):
    """
    Extract jackpots from all saved game files in directory
    
    Args:
        directory: Directory containing debug HTML files
        
    Returns:
        dict mapping game names to jackpot info
    """
    results = {}
    games = ['powerball', 'mega_millions', 'lotto_47', 'fantasy_5']
    
    for game in games:
        file_path = find_latest_debug_file(game, directory)
        if file_path:
            jackpot_info = extract_jackpot_from_file(file_path)
            if jackpot_info:
                results[game] = jackpot_info
            else:
                results[game] = {'error': 'Could not extract jackpot'}
        else:
            results[game] = {'error': 'No debug file found'}
    
    return results


if __name__ == '__main__':
    print("🎰 Jackpot Extractor from Saved Files")
    print("=" * 50)
    print()
    
    # Test on latest saved debug files
    current_dir = Path(__file__).parent
    results = extract_all_game_jackpots(current_dir)
    
    for game, info in results.items():
        game_display = game.replace('_', ' ').title()
        if 'error' in info:
            print(f"{game_display:<20}: ❌ {info['error']}")
        else:
            print(f"{game_display:<20}: ✅ {info['formatted']}")
    
    print()
    print("💡 Usage:")
    print("  from extract_jackpot_from_files import extract_jackpot_from_file")
    print("  result = extract_jackpot_from_file('enhanced_debug_powerball_20251118_140400.html')")
    print("  print(result['formatted'])  # '$445,000,000'")

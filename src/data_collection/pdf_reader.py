"""
PDF Reader Module for Michigan Lottery Analyzer
Extracts lottery winning numbers from historical PDF files (2000-2021)

This module processes PDF files containing historical lottery data and extracts:
- Winning numbers for various lottery games
- Draw dates and times
- Game types and variants
- Jackpot amounts (when available)

Author: Michigan Lottery Analyzer Team
Date: October 2025
"""

import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PDF_AVAILABLE = True
    logger.info("✅ PyPDF2 available for PDF processing")
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_AVAILABLE = True 
        logger.info("✅ PyMuPDF available for PDF processing")
    except ImportError:
        PDF_AVAILABLE = False
        logger.warning("❌ No PDF library available. Install PyPDF2 or PyMuPDF: pip install PyPDF2 pymupdf")

class LotteryPDFReader:
    """
    Comprehensive PDF reader for historical lottery data extraction
    """
    
    def __init__(self, pdf_directory: str):
        """
        Initialize the PDF reader with the directory containing PDF files
        
        Args:
            pdf_directory: Path to directory containing PDF files
        """
        self.pdf_directory = Path(pdf_directory)
        self.supported_games = {
            'Fantasy 5': {
                'numbers_count': 5,
                'numbers_range': (1, 39),
                'patterns': [
                    r'Fantasy\s*5.*?(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})',
                    r'(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})',
                ]
            },
            'Daily 4': {
                'numbers_count': 4,
                'numbers_range': (0, 9),
                'patterns': [
                    r'Daily\s*4.*?(\d)\s*[-\s]*(\d)\s*[-\s]*(\d)\s*[-\s]*(\d)',
                    r'(\d{4})',  # Four consecutive digits
                ]
            },
            'Daily 3': {
                'numbers_count': 3,
                'numbers_range': (0, 9),
                'patterns': [
                    r'Daily\s*3.*?(\d)\s*[-\s]*(\d)\s*[-\s]*(\d)',
                    r'(\d{3})',  # Three consecutive digits
                ]
            },
            'Keno': {
                'numbers_count': 20,
                'numbers_range': (1, 80),
                'patterns': [
                    r'Keno.*?(?:(\d{1,2})\s*[-,\s]\s*){19}(\d{1,2})',
                ]
            },
            'Lotto 47': {
                'numbers_count': 6,
                'numbers_range': (1, 47),
                'patterns': [
                    r'Lotto\s*47.*?(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})',
                ]
            },
            'Powerball': {
                'numbers_count': 5,
                'numbers_range': (1, 69),
                'bonus_range': (1, 26),
                'patterns': [
                    r'Powerball.*?(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2}).*?(?:PB|Power|Bonus).*?(\d{1,2})',
                ]
            },
            'Mega Millions': {
                'numbers_count': 5,
                'numbers_range': (1, 70),
                'bonus_range': (1, 25),
                'patterns': [
                    r'Mega\s*Millions.*?(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2}).*?(?:MB|Mega|Bonus).*?(\d{1,2})',
                ]
            }
        }
        
        # Date patterns for extracting draw dates
        self.date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # Month DD, YYYY
        ]
        
        # Jackpot patterns
        self.jackpot_patterns = [
            r'\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|Million|MILLION)',
            r'\$([0-9,]+(?:\.[0-9]{2})?)',
            r'Jackpot.*?\$([0-9,]+(?:\.[0-9]{2})?)',
            r'Prize.*?\$([0-9,]+(?:\.[0-9]{2})?)',
        ]
    
    def read_pdf_with_pypdf2(self, pdf_path: Path) -> str:
        """
        Read PDF content using PyPDF2
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {e}")
            return ""
    
    def read_pdf_with_pymupdf(self, pdf_path: Path) -> str:
        """
        Read PDF content using PyMuPDF (fitz)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF with PyMuPDF: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using available library
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not PDF_AVAILABLE:
            logger.error("No PDF library available")
            return ""
        
        # Try PyMuPDF first (generally better text extraction)
        try:
            import fitz
            return self.read_pdf_with_pymupdf(pdf_path)
        except ImportError:
            pass
        
        # Fall back to PyPDF2
        try:
            import PyPDF2
            return self.read_pdf_with_pypdf2(pdf_path)
        except ImportError:
            pass
        
        return ""
    
    def parse_date(self, date_text: str, year: int) -> Optional[datetime]:
        """
        Parse date from various formats
        
        Args:
            date_text: Date text to parse
            year: Year context for parsing
            
        Returns:
            Parsed datetime object or None
        """
        for pattern in self.date_patterns:
            match = re.search(pattern, date_text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 3:
                        if pattern.startswith(r'(\d{4})'):  # YYYY-MM-DD
                            year, month, day = match.groups()
                            return datetime(int(year), int(month), int(day))
                        elif pattern.startswith(r'(\w+)'):  # Month DD, YYYY
                            month_name, day, year = match.groups()
                            # Convert month name to number
                            month_names = {
                                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                'september': 9, 'october': 10, 'november': 11, 'december': 12,
                                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                            }
                            month_num = month_names.get(month_name.lower())
                            if month_num:
                                return datetime(int(year), month_num, int(day))
                        else:  # MM/DD/YYYY or MM-DD-YYYY
                            month, day, year = match.groups()
                            return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None
    
    def extract_numbers_from_text(self, text: str, game_name: str) -> List[Dict[str, Any]]:
        """
        Extract lottery numbers from text for a specific game
        
        Args:
            text: Text content to search
            game_name: Name of lottery game
            
        Returns:
            List of extracted number records
        """
        if game_name not in self.supported_games:
            return []
        
        game_config = self.supported_games[game_name]
        patterns = game_config['patterns']
        results = []
        
        # Split text into lines for better date/number association
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Try each pattern for this game
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                
                for match in matches:
                    numbers = []
                    bonus_number = None
                    
                    # Extract numbers based on pattern groups
                    groups = match.groups()
                    
                    if game_name in ['Daily 3', 'Daily 4']:
                        # Handle digit games specially
                        if len(groups) == 1 and len(groups[0]) == game_config['numbers_count']:
                            # Single group with all digits (e.g., "1234")
                            numbers = [int(d) for d in groups[0]]
                        else:
                            # Multiple groups with individual digits
                            numbers = [int(g) for g in groups if g.isdigit()]
                    else:
                        # Handle lottery games
                        if game_name in ['Powerball', 'Mega Millions'] and len(groups) > game_config['numbers_count']:
                            # Last group is bonus number
                            numbers = [int(g) for g in groups[:-1] if g.isdigit()]
                            bonus_number = int(groups[-1]) if groups[-1].isdigit() else None
                        else:
                            numbers = [int(g) for g in groups if g.isdigit()]
                    
                    # Validate numbers
                    if len(numbers) == game_config['numbers_count']:
                        numbers_range = game_config['numbers_range']
                        valid_numbers = all(numbers_range[0] <= n <= numbers_range[1] for n in numbers)
                        
                        valid_bonus = True
                        if bonus_number is not None and 'bonus_range' in game_config:
                            bonus_range = game_config['bonus_range']
                            valid_bonus = bonus_range[0] <= bonus_number <= bonus_range[1]
                        
                        if valid_numbers and valid_bonus:
                            # Try to find date in surrounding lines
                            draw_date = None
                            for j in range(max(0, i-2), min(len(lines), i+3)):
                                date_candidate = self.parse_date(lines[j], 2000)  # Default year context
                                if date_candidate:
                                    draw_date = date_candidate
                                    break
                            
                            # Try to find jackpot in surrounding lines
                            jackpot = None
                            for j in range(max(0, i-2), min(len(lines), i+3)):
                                for jackpot_pattern in self.jackpot_patterns:
                                    jackpot_match = re.search(jackpot_pattern, lines[j], re.IGNORECASE)
                                    if jackpot_match:
                                        try:
                                            jackpot_str = jackpot_match.group(1).replace(',', '')
                                            jackpot = float(jackpot_str)
                                            if 'million' in lines[j].lower():
                                                jackpot *= 1000000
                                            break
                                        except ValueError:
                                            pass
                                if jackpot:
                                    break
                            
                            result = {
                                'game': game_name,
                                'numbers': numbers,
                                'bonus_number': bonus_number,
                                'date': draw_date,
                                'jackpot': jackpot,
                                'source': 'PDF',
                                'raw_text': line.strip()
                            }
                            
                            results.append(result)
        
        return results
    
    def process_pdf_file(self, pdf_path: Path, year: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a single PDF file and extract all lottery data
        
        Args:
            pdf_path: Path to PDF file
            year: Year context for date parsing
            
        Returns:
            Dictionary with game names as keys and lists of results as values
        """
        logger.info(f"📄 Processing PDF: {pdf_path.name}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"⚠️ Could not extract text from {pdf_path}")
            return {}
        
        logger.info(f"📝 Extracted {len(text)} characters from PDF")
        
        # Extract numbers for each supported game
        all_results = {}
        
        for game_name in self.supported_games.keys():
            results = self.extract_numbers_from_text(text, game_name)
            if results:
                # Set year context if provided
                if year:
                    for result in results:
                        if result['date'] is None:
                            # Try to infer date from filename year
                            result['date'] = datetime(year, 1, 1)  # Default to start of year
                        elif result['date'].year == 1900:  # Default year from parsing
                            result['date'] = result['date'].replace(year=year)
                
                all_results[game_name] = results
                logger.info(f"🎯 Found {len(results)} {game_name} results in {pdf_path.name}")
        
        return all_results
    
    def process_all_pdfs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all PDF files in the directory
        
        Returns:
            Dictionary with all extracted lottery data organized by game
        """
        if not self.pdf_directory.exists():
            logger.error(f"❌ PDF directory not found: {self.pdf_directory}")
            return {}
        
        logger.info(f"🔍 Scanning PDF directory: {self.pdf_directory}")
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {self.pdf_directory}")
            return {}
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files to process")
        
        all_data = {}
        
        for pdf_file in sorted(pdf_files):
            # Extract year from filename (e.g., "2020.pdf" -> 2020)
            year = None
            try:
                year = int(pdf_file.stem)
            except ValueError:
                logger.warning(f"⚠️ Could not extract year from filename: {pdf_file.name}")
            
            # Process the PDF
            pdf_results = self.process_pdf_file(pdf_file, year)
            
            # Merge results into all_data
            for game_name, results in pdf_results.items():
                if game_name not in all_data:
                    all_data[game_name] = []
                all_data[game_name].extend(results)
        
        # Sort results by date within each game
        for game_name in all_data:
            all_data[game_name].sort(key=lambda x: x['date'] if x['date'] else datetime.min)
        
        # Log summary
        total_results = sum(len(results) for results in all_data.values())
        logger.info(f"✅ Processing complete! Extracted {total_results} total results across {len(all_data)} games")
        
        for game_name, results in all_data.items():
            logger.info(f"  📊 {game_name}: {len(results)} results")
        
        return all_data
    
    def save_to_json(self, data: Dict[str, List[Dict[str, Any]]], output_path: str):
        """
        Save extracted data to JSON file
        
        Args:
            data: Extracted lottery data
            output_path: Path to save JSON file
        """
        # Convert datetime objects to strings for JSON serialization
        json_data = {}
        for game_name, results in data.items():
            json_data[game_name] = []
            for result in results:
                json_result = result.copy()
                if json_result['date']:
                    json_result['date'] = json_result['date'].isoformat()
                json_data[game_name].append(json_result)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"💾 Data saved to: {output_path}")
    
    def generate_summary_report(self, data: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Generate a summary report of extracted data
        
        Args:
            data: Extracted lottery data
            
        Returns:
            Summary report as string
        """
        report_lines = [
            "=" * 60,
            "📊 LOTTERY PDF EXTRACTION SUMMARY REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}",
            ""
        ]
        
        total_results = sum(len(results) for results in data.values())
        report_lines.append(f"🎯 Total Results Extracted: {total_results}")
        report_lines.append(f"🎮 Games Processed: {len(data)}")
        report_lines.append("")
        
        for game_name, results in data.items():
            if results:
                report_lines.append(f"📈 {game_name}:")
                report_lines.append(f"   • Total Draws: {len(results)}")
                
                # Date range
                dates = [r['date'] for r in results if r['date']]
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    report_lines.append(f"   • Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                
                # Sample numbers
                if len(results) > 0:
                    sample = results[0]
                    numbers_str = ', '.join(map(str, sample['numbers']))
                    report_lines.append(f"   • Sample Numbers: {numbers_str}")
                    if sample['bonus_number']:
                        report_lines.append(f"   • Sample Bonus: {sample['bonus_number']}")
                
                report_lines.append("")
        
        return '\n'.join(report_lines)


def install_pdf_dependencies():
    """
    Install required PDF processing libraries
    """
    import subprocess
    import sys
    
    libraries = ['PyPDF2', 'pymupdf']
    
    for library in libraries:
        try:
            __import__(library.lower().replace('pdf2', 'PDF2'))
            print(f"✅ {library} is already installed")
        except ImportError:
            print(f"📦 Installing {library}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])
                print(f"✅ {library} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {library}")


def main():
    """
    Main function to demonstrate PDF reading functionality
    """
    # Directory containing PDF files
    pdf_directory = r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games"
    
    # Initialize PDF reader
    reader = LotteryPDFReader(pdf_directory)
    
    # Process all PDFs
    print("🚀 Starting PDF processing...")
    data = reader.process_all_pdfs()
    
    if data:
        # Save to JSON
        output_path = "data/historical_lottery_data.json"
        reader.save_to_json(data, output_path)
        
        # Generate and display summary
        summary = reader.generate_summary_report(data)
        print(summary)
        
        # Save summary to file
        summary_path = "data/pdf_extraction_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"📄 Summary saved to: {summary_path}")
    
    else:
        print("❌ No data extracted from PDF files")


if __name__ == "__main__":
    main()
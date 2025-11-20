"""
Enhanced PDF Reader with OCR for Michigan Lottery Charts/Tables
Extracts lottery winning numbers from image-based PDF files using OCR

This module processes PDF files containing charts and tables by:
1. Converting PDF pages to images
2. Using OCR to extract text from images
3. Parsing lottery numbers from the extracted text
4. Handling various chart/table formats

Author: Michigan Lottery Analyzer Team
Date: October 2025
"""

import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from PIL import Image

# Check for OCR dependencies
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
    logger.info("✅ OCR libraries available (pytesseract, PIL, pdf2image)")
except ImportError as e:
    OCR_AVAILABLE = False
    logger.warning(f"❌ OCR libraries not available: {e}")
    logger.warning("Install with: pip install pytesseract Pillow pdf2image")

# Check for PDF libraries
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
    logger.info("✅ PyMuPDF available for PDF processing")
except ImportError:
    try:
        import PyPDF2
        PDF_AVAILABLE = True
        logger.info("✅ PyPDF2 available for PDF processing")
    except ImportError:
        PDF_AVAILABLE = False
        logger.warning("❌ No PDF library available")

class LotteryPDFOCRReader:
    """
    Enhanced PDF reader with OCR capabilities for lottery charts and tables
    """
    
    def __init__(self, pdf_directory: str):
        """
        Initialize the OCR PDF reader
        
        Args:
            pdf_directory: Path to directory containing PDF files
        """
        self.pdf_directory = Path(pdf_directory)
        
        # Lottery game patterns optimized for OCR text
        self.game_patterns = {
            'Fantasy 5': {
                'numbers_count': 5,
                'numbers_range': (1, 39),
                'patterns': [
                    # Look for 5 numbers in various formats
                    r'(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})',
                    r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})',
                    # OCR might see numbers in tables
                    r'(?:Fantasy|F5|F-5).*?(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})',
                ],
                'keywords': ['Fantasy', 'F5', 'F-5', 'fantasy']
            },
            'Daily 4': {
                'numbers_count': 4,
                'numbers_range': (0, 9),
                'patterns': [
                    r'(\d)\s*[-\s,]\s*(\d)\s*[-\s,]\s*(\d)\s*[-\s,]\s*(\d)',
                    r'(\d)\s+(\d)\s+(\d)\s+(\d)',
                    r'(\d{4})',  # Four consecutive digits
                    # OCR patterns for Daily 4
                    r'(?:Daily|D4|D-4).*?(\d)\D*(\d)\D*(\d)\D*(\d)',
                ],
                'keywords': ['Daily 4', 'D4', 'D-4', 'daily', 'four']
            },
            'Daily 3': {
                'numbers_count': 3,
                'numbers_range': (0, 9),
                'patterns': [
                    r'(\d)\s*[-\s,]\s*(\d)\s*[-\s,]\s*(\d)',
                    r'(\d)\s+(\d)\s+(\d)',
                    r'(\d{3})',  # Three consecutive digits
                    # OCR patterns for Daily 3
                    r'(?:Daily|D3|D-3).*?(\d)\D*(\d)\D*(\d)',
                ],
                'keywords': ['Daily 3', 'D3', 'D-3', 'daily', 'three']
            },
            'Keno': {
                'numbers_count': 20,
                'numbers_range': (1, 80),
                'patterns': [
                    # Keno has 20 numbers - look for long sequences
                    r'(?:Keno|KENO).*?(?:(\d{1,2})\D+){19}(\d{1,2})',
                ],
                'keywords': ['Keno', 'KENO', 'keno']
            },
            'Lotto 47': {
                'numbers_count': 6,
                'numbers_range': (1, 47),
                'patterns': [
                    r'(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})',
                    r'(?:Lotto|L47).*?(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})',
                ],
                'keywords': ['Lotto 47', 'Lotto', 'L47', 'lotto']
            },
            'Powerball': {
                'numbers_count': 6,  # 5 main numbers + 1 Powerball
                'numbers_range': (1, 69),  # Main numbers 1-69, Powerball 1-26
                'patterns': [
                    # Standard format: 5 main numbers + Powerball
                    r'(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})',
                    r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})',
                    # OCR might see Powerball format
                    r'(?:Powerball|PB|Power).*?(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})',
                    # Sometimes main numbers and Powerball are separated
                    r'(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(?:PB|Powerball)?\D*(\d{1,2})',
                ],
                'keywords': ['Powerball', 'Power Ball', 'PB', 'powerball', 'power']
            },
            'Mega Millions': {
                'numbers_count': 6,  # 5 main numbers + 1 Mega Ball
                'numbers_range': (1, 70),  # Main numbers 1-70, Mega Ball 1-25
                'patterns': [
                    # Standard format: 5 main numbers + Mega Ball
                    r'(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})',
                    r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})',
                    # OCR might see Mega Millions format
                    r'(?:Mega|MM|Million).*?(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})',
                    # Sometimes main numbers and Mega Ball are separated
                    r'(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(?:MB|Mega)?\D*(\d{1,2})',
                ],
                'keywords': ['Mega Millions', 'Mega Million', 'MM', 'mega', 'million']
            },
            'Lucky for Life': {
                'numbers_count': 6,  # 5 main numbers + 1 Lucky Ball
                'numbers_range': (1, 48),  # Main numbers 1-48, Lucky Ball 1-18
                'patterns': [
                    # Standard format: 5 main numbers + Lucky Ball
                    r'(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})\s*[-\s,]\s*(\d{1,2})',
                    r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})',
                    # OCR might see Lucky for Life format
                    r'(?:Lucky|LFL|Life).*?(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})',
                    # Sometimes main numbers and Lucky Ball are separated
                    r'(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(?:LB|Lucky)?\D*(\d{1,2})',
                ],
                'keywords': ['Lucky for Life', 'Lucky Life', 'LFL', 'lucky', 'life']
            }
        }
        
        # Date patterns for OCR text (OCR might introduce errors)
        self.date_patterns = [
            r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})',  # MM/DD/YYYY with various separators
            r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})',   # MM/DD/YY
            r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})',   # YYYY/MM/DD
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',             # Month DD, YYYY
            # OCR might separate differently
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})',          # MM DD YYYY
        ]
    
    def setup_tesseract(self):
        """
        Setup Tesseract OCR engine with optimal settings for lottery data
        """
        if not OCR_AVAILABLE:
            return False
        
        # Try to find Tesseract executable
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
            'tesseract',  # If in PATH
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path) or path == 'tesseract':
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    # Test OCR
                    pytesseract.get_tesseract_version()
                    logger.info(f"✅ Tesseract found at: {path}")
                    return True
                except Exception as e:
                    continue
        
        logger.warning("❌ Tesseract OCR not found. Please install Tesseract-OCR")
        logger.warning("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    def convert_pdf_to_images(self, pdf_path) -> List[Any]:
        """
        Convert PDF pages to images for OCR processing using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file (string or Path object)
            
        Returns:
            List of PIL Images, one per page
        """
        try:
            import io
            import fitz  # PyMuPDF
            
            # Ensure pdf_path is a string or Path for fitz.open
            if hasattr(pdf_path, '__str__'):
                pdf_path_str = str(pdf_path)
            else:
                pdf_path_str = pdf_path
                
            # Use PyMuPDF (fitz) to convert PDF to images
            doc = fitz.open(pdf_path_str)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Create transformation matrix for DPI scaling
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
                
            doc.close()
            pdf_name = getattr(pdf_path, 'name', str(pdf_path).split('/')[-1].split('\\')[-1])
            logger.info(f"📄 Converted {pdf_name} to {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error converting PDF to images: {e}")
            return []
    
    def extract_text_from_image(self, image: Any) -> str:
        """
        Extract text from an image using OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        try:
            import pytesseract
            
            # Make sure Tesseract is set up
            self.setup_tesseract()
            
            # OCR configuration optimized for lottery data
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /\-.,:()'
            
            text = pytesseract.image_to_string(image, config=custom_config)
            return text
            
        except Exception as e:
            logger.error(f"❌ OCR error: {e}")
            return ""
    
    def preprocess_image_for_ocr(self, image: Any) -> Any:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            image: Original PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast (optional - can help with faded text)
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increase contrast
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing error: {e}")
            return image
    
    def extract_numbers_from_ocr_text(self, text: str, game_name: str) -> List[Dict[str, Any]]:
        """
        Extract lottery numbers from OCR-extracted text
        
        Args:
            text: OCR-extracted text
            game_name: Name of lottery game
            
        Returns:
            List of extracted number records
        """
        if game_name not in self.game_patterns:
            return []
        
        game_config = self.game_patterns[game_name]
        patterns = game_config['patterns']
        keywords = game_config['keywords']
        results = []
        
        # Split text into lines for better processing
        lines = text.split('\n')
        
        # Look for lines that might contain the game data
        relevant_lines = []
        for line in lines:
            line_lower = line.lower()
            if any(keyword.lower() in line_lower for keyword in keywords):
                relevant_lines.append(line)
                # Also add nearby lines (context)
                line_idx = lines.index(line)
                for i in range(max(0, line_idx-2), min(len(lines), line_idx+3)):
                    if lines[i] not in relevant_lines:
                        relevant_lines.append(lines[i])
        
        # If no keyword matches, process all lines (fallback)
        if not relevant_lines:
            relevant_lines = lines
        
        # Extract numbers from relevant lines
        for line in relevant_lines:
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                
                for match in matches:
                    numbers = []
                    groups = match.groups()
                    
                    if game_name in ['Daily 3', 'Daily 4']:
                        # Handle digit games
                        if len(groups) == 1 and len(groups[0]) == game_config['numbers_count']:
                            # Single group with all digits
                            numbers = [int(d) for d in groups[0]]
                        else:
                            # Multiple groups
                            numbers = [int(g) for g in groups if g.isdigit()]
                    else:
                        # Handle lottery games
                        numbers = [int(g) for g in groups if g.isdigit()]
                    
                    # Validate numbers based on game type
                    if len(numbers) == game_config['numbers_count']:
                        valid_numbers = False
                        main_numbers = []
                        bonus_number = None
                        
                        if game_name == 'Powerball':
                            # Powerball: 5 main numbers (1-69) + 1 Powerball (1-26)
                            if len(numbers) == 6:
                                main_numbers = numbers[:5]
                                bonus_number = numbers[5]
                                valid_main = all(1 <= n <= 69 for n in main_numbers)
                                valid_bonus = 1 <= bonus_number <= 26
                                valid_numbers = valid_main and valid_bonus
                        
                        elif game_name == 'Mega Millions':
                            # Mega Millions: 5 main numbers (1-70) + 1 Mega Ball (1-25)
                            if len(numbers) == 6:
                                main_numbers = numbers[:5]
                                bonus_number = numbers[5]
                                valid_main = all(1 <= n <= 70 for n in main_numbers)
                                valid_bonus = 1 <= bonus_number <= 25
                                valid_numbers = valid_main and valid_bonus
                        
                        elif game_name == 'Lucky for Life':
                            # Lucky for Life: 5 main numbers (1-48) + 1 Lucky Ball (1-18)
                            if len(numbers) == 6:
                                main_numbers = numbers[:5]
                                bonus_number = numbers[5]
                                valid_main = all(1 <= n <= 48 for n in main_numbers)
                                valid_bonus = 1 <= bonus_number <= 18
                                valid_numbers = valid_main and valid_bonus
                        
                        else:
                            # Other games: use standard range validation
                            numbers_range = game_config['numbers_range']
                            valid_numbers = all(numbers_range[0] <= n <= numbers_range[1] for n in numbers)
                            main_numbers = numbers
                        
                        if valid_numbers:
                            # Try to find date in the text
                            draw_date = self.extract_date_from_text('\n'.join(relevant_lines))
                            
                            result = {
                                'game': game_name,
                                'numbers': main_numbers if main_numbers else numbers,
                                'bonus_number': bonus_number,
                                'date': draw_date,
                                'jackpot': None,
                                'source': 'PDF_OCR',
                                'raw_text': line.strip()
                            }
                            
                            results.append(result)
        
        return results
    
    def extract_date_from_text(self, text: str) -> Optional[datetime]:
        """
        Extract date from OCR text using various patterns
        
        Args:
            text: OCR-extracted text
            
        Returns:
            Parsed datetime object or None
        """
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    groups = match.groups()
                    
                    if len(groups) == 3:
                        if pattern.startswith(r'(\d{4})'):  # YYYY/MM/DD
                            year, month, day = groups
                            return datetime(int(year), int(month), int(day))
                        elif pattern.startswith(r'(\w+)'):  # Month DD, YYYY
                            month_name, day, year = groups
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
                        else:  # MM/DD/YYYY or MM/DD/YY
                            month, day, year = groups
                            if len(year) == 2:
                                year = int(year)
                                if year < 50:
                                    year += 2000
                                else:
                                    year += 1900
                            return datetime(int(year), int(month), int(day))
                            
                except ValueError:
                    continue
        
        return None
    
    def process_pdf_with_ocr(self, pdf_path, year: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a PDF file using OCR to extract lottery data
        
        Args:
            pdf_path: Path to PDF file (string or Path object)
            year: Year context for date parsing
            
        Returns:
            Dictionary with game names as keys and results as values
        """
        # Ensure pdf_path is a Path object
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)
        
        logger.info(f"🔍 Processing PDF with OCR: {pdf_path.name}")
        
        if not self.setup_tesseract():
            logger.error("❌ Tesseract OCR not available")
            return {}
        
        # Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)
        if not images:
            logger.error(f"❌ Could not convert {pdf_path.name} to images")
            return {}
        
        all_results = {}
        
        # Process each page
        for page_num, image in enumerate(images, 1):
            logger.info(f"📄 Processing page {page_num}/{len(images)}")
            
            # Preprocess image for better OCR
            preprocessed_image = self.preprocess_image_for_ocr(image)
            
            # Extract text using OCR
            ocr_text = self.extract_text_from_image(preprocessed_image)
            
            if ocr_text.strip():
                logger.info(f"📝 Extracted {len(ocr_text)} characters from page {page_num}")
                
                # Extract numbers for each supported game
                for game_name in self.game_patterns.keys():
                    results = self.extract_numbers_from_ocr_text(ocr_text, game_name)
                    
                    if results:
                        if game_name not in all_results:
                            all_results[game_name] = []
                        
                        # Set year context if provided
                        if year:
                            for result in results:
                                if result['date'] is None:
                                    result['date'] = datetime(year, 1, 1)
                                elif result['date'].year < 1950:  # OCR might get year wrong
                                    result['date'] = result['date'].replace(year=year)
                        
                        all_results[game_name].extend(results)
                        logger.info(f"🎯 Found {len(results)} {game_name} results on page {page_num}")
            else:
                logger.warning(f"⚠️ No text extracted from page {page_num}")
        
        return all_results
    
    def process_all_pdfs_with_ocr(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all PDF files using OCR
        
        Returns:
            Dictionary with all extracted lottery data
        """
        if not self.pdf_directory.exists():
            logger.error(f"❌ PDF directory not found: {self.pdf_directory}")
            return {}
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {self.pdf_directory}")
            return {}
        
        logger.info(f"📚 Processing {len(pdf_files)} PDF files with OCR...")
        
        all_data = {}
        
        for pdf_file in sorted(pdf_files):
            # Extract year from filename
            year = None
            try:
                year = int(pdf_file.stem)
            except ValueError:
                pass
            
            # Process PDF with OCR
            pdf_results = self.process_pdf_with_ocr(pdf_file, year)
            
            # Merge results
            for game_name, results in pdf_results.items():
                if game_name not in all_data:
                    all_data[game_name] = []
                all_data[game_name].extend(results)
        
        # Sort results by date
        for game_name in all_data:
            all_data[game_name].sort(key=lambda x: x['date'] if x['date'] else datetime.min)
        
        # Log summary
        total_results = sum(len(results) for results in all_data.values())
        logger.info(f"✅ OCR processing complete! Extracted {total_results} total results")
        
        return all_data


def test_ocr_reader():
    """
    Test the OCR PDF reader
    """
    print("🔍 Testing OCR PDF Reader for Lottery Charts")
    print("=" * 50)
    
    pdf_directory = r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games"
    
    # Initialize OCR reader
    reader = LotteryPDFOCRReader(pdf_directory)
    
    # Test single PDF
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"🧪 Testing OCR on: {test_pdf.name}")
        
        # Extract year
        try:
            year = int(test_pdf.stem)
        except ValueError:
            year = None
        
        # Process with OCR
        results = reader.process_pdf_with_ocr(test_pdf, year)
        
        if results:
            print("✅ OCR extraction successful!")
            for game_name, game_results in results.items():
                print(f"   📊 {game_name}: {len(game_results)} results")
                if game_results:
                    sample = game_results[0]
                    numbers_str = ', '.join(map(str, sample['numbers']))
                    print(f"      Sample: {numbers_str}")
        else:
            print("❌ No data extracted with OCR")
    
    return results if 'results' in locals() else {}


if __name__ == "__main__":
    test_ocr_reader()
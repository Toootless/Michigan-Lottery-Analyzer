"""
Test OCR Path Handling Fix
Tests that the OCR system properly handles both string and Path object inputs
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from data_collection.pdf_ocr_reader import LotteryPDFOCRReader
    print("OCR system imported successfully")
except ImportError as e:
    print(f"Error importing OCR system: {e}")
    sys.exit(1)

def test_path_handling():
    """Test that the OCR system handles both string and Path inputs"""
    
    pdf_dir = Path("past_games")
    if not pdf_dir.exists():
        print("past_games directory not found")
        return False
    
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    if not pdf_files:
        print("No PDF files found")
        return False
    
    test_file = pdf_files[0]  # Use first PDF (2000.pdf)
    print(f"Testing with: {test_file.name}")
    
    # Initialize OCR reader
    reader = LotteryPDFOCRReader(str(pdf_dir))
    
    try:
        # Test 1: Path object
        print("\nTest 1: Using Path object as input")
        result = reader.process_pdf_with_ocr(test_file, year=2000)
        
        if result:
            total_entries = sum(len(entries) for entries in result.values())
            print(f"SUCCESS: Extracted {total_entries} entries using Path object")
            return True
        else:
            print("FAILED: No results with Path object")
            
        # Test 2: String path  
        print("\nTest 2: Using string path as input")
        result = reader.process_pdf_with_ocr(str(test_file), year=2000)
        
        if result:
            total_entries = sum(len(entries) for entries in result.values())
            print(f"SUCCESS: Extracted {total_entries} entries using string path")
            return True
        else:
            print("FAILED: No results with string path")
            
    except Exception as e:
        print(f"ERROR during testing: {e}")
        print("Detailed error:")
        import traceback
        traceback.print_exc()
        return False
    
    return False

if __name__ == "__main__":
    print("=== OCR Path Handling Test ===")
    
    success = test_path_handling()
    
    if success:
        print("\n✅ Path handling fix is working!")
        print("✅ OCR system ready for batch processing")
    else:
        print("\n❌ Path handling issue persists")
        print("❌ Further debugging needed")
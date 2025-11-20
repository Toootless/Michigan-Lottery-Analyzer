"""
Simple PDF Reader Test - Direct Import
Tests PDF reading without complex module imports
"""

import sys
import os
from pathlib import Path

# Direct import of PDF reader class
sys.path.insert(0, os.path.dirname(__file__))

try:
    import PyPDF2
    print("✅ PyPDF2 available")
    PDF2_AVAILABLE = True
except ImportError:
    print("❌ PyPDF2 not available")
    PDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF available")
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("❌ PyMuPDF not available")
    PYMUPDF_AVAILABLE = False

def simple_pdf_test():
    """
    Simple test of PDF reading capabilities
    """
    print("\n🧪 Simple PDF Reading Test")
    print("=" * 40)
    
    # Check if we have PDF libraries
    if not PDF2_AVAILABLE and not PYMUPDF_AVAILABLE:
        print("❌ No PDF libraries available!")
        return
    
    # Test directory
    pdf_dir = r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games"
    
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF directory not found: {pdf_dir}")
        return
    
    # Find first PDF file
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    test_pdf = pdf_files[0]
    print(f"📄 Testing with: {test_pdf.name}")
    
    # Try to extract text
    text = ""
    
    if PYMUPDF_AVAILABLE:
        print("🔍 Using PyMuPDF to extract text...")
        try:
            import fitz
            doc = fitz.open(str(test_pdf))
            for page_num in range(min(3, len(doc))):  # Test first 3 pages
                page = doc[page_num]
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            doc.close()
            print(f"✅ Extracted {len(text)} characters using PyMuPDF")
        except Exception as e:
            print(f"❌ PyMuPDF failed: {e}")
    
    elif PDF2_AVAILABLE:
        print("🔍 Using PyPDF2 to extract text...")
        try:
            with open(test_pdf, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages[:3]):  # Test first 3 pages
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            print(f"✅ Extracted {len(text)} characters using PyPDF2")
        except Exception as e:
            print(f"❌ PyPDF2 failed: {e}")
    
    if text:
        # Show sample of extracted text
        print("\n📝 Sample extracted text:")
        print("-" * 40)
        sample_text = text[:500]  # First 500 characters
        print(sample_text)
        if len(text) > 500:
            print(f"... (showing first 500 of {len(text)} characters)")
        
        # Look for potential lottery numbers
        import re
        
        # Simple patterns to find numbers
        patterns = [
            r'\b\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\b',  # 5 numbers
            r'\b\d{4}\b',  # 4-digit number
            r'\b\d{3}\b',  # 3-digit number
        ]
        
        print(f"\n🔍 Looking for potential lottery numbers...")
        found_patterns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                found_patterns.extend(matches[:5])  # Limit to first 5 matches
        
        if found_patterns:
            print("🎯 Potential lottery numbers found:")
            for i, match in enumerate(found_patterns[:10], 1):
                print(f"  {i}. {match}")
        else:
            print("⚠️ No obvious lottery number patterns found")
        
        # Look for dates
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
        ]
        
        dates_found = []
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            dates_found.extend(dates[:5])
        
        if dates_found:
            print(f"\n📅 Potential dates found:")
            for i, date in enumerate(dates_found[:5], 1):
                print(f"  {i}. {date}")
        
        return True
    else:
        print("❌ No text extracted from PDF")
        return False

if __name__ == "__main__":
    print("🎰 Simple PDF Reader Test")
    print("=" * 30)
    
    success = simple_pdf_test()
    
    if success:
        print("\n✅ PDF reading test successful!")
        print("📝 Next steps:")
        print("  1. The PDF libraries are working")
        print("  2. Text extraction is functional")
        print("  3. Ready to implement full PDF processing")
    else:
        print("\n❌ PDF reading test failed")
        print("🔧 Troubleshooting needed")
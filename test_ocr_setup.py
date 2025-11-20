"""
Simple OCR Test for PDF Charts
Tests OCR capabilities without Tesseract dependency first
"""

import sys
import os
from pathlib import Path

def test_pdf_libraries():
    """Test if PDF libraries work"""
    print("🔍 Testing PDF Libraries")
    print("-" * 30)
    
    try:
        import fitz
        print("✅ PyMuPDF (fitz) available")
        
        # Test PDF reading
        pdf_dir = Path(r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"📄 Testing with: {test_pdf.name}")
            
            # Open PDF
            doc = fitz.open(test_pdf)
            print(f"📚 PDF has {len(doc)} pages")
            
            # Try to extract text from first page
            if len(doc) > 0:
                page = doc[0]
                text = page.get_text()
                print(f"📝 Text extraction: {len(text)} characters")
                
                if len(text) < 100:
                    print("⚠️ Very little text found - PDF likely contains images/charts")
                    print("🔍 Need OCR to process image-based content")
                    
                    # Try to get page as image
                    pix = page.get_pixmap()
                    print(f"🖼️ Page image: {pix.width}x{pix.height} pixels")
                    
                    # Save first page as image for testing
                    img_path = "test_page.png"
                    pix.save(img_path)
                    print(f"💾 Saved page image: {img_path}")
                    
                    return True, img_path
                else:
                    print("✅ Text-based PDF - can process with regular text extraction")
                    print(f"Sample text: {text[:200]}...")
            
            doc.close()
            
    except ImportError:
        print("❌ PyMuPDF not available")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None
    
    return True, None

def test_image_libraries():
    """Test image processing libraries"""
    print("\n🖼️ Testing Image Libraries")
    print("-" * 30)
    
    try:
        from PIL import Image
        import pdf2image
        print("✅ PIL and pdf2image available")
        
        # Test PDF to image conversion
        pdf_dir = Path(r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"📄 Converting PDF to images: {test_pdf.name}")
            
            # Convert first page only
            images = pdf2image.convert_from_path(
                test_pdf,
                first_page=1,
                last_page=1,
                dpi=150  # Lower DPI for testing
            )
            
            if images:
                img = images[0]
                print(f"🖼️ Converted to image: {img.size[0]}x{img.size[1]} pixels")
                
                # Save for inspection
                img.save("test_converted.png")
                print("💾 Saved converted image: test_converted.png")
                
                return True, img
            
    except ImportError as e:
        print(f"❌ Image libraries not available: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return False, None
    
    return True, None

def check_tesseract():
    """Check if Tesseract OCR is available"""
    print("\n🔤 Checking Tesseract OCR")
    print("-" * 30)
    
    try:
        import pytesseract
        print("✅ pytesseract library available")
        
        # Common Tesseract installation paths on Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            'tesseract'  # If in PATH
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path) or path == 'tesseract':
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Tesseract found at: {path}")
                    print(f"   Version: {version}")
                    return True
                except Exception:
                    continue
        
        print("❌ Tesseract OCR executable not found")
        print("📥 Please install Tesseract OCR:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install to default location")
        print("   3. Or add tesseract.exe to your PATH")
        
        return False
        
    except ImportError:
        print("❌ pytesseract library not available")
        return False

def main():
    """Main test function"""
    print("🎰 Michigan Lottery PDF Chart Reader Test")
    print("=" * 50)
    
    # Test PDF libraries
    pdf_ok, sample_img = test_pdf_libraries()
    
    # Test image libraries
    img_ok, converted_img = test_image_libraries()
    
    # Check Tesseract
    ocr_ok = check_tesseract()
    
    print("\n📋 Test Summary")
    print("-" * 20)
    print(f"PDF Reading: {'✅' if pdf_ok else '❌'}")
    print(f"Image Processing: {'✅' if img_ok else '❌'}")
    print(f"OCR (Tesseract): {'✅' if ocr_ok else '❌'}")
    
    if pdf_ok and img_ok and ocr_ok:
        print("\n🎉 All systems ready for PDF chart processing!")
        print("🚀 You can now process lottery PDF charts with OCR")
    elif pdf_ok and img_ok:
        print("\n⚠️ PDF and image processing ready, but OCR needs setup")
        print("📥 Install Tesseract OCR to process chart images")
    else:
        print("\n❌ Some components need setup")
    
    print(f"\n📁 Check your directory for test images:")
    if os.path.exists("test_page.png"):
        print("   • test_page.png (direct PDF page)")
    if os.path.exists("test_converted.png"):
        print("   • test_converted.png (PDF to image conversion)")

if __name__ == "__main__":
    main()
"""
Test Multiple PDFs - Find the best ones to work with
"""

import sys
import os
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PDF2_AVAILABLE = True
except ImportError:
    PDF2_AVAILABLE = False

def analyze_all_pdfs():
    """
    Analyze all PDF files to see which ones have readable content
    """
    print("🔍 Analyzing All PDF Files")
    print("=" * 40)
    
    pdf_dir = r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games"
    
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF directory not found: {pdf_dir}")
        return
    
    pdf_files = sorted(list(Path(pdf_dir).glob("*.pdf")))
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    results = []
    
    for pdf_file in pdf_files:
        print(f"\n📄 Testing: {pdf_file.name}")
        
        # Extract text
        text = ""
        error = None
        
        if PYMUPDF_AVAILABLE:
            try:
                import fitz
                doc = fitz.open(str(pdf_file))
                
                # Get basic info
                page_count = len(doc)
                
                # Extract text from first few pages
                for page_num in range(min(5, page_count)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    text += page_text + "\n"
                
                doc.close()
                
            except Exception as e:
                error = str(e)
        
        # Analyze the extracted text
        text_length = len(text.strip())
        
        # Count potential lottery numbers
        import re
        
        # Look for various number patterns
        patterns = {
            'five_numbers': r'\b\d{1,2}[-\s]+\d{1,2}[-\s]+\d{1,2}[-\s]+\d{1,2}[-\s]+\d{1,2}\b',
            'four_digits': r'\b\d{4}\b',
            'three_digits': r'\b\d{3}\b',
            'dates': r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            'any_numbers': r'\b\d+\b'
        }
        
        pattern_counts = {}
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            pattern_counts[pattern_name] = len(matches)
        
        # Quality assessment
        quality = "Poor"
        if text_length > 1000:
            quality = "Excellent"
        elif text_length > 500:
            quality = "Good"
        elif text_length > 100:
            quality = "Fair"
        
        result = {
            'file': pdf_file.name,
            'text_length': text_length,
            'quality': quality,
            'pattern_counts': pattern_counts,
            'error': error,
            'sample_text': text[:200] if text else ""
        }
        
        results.append(result)
        
        # Print quick summary
        status = "✅" if text_length > 100 else "⚠️" if text_length > 10 else "❌"
        print(f"  {status} {text_length} chars, Quality: {quality}")
        
        if pattern_counts['five_numbers'] > 0:
            print(f"  🎯 Found {pattern_counts['five_numbers']} potential 5-number sequences")
        if pattern_counts['dates'] > 0:
            print(f"  📅 Found {pattern_counts['dates']} potential dates")
    
    # Summary report
    print("\n" + "="*50)
    print("📊 SUMMARY REPORT")
    print("="*50)
    
    # Sort by text length (best first)
    results.sort(key=lambda x: x['text_length'], reverse=True)
    
    print(f"\n🏆 TOP 5 BEST PDFs FOR PROCESSING:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['file']} - {result['text_length']} chars ({result['quality']})")
        if result['pattern_counts']['five_numbers'] > 0:
            print(f"   🎯 {result['pattern_counts']['five_numbers']} five-number patterns")
        if result['pattern_counts']['dates'] > 0:
            print(f"   📅 {result['pattern_counts']['dates']} date patterns")
    
    print(f"\n📈 QUALITY DISTRIBUTION:")
    quality_counts = {}
    for result in results:
        quality = result['quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} files")
    
    # Show sample from best file
    if results and results[0]['text_length'] > 100:
        print(f"\n📝 SAMPLE FROM BEST FILE ({results[0]['file']}):")
        print("-" * 40)
        print(results[0]['sample_text'])
        print("...")
    
    return results

if __name__ == "__main__":
    print("🎰 PDF Analysis Tool")
    print("=" * 30)
    
    if not PYMUPDF_AVAILABLE and not PDF2_AVAILABLE:
        print("❌ No PDF libraries available!")
    else:
        results = analyze_all_pdfs()
        
        if results:
            good_files = [r for r in results if r['text_length'] > 100]
            print(f"\n✅ Analysis complete!")
            print(f"📊 {len(good_files)} out of {len(results)} files have good text content")
            
            if good_files:
                print("🚀 Ready to proceed with full PDF processing using the best files!")
            else:
                print("⚠️ Limited text content found. PDFs may be image-based.")
        else:
            print("❌ No results to analyze")
"""
Test Script for PDF Reader Module
Tests the PDF reading functionality with Michigan Lottery historical data

This script:
1. Tests PDF reading capabilities
2. Extracts lottery numbers from PDF files
3. Generates summary reports
4. Saves data to JSON format

Author: Michigan Lottery Analyzer Team
Date: October 2025
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection.pdf_reader import LotteryPDFReader, install_pdf_dependencies

def test_pdf_reader():
    """
    Test the PDF reader functionality
    """
    print("🔍 Testing Michigan Lottery PDF Reader")
    print("=" * 50)
    
    # Check if PDF directory exists
    pdf_directory = r"C:\Users\johnj\OneDrive\Documents\___Online classes\__MEAGSAdvancedGenerativeAIAgenticFrameworks\Demos\Lottery_Analyzer\past_games"
    
    if not os.path.exists(pdf_directory):
        print(f"❌ PDF directory not found: {pdf_directory}")
        return
    
    print(f"📁 PDF Directory: {pdf_directory}")
    
    # List available PDF files
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    print(f"📚 Found {len(pdf_files)} PDF files:")
    for pdf_file in sorted(pdf_files)[:5]:  # Show first 5
        print(f"   • {pdf_file.name}")
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more files")
    print()
    
    # Initialize PDF reader
    print("🚀 Initializing PDF Reader...")
    reader = LotteryPDFReader(pdf_directory)
    
    # Test with a single PDF first
    if pdf_files:
        test_pdf = pdf_files[0]  # Test with first PDF
        print(f"🧪 Testing with single PDF: {test_pdf.name}")
        
        # Extract year from filename
        try:
            year = int(test_pdf.stem)
            print(f"📅 Detected year: {year}")
        except ValueError:
            year = None
            print("⚠️ Could not detect year from filename")
        
        # Process single PDF
        single_result = reader.process_pdf_file(test_pdf, year)
        
        if single_result:
            print("✅ Single PDF test successful!")
            for game_name, results in single_result.items():
                print(f"   📊 {game_name}: {len(results)} results")
                if results:
                    sample = results[0]
                    numbers_str = ', '.join(map(str, sample['numbers']))
                    print(f"      Sample: {numbers_str}")
                    if sample['bonus_number']:
                        print(f"      Bonus: {sample['bonus_number']}")
                    if sample['date']:
                        print(f"      Date: {sample['date'].strftime('%Y-%m-%d')}")
        else:
            print("❌ Single PDF test failed - no data extracted")
        
        print()
    
    # Ask user if they want to process all PDFs
    response = input("🤔 Do you want to process ALL PDF files? This may take a while. (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🚀 Processing all PDF files...")
        all_data = reader.process_all_pdfs()
        
        if all_data:
            print("✅ All PDFs processed successfully!")
            
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Save to JSON
            json_path = data_dir / "historical_lottery_data.json"
            reader.save_to_json(all_data, str(json_path))
            
            # Generate and save summary
            summary = reader.generate_summary_report(all_data)
            print("\n" + summary)
            
            summary_path = data_dir / "pdf_extraction_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print(f"\n💾 Files saved:")
            print(f"   📄 Data: {json_path}")
            print(f"   📊 Summary: {summary_path}")
            
            # Integration suggestions
            print("\n🔗 Integration with Main Application:")
            print("   1. The extracted data is saved in JSON format")
            print("   2. You can load this data in your main application")
            print("   3. Use it to supplement web scraping for historical data")
            print("   4. Data covers years 2000-2021 from PDF files")
            
        else:
            print("❌ Failed to extract data from PDF files")
    else:
        print("👍 Skipping full PDF processing")
    
    print("\n✅ PDF Reader test complete!")

def show_integration_example():
    """
    Show how to integrate PDF data with main application
    """
    print("\n" + "="*50)
    print("🔗 INTEGRATION EXAMPLE")
    print("="*50)
    
    integration_code = '''
# In your main MichiganLotteryAnalyzer.py, add this function:

def load_historical_pdf_data(game_name: str) -> List[Dict[str, Any]]:
    """
    Load historical lottery data from PDF extraction
    
    Args:
        game_name: Name of lottery game
        
    Returns:
        List of historical results
    """
    import json
    from datetime import datetime
    
    try:
        with open("data/historical_lottery_data.json", 'r') as f:
            pdf_data = json.load(f)
        
        if game_name in pdf_data:
            results = []
            for entry in pdf_data[game_name]:
                result = {
                    'numbers': entry['numbers'],
                    'bonus_number': entry.get('bonus_number'),
                    'date': datetime.fromisoformat(entry['date']) if entry['date'] else None,
                    'jackpot': entry.get('jackpot'),
                    'source': 'PDF'
                }
                results.append(result)
            return results
    
    except FileNotFoundError:
        print("📄 Historical PDF data not found. Run PDF extraction first.")
    except Exception as e:
        print(f"❌ Error loading PDF data: {e}")
    
    return []

# Modified get_lottery_data function:
def get_lottery_data(game_name: str, days: int) -> List[Dict[str, Any]]:
    """
    Get lottery data from multiple sources (web scraping + PDF + sample)
    """
    # First try to load PDF historical data (pre-2022)
    historical_data = load_historical_pdf_data(game_name)
    
    # Then try web scraping for recent data (2022+)
    recent_data = scrape_recent_data(game_name)
    
    # Combine and filter by date range
    all_data = historical_data + recent_data
    
    # Sort by date and limit to requested days
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_data = [d for d in all_data if d['date'] and d['date'] >= cutoff_date]
    filtered_data.sort(key=lambda x: x['date'], reverse=True)
    
    return filtered_data[:days] if filtered_data else generate_sample_data(game_name, days)
'''
    
    print(integration_code)

if __name__ == "__main__":
    print("🎰 Michigan Lottery PDF Reader Test")
    print("===================================")
    
    # Check dependencies
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
    except ImportError:
        print("❌ PyPDF2 not available")
    
    try:
        import fitz
        print("✅ PyMuPDF available")
    except ImportError:
        print("❌ PyMuPDF not available")
    
    print()
    
    # Run test
    test_pdf_reader()
    
    # Show integration example
    show_integration_example()
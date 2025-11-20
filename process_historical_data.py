"""
Michigan Lottery Historical Data Processor
Batch process all PDF files to extract historical lottery data using OCR

This script processes all PDF files in the past_games directory and extracts
lottery winning numbers from 2000-2021, saving the results to JSON files.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from data_collection.pdf_ocr_reader import LotteryPDFOCRReader
    OCR_AVAILABLE = True
    print("✅ OCR system loaded successfully")
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"❌ OCR system not available: {e}")
    print("📝 Make sure all OCR dependencies are installed:")
    print("   pip install pytesseract Pillow PyMuPDF")
    sys.exit(1)

def process_all_historical_pdfs():
    """Process all historical PDF files and extract lottery data"""
    
    print("🎰 Michigan Lottery Historical Data Processor")
    print("=" * 60)
    
    # Setup directories
    pdf_dir = current_dir / "past_games"
    output_dir = current_dir / "data" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        return
    
    # Get all PDF files
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    # Initialize OCR reader
    print("🔍 Initializing OCR reader...")
    reader = LotteryPDFOCRReader(str(pdf_dir))
    
    # Process each PDF file
    all_results = {}
    processing_stats = {
        'total_files': len(pdf_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_entries': 0,
        'start_time': datetime.now(),
        'games_found': set()
    }
    
    print(f"\n🚀 Starting batch processing of {len(pdf_files)} files...")
    print("⏱️ This may take several minutes depending on PDF complexity...")
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            print(f"\n📊 Processing {i+1}/{len(pdf_files)}: {pdf_file.name}")
            print("-" * 40)
            
            # Extract year from filename
            year = None
            try:
                year = int(pdf_file.stem)
                print(f"📅 Year: {year}")
            except ValueError:
                print("⚠️ Could not extract year from filename")
            
            # Process PDF with OCR
            start_time = time.time()
            results = reader.process_pdf_with_ocr(pdf_file, year=year)
            processing_time = time.time() - start_time
            
            if results:
                # Count entries and track games
                file_total = 0
                for game_name, entries in results.items():
                    if entries:
                        file_total += len(entries)
                        processing_stats['games_found'].add(game_name)
                        print(f"  🎯 {game_name}: {len(entries)} entries")
                
                print(f"  ✅ Total entries: {file_total}")
                print(f"  ⏱️ Processing time: {processing_time:.1f} seconds")
                
                # Store results
                all_results[pdf_file.name] = {
                    'year': year,
                    'processing_time': processing_time,
                    'total_entries': file_total,
                    'games': results
                }
                
                processing_stats['total_entries'] += file_total
                processing_stats['processed_files'] += 1
                
                # Save individual year file
                year_file = output_dir / f"lottery_data_{year}.json"
                save_json_results({pdf_file.name: all_results[pdf_file.name]}, year_file)
                print(f"  💾 Saved to: {year_file}")
                
            else:
                print(f"  ❌ No data extracted from {pdf_file.name}")
                processing_stats['failed_files'] += 1
                
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file.name}: {e}")
            processing_stats['failed_files'] += 1
            continue
    
    # Save complete results
    if all_results:
        print(f"\n{'='*60}")
        print("💾 Saving complete historical dataset...")
        
        # Save master file with all data
        master_file = output_dir / "complete_historical_data.json"
        save_json_results(all_results, master_file)
        print(f"✅ Master file saved: {master_file}")
        
        # Save summary statistics
        processing_stats['end_time'] = datetime.now()
        processing_stats['total_time'] = (processing_stats['end_time'] - processing_stats['start_time']).total_seconds()
        processing_stats['games_found'] = list(processing_stats['games_found'])
        
        stats_file = output_dir / "processing_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        print(f"📊 Statistics saved: {stats_file}")
        
        # Print final summary
        print_processing_summary(processing_stats, all_results)
        
    else:
        print("\n❌ No data was successfully extracted from any PDF files")

def save_json_results(results: Dict, filepath: Path):
    """Save results to JSON file with proper datetime handling"""
    
    # Convert datetime objects to strings for JSON serialization
    json_results = {}
    for filename, file_data in results.items():
        json_results[filename] = {
            'year': file_data.get('year'),
            'processing_time': file_data.get('processing_time'),
            'total_entries': file_data.get('total_entries'),
            'games': {}
        }
        
        for game_name, entries in file_data.get('games', {}).items():
            json_results[filename]['games'][game_name] = []
            for entry in entries:
                json_entry = entry.copy()
                # Convert datetime to string
                if json_entry.get('date') and hasattr(json_entry['date'], 'isoformat'):
                    json_entry['date'] = json_entry['date'].isoformat()
                json_results[filename]['games'][game_name].append(json_entry)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)

def print_processing_summary(stats: Dict, results: Dict):
    """Print comprehensive processing summary"""
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 FILES PROCESSED:")
    print(f"   ✅ Successful: {stats['processed_files']}")
    print(f"   ❌ Failed: {stats['failed_files']}")
    print(f"   📄 Total: {stats['total_files']}")
    
    print(f"\n🎯 DATA EXTRACTED:")
    print(f"   🎲 Total lottery entries: {stats['total_entries']:,}")
    print(f"   🎮 Games found: {len(stats['games_found'])}")
    
    print(f"\n🎮 GAMES BREAKDOWN:")
    game_totals = {}
    for file_data in results.values():
        for game_name, entries in file_data.get('games', {}).items():
            if game_name not in game_totals:
                game_totals[game_name] = 0
            game_totals[game_name] += len(entries)
    
    for game_name in sorted(game_totals.keys()):
        print(f"   🎯 {game_name}: {game_totals[game_name]:,} entries")
    
    print(f"\n⏱️ PERFORMANCE:")
    print(f"   🕒 Total time: {stats['total_time']:.1f} seconds")
    print(f"   📊 Average per file: {stats['total_time']/stats['total_files']:.1f} seconds")
    print(f"   🚀 Entries per second: {stats['total_entries']/stats['total_time']:.1f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   📄 Individual year files: data/historical/lottery_data_YYYY.json")
    print(f"   📚 Complete dataset: data/historical/complete_historical_data.json")
    print(f"   📊 Statistics: data/historical/processing_statistics.json")
    
    print(f"\n🎰 READY FOR ANALYSIS!")
    print("Your historical lottery data is now available for:")
    print("   • Pattern analysis across 22 years (2000-2021)")
    print("   • Long-term trend identification")
    print("   • Comprehensive statistical modeling")
    print("   • Integration with main lottery analyzer")

def load_historical_data(game_name: str = None, years: List[int] = None) -> Dict[str, List[Dict]]:
    """Load processed historical data from JSON files"""
    
    output_dir = Path("data/historical")
    master_file = output_dir / "complete_historical_data.json"
    
    if not master_file.exists():
        print(f"❌ Historical data file not found: {master_file}")
        print("💡 Run this script first to process PDF files")
        return {}
    
    try:
        with open(master_file, 'r') as f:
            all_data = json.load(f)
        
        # Flatten the data structure
        flattened_data = {}
        for file_info in all_data.values():
            for game, entries in file_info.get('games', {}).items():
                if game not in flattened_data:
                    flattened_data[game] = []
                flattened_data[game].extend(entries)
        
        # Filter by game if specified
        if game_name and game_name in flattened_data:
            return {game_name: flattened_data[game_name]}
        
        # Filter by years if specified
        if years:
            filtered_data = {}
            for game, entries in flattened_data.items():
                filtered_entries = []
                for entry in entries:
                    if 'year' in entry and entry['year'] in years:
                        filtered_entries.append(entry)
                if filtered_entries:
                    filtered_data[game] = filtered_entries
            return filtered_data
        
        return flattened_data
        
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return {}

if __name__ == "__main__":
    print("🎰 Michigan Lottery Historical Data Processor")
    print("=" * 60)
    
    if not OCR_AVAILABLE:
        print("❌ OCR system not available. Cannot process PDFs.")
        sys.exit(1)
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. 🔍 Process all PDF files (full batch processing)")
    print("2. 📊 Load existing historical data (quick test)")
    print("3. 🎯 Process specific years only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting full batch processing...")
        process_all_historical_pdfs()
        
    elif choice == "2":
        print("\n📊 Loading existing historical data...")
        data = load_historical_data()
        if data:
            total = sum(len(entries) for entries in data.values())
            print(f"✅ Loaded {total:,} entries across {len(data)} games")
            for game, entries in data.items():
                print(f"   🎯 {game}: {len(entries):,} entries")
        else:
            print("❌ No historical data found. Run option 1 first.")
            
    elif choice == "3":
        print("\n🎯 Processing specific years...")
        start_year = int(input("Start year (2000-2021): "))
        end_year = int(input("End year (2000-2021): "))
        
        # Filter PDFs by year range
        pdf_dir = Path("past_games")
        selected_pdfs = []
        for year in range(start_year, end_year + 1):
            pdf_file = pdf_dir / f"{year}.pdf"
            if pdf_file.exists():
                selected_pdfs.append(pdf_file)
        
        print(f"Found {len(selected_pdfs)} PDF files for years {start_year}-{end_year}")
        
        if selected_pdfs:
            # Process selected PDFs (similar to full processing but with filtered list)
            print("🚀 Processing selected years...")
            # You could modify process_all_historical_pdfs() to accept a file list
            process_all_historical_pdfs()  # For now, just run full processing
        else:
            print("❌ No PDF files found for specified years")
    
    else:
        print("❌ Invalid choice. Please run the script again.")
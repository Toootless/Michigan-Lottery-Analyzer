#!/usr/bin/env python3
"""
Historical Lottery Data Extractor
Extract historical lottery numbers from PDF files using OCR and save to CSV files
"""

import sys
import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

from data_collection.pdf_ocr_reader import LotteryPDFOCRReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalLotteryExtractor:
    """Extract historical lottery data from PDFs and save to CSV"""
    
    def __init__(self, pdf_directory: str, output_directory: str):
        """
        Initialize the extractor
        
        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Directory to save CSV files
        """
        self.pdf_directory = Path(pdf_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize OCR reader
        self.ocr_reader = LotteryPDFOCRReader(str(self.pdf_directory))
        
        # Games to extract
        self.target_games = [
            'Powerball',
            'Mega Millions', 
            'Lucky for Life',
            'Fantasy 5',
            'Lotto 47',
            'Daily 3',
            'Daily 4',
            'Keno'
        ]
        
        # CSV columns for different game types
        self.csv_columns = {
            'standard': ['Date', 'Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'BonusNumber', 'Source'],
            'daily_3': ['Date', 'Number1', 'Number2', 'Number3', 'Source'],
            'daily_4': ['Date', 'Number1', 'Number2', 'Number3', 'Number4', 'Source'],
            'lotto_47': ['Date', 'Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Source'],
            'keno': ['Date', 'Numbers', 'Source']  # Keno has 20 numbers - store as string
        }
    
    def get_csv_columns_for_game(self, game_name: str) -> list:
        """Get appropriate CSV columns for a game"""
        if game_name in ['Daily 3']:
            return self.csv_columns['daily_3']
        elif game_name in ['Daily 4']:
            return self.csv_columns['daily_4']
        elif game_name in ['Lotto 47']:
            return self.csv_columns['lotto_47']
        elif game_name == 'Keno':
            return self.csv_columns['keno']
        else:
            return self.csv_columns['standard']
    
    def convert_result_to_csv_row(self, result: dict, game_name: str) -> dict:
        """Convert OCR result to CSV row format"""
        numbers = result.get('numbers', [])
        bonus_number = result.get('bonus_number')
        date = result.get('date')
        source = f"PDF_OCR_{result.get('source', 'Unknown')}"
        
        # Format date
        date_str = ""
        if date:
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)
        
        if game_name in ['Daily 3']:
            return {
                'Date': date_str,
                'Number1': numbers[0] if len(numbers) > 0 else '',
                'Number2': numbers[1] if len(numbers) > 1 else '',
                'Number3': numbers[2] if len(numbers) > 2 else '',
                'Source': source
            }
        elif game_name in ['Daily 4']:
            return {
                'Date': date_str,
                'Number1': numbers[0] if len(numbers) > 0 else '',
                'Number2': numbers[1] if len(numbers) > 1 else '',
                'Number3': numbers[2] if len(numbers) > 2 else '',
                'Number4': numbers[3] if len(numbers) > 3 else '',
                'Source': source
            }
        elif game_name in ['Lotto 47']:
            return {
                'Date': date_str,
                'Number1': numbers[0] if len(numbers) > 0 else '',
                'Number2': numbers[1] if len(numbers) > 1 else '',
                'Number3': numbers[2] if len(numbers) > 2 else '',
                'Number4': numbers[3] if len(numbers) > 3 else '',
                'Number5': numbers[4] if len(numbers) > 4 else '',
                'Number6': numbers[5] if len(numbers) > 5 else '',
                'Source': source
            }
        elif game_name == 'Keno':
            numbers_str = ','.join(map(str, numbers)) if numbers else ''
            return {
                'Date': date_str,
                'Numbers': numbers_str,
                'Source': source
            }
        else:
            # Standard format (Powerball, Mega Millions, Lucky for Life, Fantasy 5)
            return {
                'Date': date_str,
                'Number1': numbers[0] if len(numbers) > 0 else '',
                'Number2': numbers[1] if len(numbers) > 1 else '',
                'Number3': numbers[2] if len(numbers) > 2 else '',
                'Number4': numbers[3] if len(numbers) > 3 else '',
                'Number5': numbers[4] if len(numbers) > 4 else '',
                'BonusNumber': bonus_number if bonus_number is not None else '',
                'Source': source
            }
    
    def process_pdf_file(self, pdf_path: Path, year: int = None) -> dict:
        """
        Process a single PDF file and extract lottery data
        
        Args:
            pdf_path: Path to PDF file
            year: Year for context (optional)
            
        Returns:
            Dictionary with game names as keys and lists of results as values
        """
        logger.info(f"🔍 Processing PDF: {pdf_path.name}")
        
        try:
            # Use OCR reader to process the PDF
            results = self.ocr_reader.process_pdf_with_ocr(pdf_path, year)
            
            if results:
                total_records = sum(len(records) for records in results.values())
                logger.info(f"✅ Extracted {total_records} records from {pdf_path.name}")
                for game, records in results.items():
                    if records:
                        logger.info(f"   - {game}: {len(records)} records")
            else:
                logger.warning(f"❌ No data extracted from {pdf_path.name}")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            return {}
    
    def save_to_csv(self, game_data: dict, game_name: str) -> str:
        """
        Save game data to CSV file
        
        Args:
            game_data: List of lottery results for the game
            game_name: Name of the lottery game
            
        Returns:
            Path to saved CSV file
        """
        if not game_data:
            logger.warning(f"No data to save for {game_name}")
            return ""
        
        # Generate filename
        filename = f"{game_name.replace(' ', '_')}_Historical_OCR.csv"
        csv_path = self.output_directory / filename
        
        # Get appropriate columns
        columns = self.get_csv_columns_for_game(game_name)
        
        try:
            # Convert results to CSV rows
            csv_rows = []
            for result in game_data:
                row = self.convert_result_to_csv_row(result, game_name)
                csv_rows.append(row)
            
            # Sort by date if available
            csv_rows.sort(key=lambda x: x.get('Date', ''), reverse=True)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            logger.info(f"💾 Saved {len(csv_rows)} {game_name} records to {filename}")
            return str(csv_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving {game_name} to CSV: {e}")
            return ""
    
    def process_all_pdfs(self, start_year: int = 2015, end_year: int = 2021, max_files: int = 5):
        """
        Process all PDF files and extract historical data
        
        Args:
            start_year: Start year for processing
            end_year: End year for processing  
            max_files: Maximum number of files to process (for testing)
        """
        logger.info(f"🚀 Starting historical data extraction ({start_year}-{end_year})")
        
        # Get PDF files in year range
        pdf_files = []
        for year in range(start_year, end_year + 1):
            pdf_path = self.pdf_directory / f"{year}.pdf"
            if pdf_path.exists():
                pdf_files.append((pdf_path, year))
        
        if not pdf_files:
            logger.error(f"❌ No PDF files found for years {start_year}-{end_year}")
            return
        
        # Limit files for testing
        pdf_files = pdf_files[:max_files]
        logger.info(f"📄 Processing {len(pdf_files)} PDF files")
        
        # Aggregate data by game
        all_game_data = {game: [] for game in self.target_games}
        
        # Process each PDF
        for i, (pdf_path, year) in enumerate(pdf_files):
            logger.info(f"📖 Processing file {i+1}/{len(pdf_files)}: {pdf_path.name}")
            
            # Extract data from PDF
            pdf_results = self.process_pdf_file(pdf_path, year)
            
            # Aggregate results by game
            for game_name in self.target_games:
                if game_name in pdf_results and pdf_results[game_name]:
                    all_game_data[game_name].extend(pdf_results[game_name])
        
        # Save aggregated data to CSV files
        logger.info("💾 Saving extracted data to CSV files...")
        saved_files = []
        
        for game_name, game_data in all_game_data.items():
            if game_data:
                csv_path = self.save_to_csv(game_data, game_name)
                if csv_path:
                    saved_files.append(csv_path)
        
        # Summary
        logger.info("🎯 Extraction Summary:")
        logger.info(f"   📄 PDFs processed: {len(pdf_files)}")
        logger.info(f"   💾 CSV files created: {len(saved_files)}")
        
        for game_name, game_data in all_game_data.items():
            if game_data:
                logger.info(f"   - {game_name}: {len(game_data)} records")
        
        if saved_files:
            logger.info("📁 Saved CSV files:")
            for filepath in saved_files:
                logger.info(f"   - {Path(filepath).name}")
        
        return all_game_data

def main():
    """Main function to run the historical data extraction"""
    
    # Paths
    pdf_dir = "past_games"
    output_dir = "historical_extracted"
    
    # Create extractor
    extractor = HistoricalLotteryExtractor(pdf_dir, output_dir)
    
    print("🎲 Historical Lottery Data Extractor")
    print("=====================================")
    print(f"📁 PDF Directory: {pdf_dir}")
    print(f"💾 Output Directory: {output_dir}")
    print()
    
    # Test with a small number of files first
    print("🧪 Testing with 3 recent PDF files...")
    results = extractor.process_all_pdfs(start_year=2019, end_year=2021, max_files=3)
    
    print("\n✅ Historical data extraction completed!")
    print(f"📊 Check the '{output_dir}' directory for CSV files with extracted data.")

if __name__ == "__main__":
    main()
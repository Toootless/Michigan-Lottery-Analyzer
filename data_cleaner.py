#!/usr/bin/env python3
"""
Data Cleaner and Validator for Historical Lottery Data
Clean and validate OCR-extracted lottery data and merge with existing CSV data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LotteryDataCleaner:
    """Clean and validate lottery data extracted from OCR"""
    
    def __init__(self, ocr_directory: str, existing_directory: str, output_directory: str):
        """
        Initialize the data cleaner
        
        Args:
            ocr_directory: Directory containing OCR-extracted CSV files
            existing_directory: Directory containing existing CSV files
            output_directory: Directory to save cleaned/merged CSV files
        """
        self.ocr_directory = Path(ocr_directory)
        self.existing_directory = Path(existing_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Game validation rules
        self.game_rules = {
            'Powerball': {
                'main_count': 5,
                'main_range': (1, 69),
                'bonus_range': (1, 26),
                'has_bonus': True
            },
            'Mega_Millions': {
                'main_count': 5,
                'main_range': (1, 70),
                'bonus_range': (1, 25),
                'has_bonus': True
            },
            'Lucky_for_Life': {
                'main_count': 5,
                'main_range': (1, 48),
                'bonus_range': (1, 18),
                'has_bonus': True
            },
            'Fantasy_5': {
                'main_count': 5,
                'main_range': (1, 39),
                'bonus_range': None,
                'has_bonus': False
            },
            'Lotto_47': {
                'main_count': 6,
                'main_range': (1, 47),
                'bonus_range': None,
                'has_bonus': False
            },
            'Daily_3': {
                'main_count': 3,
                'main_range': (0, 9),
                'bonus_range': None,
                'has_bonus': False
            },
            'Daily_4': {
                'main_count': 4,
                'main_range': (0, 9),
                'bonus_range': None,
                'has_bonus': False
            }
        }
    
    def clean_date(self, date_str: str, year_context: int = None) -> str:
        """Clean and validate date string"""
        if pd.isna(date_str) or not date_str:
            return ""
        
        try:
            # Handle common OCR date errors
            date_str = str(date_str).strip()
            
            # Fix obvious OCR errors like 8818 -> 2018
            if date_str.startswith('8818'):
                date_str = date_str.replace('8818', '2018', 1)
            elif date_str.startswith('8819'):
                date_str = date_str.replace('8819', '2019', 1)
            elif date_str.startswith('8820'):
                date_str = date_str.replace('8820', '2020', 1)
            elif date_str.startswith('8821'):
                date_str = date_str.replace('8821', '2021', 1)
            
            # Try to parse the date
            parsed_date = pd.to_datetime(date_str, errors='coerce')
            
            if pd.isna(parsed_date):
                return ""
            
            # Validate year range (lottery data should be recent)
            if parsed_date.year < 1990 or parsed_date.year > 2025:
                if year_context:
                    # Use year context if provided
                    parsed_date = parsed_date.replace(year=year_context)
                else:
                    return ""
            
            return parsed_date.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.warning(f"Date cleaning error for '{date_str}': {e}")
            return ""
    
    def validate_numbers(self, row: pd.Series, game_name: str) -> bool:
        """Validate lottery numbers for a specific game"""
        if game_name not in self.game_rules:
            return True  # Unknown game, assume valid
        
        rules = self.game_rules[game_name]
        
        # Get main numbers
        main_numbers = []
        for i in range(1, rules['main_count'] + 1):
            col_name = f'Number{i}'
            if col_name in row and pd.notna(row[col_name]):
                try:
                    num = int(row[col_name])
                    main_numbers.append(num)
                except (ValueError, TypeError):
                    return False
        
        # Check main numbers count
        if len(main_numbers) != rules['main_count']:
            return False
        
        # Check main numbers range
        main_min, main_max = rules['main_range']
        if not all(main_min <= num <= main_max for num in main_numbers):
            return False
        
        # Check bonus number if applicable
        if rules['has_bonus'] and 'BonusNumber' in row:
            if pd.notna(row['BonusNumber']):
                try:
                    bonus = int(row['BonusNumber'])
                    bonus_min, bonus_max = rules['bonus_range']
                    if not (bonus_min <= bonus <= bonus_max):
                        return False
                except (ValueError, TypeError):
                    return False
        
        return True
    
    def clean_ocr_file(self, file_path: Path, game_name: str) -> pd.DataFrame:
        """Clean a single OCR CSV file"""
        logger.info(f"🧹 Cleaning {file_path.name} for {game_name}")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            original_count = len(df)
            
            # Clean dates
            df['Date'] = df['Date'].apply(self.clean_date)
            
            # Remove rows with invalid dates
            df = df[df['Date'] != ""]
            
            # Clean source field
            df['Source'] = df['Source'].str.replace('PDF_OCR_PDF_OCR', 'PDF_OCR', regex=False)
            
            # Validate numbers
            valid_mask = df.apply(lambda row: self.validate_numbers(row, game_name), axis=1)
            df = df[valid_mask]
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Sort by date
            df = df.sort_values('Date', ascending=False)
            
            cleaned_count = len(df)
            logger.info(f"   ✅ Cleaned {file_path.name}: {original_count} → {cleaned_count} records")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cleaning {file_path.name}: {e}")
            return pd.DataFrame()
    
    def merge_with_existing(self, cleaned_df: pd.DataFrame, game_name: str) -> pd.DataFrame:
        """Merge cleaned OCR data with existing CSV data"""
        
        # Map game names to existing file names
        existing_file_map = {
            'Powerball': 'Powerball numbers from LotteryUSA.csv',
            'Mega_Millions': 'Mega Millions numbers from LotteryUSA.csv',
            'Lucky_for_Life': 'Lucky for Life numbers from LotteryUSA.csv',
            'Daily_3': 'MI Daily 3 Evening numbers from LotteryUSA.csv',
            'Daily_4': 'MI Daily 4 Evening numbers from LotteryUSA.csv',
            'Lotto_47': 'MI Lotto 47 numbers from LotteryUSA.csv'
        }
        
        existing_file = existing_file_map.get(game_name)
        if not existing_file:
            logger.info(f"   📁 No existing file mapping for {game_name}")
            return cleaned_df
        
        existing_path = self.existing_directory / existing_file
        if not existing_path.exists():
            logger.info(f"   📁 No existing file found: {existing_file}")
            return cleaned_df
        
        try:
            # Read existing data
            existing_df = pd.read_csv(existing_path)
            logger.info(f"   📊 Found existing data: {len(existing_df)} records")
            
            # Standardize column names and formats
            if 'Date' in existing_df.columns:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                existing_df = existing_df[existing_df['Date'].notna()]
            
            # Add source column if missing
            if 'Source' not in existing_df.columns:
                existing_df['Source'] = 'LotteryUSA'
            
            # Combine datasets
            combined_df = pd.concat([existing_df, cleaned_df], ignore_index=True)
            
            # Remove duplicates (keeping the first occurrence)
            combined_df = combined_df.drop_duplicates(subset=['Date'] + [col for col in combined_df.columns if col.startswith('Number')])
            
            # Sort by date
            combined_df = combined_df.sort_values('Date', ascending=False)
            
            logger.info(f"   🔄 Merged data: {len(combined_df)} total records")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error merging with existing data: {e}")
            return cleaned_df
    
    def process_all_files(self):
        """Process all OCR files"""
        logger.info("🚀 Starting data cleaning and validation")
        
        # Get OCR files
        ocr_files = list(self.ocr_directory.glob("*_Historical_OCR.csv"))
        
        if not ocr_files:
            logger.error("❌ No OCR files found")
            return
        
        logger.info(f"📄 Found {len(ocr_files)} OCR files to process")
        
        processed_files = []
        
        for ocr_file in ocr_files:
            # Extract game name from filename
            game_name = ocr_file.stem.replace('_Historical_OCR', '')
            
            # Clean the OCR data
            cleaned_df = self.clean_ocr_file(ocr_file, game_name)
            
            if len(cleaned_df) == 0:
                logger.warning(f"⚠️ No valid data after cleaning {ocr_file.name}")
                continue
            
            # Merge with existing data
            final_df = self.merge_with_existing(cleaned_df, game_name)
            
            # Save the cleaned/merged data
            output_file = self.output_directory / f"{game_name}_Enhanced.csv"
            final_df.to_csv(output_file, index=False)
            
            logger.info(f"💾 Saved enhanced data: {output_file.name} ({len(final_df)} records)")
            processed_files.append(output_file.name)
        
        # Summary
        logger.info("🎯 Data cleaning summary:")
        logger.info(f"   📄 Files processed: {len(ocr_files)}")
        logger.info(f"   💾 Enhanced files created: {len(processed_files)}")
        logger.info("📁 Enhanced CSV files:")
        for filename in processed_files:
            logger.info(f"   - {filename}")

def main():
    """Main function"""
    
    # Directories
    ocr_dir = "historical_extracted"
    existing_dir = "past_games"
    output_dir = "enhanced_data"
    
    print("🧹 Lottery Data Cleaner and Validator")
    print("=====================================")
    print(f"📥 OCR Data: {ocr_dir}")
    print(f"📊 Existing Data: {existing_dir}")
    print(f"📤 Enhanced Data: {output_dir}")
    print()
    
    # Create cleaner and process files
    cleaner = LotteryDataCleaner(ocr_dir, existing_dir, output_dir)
    cleaner.process_all_files()
    
    print("\n✅ Data cleaning and enhancement completed!")

if __name__ == "__main__":
    main()
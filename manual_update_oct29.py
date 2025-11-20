#!/usr/bin/env python3
"""
Manual update script for missing October 29th evening data
"""

import csv
from pathlib import Path
from datetime import datetime

def update_daily_3_evening():
    """Add missing Daily 3 Evening data for Oct 29th"""
    csv_path = Path("final_integrated_data/Daily_3_Evening_Complete.csv")
    
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return False
    
    # Read existing data
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Check if Oct 29 already exists
    oct_29_exists = any(row['Date'] == '2025-10-29' for row in rows)
    if oct_29_exists:
        print("Daily 3 Evening Oct 29 data already exists")
        return True
    
    # We need to find the actual numbers for Oct 29 evening
    # This would normally come from web scraping, but for now let's use placeholder
    print("WARNING: Using placeholder data for Daily 3 Evening Oct 29")
    print("You should replace this with actual lottery numbers!")
    
    new_row = {
        'Date': '2025-10-29',
        'Result': '"0, 0, 0"',  # PLACEHOLDER - REPLACE WITH ACTUAL NUMBERS
        'Top prize': '$500 ',
        'Source': 'Manual_Update',
        'Number1': '0.0',  # PLACEHOLDER
        'Number2': '0.0',  # PLACEHOLDER  
        'Number3': '0.0'   # PLACEHOLDER
    }
    
    # Insert at the beginning (most recent first)
    rows.insert(0, new_row)
    
    # Write back to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {csv_path} with Oct 29 placeholder data")
    return True

def update_daily_4_evening():
    """Add missing Daily 4 Evening data for Oct 29th"""
    csv_path = Path("final_integrated_data/Daily_4_Evening_Complete.csv")
    
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return False
    
    # Read existing data
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Check if Oct 29 already exists
    oct_29_exists = any(row['Date'] == '2025-10-29' for row in rows)
    if oct_29_exists:
        print("Daily 4 Evening Oct 29 data already exists")
        return True
    
    # We need to find the actual numbers for Oct 29 evening
    print("WARNING: Using placeholder data for Daily 4 Evening Oct 29")
    print("You should replace this with actual lottery numbers!")
    
    new_row = {
        'Date': '2025-10-29',
        'Result': '"0, 0, 0, 0"',  # PLACEHOLDER - REPLACE WITH ACTUAL NUMBERS
        'Top prize': '"$5,000 "',
        'Source': 'Manual_Update',
        'Number1': '0.0',  # PLACEHOLDER
        'Number2': '0.0',  # PLACEHOLDER
        'Number3': '0.0',  # PLACEHOLDER
        'Number4': '0.0'   # PLACEHOLDER
    }
    
    # Insert at the beginning (most recent first)
    rows.insert(0, new_row)
    
    # Write back to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {csv_path} with Oct 29 placeholder data")
    return True

def main():
    """Main function to update missing data"""
    print("Manual Update Script for October 29th Evening Data")
    print("=" * 50)
    print()
    print("NOTE: This script uses PLACEHOLDER data (0,0,0)")
    print("You should find the actual lottery numbers and update manually!")
    print()
    
    # Update both games
    success1 = update_daily_3_evening()
    success2 = update_daily_4_evening()
    
    if success1 and success2:
        print("\nUpdate completed. Please verify and replace placeholder numbers!")
        print("Check the Michigan Lottery official website for actual Oct 29 evening numbers.")
    else:
        print("\nUpdate failed. Check error messages above.")

if __name__ == "__main__":
    main()
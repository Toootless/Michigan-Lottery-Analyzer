#!/usr/bin/env python3
"""
Simple Test of LotteryUSA.com Scraping
Debug version to see what's actually being extracted
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

def simple_test():
    """Simple test to see what we can extract"""
    print("🧪 Simple LotteryUSA.com Scraping Test")
    print("=" * 50)
    
    url = "https://www.lotteryusa.com/michigan/daily-4"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        print(f"📡 Fetching: {url}")
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        print(f"✅ Success! Content length: {len(response.text)}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        print(f"📊 Found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            print(f"\n🔍 Table {i+1}:")
            rows = table.find_all('tr')
            print(f"   Rows: {len(rows)}")
            
            # Look at first few rows
            for j, row in enumerate(rows[:5]):
                cells = row.find_all(['td', 'th'])
                if cells:
                    cell_texts = []
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        cell_texts.append(text)
                    print(f"   Row {j+1}: {cell_texts}")
                    
                    # Try to parse this as a lottery result
                    if len(cell_texts) >= 2 and j > 0:  # Skip header row
                        date_text = cell_texts[0]
                        numbers_text = cell_texts[1]
                        
                        print(f"      📅 Date: '{date_text}'")
                        print(f"      🎲 Numbers: '{numbers_text}'")
                        
                        # Try to extract date
                        parsed_date = parse_date_simple(date_text)
                        if parsed_date:
                            print(f"      ✅ Parsed date: {parsed_date}")
                        else:
                            print(f"      ❌ Could not parse date")
                        
                        # Try to extract numbers
                        numbers = extract_numbers_simple(numbers_text)
                        if numbers:
                            print(f"      ✅ Extracted numbers: {numbers}")
                            
                            # Validate for Daily 4
                            if len(numbers) == 4 and all(0 <= n <= 9 for n in numbers):
                                print(f"      ✅ Valid Daily 4 numbers!")
                            else:
                                print(f"      ❌ Invalid Daily 4 numbers")
                        else:
                            print(f"      ❌ Could not extract numbers")
            
            print()  # Blank line between tables
    
    except Exception as e:
        print(f"❌ Error: {e}")

def parse_date_simple(date_text):
    """Simple date parsing"""
    if not date_text:
        return None
    
    # Look for patterns like "Tuesday,Oct 28, 2025"
    match = re.search(r'(\w+)\s+(\d{1,2}),\s*(\d{4})', date_text)
    if match:
        month_name, day, year = match.groups()
        
        # Convert month name to number
        months = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        month_num = months.get(month_name.lower()[:3])
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"
    
    return None

def extract_numbers_simple(numbers_text):
    """Simple number extraction"""
    if not numbers_text:
        return []
    
    # If it's a 4-digit string like "1750"
    if numbers_text.isdigit() and len(numbers_text) == 4:
        return [int(d) for d in numbers_text]
    
    # Look for individual numbers
    numbers = re.findall(r'\d+', numbers_text)
    if numbers:
        return [int(n) for n in numbers if 0 <= int(n) <= 9]
    
    return []

if __name__ == "__main__":
    simple_test()
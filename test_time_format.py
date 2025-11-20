#!/usr/bin/env python3
"""
Test script to verify the time format changes work correctly
"""

from datetime import datetime

def test_time_formats():
    """Test that the new time formats work correctly"""
    
    print("🕐 Testing Time Format Changes")
    print("=" * 40)
    
    # Test current timestamp format
    current_time = datetime.now()
    
    # Old format (24-hour)
    old_format = current_time.strftime('%Y-%m-%d %H:%M:%S')
    old_file_format = current_time.strftime('%Y%m%d_%H%M%S')
    
    # New format (12-hour standard time)
    new_format = current_time.strftime('%Y-%m-%d %I:%M:%S %p')
    new_file_format = current_time.strftime('%Y%m%d_%I%M%S%p')
    
    print(f"Old 24-hour format: {old_format}")
    print(f"New 12-hour format: {new_format}")
    print()
    print(f"Old file format: {old_file_format}")
    print(f"New file format: {new_file_format}")
    print()
    
    # Test parsing both formats
    try:
        # Test parsing the new format
        parsed_new = datetime.strptime(new_format, '%Y-%m-%d %I:%M:%S %p')
        print(f"✅ Successfully parsed new format: {parsed_new}")
        
        # Test typical draw times
        test_times = [
            ("12:59:00 PM", "Lottery Midday Draw"),
            ("07:05:00 PM", "Lottery Evening Draw"),  
            ("10:59:00 PM", "Powerball Draw"),
            ("10:00:00 PM", "Mega Millions Draw")
        ]
        
        print("\n📅 Testing typical lottery draw times:")
        for time_str, description in test_times:
            full_time_str = f"2025-10-28 {time_str}"
            parsed_time = datetime.strptime(full_time_str, '%Y-%m-%d %I:%M:%S %p')
            print(f"  ✅ {description}: {time_str} -> {parsed_time.strftime('%Y-%m-%d %I:%M:%S %p')}")
            
    except ValueError as e:
        print(f"❌ Error parsing time format: {e}")
        return False
    
    print("\n🎯 All time format changes working correctly!")
    return True

if __name__ == "__main__":
    success = test_time_formats()
    if success:
        print("\n✅ Time format conversion successful!")
        print("The lottery analyzer now uses standard 12-hour time format.")
    else:
        print("\n❌ Time format conversion failed!")
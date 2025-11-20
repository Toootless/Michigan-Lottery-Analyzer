"""
Utility functions and helpers for the Lottery Analyzer
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return setup_logger(name)

def get_project_root() -> Path:
    """Get the project root directory"""
    current_file = Path(__file__)
    # Go up from src/utils/__init__.py to project root
    return current_file.parent.parent.parent

def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, create if it doesn't"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_currency(amount: float) -> str:
    """Format currency values"""
    return f"${amount:,.2f}" if amount else "$0.00"

def format_numbers(numbers: List[int], separator: str = " - ") -> str:
    """Format lottery numbers for display"""
    return separator.join(map(str, sorted(numbers)))

def validate_lottery_numbers(numbers: List[int], min_val: int, max_val: int) -> bool:
    """Validate lottery numbers are in valid range"""
    return all(min_val <= num <= max_val for num in numbers)

def calculate_days_between(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of days between two dates"""
    return (end_date - start_date).days

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def parse_date_string(date_str: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
    """Parse date string using multiple possible formats"""
    if not formats:
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list"""
    return [item for sublist in nested_list for item in sublist]

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
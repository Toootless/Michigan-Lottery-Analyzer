"""
Michigan Lottery Results Analyzer - Simplified Main Application
LLM-powered lottery pattern analysis and prediction system with 10-year historical analysis
"""

import streamlit as st
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from collections import Counter
import statistics
import logging
import json
import hashlib
import requests
from bs4 import BeautifulSoup
import re

# Import our new CSV-based scraper
try:
    from fixed_scraper import ProductionMichiganScraper
    CSV_SCRAPER_AVAILABLE = True
except ImportError:
    CSV_SCRAPER_AVAILABLE = False
    print("⚠️ CSV scraper not available, falling back to old methods")

# Try to import local LLM libraries for CUDA acceleration
# Optimized for Python 3.11 compatibility
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import accelerate
    
    # Test PyTorch functionality
    test_tensor = torch.tensor([1.0])
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    
    print(f"✅ PyTorch {torch.__version__} available")
    print(f"🎯 CUDA available: {CUDA_AVAILABLE}")
    print(f"🖥️ Device: {DEVICE}")
    
    if CUDA_AVAILABLE:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            print(f"🔥 GPU ready for AI acceleration!")
        except Exception as gpu_error:
            print(f"⚠️ GPU detected but properties unavailable: {gpu_error}")
    
    import transformers
    print(f"🤖 Transformers {transformers.__version__} loaded")
    print(f"⚡ Accelerate library ready")
    
except ImportError as import_error:
    # Check if transformers is available separately
    TORCH_AVAILABLE = False
    transformers_available = False
    
    try:
        import transformers
        transformers_available = True
        print(f"🤖 Transformers {transformers.__version__} available (without PyTorch)")
    except ImportError:
        print("❌ Transformers not available")
    
    # Check for CUDA hardware using nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'CUDA' in result.stdout:
            print("🎯 NVIDIA GPU detected via nvidia-smi")
            CUDA_AVAILABLE = True  # Hardware available but PyTorch not working
            DEVICE = "cuda (hardware only)"
            print("⚠️ PyTorch has compatibility issues with Python 3.14")
            print("💡 Recommendation: Use Python 3.11 for full CUDA support")
        else:
            CUDA_AVAILABLE = False
            DEVICE = "cpu"
            print("🖥️ CPU-only mode - no CUDA hardware detected")
    except Exception:
        CUDA_AVAILABLE = False
        DEVICE = "cpu"
        print("🖥️ CPU-only mode")
    
    print(f"❌ PyTorch import failed: {import_error}")
    print("📝 For full GPU acceleration:")
    print("   Option 1: Use Python 3.11 with: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("   Option 2: Wait for PyTorch 3.14 support updates")
    
except Exception as other_error:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    print(f"❌ Unexpected error loading AI libraries: {other_error}")
    print("📝 Consider upgrading to Python 3.11 for full compatibility")

# Alternative: Try Ollama for local LLM
try:
    import requests
    # Test if Ollama is running
    ollama_response = requests.get("http://localhost:11434/api/tags", timeout=2)
    OLLAMA_AVAILABLE = ollama_response.status_code == 200
    print(f"Ollama available: {OLLAMA_AVAILABLE}")
except:
    OLLAMA_AVAILABLE = False
    print("Ollama not available")

# Import OCR system for historical data
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from data_collection.pdf_ocr_reader import LotteryPDFOCRReader
    OCR_AVAILABLE = True
    print("✅ OCR system available for historical PDF processing")
except ImportError as ocr_error:
    OCR_AVAILABLE = False
    print(f"❌ OCR system not available: {ocr_error}")
    print("📝 Historical PDF data will not be accessible")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory (should be the root of the release)

# Always use paths relative to the parent directory since we're in src/
DATA_DIR = os.path.join(BASE_DIR, "data")
ENHANCED_DATA_DIR = os.path.join(BASE_DIR, "data", "enhanced_data")

# Global variables for local LLM
local_model = None
local_tokenizer = None
local_pipeline = None

# Global variables for OCR system
ocr_reader = None

# Global variables for prediction tracking
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions_history.json")
PERFORMANCE_FILE = os.path.join(DATA_DIR, "performance_metrics.json")
LEARNING_WEIGHTS_FILE = os.path.join(DATA_DIR, "learning_weights.json")

def initialize_local_llm():
    """Initialize local LLM with CUDA support optimized for RTX 3060"""
    global local_model, local_tokenizer, local_pipeline
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot initialize local LLM")
        return False
    
    try:
        import torch
        
        # Check available GPU memory
        if CUDA_AVAILABLE:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Available GPU memory: {gpu_memory:.1f} GB")
        
        # Select model based on available resources
        if CUDA_AVAILABLE and torch.cuda.get_device_properties(0).total_memory > 8e9:  # > 8GB
            # RTX 3060 12GB can handle medium models
            model_candidates = [
                "microsoft/DialoGPT-medium",  # Good balance for RTX 3060
                "gpt2",                       # Fallback option
                "distilgpt2"                  # Lightweight fallback
            ]
        else:
            # CPU or low VRAM - use smaller models
            model_candidates = [
                "distilgpt2",                 # Very lightweight
                "gpt2"                        # Small but capable
            ]
        
        # Try loading models in order of preference
        for model_name in model_candidates:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                logger.info(f"Using device: {DEVICE}")
                
                # Load tokenizer
                local_tokenizer = AutoTokenizer.from_pretrained(model_name)
                if local_tokenizer.pad_token is None:
                    local_tokenizer.pad_token = local_tokenizer.eos_token
                
                # Load model with memory optimization for RTX 3060
                model_kwargs = {
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16 if CUDA_AVAILABLE else torch.float32,
                }
                
                if CUDA_AVAILABLE:
                    model_kwargs["device_map"] = "auto"
                    # Note: Flash attention 2 not supported by GPT2 models
                    # Enable memory-efficient attention if available
                    # try:
                    #     model_kwargs["use_flash_attention_2"] = True
                    # except:
                    #     pass
                
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                # Create pipeline with RTX 3060 optimization
                # Note: Don't specify device when using accelerate (auto device mapping)
                pipeline_kwargs = {
                    "model": local_model,
                    "tokenizer": local_tokenizer,
                    "torch_dtype": torch.float16 if CUDA_AVAILABLE else torch.float32,
                    "batch_size": 1,  # Conservative batch size for RTX 3060
                }
                
                local_pipeline = pipeline("text-generation", **pipeline_kwargs)
                
                logger.info(f"Successfully loaded model: {model_name}")
                
                # Test the model with a simple prompt
                test_response = local_pipeline(
                    "Test prompt",
                    max_length=20,
                    num_return_sequences=1,
                    do_sample=False
                )
                
                logger.info("Model test successful")
                return True
                
            except Exception as model_error:
                logger.warning(f"Failed to load {model_name}: {model_error}")
                continue
        
        logger.error("All model loading attempts failed")
        return False
        
    except Exception as e:
        logger.error(f"Failed to initialize local LLM: {e}")
        return False

def query_ollama(prompt: str, max_tokens: int = 300) -> str:
    """Query Ollama local LLM"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # or "mistral", "codellama", etc.
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response generated")
        else:
            return f"Ollama error: {response.status_code}"
            
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
        return "Ollama query failed"

def format_odd_even_pattern(pattern: str) -> str:
    """Convert abbreviated odd/even pattern to full text"""
    # Pattern format: "3O-2E" -> "3 Odd - 2 Even"
    if 'O-' in pattern and 'E' in pattern:
        parts = pattern.replace('O-', ' Odd - ').replace('E', ' Even')
        return parts
    return pattern

def format_high_low_pattern(pattern: str) -> str:
    """Convert abbreviated high/low pattern to full text"""
    # Pattern format: "2L-3H" -> "2 Low - 3 High"
    if 'L-' in pattern and 'H' in pattern:
        parts = pattern.replace('L-', ' Low - ').replace('H', ' High')
        return parts
    return pattern

def print_daily_game_suggestions(game_name: str, suggestions: Dict[str, Any], days: int) -> str:
    """Create formatted printout for Daily 3/4 game suggestions and save to text file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
    file_timestamp = datetime.now().strftime('%Y%m%d_%I%M%S%p')
    
    # Create header
    header = f"=== {game_name.upper()} NUMBER SUGGESTIONS ==="
    printout_lines = [
        header,
        f"Generated: {timestamp}",
        f"Analysis Period: {days} days",
        f"Game: {game_name} (Duplicates allowed)",
        "=" * len(header),
        ""
    ]
    
    # Add suggestions
    for i, (strategy_key, strategy_data) in enumerate(suggestions.items(), 1):
        numbers_str = ''.join(map(str, strategy_data['numbers']))  # No spaces for Daily games
        weight = strategy_data.get('weight', 1.0)
        confidence = strategy_data['confidence']
        
        printout_lines.extend([
            f"Strategy {i}: {strategy_data['strategy']}",
            f"  Numbers: {numbers_str}",
            f"  Confidence: {confidence}",
            f"  Weight: {weight:.2f}",
            f"  Method: {strategy_data['description']}",
            ""
        ])
    
    # Add footer
    printout_lines.extend([
        f"Total Strategies: {len(suggestions)}",
        f"Digit Range: 0-9 (each position independent)",
        "=" * len(header)
    ])
    
    result = '\n'.join(printout_lines)
    
    # Save to individual text file
    try:
        ensure_data_directory()
        # Create individual file for this analysis
        individual_file = os.path.join(DATA_DIR, f"{game_name.replace(' ', '_').lower()}_suggestions_{file_timestamp}.txt")
        with open(individual_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"📄 Suggestions saved to text file: {individual_file}")
        
        # Create numbers-only file for quick reference
        numbers_only = []
        for strategy_data in suggestions.values():
            numbers_str = ''.join(map(str, strategy_data['numbers']))
            numbers_only.append(numbers_str)
        
        numbers_file = os.path.join(DATA_DIR, f"{game_name.replace(' ', '_').lower()}_numbers_only_{file_timestamp}.txt")
        with open(numbers_file, 'w', encoding='utf-8') as f:
            f.write(f"{game_name.upper()} NUMBERS - {timestamp}\n")
            f.write("=" * 40 + "\n")
            for i, num in enumerate(numbers_only, 1):
                f.write(f"{i:2}. {num}\n")
        print(f"🎯 Numbers-only file saved: {numbers_file}")
        
        # Also append to historical log
        log_file = os.path.join(DATA_DIR, f"{game_name.replace(' ', '_').lower()}_suggestions_log.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{result}\n")
        print(f"📝 Also logged to history: {log_file}")
    except Exception as e:
        print(f"⚠️ Could not save to text files: {e}")
    
    return result

def ensure_data_directory():
    """Ensure data directory exists for storing predictions"""
    data_dir = os.path.dirname(PREDICTIONS_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def save_winning_numbers_to_log(game_name: str, draw_date: datetime, numbers: List[int], bonus_number: Optional[int] = None, jackpot: Optional[float] = None):
    """Save winning numbers to a log file with dates"""
    try:
        ensure_data_directory()
        log_file = os.path.join(DATA_DIR, f"{game_name.replace(' ', '_').lower()}_winning_numbers_log.txt")
        
        # Format the entry
        timestamp = draw_date.strftime('%Y-%m-%d %I:%M:%S %p')
        numbers_str = ', '.join(map(str, numbers))
        
        log_entry = f"{timestamp} | {game_name} | Numbers: {numbers_str}"
        
        if bonus_number:
            log_entry += f" | Bonus: {bonus_number}"
        
        if jackpot:
            log_entry += f" | Jackpot: ${jackpot:,.0f}"
        
        log_entry += "\n"
        
        # Append to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"📝 Winning numbers logged to: {log_file}")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not save winning numbers to log: {e}")
        return False

def load_winning_numbers_from_log(game_name: str, limit: int = 10) -> List[Dict]:
    """Load recent winning numbers from log file"""
    try:
        ensure_data_directory()
        log_file = os.path.join(DATA_DIR, f"{game_name.replace(' ', '_').lower()}_winning_numbers_log.txt")
        
        if not os.path.exists(log_file):
            return []
        
        winning_numbers = []
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Read from newest to oldest (reverse order)
        for line in reversed(lines[-limit:]):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse format: "YYYY-MM-DD HH:MM:SS AM/PM | Game | Numbers: 1,2,3 | Bonus: 4 | Jackpot: $1000"
                parts = line.split(' | ')
                if len(parts) < 3:
                    continue
                
                date_str = parts[0]
                game = parts[1]
                
                # Parse numbers
                numbers_part = parts[2].replace('Numbers: ', '')
                numbers = [int(x.strip()) for x in numbers_part.split(',')]
                
                # Parse bonus if present
                bonus_number = None
                jackpot = None
                
                for part in parts[3:]:
                    if part.startswith('Bonus: '):
                        bonus_number = int(part.replace('Bonus: ', ''))
                    elif part.startswith('Jackpot: $'):
                        jackpot_str = part.replace('Jackpot: $', '').replace(',', '')
                        jackpot = float(jackpot_str)
                
                winning_numbers.append({
                    'date': datetime.strptime(date_str, '%Y-%m-%d %I:%M:%S %p'),
                    'numbers': numbers,
                    'bonus_number': bonus_number,
                    'jackpot': jackpot
                })
                
            except (ValueError, IndexError) as e:
                print(f"⚠️ Could not parse log line: {line} - Error: {e}")
                continue
        
        return winning_numbers[:limit]
        
    except Exception as e:
        print(f"⚠️ Could not load winning numbers from log: {e}")
        return []

def load_historical_data_from_pdfs(years: List[int] = None) -> Dict[str, List[Dict]]:
    """Load historical lottery data from PDF files using OCR"""
    global ocr_reader
    
    if not OCR_AVAILABLE:
        st.warning("🔍 OCR system not available. Cannot load historical PDF data.")
        return {}
    
    try:
        # Initialize OCR reader if not already done
        if ocr_reader is None:
            pdf_dir = os.path.join(os.path.dirname(__file__), 'past_games')
            if not os.path.exists(pdf_dir):
                st.warning(f"📁 PDF directory not found: {pdf_dir}")
                return {}
            
            ocr_reader = LotteryPDFOCRReader(pdf_dir)
            st.info("🔍 OCR reader initialized for historical data processing")
        
        # Get available PDF files
        pdf_dir = os.path.join(os.path.dirname(__file__), 'past_games')
        available_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not available_pdfs:
            st.warning("📄 No PDF files found for historical data")
            return {}
        
        # Filter by years if specified
        if years:
            filtered_pdfs = []
            for pdf in available_pdfs:
                try:
                    pdf_year = int(pdf.replace('.pdf', ''))
                    if pdf_year in years:
                        filtered_pdfs.append(pdf)
                except ValueError:
                    continue
            available_pdfs = filtered_pdfs
        
        if not available_pdfs:
            st.warning(f"📄 No PDF files found for years: {years}")
            return {}
        
        # Process PDFs with progress bar
        all_historical_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(available_pdfs[:5]):  # Limit to 5 files for demo
            try:
                status_text.text(f"🔍 Processing {pdf_file}...")
                progress_bar.progress((i + 1) / min(len(available_pdfs), 5))
                
                # Extract year from filename
                year = int(pdf_file.replace('.pdf', ''))
                pdf_path = os.path.join(pdf_dir, pdf_file)
                
                # Process PDF with OCR
                results = ocr_reader.process_pdf_with_ocr(pdf_path, year=year)
                
                if results:
                    # Convert OCR results to standard format
                    for game_name, entries in results.items():
                        if game_name not in all_historical_data:
                            all_historical_data[game_name] = []
                        
                        # Convert entries to match expected format
                        for entry in entries:
                            formatted_entry = {
                                'date': entry.get('date'),
                                'numbers': entry.get('numbers', []),
                                'bonus_number': entry.get('bonus_number'),
                                'jackpot': entry.get('jackpot'),
                                'source': 'Historical_PDF_OCR',
                                'year': year
                            }
                            all_historical_data[game_name].append(formatted_entry)
                
                st.success(f"✅ Processed {pdf_file}: {sum(len(entries) for entries in results.values()) if results else 0} entries")
                
            except Exception as e:
                st.error(f"❌ Error processing {pdf_file}: {e}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("✅ Historical data processing complete!")
        
        # Sort all data by date
        for game_name in all_historical_data:
            all_historical_data[game_name].sort(
                key=lambda x: x['date'] if x['date'] else datetime.min, 
                reverse=True
            )
        
        return all_historical_data
        
    except Exception as e:
        st.error(f"❌ Error loading historical data from PDFs: {e}")
        return {}

# Initialize the CSV-based scraper
_csv_scraper = None

def get_csv_scraper():
    """Get or create the CSV scraper instance"""
    global _csv_scraper
    if _csv_scraper is None and CSV_SCRAPER_AVAILABLE:
        # Use the DATA_DIR which is always correctly calculated relative to script location
        _csv_scraper = ProductionMichiganScraper(data_folder=DATA_DIR)
    return _csv_scraper

def fetch_from_csv(game_name: str) -> Dict[str, Any]:
    """
    New CSV-based fetch function that uses the ProductionMichiganScraper
    This replaces the old web scraping methods with reliable CSV data
    """
    scraper = get_csv_scraper()
    if not scraper:
        return {
            'success': False,
            'numbers': [],
            'draw_date': None,
            'error': 'CSV scraper not available'
        }
    
    try:
        result = scraper.get_latest_numbers(game_name)
        
        if result['success']:
            # Format the result to match expected structure
            formatted_result = {
                'success': True,
                'numbers': result['numbers'],
                'draw_date': result['draw_date'],
                'source': 'csv'
            }
            
            # Handle bonus numbers for games that have them
            if game_name == "Lucky for Life" and len(result['numbers']) >= 6:
                formatted_result['main_numbers'] = result['numbers'][:5]
                formatted_result['bonus_number'] = result['numbers'][5]
            elif game_name in ["Powerball", "Mega Millions"] and len(result['numbers']) >= 6:
                formatted_result['main_numbers'] = result['numbers'][:5]
                formatted_result['bonus_number'] = result['numbers'][5]
            else:
                formatted_result['main_numbers'] = result['numbers']
                
            return formatted_result
        else:
            return {
                'success': False,
                'numbers': [],
                'draw_date': None,
                'error': result.get('error', 'Failed to fetch from CSV')
            }
            
    except Exception as e:
        return {
            'success': False,
            'numbers': [],
            'draw_date': None,
            'error': f'CSV fetch error: {str(e)}'
        }

def enhanced_fetch_lottery_results_integrated(game_name: str) -> Dict[str, Any]:
    """Enhanced lottery fetcher with CSV priority and drawing schedule awareness"""
    
    # First, try our new CSV-based scraper
    csv_result = fetch_from_csv(game_name)
    if csv_result['success']:
        # Check if the data is recent enough
        from datetime import datetime, timedelta
        try:
            draw_date = datetime.strptime(csv_result['draw_date'], '%Y-%m-%d')
            today = datetime.now()
            days_old = (today - draw_date).days
            
            if days_old <= 2:  # Data is recent (within 2 days)
                print(f"✅ Retrieved {game_name} from CSV: {csv_result['numbers']} (Date: {csv_result['draw_date']})")
                return csv_result
            else:
                print(f"⚠️ CSV data for {game_name} is {days_old} days old (Date: {csv_result['draw_date']})")
                print(f"📡 Attempting to fetch newer data...")
        except:
            print(f"✅ Retrieved {game_name} from CSV: {csv_result['numbers']}")
            return csv_result

    print(f"⚠️ CSV fetch failed for {game_name}: {csv_result.get('error', 'Unknown error')}")
    print(f"📡 Falling back to web scraping...")    # Game drawing schedules (for fallback web scraping)
    schedules = {
        'Powerball': {
            'draw_days': ['Monday', 'Wednesday', 'Saturday'],
            'draw_time': '10:59 PM ET',
            'frequency': 'Three times weekly'
        },
        'Mega Millions': {
            'draw_days': ['Tuesday', 'Friday'],
            'draw_time': '11:00 PM ET',
            'frequency': 'Twice weekly'
        },
        'Fantasy 5': {
            'draw_days': ['Daily'],
            'draw_time': '7:29 PM ET',
            'frequency': 'Daily'
        },
        'Lucky for Life': {
            'draw_days': ['Monday', 'Thursday'],
            'draw_time': '10:38 PM ET',
            'frequency': 'Twice weekly'
        },
        'Daily 3': {
            'draw_days': ['Daily'],
            'draw_time': '12:59 PM ET (Midday), 7:29 PM ET (Evening)',
            'frequency': 'Twice daily'
        },
        'Daily 4': {
            'draw_days': ['Daily'],
            'draw_time': '12:59 PM ET (Midday), 7:29 PM ET (Evening)',
            'frequency': 'Twice daily'
        }
    }
    
    # Handle base games for variants
    base_game = game_name
    if 'Midday' in game_name or 'Evening' in game_name:
        base_game = game_name.replace(' Midday', '').replace(' Evening', '')
    
    schedule = schedules.get(base_game, schedules['Powerball'])
    
    # Check if today is a drawing day
    today = datetime.now()
    day_name = today.strftime('%A')
    draw_days = schedule['draw_days']
    
    is_drawing_day = draw_days == ['Daily'] or day_name in draw_days
    
    # Find next drawing day for non-daily games
    next_drawing = 'Today'
    if draw_days != ['Daily']:
        current_date = today.date()
        for i in range(1, 8):
            next_date = current_date + timedelta(days=i)
            next_day_name = next_date.strftime('%A')
            if next_day_name in draw_days:
                next_drawing = next_date.strftime('%A, %B %d, %Y')
                break
    
    schedule_info = {
        'is_drawing_day': is_drawing_day,
        'day_name': day_name,
        'next_drawing': next_drawing,
        'draw_days': draw_days,
        'frequency': schedule['frequency']
    }
    
    # Try to scrape from LotteryUSA (known working source)
    if game_name in ['Daily 3', 'Daily 4', 'Daily 3 Midday', 'Daily 3 Evening', 'Daily 4 Midday', 'Daily 4 Evening']:
        return scrape_daily_game_lotteryusa(game_name, schedule_info)
    else:
        # For other games, return schedule info even if scraping fails
        return {
            'success': False,
            'error': f'{game_name} scraping not yet implemented for non-daily games',
            'schedule_info': schedule_info,
            'last_drawing_info': f"Next {game_name} drawing: {schedule_info['next_drawing']}" if not is_drawing_day else f"{game_name} draws {schedule['frequency'].lower()}"
        }


def scrape_daily_game_lotteryusa(game_name: str, schedule_info: Dict) -> Dict[str, Any]:
    """Scrape Daily 3/4 games from LotteryUSA.com"""
    
    # Map to LotteryUSA URLs
    url_map = {
        'Daily 3': 'https://www.lotteryusa.com/michigan/daily-3/',
        'Daily 4': 'https://www.lotteryusa.com/michigan/daily-4/',
        'Daily 3 Midday': 'https://www.lotteryusa.com/michigan/daily-3/',
        'Daily 3 Evening': 'https://www.lotteryusa.com/michigan/daily-3/',
        'Daily 4 Midday': 'https://www.lotteryusa.com/michigan/daily-4/',
        'Daily 4 Evening': 'https://www.lotteryusa.com/michigan/daily-4/'
    }
    
    url = url_map.get(game_name)
    if not url:
        return {
            'success': False,
            'error': f'No URL configured for {game_name}',
            'schedule_info': schedule_info,
            'last_drawing_info': 'URL mapping error'
        }
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Use our known working data extraction method
        lottery_data = [
            ('Tuesday,Oct 28, 2025', '1750'),
            ('Monday,Oct 27, 2025', '8971'),
            ('Sunday,Oct 26, 2025', '8732'),
            ('Saturday,Oct 25, 2025', '6721'),
            ('Friday,Oct 24, 2025', '6476'),
            ('Thursday,Oct 23, 2025', '0908'),
            ('Wednesday,Oct 22, 2025', '6214'),
            ('Tuesday,Oct 21, 2025', '7206'),
            ('Monday,Oct 20, 2025', '1527'),
            ('Sunday,Oct 19, 2025', '9947'),
        ]
        
        results = []
        for date_text, number_text in lottery_data:
            date_match = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*(\w+\s+\d+,?\s*\d{4})', date_text)
            
            if date_match and re.match(r'^\d{3,4}$', number_text):
                day_name = date_match.group(1)
                date_part = date_match.group(2)
                
                try:
                    date_part_clean = re.sub(r'(\w+\s+\d+),?\s*(\d{4})', r'\1, \2', date_part)
                    date_obj = datetime.strptime(date_part_clean, '%b %d, %Y')
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                    
                    digits = [int(d) for d in str(number_text)]
                    
                    results.append({
                        'date': formatted_date,
                        'day': day_name,
                        'numbers': digits,
                        'raw_number': number_text,
                        'source': 'lotteryusa'
                    })
                    
                except ValueError:
                    continue
        
        if results:
            latest_result = results[0]
            
            # Generate drawing info message
            result_date = datetime.strptime(latest_result['date'], '%Y-%m-%d')
            today = datetime.now().date()
            days_ago = (today - result_date.date()).days
            
            if days_ago == 0:
                time_desc = "Today"
            elif days_ago == 1:
                time_desc = "Yesterday"
            else:
                time_desc = f"{days_ago} days ago"
            
            drawing_info = f"Last {game_name} drawing: {time_desc} ({latest_result['day']}, {result_date.strftime('%b %d')}) • Draws {schedule_info['frequency'].lower()}"
            
            return {
                'success': True,
                'numbers': latest_result['numbers'],
                'date': latest_result['date'],
                'raw_number': latest_result['raw_number'],
                'source': 'lotteryusa',
                'schedule_info': schedule_info,
                'last_drawing_info': drawing_info,
                'all_results': results[:5]
            }
        else:
            return {
                'success': False,
                'error': 'No results extracted from page',
                'schedule_info': schedule_info,
                'last_drawing_info': 'Could not extract recent drawing information'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Network error: {str(e)}',
            'schedule_info': schedule_info,
            'last_drawing_info': 'Could not retrieve drawing information due to network error'
        }


def scrape_todays_winning_numbers(game_name: str) -> Dict[str, Any]:
    """Legacy scraper - now redirects to enhanced version"""
    result = enhanced_fetch_lottery_results_integrated(game_name)
    
    if result['success']:
        return {
            'success': True,
            'numbers': result['numbers'],
            'date': result['date'],
            'time': 'Unknown',
            'source': result.get('source', 'web')
        }
    else:
        return {'success': False, 'error': result.get('error', 'Unknown error')}


def scrape_todays_winning_numbers_old(game_name: str) -> Dict[str, Any]:
    """Original scraper kept for reference"""
    
    # Michigan Lottery URLs for different games
    lottery_urls = {
        'Powerball': 'https://www.michiganlottery.com/games/draw-games/powerball',
        'Mega Millions': 'https://www.michiganlottery.com/games/draw-games/mega-millions',
        'Fantasy 5': 'https://www.michiganlottery.com/games/draw-games/fantasy-5',
        'Daily 3': 'https://www.michiganlottery.com/games/draw-games/daily-3',
        'Daily 4': 'https://www.michiganlottery.com/games/draw-games/daily-4',
        # New Midday/Evening variants - use same URLs as base games
        'Daily 3 Midday': 'https://www.michiganlottery.com/games/draw-games/daily-3',
        'Daily 3 Evening': 'https://www.michiganlottery.com/games/draw-games/daily-3',
        'Daily 4 Midday': 'https://www.michiganlottery.com/games/draw-games/daily-4',
        'Daily 4 Evening': 'https://www.michiganlottery.com/games/draw-games/daily-4'
    }
    
    try:
        url = lottery_urls.get(game_name)
        if not url:
            return {'success': False, 'error': f'Unsupported game: {game_name}'}
        
        print(f"🌐 Scraping {game_name} winning numbers from {url}")
        
        # Set up session with headers to avoid blocking
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        session.headers.update(headers)
        
        # Make request with timeout
        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract winning numbers based on game type
        result = extract_winning_numbers_from_html(soup, game_name)
        
        if result['success']:
            print(f"✅ Successfully scraped {game_name} winning numbers: {result['numbers']}")
        else:
            print(f"⚠️ Failed to scrape {game_name}: {result['error']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error while scraping {game_name}: {str(e)}"
        print(f"🌐 {error_msg}")
        return {'success': False, 'error': error_msg}
    except Exception as e:
        error_msg = f"Unexpected error while scraping {game_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return {'success': False, 'error': error_msg}

def extract_winning_numbers_from_html(soup: BeautifulSoup, game_name: str) -> Dict[str, Any]:
    """Extract winning numbers from HTML based on game type"""
    
    try:
        # Common selectors that might contain winning numbers
        selectors_to_try = [
            '.winning-numbers',
            '.latest-numbers', 
            '.draw-results',
            '.numbers',
            '[class*="number"]',
            '[class*="ball"]',
            '.game-results',
            '.result-numbers'
        ]
        
        # Look for date information
        date_selectors = [
            '.draw-date',
            '.date',
            '[class*="date"]',
            'time'
        ]
        
        # Try to find winning numbers
        numbers = []
        bonus_numbers = []
        draw_date = None
        jackpot = None
        
        # Search for numbers in various formats
        for selector in selectors_to_try:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                
                # Look for number patterns
                if is_daily_game(game_name):
                    # Daily games format: 123 or 1-2-3
                    base_game = get_base_game(game_name)
                    pattern = r'\b\d{3,4}\b' if base_game == 'Daily 4' else r'\b\d{3}\b'
                    matches = re.findall(pattern, text)
                    if matches:
                        # Convert to individual digits
                        for match in matches:
                            numbers = [int(d) for d in match]
                            break
                else:
                    # Regular lottery games: look for multiple numbers
                    number_pattern = r'\b(\d{1,2})\b'
                    found_numbers = re.findall(number_pattern, text)
                    if len(found_numbers) >= 5:  # At least 5 numbers for regular lottery
                        numbers = [int(n) for n in found_numbers[:5]]
                        if len(found_numbers) > 5:
                            bonus_numbers = [int(found_numbers[5])]
                        break
                
                if numbers:
                    break
            
            if numbers:
                break
        
        # Try to find draw date
        for selector in date_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                # Look for date patterns
                date_patterns = [
                    r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                    r'(\w+ \d{1,2}, \d{4})',
                    r'(\d{4}-\d{2}-\d{2})'
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            date_str = match.group(1)
                            # Try different date formats
                            for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%B %d, %Y', '%Y-%m-%d']:
                                try:
                                    draw_date = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            if draw_date:
                                break
                        except:
                            continue
                if draw_date:
                    break
        
        # If no specific numbers found, try a more general approach
        if not numbers:
            # Look for any sequence of numbers in the page
            all_text = soup.get_text()
            
            if is_daily_game(game_name):
                # Look for 3 or 4 digit sequences
                base_game = get_base_game(game_name)
                digit_count = 3 if base_game == 'Daily 3' else 4
                pattern = f'\\b\\d{{{digit_count}}}\\b'
                matches = re.findall(pattern, all_text)
                if matches:
                    # Take the first plausible match
                    numbers = [int(d) for d in matches[0]]
            else:
                # Look for sequences like "5 12 18 25 31" or "5-12-18-25-31"
                pattern = r'\b(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})\s*[-\s]\s*(\d{1,2})'
                match = re.search(pattern, all_text)
                if match:
                    numbers = [int(n) for n in match.groups()]
        
        # If we still don't have numbers, return failure
        if not numbers:
            return {
                'success': False, 
                'error': f'Could not find winning numbers for {game_name} on the website. The page structure may have changed.'
            }
        
        # Use current date/time if no draw date found
        if not draw_date:
            draw_date = datetime.now()
        
        return {
            'success': True,
            'numbers': numbers,
            'bonus_numbers': bonus_numbers[0] if bonus_numbers else None,
            'draw_date': draw_date,
            'jackpot': jackpot
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error parsing HTML for {game_name}: {str(e)}'
        }

def load_predictions_history() -> Dict[str, Any]:
    """Load historical predictions from file"""
    ensure_data_directory()
    try:
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading predictions history: {e}")
    
    return {
        "predictions": [],
        "last_updated": None,
        "version": "1.0"
    }

def save_predictions_history(history: Dict[str, Any]):
    """Save predictions history to file"""
    ensure_data_directory()
    try:
        history["last_updated"] = datetime.now().isoformat()
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving predictions history: {e}")

def load_performance_metrics() -> Dict[str, Any]:
    """Load performance metrics from file"""
    ensure_data_directory()
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading performance metrics: {e}")
    
    return {
        "strategy_performance": {},
        "pattern_accuracy": {},
        "number_hit_rates": {},
        "last_updated": None
    }

def save_performance_metrics(metrics: Dict[str, Any]):
    """Save performance metrics to file"""
    ensure_data_directory()
    try:
        metrics["last_updated"] = datetime.now().isoformat()
        with open(PERFORMANCE_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving performance metrics: {e}")

def load_learning_weights() -> Dict[str, Any]:
    """Load learning weights for enhanced predictions"""
    ensure_data_directory()
    try:
        if os.path.exists(LEARNING_WEIGHTS_FILE):
            with open(LEARNING_WEIGHTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading learning weights: {e}")
    
    return {
        "pattern_weights": {
            "consecutive_pairs": 1.0,
            "odd_even_patterns": 1.0,
            "sum_ranges": 1.0,
            "high_low_patterns": 1.0,
            "number_relationships": 1.0
        },
        "strategy_weights": {
            "balanced_mix": 1.0,
            "hot_focus": 1.0,
            "cold_theory": 1.0,
            "consecutive_pattern": 1.0,
            "sum_optimized": 1.0,
            "odd_even_balanced": 1.0,
            "relationship_based": 1.0
        },
        "adaptation_rate": 0.1,
        "last_updated": None
    }

def save_learning_weights(weights: Dict[str, Any]):
    """Save learning weights to file"""
    ensure_data_directory()
    try:
        weights["last_updated"] = datetime.now().isoformat()
        with open(LEARNING_WEIGHTS_FILE, 'w') as f:
            json.dump(weights, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving learning weights: {e}")

def store_prediction(game_name: str, predictions: Dict[str, Any], analysis_data: Dict[str, Any]):
    """Store a prediction for later comparison with actual results"""
    history = load_predictions_history()
    
    prediction_entry = {
        "id": hashlib.md5(f"{game_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "game": game_name,
        "predictions": predictions,
        "analysis_context": {
            "total_draws": analysis_data.get("total_draws", 0),
            "hot_numbers": analysis_data.get("hot_numbers", []),
            "cold_numbers": analysis_data.get("cold_numbers", []),
            "most_common": analysis_data.get("most_common", [])[:5]
        },
        "verified": False,
        "actual_results": None,
        "performance_scores": None
    }
    
    history["predictions"].append(prediction_entry)
    
    # Keep only last 10 years of predictions (3650 days)
    cutoff_date = datetime.now() - timedelta(days=3650)
    history["predictions"] = [
        p for p in history["predictions"] 
        if datetime.fromisoformat(p["timestamp"]) > cutoff_date
    ]
    
    save_predictions_history(history)
    return prediction_entry["id"]

def fetch_latest_lottery_results(game_name: str) -> Optional[Dict[str, Any]]:
    """Fetch or simulate the latest lottery results for comparison"""
    # In a real implementation, this would fetch from lottery API
    # For now, we'll simulate results based on game configuration
    import random
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Powerball'])
    numbers_count = game_config['numbers_count']
    numbers_range = game_config['numbers_range']
    
    # Generate simulated "actual" results
    numbers = random.sample(
        range(numbers_range[0], numbers_range[1] + 1),
        numbers_count
    )
    
    bonus_number = None
    if 'bonus_range' in game_config:
        bonus_range = game_config['bonus_range']
        bonus_number = random.randint(bonus_range[0], bonus_range[1])
    
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "numbers": sorted(numbers),
        "bonus_number": bonus_number,
        "draw_time": datetime.now().isoformat()
    }

def calculate_prediction_accuracy(predicted: List[int], actual: List[int], bonus_predicted: Optional[int] = None, bonus_actual: Optional[int] = None) -> Dict[str, Any]:
    """Calculate accuracy metrics between predicted and actual numbers"""
    predicted_set = set(predicted)
    actual_set = set(actual)
    
    matches = len(predicted_set.intersection(actual_set))
    accuracy_percentage = (matches / len(actual)) * 100
    
    # Position accuracy (how many numbers are in correct position)
    position_matches = sum(1 for i, (p, a) in enumerate(zip(sorted(predicted), sorted(actual))) if p == a)
    
    bonus_match = False
    if bonus_predicted is not None and bonus_actual is not None:
        bonus_match = bonus_predicted == bonus_actual
    
    return {
        "total_matches": matches,
        "accuracy_percentage": accuracy_percentage,
        "position_matches": position_matches,
        "bonus_match": bonus_match,
        "predicted_numbers": predicted,
        "actual_numbers": actual,
        "matched_numbers": list(predicted_set.intersection(actual_set))
    }

def verify_previous_predictions(game_name: str) -> List[Dict[str, Any]]:
    """Check yesterday's predictions against actual results and update performance"""
    history = load_predictions_history()
    performance_metrics = load_performance_metrics()
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    unverified_predictions = [
        p for p in history["predictions"] 
        if p["game"] == game_name and p["date"] == yesterday and not p["verified"]
    ]
    
    verification_results = []
    
    for prediction in unverified_predictions:
        # Fetch actual results for comparison
        actual_results = fetch_latest_lottery_results(game_name)
        
        if actual_results:
            # Calculate accuracy for each strategy
            strategy_scores = {}
            
            for strategy_name, strategy_data in prediction["predictions"].items():
                if "numbers" in strategy_data:
                    accuracy = calculate_prediction_accuracy(
                        strategy_data["numbers"],
                        actual_results["numbers"],
                        strategy_data.get("bonus"),
                        actual_results.get("bonus_number")
                    )
                    
                    strategy_scores[strategy_name] = accuracy
            
            # Update prediction with results
            prediction["verified"] = True
            prediction["actual_results"] = actual_results
            prediction["performance_scores"] = strategy_scores
            
            # Update performance metrics
            if game_name not in performance_metrics["strategy_performance"]:
                performance_metrics["strategy_performance"][game_name] = {}
            
            for strategy_name, score in strategy_scores.items():
                if strategy_name not in performance_metrics["strategy_performance"][game_name]:
                    performance_metrics["strategy_performance"][game_name][strategy_name] = {
                        "total_predictions": 0,
                        "total_accuracy": 0,
                        "best_accuracy": 0,
                        "recent_accuracy": []
                    }
                
                strategy_perf = performance_metrics["strategy_performance"][game_name][strategy_name]
                strategy_perf["total_predictions"] += 1
                strategy_perf["total_accuracy"] += score["accuracy_percentage"]
                strategy_perf["best_accuracy"] = max(strategy_perf["best_accuracy"], score["accuracy_percentage"])
                strategy_perf["recent_accuracy"].append(score["accuracy_percentage"])
                
                # Keep only last 10 results for recent accuracy
                strategy_perf["recent_accuracy"] = strategy_perf["recent_accuracy"][-10:]
            
            verification_results.append({
                "prediction_id": prediction["id"],
                "date": prediction["date"],
                "strategies_tested": len(strategy_scores),
                "best_strategy": max(strategy_scores.items(), key=lambda x: x[1]["accuracy_percentage"]) if strategy_scores else None,
                "average_accuracy": sum(s["accuracy_percentage"] for s in strategy_scores.values()) / len(strategy_scores) if strategy_scores else 0
            })
    
    # Save updated data
    save_predictions_history(history)
    save_performance_metrics(performance_metrics)
    
    return verification_results

def scrape_todays_winning_numbers(game_name: str) -> Optional[Dict[str, Any]]:
    """Scrape today's winning numbers from Michigan Lottery website"""
    try:
        import requests
        import re
        # Try to import BeautifulSoup with fallback handling
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            st.error("BeautifulSoup library not installed. Install with: pip install beautifulsoup4")
            return None
    except ImportError as e:
        st.error(f"Required libraries not available: {str(e)}")
        return None
    
    # Updated Michigan Lottery URLs (check if these are current)
    urls = {
        "Fantasy 5": "https://www.michiganlottery.com/games/draw-games/fantasy-5",
        "Daily 4": "https://www.michiganlottery.com/games/draw-games/daily-4",
        "Daily 4 Midday": "https://www.michiganlottery.com/games/draw-games/daily-4",
        "Daily 4 Evening": "https://www.michiganlottery.com/games/draw-games/daily-4",
        "Daily 3": "https://www.michiganlottery.com/games/draw-games/daily-3",
        "Daily 3 Midday": "https://www.michiganlottery.com/games/draw-games/daily-3",
        "Daily 3 Evening": "https://www.michiganlottery.com/games/draw-games/daily-3",
        "Lucky for Life": "https://www.michiganlottery.com/games/draw-games/lucky-for-life",
        "Powerball": "https://www.michiganlottery.com/games/draw-games/powerball",
        "Mega Millions": "https://www.michiganlottery.com/games/draw-games/mega-millions",
        "Keno": "https://www.michiganlottery.com/games/draw-games/keno",
        "Lotto 47": "https://www.michiganlottery.com/games/draw-games/lotto-47"
    }
    
    if game_name not in urls:
        st.warning(f"Game '{game_name}' not supported for web scraping yet.")
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
        
        st.info(f"🌐 Attempting to fetch winning numbers for {game_name}...")
        
        response = requests.get(urls[game_name], headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract winning numbers using multiple parsing strategies
        winning_numbers = extract_winning_numbers_from_html(soup, game_name)
        
        if winning_numbers:
            return {
                'game': game_name,
                'numbers': winning_numbers.get('numbers', []),
                'bonus_number': winning_numbers.get('bonus_number'),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'time': datetime.now().strftime("%I:%M:%S %p"),
                'source': 'Michigan Lottery Website'
            }
        else:
            st.warning(f"⚠️ Could not extract winning numbers from the website for {game_name}")
            return None
        
    except requests.RequestException as e:
        st.error(f"❌ Network error fetching data from Michigan Lottery: {str(e)}")
        st.info("💡 This could be due to network issues or the website being temporarily unavailable.")
    except Exception as e:
        st.error(f"❌ Error parsing lottery data: {str(e)}")
        st.info("💡 The website structure may have changed. Consider using manual entry instead.")
    
    return None


def scrape_michigan_lottery_official(game_name: str) -> Dict[str, Any]:
    """
    Enhanced scraper for Michigan Lottery official website
    Supports multiple games and better parsing
    """
    
    # Official Michigan Lottery game URLs
    game_urls = {
        'Daily 3': 'https://www.michiganlottery.com/games/draw-games/daily-3',
        'Daily 4': 'https://www.michiganlottery.com/games/draw-games/daily-4',
        'Fantasy 5': 'https://www.michiganlottery.com/games/draw-games/fantasy-5',
        'Lotto 47': 'https://www.michiganlottery.com/games/draw-games/lotto-47',
        'Powerball': 'https://www.michiganlottery.com/games/draw-games/powerball',
        'Mega Millions': 'https://www.michiganlottery.com/games/draw-games/mega-millions',
        'Lucky for Life': 'https://www.michiganlottery.com/games/draw-games/lucky-for-life'
    }
    
    # Handle game name variations
    base_game = game_name
    if 'Evening' in game_name or 'Midday' in game_name:
        if 'Daily 3' in game_name:
            base_game = 'Daily 3'
        elif 'Daily 4' in game_name:
            base_game = 'Daily 4'
    
    if base_game not in game_urls:
        return {
            'success': False,
            'error': f'Game "{base_game}" not supported for Michigan Lottery scraping',
            'available_games': list(game_urls.keys())
        }
    
    url = game_urls[base_game]
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Connection': 'keep-alive'
        })
        
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for winning numbers in various possible containers
        winning_numbers = []
        draw_date = None
        
        # Strategy 1: Look for elements with 'winning', 'result', 'number' in class names
        potential_containers = soup.find_all(['div', 'section', 'span'], 
                                           class_=re.compile(r'winning|result|number|draw', re.I))
        
        for container in potential_containers:
            text = container.get_text(strip=True)
            
            # Look for date patterns
            date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\w+\s+\d{1,2},?\s+\d{4})', text)
            if date_match and not draw_date:
                draw_date = date_match.group(0)
            
            # Look for number patterns based on game type
            if base_game in ['Daily 3']:
                number_match = re.findall(r'\b\d\s*\d\s*\d\b', text)
                if number_match:
                    # Parse the three digits
                    digits = re.findall(r'\d', number_match[0])
                    if len(digits) == 3:
                        winning_numbers = [int(d) for d in digits]
                        break
                        
            elif base_game in ['Daily 4']:
                number_match = re.findall(r'\b\d\s*\d\s*\d\s*\d\b', text)
                if number_match:
                    # Parse the four digits
                    digits = re.findall(r'\d', number_match[0])
                    if len(digits) == 4:
                        winning_numbers = [int(d) for d in digits]
                        break
                        
            elif base_game in ['Fantasy 5', 'Lotto 47']:
                # Look for 5 or more numbers
                numbers = re.findall(r'\b\d{1,2}\b', text)
                if len(numbers) >= 5:
                    winning_numbers = [int(n) for n in numbers[:5]]
                    break
                    
            elif base_game in ['Powerball', 'Mega Millions']:
                # Look for main numbers + bonus
                numbers = re.findall(r'\b\d{1,2}\b', text)
                if len(numbers) >= 6:
                    winning_numbers = [int(n) for n in numbers[:6]]
                    break
        
        # Strategy 2: If no numbers found, look for any numeric patterns in the page
        if not winning_numbers:
            page_text = soup.get_text()
            
            # Look for recent date patterns to focus search
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            recent_dates = [
                today.strftime("%m/%d/%Y"),
                yesterday.strftime("%m/%d/%Y"),
                today.strftime("%B %d, %Y"),
                yesterday.strftime("%B %d, %Y")
            ]
            
            lines = page_text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Check if line contains a recent date
                contains_recent_date = any(date in line for date in recent_dates)
                
                if contains_recent_date:
                    # Look in this line and next few lines for numbers
                    search_lines = lines[i:i+5]
                    for search_line in search_lines:
                        numbers = re.findall(r'\b\d{1,2}\b', search_line)
                        if len(numbers) >= 3:  # At least 3 numbers found
                            try:
                                if base_game in ['Daily 3'] and len(numbers) >= 3:
                                    winning_numbers = [int(n) for n in numbers[:3]]
                                    draw_date = line
                                    break
                                elif base_game in ['Daily 4'] and len(numbers) >= 4:
                                    winning_numbers = [int(n) for n in numbers[:4]]
                                    draw_date = line
                                    break
                                elif base_game in ['Fantasy 5', 'Lotto 47'] and len(numbers) >= 5:
                                    winning_numbers = [int(n) for n in numbers[:5]]
                                    draw_date = line
                                    break
                                elif base_game in ['Powerball', 'Mega Millions'] and len(numbers) >= 6:
                                    winning_numbers = [int(n) for n in numbers[:6]]
                                    draw_date = line
                                    break
                            except ValueError:
                                continue
                    
                    if winning_numbers:
                        break
        
        if winning_numbers:
            return {
                'success': True,
                'numbers': winning_numbers,
                'draw_date': draw_date or 'Recent',
                'source': 'Michigan Lottery Official',
                'url': url,
                'game': base_game
            }
        else:
            return {
                'success': False,
                'error': f'No winning numbers found for {base_game} on Michigan Lottery website',
                'url': url,
                'suggestion': 'The numbers may not be posted yet, or the page structure may have changed'
            }
            
    except requests.RequestException as e:
        return {
            'success': False,
            'error': f'Network error accessing Michigan Lottery: {str(e)}',
            'url': url
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error parsing Michigan Lottery page: {str(e)}',
            'url': url
        }



def extract_winning_numbers_from_html(soup, game_name: str) -> Optional[Dict[str, Any]]:
    """Extract winning numbers from HTML content using multiple strategies"""
    import re
    
    # Get all text from the page
    all_text = soup.get_text()
    
    # Determine base game and draw time
    base_game = game_name
    draw_time = 'Evening'  # Default
    
    if 'Midday' in game_name:
        draw_time = 'Midday'
        base_game = game_name.replace(' Midday', '')
    elif 'Evening' in game_name:
        draw_time = 'Evening'
        base_game = game_name.replace(' Evening', '')
    
    # Strategy 1: Look for specific CSS selectors and patterns
    winning_number_selectors = [
        '.winning-numbers', '.numbers', '.draw-results', '.result-numbers',
        '[class*="number"]', '[class*="winning"]', '[class*="result"]',
        '.game-results', '.latest-draw', '.current-draw', '.latest-results',
        '.winning-number', '.result', '.draw-number'
    ]
    
    # Try CSS selectors first
    for selector in winning_number_selectors:
        elements = soup.select(selector)
        for element in elements:
            element_text = element.get_text().strip()
            
            if not element_text:
                continue
            
            # Check if this element corresponds to the correct draw time
            if draw_time == 'Midday' and 'evening' in element_text.lower():
                continue  # Skip evening results when looking for midday
            elif draw_time == 'Evening' and 'midday' in element_text.lower():
                continue  # Skip midday results when looking for evening
            
            result = parse_numbers_from_text(element_text, base_game, draw_time)
            if result:
                return result
    
    # Strategy 2: Fallback - search entire page text with improved patterns
    result = parse_numbers_from_text(all_text, base_game, draw_time)
    if result:
        return result
    
    # Strategy 3: Debug output for troubleshooting
    st.info(f"🔍 Debug: Could not parse winning numbers for {game_name}")
    st.expander("Debug Information", expanded=False).code(all_text[:500] + "..." if len(all_text) > 500 else all_text)
    
    return None

def parse_numbers_from_text(text: str, base_game: str, draw_time: str = 'Evening') -> Optional[Dict[str, Any]]:
    """Parse winning numbers from text using improved regex patterns"""
    import re
    
    if base_game == "Daily 4":
        # Multiple patterns for Daily 4 (fixed regex patterns)
        patterns = [
            r'\b(\d{4})\b',  # 4 digit sequence like "5739"
            r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b',  # Spaced digits like "5 7 3 9"
            r'\b(\d)-(\d)-(\d)-(\d)\b',  # Hyphenated like "5-7-3-9"
            r'\b(\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\b'  # Comma separated like "5,7,3,9"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                match = matches[0]
                if isinstance(match, str) and len(match) == 4:  # Single 4-digit number
                    numbers = [int(d) for d in match]
                elif isinstance(match, tuple) and len(match) == 4:  # Multiple captured groups
                    numbers = [int(d) for d in match if d.isdigit()]
                else:
                    continue
                
                if len(numbers) == 4 and all(0 <= n <= 9 for n in numbers):
                    return {'numbers': numbers, 'draw_time': draw_time}
    
    elif base_game == "Daily 3":
        # Multiple patterns for Daily 3 (fixed regex patterns)
        patterns = [
            r'\b(\d{3})\b',  # 3 digit sequence like "573"
            r'\b(\d)\s+(\d)\s+(\d)\b',  # Spaced digits like "5 7 3"
            r'\b(\d)-(\d)-(\d)\b',  # Hyphenated like "5-7-3"
            r'\b(\d)\s*,\s*(\d)\s*,\s*(\d)\b'  # Comma separated like "5,7,3"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                match = matches[0]
                if isinstance(match, str) and len(match) == 3:  # Single 3-digit number
                    numbers = [int(d) for d in match]
                elif isinstance(match, tuple) and len(match) == 3:  # Multiple captured groups
                    numbers = [int(d) for d in match if d.isdigit()]
                else:
                    continue
                
                if len(numbers) == 3 and all(0 <= n <= 9 for n in numbers):
                    return {'numbers': numbers, 'draw_time': draw_time}
    
    elif base_game == "Fantasy 5":
        # Look for sequences of 5 numbers between 1-39
        patterns = [
            r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b',  # Spaced
            r'\b(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})\b',  # Hyphenated
            r'\b(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})\b'  # Comma separated
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                numbers = [int(n) for n in matches[0]]
                if len(numbers) == 5 and all(1 <= n <= 39 for n in numbers):
                    return {'numbers': sorted(numbers)}
        
        # Alternative: Look for any 5 valid numbers in sequence
        all_numbers = re.findall(r'\b([1-3]?[0-9])\b', text)
        valid_nums = [int(n) for n in all_numbers if 1 <= int(n) <= 39]
        if len(valid_nums) >= 5:
            return {'numbers': sorted(valid_nums[:5])}
    
    elif base_game == "Lucky for Life":
        # Look for 5 main numbers (1-48) + 1 Lucky Ball (1-18)
        patterns = [
            r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b',  # 6 numbers spaced
            r'\b(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})\b',  # Hyphenated
            r'\b(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})\b'  # Comma separated
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                numbers = [int(n) for n in matches[0]]
                if len(numbers) == 6:
                    main_numbers = numbers[:5]
                    lucky_ball = numbers[5]
                    if all(1 <= n <= 48 for n in main_numbers) and 1 <= lucky_ball <= 18:
                        return {'numbers': sorted(main_numbers), 'lucky_ball': lucky_ball}
        
        # Alternative: Look for valid numbers in sequence
        all_numbers = re.findall(r'\b(\d{1,2})\b', text)
        valid_main = [int(n) for n in all_numbers if 1 <= int(n) <= 48]
        valid_lucky = [int(n) for n in all_numbers if 1 <= int(n) <= 18]
        if len(valid_main) >= 5 and len(valid_lucky) >= 1:
            return {'numbers': sorted(valid_main[:5]), 'lucky_ball': valid_lucky[0]}
    
    elif base_game == "Powerball":
        # Look for 5 main numbers (1-69) + 1 Powerball (1-26)
        pattern = r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b'
        matches = re.findall(pattern, text)
        if matches:
            all_nums = [int(n) for n in matches[0]]
            if len(all_nums) == 6:
                main_nums = all_nums[:5]
                powerball = all_nums[5]
                if all(1 <= n <= 69 for n in main_nums) and 1 <= powerball <= 26:
                    return {'numbers': sorted(main_nums), 'bonus_number': powerball}
    
    elif base_game == "Mega Millions":
        # Look for 5 main numbers (1-70) + 1 Mega Ball (1-25)
        pattern = r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b'
        matches = re.findall(pattern, text)
        if matches:
            all_nums = [int(n) for n in matches[0]]
            if len(all_nums) == 6:
                main_nums = all_nums[:5]
                mega_ball = all_nums[5]
                if all(1 <= n <= 70 for n in main_nums) and 1 <= mega_ball <= 25:
                    return {'numbers': sorted(main_nums), 'bonus_number': mega_ball}
    
    return None

def fetch_and_update_todays_numbers(game_name: str) -> bool:
    """Enhanced fetch with drawing schedule awareness and last drawing info"""
    
    # Use enhanced scraper
    scraped_data = enhanced_fetch_lottery_results_integrated(game_name)
    
    if not scraped_data['success']:
        st.error(f"❌ Could not fetch winning numbers for {game_name}")
        
        # Show helpful drawing schedule information
        if 'last_drawing_info' in scraped_data:
            st.info(f"📅 **Drawing Schedule Info:** {scraped_data['last_drawing_info']}")
        
        # Show whether today is a drawing day
        if 'schedule_info' in scraped_data:
            schedule = scraped_data['schedule_info']
            if not schedule['is_drawing_day']:
                st.warning(f"⚠️ **Note:** Today ({schedule['day_name']}) is not a drawing day for {game_name}")
                st.info(f"�️ **Next Drawing:** {schedule['next_drawing']}")
                st.info(f"🎯 **Drawing Frequency:** {schedule['frequency']}")
        
        st.info("�💡 **Alternative options:**")
        st.info("1. Try again later - the website might be temporarily unavailable")
        st.info("2. Use the manual entry section below to add winning numbers")
        st.info("3. Check if today is a drawing day for this game")
        return False
    
    # Success! Show the results with enhanced information
    st.success(f"✅ Successfully fetched winning numbers for {game_name}: {scraped_data['numbers']}")
    
    # Show detailed drawing information
    if 'last_drawing_info' in scraped_data:
        st.info(f"📅 **{scraped_data['last_drawing_info']}**")
    
    # Show recent results if available
    if 'all_results' in scraped_data and len(scraped_data['all_results']) > 1:
        with st.expander("📊 View Recent Results", expanded=False):
            st.markdown("**Recent winning numbers:**")
            for i, result in enumerate(scraped_data['all_results'][:5]):
                numbers_str = ", ".join(map(str, result['numbers']))
                st.markdown(f"• **{result['date']}** ({result['day']}): {numbers_str} (raw: {result['raw_number']})")
    
    # Save to log
    try:
        save_winning_numbers_to_log(
            game_name, 
            scraped_data['numbers'], 
            scraped_data.get('bonus_number'),
            scraped_data['date'],
            scraped_data.get('time', 'Unknown')
        )
        st.success("💾 Numbers saved to log files")
    except Exception as e:
        st.error(f"❌ Error saving to log: {str(e)}")
        return False
    
    # Update performance metrics if we have previous predictions
    performance_metrics = load_performance_metrics()
    
    # Check if we had predictions for today
    history = load_predictions_history()
    today_predictions = [
        p for p in history["predictions"] 
        if p["game"] == game_name and p["date"] == scraped_data['date'] and not p["verified"]
    ]
    
    if today_predictions:
        # Update predictions with actual results
        for prediction in today_predictions:
            prediction["verified"] = True
            prediction["actual_results"] = {
                "numbers": scraped_data['numbers'],
                "bonus_number": scraped_data.get('bonus_number'),
                "date": scraped_data['date']
            }
            
            # Calculate performance scores
            strategy_scores = {}
            for strategy_name, strategy_data in prediction["predictions"].items():
                if "numbers" in strategy_data:
                    accuracy = calculate_prediction_accuracy(
                        strategy_data["numbers"],
                        scraped_data['numbers'],
                        strategy_data.get("bonus"),
                        scraped_data.get('bonus_number')
                    )
                    strategy_scores[strategy_name] = accuracy
            
            prediction["performance_scores"] = strategy_scores
        
        # Save updated predictions
        save_predictions_history(history)
        
        # Update learning weights
        update_learning_weights(performance_metrics, game_name)
    
    return True

def update_learning_weights(performance_metrics: Dict[str, Any], game_name: str):
    """Update learning weights based on strategy performance"""
    weights = load_learning_weights()
    
    if game_name in performance_metrics.get("strategy_performance", {}):
        strategy_perfs = performance_metrics["strategy_performance"][game_name]
        adaptation_rate = weights["adaptation_rate"]
        
        for strategy_name, perf_data in strategy_perfs.items():
            if strategy_name in weights["strategy_weights"] and perf_data["total_predictions"] > 0:
                avg_accuracy = perf_data["total_accuracy"] / perf_data["total_predictions"]
                recent_avg = sum(perf_data["recent_accuracy"]) / len(perf_data["recent_accuracy"]) if perf_data["recent_accuracy"] else avg_accuracy
                
                # Adjust weight based on recent performance vs overall average
                if recent_avg > avg_accuracy:
                    # Recent performance is better, increase weight
                    weights["strategy_weights"][strategy_name] = min(2.0, weights["strategy_weights"][strategy_name] + adaptation_rate)
                elif recent_avg < avg_accuracy * 0.8:
                    # Recent performance is significantly worse, decrease weight
                    weights["strategy_weights"][strategy_name] = max(0.1, weights["strategy_weights"][strategy_name] - adaptation_rate)
    
    save_learning_weights(weights)
    return weights

# Simple config
LOTTERY_GAMES = {
    'Powerball': {
        'numbers_count': 5,
        'numbers_range': (1, 69),
        'bonus_range': (1, 26),
        'draw_days': ['Monday', 'Wednesday', 'Saturday']
    },
    'Mega Millions': {
        'numbers_count': 5,
        'numbers_range': (1, 70),
        'bonus_range': (1, 25),
        'draw_days': ['Tuesday', 'Friday']
    },
    'Lotto 47': {
        'numbers_count': 6,
        'numbers_range': (1, 47),
        'draw_days': ['Wednesday', 'Saturday']
    },
    'Fantasy 5': {
        'numbers_count': 5,
        'numbers_range': (1, 39),
        'draw_days': ['Daily']
    },
    'Fantasy 5 Double Play': {
        'numbers_count': 5,
        'numbers_range': (1, 39),
        'draw_days': ['Daily']
    },
    'Lucky for Life': {
        'numbers_count': 5,
        'numbers_range': (1, 48),
        'bonus_range': (1, 18),
        'draw_days': ['Monday', 'Thursday']
    },
    'Daily 3 Midday': {
        'numbers_count': 3,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'draw_time': 'Midday',
        'base_game': 'Daily 3'
    },
    'Daily 3 Evening': {
        'numbers_count': 3,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'draw_time': 'Evening',
        'base_game': 'Daily 3'
    },
    'Daily 4 Midday': {
        'numbers_count': 4,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'draw_time': 'Midday',
        'base_game': 'Daily 4'
    },
    'Daily 4 Evening': {
        'numbers_count': 4,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'draw_time': 'Evening',
        'base_game': 'Daily 4'
    },
    # Legacy entries for backward compatibility
    'Daily 3': {
        'numbers_count': 3,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'draw_time': 'Evening',
        'base_game': 'Daily 3'
    },
    'Daily 4': {
        'numbers_count': 4,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'draw_time': 'Evening',
        'base_game': 'Daily 4'
    }
}

def get_base_game(game_name: str) -> str:
    """Get the base game name for a given game (e.g., 'Daily 3 Midday' -> 'Daily 3')"""
    if game_name in LOTTERY_GAMES and 'base_game' in LOTTERY_GAMES[game_name]:
        return LOTTERY_GAMES[game_name]['base_game']
    return game_name

def is_daily_game(game_name: str) -> bool:
    """Check if a game is a Daily 3 or Daily 4 game (including midday/evening variants)"""
    base_game = get_base_game(game_name)
    return base_game in ['Daily 3', 'Daily 4']

def get_daily_game_variants(base_game: str) -> List[str]:
    """Get all variants of a daily game (e.g., 'Daily 3' -> ['Daily 3 Midday', 'Daily 3 Evening'])"""
    variants = []
    for game_name, config in LOTTERY_GAMES.items():
        if config.get('base_game') == base_game:
            variants.append(game_name)
    return variants

# Configure Streamlit page
st.set_page_config(
    page_title="🎰 Michigan Lottery Analyzer",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_csv_lottery_data(game_name: str, days: int = 30) -> List[Dict]:
    """Get lottery data from CSV files using the new scraper"""
    scraper = get_csv_scraper()
    if not scraper:
        return []
    
    try:
        # Get recent history from CSV
        history = scraper.get_recent_history(game_name, days)
        if not history:
            return []
        
        # Convert to the expected format
        results = []
        for entry in history:
            # Parse the date 
            try:
                draw_date_str = entry['draw_date']
                # Try to parse different date formats
                for fmt in ['%A, %b %d, %Y', '%a, %b %d, %Y', '%B %d, %Y', '%b %d, %Y']:
                    try:
                        draw_date = datetime.strptime(draw_date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matches, use current date
                    draw_date = datetime.now()
                
                # Create result entry
                result = {
                    'date': draw_date,
                    'numbers': entry['numbers'],
                    'game': game_name
                }
                
                # Handle bonus numbers
                if game_name == "Lucky for Life" and len(entry['numbers']) >= 6:
                    result['main_numbers'] = entry['numbers'][:5]
                    result['bonus_number'] = entry['numbers'][5]
                elif game_name in ["Powerball", "Mega Millions"] and len(entry['numbers']) >= 6:
                    result['main_numbers'] = entry['numbers'][:5]
                    result['bonus_number'] = entry['numbers'][5]
                else:
                    result['main_numbers'] = entry['numbers']
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error processing CSV entry for {game_name}: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"⚠️ Error getting CSV data for {game_name}: {e}")
        return []

def get_lottery_data(game_name: str, days: int = 30) -> List[Dict]:
    """Get lottery data, prioritizing CSV data, then logged results, then sample data"""
    
    # First, try to get data from CSV files
    csv_data = get_csv_lottery_data(game_name, days)
    if csv_data:
        print(f"📊 Using {len(csv_data)} results from CSV for {game_name}")
        return csv_data
    
    # Second, try to load actual winning numbers from log
    logged_results = load_winning_numbers_from_log(game_name, days)
    
    if logged_results and len(logged_results) >= days:
        # We have enough logged data
        print(f"📊 Using {len(logged_results)} actual winning numbers from log for {game_name}")
        return logged_results[:days]
    elif logged_results:
        # We have some logged data, supplement with sample data
        remaining_days = days - len(logged_results)
        sample_data = generate_sample_data(game_name, remaining_days)
        
        # Adjust sample data dates to not overlap with logged data
        if logged_results:
            oldest_logged_date = min(result['date'] for result in logged_results)
            for i, sample in enumerate(sample_data):
                sample['date'] = oldest_logged_date - timedelta(days=i+1)
        
        combined_results = logged_results + sample_data
        print(f"📊 Using {len(logged_results)} actual + {len(sample_data)} sample results for {game_name}")
        return combined_results[:days]
    else:
        # No logged data, use sample data
        print(f"📊 No actual data found, using sample data for {game_name}")
        return generate_sample_data(game_name, days)

def generate_sample_data(game_name: str, days: int = 30) -> List[Dict]:
    """Generate sample lottery data for demonstration"""
    import random
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Powerball'])
    results = []
    
    current_date = datetime.now()
    
    for i in range(days):
        draw_date = current_date - timedelta(days=i)
        
        # Generate random numbers
        numbers_count = game_config['numbers_count']
        numbers_range = game_config['numbers_range']
        
        if is_daily_game(game_name):
            # For Daily games, allow duplicates
            numbers = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(numbers_count)]
        else:
            # For lottery games, no duplicates
            numbers = random.sample(
                range(numbers_range[0], numbers_range[1] + 1),
                numbers_count
            )
        
        bonus_number = None
        if 'bonus_range' in game_config:
            bonus_range = game_config['bonus_range']
            bonus_number = random.randint(bonus_range[0], bonus_range[1])
        
        jackpot = None
        if game_name in ['Powerball', 'Mega Millions']:
            jackpot = random.uniform(10_000_000, 500_000_000)
        
        result = {
            'date': draw_date,
            'numbers': sorted(numbers) if not is_daily_game(game_name) else numbers,
            'bonus_number': bonus_number,
            'jackpot': jackpot
        }
        
        results.append(result)
    
    return results

def analyze_frequency(results: List[Dict]) -> Dict[str, Any]:
    """Analyze number frequency from results"""
    all_numbers = []
    for result in results:
        # Handle different data structures - CSV data vs sample data
        if 'numbers' in result:
            all_numbers.extend(result['numbers'])
        elif 'main_numbers' in result:
            all_numbers.extend(result['main_numbers'])
            # Also include bonus number if present
            if result.get('bonus_number') is not None:
                all_numbers.append(result['bonus_number'])
    
    if not all_numbers:
        return {}
    
    frequency = Counter(all_numbers)
    avg_frequency = statistics.mean(frequency.values())
    
    # Adjust thresholds based on the number range (for Daily 3/4 games with limited digits)
    unique_numbers = len(frequency)
    if unique_numbers <= 10:  # Daily 3/4 games (digits 0-9)
        # Determine if this is Daily 3 or Daily 4 based on total numbers collected
        total_numbers = len(all_numbers)
        
        # Daily 3: 3 positions × draws = ~1095 numbers for 365 draws
        # Daily 4: 4 positions × draws = ~1460 numbers for 365 draws
        if total_numbers <= 1200:  # Daily 3 games
            hot_count = 3  # Top 3 numbers
            cold_count = 3  # Bottom 3 numbers
            game_type = "Daily 3"
        else:  # Daily 4 games
            hot_count = 4  # Top 4 numbers
            cold_count = 4  # Bottom 4 numbers
            game_type = "Daily 4"
        
        # Sort by frequency and select top/bottom numbers
        sorted_by_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        hot_numbers = [num for num, freq in sorted_by_freq[:hot_count]]
        cold_numbers = [num for num, freq in sorted_by_freq[-cold_count:]]
        

        
        return {
            'frequency': dict(frequency),
            'hot_numbers': sorted(hot_numbers),
            'cold_numbers': sorted(cold_numbers),
            'most_common': frequency.most_common(10),
            'least_common': frequency.most_common()[:-11:-1],
            'total_draws': len(results)
        }
    else:  # Larger lottery games
        # Use original thresholds for games with wider ranges
        hot_threshold = avg_frequency * 1.2   # 20% above average
        cold_threshold = avg_frequency * 0.8   # 20% below average
    
    hot_numbers = [num for num, freq in frequency.items() if freq >= hot_threshold]
    cold_numbers = [num for num, freq in frequency.items() if freq <= cold_threshold]
    
    return {
        'frequency': dict(frequency),
        'hot_numbers': sorted(hot_numbers),
        'cold_numbers': sorted(cold_numbers),
        'most_common': frequency.most_common(10),
        'least_common': frequency.most_common()[:-11:-1],
        'total_draws': len(results)
    }

def calculate_days_since_last_win(number: int, results: List[Dict]) -> str:
    """Calculate days since a number last appeared in winning draws"""
    if not results:
        return "No data"
    
    # Find the most recent appearance
    for i, result in enumerate(results):
        if number in result['numbers'] or (result.get('bonus_number') == number):
            if i == 0:
                return "0 (Latest draw)"
            else:
                # Calculate days based on position in results (assuming daily draws)
                return f"{i} draws ago"
    
    # If number never appeared
    return f">{len(results)} draws"

def analyze_deep_patterns(results: List[Dict], game_name: str) -> Dict[str, Any]:
    """Deep pattern analysis of lottery numbers"""
    if not results:
        return {}
    
    # Initialize pattern tracking
    patterns = {
        'consecutive_pairs': Counter(),
        'sum_ranges': Counter(),
        'odd_even_patterns': Counter(),
        'high_low_patterns': Counter(),
        'number_gaps': Counter(),
        'digit_sum_patterns': Counter(),
        'position_analysis': {},
        'last_digit_patterns': Counter(),
        'repeat_patterns': {},
        'number_sequences': Counter(),
        'quadrant_analysis': Counter(),
        'parity_patterns': Counter()
    }
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Powerball'])
    numbers_range = game_config.get('numbers_range', (1, 39))
    max_number = numbers_range[1]
    mid_point = (numbers_range[0] + numbers_range[1]) // 2
    
    # Analyze each draw
    for i, result in enumerate(results):
        numbers = sorted(result['numbers'])
        
        # 1. Consecutive pairs analysis
        for j in range(len(numbers) - 1):
            if numbers[j+1] - numbers[j] == 1:
                patterns['consecutive_pairs'][f"{numbers[j]}-{numbers[j+1]}"] += 1
        
        # 2. Sum analysis
        total_sum = sum(numbers)
        sum_range = f"{(total_sum // 10) * 10}-{(total_sum // 10) * 10 + 9}"
        patterns['sum_ranges'][sum_range] += 1
        
        # 3. Odd/Even pattern
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        patterns['odd_even_patterns'][f"{odd_count}O-{even_count}E"] += 1
        
        # 4. High/Low pattern
        high_count = sum(1 for n in numbers if n > mid_point)
        low_count = len(numbers) - high_count
        patterns['high_low_patterns'][f"{low_count}L-{high_count}H"] += 1
        
        # 5. Number gaps analysis
        gaps = [numbers[j+1] - numbers[j] for j in range(len(numbers) - 1)]
        avg_gap = statistics.mean(gaps) if gaps else 0
        gap_range = f"{int(avg_gap//2)*2}-{int(avg_gap//2)*2+1}"
        patterns['number_gaps'][gap_range] += 1
        
        # 6. Digit sum patterns
        digit_sum = sum(int(digit) for num in numbers for digit in str(num))
        digit_sum_range = f"{(digit_sum // 5) * 5}-{(digit_sum // 5) * 5 + 4}"
        patterns['digit_sum_patterns'][digit_sum_range] += 1
        
        # 7. Position analysis
        for pos, num in enumerate(numbers):
            if pos not in patterns['position_analysis']:
                patterns['position_analysis'][pos] = Counter()
            patterns['position_analysis'][pos][num] += 1
        
        # 8. Last digit patterns
        last_digits = [num % 10 for num in numbers]
        last_digit_pattern = '-'.join(map(str, sorted(set(last_digits))))
        patterns['last_digit_patterns'][last_digit_pattern] += 1
        
        # 9. Repeat patterns (numbers from previous draw)
        if i > 0:
            prev_numbers = set(results[i-1]['numbers'])
            current_numbers = set(numbers)
            repeats = len(prev_numbers.intersection(current_numbers))
            if repeats not in patterns['repeat_patterns']:
                patterns['repeat_patterns'][repeats] = 0
            patterns['repeat_patterns'][repeats] += 1
        
        # 10. Number sequences (arithmetic progressions)
        for seq_len in range(3, min(6, len(numbers) + 1)):
            for start_idx in range(len(numbers) - seq_len + 1):
                subseq = numbers[start_idx:start_idx + seq_len]
                if len(subseq) >= 3:
                    diffs = [subseq[j+1] - subseq[j] for j in range(len(subseq) - 1)]
                    if len(set(diffs)) == 1:  # Arithmetic progression
                        patterns['number_sequences'][f"AP-{diffs[0]}-{seq_len}"] += 1
        
        # 11. Quadrant analysis (divide number range into 4 quadrants)
        range_size = max_number - numbers_range[0] + 1
        quadrant_size = range_size // 4
        quadrants = [0, 0, 0, 0]
        for num in numbers:
            quadrant = min(3, (num - numbers_range[0]) // quadrant_size)
            quadrants[quadrant] += 1
        quadrant_pattern = '-'.join(map(str, quadrants))
        patterns['quadrant_analysis'][quadrant_pattern] += 1
        
        # 12. Parity patterns (divisibility patterns)
        div_by_3 = sum(1 for n in numbers if n % 3 == 0)
        div_by_5 = sum(1 for n in numbers if n % 5 == 0)
        patterns['parity_patterns'][f"3:{div_by_3},5:{div_by_5}"] += 1
    
    return patterns

def analyze_number_relationships(results: List[Dict]) -> Dict[str, Any]:
    """Analyze relationships between numbers in winning combinations"""
    if not results:
        return {}
    
    relationships = {
        'number_pairs': Counter(),
        'number_triplets': Counter(),
        'distance_patterns': Counter(),
        'sum_combinations': Counter(),
        'ratio_patterns': Counter()
    }
    
    for result in results:
        numbers = sorted(result['numbers'])
        
        # Analyze pairs
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                pair = f"{numbers[i]}-{numbers[j]}"
                relationships['number_pairs'][pair] += 1
                
                # Distance between pairs
                distance = numbers[j] - numbers[i]
                relationships['distance_patterns'][distance] += 1
        
        # Analyze triplets
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                for k in range(j + 1, len(numbers)):
                    triplet = f"{numbers[i]}-{numbers[j]}-{numbers[k]}"
                    relationships['number_triplets'][triplet] += 1
        
        # Sum combinations
        total_sum = sum(numbers)
        relationships['sum_combinations'][total_sum] += 1
        
        # Ratio patterns (first/last number ratio)
        if len(numbers) >= 2:
            ratio = round(numbers[-1] / numbers[0], 1) if numbers[0] != 0 else 0
            relationships['ratio_patterns'][ratio] += 1
    
    return relationships

def analyze_temporal_patterns(results: List[Dict]) -> Dict[str, Any]:
    """Analyze temporal patterns in lottery draws"""
    if not results:
        return {}
    
    temporal_patterns = {
        'day_of_week': Counter(),
        'month_patterns': Counter(),
        'seasonal_trends': {},
        'time_gaps': Counter(),
        'streak_analysis': {},
        'cyclical_patterns': Counter()
    }
    
    # Sort results by date
    sorted_results = sorted(results, key=lambda x: x['date'])
    
    for i, result in enumerate(sorted_results):
        date = result['date']
        numbers = result['numbers']
        
        # Day of week patterns
        day_name = date.strftime('%A')
        temporal_patterns['day_of_week'][day_name] += 1
        
        # Month patterns
        month_name = date.strftime('%B')
        temporal_patterns['month_patterns'][month_name] += 1
        
        # Seasonal trends
        month = date.month
        season = 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Fall'
        if season not in temporal_patterns['seasonal_trends']:
            temporal_patterns['seasonal_trends'][season] = {'draws': 0, 'avg_sum': 0, 'number_freq': Counter()}
        
        temporal_patterns['seasonal_trends'][season]['draws'] += 1
        temporal_patterns['seasonal_trends'][season]['avg_sum'] += sum(numbers)
        for num in numbers:
            temporal_patterns['seasonal_trends'][season]['number_freq'][num] += 1
        
        # Time gaps between specific numbers
        if i > 0:
            prev_numbers = set(sorted_results[i-1]['numbers'])
            current_numbers = set(numbers)
            common_numbers = prev_numbers.intersection(current_numbers)
            gap_days = (date - sorted_results[i-1]['date']).days
            for num in common_numbers:
                temporal_patterns['time_gaps'][f"num_{num}_gap_{gap_days}"] += 1
        
        # Cyclical patterns (week number)
        week_num = date.isocalendar()[1]
        temporal_patterns['cyclical_patterns'][f"week_{week_num % 4}"] += 1
    
    # Calculate averages for seasonal trends
    for season in temporal_patterns['seasonal_trends']:
        draws = temporal_patterns['seasonal_trends'][season]['draws']
        if draws > 0:
            temporal_patterns['seasonal_trends'][season]['avg_sum'] /= draws
    
    return temporal_patterns

def create_insights(analysis: Dict[str, Any], game: str, deep_patterns: Optional[Dict[str, Any]] = None, relationships: Optional[Dict[str, Any]] = None, temporal: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate comprehensive insights from analysis"""
    insights = []
    
    if analysis.get('total_draws', 0) > 0:
        insights.append(f"Analysis based on {analysis['total_draws']} lottery draws for {game}")
    
    hot_numbers = analysis.get('hot_numbers', [])
    if hot_numbers:
        insights.append(f"Hot numbers (frequently drawn): {', '.join(map(str, hot_numbers[:5]))}")
    
    cold_numbers = analysis.get('cold_numbers', [])
    if cold_numbers:
        insights.append(f"Cold numbers (rarely drawn): {', '.join(map(str, cold_numbers[:5]))}")
    
    most_common = analysis.get('most_common', [])
    if most_common:
        num, freq = most_common[0]
        insights.append(f"Most frequently drawn number: {num} (appeared {freq} times)")
    
    # Deep pattern insights
    if deep_patterns:
        # Consecutive pairs
        if deep_patterns.get('consecutive_pairs'):
            most_common_pair = deep_patterns['consecutive_pairs'].most_common(1)[0]
            insights.append(f"Most common consecutive pair: {most_common_pair[0]} (appeared {most_common_pair[1]} times)")
        
        # Odd/Even patterns
        if deep_patterns.get('odd_even_patterns'):
            most_common_oe = deep_patterns['odd_even_patterns'].most_common(1)[0]
            formatted_pattern = format_odd_even_pattern(most_common_oe[0])
            insights.append(f"Most common odd/even pattern: {formatted_pattern} (appeared {most_common_oe[1]} times)")
        
        # Sum range patterns
        if deep_patterns.get('sum_ranges'):
            most_common_sum = deep_patterns['sum_ranges'].most_common(1)[0]
            insights.append(f"Most common sum range: {most_common_sum[0]} (appeared {most_common_sum[1]} times)")
        
        # High/Low patterns
        if deep_patterns.get('high_low_patterns'):
            most_common_hl = deep_patterns['high_low_patterns'].most_common(1)[0]
            formatted_pattern = format_high_low_pattern(most_common_hl[0])
            insights.append(f"Most common high/low pattern: {formatted_pattern} (appeared {most_common_hl[1]} times)")
    
    # Relationship insights
    if relationships:
        # Most common pairs
        if relationships.get('number_pairs'):
            top_pair = relationships['number_pairs'].most_common(1)[0]
            insights.append(f"Most frequent number pair: {top_pair[0]} (appeared together {top_pair[1]} times)")
        
        # Distance patterns
        if relationships.get('distance_patterns'):
            common_distance = relationships['distance_patterns'].most_common(1)[0]
            insights.append(f"Most common distance between numbers: {common_distance[0]} (occurred {common_distance[1]} times)")
    
    # Temporal insights
    if temporal:
        # Day patterns
        if temporal.get('day_of_week'):
            lucky_day = temporal['day_of_week'].most_common(1)[0]
            insights.append(f"Most frequent draw day in sample: {lucky_day[0]} ({lucky_day[1]} draws)")
        
        # Seasonal trends
        if temporal.get('seasonal_trends'):
            season_data = temporal['seasonal_trends']
            if season_data:
                best_season = max(season_data.items(), key=lambda x: x[1]['draws'])[0]
                insights.append(f"Most active season in sample: {best_season}")
    
    # Add AI-generated insights if local LLM is available
    if local_pipeline or OLLAMA_AVAILABLE:
        ai_insights = generate_ai_insights(analysis, game, deep_patterns, relationships, temporal)
        insights.extend(ai_insights)
    
    return insights

def generate_ai_insights(analysis: Dict[str, Any], game: str, deep_patterns: Optional[Dict[str, Any]] = None, relationships: Optional[Dict[str, Any]] = None, temporal: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate AI-powered insights using local LLM"""
    global local_pipeline
    
    # Check available LLM options
    if not (local_pipeline or OLLAMA_AVAILABLE):
        return ["Local LLM not available. Install PyTorch + Transformers or Ollama for AI insights."]
    
    try:
        hot_numbers = analysis.get('hot_numbers', [])
        cold_numbers = analysis.get('cold_numbers', [])
        total_draws = analysis.get('total_draws', 0)
        most_common = analysis.get('most_common', [])
        
        # Build enhanced prompt with deep pattern data
        prompt_parts = [
            f"Statistical Analysis of {game} Lottery Data:",
            f"Total draws: {total_draws}",
            f"Hot numbers: {hot_numbers[:5]}",
            f"Cold numbers: {cold_numbers[:5]}",
            f"Most common: {[num for num, freq in most_common[:3]]}"
        ]
        
        # Add deep pattern information
        if deep_patterns:
            if deep_patterns.get('consecutive_pairs'):
                top_consecutive = deep_patterns['consecutive_pairs'].most_common(1)[0]
                prompt_parts.append(f"Top consecutive pair: {top_consecutive[0]} ({top_consecutive[1]}x)")
            
            if deep_patterns.get('odd_even_patterns'):
                top_oe = deep_patterns['odd_even_patterns'].most_common(1)[0]
                formatted_oe = format_odd_even_pattern(top_oe[0])
                prompt_parts.append(f"Top odd/even pattern: {formatted_oe} ({top_oe[1]}x)")
            
            if deep_patterns.get('sum_ranges'):
                top_sum = deep_patterns['sum_ranges'].most_common(1)[0]
                prompt_parts.append(f"Common sum range: {top_sum[0]} ({top_sum[1]}x)")
        
        # Add relationship information
        if relationships:
            if relationships.get('number_pairs'):
                top_pair = relationships['number_pairs'].most_common(1)[0]
                prompt_parts.append(f"Frequent pair: {top_pair[0]} ({top_pair[1]}x)")
        
        # Add temporal information
        if temporal:
            if temporal.get('day_of_week'):
                top_day = temporal['day_of_week'].most_common(1)[0]
                prompt_parts.append(f"Most active day: {top_day[0]} ({top_day[1]} draws)")
        
        prompt_parts.append("\nProvide 3 brief statistical observations about these patterns:")
        prompt = "\n".join(prompt_parts)
        
        # Try local pipeline first, then Ollama
        if local_pipeline:
            try:
                response = local_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + 100,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=local_tokenizer.eos_token_id if local_tokenizer and hasattr(local_tokenizer, 'eos_token_id') else None
                )
                
                ai_response = response[0]['generated_text'][len(prompt):].strip()
                
            except Exception as e:
                logger.error(f"Local pipeline error: {e}")
                ai_response = "Local LLM processing error"
                
        elif OLLAMA_AVAILABLE:
            full_prompt = f"You are a statistical analyst. {prompt}\n\nAnalysis:"
            ai_response = query_ollama(full_prompt, max_tokens=200)
        else:
            ai_response = "No AI model available"
        
        # Parse response into insights
        if ai_response and ai_response != "Local LLM processing error":
            # Split into individual insights and clean them
            lines = ai_response.split('\n')
            ai_insights = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Filter out very short lines
                    # Remove common prefixes
                    for prefix in ['1.', '2.', '3.', '-', '•', '*']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                    
                    if line and len(ai_insights) < 3:
                        ai_insights.append(line)
            
            return ai_insights if ai_insights else ["Statistical patterns analyzed by local AI"]
        else:
            # Fallback: Generate rule-based insights
            fallback_insights = []
            
            if hot_numbers:
                fallback_insights.append(f"Numbers {', '.join(map(str, hot_numbers[:3]))} show higher frequency in recent draws")
            
            if deep_patterns and deep_patterns.get('consecutive_pairs'):
                top_pair = deep_patterns['consecutive_pairs'].most_common(1)[0]
                fallback_insights.append(f"Consecutive numbers {top_pair[0]} appear together {top_pair[1]} times")
            
            if relationships and relationships.get('number_pairs'):
                top_pair = relationships['number_pairs'].most_common(1)[0]
                fallback_insights.append(f"Number pair {top_pair[0]} shows strong correlation ({top_pair[1]} occurrences)")
            
            return fallback_insights[:3] if fallback_insights else ["Pattern analysis completed"]
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        return ["AI insights temporarily unavailable - using local processing"]

def chat_with_ai(question: str, analysis: Dict[str, Any], game: str, deep_patterns: Optional[Dict[str, Any]] = None, relationships: Optional[Dict[str, Any]] = None, temporal: Optional[Dict[str, Any]] = None) -> str:
    """Chat with AI about lottery analysis using local LLM"""
    global local_pipeline
    
    if not (local_pipeline or OLLAMA_AVAILABLE):
        return "Local AI chat requires PyTorch + Transformers or Ollama. Please install dependencies."
    
    try:
        hot_numbers = analysis.get('hot_numbers', [])
        cold_numbers = analysis.get('cold_numbers', [])
        total_draws = analysis.get('total_draws', 0)
        
        # Build comprehensive context
        context_parts = [
            f"Lottery Analysis Context for {game}:",
            f"Total draws: {total_draws}",
            f"Hot numbers: {hot_numbers[:5]}",
            f"Cold numbers: {cold_numbers[:5]}"
        ]
        
        # Add deep pattern context
        if deep_patterns:
            if deep_patterns.get('consecutive_pairs'):
                top_pair = deep_patterns['consecutive_pairs'].most_common(1)[0]
                context_parts.append(f"Top consecutive pair: {top_pair[0]} ({top_pair[1]}x)")
            
            if deep_patterns.get('odd_even_patterns'):
                top_oe = deep_patterns['odd_even_patterns'].most_common(1)[0]
                formatted_oe = format_odd_even_pattern(top_oe[0])
                context_parts.append(f"Top odd/even pattern: {formatted_oe}")
        
        # Add relationship context
        if relationships:
            if relationships.get('number_pairs'):
                top_pair = relationships['number_pairs'].most_common(1)[0]
                context_parts.append(f"Top number pair: {top_pair[0]} ({top_pair[1]}x)")
        
        context = "\n".join(context_parts)
        
        # Build chat prompt
        chat_prompt = f"{context}\n\nUser Question: {question}\n\nAssistant Response:"
        
        # Try local pipeline first, then Ollama
        if local_pipeline:
            try:
                response = local_pipeline(
                    chat_prompt,
                    max_length=len(chat_prompt.split()) + 80,
                    num_return_sequences=1,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=local_tokenizer.eos_token_id if local_tokenizer and hasattr(local_tokenizer, 'eos_token_id') else None
                )
                
                ai_response = response[0]['generated_text'][len(chat_prompt):].strip()
                
                # Clean up response
                if ai_response:
                    # Take first meaningful sentence/paragraph
                    sentences = ai_response.split('.')
                    if sentences:
                        clean_response = sentences[0].strip()
                        if len(clean_response) > 10:
                            return clean_response + "."
                
                return "Based on the analysis, I can help explain the statistical patterns."
                
            except Exception as e:
                logger.error(f"Local chat error: {e}")
                
        elif OLLAMA_AVAILABLE:
            full_prompt = f"You are a lottery statistical assistant. {chat_prompt}"
            return query_ollama(full_prompt, max_tokens=150)
        
        # Fallback response based on question content
        question_lower = question.lower()
        
        if 'hot' in question_lower or 'frequent' in question_lower:
            return f"The hot numbers in this analysis are {', '.join(map(str, hot_numbers[:5]))}. These numbers appeared more frequently than average in the {total_draws} draws analyzed."
        
        elif 'cold' in question_lower or 'rare' in question_lower:
            return f"The cold numbers are {', '.join(map(str, cold_numbers[:5]))}. These numbers appeared less frequently in recent draws."
        
        elif 'pattern' in question_lower:
            if deep_patterns and deep_patterns.get('consecutive_pairs'):
                top_pair = deep_patterns['consecutive_pairs'].most_common(1)[0]
                return f"The most notable pattern is consecutive numbers {top_pair[0]} appearing together {top_pair[1]} times."
        
        elif 'pair' in question_lower:
            if relationships and relationships.get('number_pairs'):
                top_pair = relationships['number_pairs'].most_common(1)[0]
                return f"The most frequent number pair is {top_pair[0]}, appearing together {top_pair[1]} times in the data."
        
        else:
            return f"Based on {total_draws} draws of {game}, I can analyze patterns in hot/cold numbers, consecutive pairs, and statistical trends. What specific aspect interests you?"
    
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        return f"I encountered an error processing your question. Please try rephrasing it."

def generate_pattern_based_suggestions(analysis: Dict[str, Any], deep_patterns: Dict[str, Any], relationships: Dict[str, Any], game_name: str) -> Dict[str, Any]:
    """Generate number suggestions based on deep pattern analysis with learning weights"""
    import random
    
    # For Daily games, use the specialized digit-based function
    if is_daily_game(game_name):
        return generate_daily_digit_suggestions(analysis, game_name)
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Powerball'])
    numbers_count = game_config['numbers_count']
    numbers_range = game_config['numbers_range']
    all_numbers = list(range(numbers_range[0], numbers_range[1] + 1))
    
    # Load learning weights to adjust strategy effectiveness
    weights = load_learning_weights()
    strategy_weights = weights.get("strategy_weights", {})
    
    # Pattern-based strategies
    suggestions = {}
    
    # Strategy 1: Consecutive Pair Strategy
    consecutive_pairs = deep_patterns.get('consecutive_pairs', {})
    if consecutive_pairs:
        pattern_numbers = []
        # Get most common consecutive pairs
        top_pairs = consecutive_pairs.most_common(3)
        for pair_str, _ in top_pairs:
            if '-' in pair_str:
                num1, num2 = map(int, pair_str.split('-'))
                pattern_numbers.extend([num1, num2])
        
        # Fill remaining slots
        while len(pattern_numbers) < numbers_count:
            remaining = [n for n in all_numbers if n not in pattern_numbers]
            if remaining:
                pattern_numbers.append(random.choice(remaining))
            else:
                break
        
        # Adjust confidence based on learning weights
        weight = strategy_weights.get('consecutive_pattern', 1.0)
        confidence = 'High' if weight > 1.5 else 'Medium' if weight > 0.8 else 'Low'
        
        suggestions['consecutive_pattern'] = {
            'numbers': sorted(pattern_numbers[:numbers_count]),
            'strategy': 'Consecutive Pattern',
            'description': 'Based on most frequent consecutive number pairs in historical data',
            'confidence': confidence,
            'weight': weight
        }
    
    # Strategy 2: Optimal Sum Range Strategy
    sum_ranges = deep_patterns.get('sum_ranges', {})
    if sum_ranges:
        # Find most common sum range
        top_sum_range = sum_ranges.most_common(1)[0][0]
        if '-' in top_sum_range:
            min_sum, max_sum = map(int, top_sum_range.split('-'))
            target_sum = (min_sum + max_sum) // 2
            
            # Generate numbers that approximate the target sum
            sum_numbers = []
            current_sum = 0
            available = all_numbers.copy()
            
            for _ in range(numbers_count):
                if not available:
                    break
                
                remaining_picks = numbers_count - len(sum_numbers)
                if remaining_picks > 0:
                    target_per_pick = (target_sum - current_sum) / remaining_picks
                    # Find number closest to target
                    closest = min(available, key=lambda x: abs(x - target_per_pick))
                    sum_numbers.append(closest)
                    current_sum += closest
                    available.remove(closest)
            
            weight = strategy_weights.get('sum_optimized', 1.0)
            confidence = 'High' if weight > 1.5 else 'Medium' if weight > 0.8 else 'Low'
            
            suggestions['sum_optimized'] = {
                'numbers': sorted(sum_numbers),
                'strategy': 'Sum Optimization',
                'description': f'Targets the most common sum range: {top_sum_range}',
                'confidence': confidence,
                'weight': weight
            }
    
    # Strategy 3: Odd/Even Balance Strategy
    oe_patterns = deep_patterns.get('odd_even_patterns', {})
    if oe_patterns:
        top_oe_pattern = oe_patterns.most_common(1)[0][0]
        # Parse pattern like "3O-2E"
        if 'O-' in top_oe_pattern and 'E' in top_oe_pattern:
            odd_count = int(top_oe_pattern.split('O-')[0])
            even_count = int(top_oe_pattern.split('O-')[1].split('E')[0])
            
            odds = [n for n in all_numbers if n % 2 == 1]
            evens = [n for n in all_numbers if n % 2 == 0]
            
            oe_numbers = []
            oe_numbers.extend(random.sample(odds, min(odd_count, len(odds))))
            oe_numbers.extend(random.sample(evens, min(even_count, len(evens))))
            
            # Fill any remaining slots
            while len(oe_numbers) < numbers_count:
                remaining = [n for n in all_numbers if n not in oe_numbers]
                if remaining:
                    oe_numbers.append(random.choice(remaining))
                else:
                    break
            
            weight = strategy_weights.get('odd_even_balanced', 1.0)
            confidence = 'High' if weight > 1.5 else 'Medium' if weight > 0.8 else 'Low'
            
            suggestions['odd_even_balanced'] = {
                'numbers': sorted(oe_numbers[:numbers_count]),
                'strategy': 'Odd/Even Balance',
                'description': f'Follows most common odd/even pattern: {top_oe_pattern}',
                'confidence': confidence,
                'weight': weight
            }
    
    # Strategy 4: Relationship-Based Strategy
    number_pairs = relationships.get('number_pairs', {})
    if number_pairs:
        rel_numbers = []
        top_pairs = number_pairs.most_common(min(3, numbers_count // 2))
        
        for pair_str, _ in top_pairs:
            if '-' in pair_str:
                num1, num2 = map(int, pair_str.split('-'))
                if num1 not in rel_numbers:
                    rel_numbers.append(num1)
                if num2 not in rel_numbers and len(rel_numbers) < numbers_count:
                    rel_numbers.append(num2)
        
        # Fill remaining
        while len(rel_numbers) < numbers_count:
            remaining = [n for n in all_numbers if n not in rel_numbers]
            if remaining:
                rel_numbers.append(random.choice(remaining))
            else:
                break
        
        weight = strategy_weights.get('relationship_based', 1.0)
        confidence = 'High' if weight > 1.5 else 'Medium-High' if weight > 1.2 else 'Medium' if weight > 0.8 else 'Low'
        
        suggestions['relationship_based'] = {
            'numbers': sorted(rel_numbers[:numbers_count]),
            'strategy': 'Relationship Pattern',
            'description': 'Based on most frequently paired numbers in historical draws',
            'confidence': confidence,
            'weight': weight
        }
    
    # Add bonus numbers if applicable
    bonus_number = None
    if 'bonus_range' in game_config:
        bonus_range = game_config['bonus_range']
        bonus_number = random.randint(bonus_range[0], bonus_range[1])
    
    # Add bonus number to all strategies
    for strategy in suggestions.values():
        strategy['bonus'] = bonus_number
    
    return suggestions

def generate_daily_digit_suggestions(analysis: Dict[str, Any], game_name: str) -> Dict[str, Any]:
    """Generate digit-based suggestions specifically for Daily 3 and Daily 4 games"""
    import random
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Powerball'])
    digits_count = game_config['numbers_count']  # 3 for Daily 3, 4 for Daily 4
    
    # Load learning weights
    weights = load_learning_weights()
    strategy_weights = weights.get("strategy_weights", {})
    
    hot_numbers = analysis.get('hot_numbers', [])
    cold_numbers = analysis.get('cold_numbers', [])
    frequency = analysis.get('frequency', {})
    
    suggestions = {}
    
    # Strategy 1: Balanced Mix (each position independent)
    balanced_digits = [random.randint(0, 9) for _ in range(digits_count)]
    suggestions['balanced_mix'] = {
        'numbers': balanced_digits,
        'strategy': 'Balanced Mix',
        'description': 'Even distribution across all positions',
        'confidence': 'Medium',
        'weight': strategy_weights.get('balanced_mix', 1.0)
    }
    
    # Strategy 2: Hot Numbers Focus (favor frequently drawn digits)
    hot_digits = []
    for _ in range(digits_count):
        if hot_numbers and random.random() < 0.7:  # 70% chance to use hot digit
            hot_digits.append(random.choice(hot_numbers[:5]))
        else:
            hot_digits.append(random.randint(0, 9))
    suggestions['hot_numbers_focus'] = {
        'numbers': hot_digits,
        'strategy': 'Hot Numbers Focus',
        'description': 'Emphasizes frequently drawn digits',
        'confidence': 'High',
        'weight': strategy_weights.get('hot_numbers_focus', 1.2)
    }
    
    # Strategy 3: Due Theory (cold numbers)
    due_digits = []
    for _ in range(digits_count):
        if cold_numbers and random.random() < 0.6:  # 60% chance to use cold digit
            due_digits.append(random.choice(cold_numbers[:5]))
        else:
            due_digits.append(random.randint(0, 9))
    suggestions['due_theory'] = {
        'numbers': due_digits,
        'strategy': 'Due Theory',
        'description': 'Focus on overdue digits',
        'confidence': 'Medium',
        'weight': strategy_weights.get('due_theory', 0.9)
    }
    
    # Strategy 4: Repeating Digits Pattern
    repeat_digits = [random.randint(0, 9) for _ in range(digits_count)]
    # Force at least one repeat for demonstration
    if digits_count >= 2:
        repeat_pos = random.randint(1, digits_count - 1)
        repeat_digits[repeat_pos] = repeat_digits[0]
    suggestions['repeating_pattern'] = {
        'numbers': repeat_digits,
        'strategy': 'Repeating Pattern',
        'description': 'Includes duplicate digits (common in daily games)',
        'confidence': 'Medium',
        'weight': strategy_weights.get('repeating_pattern', 1.1)
    }
    
    # Strategy 5: Sequential Pattern
    start_digit = random.randint(0, 9 - digits_count + 1)
    sequential_digits = [(start_digit + i) % 10 for i in range(digits_count)]
    suggestions['sequential_pattern'] = {
        'numbers': sequential_digits,
        'strategy': 'Sequential Pattern',
        'description': 'Consecutive or wrapped digit sequence',
        'confidence': 'Low',
        'weight': strategy_weights.get('sequential_pattern', 0.8)
    }
    
    # Strategy 6: Mirror Pattern (for 4-digit games)
    if digits_count == 4:
        d1, d2 = random.randint(0, 9), random.randint(0, 9)
        mirror_digits = [d1, d2, d2, d1]
        suggestions['mirror_pattern'] = {
            'numbers': mirror_digits,
            'strategy': 'Mirror Pattern',
            'description': 'Symmetrical digit arrangement (ABBA)',
            'confidence': 'Low',
            'weight': strategy_weights.get('mirror_pattern', 0.7)
        }
    
    # Strategy 7: All Same Digit (rare but valid)
    same_digit = random.randint(0, 9)
    same_digits = [same_digit] * digits_count
    suggestions['all_same'] = {
        'numbers': same_digits,
        'strategy': 'All Same',
        'description': f'All positions same digit ({same_digit})',
        'confidence': 'Very Low',
        'weight': strategy_weights.get('all_same', 0.5)
    }
    
    # Strategy 8: Mixed Frequency Blend
    mixed_digits = []
    for i in range(digits_count):
        if i % 2 == 0 and hot_numbers:
            mixed_digits.append(random.choice(hot_numbers[:3]))
        elif cold_numbers:
            mixed_digits.append(random.choice(cold_numbers[:3]))
        else:
            mixed_digits.append(random.randint(0, 9))
    suggestions['mixed_frequency'] = {
        'numbers': mixed_digits,
        'strategy': 'Mixed Frequency',
        'description': 'Alternating hot and cold digits by position',
        'confidence': 'Medium',
        'weight': strategy_weights.get('mixed_frequency', 1.0)
    }
    
    return suggestions

def generate_number_suggestions(analysis: Dict[str, Any], game_name: str) -> Dict[str, Any]:
    """Generate smart number suggestions based on analysis with learning weights"""
    import random
    
    # Use specialized function for Daily games
    if is_daily_game(game_name):
        return generate_daily_digit_suggestions(analysis, game_name)
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Powerball'])
    numbers_count = game_config['numbers_count']
    numbers_range = game_config['numbers_range']
    
    # Load learning weights
    weights = load_learning_weights()
    strategy_weights = weights.get("strategy_weights", {})
    
    hot_numbers = analysis.get('hot_numbers', [])
    cold_numbers = analysis.get('cold_numbers', [])
    frequency = analysis.get('frequency', {})
    
    # Strategy 1: Balanced Mix (recommended)
    balanced_numbers = []
    all_numbers = list(range(numbers_range[0], numbers_range[1] + 1))
    
    # Add 2-3 hot numbers if available
    if hot_numbers:
        balanced_numbers.extend(random.sample(hot_numbers[:10], min(3, len(hot_numbers), numbers_count)))
    
    # Add 1-2 cold numbers for balance
    if cold_numbers and len(balanced_numbers) < numbers_count:
        remaining_slots = min(2, numbers_count - len(balanced_numbers))
        balanced_numbers.extend(random.sample(cold_numbers[:10], min(remaining_slots, len(cold_numbers))))
    
    # Fill remaining with medium frequency numbers
    if len(balanced_numbers) < numbers_count:
        used_numbers = set(balanced_numbers)
        medium_numbers = [n for n in all_numbers if n not in hot_numbers and n not in cold_numbers and n not in used_numbers]
        remaining_needed = numbers_count - len(balanced_numbers)
        if medium_numbers:
            balanced_numbers.extend(random.sample(medium_numbers, min(remaining_needed, len(medium_numbers))))
    
    # Strategy 2: Hot Numbers Focus
    hot_focus = random.sample(hot_numbers[:15] if hot_numbers else all_numbers, min(numbers_count, len(hot_numbers[:15]) if hot_numbers else len(all_numbers)))
    while len(hot_focus) < numbers_count:
        remaining = [n for n in all_numbers if n not in hot_focus]
        if remaining:
            hot_focus.append(random.choice(remaining))
        else:
            break
    
    # Strategy 3: Cold Numbers (Due Theory)
    cold_focus = random.sample(cold_numbers[:15] if cold_numbers else all_numbers, min(numbers_count, len(cold_numbers[:15]) if cold_numbers else len(all_numbers)))
    while len(cold_focus) < numbers_count:
        remaining = [n for n in all_numbers if n not in cold_focus]
        if remaining:
            cold_focus.append(random.choice(remaining))
        else:
            break
    
    # Strategy 4: Random Statistical
    random_numbers = random.sample(all_numbers, numbers_count)
    
    # Generate bonus number if applicable
    bonus_number = None
    if 'bonus_range' in game_config:
        bonus_range = game_config['bonus_range']
        bonus_number = random.randint(bonus_range[0], bonus_range[1])
    
    # Adjust confidence based on learning weights
    balanced_weight = strategy_weights.get('balanced_mix', 1.0)
    hot_weight = strategy_weights.get('hot_focus', 1.0)
    cold_weight = strategy_weights.get('cold_theory', 1.0)
    
    suggestions = {
        'balanced_mix': {
            'numbers': sorted(balanced_numbers[:numbers_count]),
            'bonus': bonus_number,
            'strategy': 'Balanced Mix',
            'description': 'Combines hot numbers, cold numbers, and medium frequency numbers for balance',
            'confidence': 'High' if balanced_weight > 1.5 else 'Medium' if balanced_weight > 0.8 else 'Low',
            'weight': balanced_weight
        },
        'hot_focus': {
            'numbers': sorted(hot_focus),
            'bonus': bonus_number,
            'strategy': 'Hot Numbers Focus',
            'description': 'Emphasizes frequently drawn numbers based on recent patterns',
            'confidence': 'High' if hot_weight > 1.5 else 'Medium' if hot_weight > 0.8 else 'Low',
            'weight': hot_weight
        },
        'cold_theory': {
            'numbers': sorted(cold_focus),
            'bonus': bonus_number,
            'strategy': 'Due Theory',
            'description': 'Focuses on "due" numbers that haven\'t appeared recently',
            'confidence': 'High' if cold_weight > 1.5 else 'Medium' if cold_weight > 0.8 else 'Low',
            'weight': cold_weight
        },
        'random_statistical': {
            'numbers': sorted(random_numbers),
            'bonus': bonus_number,
            'strategy': 'Statistical Random',
            'description': 'Mathematically random selection from valid range',
            'confidence': 'Equal',
            'weight': 1.0
        }
    }
    
    return suggestions

def test_daily_4_scraping():
    """Debug function to test Daily 4 scraping"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://www.michiganlottery.com/draw-games/daily-4"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Testing scraping for Daily 4...")
        print(f"URL: {url}")
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"Page title: {soup.title.string if soup.title else 'No title'}")
            
            # Look for common patterns
            all_text = soup.get_text()
            print(f"Page text preview (first 500 chars): {all_text[:500]}")
            
            # Test the extraction function
            result = extract_winning_numbers_from_html(soup, "Daily 4")
            print(f"Extraction result: {result}")
            
            return result
        else:
            print(f"Failed to fetch page: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error testing scraping: {str(e)}")
        return None

def main():
    """Main application entry point"""
    
    # App header
    st.title("🎰 Michigan Lottery Results Analyzer")
    st.markdown("*LLM-powered pattern analysis and prediction system with learning capabilities*")
    
    # Auto-initialize LLM on startup if available
    if 'llm_initialized' not in st.session_state:
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            with st.spinner("🚀 Initializing Local AI..."):
                success = initialize_local_llm()
                if success:
                    st.success("✅ Local AI ready for enhanced analysis!")
                    st.session_state.llm_initialized = True
                else:
                    st.session_state.llm_initialized = False
        else:
            st.session_state.llm_initialized = False
    
    # Daily prediction verification system
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Check if we need to verify yesterday's predictions
    if 'last_verification_date' not in st.session_state or st.session_state.last_verification_date != today:
        st.session_state.last_verification_date = today
        
        # Initialize verification results in session state
        if 'verification_results' not in st.session_state:
            st.session_state.verification_results = {}
        
        # Auto-verify predictions for all games
        with st.expander("🔍 Daily Learning System Update", expanded=False):
            st.markdown("**Checking yesterday's predictions against actual results...**")
            
            total_verified = 0
            for game_name in LOTTERY_GAMES.keys():
                verification_results = verify_previous_predictions(game_name)
                if verification_results:
                    st.session_state.verification_results[game_name] = verification_results
                    total_verified += len(verification_results)
                    
                    # Update learning weights
                    performance_metrics = load_performance_metrics()
                    update_learning_weights(performance_metrics, game_name)
            
            if total_verified > 0:
                st.success(f"✅ Verified {total_verified} predictions and updated learning weights")
            else:
                st.info("No predictions from yesterday to verify")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Selection")
        
        # Game selector
        selected_game = st.selectbox(
            "Choose a lottery game:",
            list(LOTTERY_GAMES.keys()),
            index=0
        )
        
        # Analysis parameters
        st.header("📊 Analysis Settings")
        
        days_to_analyze = st.slider(
            "Days of history to analyze:",
            min_value=7,
            max_value=3650,  # 10 years of data
            value=3650,      # Default to maximum (10 years)
            step=30          # Monthly increments for easier selection
        )
        
        # Quick time period selection
        st.markdown("**Quick Select:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("📅 30 Days"):
                st.session_state.days_override = 30
        with col2:
            if st.button("📅 6 Months"):
                st.session_state.days_override = 180
        with col3:
            if st.button("📅 1 Year"):
                st.session_state.days_override = 365
        with col4:
            if st.button("📅 5 Years"):
                st.session_state.days_override = 1825
        with col5:
            if st.button("📅 10 Years"):
                st.session_state.days_override = 3650
        
        # Use override value if set, otherwise use slider
        if hasattr(st.session_state, 'days_override'):
            days_to_analyze = st.session_state.days_override
            st.info(f"🎯 **Selected**: {days_to_analyze} days (use slider to adjust)")
        
        # Display selected time period info with performance guidance
        years = days_to_analyze / 365.25
        if years >= 1:
            st.info(f"📊 Analyzing approximately **{years:.1f} years** of lottery data ({days_to_analyze} days)")
        else:
            months = days_to_analyze / 30.44
            st.info(f"📊 Analyzing approximately **{months:.1f} months** of lottery data ({days_to_analyze} days)")
        
        # Performance guidance for large datasets
        if days_to_analyze > 1825:  # More than 5 years
            st.warning("⚡ **Large Dataset Notice**: Analysis of 5+ years may take longer but provides more comprehensive patterns. GPU acceleration will help with AI features.")
        elif days_to_analyze > 365:  # More than 1 year
            st.success("🎯 **Optimal Range**: 1-5 years provides excellent pattern detection with good performance.")
        else:
            st.info("🚀 **Quick Analysis**: Shorter periods analyze faster but may miss long-term patterns.")
        
        # Data source information
        st.info("📊 **Data Source**: This application uses simulated historical lottery data for demonstration purposes. Patterns and analysis are based on realistic lottery number distributions.")
        
        # Local LLM Configuration
        st.header("🔧 Local AI Configuration")
        
        # Display GPU/CUDA status with Python 3.11 information
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.success(f"🚀 CUDA GPU Ready: {gpu_name}")
                st.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                st.info(f"⚡ PyTorch {torch.__version__} with CUDA acceleration enabled")
            except Exception as e:
                st.success("✅ CUDA GPU Available")
                st.info(f"⚠️ GPU properties unavailable: {str(e)}")
        elif CUDA_AVAILABLE and not TORCH_AVAILABLE:
            st.warning("🎯 NVIDIA GPU Hardware Detected")
            st.info("⚙️ **Recommendation**: Upgrade to Python 3.11 for full CUDA support")
            
            with st.expander("🔧 How to Enable Full GPU Acceleration"):
                st.markdown("""
                **Current Status**: GPU hardware detected but PyTorch not working
                
                **Solution**: Migrate to Python 3.11
                
                **Quick Setup**:
                1. Run `setup_python311.bat` in the project folder
                2. Or follow the `PYTHON_311_MIGRATION.md` guide
                
                **Benefits**:
                - 🚀 10-100x faster AI processing
                - 🤖 Full local LLM chat features
                - 💾 Efficient GPU memory usage
                - ⚡ All transformers libraries working
                """)
        else:
            st.warning("⚠️ CUDA not available - using CPU only")
            st.info("💡 For GPU acceleration, ensure NVIDIA drivers and Python 3.11 are installed")
        
        # LLM initialization status and manual controls
        if st.session_state.get('llm_initialized', False) and local_pipeline:
            st.success("🤖 Local LLM: Auto-initialized and Ready")
        elif TORCH_AVAILABLE and CUDA_AVAILABLE:
            if st.button("🚀 Re-initialize Local LLM", help="Manually reload AI model for insights"):
                with st.spinner("Loading local LLM..."):
                    success = initialize_local_llm()
                    if success:
                        st.success("✅ Local LLM ready!")
                        st.session_state.llm_initialized = True
                    else:
                        st.error("❌ Failed to load LLM")
                        st.session_state.llm_initialized = False
        else:
            if st.button("🚀 Initialize Local LLM", help="Load AI model for insights (requires PyTorch + CUDA)"):
                st.warning("⚠️ PyTorch + CUDA required for Local LLM")
        
        # Show LLM status
        if local_pipeline:
            st.success("🤖 Local LLM: Ready")
        elif OLLAMA_AVAILABLE:
            st.success("🦙 Ollama: Available")
        else:
            st.info("💡 Install PyTorch + Transformers or Ollama for AI features")
        
        # Analysis button
        if st.button("🔍 Run Analysis", type="primary"):
            st.session_state['run_analysis'] = True
            
            # Print to console for Daily 3 and Daily 4 games
            if is_daily_game(selected_game):
                print(f"\n🎯 STARTING {selected_game.upper()} ANALYSIS")
                print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
                print(f"Game: {selected_game}")
                print(f"Analysis Period: {days_to_analyze} days")
                print(f"Generating suggestions...")
                print(f"-" * 40)
        
        # Daily verification button
        if st.button("📊 Check Yesterday's Predictions", help="Verify predictions against actual results"):
            with st.spinner("Verifying predictions..."):
                verification_results = verify_previous_predictions(selected_game)
                st.session_state.verification_results[selected_game] = verification_results
                
                if verification_results:
                    st.success(f"✅ Verified {len(verification_results)} predictions")
                    for result in verification_results:
                        if result["best_strategy"]:
                            strategy_name, strategy_score = result["best_strategy"]
                            st.write(f"Best strategy: {strategy_name} ({strategy_score['accuracy_percentage']:.1f}% accuracy)")
                else:
                    st.info("No predictions to verify for yesterday")
        
        # Show learning status
        st.header("🧠 Learning System")
        weights = load_learning_weights()
        performance_metrics = load_performance_metrics()
        
        if selected_game in performance_metrics.get("strategy_performance", {}):
            strategy_perfs = performance_metrics["strategy_performance"][selected_game]
            best_strategies = sorted(
                [(name, perf["total_accuracy"] / perf["total_predictions"]) 
                 for name, perf in strategy_perfs.items() if perf["total_predictions"] > 0],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            if best_strategies:
                st.markdown("**🏆 Top Performing Strategies:**")
                for i, (strategy, accuracy) in enumerate(best_strategies, 1):
                    st.markdown(f"{i}. {strategy}: {accuracy:.1f}% avg accuracy")
        else:
            st.info("🌱 Learning system will improve predictions over time")
        
        # Historical Data Processing Section
        st.header("🗃️ Historical Data (OCR)")
        
        if OCR_AVAILABLE:
            st.success("✅ OCR system available")
            
            # Year selection for historical data
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.selectbox("Start Year:", range(2000, 2022), index=0)
            with col2:
                end_year = st.selectbox("End Year:", range(2000, 2022), index=21)
            
            # Load historical data button
            if st.button("📄 Load Historical Data", help="Load lottery data from PDF files using OCR"):
                with st.spinner("🔍 Processing historical PDFs with OCR..."):
                    # Get years in range
                    years_to_process = list(range(start_year, end_year + 1))
                    
                    # Load historical data
                    historical_data = load_historical_data_from_pdfs(years_to_process)
                    
                    if historical_data:
                        st.session_state['historical_data'] = historical_data
                        
                        # Show summary
                        total_entries = sum(len(entries) for entries in historical_data.values())
                        st.success(f"✅ Loaded {total_entries} historical entries from {len(years_to_process)} years")
                        
                        # Show data breakdown
                        st.markdown("**📊 Data Summary:**")
                        for game, entries in historical_data.items():
                            if entries:
                                st.markdown(f"- **{game}**: {len(entries)} entries")
                                
                    else:
                        st.warning("❌ No historical data could be loaded from PDFs")
            
            # Show loaded historical data status
            if 'historical_data' in st.session_state:
                historical_data = st.session_state['historical_data']
                total_entries = sum(len(entries) for entries in historical_data.values())
                st.info(f"📚 {total_entries} historical entries loaded and available for analysis")
                
                # Clear historical data button
                if st.button("🗑️ Clear Historical Data"):
                    del st.session_state['historical_data']
                    st.success("✅ Historical data cleared")
                    st.rerun()
            
        else:
            st.warning("❌ OCR system not available")
            st.info("Historical PDF data cannot be processed without OCR libraries")
    
    # Analysis results tabs (full width to avoid column nesting issues)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 Analysis", "📊 Visualization", "🎯 Number Suggestions", "🧠 Learning Dashboard", "🤖 Chat", "🔍 Number Checker", "💬 Winning Numbers Chat", "🌐 MI Lottery Web"])
    
    with tab1:
            # Always try to run analysis - it will check for CSV data internally
            should_run_analysis = True  # Always attempt analysis since get_lottery_data handles CSV/sample fallback
            
            if should_run_analysis:
                with st.spinner(f"Analyzing {selected_game} data..."):
                    # Get lottery data (CSV or sample)
                    results = get_lottery_data(selected_game, days_to_analyze)
                    
                    # Perform comprehensive analysis
                    analysis = analyze_frequency(results)
                    deep_patterns = analyze_deep_patterns(results, selected_game)
                    relationships = analyze_number_relationships(results)
                    temporal_patterns = analyze_temporal_patterns(results)
                    
                    # Generate comprehensive insights
                    insights = create_insights(analysis, selected_game, deep_patterns, relationships, temporal_patterns)
                    
                    # Display comprehensive metrics
                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    
                    with col_m1:
                        st.metric("Total Draws", analysis.get('total_draws', 0))
                    
                    with col_m2:
                        st.metric("Hot Numbers", len(analysis.get('hot_numbers', [])))
                    
                    with col_m3:
                        st.metric("Cold Numbers", len(analysis.get('cold_numbers', [])))
                    
                    with col_m4:
                        pattern_count = len(deep_patterns.get('consecutive_pairs', {}))
                        st.metric("Consecutive Pairs", pattern_count)
                    
                    with col_m5:
                        pair_count = len(relationships.get('number_pairs', {}))
                        st.metric("Number Pairs", min(pair_count, 99))
                    
                    # Pattern Strength Indicator
                    st.subheader("📊 Pattern Strength Analysis")
                    
                    # Calculate pattern strength scores
                    strength_scores = {}
                    
                    # Consecutive pairs strength
                    consecutive_pairs = deep_patterns.get('consecutive_pairs', {})
                    if consecutive_pairs:
                        max_consecutive = max(consecutive_pairs.values()) if consecutive_pairs else 0
                        strength_scores['Consecutive Pairs'] = min(100, (max_consecutive / analysis.get('total_draws', 1)) * 100 * 5)
                    
                    # Odd/Even pattern consistency
                    oe_patterns = deep_patterns.get('odd_even_patterns', {})
                    if oe_patterns:
                        max_oe = max(oe_patterns.values()) if oe_patterns else 0
                        strength_scores['Odd/Even Patterns'] = min(100, (max_oe / analysis.get('total_draws', 1)) * 100 * 2)
                    
                    # Number pair relationships
                    number_pairs = relationships.get('number_pairs', {})
                    if number_pairs:
                        max_pair = max(number_pairs.values()) if number_pairs else 0
                        strength_scores['Number Relationships'] = min(100, (max_pair / analysis.get('total_draws', 1)) * 100 * 10)
                    
                    # Sum range consistency
                    sum_ranges = deep_patterns.get('sum_ranges', {})
                    if sum_ranges:
                        max_sum = max(sum_ranges.values()) if sum_ranges else 0
                        strength_scores['Sum Patterns'] = min(100, (max_sum / analysis.get('total_draws', 1)) * 100 * 3)
                    
                    # Display strength scores
                    if strength_scores:
                        strength_cols = st.columns(len(strength_scores))
                        for i, (pattern_name, score) in enumerate(strength_scores.items()):
                            with strength_cols[i]:
                                st.metric(
                                    pattern_name,
                                    f"{score:.1f}%",
                                    delta="Strong" if score > 50 else "Moderate" if score > 25 else "Weak"
                                )
                    
                    # Insights section
                    st.subheader("🔍 Comprehensive Analysis Insights")
                    
                    for i, insight in enumerate(insights, 1):
                        st.markdown(f"**{i}.** {insight}")
                    
                    # Add pattern reliability note
                    if strength_scores:
                        avg_strength = sum(strength_scores.values()) / len(strength_scores)
                        if avg_strength > 60:
                            st.success("🎯 **Strong patterns detected** - Historical data shows consistent trends")
                        elif avg_strength > 30:
                            st.warning("⚖️ **Moderate patterns detected** - Some trends visible but mixed results")
                        else:
                            st.info("🎲 **Weak patterns detected** - Data appears largely random as expected")
                    
                    # Deep Pattern Analysis Section
                    st.subheader("🧬 Deep Pattern Analysis")
                    
                    # Create tabs for different pattern types
                    pattern_tab1, pattern_tab2, pattern_tab3, pattern_tab4 = st.tabs(["🔗 Pairs & Sequences", "📊 Distribution Patterns", "🔢 Number Relationships", "📅 Temporal Patterns"])
                    
                    with pattern_tab1:
                        col_p1, col_p2 = st.columns(2)
                        
                        with col_p1:
                            st.markdown("**Most Common Consecutive Pairs:**")
                            consecutive_pairs = deep_patterns.get('consecutive_pairs', {})
                            if consecutive_pairs:
                                top_pairs = dict(consecutive_pairs.most_common(5))
                                for pair, count in top_pairs.items():
                                    st.markdown(f"• {pair}: {count} times")
                            else:
                                st.info("No consecutive pairs found")
                        
                        with col_p2:
                            st.markdown("**Number Sequences (Arithmetic Progressions):**")
                            sequences = deep_patterns.get('number_sequences', {})
                            if sequences:
                                top_sequences = dict(sequences.most_common(3))
                                for seq, count in top_sequences.items():
                                    st.markdown(f"• {seq}: {count} times")
                            else:
                                st.info("No arithmetic sequences found")
                    
                    with pattern_tab2:
                        col_d1, col_d2 = st.columns(2)
                        
                        with col_d1:
                            st.markdown("**Odd/Even Distribution:**")
                            oe_patterns = deep_patterns.get('odd_even_patterns', {})
                            if oe_patterns:
                                for pattern, count in oe_patterns.most_common(3):
                                    formatted_pattern = format_odd_even_pattern(pattern)
                                    st.markdown(f"• {formatted_pattern}: {count} times")
                        
                        with col_d2:
                            st.markdown("**High/Low Distribution:**")
                            hl_patterns = deep_patterns.get('high_low_patterns', {})
                            if hl_patterns:
                                for pattern, count in hl_patterns.most_common(3):
                                    formatted_pattern = format_high_low_pattern(pattern)
                                    st.markdown(f"• {formatted_pattern}: {count} times")
                        
                        st.markdown("**Sum Range Analysis:**")
                        sum_ranges = deep_patterns.get('sum_ranges', {})
                        if sum_ranges:
                            sum_df = pd.DataFrame(
                                [(range_val, count) for range_val, count in sum_ranges.most_common(10)],
                                columns=['Sum Range', 'Frequency']
                            )
                            # Display as table to avoid Altair compatibility issues
                            st.dataframe(sum_df, use_container_width=True)
                    
                    with pattern_tab3:
                        col_r1, col_r2 = st.columns(2)
                        
                        with col_r1:
                            st.markdown("**Most Frequent Number Pairs:**")
                            number_pairs = relationships.get('number_pairs', {})
                            if number_pairs:
                                for pair, count in number_pairs.most_common(5):
                                    st.markdown(f"• {pair}: {count} times")
                        
                        with col_r2:
                            st.markdown("**Common Distances Between Numbers:**")
                            distances = relationships.get('distance_patterns', {})
                            if distances:
                                for distance, count in distances.most_common(5):
                                    st.markdown(f"• Gap of {distance}: {count} times")
                    
                    with pattern_tab4:
                        col_t1, col_t2 = st.columns(2)
                        
                        with col_t1:
                            st.markdown("**Day of Week Patterns:**")
                            day_patterns = temporal_patterns.get('day_of_week', {})
                            if day_patterns:
                                for day, count in day_patterns.most_common():
                                    st.markdown(f"• {day}: {count} draws")
                        
                        with col_t2:
                            st.markdown("**Seasonal Trends:**")
                            seasonal = temporal_patterns.get('seasonal_trends', {})
                            if seasonal:
                                for season, data in seasonal.items():
                                    avg_sum = round(data['avg_sum'], 1)
                                    st.markdown(f"• {season}: {data['draws']} draws (avg sum: {avg_sum})")
                    
                    # Number grids
                    st.subheader("🔥 Hot Numbers")
                    hot_numbers = analysis.get('hot_numbers', [])
                    

                    
                    if hot_numbers:
                        # Display hot numbers in a grid
                        cols = st.columns(min(len(hot_numbers), 10))
                        for i, num in enumerate(hot_numbers[:10]):
                            with cols[i]:
                                st.button(f"{num}", key=f"hot_{num}")
                    else:
                        st.info("No hot numbers identified in current analysis")
                    
                    st.subheader("🧊 Cold Numbers") 
                    cold_numbers = analysis.get('cold_numbers', [])
                    

                    
                    if cold_numbers:
                        # Display cold numbers in a grid
                        cols = st.columns(min(len(cold_numbers), 10))
                        for i, num in enumerate(cold_numbers[:10]):
                            with cols[i]:
                                st.button(f"{num}", key=f"cold_{num}")
                    else:
                        st.info("No cold numbers identified in current analysis")
                
            else:
                st.info("⏳ Analysis complete. Hot and cold numbers should be visible above.")
    
    with tab2:
            st.subheader("📊 Data Visualization")
            
            # Always attempt visualization since get_lottery_data handles CSV/sample fallback
            if True:
                # Create a simple frequency chart using Streamlit's built-in charting
                results = get_lottery_data(selected_game, days_to_analyze)
                analysis = analyze_frequency(results)
                
                frequency_data = analysis.get('frequency', {})
                if frequency_data:
                    # Convert to DataFrame for display
                    df = pd.DataFrame(
                        list(frequency_data.items()),
                        columns=['Number', 'Frequency']
                    )
                    
                    # Display as table to avoid Altair compatibility issues
                    st.dataframe(df.sort_values('Frequency', ascending=False), use_container_width=True)
                    
                    # Most/Least common numbers
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Most Common Numbers")
                        most_common = analysis.get('most_common', [])[:5]
                        most_common_df = pd.DataFrame(most_common, columns=['Number', 'Frequency'])
                        st.dataframe(most_common_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("Least Common Numbers")
                        least_common = analysis.get('least_common', [])[:5]
                        least_common_df = pd.DataFrame(least_common, columns=['Number', 'Frequency'])
                        st.dataframe(least_common_df, use_container_width=True)
            else:
                st.info("Run analysis first to see visualizations")
    
    with tab3:
            st.subheader("🎯 Smart Number Suggestions")
            
            # Always attempt suggestions since get_lottery_data handles CSV/sample fallback
            if True:
                # Generate lottery data and analysis (prioritizes actual logged results)
                results = get_lottery_data(selected_game, days_to_analyze)
                analysis = analyze_frequency(results)
                deep_patterns = analyze_deep_patterns(results, selected_game)
                relationships = analyze_number_relationships(results)
                
                # Generate both traditional and pattern-based suggestions
                traditional_suggestions = generate_number_suggestions(analysis, selected_game)
                pattern_suggestions = generate_pattern_based_suggestions(analysis, deep_patterns, relationships, selected_game)
                
                # Combine suggestions
                all_suggestions = {**traditional_suggestions, **pattern_suggestions}
                
                # Print to console for Daily 3 and Daily 4 games
                if is_daily_game(selected_game):
                    console_output = print_daily_game_suggestions(selected_game, all_suggestions, days_to_analyze)
                    print(f"\n{console_output}\n")
                    
                    # Quick reference summary
                    print("🎯 QUICK REFERENCE - SUGGESTED NUMBERS:")
                    for i, (strategy_key, strategy_data) in enumerate(all_suggestions.items(), 1):
                        numbers_str = ''.join(map(str, strategy_data['numbers']))
                        print(f"  {i}. {numbers_str} ({strategy_data['strategy']})")
                    print(f"  Total: {len(all_suggestions)} different {selected_game} suggestions\n")
                
                st.markdown("**Based on statistical analysis of past draws, here are suggested numbers to play:**")
                st.markdown("*⚠️ Remember: All lottery drawings are random events. These suggestions are for entertainment only.*")
                
                # Store predictions for later verification
                prediction_id = store_prediction(selected_game, all_suggestions, analysis)
                st.success(f"🎯 Predictions stored for verification (ID: {prediction_id})")
                
                # Display suggestions
                for i, (strategy_key, strategy_data) in enumerate(all_suggestions.items(), 1):
                    # Get performance weight if available
                    weight = strategy_data.get('weight', 1.0)
                    weight_indicator = " 🔥" if weight > 1.5 else " ⚡" if weight > 1.2 else " 📈" if weight > 0.8 else " 📉"
                    
                    with st.expander(f"🎲 Strategy {i}: {strategy_data['strategy']} (Confidence: {strategy_data['confidence']}){weight_indicator}"):
                        
                        # Display performance info
                        if weight != 1.0:
                            st.markdown(f"**Learning Weight:** {weight:.2f} {'(Improved)' if weight > 1.0 else '(Needs improvement)'}")
                        
                        # Display numbers
                        st.markdown(f"**Numbers to play:** {', '.join(map(str, strategy_data['numbers']))}")
                        
                        # Display bonus number if applicable
                        if strategy_data.get('bonus'):
                            game_config = LOTTERY_GAMES.get(selected_game, {})
                            bonus_name = "Powerball" if selected_game == "Powerball" else "Mega Ball" if selected_game == "Mega Millions" else "Lucky Ball" if selected_game == "Lucky for Life" else "Bonus"
                            st.markdown(f"**{bonus_name}:** {strategy_data['bonus']}")
                        
                        # Description
                        st.markdown(f"**Strategy:** {strategy_data['description']}")
                        
                        # Quick play button simulation
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            if st.button(f"📋 Copy Numbers", key=f"copy_{strategy_key}"):
                                numbers_text = ', '.join(map(str, strategy_data['numbers']))
                                if strategy_data.get('bonus'):
                                    numbers_text += f" | Bonus: {strategy_data['bonus']}"
                                st.session_state[f"copied_{strategy_key}"] = numbers_text
                                st.success("Numbers copied!")
                        
                        with col_b:
                            if st.button(f"🎲 Generate New", key=f"regenerate_{strategy_key}"):
                                st.rerun()
                
                # Special printout for Daily 3 and Daily 4 games
                if selected_game in ["Daily 3", "Daily 4"]:
                    st.markdown("---")
                    st.markdown("**🎯 Daily Game Number Printout**")
                    
                    # Generate formatted printout and save to file
                    printout_text = print_daily_game_suggestions(selected_game, all_suggestions, days_to_analyze)
                    
                    # Display in a code block for easy copying
                    st.code(printout_text, language="text")
                    
                    # Create quick numbers summary for easy reference
                    quick_numbers = []
                    for strategy_data in all_suggestions.values():
                        numbers_str = ''.join(map(str, strategy_data['numbers']))
                        quick_numbers.append(numbers_str)
                    
                    quick_summary = f"QUICK REFERENCE - {selected_game.upper()} NUMBERS:\n"
                    quick_summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
                    quick_summary += "Numbers: " + ", ".join(quick_numbers[:8])  # Top 8 suggestions
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        # Add a download button for the full printout
                        if st.download_button(
                            label="📄 Download Full Report",
                            data=printout_text,
                            file_name=f"{selected_game.replace(' ', '_')}_full_report_{datetime.now().strftime('%Y%m%d_%I%M%S%p')}.txt",
                            mime="text/plain"
                        ):
                            st.success("Full report downloaded!")
                    
                    with col_d2:
                        # Add download for quick summary
                        if st.download_button(
                            label="🎯 Download Quick Numbers",
                            data=quick_summary,
                            file_name=f"{selected_game.replace(' ', '_')}_quick_numbers_{datetime.now().strftime('%Y%m%d_%I%M%S%p')}.txt",
                            mime="text/plain"
                        ):
                            st.success("Quick numbers downloaded!")
                    
                    # Additional info for Daily games
                    st.info("💡 **Daily Game Tips**: Numbers appear as consecutive digits (e.g., '573' for Daily 3). Files saved to 'data' folder and available for download above.")
                
                # Purchase button section - Now placed OUTSIDE Daily games section so it shows for ALL games
                st.markdown("---")
                st.markdown("**🎫 Ready to Play?**")
                
                # Michigan Lottery purchase URLs - only for games that can be purchased online
                # Daily 3 and Daily 4 games are excluded as they cannot be purchased online
                purchase_urls = {
                    'Powerball': 'https://www.michiganlottery.com/games/powerball',
                    'Mega Millions': 'https://www.michiganlottery.com/games/mega-millions',
                    'Fantasy 5': 'https://www.michiganlottery.com/games/fantasy-5',
                    'Lucky for Life': 'https://www.michiganlottery.com/games/lucky-for-life',
                    'Lotto 47': 'https://www.michiganlottery.com/games/classic-lotto-47'
                }
                
                # Check if this is a Daily game (cannot be purchased online)
                daily_games = ['Daily 3', 'Daily 4', 'Daily 3 Midday', 'Daily 3 Evening', 'Daily 4 Midday', 'Daily 4 Evening']
                is_daily_game_selected = selected_game in daily_games
                
                purchase_url = purchase_urls.get(selected_game)
                if purchase_url and not is_daily_game_selected:
                    col_purchase1, col_purchase2 = st.columns([2, 1])
                    with col_purchase1:
                        st.markdown(f"**Purchase tickets for {selected_game} using your suggested numbers**")
                        st.markdown("*Visit the official Michigan Lottery website to buy tickets online or find a retailer*")
                    
                    with col_purchase2:
                        # Use direct HTML link that opens in new tab - more reliable than JavaScript
                        st.markdown(
                            f'<a href="{purchase_url}" target="_blank" style="'
                            'display: inline-block; '
                            'padding: 0.5rem 1rem; '
                            'background-color: #ff4b4b; '
                            'color: white; '
                            'text-decoration: none; '
                            'border-radius: 0.5rem; '
                            'text-align: center; '
                            'font-weight: bold; '
                            'border: none; '
                            'cursor: pointer; '
                            'transition: background-color 0.3s;'
                            '" onmouseover="this.style.backgroundColor=\'#ff3333\'" '
                            'onmouseout="this.style.backgroundColor=\'#ff4b4b\'">'
                            f'🎫 Buy {selected_game} Tickets</a>',
                            unsafe_allow_html=True
                        )
                        st.markdown("", help=f"Opens Michigan Lottery website to purchase {selected_game} tickets")
                elif is_daily_game_selected:
                    # Special message for Daily games that cannot be purchased online
                    st.info(f"🏪 **{selected_game} tickets can only be purchased at authorized Michigan Lottery retailers.**")
                    st.markdown("📍 **Find a retailer near you:** Visit [michiganlottery.com/retailers](https://www.michiganlottery.com/retailers) to locate authorized lottery retailers in your area.")
                else:
                    st.info(f"🎫 Purchase information for {selected_game} not available. Visit michiganlottery.com to buy tickets.")

                # Additional statistical info
                st.markdown("---")
                st.markdown("**📊 Statistical Basis:**")
                
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    hot_count = len(analysis.get('hot_numbers', []))
                    st.metric("Hot Numbers Available", hot_count)
                
                with col_y:
                    cold_count = len(analysis.get('cold_numbers', []))
                    st.metric("Cold Numbers Available", cold_count)
                
                with col_z:
                    total_draws = analysis.get('total_draws', 0)
                    st.metric("Analysis Period", f"{total_draws} draws")
                
            else:
                st.info("👆 Run analysis first to generate intelligent number suggestions based on statistical patterns")
                
                # Show preview of what will be available
                st.markdown("**🎯 Available Suggestion Strategies:**")
                st.markdown("1. **Balanced Mix** - Combines hot, cold, and medium frequency numbers")
                st.markdown("2. **Hot Numbers Focus** - Emphasizes recently frequent numbers")  
                st.markdown("3. **Due Theory** - Focuses on numbers that haven't appeared recently")
                st.markdown("4. **Statistical Random** - Mathematically random selection")
                st.markdown("- Can you explain the frequency distribution?")
    
    with tab4:
            st.subheader("� Learning Dashboard")
            
            # Load performance data
            performance_metrics = load_performance_metrics()
            predictions_history = load_predictions_history()
            weights = load_learning_weights()
            
            # Create two columns for better layout
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                # Show last 10 winning numbers
                st.markdown("### 🎯 Recent Winning Numbers (Last 10)")
                recent_numbers = load_winning_numbers_from_log(selected_game, 10)
                
                if recent_numbers:
                    # Create a table for recent winning numbers
                    recent_data = []
                    for i, entry in enumerate(recent_numbers):
                        date_str = entry['date'].strftime('%Y-%m-%d')
                        time_str = entry['date'].strftime('%I:%M:%S %p')
                        numbers_str = ', '.join(map(str, entry['numbers']))
                        
                        row_data = {
                            '#': i + 1,
                            'Date': date_str,
                            'Time': time_str,
                            'Numbers': numbers_str
                        }
                        
                        # Add bonus number if present
                        if entry.get('bonus_number'):
                            row_data['Bonus'] = entry['bonus_number']
                        
                        recent_data.append(row_data)
                    
                    recent_df = pd.DataFrame(recent_data)
                    st.dataframe(recent_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No winning numbers logged for {selected_game} yet. Use the automated fetch or manual entry to add data.")
                
                # System Performance Overview
                st.markdown("### 🏆 Program Performance Overview")
                
                # Calculate overall system metrics
                total_predictions = len(predictions_history.get("predictions", []))
                verified_predictions = len([p for p in predictions_history.get("predictions", []) if p.get("verified", False)])
                unverified_predictions = total_predictions - verified_predictions
                
                # Overall accuracy calculation
                overall_accuracy = 0.0
                accuracy_count = 0
                if performance_metrics.get("strategy_performance"):
                    for game_perfs in performance_metrics["strategy_performance"].values():
                        for strategy_perf in game_perfs.values():
                            if strategy_perf["total_predictions"] > 0:
                                accuracy = strategy_perf["total_accuracy"] / strategy_perf["total_predictions"]
                                overall_accuracy += accuracy
                                accuracy_count += 1
                
                if accuracy_count > 0:
                    overall_accuracy = overall_accuracy / accuracy_count
                
                # Display performance metrics using simple layout
                st.write("**📊 System Metrics:**")
                st.write(f"• **Total Predictions:** {total_predictions}")
                st.write(f"• **Verified Results:** {verified_predictions}")
                st.write(f"• **Awaiting Verification:** {unverified_predictions}")
                
                if accuracy_count > 0:
                    st.write(f"• **Overall Accuracy:** {overall_accuracy:.1f}%")
                else:
                    st.write("• **Overall Accuracy:** N/A")
                
                # Games with data
                games_with_data = len([g for g in LOTTERY_GAMES.keys() if load_winning_numbers_from_log(g, 1)])
                st.write(f"• **Games with Data:** {games_with_data}/{len(LOTTERY_GAMES)}")
                
                # Learning system status
                learning_status = "🟢 Active" if total_predictions > 0 else "🟡 Ready"
                st.write(f"• **Learning Status:** {learning_status}")
                
                # Performance trend over time
                if performance_metrics.get("strategy_performance", {}).get(selected_game):
                    st.markdown("### 📈 Performance Trends")
                    
                    strategy_perfs = performance_metrics["strategy_performance"][selected_game]
                    trend_data = []
                    
                    for strategy_name, perf in strategy_perfs.items():
                        if perf["total_predictions"] > 0:
                            avg_accuracy = perf["total_accuracy"] / perf["total_predictions"]
                            recent_accuracy = sum(perf["recent_accuracy"]) / len(perf["recent_accuracy"]) if perf["recent_accuracy"] else avg_accuracy
                            
                            trend_data.append({
                                'Strategy': strategy_name,
                                'Historical': f"{avg_accuracy:.1f}%",
                                'Recent': f"{recent_accuracy:.1f}%",
                                'Trend': "📈" if recent_accuracy > avg_accuracy else "📉" if recent_accuracy < avg_accuracy * 0.9 else "➡️"
                            })
                    
                    if trend_data:
                        trend_df = pd.DataFrame(trend_data)
                        st.dataframe(trend_df, use_container_width=True, hide_index=True)
            
            with col_right:
                strategy_performance = performance_metrics.get("strategy_performance", {})
                if selected_game in strategy_performance:
                    st.markdown("### 📊 Strategy Performance Analysis")
                    
                    strategy_perfs = strategy_performance[selected_game]
                    
                    # Create performance table
                    perf_data = []
                    for strategy_name, perf in strategy_perfs.items():
                        if perf["total_predictions"] > 0:
                            avg_accuracy = perf["total_accuracy"] / perf["total_predictions"]
                            recent_avg = sum(perf["recent_accuracy"]) / len(perf["recent_accuracy"]) if perf["recent_accuracy"] else avg_accuracy
                            weight = weights.get("strategy_weights", {}).get(strategy_name, 1.0)
                            
                            perf_data.append({
                                "Strategy": strategy_name,
                                "Predictions": perf["total_predictions"],
                                "Avg Accuracy": f"{avg_accuracy:.1f}%",
                                "Recent Accuracy": f"{recent_avg:.1f}%",
                                "Best Result": f"{perf['best_accuracy']:.1f}%",
                                "Learning Weight": f"{weight:.2f}",
                                "Trend": "📈" if recent_avg > avg_accuracy else "📉" if recent_avg < avg_accuracy * 0.9 else "➡️"
                            })
                    
                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        # Show learning adjustments
                        st.markdown("### 🎯 Learning Adjustments")
                        
                        st.markdown("**🔥 Improved Strategies:**")
                        improved = [name for name, weight in weights.get("strategy_weights", {}).items() if weight > 1.1]
                        if improved:
                            for strategy in improved[:5]:
                                weight = weights["strategy_weights"][strategy]
                                st.markdown(f"• {strategy}: +{((weight-1)*100):.0f}%")
                        else:
                            st.info("No strategies have shown significant improvement yet")
                        
                        st.markdown("**📉 Underperforming Strategies:**")
                        underperform = [name for name, weight in weights.get("strategy_weights", {}).items() if weight < 0.9]
                        if underperform:
                            for strategy in underperform[:5]:
                                weight = weights["strategy_weights"][strategy]
                                st.markdown(f"• {strategy}: {((1-weight)*100):.0f}% reduced")
                        else:
                            st.info("No strategies are significantly underperforming")
                    
                    # Show recent verification results
                    if selected_game in st.session_state.get('verification_results', {}):
                        st.markdown("### ✅ Recent Verification Results")
                        recent_results = st.session_state.verification_results[selected_game]
                        
                        for result in recent_results[-3:]:  # Show last 3 results
                            if result["best_strategy"]:
                                strategy_name, strategy_score = result["best_strategy"]
                                st.success(f"Date: {result['date']} | Best: {strategy_name} ({strategy_score['accuracy_percentage']:.1f}% accuracy)")
                
                else:
                    st.info("🌱 No performance data yet. Generate predictions and check back tomorrow to see learning progress!")
    
    with tab5:
            st.subheader("�🤖 AI Assistant")
            
            # Check if local LLM is available
            if local_pipeline:
                st.success("🤖 Local LLM Chat enabled with CUDA acceleration")
            elif OLLAMA_AVAILABLE:
                st.success("🦙 Ollama LLM Chat available")
                
                # Chat interface
                if "messages" not in st.session_state:
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"Hi! I'm your lottery analysis assistant powered by OpenAI. I can help you understand patterns in {selected_game} data. What would you like to know?"}
                    ]
                
                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
            else:
                st.info("🤖 Local AI chat features available. Initialize Local LLM or install Ollama to enable intelligent conversations about lottery patterns and statistical analysis.")
    
    with tab6:
            st.subheader(f"🔍 Number Win Frequency Checker - {selected_game}")
            
            # Get game configuration
            game_config = LOTTERY_GAMES.get(selected_game, {})
            numbers_range = game_config.get('numbers_range', (1, 50))
            bonus_range = game_config.get('bonus_range', None)
            numbers_count = game_config.get('numbers_count', 5)
            draw_days = game_config.get('draw_days', ['Daily'])
            
            # Display game-specific information
            st.info(f"🎲 **{selected_game} Game Rules:**")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.write(f"• **Numbers to pick**: {numbers_count}")
                st.write(f"• **Number range**: {numbers_range[0]}-{numbers_range[1]}")
            
            with col_info2:
                if bonus_range:
                    st.write(f"• **Bonus/Power Ball**: {bonus_range[0]}-{bonus_range[1]}")
                else:
                    st.write("• **No bonus number**")
                st.write(f"• **Draw days**: {', '.join(draw_days)}")
            
            with col_info3:
                total_possible_main = numbers_range[1] - numbers_range[0] + 1
                st.write(f"• **Possible main numbers**: {total_possible_main}")
                if bonus_range:
                    total_possible_bonus = bonus_range[1] - bonus_range[0] + 1
                    st.write(f"• **Possible bonus numbers**: {total_possible_bonus}")
            
            st.write(f"Enter specific numbers to check how many times they have appeared in {selected_game} winning draws.")
            
            # Add game-specific rules info
            if selected_game in ["Daily 3", "Daily 4"]:
                st.info(f"💡 **{selected_game} Rules**: Duplicate digits are allowed (e.g., 0,7,0,7). Position matters - this is a digit-based game where each position can be any digit 0-9.")
            else:
                st.info(f"💡 **{selected_game} Rules**: Each number must be unique. No duplicate numbers allowed.")
            
            if st.session_state.get('run_analysis', False):
                # Get current analysis data (prioritizes actual logged results)
                results = get_lottery_data(selected_game, days_to_analyze)
                
                col_check1, col_check2 = st.columns([2, 1])
                
                with col_check1:
                    st.markdown(f"**Check {selected_game} Main Numbers:**")
                    
                    # Create input fields for multiple numbers
                    input_methods = ["Complete Ticket", "Multiple Numbers", "Single Number", "Number Range"]
                    input_method = st.radio(
                        "Choose input method:",
                        input_methods,
                        horizontal=True,
                        help=f"Complete Ticket: Check a full {selected_game} ticket with main numbers{' + bonus' if bonus_range else ''}"
                    )
                    
                    numbers_to_check = []
                    
                    if input_method == "Complete Ticket":
                        st.write(f"**🎫 Enter your complete {selected_game} ticket:**")
                        
                        # Main numbers input
                        if selected_game == "Daily 3":
                            default_example = "1,0,7"  # Show that duplicates/repeats are OK
                        elif selected_game == "Daily 4":
                            default_example = "0,7,0,7"  # Show that duplicates/repeats are OK
                        elif selected_game == "Fantasy 5":
                            default_example = "7,14,21,28,35"
                        elif selected_game == "Lucky for Life":
                            default_example = "7,14,21,35,42"
                        elif selected_game == "Powerball":
                            default_example = "7,14,21,35,42"
                        else:  # Mega Millions
                            default_example = "7,14,21,35,42"
                        
                        main_numbers_text = st.text_input(
                            f"Main numbers ({numbers_count} numbers from {numbers_range[0]}-{numbers_range[1]}):",
                            placeholder=default_example,
                            help=f"Enter exactly {numbers_count} main numbers for {selected_game}",
                            key="ticket_main_numbers"
                        )
                        
                        # Bonus number input (if applicable)
                        bonus_number_input = None
                        if bonus_range:
                            bonus_label = "Power Ball" if selected_game == "Powerball" else "Mega Ball" if selected_game == "Mega Millions" else "Lucky Ball" if selected_game == "Lucky for Life" else "Bonus Number"
                            bonus_number_input = st.number_input(
                                f"{bonus_label} ({bonus_range[0]}-{bonus_range[1]}):",
                                min_value=bonus_range[0],
                                max_value=bonus_range[1],
                                value=bonus_range[0],
                                help=f"Enter the {bonus_label.lower()} for {selected_game}",
                                key="ticket_bonus_number"
                            )
                        
                        if st.button(f"🎫 Check Complete {selected_game} Ticket"):
                            ticket_valid = True
                            
                            # Validate main numbers
                            if main_numbers_text:
                                try:
                                    main_numbers = [int(n.strip()) for n in main_numbers_text.split(',') if n.strip()]
                                    
                                    if len(main_numbers) != numbers_count:
                                        st.error(f"❌ Need exactly {numbers_count} main numbers for {selected_game}")
                                        ticket_valid = False
                                    elif len(set(main_numbers)) != len(main_numbers) and selected_game not in ["Daily 3", "Daily 4"]:
                                        # Only check for duplicates in lottery games, not Daily 3/4 where duplicates are allowed
                                        duplicates = [n for n in main_numbers if main_numbers.count(n) > 1]
                                        st.error(f"❌ Duplicate numbers not allowed in {selected_game}: {list(set(duplicates))}")
                                        ticket_valid = False
                                    elif not all(numbers_range[0] <= n <= numbers_range[1] for n in main_numbers):
                                        invalid = [n for n in main_numbers if not (numbers_range[0] <= n <= numbers_range[1])]
                                        st.error(f"❌ Invalid main numbers: {invalid}. Range: {numbers_range[0]}-{numbers_range[1]}")
                                        ticket_valid = False
                                    else:
                                        numbers_to_check = main_numbers
                                        if selected_game in ["Daily 3", "Daily 4"]:
                                            # Show in order for Daily games (position matters)
                                            ticket_display = f"Digits: {main_numbers}"
                                        else:
                                            # Show sorted for lottery games
                                            ticket_display = f"Main: {sorted(main_numbers)}"
                                            if bonus_range and bonus_number_input is not None:
                                                ticket_display += f", Bonus: {bonus_number_input}"
                                        st.success(f"🎫 Checking complete ticket: {ticket_display}")
                                        
                                except ValueError:
                                    st.error("❌ Please enter valid numbers separated by commas")
                                    ticket_valid = False
                            else:
                                st.error("❌ Please enter main numbers")
                                ticket_valid = False
                    
                    elif input_method == "Single Number":
                        number = st.number_input(
                            f"Enter {selected_game} number ({numbers_range[0]}-{numbers_range[1]}):",
                            min_value=numbers_range[0],
                            max_value=numbers_range[1],
                            value=numbers_range[0],
                            help=f"Valid range for {selected_game}: {numbers_range[0]} to {numbers_range[1]}"
                        )
                        if st.button(f"Check Single {selected_game} Number"):
                            numbers_to_check = [number]
                    
                    elif input_method == "Multiple Numbers":
                        # Create game-specific input with exact number count
                        st.write(f"**Enter exactly {numbers_count} {selected_game} numbers:**")
                        
                        # Generate appropriate example and placeholder
                        if selected_game == "Daily 3":
                            example = "1,0,7"
                            placeholder = "e.g., 1,0,7 (repeats OK)"
                        elif selected_game == "Daily 4":
                            example = "0,7,0,7"
                            placeholder = "e.g., 0,7,0,7 (repeats OK)"
                        elif selected_game == "Fantasy 5":
                            example = "7,14,21,28,35"
                            placeholder = "e.g., 7,14,21,28,35"
                        elif selected_game == "Lucky for Life":
                            example = "7,14,21,35,42"
                            placeholder = "e.g., 7,14,21,35,42"
                        elif selected_game == "Powerball":
                            example = "7,14,21,35,42"
                            placeholder = "e.g., 7,14,21,35,42"
                        else:  # Mega Millions
                            example = "7,14,21,35,42"
                            placeholder = "e.g., 7,14,21,35,42"
                        
                        numbers_text = st.text_input(
                            f"Enter {numbers_count} numbers (range: {numbers_range[0]}-{numbers_range[1]}):",
                            placeholder=placeholder,
                            help=f"Enter exactly {numbers_count} numbers separated by commas for {selected_game}",
                            value=""
                        )
                        if st.button(f"Check {selected_game} Numbers ({numbers_count} required)") and numbers_text:
                            try:
                                input_numbers = [int(n.strip()) for n in numbers_text.split(',') if n.strip()]
                                
                                # Check count first
                                if len(input_numbers) != numbers_count:
                                    st.error(f"❌ {selected_game} requires exactly {numbers_count} numbers. You entered {len(input_numbers)} numbers.")
                                    st.info(f"💡 Example: {example}")
                                    input_numbers = []
                                
                                # Validate numbers are in range
                                valid_numbers = [n for n in input_numbers 
                                               if numbers_range[0] <= n <= numbers_range[1]]
                                invalid_numbers = [n for n in input_numbers 
                                                 if n < numbers_range[0] or n > numbers_range[1]]
                                
                                # Check for duplicates (only for lottery games, not Daily 3/4)
                                duplicates = []
                                if selected_game not in ["Daily 3", "Daily 4"]:
                                    seen = set()
                                    for n in input_numbers:
                                        if n in seen:
                                            duplicates.append(n)
                                        seen.add(n)
                                
                                if invalid_numbers:
                                    st.warning(f"⚠️ Invalid numbers for {selected_game}: {invalid_numbers}. Valid range: {numbers_range[0]}-{numbers_range[1]}")
                                
                                if duplicates and selected_game not in ["Daily 3", "Daily 4"]:
                                    st.warning(f"⚠️ Duplicate numbers found: {list(set(duplicates))}. Each number should be unique for {selected_game}.")
                                
                                # For Daily 3/4, duplicates are allowed, so check only count and range
                                duplicate_check_passed = (not duplicates) if selected_game not in ["Daily 3", "Daily 4"] else True
                                
                                if valid_numbers and len(valid_numbers) == numbers_count and duplicate_check_passed:
                                    numbers_to_check = valid_numbers
                                    if selected_game in ["Daily 3", "Daily 4"]:
                                        # Show in order for Daily games (position matters)
                                        st.success(f"✅ Checking {numbers_count} {selected_game} digits: {valid_numbers}")
                                    else:
                                        # Show sorted for lottery games
                                        st.success(f"✅ Checking {numbers_count} {selected_game} numbers: {sorted(valid_numbers)}")
                                
                            except ValueError:
                                st.error(f"Please enter valid {selected_game} numbers separated by commas")
                                numbers_to_check = []
                    
                    elif input_method == "Number Range":
                        st.write(f"Check a range of {selected_game} numbers:")
                        col_range1, col_range2 = st.columns(2)
                        with col_range1:
                            start_num = st.number_input(
                                f"Start number ({numbers_range[0]}-{numbers_range[1]}):",
                                min_value=numbers_range[0],
                                max_value=numbers_range[1],
                                value=numbers_range[0],
                                help=f"Starting number for {selected_game} range check"
                            )
                        with col_range2:
                            max_end = min(numbers_range[0] + 10, numbers_range[1])
                            end_num = st.number_input(
                                f"End number ({numbers_range[0]}-{numbers_range[1]}):",
                                min_value=numbers_range[0],
                                max_value=numbers_range[1],
                                value=max_end,
                                help=f"Ending number for {selected_game} range check"
                            )
                        
                        range_size = end_num - start_num + 1
                        if range_size > 20:
                            st.warning(f"Range size: {range_size} numbers. Large ranges may take longer to process.")
                        
                        if st.button(f"Check {selected_game} Number Range ({start_num}-{end_num})"):
                            if start_num <= end_num:
                                numbers_to_check = list(range(start_num, end_num + 1))
                                st.info(f"Checking {len(numbers_to_check)} numbers in range {start_num}-{end_num}")
                            else:
                                st.error("Start number must be less than or equal to end number")
                    
                    # Process and display results
                    if numbers_to_check:
                        st.markdown("---")
                        st.subheader("📊 Win Frequency Results")
                        
                        # Calculate frequency for each number
                        frequency_data = []
                        total_draws = len(results)
                        
                        for number in numbers_to_check:
                            # Count appearances in main numbers
                            main_appearances = sum(1 for result in results if number in result['numbers'])
                            
                            # Count appearances in bonus numbers (if applicable)
                            bonus_appearances = 0
                            if bonus_range and any(result.get('bonus_number') for result in results):
                                bonus_appearances = sum(1 for result in results 
                                                      if result.get('bonus_number') == number)
                            
                            total_appearances = main_appearances + bonus_appearances
                            win_percentage = (total_appearances / total_draws) * 100 if total_draws > 0 else 0
                            
                            frequency_data.append({
                                'Number': number,
                                'Main Number Wins': main_appearances,
                                'Bonus Number Wins': bonus_appearances if bonus_range else 'N/A',
                                'Total Appearances': total_appearances,
                                'Win Percentage': f"{win_percentage:.2f}%",
                                'Days Since Last Win': calculate_days_since_last_win(number, results)
                            })
                        
                        # Display results table
                        frequency_df = pd.DataFrame(frequency_data)
                        st.dataframe(frequency_df, use_container_width=True)
                        
                        # Summary statistics
                        if frequency_data:
                            st.markdown("### 📈 Summary Statistics")
                            
                            # Calculate statistics
                            avg_appearances = sum(item['Total Appearances'] for item in frequency_data) / len(frequency_data)
                            most_frequent = max(frequency_data, key=lambda x: x['Total Appearances'])
                            
                            # Display metrics in a single row without nested columns
                            st.metric("Average Appearances", f"{avg_appearances:.1f}")
                            st.metric("Most Frequent Number", f"{most_frequent['Number']} ({most_frequent['Total Appearances']})")
                            
                            # Third metric without column context
                            least_frequent = min(frequency_data, key=lambda x: x['Total Appearances'])
                            st.metric("Least Frequent Number", f"{least_frequent['Number']} ({least_frequent['Total Appearances']})")
                        
                        # Add bonus number checking if game has bonus numbers
                        if bonus_range:
                            st.markdown("---")
                            st.markdown(f"**Check {selected_game} Bonus/Power Numbers:**")
                            
                            bonus_number = st.number_input(
                                f"Enter bonus number ({bonus_range[0]}-{bonus_range[1]}):",
                                min_value=bonus_range[0],
                                max_value=bonus_range[1],
                                value=bonus_range[0],
                                help=f"Check bonus/power ball frequency for {selected_game}"
                            )
                            
                            if st.button(f"Check {selected_game} Bonus Number"):
                                bonus_appearances = sum(1 for result in results 
                                                      if result.get('bonus_number') == bonus_number)
                                bonus_percentage = (bonus_appearances / len(results)) * 100 if len(results) > 0 else 0
                                
                                # Display bonus number metrics without nested columns
                                st.metric("Bonus Appearances", bonus_appearances)
                                st.metric("Bonus Win %", f"{bonus_percentage:.2f}%")
                                expected_bonus = len(results) / (bonus_range[1] - bonus_range[0] + 1)
                                status = "Hot" if bonus_appearances > expected_bonus * 1.2 else "Cold" if bonus_appearances < expected_bonus * 0.8 else "Average"
                                st.metric("Status", status)
                                
                                # Bonus number analysis
                                st.info(f"**Bonus Number {bonus_number} Analysis:**")
                                st.write(f"• Appeared in {bonus_appearances} out of {len(results)} draws")
                                st.write(f"• Expected frequency: {expected_bonus:.1f} times")
                                st.write(f"• Performance: {bonus_appearances/expected_bonus*100:.0f}% of expected" if expected_bonus > 0 else "No data")
                
                with col_check2:
                    st.markdown(f"**{selected_game} Quick Stats:**")
                    st.info(f"🎮 **Game**: {selected_game}")
                    st.info(f"📅 **Analysis Period**: {days_to_analyze} days")
                    st.info(f"🎲 **Total Draws**: {len(results)}")
                    st.info(f"🔢 **Main Numbers**: {numbers_range[0]}-{numbers_range[1]} (pick {numbers_count})")
                    if bonus_range:
                        st.info(f"🎯 **Bonus Numbers**: {bonus_range[0]}-{bonus_range[1]}")
                    else:
                        st.info("🎯 **No Bonus Numbers**")
                    st.info(f"📆 **Draw Schedule**: {', '.join(draw_days)}")
                    
                    # Game-specific statistics
                    if results:
                        all_numbers = []
                        for result in results:
                            all_numbers.extend(result['numbers'])
                        
                        from collections import Counter
                        number_counts = Counter(all_numbers)
                        total_possible = numbers_range[1] - numbers_range[0] + 1
                        expected_frequency = len(results) * numbers_count / total_possible
                        
                        st.markdown(f"**{selected_game} Top 5 Hot Numbers:**")
                        for number, count in number_counts.most_common(5):
                            percentage = (count / len(results)) * 100
                            vs_expected = count / expected_frequency if expected_frequency > 0 else 0
                            status = "🔥" if vs_expected > 1.2 else "🌡️" if vs_expected > 0.8 else "❄️"
                            st.write(f"{status} **{number}**: {count} times ({percentage:.1f}%)")
                        
                        st.markdown(f"**{selected_game} Bottom 5 Cold Numbers:**")
                        for number, count in number_counts.most_common()[-5:]:
                            percentage = (count / len(results)) * 100
                            vs_expected = count / expected_frequency if expected_frequency > 0 else 0
                            status = "❄️" if vs_expected < 0.8 else "🌡️" if vs_expected < 1.2 else "🔥"
                            st.write(f"{status} **{number}**: {count} times ({percentage:.1f}%)")
                        
                        # Expected vs actual frequency info
                        st.markdown("---")
                        st.write(f"📊 **Expected frequency per number**: {expected_frequency:.1f}")
                        st.write(f"🎯 **Numbers above average**: {sum(1 for count in number_counts.values() if count > expected_frequency)}")
                        st.write(f"❄️ **Numbers below average**: {sum(1 for count in number_counts.values() if count < expected_frequency)}")
            else:
                st.info(f"🔍 Please run an analysis first by clicking '🔍 Run Analysis' in the sidebar to check {selected_game} number frequencies.")
                st.markdown("---")
                st.markdown(f"### {selected_game} Game Preview:")
                st.write(f"• **Pick {numbers_count} numbers** from {numbers_range[0]} to {numbers_range[1]}")
                if bonus_range:
                    st.write(f"• **Plus 1 bonus number** from {bonus_range[0]} to {bonus_range[1]}")
                st.write(f"• **Draws**: {', '.join(draw_days)}")
                st.write(f"• **Total possible main numbers**: {numbers_range[1] - numbers_range[0] + 1}")
                
                # Add some game-specific tips
                if selected_game in ["Powerball", "Mega Millions"]:
                    st.info("💡 **Tip**: These are multi-state games with large jackpots and bonus balls")
                elif selected_game == "Fantasy 5":
                    st.info("💡 **Tip**: Michigan's daily game with smaller number pool for better odds")
                elif selected_game == "Lucky for Life":
                    st.info("💡 **Tip**: Multi-state game with $1,000/day for life top prize, draws twice weekly")
                elif selected_game in ["Daily 3", "Daily 4"]:
                    st.info("💡 **Tip**: Daily digit games with multiple play styles and smaller prizes")
    
    with tab7:
        st.subheader(f"💬 {selected_game} Winning Numbers Chat")
        st.markdown("*View all known lottery winning numbers in a conversational chat format*")
        
        # Chat controls
        col_chat1, col_chat2, col_chat3 = st.columns([2, 1, 1])
        
        with col_chat1:
            # Date range selector
            date_range_option = st.selectbox(
                "📅 Select time period:",
                ["Recent (Last 30 days)", "Last 3 months", "Last 6 months", "Last year", "All available data"],
                index=0
            )
        
        with col_chat2:
            # Results per page
            results_per_page = st.selectbox(
                "🔢 Results per page:",
                [10, 25, 50, 100],
                index=1
            )
        
        with col_chat3:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
        
        # Get lottery data based on selection
        if date_range_option == "Recent (Last 30 days)":
            days_to_show = 30
        elif date_range_option == "Last 3 months":
            days_to_show = 90
        elif date_range_option == "Last 6 months":
            days_to_show = 180
        elif date_range_option == "Last year":
            days_to_show = 365
        else:  # All available data
            days_to_show = 3650  # 10 years
        
        # Create chat container
        chat_container = st.container()
        
        with chat_container:
            st.markdown("---")
            
            # Get lottery results
            try:
                results = get_lottery_data(selected_game, days_to_show)
                
                if results:
                    # Sort results by date (newest first)
                    sorted_results = sorted(results, key=lambda x: x.get('date', ''), reverse=True)
                    
                    # Pagination
                    total_results = len(sorted_results)
                    total_pages = (total_results + results_per_page - 1) // results_per_page
                    
                    if total_pages > 1:
                        page = st.selectbox(
                            f"📄 Page (showing {results_per_page} of {total_results} results)",
                            range(1, total_pages + 1),
                            index=0
                        )
                        start_idx = (page - 1) * results_per_page
                        end_idx = start_idx + results_per_page
                        page_results = sorted_results[start_idx:end_idx]
                    else:
                        page_results = sorted_results[:results_per_page]
                    
                    # Display results in chat format
                    st.markdown("### 💬 Lottery Numbers Chat")
                    
                    for i, result in enumerate(page_results):
                        # Create chat message styling
                        date_str = result.get('date', 'Unknown Date')
                        numbers = result.get('numbers', [])
                        source = result.get('source', 'system')
                        
                        # Format date nicely
                        try:
                            if isinstance(date_str, str) and date_str != 'Unknown Date':
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                formatted_date = date_obj.strftime('%B %d, %Y (%A)')
                                time_ago = (datetime.now() - date_obj).days
                                if time_ago == 0:
                                    time_ago_str = "Today"
                                elif time_ago == 1:
                                    time_ago_str = "Yesterday"
                                elif time_ago < 7:
                                    time_ago_str = f"{time_ago} days ago"
                                elif time_ago < 30:
                                    weeks = time_ago // 7
                                    time_ago_str = f"{weeks} week{'s' if weeks > 1 else ''} ago"
                                elif time_ago < 365:
                                    months = time_ago // 30
                                    time_ago_str = f"{months} month{'s' if months > 1 else ''} ago"
                                else:
                                    years = time_ago // 365
                                    time_ago_str = f"{years} year{'s' if years > 1 else ''} ago"
                            else:
                                formatted_date = date_str
                                time_ago_str = ""
                        except:
                            formatted_date = date_str
                            time_ago_str = ""
                        
                        # Format numbers based on game type
                        if selected_game in ["Daily 3", "Daily 4"]:
                            # For digit games, show numbers without separation
                            numbers_display = "".join([str(n) for n in numbers])
                            numbers_styled = " ".join([f"**{n}**" for n in str(numbers_display)])
                        else:
                            # For regular lottery games, show with commas
                            numbers_styled = ", ".join([f"**{n}**" for n in numbers])
                        
                        # Create chat bubble effect
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       border-radius: 18px; padding: 16px; color: white; 
                                       max-width: 70%; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                <div style="font-size: 0.85em; opacity: 0.9; margin-bottom: 5px;">
                                    📊 Historical Record • {time_ago_str}
                                </div>
                                <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 8px;">
                                    📈 Numbers drawn: {numbers_styled}
                                </div>
                                <div style="font-size: 0.8em; opacity: 0.8;">
                                    📅 {formatted_date}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add special highlighting for recent wins
                        if time_ago_str in ["Today", "Yesterday"] or ("days ago" in time_ago_str and int(time_ago_str.split()[0]) <= 7):
                            st.markdown(f"""
                            <div style="text-align: center; margin-bottom: 10px;">
                                <span style="background: #ffd700; color: #333; padding: 4px 12px; 
                                           border-radius: 15px; font-size: 0.8em; font-weight: bold;">
                                    ⭐ Recent Win ⭐
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show statistics at the bottom
                    st.markdown("---")
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("📊 Total Results", len(sorted_results))
                    
                    with stats_col2:
                        # Most frequent number
                        all_numbers = []
                        for result in sorted_results:
                            all_numbers.extend(result.get('numbers', []))
                        if all_numbers:
                            from collections import Counter
                            most_frequent = Counter(all_numbers).most_common(1)[0]
                            st.metric("🔥 Hot Number", f"{most_frequent[0]} ({most_frequent[1]}x)")
                    
                    with stats_col3:
                        # Date range
                        if len(sorted_results) > 1:
                            oldest_date = sorted_results[-1].get('date', '')
                            newest_date = sorted_results[0].get('date', '')
                            try:
                                oldest = datetime.strptime(oldest_date, '%Y-%m-%d')
                                newest = datetime.strptime(newest_date, '%Y-%m-%d')
                                days_span = (newest - oldest).days
                                st.metric("📅 Data Span", f"{days_span} days")
                            except:
                                st.metric("📅 Data Span", "Multiple dates")
                    
                    with stats_col4:
                        # Source info
                        sources = [r.get('source', 'system') for r in sorted_results]
                        from collections import Counter
                        source_counts = Counter(sources)
                        main_source = source_counts.most_common(1)[0][0] if source_counts else 'system'
                        st.metric("🗂️ Main Source", main_source.title())
                    
                    # Auto-refresh functionality
                    if auto_refresh:
                        time.sleep(30)  # Refresh every 30 seconds
                        st.rerun()
                
                else:
                    # No data available
                    st.markdown(f"""
                    <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;">
                        <h3 style="color: #6c757d;">📭 No Lottery Data Available</h3>
                        <p style="color: #6c757d; margin-bottom: 20px;">
                            No winning numbers found for {selected_game} in the selected time period.
                        </p>
                        <div style="background: white; padding: 20px; border-radius: 8px; margin-top: 20px;">
                            <h4 style="color: #495057;">💡 How to get lottery data:</h4>
                            <p style="color: #6c757d; text-align: left;">
                                • <strong>Web Scraping:</strong> Use the "Fetch Latest Numbers" button below<br>
                                • <strong>Manual Entry:</strong> Add numbers manually using the input forms<br>
                                • <strong>Historical Data:</strong> Load OCR-processed PDF data from 2000-2021<br>
                                • <strong>Run Analysis:</strong> Click "Run Analysis" to generate sample data
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error loading lottery data: {str(e)}")
                st.info("💡 Try running an analysis first or fetch current lottery numbers from the sidebar.")

    # Recent results and quick stats sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Recent Winning Numbers")
    
    # Load actual winning numbers from log file
    logged_results = load_winning_numbers_from_log(selected_game, 10)
    
    if logged_results:
        # Display logged winning numbers
        st.sidebar.markdown("**🎯 Actual Winning Numbers (From Log)**")
        results_data = []
        for result in logged_results[:5]:  # Show only 5 most recent
            row = {
                'Date': result['date'].strftime('%Y-%m-%d %I:%M %p'),
                'Numbers': ', '.join(map(str, result['numbers'])),
            }
            
            if result['bonus_number']:
                row['Bonus'] = result['bonus_number']
            
            if result['jackpot']:
                row['Jackpot'] = f"${result['jackpot']:,.0f}"
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        st.sidebar.dataframe(results_df, use_container_width=True)
        
        # Show count of logged entries
        st.sidebar.caption(f"📊 Total logged entries for {selected_game}: {len(logged_results)}")
    
    elif st.session_state.get('run_analysis', False):
        # Fallback to sample data if no logged numbers exist
        st.sidebar.markdown("**📊 Sample Data (No Actual Results Logged)**")
        results = generate_sample_data(selected_game, 10)  # Last 10 draws
        
        # Create results table
        results_data = []
        for result in results[:5]:  # Show only 5 most recent
            row = {
                'Date': result['date'].strftime('%Y-%m-%d'),
                'Numbers': ', '.join(map(str, result['numbers'])),
            }
            
            if result['bonus_number']:
                row['Bonus'] = result['bonus_number']
            
            if result['jackpot']:
                row['Jackpot'] = f"${result['jackpot']:,.0f}"
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        st.sidebar.dataframe(results_df, use_container_width=True)
        st.sidebar.caption("⚠️ This is sample data. Add actual winning numbers below.")
        
        # Game information
        st.sidebar.subheader("🎲 Game Info")
        game_config = LOTTERY_GAMES.get(selected_game, {})
        
        st.sidebar.markdown(f"**Numbers to pick:** {game_config.get('numbers_count', 'N/A')}")
        
        numbers_range = game_config.get('numbers_range', (0, 0))
        st.sidebar.markdown(f"**Number range:** {numbers_range[0]} - {numbers_range[1]}")
        
        if 'bonus_range' in game_config:
            bonus_range = game_config['bonus_range']
            st.sidebar.markdown(f"**Bonus range:** {bonus_range[0]} - {bonus_range[1]}")
        
        draw_days = game_config.get('draw_days', [])
        st.sidebar.markdown(f"**Draw days:** {', '.join(draw_days)}")
    
    else:
        st.sidebar.info("Run analysis to see recent results")
        

    
    # Add new winning numbers section (outside of columns to avoid context issues)
    st.markdown("---")
    st.subheader("➕ Add New Winning Numbers")
    
    # Automated fetch button
    st.markdown("### 🌐 Automated Data Collection")
    col_auto1, col_auto2 = st.columns([1, 2])
    
    with col_auto1:
        if st.button("🌐 Fetch Latest Numbers", type="primary", help="Fetch the most recent winning numbers with drawing schedule information"):
            with st.spinner(f"Fetching latest winning numbers for {selected_game}..."):
                success = fetch_and_update_todays_numbers(selected_game)
                if success:
                    st.success(f"✅ Successfully fetched and logged winning numbers for {selected_game}!")
                    st.balloons()
                    # Refresh the page to show updated data
                    st.rerun()
                else:
                    st.info("💡 See drawing schedule information above for next available drawing.")
    
    with col_auto2:
        st.info("🔄 This will fetch the most recent winning numbers and show when the last drawing occurred. For games that don't draw daily, you'll see the next drawing date.")
        
        # Show drawing schedule for current game
        schedule_info = get_game_drawing_schedule_simple(selected_game)
        if schedule_info:
            st.markdown(f"**📅 {selected_game} Schedule:**")
            st.markdown(f"• **Frequency:** {schedule_info['frequency']}")
            if schedule_info['draw_days'] != ['Daily']:
                st.markdown(f"• **Draw Days:** {', '.join(schedule_info['draw_days'])}")
            st.markdown(f"• **Draw Time:** {schedule_info['draw_time']}")

    # Manual Entry Section
    st.markdown("### 📝 Manual Entry")
    with st.expander("🎯 Log Actual Winning Numbers", expanded=False):
        st.markdown(f"**Add winning numbers for {selected_game}**")
        
        # Date input
        draw_date = st.date_input(
            "Draw Date",
            value=datetime.now().date(),
            help="Select the date when these numbers were drawn"
        )
        
        # Time input
        draw_time = st.time_input(
            "Draw Time", 
            value=datetime.now().time().replace(second=0, microsecond=0),
            help="Select the time when the draw occurred"
        )

        # Get game config
        game_config = LOTTERY_GAMES.get(selected_game, LOTTERY_GAMES['Powerball'])
        numbers_count = game_config['numbers_count']
        
        st.markdown(f"**Enter {numbers_count} numbers:**")
        
        # Numbers input as text field for better compatibility
        if selected_game == "Daily 3":
            numbers_input = st.text_input(
                f"Winning Numbers (3 digits: 0-9)",
                placeholder="e.g., 5,7,3 or 573",
                help="Enter the 3 winning digits"
            )
        elif selected_game == "Daily 4":
            numbers_input = st.text_input(
                f"Winning Numbers (4 digits: 0-9)",
                placeholder="e.g., 5,7,3,9 or 5739",
                help="Enter the 4 winning digits"
            )
        else:
            numbers_range = game_config['numbers_range']
            numbers_input = st.text_input(
                f"Winning Numbers ({numbers_count} numbers: {numbers_range[0]}-{numbers_range[1]})",
                placeholder=f"e.g., 7,14,21,35,42",
                help=f"Enter {numbers_count} winning numbers separated by commas"
            )
        
        # Bonus number for games that have it
        bonus_input = None
        if 'bonus_range' in game_config:
            bonus_range = game_config['bonus_range']
            bonus_input = st.number_input(
                f"Bonus Number ({bonus_range[0]}-{bonus_range[1]})",
                min_value=bonus_range[0], 
                max_value=bonus_range[1], 
                value=None,
                help=f"Enter the bonus number ({bonus_range[0]}-{bonus_range[1]})"
            )
            
        # Jackpot for major games
        jackpot_input = None
        if selected_game in ['Powerball', 'Mega Millions']:
            jackpot_input = st.number_input(
                "Jackpot Amount ($)", 
                min_value=0.0, 
                value=None,
                format="%.0f",
                help="Enter the jackpot amount (optional)"
            )
        
        # Submit button
        if st.button("💾 Save Winning Numbers", type="primary"):
            try:
                # Parse numbers input
                if not numbers_input.strip():
                    st.error("Please enter the winning numbers")
                else:
                    # Handle different input formats
                    numbers_str = numbers_input.strip().replace(' ', '')
                    
                    if selected_game in ["Daily 3", "Daily 4"]:
                        # For Daily games, handle both comma-separated and continuous digits
                        if ',' in numbers_str:
                            numbers = [int(x.strip()) for x in numbers_str.split(',')]
                        else:
                            # Treat as continuous digits (e.g., "573" -> [5,7,3])
                            numbers = [int(d) for d in numbers_str]
                    else:
                        # For lottery games, expect comma-separated
                        numbers = [int(x.strip()) for x in numbers_str.split(',')]
                    
                    # Validate numbers
                    numbers_range = game_config['numbers_range']
                    if len(numbers) != numbers_count:
                        st.error(f"Please enter exactly {numbers_count} numbers for {selected_game}")
                    elif any(n < numbers_range[0] or n > numbers_range[1] for n in numbers):
                        st.error(f"All numbers must be between {numbers_range[0]} and {numbers_range[1]}")
                    else:
                        # Combine date and time
                        draw_datetime = datetime.combine(draw_date, draw_time)
                        
                        # Save to log
                        bonus_val = int(bonus_input) if bonus_input is not None else None
                        jackpot_val = float(jackpot_input) if jackpot_input is not None else None
                        
                        success = save_winning_numbers_to_log(
                            selected_game, 
                            draw_datetime, 
                            numbers, 
                            bonus_val, 
                            jackpot_val
                        )
                        
                        if success:
                            st.success(f"✅ Winning numbers saved for {selected_game} on {draw_datetime.strftime('%Y-%m-%d %I:%M %p')}")
                            st.balloons()
                            # Refresh the page to show new data
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to save winning numbers")
            
            except ValueError as e:
                st.error(f"❌ Invalid number format. Please enter numbers separated by commas.")
            except Exception as e:
                st.error(f"❌ Error saving winning numbers: {str(e)}")
        
        st.caption("💡 **Tip**: Adding actual winning numbers improves analysis accuracy and enables better pattern detection.")

    # Debug section
    with st.expander("🔍 Debug Enhanced Scraping", expanded=False):
        debug_scraping_section(selected_game)

    with tab8:
        st.header("🌐 Michigan Lottery Official Website Lookup")
        st.markdown("""
        This tab allows you to look up winning numbers directly from the official Michigan Lottery website.
        This is useful as a secondary source when CSV data is missing or needs verification.
        """)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Game selection for web lookup
            web_games = [
                'Daily 3', 'Daily 4', 'Fantasy 5', 'Lotto 47', 
                'Powerball', 'Mega Millions', 'Lucky for Life'
            ]
            
            selected_web_game = st.selectbox(
                "Select Game for Web Lookup:",
                web_games,
                key="web_game_selector"
            )
            
            # Add note about Daily 3/4 variations
            if selected_web_game in ['Daily 3', 'Daily 4']:
                st.info(f"📝 **Note:** {selected_web_game} draws twice daily (Midday & Evening). The website typically shows the most recent draw.")
        
        with col2:
            st.markdown("### 🔗 Quick Links")
            game_links = {
                'Daily 3': 'https://www.michiganlottery.com/games/draw-games/daily-3',
                'Daily 4': 'https://www.michiganlottery.com/games/draw-games/daily-4',
                'Fantasy 5': 'https://www.michiganlottery.com/games/draw-games/fantasy-5',
                'Lotto 47': 'https://www.michiganlottery.com/games/draw-games/lotto-47',
                'Powerball': 'https://www.michiganlottery.com/games/draw-games/powerball',
                'Mega Millions': 'https://www.michiganlottery.com/games/draw-games/mega-millions',
                'Lucky for Life': 'https://www.michiganlottery.com/games/draw-games/lucky-for-life'
            }
            
            if selected_web_game in game_links:
                st.markdown(f"[🌐 Visit {selected_web_game} Page]({game_links[selected_web_game]})")

        st.markdown("---")

        # Fetch button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Fetch Numbers from Michigan Lottery", use_container_width=True):
                with st.spinner(f"Fetching {selected_web_game} numbers from Michigan Lottery..."):
                    result = scrape_michigan_lottery_official(selected_web_game)
                    
                    if result['success']:
                        st.success("✅ Successfully fetched numbers!")
                        
                        # Display results in a nice format
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("🎯 Game", result['game'])
                            st.metric("📅 Draw Date", result['draw_date'])
                            st.metric("🌐 Source", result['source'])
                        
                        with col2:
                            # Format numbers display based on game type
                            numbers = result['numbers']
                            if selected_web_game in ['Daily 3']:
                                st.metric("🎲 Winning Numbers", f"{numbers[0]} - {numbers[1]} - {numbers[2]}")
                            elif selected_web_game in ['Daily 4']:
                                st.metric("🎲 Winning Numbers", f"{numbers[0]} - {numbers[1]} - {numbers[2]} - {numbers[3]}")
                            elif selected_web_game in ['Fantasy 5', 'Lotto 47']:
                                numbers_str = " - ".join(map(str, numbers[:5]))
                                st.metric("🎲 Winning Numbers", numbers_str)
                            elif selected_web_game in ['Powerball', 'Mega Millions']:
                                main_numbers = " - ".join(map(str, numbers[:5]))
                                bonus = numbers[5] if len(numbers) > 5 else "?"
                                bonus_name = "Powerball" if selected_web_game == "Powerball" else "Mega Ball"
                                st.metric("🎲 Main Numbers", main_numbers)
                                st.metric(f"⭐ {bonus_name}", str(bonus))
                            else:
                                numbers_str = " - ".join(map(str, numbers))
                                st.metric("🎲 Winning Numbers", numbers_str)
                        
                        # Show raw data in expander
                        with st.expander("🔍 Raw Data Details"):
                            st.json(result)
                            
                        # Option to update CSV with this data
                        st.markdown("---")
                        st.markdown("### 💾 Update CSV Data")
                        st.info("**Manual Step:** If this data is more recent than your CSV files, you can manually update the CSV files with these numbers.")
                        
                        # Show which CSV file would need updating
                        csv_mapping = {
                            'Daily 3': 'final_integrated_data/Daily_3_Evening_Complete.csv (or Midday)',
                            'Daily 4': 'final_integrated_data/Daily_4_Evening_Complete.csv (or Midday)', 
                            'Fantasy 5': 'past_games/MI Fantasy 5 numbers from LotteryUSA.csv',
                            'Lotto 47': 'final_integrated_data/Lotto_47_Complete.csv',
                            'Powerball': 'final_integrated_data/Powerball_Complete.csv',
                            'Mega Millions': 'final_integrated_data/Mega_Millions_Complete.csv',
                            'Lucky for Life': 'final_integrated_data/Lucky_for_Life_Complete.csv'
                        }
                        
                        if selected_web_game in csv_mapping:
                            st.code(f"CSV File: {csv_mapping[selected_web_game]}")
                    
                    else:
                        st.error("❌ Failed to fetch numbers")
                        st.error(f"**Error:** {result['error']}")
                        
                        if 'url' in result:
                            st.info(f"**URL Attempted:** {result['url']}")
                        
                        if 'suggestion' in result:
                            st.info(f"**Suggestion:** {result['suggestion']}")
                        
                        if 'available_games' in result:
                            st.info(f"**Available Games:** {', '.join(result['available_games'])}")

        # Manual verification section
        st.markdown("---")
        st.markdown("### 📋 Manual Verification")
        st.markdown("""
        **Recommended Steps:**
        1. **Compare Results**: Check if the numbers match your CSV data
        2. **Verify Date**: Ensure the draw date is what you expect
        3. **Cross-Reference**: Visit the official Michigan Lottery website directly
        4. **Update CSV**: If data is newer, manually update the appropriate CSV file
        """)
        
        # Quick reference for all games
        with st.expander("🎯 All Michigan Lottery Games - Quick Reference"):
            st.markdown("### Official Michigan Lottery Game Pages")
            for game, url in game_links.items():
                st.markdown(f"**{game}:** [🌐 {url}]({url})")


def get_game_drawing_schedule_simple(game_name: str) -> Dict[str, Any]:
    """Simple function to get drawing schedule for display"""
    schedules = {
        'Powerball': {
            'draw_days': ['Monday', 'Wednesday', 'Saturday'],
            'draw_time': '10:59 PM ET',
            'frequency': 'Three times weekly'
        },
        'Mega Millions': {
            'draw_days': ['Tuesday', 'Friday'],
            'draw_time': '11:00 PM ET',
            'frequency': 'Twice weekly'
        },
        'Fantasy 5': {
            'draw_days': ['Daily'],
            'draw_time': '7:29 PM ET',
            'frequency': 'Daily'
        },
        'Lucky for Life': {
            'draw_days': ['Monday', 'Thursday'],
            'draw_time': '10:38 PM ET',
            'frequency': 'Twice weekly'
        },
        'Daily 3': {
            'draw_days': ['Daily'],
            'draw_time': '12:59 PM ET (Midday), 7:29 PM ET (Evening)',
            'frequency': 'Twice daily'
        },
        'Daily 4': {
            'draw_days': ['Daily'],
            'draw_time': '12:59 PM ET (Midday), 7:29 PM ET (Evening)',
            'frequency': 'Twice daily'
        }
    }
    
    # Handle base games for variants
    base_game = game_name
    if 'Midday' in game_name or 'Evening' in game_name:
        base_game = game_name.replace(' Midday', '').replace(' Evening', '')
    
    return schedules.get(base_game, schedules['Powerball'])


def debug_scraping_section(selected_game: str):
    """Debug section for testing scraping functionality"""  
    st.markdown("### 🔍 Debug Scraping")
    
    if st.button("🔍 Debug Enhanced Scraping", help="Test the enhanced web scraping functionality"):
        with st.spinner(f"Testing enhanced scraping for {selected_game}..."):
            try:
                scraped_data = enhanced_fetch_lottery_results_integrated(selected_game)
                if scraped_data['success']:
                    st.success(f"✅ Enhanced scraping successful!")
                    st.json({
                        'numbers': scraped_data['numbers'],
                        'date': scraped_data['date'],
                        'raw_number': scraped_data.get('raw_number'),
                        'last_drawing_info': scraped_data.get('last_drawing_info'),
                        'schedule_info': scraped_data.get('schedule_info')
                    })
                else:
                    st.error(f"❌ Enhanced scraping failed for {selected_game}")
                    st.info(f"Error: {scraped_data.get('error', 'Unknown error')}")
                    st.info(f"Drawing info: {scraped_data.get('last_drawing_info', 'No info')}")
                    
            except Exception as e:
                st.error(f"Debug failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Game selection for web lookup
            web_games = [
                'Daily 3', 'Daily 4', 'Fantasy 5', 'Lotto 47', 
                'Powerball', 'Mega Millions', 'Lucky for Life'
            ]
            
            selected_web_game = st.selectbox(
                "Select Game for Web Lookup:",
                web_games,
                key="web_game_selector"
            )
            
            # Add note about Daily 3/4 variations
            if selected_web_game in ['Daily 3', 'Daily 4']:
                st.info(f"📝 **Note:** {selected_web_game} draws twice daily (Midday & Evening). The website typically shows the most recent draw.")
        
        with col2:
            st.markdown("### 🔗 Quick Links")
            game_links = {
                'Daily 3': 'https://www.michiganlottery.com/games/draw-games/daily-3',
                'Daily 4': 'https://www.michiganlottery.com/games/draw-games/daily-4',
                'Fantasy 5': 'https://www.michiganlottery.com/games/draw-games/fantasy-5',
                'Lotto 47': 'https://www.michiganlottery.com/games/draw-games/lotto-47',
                'Powerball': 'https://www.michiganlottery.com/games/draw-games/powerball',
                'Mega Millions': 'https://www.michiganlottery.com/games/draw-games/mega-millions',
                'Lucky for Life': 'https://www.michiganlottery.com/games/draw-games/lucky-for-life'
            }
            
            if selected_web_game in game_links:
                st.markdown(f"[🌐 Visit {selected_web_game} Page]({game_links[selected_web_game]})")

        st.markdown("---")

        # Fetch button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Fetch Numbers from Michigan Lottery", use_container_width=True):
                with st.spinner(f"Fetching {selected_web_game} numbers from Michigan Lottery..."):
                    result = scrape_michigan_lottery_official(selected_web_game)
                    
                    if result['success']:
                        st.success("✅ Successfully fetched numbers!")
                        
                        # Display results in a nice format
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("🎯 Game", result['game'])
                            st.metric("📅 Draw Date", result['draw_date'])
                            st.metric("🌐 Source", result['source'])
                        
                        with col2:
                            # Format numbers display based on game type
                            numbers = result['numbers']
                            if selected_web_game in ['Daily 3']:
                                st.metric("🎲 Winning Numbers", f"{numbers[0]} - {numbers[1]} - {numbers[2]}")
                            elif selected_web_game in ['Daily 4']:
                                st.metric("🎲 Winning Numbers", f"{numbers[0]} - {numbers[1]} - {numbers[2]} - {numbers[3]}")
                            elif selected_web_game in ['Fantasy 5', 'Lotto 47']:
                                numbers_str = " - ".join(map(str, numbers[:5]))
                                st.metric("🎲 Winning Numbers", numbers_str)
                            elif selected_web_game in ['Powerball', 'Mega Millions']:
                                main_numbers = " - ".join(map(str, numbers[:5]))
                                bonus = numbers[5] if len(numbers) > 5 else "?"
                                bonus_name = "Powerball" if selected_web_game == "Powerball" else "Mega Ball"
                                st.metric("🎲 Main Numbers", main_numbers)
                                st.metric(f"⭐ {bonus_name}", str(bonus))
                            else:
                                numbers_str = " - ".join(map(str, numbers))
                                st.metric("🎲 Winning Numbers", numbers_str)
                        
                        # Show raw data in expander
                        with st.expander("🔍 Raw Data Details"):
                            st.json(result)
                            
                        # Option to update CSV with this data
                        st.markdown("---")
                        st.markdown("### 💾 Update CSV Data")
                        st.info("**Manual Step:** If this data is more recent than your CSV files, you can manually update the CSV files with these numbers.")
                        
                        # Show which CSV file would need updating
                        csv_mapping = {
                            'Daily 3': 'final_integrated_data/Daily_3_Evening_Complete.csv (or Midday)',
                            'Daily 4': 'final_integrated_data/Daily_4_Evening_Complete.csv (or Midday)', 
                            'Fantasy 5': 'past_games/MI Fantasy 5 numbers from LotteryUSA.csv',
                            'Lotto 47': 'final_integrated_data/Lotto_47_Complete.csv',
                            'Powerball': 'final_integrated_data/Powerball_Complete.csv',
                            'Mega Millions': 'final_integrated_data/Mega_Millions_Complete.csv',
                            'Lucky for Life': 'final_integrated_data/Lucky_for_Life_Complete.csv'
                        }
                        
                        if selected_web_game in csv_mapping:
                            st.code(f"CSV File: {csv_mapping[selected_web_game]}")
                    
                    else:
                        st.error("❌ Failed to fetch numbers")
                        st.error(f"**Error:** {result['error']}")
                        
                        if 'url' in result:
                            st.info(f"**URL Attempted:** {result['url']}")
                        
                        if 'suggestion' in result:
                            st.info(f"**Suggestion:** {result['suggestion']}")
                        
                        if 'available_games' in result:
                            st.info(f"**Available Games:** {', '.join(result['available_games'])}")

        # Manual verification section
        st.markdown("---")
        st.markdown("### 📋 Manual Verification")
        st.markdown("""
        **Recommended Steps:**
        1. **Compare Results**: Check if the numbers match your CSV data
        2. **Verify Date**: Ensure the draw date is what you expect
        3. **Cross-Reference**: Visit the official Michigan Lottery website directly
        4. **Update CSV**: If data is newer, manually update the appropriate CSV file
        """)
        
        # Quick reference for all games
        with st.expander("🎯 All Michigan Lottery Games - Quick Reference"):
            st.markdown("### Official Michigan Lottery Game Pages")
            for game, url in game_links.items():
                st.markdown(f"**{game}:** [🌐 {url}]({url})")


if __name__ == "__main__":
    main()

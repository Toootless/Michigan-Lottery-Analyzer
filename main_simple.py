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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for local LLM
local_model = None
local_tokenizer = None
local_pipeline = None

# Global variables for prediction tracking
PREDICTIONS_FILE = "data/predictions_history.json"
PERFORMANCE_FILE = "data/performance_metrics.json"
LEARNING_WEIGHTS_FILE = "data/learning_weights.json"

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
        individual_file = f"data/{game_name.replace(' ', '_').lower()}_suggestions_{file_timestamp}.txt"
        with open(individual_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"📄 Suggestions saved to text file: {individual_file}")
        
        # Create numbers-only file for quick reference
        numbers_only = []
        for strategy_data in suggestions.values():
            numbers_str = ''.join(map(str, strategy_data['numbers']))
            numbers_only.append(numbers_str)
        
        numbers_file = f"data/{game_name.replace(' ', '_').lower()}_numbers_only_{file_timestamp}.txt"
        with open(numbers_file, 'w', encoding='utf-8') as f:
            f.write(f"{game_name.upper()} NUMBERS - {timestamp}\n")
            f.write("=" * 40 + "\n")
            for i, num in enumerate(numbers_only, 1):
                f.write(f"{i:2}. {num}\n")
        print(f"🎯 Numbers-only file saved: {numbers_file}")
        
        # Also append to historical log
        log_file = f"data/{game_name.replace(' ', '_').lower()}_suggestions_log.txt"
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

def save_winning_numbers_to_log(game_name: str, draw_date: datetime, numbers: List[int], bonus_number: int = None, jackpot: float = None):
    """Save winning numbers to a log file with dates"""
    try:
        ensure_data_directory()
        log_file = f"data/{game_name.replace(' ', '_').lower()}_winning_numbers_log.txt"
        
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
        log_file = f"data/{game_name.replace(' ', '_').lower()}_winning_numbers_log.txt"
        
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

def scrape_todays_winning_numbers(game_name: str) -> Dict[str, Any]:
    """Scrape today's winning numbers from Michigan Lottery website"""
    
    # Michigan Lottery URLs for different games
    lottery_urls = {
        'Powerball': 'https://www.michiganlottery.com/games/draw-games/powerball',
        'Mega Millions': 'https://www.michiganlottery.com/games/draw-games/mega-millions',
        'Fantasy 5': 'https://www.michiganlottery.com/games/draw-games/fantasy-5',
        'Daily 3': 'https://www.michiganlottery.com/games/draw-games/daily-3',
        'Daily 4': 'https://www.michiganlottery.com/games/draw-games/daily-4'
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
                if game_name in ['Daily 3', 'Daily 4']:
                    # Daily games format: 123 or 1-2-3
                    pattern = r'\b\d{3,4}\b' if game_name == 'Daily 4' else r'\b\d{3}\b'
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
            
            if game_name in ['Daily 3', 'Daily 4']:
                # Look for 3 or 4 digit sequences
                digit_count = 3 if game_name == 'Daily 3' else 4
                pattern = r'\b\d{' + str(digit_count) + r'}\b'
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
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Fantasy 5'])
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
    
    # Michigan Lottery URLs for different games
    urls = {
        "Fantasy 5": "https://www.michiganlottery.com/draw-games/fantasy-5",
        "Daily 4": "https://www.michiganlottery.com/draw-games/daily-4",
        "Daily 3": "https://www.michiganlottery.com/draw-games/daily-3",
        "Keno": "https://www.michiganlottery.com/draw-games/keno",
        "Lotto 47": "https://www.michiganlottery.com/draw-games/lotto-47"
    }
    
    if game_name not in urls:
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(urls[game_name], headers=headers, timeout=10)
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
        
    except requests.RequestException as e:
        st.error(f"Error fetching data from Michigan Lottery: {str(e)}")
    except Exception as e:
        st.error(f"Error parsing lottery data: {str(e)}")
    
    return None

def extract_winning_numbers_from_html(soup, game_name: str) -> Optional[Dict[str, Any]]:
    """Extract winning numbers from HTML content using multiple strategies"""
    
    # Get all text from the page
    all_text = soup.get_text()
    
    # Strategy 1: Look for specific CSS selectors and patterns
    winning_number_selectors = [
        '.winning-numbers', '.numbers', '.draw-results', '.result-numbers',
        '[class*="number"]', '[class*="winning"]', '[class*="result"]',
        '.game-results', '.latest-draw', '.current-draw'
    ]
    
    numbers = []
    
    # Try CSS selectors first
    for selector in winning_number_selectors:
        elements = soup.select(selector)
        for element in elements:
            element_text = element.get_text()
            
            if game_name in ['Daily 3', 'Daily 4']:
                # Look for 3 or 4 digit sequences
                digit_count = 3 if game_name == 'Daily 3' else 4
                pattern = r'\b\d{' + str(digit_count) + r'}\b'
                matches = re.findall(pattern, element_text)
                if matches:
                    # Take the first plausible match
                    numbers = [int(d) for d in matches[0]]
                    if len(numbers) == digit_count:
                        return {'numbers': numbers}
                
                # Also try looking for spaced digits like "5 7 3 9"
                digit_pattern = r'\d'
                spaced_pattern = r'\b(' + r'\s+'.join([digit_pattern] * digit_count) + r')\b'
                spaced_matches = re.findall(spaced_pattern, element_text)
                if spaced_matches:
                    numbers = [int(d) for d in spaced_matches[0].split()]
                    if len(numbers) == digit_count:
                        return {'numbers': numbers}
            
            elif game_name == "Fantasy 5":
                # Look for 5 numbers between 1-39
                found_nums = re.findall(r'\\b([1-3]?[0-9])\\b', element_text)
                valid_nums = [int(n) for n in found_nums if 1 <= int(n) <= 39]
                if len(valid_nums) >= 5:
                    return {'numbers': sorted(valid_nums[:5])}
    
    # Strategy 2: Fallback - search entire page text
    if game_name == "Daily 4":
        # Multiple patterns for Daily 4
        patterns = [
            r'\b(\d{4})\b',  # 4 digit sequence like "5739"
            r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b',  # Spaced digits like "5 7 3 9"
            r'\b(\d)-(\d)-(\d)-(\d)\b',  # Hyphenated like "5-7-3-9"
            r'\b(\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\b'  # Comma separated like "5,7,3,9"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                if len(matches[0]) == 1:  # Single 4-digit number
                    numbers = [int(d) for d in matches[0]]
                else:  # Multiple captured groups
                    numbers = [int(d) for d in matches[0] if d.isdigit()]
                
                if len(numbers) == 4 and all(0 <= n <= 9 for n in numbers):
                    return {'numbers': numbers}
    
    elif game_name == "Daily 3":
        # Multiple patterns for Daily 3
        patterns = [
            r'\b(\d{3})\b',  # 3 digit sequence like "573"
            r'\b(\d)\s+(\d)\s+(\d)\b',  # Spaced digits like "5 7 3"
            r'\b(\d)-(\d)-(\d)\b',  # Hyphenated like "5-7-3"
            r'\b(\d)\s*,\s*(\d)\s*,\s*(\d)\b'  # Comma separated like "5,7,3"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                if len(matches[0]) == 1:  # Single 3-digit number
                    numbers = [int(d) for d in matches[0]]
                else:  # Multiple captured groups
                    numbers = [int(d) for d in matches[0] if d.isdigit()]
                
                if len(numbers) == 3 and all(0 <= n <= 9 for n in numbers):
                    return {'numbers': numbers}
    
    elif game_name == "Fantasy 5":
        # Look for sequences like "7 14 21 35 42"
        pattern = r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b'
        matches = re.findall(pattern, all_text)
        if matches:
            numbers = [int(n) for n in matches[0]]
            if len(numbers) == 5 and all(1 <= n <= 39 for n in numbers):
                return {'numbers': sorted(numbers)}
    
    # Strategy 3: Debug output and basic fallback
    st.warning(f"Debug: Could not parse winning numbers for {game_name}. Page contains: {all_text[:200]}...")
    
    # Very basic fallback - look for any digit sequences
    all_digits = re.findall(r'\\d', all_text)
    if len(all_digits) >= (3 if game_name == "Daily 3" else 4 if game_name == "Daily 4" else 5):
        digit_count = 3 if game_name == "Daily 3" else 4 if game_name == "Daily 4" else 5
        numbers = [int(d) for d in all_digits[:digit_count]]
        return {'numbers': numbers}
    
    return None

def fetch_and_update_todays_numbers(game_name: str) -> bool:
    """Fetch today's winning numbers and update logs and learning system"""
    
    # Scrape the numbers
    scraped_data = scrape_todays_winning_numbers(game_name)
    
    if not scraped_data:
        st.error(f"Could not fetch today's winning numbers for {game_name}")
        return False
    
    # Save to log
    save_winning_numbers_to_log(
        game_name, 
        scraped_data['numbers'], 
        scraped_data.get('bonus_number'),
        scraped_data['date'],
        scraped_data['time']
    )
    
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
    'Fantasy 5': {
        'numbers_count': 5,
        'numbers_range': (1, 39),
        'draw_days': ['Daily']
    },
    'Daily 3': {
        'numbers_count': 3,
        'numbers_range': (0, 9),
        'draw_days': ['Daily']
    },
    'Daily 4': {
        'numbers_count': 4,
        'numbers_range': (0, 9),
        'draw_days': ['Daily']
    }
}

# Configure Streamlit page
st.set_page_config(
    page_title="🎰 Michigan Lottery Analyzer",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_lottery_data(game_name: str, days: int = 30) -> List[Dict]:
    """Get lottery data, prioritizing logged actual results over sample data"""
    # First, try to load actual winning numbers from log
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
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Fantasy 5'])
    results = []
    
    current_date = datetime.now()
    
    for i in range(days):
        draw_date = current_date - timedelta(days=i)
        
        # Generate random numbers
        numbers_count = game_config['numbers_count']
        numbers_range = game_config['numbers_range']
        
        if game_name in ["Daily 3", "Daily 4"]:
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
            'numbers': sorted(numbers) if game_name not in ["Daily 3", "Daily 4"] else numbers,
            'bonus_number': bonus_number,
            'jackpot': jackpot
        }
        
        results.append(result)
    
    return results

def analyze_frequency(results: List[Dict]) -> Dict[str, Any]:
    """Analyze number frequency from results"""
    all_numbers = []
    for result in results:
        all_numbers.extend(result['numbers'])
    
    if not all_numbers:
        return {}
    
    frequency = Counter(all_numbers)
    avg_frequency = statistics.mean(frequency.values())
    
    hot_numbers = [num for num, freq in frequency.items() if freq >= avg_frequency * 1.2]
    cold_numbers = [num for num, freq in frequency.items() if freq <= avg_frequency * 0.8]
    
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
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Fantasy 5'])
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
    if game_name in ["Daily 3", "Daily 4"]:
        return generate_daily_digit_suggestions(analysis, game_name)
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Fantasy 5'])
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
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Fantasy 5'])
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
    if game_name in ["Daily 3", "Daily 4"]:
        return generate_daily_digit_suggestions(analysis, game_name)
    
    game_config = LOTTERY_GAMES.get(game_name, LOTTERY_GAMES['Fantasy 5'])
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
            if selected_game in ["Daily 3", "Daily 4"]:
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
    
    # Analysis results tabs (full width to avoid column nesting issues)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Analysis", "📊 Visualization", "🎯 Number Suggestions", "🧠 Learning Dashboard", "🤖 Chat", "🔍 Number Checker"])
    
    with tab1:
            if st.session_state.get('run_analysis', False):
                with st.spinner(f"Analyzing {selected_game} data..."):
                    # Generate sample data
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
                            st.dataframe(sum_df, width='stretch')
                    
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
                st.info("👆 Select a game and click 'Run Analysis' to start")
    
    with tab2:
            st.subheader("📊 Data Visualization")
            
            if st.session_state.get('run_analysis', False):
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
                    st.dataframe(df.sort_values('Frequency', ascending=False), width='stretch')
                    
                    # Most/Least common numbers
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Most Common Numbers")
                        most_common = analysis.get('most_common', [])[:5]
                        most_common_df = pd.DataFrame(most_common, columns=['Number', 'Frequency'])
                        st.dataframe(most_common_df, width='stretch')
                    
                    with col2:
                        st.subheader("Least Common Numbers")
                        least_common = analysis.get('least_common', [])[:5]
                        least_common_df = pd.DataFrame(least_common, columns=['Number', 'Frequency'])
                        st.dataframe(least_common_df, width='stretch')
            else:
                st.info("Run analysis first to see visualizations")
    
    with tab3:
            st.subheader("🎯 Smart Number Suggestions")
            
            if st.session_state.get('run_analysis', False):
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
                if selected_game in ["Daily 3", "Daily 4"]:
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
                            bonus_name = "Powerball" if selected_game == "Powerball" else "Mega Ball" if selected_game == "Mega Millions" else "Bonus"
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
                    st.dataframe(recent_df, width='stretch', hide_index=True)
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
                        st.dataframe(trend_df, width='stretch', hide_index=True)
            
            with col_right:
                if selected_game in performance_metrics.get("strategy_performance", {}):
                    st.markdown("### 📊 Strategy Performance Analysis")
                
                strategy_perfs = performance_metrics["strategy_performance"][selected_game]
                
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
                    st.dataframe(perf_df, width='stretch')
                    
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
                            bonus_label = "Power Ball" if selected_game == "Powerball" else "Mega Ball" if selected_game == "Mega Millions" else "Bonus Number"
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
                        st.dataframe(frequency_df, width='stretch')
                        
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
                elif selected_game in ["Daily 3", "Daily 4"]:
                    st.info("💡 **Tip**: Daily digit games with multiple play styles and smaller prizes")
    
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
    st.sidebar.dataframe(results_df, width='stretch')
        
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
    st.sidebar.dataframe(results_df, width='stretch')
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
        if st.button("🌐 Fetch Today's Numbers", type="primary", help="Automatically fetch today's winning numbers from Michigan Lottery website"):
            with st.spinner(f"Fetching today's winning numbers for {selected_game}..."):
                success = fetch_and_update_todays_numbers(selected_game)
                if success:
                    st.success(f"✅ Successfully fetched and logged today's winning numbers for {selected_game}!")
                    st.balloons()
                    # Refresh the page to show updated data
                    st.rerun()
                else:
                    st.error(f"❌ Could not fetch today's numbers for {selected_game}. Please try manual entry below.")
    
    with col_auto2:
        st.info("🔄 This will automatically fetch today's winning numbers from the Michigan Lottery website and update your logs and learning dashboard.")
        
        # Debug button for testing scraping
        if st.button("🔍 Debug Scraping", help="Test the web scraping functionality for debugging"):
            with st.spinner(f"Testing scraping for {selected_game}..."):
                try:
                    scraped_data = scrape_todays_winning_numbers(selected_game)
                    if scraped_data:
                        st.success(f"✅ Scraping successful!")
                        st.json(scraped_data)
                    else:
                        st.error(f"❌ Scraping failed for {selected_game}")
                        
                        # Try to show debug info
                        import requests
                        from bs4 import BeautifulSoup
                        
                        urls = {
                            "Fantasy 5": "https://www.michiganlottery.com/draw-games/fantasy-5",
                            "Daily 4": "https://www.michiganlottery.com/draw-games/daily-4",
                            "Daily 3": "https://www.michiganlottery.com/draw-games/daily-3",
                            "Keno": "https://www.michiganlottery.com/draw-games/keno",
                            "Lotto 47": "https://www.michiganlottery.com/draw-games/lotto-47"
                        }
                        
                        if selected_game in urls:
                            st.info(f"Attempting to fetch from: {urls[selected_game]}")
                            
                except Exception as e:
                    st.error(f"Debug failed: {str(e)}")
    
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
        
        game_config = LOTTERY_GAMES.get(selected_game, {})
        numbers_count = game_config.get('numbers_count', 5)
        numbers_range = game_config.get('numbers_range', (1, 50))
        
        # Numbers input (no nested columns to avoid Streamlit nesting error)
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
            numbers_input = st.text_input(
                f"Winning Numbers ({numbers_count} numbers: {numbers_range[0]}-{numbers_range[1]})",
                placeholder=f"e.g., 7,14,21,35,42",
                help=f"Enter {numbers_count} winning numbers separated by commas"
            )
        
        # Bonus number input for applicable games
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
        
        # Jackpot input for major games
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
                    if len(numbers) != numbers_count:
                        st.error(f"Please enter exactly {numbers_count} numbers for {selected_game}")
                    elif any(n < numbers_range[0] or n > numbers_range[1] for n in numbers):
                        st.error(f"All numbers must be between {numbers_range[0]} and {numbers_range[1]}")
                    else:
                        # Combine date and time
                        draw_datetime = datetime.combine(draw_date, draw_time)
                        
                        # Save to log
                        success = save_winning_numbers_to_log(
                            selected_game, 
                            draw_datetime, 
                            numbers, 
                            bonus_input, 
                            jackpot_input
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

    # Chat input (outside of columns/tabs to avoid Streamlit restrictions)
    if local_pipeline or OLLAMA_AVAILABLE:
        st.markdown("---")
        st.subheader("💬 Ask the Local AI Assistant")
        
        # Chat input
        if prompt := st.chat_input("Ask about lottery patterns..."):
            if "messages" not in st.session_state:
                llm_type = "Local GPU-accelerated" if local_pipeline else "Ollama"
                st.session_state.messages = [
                    {"role": "assistant", "content": f"Hi! I'm your lottery analysis assistant powered by {llm_type} LLM. I can help you understand patterns in {selected_game} data. What would you like to know?"}
                ]
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate AI response
            if st.session_state.get('run_analysis', False):
                results = get_lottery_data(selected_game, days_to_analyze)
                analysis = analyze_frequency(results)
                deep_patterns = analyze_deep_patterns(results, selected_game)
                relationships = analyze_number_relationships(results)
                temporal_patterns = analyze_temporal_patterns(results)
                
                with st.spinner("🤖 AI is thinking..."):
                    response = chat_with_ai(prompt, analysis, selected_game, deep_patterns, relationships, temporal_patterns)
            else:
                response = f"Please run an analysis first by clicking '🔍 Run Analysis' in the sidebar, then I can help you understand the {selected_game} patterns and statistics!"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Refresh the page to show new messages
            st.rerun()

if __name__ == "__main__":
    main()

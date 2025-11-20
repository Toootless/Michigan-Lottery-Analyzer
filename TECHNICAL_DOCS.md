# 🔧 Technical Documentation - Michigan Lottery Analyzer

> **Main Application**: `MichiganLotteryAnalyzer.py`

## 📋 System Overview

The Michigan Lottery Analyzer is a sophisticated AI-powered application built on modern Python technologies with GPU acceleration capabilities. This document provides comprehensive technical details for developers, system administrators, and advanced users.

## 🏗️ Architecture

### Core Components

```
Michigan Lottery Analyzer/
├── MichiganLotteryAnalyzer.py  # Main application (3,600+ lines)
│   ├── AI/GPU Integration      # PyTorch + CUDA acceleration  
│   ├── Game Engine            # Lottery game logic
│   ├── Pattern Analysis       # Statistical algorithms
│   ├── Learning System        # Adaptive AI improvements
│   ├── Web Scraping System    # Automated data collection
│   ├── Date/Time Logging      # Standard time format (12-hour AM/PM)
│   ├── Data Management        # JSON + text file persistence
│   └── Web Interface          # Enhanced Streamlit frontend
├── data/                      # Runtime data storage
├── config/                    # Configuration files
└── documentation/             # User and technical docs
```

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | Streamlit | 1.28+ | Web UI framework |
| **AI/ML** | PyTorch | 2.7.1+cu118 | GPU acceleration |
| **NLP** | Transformers | 4.57.1 | Language models |
| **Data** | Pandas/NumPy | Latest | Data processing |
| **Visualization** | Plotly | Latest | Interactive charts |
| **GPU** | CUDA | 11.8 | Hardware acceleration |
| **Text Output** | Native Python | 3.9+ | File generation system |
| **Web Scraping** | BeautifulSoup + Requests | Latest | Automated data collection |
| **Date/Time** | datetime | Standard | Timestamp tracking |

### 🌐 New Features (v2.4+)

#### Automated Web Scraping System
- **Real-time Data Collection**: Automatically fetch winning numbers from Michigan Lottery website
- **Multi-Game Support**: Fantasy 5, Daily 3, Daily 4, Keno, Lotto 47
- **Intelligent Parsing**: Multiple HTML parsing strategies with fallback mechanisms
- **Error Handling**: Graceful degradation with user-friendly error messages

#### Date/Time Logging System
- **Timestamp Tracking**: Full date and time logging for all winning number entries using standard 12-hour AM/PM format
- **Historical Data**: Persistent storage with JSON and text file logging
- **Manual Override**: User can specify custom dates and times for historical entries
- **Automatic Dating**: Real-time timestamps for scraped data in standard time format
- **Time Format**: All timestamps use 12-hour format with AM/PM indicators (e.g., "07:05:00 PM" instead of "19:05:00")

## 🎮 Game Engine Architecture

### Game Configuration System

```python
LOTTERY_GAMES = {
    'Powerball': {
        'numbers_count': 5,          # Main numbers to select
        'numbers_range': (1, 69),    # Valid number range
        'bonus_range': (1, 26),      # Power Ball range
        'draw_days': ['Monday', 'Wednesday', 'Saturday'],
        'allows_duplicates': False   # Unique numbers only
    },
    'Daily 4': {
        'numbers_count': 4,
        'numbers_range': (0, 9),
        'draw_days': ['Daily'],
        'allows_duplicates': True    # Digits can repeat
    }
    # ... additional games
}
```

### Validation Logic

#### Lottery Games (Powerball, Mega Millions, Fantasy 5)
- **Unique Numbers**: Each number must be different
- **Range Validation**: Numbers must be within game-specific range
- **Count Validation**: Exact number count required
- **Bonus Numbers**: Separate validation for secondary numbers

#### Daily Games (Daily 3, Daily 4)
- **Duplicates Allowed**: Positions can have same digit (e.g., 0077)
- **Order Matters**: Position-dependent validation (123 ≠ 321)
- **Digit Range**: 0-9 for each position independently
- **No Bonus**: Single set of numbers only
- **Text Output**: Automated file generation with multiple formats

## 🚀 AI/GPU Integration

### PyTorch Configuration

```python
# GPU Detection and Setup
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# Memory Optimization for RTX 3060
model_kwargs = {
    "low_cpu_mem_usage": True,
    "torch_dtype": torch.float16,  # Half precision
    "device_map": "auto"           # Automatic GPU mapping
}
```

### Model Loading Strategy

1. **Model Candidates** (in priority order):
   - `microsoft/DialoGPT-medium` (Primary choice for RTX 3060)
   - `gpt2` (Fallback option)
   - `distilgpt2` (Minimal resource usage)

2. **Memory Management**:
   - 90% GPU memory allocation for model
   - 10% buffer for processing
   - Automatic batch size adjustment

3. **Error Handling**:
   - Graceful degradation to CPU mode
   - Model fallback chain
   - User-friendly error messages

### Local LLM Pipeline

```python
def initialize_local_llm():
    """Initialize local LLM with GPU optimization"""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        
        # Create optimized pipeline
        pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            batch_size=1
        )
        
        return True
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        return False
```

## 📊 Pattern Analysis Engine

### Statistical Algorithms

#### 1. Hot/Cold Number Analysis
```python
def analyze_hot_cold_numbers(results, threshold_hot=1.2, threshold_cold=0.8):
    """
    Identifies frequently appearing (hot) and rare (cold) numbers
    
    Args:
        results: Historical lottery results
        threshold_hot: Multiplier for hot classification
        threshold_cold: Multiplier for cold classification
    
    Returns:
        dict: Hot and cold number classifications
    """
    frequency = calculate_number_frequency(results)
    expected = len(results) / get_total_possible_numbers()
    
    hot_numbers = [n for n, freq in frequency.items() 
                   if freq > expected * threshold_hot]
    cold_numbers = [n for n, freq in frequency.items() 
                    if freq < expected * threshold_cold]
    
    return {'hot': hot_numbers, 'cold': cold_numbers}
```

#### 2. Sum Range Analysis
```python
def analyze_sum_ranges(results):
    """
    Analyzes the sum of winning numbers for pattern detection
    
    Statistical theory: Lottery sums tend to cluster around mean
    """
    sums = [sum(result['numbers']) for result in results]
    
    return {
        'mean': np.mean(sums),
        'std': np.std(sums),
        'common_ranges': identify_common_ranges(sums),
        'recommendations': suggest_sum_targets(sums)
    }
```

#### 3. Consecutive Number Patterns
```python
def analyze_consecutive_patterns(results):
    """
    Identifies patterns in consecutive number appearances
    """
    consecutive_counts = []
    for result in results:
        numbers = sorted(result['numbers'])
        consecutive = count_consecutive_numbers(numbers)
        consecutive_counts.append(consecutive)
    
    return {
        'avg_consecutive': np.mean(consecutive_counts),
        'distribution': np.bincount(consecutive_counts),
        'recommendations': generate_consecutive_advice(consecutive_counts)
    }
```

### 12 Deep Pattern Types

1. **Hot/Cold Analysis**: Frequency-based number classification
2. **Sum Range Analysis**: Total sum pattern detection
3. **Even/Odd Patterns**: Balance analysis between even and odd numbers
4. **Consecutive Analysis**: Sequential number appearance patterns
5. **Decade Distribution**: Number spread across 10s (1-10, 11-20, etc.)
6. **Gap Analysis**: Spacing between drawn numbers
7. **Repeat Analysis**: Numbers appearing in consecutive draws
8. **Last Digit Patterns**: Ending digit distribution (1, 11, 21, etc.)
9. **Prime Number Analysis**: Prime vs composite number frequency
10. **Quadrant Analysis**: Number distribution across range quarters
11. **Seasonal Patterns**: Time-based drawing variations
12. **Combination Patterns**: Multi-number sequence analysis

## 🎯 Daily Games Digit Generation System

### Specialized Daily 3/4 Algorithm

```python
def generate_daily_digit_suggestions(analysis: Dict[str, Any], game_name: str) -> Dict[str, Any]:
    """
    Generate digit-based suggestions specifically for Daily 3 and Daily 4 games
    
    Key Differences from Lottery Games:
    - Each position is independent (allows duplicates)
    - Order matters (123 ≠ 321)
    - Range is 0-9 for all positions
    - No sorting applied to maintain position integrity
    """
    digits_count = 3 if game_name == "Daily 3" else 4
    
    strategies = {
        'balanced_mix': "Even distribution across all positions",
        'hot_numbers_focus': "Emphasizes frequently drawn digits", 
        'due_theory': "Focus on overdue digits",
        'repeating_pattern': "Includes duplicate digits (common in daily games)",
        'sequential_pattern': "Consecutive or wrapped digit sequence",
        'mirror_pattern': "Symmetrical digit arrangement (Daily 4 only)",
        'all_same': "All positions same digit",
        'mixed_frequency': "Alternating hot and cold digits by position"
    }
    
    # Generate 8 different digit combinations
    # Each maintains position order and allows duplicates
    return digit_suggestions
```

### Text Output Integration

The Daily Games system automatically generates multiple file formats:

```python
# File outputs for each analysis
formats = {
    'full_report': 'daily_3_suggestions_20251027_193006.txt',    # Complete analysis
    'numbers_only': 'daily_3_numbers_only_20251027_193006.txt',  # Just the digits
    'historical_log': 'daily_3_suggestions_log.txt'             # Append-only history
}

# Example output format:
"""
=== DAILY 3 NUMBER SUGGESTIONS ===
Generated: 2025-10-27 07:30:06 PM
Analysis Period: 3650 days
Game: Daily 3 (Duplicates allowed)

Strategy 1: Balanced Mix
  Numbers: 468
  Confidence: Medium
  Method: Even distribution across positions

Strategy 2: Hot Numbers Focus  
  Numbers: 035
  Confidence: High
  Method: Emphasizes frequently drawn digits
"""
```

## 🌐 Web Scraping System (v2.4+)

### Automated Data Collection

The system now includes automated web scraping capabilities to fetch real-time winning numbers from the Michigan Lottery website.

#### Supported Games
- **Fantasy 5**: 5 numbers (1-39) ✅ Real data available
- **Daily 4**: 4 digits (0-9) ✅ Real data available (Midday/Evening variants)  
- **Daily 3**: 3 digits (0-9) ✅ Real data available (Midday/Evening variants)
- **Powerball**: 5 numbers (1-69) + Power Ball (1-26) ✅ Real data available
- **Mega Millions**: 5 numbers (1-70) + Mega Ball (1-25) ✅ Real data available

#### Web Scraping Architecture

```python
def scrape_todays_winning_numbers(game_name: str) -> Optional[Dict[str, Any]]:
    """
    Scrape today's winning numbers from Michigan Lottery website
    
    Process:
    1. Fetch HTML from game-specific URL
    2. Parse using BeautifulSoup with multiple strategies
    3. Extract numbers using regex patterns
    4. Validate results against game rules
    5. Return structured data with metadata
    """
    
    # Michigan Lottery URLs
    urls = {
        "Fantasy 5": "https://www.michiganlottery.com/draw-games/fantasy-5",
        "Daily 4": "https://www.michiganlottery.com/draw-games/daily-4",
        # ... other games
    }
    
    # HTTP request with proper headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...'
    }
    
    response = requests.get(urls[game_name], headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    return extract_winning_numbers_from_html(soup, game_name)
```

#### HTML Parsing Strategies

1. **CSS Selector Approach**: Look for common winning number containers
   ```python
   selectors = [
       '.winning-numbers', '.numbers', '.draw-results',
       '[class*="number"]', '[class*="winning"]'
   ]
   ```

2. **Regex Pattern Matching**: Multiple patterns for different number formats
   ```python
   patterns = {
       "Daily 4": [
           r'\b(\d{4})\b',                    # 4-digit sequence
           r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b',  # Spaced digits
           r'\b(\d)-(\d)-(\d)-(\d)\b'         # Hyphenated
       ]
   }
   ```

3. **Fallback Mechanisms**: Graceful degradation when primary parsing fails

#### Integration with Learning System

```python
def fetch_and_update_todays_numbers(game_name: str) -> bool:
    """
    Complete workflow: scrape → validate → log → update learning
    
    1. Scrape winning numbers from website
    2. Save to JSON log with timestamp
    3. Update text log file
    4. Check against existing predictions
    5. Update performance metrics
    6. Trigger learning weight adjustments
    """
```

#### Error Handling & Debugging

- **Request Failures**: Network timeouts, HTTP errors
- **Parsing Failures**: Malformed HTML, changed website structure  
- **Validation Failures**: Numbers outside valid ranges
- **Debug Mode**: Detailed logging and HTML content preview

## 🖥️ Enhanced User Interface (v2.4+)

### Automated Data Collection Interface

The dashboard now includes dedicated sections for automated and manual data collection:

#### 🌐 Automated Data Collection Section
- **Fetch Today's Numbers Button**: One-click automated scraping with progress indicators
- **Real-time Feedback**: Success/error messages with detailed status information
- **Auto-refresh**: Dashboard updates automatically after successful data collection
- **Debug Tools**: Testing and troubleshooting interface for development

#### 📝 Manual Entry Section
- **Date/Time Controls**: Full timestamp specification for historical entries
- **Game-specific Input**: Customized input fields based on selected game type
- **Input Validation**: Real-time validation with helpful error messages
- **Flexible Formats**: Support for comma-separated, spaced, or continuous number entry

#### Enhanced Dashboard Features
- **Timestamp Display**: All winning numbers now show date and time of entry
- **Data Source Indicators**: Clear labeling of manual vs. scraped data
- **Improved Layout**: Better organization with collapsible sections
- **Error Prevention**: Fixed Streamlit column nesting issues

### UI Components

```python
# Automated fetch button with spinner and feedback
if st.button("🌐 Fetch Today's Numbers", type="primary"):
    with st.spinner(f"Fetching today's winning numbers for {selected_game}..."):
        success = fetch_and_update_todays_numbers(selected_game)
        if success:
            st.success("✅ Successfully fetched and logged!")
            st.balloons()
            st.rerun()  # Refresh to show new data

# Debug interface for testing scraping
if st.button("🔍 Debug Scraping"):
    scraped_data = scrape_todays_winning_numbers(selected_game)
    if scraped_data:
        st.json(scraped_data)  # Show parsed results
```

### User Experience Improvements
- **Progressive Disclosure**: Advanced features hidden in expandable sections
- **Visual Feedback**: Icons, colors, and animations for better user guidance
- **Responsive Design**: Optimal layout across different screen sizes
- **Accessibility**: Clear labels and help tooltips throughout interface

## 🧠 Learning System

### Daily Verification Process

```python
def verify_previous_predictions(game_name):
    """
    Daily verification system that compares predictions to actual results
    
    Process:
    1. Fetch yesterday's predictions
    2. Get actual lottery results
    3. Calculate accuracy metrics
    4. Update learning weights
    5. Store performance data
    """
    predictions = load_predictions_for_date(yesterday)
    actual_results = fetch_actual_results(game_name, yesterday)
    
    accuracy_metrics = calculate_accuracy(predictions, actual_results)
    updated_weights = adjust_learning_weights(accuracy_metrics)
    
    save_performance_metrics(accuracy_metrics)
    save_learning_weights(updated_weights)
    
    return accuracy_metrics
```

### Adaptive Weight System

```python
# Learning weights structure
learning_weights = {
    "hot_numbers": 1.0,      # Hot number emphasis
    "cold_numbers": 0.8,     # Cold number emphasis  
    "sum_ranges": 1.2,       # Sum range importance
    "even_odd": 0.9,         # Even/odd balance
    "consecutive": 0.7,      # Consecutive patterns
    "last_updated": timestamp
}

def adjust_learning_weights(accuracy_metrics):
    """
    Adjusts algorithm weights based on prediction accuracy
    
    Successful patterns get increased weight
    Poor performers get decreased weight
    """
    for pattern, accuracy in accuracy_metrics.items():
        if accuracy > 0.6:  # Above average
            learning_weights[pattern] *= 1.1  # Increase weight
        elif accuracy < 0.3:  # Below average
            learning_weights[pattern] *= 0.9  # Decrease weight
    
    return learning_weights
```

## 🔍 Number Checker System

### Input Validation Pipeline

```python
def validate_lottery_input(numbers, game_config):
    """
    Comprehensive validation for lottery number input
    
    Validation Steps:
    1. Count validation (exact number required)
    2. Range validation (within game limits)
    3. Duplicate validation (game-dependent)
    4. Type validation (integer conversion)
    """
    numbers_count = game_config['numbers_count']
    numbers_range = game_config['numbers_range']
    allows_duplicates = game_config.get('allows_duplicates', False)
    
    # Step 1: Count validation
    if len(numbers) != numbers_count:
        return ValidationResult(
            valid=False,
            error=f"Need exactly {numbers_count} numbers"
        )
    
    # Step 2: Range validation
    invalid_numbers = [n for n in numbers 
                      if not (numbers_range[0] <= n <= numbers_range[1])]
    if invalid_numbers:
        return ValidationResult(
            valid=False,
            error=f"Invalid numbers: {invalid_numbers}"
        )
    
    # Step 3: Duplicate validation (game-dependent)
    if not allows_duplicates and len(set(numbers)) != len(numbers):
        duplicates = find_duplicates(numbers)
        return ValidationResult(
            valid=False,
            error=f"Duplicates not allowed: {duplicates}"
        )
    
    return ValidationResult(valid=True, numbers=numbers)
```

### Game-Specific Validation Rules

#### Daily Games (3 & 4)
```python
# Daily games allow duplicates - each position is independent
def validate_daily_game(digits):
    """
    Daily 3: [0,7,0] is valid
    Daily 4: [0,7,0,7] is valid
    Position matters - order is preserved
    """
    return all(0 <= digit <= 9 for digit in digits)
```

#### Lottery Games
```python
# Lottery games require unique numbers
def validate_lottery_game(numbers, game_range):
    """
    Powerball: [7,14,21,35,42] is valid
    [7,14,21,35,35] is invalid (duplicate 35)
    Order doesn't matter - numbers are sorted for display
    """
    return (len(set(numbers)) == len(numbers) and
            all(game_range[0] <= n <= game_range[1] for n in numbers))
```

## 💾 Data Management

### Storage Architecture

```python
# Enhanced data storage structure (v2.4+)
DATA_STRUCTURE = {
    # New: Winning Numbers Log with date/time tracking
    "winning_numbers_log.json": {
        "Fantasy 5": [
            {
                "numbers": [5, 12, 18, 25, 31],
                "bonus_number": null,
                "date": "2025-10-27",
                "time": "19:00:00",
                "source": "Michigan Lottery Website"
            }
        ],
        "Daily 4": [
            {
                "numbers": [5, 7, 3, 9],
                "date": "2025-10-27", 
                "time": "12:59:00 PM",
                "source": "User Entry"
            }
        ]
    },
    
    # New: Human-readable log file
    "winning_numbers_log.txt": "Append-only text log with timestamps",
    
    "predictions_history.json": {
        "predictions": [
            {
                "id": "unique_identifier",
                "timestamp": "2025-10-27T12:00:00",
                "game_name": "Powerball",
                "numbers": [7, 14, 21, 35, 42],
                "bonus_number": 15,
                "confidence": 0.85,
                "patterns_used": ["hot_numbers", "sum_ranges"]
            }
        ],
        "last_updated": "2025-10-27T12:00:00"
    },
    
    # Text file outputs for Daily games
    "daily_3_suggestions_YYYYMMDD_HHMMSS.txt": "Full analysis report",
    "daily_3_numbers_only_YYYYMMDD_HHMMSS.txt": "Numbers-only format",
    "daily_3_suggestions_log.txt": "Historical append-only log",
    "daily_4_suggestions_YYYYMMDD_HHMMSS.txt": "Full analysis report",
    "daily_4_numbers_only_YYYYMMDD_HHMMSS.txt": "Numbers-only format", 
    "daily_4_suggestions_log.txt": "Historical append-only log",
    
    "learning_weights.json": {
        "weights": {
            "hot_numbers": 1.2,
            "cold_numbers": 0.8,
            # ... all pattern weights
        },
        "last_updated": "2025-10-27T12:00:00"
    },
    
    "performance_metrics.json": {
        "daily_metrics": [
            {
                "date": "2025-10-27",
                "game": "Powerball",
                "predictions": 3,
                "matches": 1,
                "accuracy": 0.33
            }
        ],
        "overall_stats": {
            "total_predictions": 150,
            "total_matches": 45,
            "overall_accuracy": 0.30
        }
    }
}
```

### Daily Games Text Output System

```python
def print_daily_game_suggestions(game_name: str, suggestions: Dict[str, Any], days: int) -> str:
    """
    Generate formatted text files for Daily 3/4 games
    
    Creates multiple output formats:
    1. Full detailed report with strategies and confidence levels
    2. Numbers-only file for quick reference  
    3. Historical log for tracking over time
    
    File naming convention:
    - daily_3_suggestions_YYYYMMDD_HHMMSS.txt (full report)
    - daily_3_numbers_only_YYYYMMDD_HHMMSS.txt (numbers only)
    - daily_3_suggestions_log.txt (historical append-only)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
    file_timestamp = datetime.now().strftime('%Y%m%d_%I%M%S%p')
    
    # Generate formatted output with strategies
    printout_lines = [
        f"=== {game_name.upper()} NUMBER SUGGESTIONS ===",
        f"Generated: {timestamp}",
        f"Analysis Period: {days} days", 
        f"Game: {game_name} (Duplicates allowed)",
        "=" * 40
    ]
    
    # Add 8 different strategies with digit-based logic
    for i, (strategy_key, strategy_data) in enumerate(suggestions.items(), 1):
        numbers_str = ''.join(map(str, strategy_data['numbers']))
        printout_lines.extend([
            f"Strategy {i}: {strategy_data['strategy']}",
            f"  Numbers: {numbers_str}",
            f"  Confidence: {strategy_data['confidence']}",
            f"  Method: {strategy_data['description']}"
        ])
    
    return '\n'.join(printout_lines)
```

### Data Retention Policy

```python
def cleanup_old_data():
    """
    Automatic data cleanup to maintain performance
    
    Retention Periods:
    - Predictions: 10 years (3,650 days)
    - Learning weights: Persistent
    - Performance metrics: 10 years
    - Analysis cache: 30 days (for large datasets)
    - Text output files: User-managed (stored in data/ folder)
    """
    cutoff_date = datetime.now() - timedelta(days=3650)  # 10 years
    
    # Remove old predictions
    predictions = load_predictions_history()
    predictions['predictions'] = [
        p for p in predictions['predictions']
        if datetime.fromisoformat(p['timestamp']) > cutoff_date
    ]
    save_predictions_history(predictions)
```

## 🔧 Configuration System

### Environment Variables

```bash
# Optional configuration via .env file
OPENAI_API_KEY=your_api_key_here
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
STREAMLIT_SERVER_PORT=8501
DEBUG_MODE=False
```

### Runtime Configuration

```python
# Application configuration
APP_CONFIG = {
    "gpu": {
        "enable_cuda": True,
        "memory_fraction": 0.9,
        "fallback_to_cpu": True
    },
    "models": {
        "auto_initialize": True,
        "model_priority": [
            "microsoft/DialoGPT-medium",
            "gpt2",
            "distilgpt2"
        ]
    },
    "analysis": {
        "default_days": 3650,  # Default to 10 years
        "max_days": 3650,      # 10 years maximum
        "pattern_count": 12,
        "quick_select_periods": [30, 180, 365, 1825, 3650],
        "performance_thresholds": {
            "fast": 365,      # 1 year
            "optimal": 1825,  # 5 years
            "large": 3650     # 10 years
        }
    },
    "ui": {
        "tabs": 6,
        "default_tab": "Analysis",
        "theme": "light"
    }
}
```

## 🔍 Debugging & Monitoring

### Logging System

```python
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lottery_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage throughout application
logger.info("GPU initialization successful")
logger.warning("Falling back to CPU mode")
logger.error("Model loading failed", exc_info=True)
```

### Performance Monitoring

```python
def monitor_performance():
    """
    Real-time performance monitoring
    """
    metrics = {
        "gpu_memory": get_gpu_memory_usage(),
        "cpu_usage": get_cpu_usage(),
        "memory_usage": get_ram_usage(),
        "model_load_time": measure_model_load_time(),
        "analysis_speed": measure_analysis_speed()
    }
    
    return metrics
```

### Error Handling Strategy

```python
class LotteryAnalyzerError(Exception):
    """Base exception for lottery analyzer"""
    pass

class GPUInitializationError(LotteryAnalyzerError):
    """GPU setup failed"""
    pass

class ModelLoadError(LotteryAnalyzerError):
    """AI model loading failed"""
    pass

class ValidationError(LotteryAnalyzerError):
    """Input validation failed"""
    pass

# Comprehensive error handling
try:
    initialize_gpu()
    load_ai_models()
    start_analysis()
except GPUInitializationError:
    logger.warning("GPU unavailable, falling back to CPU")
    initialize_cpu_mode()
except ModelLoadError:
    logger.error("AI models unavailable, using statistical mode")
    disable_ai_features()
except ValidationError as e:
    display_user_error(str(e))
```

## 🚀 Performance Optimization

### GPU Memory Management

```python
def optimize_gpu_memory():
    """
    RTX 3060 memory optimization strategies
    """
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory efficient attention
        torch.backends.cuda.enable_math_sdp(True)
        
        # Monitor memory usage
        memory_stats = torch.cuda.memory_stats()
        logger.info(f"GPU memory: {memory_stats}")
```

### Processing Optimization

```python
def optimize_data_processing():
    """
    Data processing optimizations
    """
    # Use numpy for numerical operations
    import numpy as np
    
    # Vectorized operations instead of loops
    def calculate_frequencies_vectorized(numbers_array):
        unique, counts = np.unique(numbers_array, return_counts=True)
        return dict(zip(unique, counts))
    
    # Pandas optimizations
    pd.set_option('mode.chained_assignment', None)
    
    # Caching for repeated calculations
    from functools import lru_cache
    
    @lru_cache(maxsize=128)
    def cached_pattern_analysis(data_hash):
        return expensive_pattern_calculation(data_hash)
```

### 10-Year Analysis Optimization

```python
def optimize_large_dataset_analysis():
    """
    Performance optimizations for 10-year historical analysis
    """
    # Batch processing for large datasets
    def process_in_batches(data, batch_size=1000):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    # Memory-efficient pattern analysis
    def streaming_pattern_analysis(results):
        patterns = defaultdict(Counter)
        for batch in process_in_batches(results, 1000):
            batch_patterns = analyze_patterns_batch(batch)
            for pattern_type, counts in batch_patterns.items():
                patterns[pattern_type].update(counts)
        return patterns
    
    # Progressive loading for very large datasets
    def progressive_data_loading(days):
        if days > 1825:  # More than 5 years
            # Load data in yearly chunks
            chunk_size = 365
            logger.info(f"Processing {days} days in {chunk_size}-day chunks")
        else:
            # Standard loading for smaller datasets
            chunk_size = days
        return chunk_size

# Performance scaling recommendations
DATASET_PERFORMANCE_GUIDE = {
    "small": {
        "days": "7-90",
        "performance": "Excellent",
        "recommendation": "Real-time analysis"
    },
    "medium": {
        "days": "91-365", 
        "performance": "Very Good",
        "recommendation": "Quick processing with comprehensive patterns"
    },
    "large": {
        "days": "366-1825",
        "performance": "Good",
        "recommendation": "GPU acceleration recommended"
    },
    "extra_large": {
        "days": "1826-3650",
        "performance": "Moderate",
        "recommendation": "GPU acceleration essential, longer processing time"
    }
}
```

## 🧪 Testing Strategy

### Unit Tests

```python
import unittest
from unittest.mock import patch, MagicMock

class TestLotteryAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = LotteryAnalyzer()
        self.sample_data = generate_test_data()
    
    def test_number_validation(self):
        """Test number validation for different games"""
        # Test Powerball validation
        valid_powerball = [1, 15, 23, 45, 67]
        self.assertTrue(self.analyzer.validate_numbers(
            valid_powerball, 'Powerball'
        ))
        
        # Test Daily 4 with duplicates
        valid_daily4 = [0, 7, 0, 7]
        self.assertTrue(self.analyzer.validate_numbers(
            valid_daily4, 'Daily 4'
        ))
    
    def test_pattern_analysis(self):
        """Test pattern analysis algorithms"""
        patterns = self.analyzer.analyze_patterns(self.sample_data)
        
        self.assertIn('hot_numbers', patterns)
        self.assertIn('cold_numbers', patterns)
        self.assertIsInstance(patterns['hot_numbers'], list)
    
    @patch('torch.cuda.is_available')
    def test_gpu_fallback(self, mock_cuda):
        """Test graceful fallback when GPU unavailable"""
        mock_cuda.return_value = False
        
        result = self.analyzer.initialize_gpu()
        self.assertFalse(result)
        self.assertEqual(self.analyzer.device, 'cpu')
```

### Integration Tests

```python
class TestIntegration(unittest.TestCase):
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis workflow"""
        # Generate sample data
        data = generate_sample_lottery_data('Powerball', 90)
        
        # Run analysis
        results = run_complete_analysis(data)
        
        # Verify results structure
        self.assertIn('patterns', results)
        self.assertIn('suggestions', results)
        self.assertIn('confidence', results)
        
        # Verify data types
        self.assertIsInstance(results['suggestions'], list)
        self.assertIsInstance(results['confidence'], float)
```

## 📦 Deployment & Distribution

### Package Structure

```
lottery_analyzer/
├── MichiganLotteryAnalyzer.py  # Main application
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── README.md              # User documentation
├── CHANGELOG.md           # Version history
├── LICENSE                # Software license
├── data/                  # Data directory
├── config/                # Configuration files
├── tests/                 # Test suite
├── docs/                  # Documentation
└── scripts/               # Utility scripts
    ├── setup_python311.bat
    ├── install_cuda.sh
    └── benchmark.py
```

### Installation Scripts

```bash
#!/bin/bash
# install.sh - Automated installation script

set -e

echo "Installing Michigan Lottery Analyzer..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$python_version" < "3.10" ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Install dependencies
pip3 install -r requirements.txt

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA packages..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No GPU detected, installing CPU-only packages..."
    pip3 install torch torchvision torchaudio
fi

# Install additional AI packages
pip3 install transformers accelerate safetensors

echo "Installation complete!"
echo "Run with: streamlit run MichiganLotteryAnalyzer.py"
```

### Docker Support

```dockerfile
# Dockerfile for containerized deployment
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run application
CMD ["streamlit", "run", "MichiganLotteryAnalyzer.py"]
```

## 🔒 Security Considerations

### API Key Management

```python
# Secure API key handling
import os
from pathlib import Path

def load_api_keys():
    """
    Load API keys from secure sources
    Priority: Environment variables > .env file > user input
    """
    api_key = None
    
    # 1. Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 2. Check .env file
    if not api_key:
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    
    # 3. Secure input prompt
    if not api_key:
        api_key = getpass.getpass("Enter OpenAI API key: ")
    
    return api_key
```

### Input Sanitization

```python
def sanitize_user_input(user_input):
    """
    Sanitize user input to prevent injection attacks
    """
    # Remove potentially dangerous characters
    safe_chars = set("0123456789,-. ")
    sanitized = ''.join(c for c in user_input if c in safe_chars)
    
    # Limit input length
    max_length = 100
    sanitized = sanitized[:max_length]
    
    # Validate format
    if not re.match(r'^[\d,\s.-]+$', sanitized):
        raise ValidationError("Invalid input format")
    
    return sanitized
```

## 📈 Monitoring & Analytics

### Usage Analytics

```python
def track_usage_analytics():
    """
    Anonymous usage analytics for improvement
    """
    analytics = {
        "session_start": datetime.now().isoformat(),
        "features_used": [],
        "analysis_count": 0,
        "games_analyzed": set(),
        "gpu_enabled": torch.cuda.is_available(),
        "python_version": sys.version_info[:2]
    }
    
    return analytics

def log_feature_usage(feature_name):
    """Log feature usage for analytics"""
    if "analytics" in st.session_state:
        st.session_state.analytics["features_used"].append({
            "feature": feature_name,
            "timestamp": datetime.now().isoformat()
        })
```

## 🛡️ Error Recovery

### Graceful Degradation

```python
class FeatureManager:
    """
    Manages feature availability and graceful degradation
    """
    def __init__(self):
        self.features = {
            "gpu_acceleration": False,
            "local_llm": False,
            "advanced_analysis": True,
            "basic_analysis": True
        }
    
    def check_feature_availability(self):
        """Check and update feature availability"""
        # GPU features
        if torch.cuda.is_available():
            self.features["gpu_acceleration"] = True
        
        # LLM features
        try:
            initialize_local_llm()
            self.features["local_llm"] = True
        except Exception:
            self.features["local_llm"] = False
    
    def get_available_features(self):
        """Return list of available features"""
        return [k for k, v in self.features.items() if v]
```

---

## 🔗 API Reference

### Core Classes

#### `LotteryAnalyzer`
Main analysis engine for lottery pattern detection and prediction.

**Methods:**
- `analyze_patterns(data, days)`: Comprehensive pattern analysis
- `generate_suggestions(patterns)`: AI-powered number suggestions
- `validate_numbers(numbers, game)`: Input validation
- `calculate_confidence(patterns)`: Prediction confidence scoring

#### `GameEngine`
Handles game-specific logic and validation.

**Methods:**
- `get_game_config(game_name)`: Retrieve game configuration
- `validate_game_input(numbers, game)`: Game-specific validation
- `format_display(numbers, game)`: Format numbers for display

#### `LearningSystem`
Manages adaptive learning and prediction improvement.

**Methods:**
- `verify_predictions(date)`: Daily prediction verification
- `update_weights(accuracy)`: Adjust learning weights
- `get_performance_metrics()`: Retrieve accuracy statistics

## 📝 Version History & Changelog

### Version 2.4.0 (October 2025) - Current
**Major Features Added:**
- ✅ **Automated Web Scraping System**: Real-time data collection from Michigan Lottery website
- ✅ **Date/Time Logging**: Full timestamp tracking for all winning number entries
- ✅ **Enhanced Dashboard**: Automated and manual data collection interfaces
- ✅ **Midday/Evening Draw Support**: Separate Daily 3/4 Midday and Evening game variants
- ✅ **Real Data Integration**: Winning numbers log with actual lottery results
- ✅ **Improved Error Handling**: Better web scraping reliability and user feedback
- ✅ **Debug Tools**: Testing and troubleshooting interfaces for developers

**Technical Improvements:**
- Enhanced HTML parsing with multiple fallback strategies
- Expanded data persistence with JSON and text file logging  
- Fixed Streamlit column nesting issues
- Added comprehensive input validation and sanitization
- Improved user experience with progress indicators and feedback

**Bug Fixes:**
- Fixed f-string and raw string syntax error in regex patterns
- Resolved Streamlit UI layout issues with nested columns
- Enhanced error messages for failed web scraping attempts

### Version 2.3.0 (Prior Release)
**Features:**
- AI/GPU Integration with PyTorch 2.7.1+cu118
- Learning system with adaptive weights
- Multi-game support for 7+ lottery types
- Pattern analysis engine with 10 analysis strategies
- JSON-based data persistence
- Streamlit web interface

### Version History Summary
| Version | Release Date | Key Features | Lines of Code |
|---------|-------------|--------------|---------------|
| v2.4.0 | Oct 2025 | Web scraping, date logging, enhanced UI | 3,600+ |
| v2.3.0 | Sep 2025 | AI/GPU integration, learning system | 3,100+ |
| v2.2.0 | Aug 2025 | Multi-game support, pattern analysis | 2,500+ |
| v2.1.0 | Jul 2025 | Basic prediction system | 2,000+ |
| v2.0.0 | Jun 2025 | Initial Streamlit interface | 1,500+ |

---

This technical documentation provides comprehensive coverage of the system architecture, implementation details, and operational procedures for the Michigan Lottery Analyzer v2.4.0. The system now includes automated web scraping, comprehensive date/time logging, and enhanced user interface features. For additional support or contributions, please refer to the project repository and community guidelines.
"""
Configuration Management for Lottery Analyzer
Handles environment variables and application settings
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class Config:
    """Application configuration manager"""
    
    def __init__(self):
        self.load_environment()
        self.setup_logging()
    
    def load_environment(self):
        """Load environment variables from .env file"""
        env_file = CONFIG_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Azure OpenAI (Optional)
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        self.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        # Michigan Lottery Configuration
        self.MICHIGAN_LOTTERY_BASE_URL = os.getenv("MICHIGAN_LOTTERY_BASE_URL", "https://www.michiganlottery.com")
        self.SCRAPING_DELAY = int(os.getenv("SCRAPING_DELAY", "2"))
        self.MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
        
        # Database Configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/lottery_results.db")
        self.BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
        
        # Analysis Configuration
        self.DEFAULT_ANALYSIS_DAYS = int(os.getenv("DEFAULT_ANALYSIS_DAYS", "365"))
        self.MINIMUM_SAMPLE_SIZE = int(os.getenv("MINIMUM_SAMPLE_SIZE", "100"))
        self.PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.75"))
        
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "localhost")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = LOGS_DIR / os.getenv("LOG_FILE", "lottery_analyzer.log")
        self.ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
        
        # Visualization Settings
        self.DEFAULT_CHART_THEME = os.getenv("DEFAULT_CHART_THEME", "plotly_white")
        self.MAX_CHART_DATA_POINTS = int(os.getenv("MAX_CHART_DATA_POINTS", "1000"))
        self.CHART_WIDTH = int(os.getenv("CHART_WIDTH", "800"))
        self.CHART_HEIGHT = int(os.getenv("CHART_HEIGHT", "600"))
        
        # Machine Learning Settings
        self.MODEL_CACHE_ENABLED = os.getenv("MODEL_CACHE_ENABLED", "true").lower() == "true"
        self.MODEL_CACHE_DIR = DATA_DIR / os.getenv("MODEL_CACHE_DIR", "models")
        self.TRAINING_VALIDATION_SPLIT = float(os.getenv("TRAINING_VALIDATION_SPLIT", "0.8"))
        self.RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
        
        # Web Scraping Settings
        self.USER_AGENT = os.getenv("USER_AGENT", "LotteryAnalyzer/1.0 (Educational Research)")
        self.SELENIUM_HEADLESS = os.getenv("SELENIUM_HEADLESS", "true").lower() == "true"
        self.SELENIUM_TIMEOUT = int(os.getenv("SELENIUM_TIMEOUT", "30"))
        
        # Performance Settings
        self.ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
        
        # Ensure model cache directory exists
        self.MODEL_CACHE_DIR.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Configure application logging"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            format=log_format
        )
        
        # Add file handler if enabled
        if self.ENABLE_FILE_LOGGING:
            file_handler = logging.FileHandler(self.LOG_FILE)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
    
    def get_lottery_games(self) -> Dict[str, Dict[str, Any]]:
        """Get supported lottery games configuration"""
        return {
            "powerball": {
                "name": "Powerball",
                "numbers_count": 5,
                "numbers_range": (1, 69),
                "powerball_range": (1, 26),
                "url_path": "/powerball/winning-numbers",
                "draw_days": ["Monday", "Wednesday", "Saturday"]
            },
            "mega_millions": {
                "name": "Mega Millions", 
                "numbers_count": 5,
                "numbers_range": (1, 70),
                "mega_ball_range": (1, 25),
                "url_path": "/mega-millions/winning-numbers",
                "draw_days": ["Tuesday", "Friday"]
            },
            "fantasy_5": {
                "name": "Fantasy 5",
                "numbers_count": 5,
                "numbers_range": (1, 39),
                "url_path": "/fantasy-5/winning-numbers",
                "draw_days": ["Daily"]
            },
            "daily_3": {
                "name": "Daily 3",
                "numbers_count": 3,
                "numbers_range": (0, 9),
                "url_path": "/daily-3/winning-numbers", 
                "draw_days": ["Daily"]
            },
            "daily_4": {
                "name": "Daily 4",
                "numbers_count": 4,
                "numbers_range": (0, 9),
                "url_path": "/daily-4/winning-numbers",
                "draw_days": ["Daily"]
            },
            "keno": {
                "name": "Keno",
                "numbers_count": 20,
                "numbers_range": (1, 80),
                "url_path": "/keno/winning-numbers",
                "draw_days": ["Every 4 minutes"]
            },
            "lucky_for_life": {
                "name": "Lucky for Life",
                "numbers_count": 5,
                "numbers_range": (1, 48),
                "lucky_ball_range": (1, 18),
                "url_path": "/lucky-for-life/winning-numbers",
                "draw_days": ["Monday", "Thursday"]
            }
        }
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate critical configuration settings"""
        validation_results = {
            "openai_configured": bool(self.OPENAI_API_KEY),
            "database_accessible": True,  # TODO: Implement actual DB check
            "model_cache_writable": self.MODEL_CACHE_DIR.is_dir() and os.access(self.MODEL_CACHE_DIR, os.W_OK),
            "log_directory_writable": LOGS_DIR.is_dir() and os.access(LOGS_DIR, os.W_OK),
            "data_directory_writable": DATA_DIR.is_dir() and os.access(DATA_DIR, os.W_OK)
        }
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                # Hide sensitive information
                if 'api_key' in attr_name.lower() or 'secret' in attr_name.lower():
                    config_dict[attr_name] = "***HIDDEN***" if attr_value else None
                else:
                    config_dict[attr_name] = attr_value
        return config_dict


# Global configuration instance
_config_instance = None

def load_config() -> Config:
    """Load and return global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)
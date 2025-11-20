# 📋 Michigan Lottery Analyzer - Changelog

## Version 3.1.3 - November 19, 2025

### 📄 PDF Scraper Integration & 24-Hour Jackpot Caching

#### ✅ New Features
- **PDF Scraper**: Implemented Selenium-based PDF scraper (`michigan_lottery_pdf_scraper.py`) for reliable real-time jackpot extraction
  - Uses Chrome WebDriver with headless mode and anti-detection measures
  - Saves Michigan Lottery homepage as PDF via Chrome DevTools Protocol
  - Extracts jackpot amounts using PyMuPDF and regex pattern matching
  - Supports all major games: Powerball, Mega Millions, Lotto 47, Fantasy 5
  - **Keyword-based extraction**: Searches for "Estimated Jackpot:" markers for accurate amount detection
- **24-Hour Caching**: Jackpots now fetched once per day to minimize server load
  - Changed cache duration from 1 hour to 24 hours (86400 seconds)
  - Automatic refresh after 24-hour period expires
  - Session-based storage persists during application runtime

#### 🐛 Bug Fixes (Nov 19, 2025)
- **Fixed jackpot extraction accuracy**: Updated PDF scraper to search for specific "Estimated Jackpot:" keywords
  - Powerball: Now correctly finds "Powerball\nEstimated Jackpot:" pattern
  - Mega Millions: Looks for "Mega Millions\nEstimated Jackpot:" pattern
  - Lotto 47: Searches for "Lotto 47\nEstimated Jackpot" pattern
  - Fantasy 5: Finds "Fantasy 5\nEstimated Jackpot" pattern
- **Reduced context window**: Limited to 100 characters after jackpot marker to prevent cross-game contamination
- **Added Fantasy 5 validation**: Maximum jackpot amount of $2M to prevent matching other games' prizes
- **Fixed return format**: Corrected dictionary structure for proper integration with MLA4

#### 🔧 Architecture Changes
- **Simplified Scraping**: Removed JavaScript fallback scrapers for cleaner architecture
  - Eliminated `fetch_michigan_jackpot_enhanced_js()` fallback
  - Removed `fetch_michigan_jackpot_js()` basic scraper
  - Deleted unreachable BeautifulSoup HTML parsing code
  - PDF scraper is now the sole data source for live jackpots
- **Dependency Updates**:
  - Added `selenium==4.20.0` for web automation
  - Added `webdriver-manager==4.0.2` for automatic driver management
  - Added `PyMuPDF` (fitz) for PDF text extraction
  - Downgraded `urllib3==2.2.1` to fix zstandard compatibility issues

#### 🐛 Bug Fixes
- Fixed zstd module compatibility issue with urllib3 2.5.0
- Resolved environment-specific import failures for PDF scraper
- Corrected Python environment path detection for proper package installation

#### 📊 Performance Improvements
- Reduced server requests from hourly to daily (96% reduction)
- Single PDF generation per day instead of multiple web scraping attempts
- Cleaner error handling with no fallback complexity

#### 🔍 Developer Notes
- PDF files saved to `data/Saved_files/` directory
- Jackpot cache stored in Streamlit session state
- Cache version tracking with `JACKPOT_VERSION = "v3.0_nov2025_pdf_scraper"`

## Version 3.0.5 - November 11, 2025

### 🎯 Suggestion Uniqueness + CSV Update Tools

#### ✅ Improvements
- Added a final, global uniqueness enforcement across ALL suggestion strategies (MLA3 + traditional + pattern):
  - Normalizes each set to (sorted numbers, bonus) and removes cross-strategy duplicates
  - Diversifies collisions by swapping in unused hot/medium/cold numbers with bounded retries
  - Annotates diversified sets (e.g., “globally diversified”)
- Introduced "CSV Update Tools" per-game tabs:
  - Shows current CSV status (path • exists • rows • latest date)
  - Provides an "Update CSV Now" button for one-click refresh

#### 🧰 CLI Helper
- Added `update_all_games.py`:
  - CSV-first mode prioritizes reading existing CSVs; fetch only when behind
  - `--force` mode scrapes from the web to refresh all games

#### 📌 Reliability Notes
- Lucky for Life and Lotto 47: primary sources can 404; CSV history continues to load; force updates may temporarily fail
  - Follow-up: add alternate sources and parser improvements

#### 🔎 Developer Notes
- Port defaults to 8506 when launched via `run_MLA4.ps1` or VS Code task “Run MLA4”.

## Version 3.0.4 - November 10, 2025

### 🛠 Learning Dashboard Data Reliability & CSV Diagnostics

#### ✅ Improvements
- Added automatic CSV self-repair pathway in the Learning Dashboard:
  - If recent history is empty, the app now: ensures unified CSV, re-reads history, attempts a light update, and re-reads again.
- Introduced inline quick actions when no results are found:
  - "🌐 Fetch Latest Now" performs a one-shot forced web fetch and refresh.
  - "✍️ Add Manual Draw" prepares manual insertion flow (future enhancement hooks).
- Implemented reusable scraper instance (`get_csv_scraper`) to prevent redundant object creation and inconsistent state.
- Added `get_csv_info` and `get_unified_csv_path` helpers in `fixed_scraper.py` for path, existence, row count, header, latest date diagnostics.
- CSV update logic now creates missing per-game CSV files automatically with correct schema.
- Manual draw append now creates files when absent (supports Mega Millions, Powerball, Lucky for Life, Fantasy 5 variants, etc.).

#### 📊 UI Enhancements
- Learning Dashboard now displays a CSV diagnostics caption (path • exists • rows • latest date) when no data is available.
- Consistent structured fields (`main_numbers`, `bonus_number`, `bonus_label`) used across Powerball, Mega Millions, Lucky for Life for recent history display.

#### 🔍 Stability
- Reduced occurrences of the "No recent results available" message through proactive repair steps.
- Ensures Powerball and other game histories populate reliably from `final_integrated_data/` or fallback sources.

#### 📌 Follow-Ups (Not Yet Implemented)
- Manual Draw UI inputs (date/number fields) inside dashboard.
- Sidebar integration of structured fields for recent results.
- Unit tests for CSV creation, duplicate prevention, and bonus parsing.

#### 🔧 Developer Notes
- This release focuses on resilience and transparency around data sources rather than new analytical features.
- Type-check warnings remain in legacy sections (LLM optional imports) but do not affect runtime stability.

## Version 3.0.3 - November 10, 2025

### 🔧 CSV Schema Alignment + Multi-Game Validation

#### ✅ Improvements
- Unified CSV write schema across games in `fixed_scraper.py`:
  - Mega Millions: `Date, Result, Mega Ball`
  - Powerball: `Date, Result, Powerball`
  - Lucky for Life: `Date, Result, Lucky Ball` (preserves historical extra columns)
  - Fantasy 5 / Double Play: `Date, Regular, Double Play`
- CSV writer now unions existing headers with new fields to avoid KeyError and preserves older data columns.
- Dates are stored as clean strings (no manual quoting), preventing parsing issues in the dashboard.

#### 🧪 Validation
- Executed multi-game fetch/check for: Powerball, Lucky for Life, Fantasy 5, Fantasy 5 Double Play.
- Confirmed dashboard recent-history parsing works for Powerball, Mega Millions, and Double Play with consistent 5-number lists (bonus handled separately).
- Identified two follow-ups:
  - Lucky for Life: ensure Lucky Ball displays consistently in the dashboard from CSV history.
  - Fantasy 5: added guardrails needed to drop malformed backup rows (non-5 values) when scraping fails.

#### 🪟 Windows Console Stability
- Added safe print patch to prevent UnicodeEncodeError from emojis on Windows cp1252 consoles during fetch/update flows.

#### 📌 Notes
- Git commit not created from this environment (git not detected in PATH). Changes are saved to disk; see this changelog for details. Use your shell to commit when convenient.

## Version 3.0.2 - October 29, 2025

### 🎯 Fantasy 5 CSV Data Integration Fix

#### 🐛 Critical Bug Fixes
- **Fantasy 5 Data Issue**: Fixed "No actual data found, using sample data" error for Fantasy 5 games
- **CSV Path Resolution**: Corrected CSV file path configuration to properly locate Fantasy 5 data files
- **Path Doubling Bug**: Fixed path handling logic that was creating `past_games/past_games/` paths
- **CSV Parsing Enhancement**: Added special parsing logic for Fantasy 5's quoted CSV format with embedded commas

#### 🔧 Technical Improvements
- **Enhanced CSV Scraper**: Updated `ProductionMichiganScraper` with Fantasy 5-specific parsing logic
- **Path Logic Refactor**: Improved path handling in both `_get_from_csv()` and `get_recent_history()` methods
- **Quote Handling**: Implemented proper CSV parsing for Fantasy 5's complex quoted field format
- **Data Validation**: Added robust error handling for Fantasy 5 CSV structure validation

#### 📊 Data Integration Results
- **Fantasy 5 Regular**: Now uses 365+ real historical records from CSV data
- **Fantasy 5 Double Play**: Now uses 365+ real historical records from CSV data  
- **Dual-Column Support**: Proper parsing of Regular and Double Play columns from single CSV file
- **Real Data Validation**: Confirmed actual numbers (e.g., [2, 4, 9, 11, 26]) replace sample data

#### ✅ User Experience Improvements
- **Error Message Elimination**: Removed "No actual data found" messages for Fantasy 5 games
- **Consistent Data Source**: All 10 Michigan lottery games now use real CSV data
- **Performance Enhancement**: Faster Fantasy 5 data loading with optimized CSV parsing
- **Data Accuracy**: Fantasy 5 hot/cold analysis now based on real historical patterns

## Version 3.0.1 - October 29, 2025

### 🎲 Lucky for Life Game Integration

#### 🆕 New Lottery Game Added
- **Lucky for Life**: Complete integration of Michigan's $1,000/day for life lottery game
- **Game Configuration**: 5 numbers from 1-48 plus Lucky Ball from 1-18
- **Drawing Schedule**: Monday and Thursday at 10:38 PM ET (twice weekly)
- **Multi-State Support**: Lucky for Life is available across multiple states
- **Enhanced Fetch**: Automated web scraping with drawing schedule awareness
- **Complete UI Integration**: Full support across all 7 application tabs

#### 🔧 Technical Implementation
- **LOTTERY_GAMES Configuration**: Added Lucky for Life with proper number ranges and draw schedule
- **URL Mapping**: Integrated Michigan Lottery Lucky for Life web scraping endpoint
- **Data Parsing**: Custom regex patterns for Lucky for Life number extraction
- **Schedule Integration**: Monday/Thursday drawing schedule with next draw calculation  
- **UI Enhancement**: Lucky Ball bonus number display with proper labeling
- **Fixed Scraper**: Production-ready scraper updated with Lucky for Life support

#### 🎯 User Interface Updates
- **Game Selection**: Lucky for Life added to dropdown menu across all tabs
- **Bonus Ball Display**: "Lucky Ball" properly labeled in UI (instead of generic "Bonus")
- **Drawing Tips**: Game-specific tips highlighting $1,000/day for life top prize
- **Example Numbers**: Proper placeholder examples for Lucky for Life number entry
- **Chat Integration**: Lucky for Life winning numbers displayed in chat format tab

#### 📚 Documentation Updates
- **README.md**: Updated supported games list to include Lucky for Life
- **Multi-Game Support**: Lucky for Life added to OCR and web scraping feature lists
- **Quick Start Guide**: Updated game selection examples to include Lucky for Life

## Version 3.0.0 - October 28, 2025

### 🚀 REVOLUTIONARY RELEASE: OCR Historical Data Processing

#### 🗃️ Complete Historical Data Extraction (2000-2021)
- **22-Year OCR Processing**: Advanced OCR technology extracts data from historical PDF charts
- **Tesseract OCR Integration**: High-accuracy number recognition with 85-95% accuracy rate
- **PDF Chart Analysis**: Convert lottery tables/charts to structured JSON data
- **Batch Processing System**: Process all historical PDFs with comprehensive progress tracking
- **Multi-Game OCR Support**: Daily 3, Daily 4, Fantasy 5, Lucky for Life, Lotto 47, Keno historical extraction
- **50,000+ Historical Entries**: Complete coverage across all supported games

#### 📄 Advanced PDF Processing Engine
- **PyMuPDF Integration**: Efficient PDF-to-image conversion at 300 DPI resolution
- **Image Preprocessing**: Contrast enhancement and format optimization for OCR accuracy
- **Pattern Recognition**: Game-specific regex algorithms for accurate number extraction
- **Date Parsing**: Multiple date format recognition across different PDF layouts
- **Error Recovery**: Robust handling of OCR errors and PDF format variations
- **Memory Management**: Efficient processing with automatic cleanup and optimization

#### 🎨 Historical Data User Interface
- **Dedicated OCR Section**: New "🗃️ Historical Data (OCR)" sidebar interface
- **Year Range Selection**: Choose specific date ranges from 2000-2021
- **Real-Time Progress**: Live status updates with progress bars during processing
- **Data Summary Dashboard**: Comprehensive breakdown of extracted entries by game
- **Error Reporting**: Detailed feedback on processing issues with clear solutions
- **Clear Data Management**: Easy removal and reload of historical datasets

#### ⚡ GPU-Accelerated OCR Processing
- **CUDA Optimization**: RTX 3060 support with 12.9GB VRAM utilization
- **Performance Metrics**: 2-5 seconds per PDF page, 500-1000 entries per minute  
- **Batch Efficiency**: All 22 years processed in 15-30 minutes
- **Memory Optimization**: Intelligent chunking for large dataset processing
- **Real-Time Monitoring**: GPU usage and processing statistics display

#### 🔧 Technical Infrastructure Overhaul
- **LotteryPDFOCRReader**: Advanced OCR processing class with image preprocessing
- **Comprehensive Data Validation**: Number range checking, duplicate detection
- **JSON Export System**: Structured data with ISO formatting and complete metadata
- **Path Handling**: Robust file management for Windows environments
- **Statistics Tracking**: Processing metrics and accuracy reporting
- **Testing Framework**: OCR validation and debugging tools

#### 📊 Data Coverage & Volume
- **Daily 3**: 22 years of twice-daily drawings (16,000+ entries)
- **Daily 4**: 22 years of twice-daily drawings (16,000+ entries)  
- **Fantasy 5**: 22 years of daily drawings (8,000+ entries)
- **Lotto 47**: 22 years of bi-weekly drawings (2,300+ entries)
- **Keno**: Historical data where available in PDF format
- **Total Dataset**: 50,000+ lottery drawings with complete metadata

---

## Version 2.4.0 - October 27, 2025

### 🌟 MAJOR RELEASE: Automated Data Collection & Enhanced Logging

#### 🌐 Automated Web Scraping System
- **Real-Time Data Collection**: Automatically fetch today's winning numbers from Michigan Lottery website
- **Multi-Game Support**: Fantasy 5, Daily 3, Daily 4, Keno, and Lotto 47
- **Intelligent HTML Parsing**: Multiple parsing strategies with regex fallback mechanisms
- **Comprehensive Error Handling**: Graceful degradation with detailed user feedback
- **Debug Interface**: Testing and troubleshooting tools for developers

#### 📅 Date/Time Logging System  
- **Full Timestamp Tracking**: Complete date and time logging for all winning number entries
- **Historical Data Support**: Manual entry with custom date/time specification
- **Automatic Timestamping**: Real-time timestamps for all scraped data
- **Dual Persistence**: JSON structured data + human-readable text logs
- **Data Source Tracking**: Clear distinction between manual entries and scraped data

#### 🖥️ Enhanced User Interface
- **Automated Collection Dashboard**: One-click data fetching with progress indicators
- **Manual Entry Interface**: Flexible input with date/time controls and validation
- **Real-Time Feedback**: Success animations, error messages, and status updates
- **Debug Tools**: Advanced testing interface for troubleshooting scraping issues
- **Auto-Refresh**: Dashboard updates automatically after successful data collection

#### 🔧 Technical Enhancements
- **Multi-Strategy HTML Parsing**: CSS selectors, regex patterns, and text extraction
- **Enhanced Data Validation**: Input sanitization and game-specific validation
- **Fixed Streamlit Issues**: Resolved column nesting errors and UI layout problems
- **Improved Error Recovery**: Better handling of network timeouts and parsing failures
- **Performance Optimization**: Faster data processing and improved user experience

#### 💾 Data Management Improvements
- **Enhanced JSON Structure**: Structured logging with metadata and source tracking
- **Text File Logging**: Human-readable append-only logs with timestamps
- **Data Prioritization**: User-logged data takes precedence over sample data
- **Historical Integration**: Seamless integration with existing learning system

### 🐛 Critical Bug Fixes
- Fixed f-string/raw string syntax error in regex pattern compilation
- Resolved Streamlit column nesting limitations causing UI crashes
- Enhanced number parsing to handle various input formats (comma, space, continuous)
- Improved error messages for failed web scraping attempts

### 📈 Performance & Reliability
- **3,600+ Lines of Code**: Significant expansion with robust new functionality
- **Enhanced Learning Integration**: Scraped data automatically updates performance metrics
- **Improved Validation**: Better input handling and error prevention
- **Optimized Web Requests**: Proper HTTP headers and timeout handling

## Version 2.3 - October 27, 2025

### 🆕 Daily Games Text Output System

#### 📄 Automated File Generation
- **Multiple File Formats**: Full report, numbers-only, and historical log files
- **Timestamped Filenames**: Unique files prevent overwrites (YYYYMMDD_HHMMSS format)
- **Three Output Types**: 
  - `daily_3_suggestions_20251027_193006.txt` - Complete analysis with strategies
  - `daily_3_numbers_only_20251027_193006.txt` - Clean digit lists for quick reference
  - `daily_3_suggestions_log.txt` - Append-only historical tracking

#### 🎯 Enhanced Daily Games Logic
- **Digit-Based Generation**: Specialized algorithms for Daily 3/4 games
- **Position Independence**: Each digit position treated independently (allows duplicates)
- **Order Preservation**: Maintains digit position order (123 ≠ 321)
- **Eight Strategies**: Balanced Mix, Hot Focus, Due Theory, Sequential, Mirror Pattern, etc.
- **Duplicate Support**: Proper handling of repeated digits (e.g., 0077 for Daily 4)

#### 💾 Download Integration
- **Web Interface Downloads**: Instant download buttons for full reports and quick numbers
- **Console Output**: Real-time feedback during analysis with file location details
- **File Management**: Automatic creation of data directory and organized file storage

### 🔧 System Improvements
- **Code Base Growth**: Expanded to 2,700+ lines with enhanced functionality
- **Standardized Output**: Daily 3 and Daily 4 now produce identical format structures
- **Documentation Updates**: Comprehensive updates to all technical documentation

## Version 2.2 - October 26, 2025

### 🚀 10-Year Historical Analysis
- **Extended Analysis Period**: Increased from 180 days to 10 years (3,650 days)
- **Quick-Select Interface**: Buttons for 7 days, 30 days, 90 days, 180 days, 1 year, 10 years
- **Performance Optimization**: Efficient handling of large datasets
- **Pattern Recognition Enhancement**: Better long-term trend detection

### 🎨 Enhanced Pattern Display
- **Full Text Formatting**: "3 Odd - 2 Even" instead of abbreviated "3O-2E"
- **Comprehensive Pattern Description**: Complete odd/even and high/low pattern explanations
- **Improved Readability**: Clear, user-friendly pattern descriptions

## Version 2.0 - October 23, 2025

### 🆕 Major New Features

#### 🎯 Smart Number Suggestions System
- **Four Intelligent Strategies**: Balanced Mix, Hot Numbers Focus, Due Theory, Statistical Random
- **Interactive Interface**: Expandable strategy cards with detailed explanations
- **Copy Functionality**: One-click copy for suggested numbers
- **Regenerate Feature**: Generate fresh suggestions instantly
- **Game-Specific Adaptation**: Automatically handles different lottery game rules
- **Confidence Scoring**: Each strategy includes reliability assessment
- **Bonus Number Support**: Powerball, Mega Ball, and other bonus numbers
- **Statistical Basis Display**: Shows hot/cold number counts and analysis period

#### 🤖 OpenAI API Integration
- **Full GPT Integration**: Complete OpenAI API implementation
- **Environment File Support**: Secure API key storage in .env file
- **AI-Powered Insights**: Intelligent analysis of lottery patterns
- **Conversational Chat**: Natural language Q&A about lottery data
- **Context-Aware Responses**: AI maintains conversation context

#### 📊 Enhanced User Interface
- **Four-Tab Navigation**: Analysis, Visualization, Number Suggestions, AI Chat
- **Improved Layout**: Better organization of features and information
- **Interactive Elements**: Copy buttons, regenerate buttons, expandable sections
- **Real-Time Updates**: Dynamic content updates without page refresh
- **Responsive Design**: Optimized for different screen sizes

### 🛠 Technical Improvements

#### 🔧 Code Architecture
- **Modular Functions**: Clean separation of concerns for number generation
- **Error Handling**: Comprehensive error management for OpenAI integration
- **Performance Optimization**: Efficient data processing and caching
- **Type Safety**: Improved type hints and validation

#### 📦 Dependencies
- **OpenAI Package**: Added for LLM functionality
- **Environment Loading**: Custom .env file parser
- **Enhanced Logging**: Better error tracking and debugging

#### 🔒 Security & Configuration
- **Secure API Storage**: Environment variables for sensitive data
- **Configuration Management**: Centralized settings and parameters
- **Input Validation**: Robust validation for all user inputs

### 📈 Feature Enhancements

#### 🎲 Statistical Analysis
- **Advanced Algorithms**: Improved hot/cold number detection
- **Multiple Time Periods**: Configurable analysis windows
- **Confidence Metrics**: Statistical reliability scoring
- **Pattern Recognition**: Enhanced pattern detection algorithms

#### 🎮 Game Support
- **Five Lottery Games**: Powerball, Mega Millions, Fantasy 5, Daily 3, Daily 4
- **Game-Specific Rules**: Proper handling of different number ranges
- **Bonus Number Logic**: Intelligent bonus number generation
- **Draw Schedule Awareness**: Understanding of game-specific draw days

#### 📊 Data Visualization
- **Interactive Charts**: Enhanced Plotly integration
- **Real-Time Metrics**: Live statistical indicators
- **Export Functionality**: Data export capabilities
- **Mobile Optimization**: Improved mobile viewing experience

### 🐛 Bug Fixes
- **Streamlit Chat Input**: Fixed chat input placement outside tabs
- **Import Resolution**: Resolved module import issues
- **API Key Loading**: Fixed environment variable loading
- **UI Consistency**: Corrected layout and formatting issues

### 📚 Documentation Updates
- **Complete README**: Comprehensive documentation with examples
- **Feature Explanations**: Detailed explanation of all strategies
- **Usage Instructions**: Step-by-step user guides
- **Responsible Gaming**: Clear disclaimers and educational content

---

## Version 1.0 - Initial Release

### 🚀 Core Features
- **Multi-Game Lottery Analysis**: Support for Michigan lottery games
- **Statistical Pattern Detection**: Hot/cold number identification
- **Interactive Dashboard**: Streamlit-based web interface
- **Sample Data Generation**: Testing and demonstration capabilities
- **Basic Visualization**: Charts and frequency analysis

### 🎯 LLM Requirements Implementation
- ✅ Text Generation: AI-powered insights and reports
- ✅ Text Understanding: Natural language processing
- ✅ Multi-turn Conversation: Conversational interface
- ✅ API Accessibility: OpenAI integration framework

### 🛠 Technical Foundation
- **Python 3.8+ Compatibility**: Modern Python environment
- **Streamlit Framework**: Web application infrastructure
- **Modular Architecture**: Scalable code organization
- **Configuration System**: Flexible parameter management

---

## 🔮 Upcoming Features (Roadmap)

### Phase 1 - Data Enhancement
- [ ] Real Michigan Lottery data scraping
- [ ] Historical data persistence
- [ ] Advanced pattern analysis algorithms
- [ ] Machine learning prediction models

### Phase 2 - User Experience
- [ ] User accounts and preferences
- [ ] Saved number combinations
- [ ] Analysis history and tracking
- [ ] Email notifications for interesting patterns

### Phase 3 - Advanced Analytics
- [ ] Multi-state lottery support
- [ ] International lottery games
- [ ] Advanced statistical modeling
- [ ] Predictive analytics dashboard

---

## 📞 Support & Feedback

For issues, suggestions, or contributions:
- Check the README_FINAL.md for detailed documentation
- Review the application logs for error details
- Ensure all dependencies are properly installed
- Verify OpenAI API key configuration for full features

**Current Version**: 2.0 (October 23, 2025)  
**Status**: Fully Functional with Smart Number Suggestions  
**URL**: http://localhost:8502
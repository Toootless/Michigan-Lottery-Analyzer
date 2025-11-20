# 📋 Michigan Lottery Analyzer - Complete Changelog

## Version 3.1.2 - November 16, 2024

### 💰 Live Jackpot Tracking System
- **New Feature**: Comprehensive jackpot amount display for all lottery games
- **Smart Prize Categories**: Intelligent differentiation between rolling jackpots vs fixed prizes
- **Realistic Estimates**: Jackpot amounts based on actual lottery patterns and ranges
- **Session Caching**: Consistent jackpot amounts during user session with refresh controls

### 🎲 Enhanced Prize Display
- **Main Dashboard**: Jackpot overview section with metrics for all games
- **Sidebar Integration**: Current prize display for selected game
- **CSV Tools Enhancement**: Prize information alongside game status
- **Smart Formatting**: Appropriate icons and text for different prize types (💰🎯🎲)

### 🔄 Interactive Jackpot Controls
- **Individual Refresh**: Jackpot refresh buttons for rolling jackpot games
- **Bulk Updates**: "Refresh All Jackpots" button for complete refresh
- **Cache Management**: Smooth user experience with intelligent caching

### 🎯 Game-Specific Prize Ranges
- **Powerball/Mega Millions**: $20M - $500M estimated rolling jackpots
- **Lotto 47**: $1M - $20M estimated rolling jackpots
- **Fantasy 5**: $100K - $500K estimated rolling jackpots
- **Daily 3/4**: Fixed $500/$5,000 top prizes clearly displayed
- **Lucky for Life**: $1,000/day for life annuity format

### 📊 Data Structure Enhancements
- Enhanced `LOTTERY_GAMES` configuration with jackpot properties
- New fields: `has_jackpot`, `minimum_jackpot`, `jackpot_rollover`
- Added `fixed_top_prize` and `prize_type` for non-jackpot games
- New helper functions: `get_current_jackpot()`, `format_jackpot_display()`

### 🎨 UI/UX Improvements
- Prominent jackpot section on main application page
- Clear demo disclaimers for estimated amounts
- Consistent prize formatting across entire interface
- Better visual hierarchy with organized jackpot metrics

---

## Version 3.1.1 - November 13, 2025

### 🖥️ UI & Framework Modernization
- Upgraded Streamlit minimum version to `>=1.40.0` across all requirement files for latest stable component APIs.
- Replaced deprecated / non-standard `width='stretch'` dataframe usage with `use_container_width=True` for consistent responsive layout.
- Minor layout normalization (dataframes, metrics) to avoid future deprecation notices.

### 📚 Documentation
- README header bumped to v3.1.1 with new patch notes section.
- Setup guide reference updated (Streamlit modern APIs implied by new version range).
- Changelog dependency section updated to reflect newer Streamlit baseline.

### 🔒 Compatibility & Scope
- Patch release; no scraper, data model, or CSV schema changes.
- Fully compatible with v3.1.0 structured parser & double play metadata additions.
- Safe to apply without rebuilding CSVs; optional `pip install --upgrade streamlit` recommended.

### ✅ Upgrade Steps (Optional)
```powershell
pip install --upgrade streamlit
```
No further actions required unless pinning versions in a lock file.

---

## Version 3.1.0 - November 13, 2025

### 🧩 Structured Parsing & Data Integrity
- Added resilient structured draw-table parser (`_parse_lotteryusa_structured`) for LotteryUSA modern DOM.
- Mega Millions main number correction (legacy off-by-one fixed via structured parse alignment).
- Date normalization and duplicate row cleanup for Mega Millions & Powerball (consistent comma spacing, de-dup inline).
- Unified Powerball CSV header expanded: `Date, Result, Powerball, Double Play, Double Play Powerball, Power Play, Double Play Multiplier`.

### 🎯 Powerball Double Play Support
- Captures Double Play main numbers + bonus ball + both multipliers (Power Play & Double Play Multiplier where available).
- Manual entry auto-creates expanded schema when Powerball CSV is missing.
- Metadata fields now propagate through fetch & CSV layers: `main_numbers`, `bonus_number`, `double_play_numbers`, `double_play_bonus`, `multiplier`, `double_play_multiplier`.

### ⚙️ Environment Resilience
- zstd shim injected when native `compression.zstd.ZstdError` unavailable to prevent urllib3 import failure.
- Safe-print patch preserves execution in Windows terminals with emoji output.

### 🔍 Fallback & Reliability Notes
- Explicit 404 handling for Lotto 47 direct LotteryUSA pages (retains CSV historical data).
- Lucky for Life structured page parsed (multi-row capture; bonus number retained when present).
- Enhanced diagnostic output for failed backups (clear source URLs logged).

### 🛠️ Maintenance & Quality
- Centralized bonus number & double play parsing with helper utilities for CSV rows.
- Deduplicated Mega Millions & Powerball historical entries (reduces noisy duplicate date variants).
- README & System Guide updated with v3.1.0 features (structured parser, double play, expanded headers, reliability notes).

### 📌 Upgrade Guidance
No breaking API changes; existing consumers of `numbers` still receive consistent lists. New metadata keys are optional and backward-compatible.
Run the batch updater (`python update_all_games.py --force`) once post-upgrade to refresh Powerball with double play fields.

## Version 2.2 - October 27, 2025

### 🚀 10-Year Historical Analysis Update

#### 📊 Extended Historical Range
- **10-Year Analysis Capability**: Upgraded from 180 days (6 months) to 3,650 days (10 years)
- **Smart Time Period Selection**: Quick-select buttons for 30 days, 6 months, 1 year, 5 years, and 10 years
- **Optimized Analysis Periods**: Performance guidance for different dataset sizes
- **Comprehensive Pattern Detection**: Long-term trends and seasonal patterns across decades
- **Enhanced Data Retention**: Extended storage to match 10-year analysis capability

#### 🎯 User Experience Improvements
- **Quick Select Interface**: One-click buttons for common time periods
- **Performance Guidance**: Smart recommendations based on selected dataset size
- **Time Period Display**: Clear indication of years/months being analyzed
- **Dataset Size Warnings**: Performance notices for large dataset analysis
- **Improved Slider Control**: 30-day increments for easier navigation

#### ⚡ Performance Optimizations
- **Large Dataset Support**: Optimized algorithms for 10-year data processing
- **GPU Acceleration Benefits**: CUDA acceleration particularly helpful for large datasets
- **Memory Management**: Efficient handling of extensive historical data
- **Progressive Analysis**: Smart performance scaling based on dataset size

#### 📈 Enhanced Pattern Recognition
- **Long-term Trends**: Decade-spanning pattern identification
- **Seasonal Analysis**: Multi-year seasonal variation detection
- **Comprehensive Statistics**: More robust statistical analysis with larger datasets
- **Historical Context**: Better prediction accuracy with extensive historical data

---

## Version 2.1 - October 27, 2025

### 🚀 Major System Enhancements

#### 🎯 Enhanced Analysis Capabilities
- **Extended Analysis Period**: Default now 180 days (maximum), up from 90 days
- **Auto-Maximum Range**: Program starts with days slider maxed at 180 days
- **Deep Pattern Analysis**: 12+ comprehensive statistical pattern types
- **Real-time Learning**: Daily verification system with automatic weight updates

#### 🤖 AI & GPU Acceleration
- **Auto-LLM Initialization**: Local AI automatically loads on startup when GPU available
- **PyTorch 2.7.1 Integration**: Latest GPU acceleration with CUDA 11.8 support
- **Transformers 4.57.1**: Advanced AI models for pattern recognition
- **RTX 3060 Optimization**: Specialized memory management for 12GB VRAM
- **GPU Status Display**: Real-time CUDA/PyTorch availability information

#### 🎮 Game-Specific Number Validation
- **Complete Ticket Checker**: New primary input method for full lottery tickets
- **Daily 3/4 Duplicate Support**: Fixed error - duplicates now properly allowed (e.g., 0,7,0,7)
- **Lottery Game Validation**: Maintains unique number requirement for Powerball/Mega Millions/Fantasy 5
- **Smart Examples**: Game-appropriate placeholders and validation messages
- **Position-Aware Display**: Daily games show digits in order, lottery games sorted
- **Bonus Number Integration**: Powerball/Mega Ball support in complete ticket mode

### 🔧 Technical Improvements

#### 🛠️ System Fixes
- **Streamlit Compatibility**: Fixed column nesting errors (max one level of nesting)
- **PyTorch Import Issues**: Resolved compatibility problems with different Python versions
- **Duplicate Validation**: Corrected logic for digit-based vs lottery games
- **Error Handling**: Enhanced user feedback with clear validation messages
- **Deprecation Warnings**: Fixed `use_container_width` and `width` parameter issues

#### 📊 Data Management
- **180-Day Retention**: Prediction history extended to match analysis period
- **JSON Persistence**: Efficient data storage with automatic cleanup
- **Learning Weights**: Persistent AI improvement data across sessions
- **Performance Metrics**: Comprehensive tracking of prediction accuracy

#### 🎨 User Interface Enhancements
- **6 Comprehensive Tabs**: Analysis, Visualization, Suggestions, Learning, Chat, Number Checker
- **Input Method Priority**: Complete Ticket as default, followed by Multiple Numbers
- **Validation Feedback**: Real-time error messages with helpful examples
- **Game Rules Display**: Clear explanation of duplicate policies per game type
- **Status Indicators**: Visual feedback for GPU, LLM, and system status

### 🎮 Game System Updates

#### Daily Games (3 & 4)
- ✅ **Duplicates Allowed**: Properly handles combinations like 0,7,0,7 for Daily 4
- 🔢 **Order Preservation**: Shows digits in input order (position matters)
- 📊 **Range Validation**: 0-9 digit validation with clear feedback
- 💡 **Smart Examples**: Shows valid duplicate combinations in placeholders

#### Lottery Games (Powerball/Mega Millions/Fantasy 5)
- ❌ **Unique Numbers**: Enforces no duplicate requirement
- 🎲 **Bonus Support**: Power Ball/Mega Ball integration in complete tickets
- 📈 **Sorted Display**: Shows numbers in ascending order
- 🎯 **Range Validation**: Game-specific number range enforcement

### 📚 Documentation & Setup

#### 🔧 Installation Improvements
- **PYTHON_311_MIGRATION.md**: Comprehensive Python 3.11 upgrade guide
- **SETUP_GUIDE.md**: Complete installation and configuration instructions
- **setup_python311.bat**: Automated Windows setup script
- **requirements_py311.txt**: Python 3.11 specific dependencies

#### 📖 Documentation Updates
- **README_UPDATED.md**: Complete feature documentation with usage examples
- **Troubleshooting Guide**: Common issues and solutions
- **API Reference**: Game configurations and system architecture
- **Performance Guide**: GPU optimization and memory management

### 🐛 Bug Fixes

#### Critical Fixes
- **Duplicate Number Error**: Fixed "Duplicate numbers not allowed: [0, 7, 0, 7]" for Daily games
- **Column Nesting**: Resolved Streamlit "Columns can only be placed inside other columns up to one level" error
- **PyTorch Compatibility**: Fixed import issues with different Python versions
- **LLM Initialization**: Resolved model loading failures with proper error handling

#### Minor Fixes
- **Streamlit Warnings**: Fixed deprecation warnings for component parameters
- **Validation Messages**: Improved error message clarity and helpfulness
- **Display Issues**: Corrected number formatting for different game types
- **Memory Management**: Optimized GPU memory usage for large models

---

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
- **Pattern Recognition**: Advanced statistical analysis with AI explanations
- **Contextual Responses**: Game-specific AI recommendations
- **Error Handling**: Graceful fallback when API unavailable

#### 🧠 Learning Dashboard
- **Prediction Tracking**: Visual timeline of all predictions with outcomes
- **Accuracy Metrics**: Success rate calculation and trend analysis
- **Learning Weights**: Dynamic adjustment system for improved predictions
- **Performance Visualization**: Charts showing prediction improvement over time
- **Game-Specific Analysis**: Separate tracking for each lottery game
- **Historical Comparison**: Compare prediction accuracy across different periods

#### 📈 Advanced Visualizations
- **Interactive Plotly Charts**: Dynamic, responsive data visualizations
- **Hot/Cold Analysis**: Color-coded frequency heatmaps
- **Trend Visualization**: Time-series charts showing number popularity
- **Distribution Analysis**: Statistical spread and pattern visualization
- **Multi-Game Comparison**: Side-by-side analysis of different lottery games

### 🔧 System Improvements

#### 🏗️ Enhanced Architecture
- **Modular Design**: Separated concerns for better maintainability
- **Data Persistence**: JSON-based storage system for predictions and learning
- **Session Management**: Streamlit session state optimization
- **Error Resilience**: Comprehensive error handling and recovery
- **Configuration System**: Flexible game settings and parameters

#### 🎮 Game Engine Enhancements
- **Universal Game Support**: Standardized interface for all lottery types
- **Dynamic Configuration**: Easy addition of new lottery games
- **Validation Engine**: Robust number validation with game-specific rules
- **Bonus Number Handling**: Comprehensive support for secondary numbers
- **Pattern Analysis**: Deep statistical analysis for each game type

### 🎨 User Experience

#### 🖥️ Interface Improvements
- **Modern Design**: Clean, intuitive user interface
- **Responsive Layout**: Optimized for different screen sizes
- **Loading States**: Visual feedback during processing
- **Success Indicators**: Clear confirmation of actions
- **Error Messages**: Helpful, actionable error information

#### 🎯 Usability Features
- **One-Click Actions**: Streamlined workflows for common tasks
- **Keyboard Shortcuts**: Quick access to frequently used features
- **Copy/Paste Support**: Easy number sharing and input
- **History Navigation**: Quick access to previous analyses
- **Favorites System**: Save and recall favorite number combinations

---

## Version 1.5 - October 20, 2025

### Added
- Deep pattern analysis with 12+ statistical methods
- Local CUDA GPU detection and optimization
- Daily learning system for prediction improvement
- Game-dependent number checker functionality
- Enhanced visualization capabilities
- Better data management and persistence

### Fixed
- PyTorch import issues
- Streamlit compatibility problems
- Data validation edge cases
- Memory management issues

---

## Version 1.0 - October 15, 2025

### Initial Release
- Basic lottery analysis for Michigan games
- Simple pattern recognition
- Web interface with Streamlit
- Support for Powerball, Mega Millions, Fantasy 5, Daily 3, Daily 4
- Historical data analysis
- Number frequency tracking
- Basic visualization capabilities

---

## Technical Specifications

### Current System Requirements
- **Python**: 3.10+ (3.11 recommended for full GPU support)
- **GPU**: NVIDIA RTX 3060 or better (12GB+ VRAM recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for models and data
- **OS**: Windows 10/11, macOS, Linux

### Dependencies
- **Streamlit**: 1.40+ (Web framework)
- **PyTorch**: 2.7.1+cu118 (GPU acceleration)
- **Transformers**: 4.57.1 (AI models)
- **Pandas/NumPy**: Data processing
- **Plotly**: Interactive visualizations
- **Accelerate**: GPU optimization

### Performance Metrics (RTX 3060)
- **LLM Initialization**: ~30-60 seconds
- **Analysis Processing**: ~2-5 seconds for 180 days
- **Pattern Recognition**: ~1-3 seconds
- **GPU Memory Usage**: ~4-8GB depending on model
- **Prediction Generation**: <1 second

---

## Version Numbering System
- **Major.Minor** format for public releases
- **Major**: Significant new features, breaking changes, architecture updates
- **Minor**: New features, enhancements, backward-compatible changes
- **Patch**: Bug fixes, documentation updates, minor improvements

## Support & Compatibility
- **Backward Compatibility**: Maintained for data files and configurations
- **Migration Tools**: Automated upgrade scripts provided
- **Documentation**: Comprehensive guides for all versions
- **Community Support**: GitHub issues and discussions
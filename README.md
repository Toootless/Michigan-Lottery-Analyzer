# Michigan Lottery Results Analyzer v3.1.2

## 🎰 AI-Powered Lottery Analysis with 22-Year Historical OCR Processing

### 🌟 Project Overview
This advanced AI-powered application combines Large Language Models (LLMs), GPU acceleration, automated web scraping, and **revolutionary OCR technology** to analyze Michigan Lottery patterns with unprecedented historical depth. The system now processes 22 years of historical data (2000-2021) from PDF charts and generates intelligent predictions based on comprehensive pattern analysis.

### 🚀 New in recent versions
- v3.1.3 (Nov 19, 2025):
   - 📄 **PDF Scraper Integration**: Selenium-based PDF scraper for real-time jackpot data from Michigan Lottery website
   - 🎯 **Keyword-Based Extraction**: Searches for "Estimated Jackpot:" markers for accurate jackpot detection
   - ⏰ **24-Hour Caching**: Jackpots fetched once per day to minimize server load and improve performance
   - 🎯 **Simplified Architecture**: Removed JavaScript fallback scrapers, PDF-only approach for reliability
   - 🔧 **Dependency Management**: Added selenium 4.20.0, webdriver-manager, and PyMuPDF for PDF extraction
   - 💾 **Persistent Cache**: Session-based jackpot storage with automatic 24-hour refresh cycle
   - ✅ **Fixed Jackpot Accuracy**: Corrected extraction logic to prevent cross-game contamination
- v3.1.2 (Nov 16, 2024):
   - 💰 **Live Jackpot Tracking**: Added comprehensive jackpot amount display for all lottery games
   - 🎲 **Smart Prize Display**: Realistic jackpot estimates with rolling/fixed prize differentiation  
   - 🔄 **Refresh Controls**: Individual and bulk jackpot refresh functionality
   - 🎯 **Enhanced UI**: Prize information integrated throughout the interface
   - 📊 **Prize Analytics**: Session-based jackpot caching for consistency
- v3.1.1 (Nov 13, 2025):
   - Streamlit dependency bumped to `>=1.40.0` (aligns with latest layout & dataframe APIs)
   - Replaced legacy `width='stretch'` usages with `use_container_width=True` throughout UI for forward compatibility
   - Minor UI polish: consistent display across tabs, no nested column warnings
   - Documentation updates (README, CHANGELOG, setup guide) to reflect modern Streamlit usage
   - Non-breaking patch (pure frontend + dependency range); safe to apply alongside v3.1.0 structured parser release
- v3.1.0 (Nov 13, 2025):
   - Structured LotteryUSA draw-table parser (`_parse_lotteryusa_structured`) for resilient DOM changes
   - Full Powerball Double Play support (numbers + bonus + Power Play + Double Play Multiplier) in unified CSV schema
   - Mega Millions number correction (fixed off-by-one mismatch from legacy parser)
   - Metadata propagation for `main_numbers`, `bonus_number`, `double_play_*`, and `multiplier` in CSV + fetch responses
   - Date normalization & duplicate row cleanup for Powerball & Mega Millions (dedup inline utility)
   - Environment resilience: zstd shim to prevent import failures when `compression.zstd.ZstdError` missing
   - Manual entry & auto-creation now include expanded Powerball headers
   - Improved fallback diagnostics (explicit 404 reporting for Lotto 47; Lucky for Life structured page parsed)
   - Unified Powerball header: `Date, Result, Powerball, Double Play, Double Play Powerball, Power Play, Double Play Multiplier`
   - New upgrade guide: see `UPGRADE.md` for safe migration & verification checklist
- v3.0.5 (Nov 11, 2025):
   - Global uniqueness enforcement across ALL suggestion strategies (MLA3 + traditional + pattern) so no two cards show the same number set
   - New "CSV Update Tools" per-game tabs with an "Update CSV Now" button and current CSV status (path • exists • rows • latest date)
   - Bulk updater utility `update_all_games.py` (csv-first and `--force` modes) for fast refresh outside the UI
   - Reliability note: Lucky for Life and Lotto 47 primary sources may 404; CSV history still loads but force web updates can fail (tracked)
- v3.0.4 (Nov 10, 2025): Learning Dashboard CSV self-repair, inline fetch/manual actions, CSV diagnostics caption; unified scraper instance; automatic CSV creation on write paths.
- v3.0.3 (Nov 10, 2025): Unified CSV schemas across games, multi-game fetch validation, Windows console safe-print, structured fields for bonus games.
- v3.0.0 (Oct 28, 2025) - MAJOR RELEASE:
- **🗃️ Historical OCR Processing**: Extract 22 years of data (2000-2021) from PDF charts using advanced OCR
- **📄 PDF Chart Analysis**: Convert lottery charts and tables to structured data
- **🔍 Tesseract OCR Integration**: High-accuracy number recognition from image-based PDFs
- **🎯 Multi-Game OCR Support**: Daily 3, Daily 4, Fantasy 5, Lucky for Life, Lotto 47, Keno historical data
- **💰 Live Jackpot Tracking**: Real-time prize information display with smart refresh controls
- **📦 Batch Processing**: Process all historical PDFs with comprehensive error handling
- **💾 JSON Data Export**: Structured historical data storage and retrieval
- **🤖 Enhanced AI Analysis**: Leverage 22 years of data for superior pattern recognition
- **⚡ GPU-Accelerated OCR**: RTX 3060 optimization for faster processing
- **✅ Fantasy 5 CSV Integration**: Complete real data integration for Fantasy 5 and Fantasy 5 Double Play games (365+ records)

## 1. Functional Requirements

### Core LLM Capabilities
- **Text Generation**: Generate coherent lottery analysis reports and predictions
- **Text Understanding**: Interpret lottery data and user queries about patterns
- **Multi-turn Conversation**: Maintain context in lottery analysis discussions
- **Language Support**: American English with extensibility
- **Customizable Output**: Analysis summaries, tables, charts, and reports
- **API Accessibility**: Python API for automation and integration
- **Model Training**: Custom training on lottery datasets
- **Inference Efficiency**: Low-latency pattern analysis

### Lottery-Specific Features
- **🌐 Automated Data Collection**: Real-time web scraping from Michigan Lottery website
- **📊 Pattern Recognition**: AI-powered frequency analysis, sequences, and anomaly detection
- **📈 Statistical Analysis**: Hot/cold numbers, overdue analysis, trend forecasting
- **🤖 Predictive Modeling**: GPU-accelerated ML prediction algorithms with learning weights
- 🎯 **Multiple Games**: Support for Powerball, Mega Millions, Lotto 47, Fantasy 5, Lucky for Life, Daily 3/4
- 💰 **Live Jackpots**: Real-time jackpot tracking and prize information display
- 📊 **Pattern Analysis**: Hot/cold numbers, frequency analysis, consecutive patterns
- 🤖 **AI Predictions**: Multiple prediction strategies with accuracy tracking
- 📈 **Performance Metrics**: Strategy comparison and learning system
- 🔄 **Live Data**: Web scraping integration for real-time results
- 💾 **Data Management**: CSV-based storage with historical data processing
- 🎨 **Modern UI**: Clean Streamlit interface with responsive design
- ⚡ **GPU Acceleration**: CUDA support for enhanced AI processing (Python 3.11+)
- **� Complete CSV Data Integration**: All games now use real historical data including Fantasy 5 variants
- **�📅 Timestamp Tracking**: Full date/time logging for all winning number entries
- **📱 Interactive Dashboard**: Streamlit-based web interface with real-time updates
- **📋 Automated Reports**: AI-generated analysis reports with confidence scoring
- **💾 Data Persistence**: JSON and text file logging with historical data management

## 🚀 Quick Start Guide

### 1. Launch the Application
Preferred (VS Code Task):
```powershell
# In VS Code, run the task: Run MLA4
# Or directly:
./run_MLA4.ps1 -Address localhost -Port 8506
```

Cross-platform (Streamlit):
```bash
python -m streamlit run MLA4.py --server.address localhost --server.port 8506
```

### 2. Using Automated Data Collection
1. Select your preferred lottery game (Fantasy 5, Daily 4, Lucky for Life, etc.)
2. Click **"🌐 Fetch Today's Numbers"** to automatically collect winning numbers
3. The system will update CSVs first; when needed, it scrapes and validates from the web and logs timestamps
4. Dashboard updates automatically with new data and analysis

### 3. Manual Data Entry (Optional)
1. Expand the **"📝 Manual Entry"** section
2. Choose date and time for historical entries
3. Enter winning numbers in any supported format
4. Click **"💾 Save Winning Numbers"** to log the data
   - If a CSV for the selected game doesn’t exist yet, it will be created automatically with the correct schema

### 4. View Analysis & Predictions
- AI-powered pattern analysis with confidence scoring
- Historical trends and statistical insights
- Learning system adapts based on actual results
- Export reports in multiple formats

## 2. Non-Functional Requirements

- **Performance**: Generate analysis within 2 seconds for standard datasets
- **Scalability**: Deploy locally or on cloud infrastructure
- **Security**: Ensure data privacy and GDPR compliance
- **Reliability**: Robust error handling and data validation
- **Documentation**: Comprehensive guides and API references

## 3. Technical Requirements

- **Python Version**: 3.11+
- **Architecture**: Transformer-based LLM with statistical ML models
- **Input**: Web scraping, file upload, API endpoints
- **Output**: JSON, plain text, Markdown, CSV, interactive dashboards

## 4. Project Structure

```
Lottery_Analyzer/
├── src/
│   ├── data_collection/
│   │   ├── michigan_scraper.py
│   │   ├── data_validator.py
│   │   └── storage_manager.py
│   ├── analysis/
│   │   ├── pattern_analyzer.py
│   │   ├── statistical_models.py
│   │   ├── llm_analyzer.py
│   │   └── predictor.py
│   ├── visualization/
│   │   ├── charts.py
│   │   ├── dashboard.py
│   │   └── reports.py
│   ├── api/
│   │   ├── endpoints.py
│   │   └── models.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── tests/
├── docs/
├── config/
├── requirements.txt
├── main.py
└── README.md
```

## 5. Installation and Setup

1. **Clone/Create Project**:
   ```bash
   cd Lottery_Analyzer
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp config/.env.example config/.env
   # Edit .env with your API keys
   ```

4. **Run Application**:
   ```powershell
   ./run_MLA4.ps1 -Address localhost -Port 8520
   ```

## 6. Features

### Data Collection
- Automated Michigan Lottery website scraping
- Historical data download and storage
- Real-time lottery result updates
- Data validation and cleaning

### Pattern Analysis
- Number frequency analysis
- Hot/cold number identification
- Sequence pattern detection
- Statistical correlation analysis
- Overdue number tracking

### LLM Integration
- Natural language queries about lottery data
- Automated insight generation
- Conversational analysis interface
- Custom report generation

### Machine Learning
- Predictive modeling algorithms
- Pattern classification models
- Anomaly detection systems
- Ensemble prediction methods

### Visualization
- Interactive lottery number charts
- Frequency heatmaps
- Trend analysis graphs
- Prediction confidence intervals
- Statistical distribution plots

## 7. API Endpoints

- `GET /api/games` - List available lottery games
- `GET /api/results/{game}` - Get historical results
- `POST /api/analyze` - Analyze patterns
- `POST /api/predict` - Generate predictions
- `GET /api/stats/{game}` - Get game statistics

## 8. Usage Examples

### Basic Analysis
```python
from lottery_analyzer import LotteryAnalyzer

analyzer = LotteryAnalyzer()
results = analyzer.get_michigan_results("powerball", days=365)
patterns = analyzer.analyze_patterns(results)
predictions = analyzer.generate_predictions(results)
```

### Natural Language Interface
```python
# Ask questions about lottery data
response = analyzer.chat("What are the most frequent Powerball numbers?")
response = analyzer.chat("Show me the overdue numbers for Fantasy 5")
```

## 9. Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 10. License

Educational use - Advanced Generative AI Agentic Frameworks Course

---

**Ready to analyze lottery patterns with cutting-edge LLM technology!** 🎰🤖

---

## 📊 Learning Dashboard: Data Sources & Auto-Repair

- CSV-first history: recent results are read from unified per-game CSVs under `final_integrated_data/` (fallbacks are used when configured).
- Structured fields in history: for bonus games we expose `main_numbers`, `bonus_number`, and `bonus_label` to render Mega Ball / Powerball / Lucky Ball consistently.
- Auto-repair when empty:
   - Ensure unified CSV exists; re-read history
   - Attempt a lightweight update; re-read again
   - Show a CSV diagnostics caption (path • exists • rows • latest date) when no data is available
- Quick actions:
   - "🌐 Fetch Latest Now" to force fetch and update the CSV, then auto-refresh
   - "✍️ Add Manual Draw" to prepare adding a row manually (future enhancement hooks)
   - "CSV Update Tools" tabs (one per game) with an "Update CSV Now" button and current CSV status

### CSV Diagnostics Helpers (Developer)
- `get_csv_info(game)`: returns `{ path, exists, rows, latest_date, headers }`
- `get_unified_csv_path(game)`: resolves the path used for the game’s CSV

### Troubleshooting
- Warning: “OpenAI not available. LLM features will be limited.” → Set `OPENAI_API_KEY` to enable AI chat; not required for CSV/analysis.
- Powerball/Mega Millions not showing? Use “🌐 Fetch Latest Now” in the Learning Dashboard or the sidebar “Force Refresh from Web (Update CSV)”.
- Missing CSV files are created automatically on write (fetch or manual append). If still empty, check `logs/rebuild_summary.csv` and the CSV diagnostics caption.

## 🔄 Updating CSVs (UI and CLI)

### In-App: CSV Update Tools
- Navigate to the new "CSV Update Tools" section (per-game tabs)
- Review the CSV status for the selected game
- Click "Update CSV Now" to force a refresh from the web and update the local CSV

### Command-line: Bulk Update Helper
```powershell
# CSV-first mode (fast, reads existing CSVs; fetch only when needed)
python update_all_games.py

# Force web updates for all games
python update_all_games.py --force
```
Notes:
- Mega Millions and Powerball should update reliably
- Lucky for Life and Lotto 47 may hit transient 404s on primary sources; CSV history still loads

## ✅ Suggestion Uniqueness Guarantee (v3.0.5)

All suggestion cards now undergo a final global de-duplication pass:
- Normalizes to (sorted numbers, bonus) and removes collisions across MLA3, traditional, and pattern strategies
- Diversifies duplicates by swapping in unused hot/medium/cold numbers with safe fallbacks
- Annotates diversified sets (e.g., “globally diversified”) for transparency
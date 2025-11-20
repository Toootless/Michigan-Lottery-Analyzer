# 🎰 Michigan Lottery Results Analyzer - Enhanced Version

## 🌟 Overview

A comprehensive, AI-powered lottery analysis system with GPU acceleration, deep pattern analysis, and learning capabilities. This enhanced version features local LLM integration, extended analysis periods, and game-specific number checking.

## ✨ Key Features

### 🚀 **AI & GPU Acceleration**
- **Local LLM Integration**: Auto-initializes PyTorch-based language models
- **CUDA GPU Support**: Optimized for NVIDIA RTX 3060 (12GB VRAM)
- **PyTorch 2.7.1**: Latest GPU acceleration with CUDA 11.8
- **Transformers 4.57.1**: Hugging Face models for pattern analysis

### 📊 **Enhanced Analysis**
- **12+ Deep Pattern Types**: Hot/cold numbers, sum ranges, even/odd patterns, consecutive analysis
- **10-Year Historical Analysis**: 7 days to 10 years (3,650 days) - default: 10 years maximum
- **Quick Time Selection**: One-click buttons for 30 days, 6 months, 1 year, 5 years, 10 years
- **Statistical Learning**: Daily verification system improves predictions
- **Real-time Insights**: AI-powered pattern recognition and suggestions
- **Performance Optimization**: Smart scaling for large dataset analysis

### 🎮 **Game Support**
- **Powerball**: 5 numbers (1-69) + Power Ball (1-26)
- **Mega Millions**: 5 numbers (1-70) + Mega Ball (1-25)
- **Fantasy 5**: 5 numbers (1-39)
- **Daily 4**: 4 digits (0-9) - duplicates allowed
- **Daily 3**: 3 digits (0-9) - duplicates allowed

### 🔍 **Smart Number Checker**
- **Complete Ticket**: Check entire lottery tickets with bonus numbers
- **Game-Specific Validation**: Handles duplicates correctly for Daily games
- **Multiple Input Methods**: Single, multiple, range, or complete ticket checking
- **Position-Aware**: Daily games show digits in order, lottery games sorted

## 🏗️ System Architecture

```
Michigan Lottery Analyzer/
├── main_simple.py          # Main application (2,500+ lines)
├── data/                   # Data storage
│   ├── predictions/        # Historical predictions
│   ├── learning_weights/   # AI learning data
│   └── performance_metrics/# Analysis metrics
├── config/                 # Configuration files
├── requirements.txt        # Python dependencies
├── PYTHON_311_MIGRATION.md # Python 3.11 setup guide
├── SETUP_GUIDE.md         # Complete setup instructions
└── setup_python311.bat    # Automated setup script
```

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended for full GPU support)
- **NVIDIA GPU** with CUDA support (RTX 3060 or better)
- **8GB+ RAM** (12GB+ recommended)

### Quick Setup
1. **Clone/Download** the project files
2. **Run Setup**: `setup_python311.bat` (Windows) or follow `SETUP_GUIDE.md`
3. **Install Dependencies**:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers accelerate safetensors
   ```
4. **Launch Application**:
   ```bash
   streamlit run main_simple.py
   ```

### Manual GPU Setup
If automatic GPU detection fails:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify GPU
nvidia-smi
```

## 🎯 Application Tabs

### 1. 📈 **Analysis Tab**
- **Deep Pattern Analysis**: 12 advanced statistical patterns
- **Game Selection**: Choose from 5 lottery types
- **Period Control**: 7 days to 10 years analysis range with quick-select buttons
- **GPU Status**: Real-time CUDA/PyTorch information
- **Auto-Learning**: Daily prediction verification

### 2. 📊 **Visualization Tab**
- **Interactive Charts**: Plotly-powered visualizations
- **Frequency Analysis**: Hot/cold number patterns
- **Trend Charts**: Historical number performance
- **Statistical Summaries**: Comprehensive data insights

### 3. 🎯 **Number Suggestions Tab**
- **AI-Powered Recommendations**: LLM-generated suggestions
- **Pattern-Based Selection**: Statistical likelihood analysis
- **Multi-Game Support**: Optimized for each lottery type
- **Confidence Scoring**: Weighted recommendation system

### 4. 🧠 **Learning Dashboard Tab**
- **Prediction Tracking**: Historical accuracy monitoring
- **Learning Weights**: Adaptive algorithm improvements
- **Performance Metrics**: Success rate analysis
- **Daily Updates**: Automatic learning system status

### 5. 🤖 **Chat Tab**
- **Local AI Assistant**: PyTorch-powered conversational AI
- **Pattern Discussion**: Intelligent lottery analysis chat
- **Statistical Questions**: Ask about number patterns
- **Real-time Responses**: GPU-accelerated inference

### 6. 🔍 **Number Checker Tab**
- **Complete Ticket**: Full lottery ticket validation
- **Game-Specific Rules**: Handles duplicates correctly
- **Multiple Methods**: Single, multiple, range, or ticket input
- **Frequency Analysis**: Historical appearance tracking
- **Bonus Number Support**: Powerball/Mega Millions integration

## 🎮 Game-Specific Features

### Daily Games (3 & 4)
- ✅ **Duplicates Allowed**: `0,7,0,7` is valid for Daily 4
- 🔢 **Position Matters**: Order-dependent digit checking
- 📊 **Digit Analysis**: 0-9 frequency tracking
- 💡 **Smart Examples**: Shows valid duplicate combinations

### Lottery Games (Powerball/Mega Millions/Fantasy 5)
- ❌ **No Duplicates**: Each number must be unique
- 🎲 **Bonus Support**: Power Ball/Mega Ball integration
- 📈 **Range Analysis**: Number frequency within valid ranges
- 🎯 **Sorted Display**: Shows numbers in ascending order

## 🧠 Learning System

### Daily Verification
- **Automatic Checks**: Compares predictions vs actual results
- **Weight Updates**: Improves future predictions based on accuracy
- **Performance Tracking**: Monitors prediction success rates
- **Adaptive Learning**: Adjusts algorithms based on outcomes

### Pattern Recognition
- **Statistical Analysis**: 12+ deep pattern types
- **Trend Detection**: Identifies emerging number patterns
- **Frequency Tracking**: Historical appearance analysis
- **Predictive Modeling**: AI-enhanced number selection

## 🚀 Performance Optimizations

### GPU Acceleration
- **CUDA Integration**: PyTorch GPU processing
- **Model Optimization**: RTX 3060 memory management
- **Batch Processing**: Efficient inference operations
- **Memory Management**: Automatic GPU memory optimization

### Data Processing
- **Efficient Storage**: JSON-based data persistence
- **Fast Retrieval**: Optimized data access patterns
- **Concurrent Operations**: Multi-threaded processing
- **Cache Management**: Intelligent data caching

## 🔧 Configuration

### Default Settings
- **Analysis Period**: 10 years (3,650 days) maximum with quick-select options
- **LLM Initialization**: Automatic on startup
- **GPU Usage**: Enabled if available
- **Learning System**: Active daily verification

### Customization Options
- **Date Range**: 7 days to 10 years analysis with performance optimization
- **Model Selection**: Multiple LLM options
- **Game Types**: All 5 lottery games supported
- **Display Preferences**: Customizable visualizations

## � 10-Year Historical Analysis

### Time Period Selection
- **Quick Select Buttons**: One-click selection for common periods
  - 📅 **30 Days**: Fast analysis for recent trends
  - 📅 **6 Months**: Balanced performance and pattern depth
  - 📅 **1 Year**: Comprehensive seasonal analysis
  - 📅 **5 Years**: Long-term trend identification
  - 📅 **10 Years**: Maximum historical context

### Performance Guidance
- **🚀 Quick Analysis** (weeks/months): Fast processing, recent patterns
- **🎯 Optimal Range** (1-5 years): Best balance of performance and insights
- **⚡ Large Dataset** (5+ years): Comprehensive patterns, benefits from GPU acceleration

### Enhanced Pattern Recognition
- **Long-term Trends**: Decade-spanning pattern identification
- **Seasonal Analysis**: Multi-year seasonal variation detection
- **Comprehensive Statistics**: More robust analysis with larger datasets
- **Historical Context**: Better prediction accuracy with extensive data

## �📊 Data Management

### Storage Structure
```
data/
├── predictions_history.json    # Historical predictions
├── learning_weights.json      # AI learning parameters
└── performance_metrics.json   # System performance data
```

### Data Retention
- **Predictions**: 10 years retention with automated cleanup
- **Learning Data**: Persistent across sessions
- **Performance Metrics**: Continuous tracking
- **Analysis Results**: Session-based storage

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Solution 1: Check CUDA installation
nvidia-smi

# Solution 2: Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### LLM Initialization Fails
```bash
# Solution: Use CPU-only mode or upgrade to Python 3.11
pip install torch torchvision torchaudio
```

#### Streamlit Errors
```bash
# Solution: Update Streamlit
pip install --upgrade streamlit
```

### Performance Issues
- **Memory**: Ensure 8GB+ RAM available
- **GPU Memory**: RTX 3060 12GB recommended
- **Python Version**: Use Python 3.11 for best compatibility

## 🔄 Version History

### v2.0 (Current)
- ✅ Extended analysis period to 10 years (3,650 days) with quick-select interface
- ✅ Auto-LLM initialization
- ✅ Game-specific number validation
- ✅ Complete ticket checking
- ✅ Duplicate handling for Daily games
- ✅ Enhanced GPU acceleration
- ✅ Improved learning system

### v1.0 (Initial)
- Basic lottery analysis
- 30-90 day analysis range
- Manual LLM initialization
- Simple number checking

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests and validation
5. Submit pull request

### Code Structure
- **Main App**: `main_simple.py` (modular design)
- **Configuration**: Game settings and parameters
- **Data Layer**: JSON-based persistence
- **AI Integration**: PyTorch/Transformers integration

## 📄 License

This project is for educational and research purposes. Lottery participation should be done responsibly.

## 🎯 Future Enhancements

### Planned Features
- [ ] Real lottery API integration
- [ ] Advanced ML models
- [ ] Mobile-responsive design
- [ ] Multi-state lottery support
- [ ] Export/import functionality

### Optimization Goals
- [ ] Faster model loading
- [ ] Enhanced GPU utilization
- [ ] Real-time data streaming
- [ ] Advanced visualization options

---

## 🚀 Quick Start Commands

```bash
# Launch application
streamlit run main_simple.py

# Check GPU status
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify installation
python -c "import streamlit, torch, transformers; print('All dependencies OK')"
```

**Access the application at**: http://localhost:8501

---

*Built with ❤️ using Python, Streamlit, PyTorch, and CUDA acceleration*
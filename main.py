"""
Michigan Lottery Results Analyzer - Main Application
LLM-powered lottery pattern analysis and prediction system
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import custom modules
from src.utils.config import load_config
from src.data_collection.michigan_scraper import MichiganLotteryScraper
from src.analysis.llm_analyzer import LotteryLLMAnalyzer
from src.visualization.dashboard import LotteryDashboard

# Configure Streamlit page
st.set_page_config(
    page_title="🎰 Michigan Lottery Analyzer",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    scraper = MichiganLotteryScraper(config)
    llm_analyzer = LotteryLLMAnalyzer(config.OPENAI_API_KEY)
    dashboard = LotteryDashboard()
    
    # App header
    st.title("🎰 Michigan Lottery Results Analyzer")
    st.markdown("*LLM-powered pattern analysis and prediction system*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Selection")
        
        # Available games
        games = [
            "Powerball",
            "Mega Millions", 
            "Fantasy 5",
            "Daily 3",
            "Daily 4",
            "Keno",
            "Lucky for Life"
        ]
        
        selected_game = st.selectbox("Choose a lottery game", games)
        
        # Analysis parameters
        st.header("📊 Analysis Settings")
        analysis_days = st.slider("Historical days to analyze", 30, 1095, 365)
        prediction_mode = st.selectbox(
            "Prediction Method",
            ["Statistical", "LLM-based", "Hybrid", "Ensemble"]
        )
        
        # Data refresh
        st.header("🔄 Data Management")
        if st.button("🔄 Refresh Data"):
            with st.spinner("Fetching latest lottery results..."):
                try:
                    results = scraper.get_recent_results(selected_game, days=analysis_days)
                    st.success(f"✅ Loaded {len(results)} results for {selected_game}")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
        
        # API status
        st.header("🔧 System Status")
        st.text(f"🤖 LLM: {'Connected' if llm_analyzer.is_ready() else 'Disconnected'}")
        st.text(f"🌐 Scraper: {'Ready' if scraper.is_ready() else 'Error'}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🎯 {selected_game} Analysis")
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Patterns", "🤖 LLM Insights", "🔮 Predictions", "📊 Statistics"])
        
        with tab1:
            st.subheader("Number Pattern Analysis")
            # TODO: Implement pattern visualization
            st.info("📊 Pattern analysis charts will be displayed here")
            
        with tab2:
            st.subheader("AI-Powered Insights")
            
            # Chat interface
            st.markdown("### 💬 Ask about lottery patterns")
            user_query = st.text_input(
                "Ask a question about lottery data:",
                placeholder="What are the most frequent numbers in Powerball?"
            )
            
            if user_query:
                with st.spinner("🤖 Analyzing with LLM..."):
                    try:
                        response = llm_analyzer.analyze_query(user_query, selected_game)
                        st.write("**🎯 Analysis Result:**")
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
            
            # Pre-generated insights
            st.markdown("### 🧠 Automatic Insights")
            st.info("🔄 LLM-generated insights will appear here")
            
        with tab3:
            st.subheader("Number Predictions")
            
            if st.button("🎲 Generate Predictions"):
                with st.spinner("🔮 Generating predictions..."):
                    # TODO: Implement prediction logic
                    st.info("🎯 Predictions will be displayed here")
            
        with tab4:
            st.subheader("Statistical Summary")
            # TODO: Implement statistics display
            st.info("📊 Statistical summaries will be shown here")
    
    with col2:
        st.header("🎲 Quick Stats")
        
        # Sample metrics (placeholder)
        st.metric("🔥 Hot Numbers", "7, 21, 35", "+2 this week")
        st.metric("❄️ Cold Numbers", "13, 42, 58", "-5 this week")
        st.metric("⏰ Overdue Numbers", "3, 17, 49", "45+ days")
        
        # Recent results placeholder
        st.subheader("📅 Recent Results")
        st.info("Latest lottery results will be displayed here")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("📊 Full Report"):
            st.info("Generating comprehensive report...")
        
        if st.button("📤 Export Data"):
            st.info("Preparing data export...")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🎰 **Michigan Lottery Analyzer** | "
        "Powered by LLMs and Advanced Analytics | "
        "Educational Research Project"
    )

if __name__ == "__main__":
    main()
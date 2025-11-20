"""
Interactive Dashboard for Lottery Analysis Visualization
Creates charts, graphs, and interactive elements for data presentation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter
import logging

# Simple logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LotteryDashboard:
    """Interactive dashboard for lottery analysis visualization"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'hot': '#ff4444',
            'cold': '#4444ff'
        }
        
    def create_frequency_chart(self, results: List[Any], title: str = "Number Frequency Analysis") -> go.Figure:
        """Create a frequency chart showing how often each number appears"""
        try:
            # Extract all numbers from results
            all_numbers = []
            for result in results:
                if hasattr(result, 'numbers') and result.numbers:
                    all_numbers.extend(result.numbers)
            
            if not all_numbers:
                return self._create_empty_chart("No data available")
            
            # Count frequencies
            number_freq = Counter(all_numbers)
            numbers = sorted(number_freq.keys())
            frequencies = [number_freq[num] for num in numbers]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=numbers,
                    y=frequencies,
                    marker_color=self.colors['primary'],
                    hovertemplate='Number: %{x}<br>Frequency: %{y}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Numbers",
                yaxis_title="Frequency",
                showlegend=False,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating frequency chart: {e}")
            return self._create_empty_chart("Error creating chart")
    
    def create_hot_cold_chart(self, hot_numbers: List[int], cold_numbers: List[int]) -> go.Figure:
        """Create a visualization of hot and cold numbers"""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Hot Numbers', 'Cold Numbers'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Hot numbers
            if hot_numbers:
                fig.add_trace(
                    go.Bar(
                        x=hot_numbers[:10],  # Top 10
                        y=[1] * len(hot_numbers[:10]),
                        marker_color=self.colors['hot'],
                        name="Hot Numbers",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Cold numbers
            if cold_numbers:
                fig.add_trace(
                    go.Bar(
                        x=cold_numbers[:10],  # Top 10
                        y=[1] * len(cold_numbers[:10]),
                        marker_color=self.colors['cold'],
                        name="Cold Numbers",
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="Hot vs Cold Numbers Analysis",
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Numbers", row=1, col=1)
            fig.update_xaxes(title_text="Numbers", row=1, col=2)
            fig.update_yaxes(title_text="Relative Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Relative Frequency", row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating hot/cold chart: {e}")
            return self._create_empty_chart("Error creating hot/cold chart")
    
    def create_timeline_chart(self, results: List[Any], game: str) -> go.Figure:
        """Create a timeline chart of lottery draws"""
        try:
            if not results:
                return self._create_empty_chart("No timeline data available")
            
            dates = []
            jackpots = []
            
            for result in results:
                if hasattr(result, 'draw_date') and hasattr(result, 'jackpot_amount'):
                    dates.append(result.draw_date)
                    jackpots.append(result.jackpot_amount or 0)
            
            if not dates:
                return self._create_empty_chart("No timeline data available")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=jackpots,
                mode='lines+markers',
                name=f'{game} Jackpots',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Jackpot: $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{game} Jackpot Timeline",
                xaxis_title="Date",
                yaxis_title="Jackpot Amount ($)",
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating timeline chart: {e}")
            return self._create_empty_chart("Error creating timeline chart")
    
    def create_pattern_analysis_chart(self, patterns: Dict[str, Any]) -> go.Figure:
        """Create visualization of number patterns"""
        try:
            if not patterns:
                return self._create_empty_chart("No pattern data available")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Consecutive Numbers', 'Sum Distribution', 'Odd/Even Distribution', 'Range Analysis'),
                specs=[
                    [{"type": "bar"}, {"type": "histogram"}],
                    [{"type": "pie"}, {"type": "bar"}]
                ]
            )
            
            # Consecutive numbers pattern
            consecutive_data = patterns.get('consecutive_numbers', {})
            if consecutive_data and 'consecutive_frequency' in consecutive_data:
                freq_data = consecutive_data['consecutive_frequency']
                fig.add_trace(
                    go.Bar(
                        x=list(freq_data.keys()),
                        y=list(freq_data.values()),
                        marker_color=self.colors['primary'],
                        name="Consecutive"
                    ),
                    row=1, col=1
                )
            
            # Odd/Even pattern
            odd_even_data = patterns.get('odd_even_patterns', {})
            if odd_even_data:
                avg_odd = odd_even_data.get('average_odd', 0)
                avg_even = odd_even_data.get('average_even', 0)
                
                fig.add_trace(
                    go.Pie(
                        labels=['Odd', 'Even'],
                        values=[avg_odd, avg_even],
                        marker_colors=[self.colors['hot'], self.colors['cold']],
                        name="Odd/Even"
                    ),
                    row=2, col=1
                )
            
            # Range analysis
            range_data = patterns.get('range_patterns', {})
            if range_data:
                range_values = ['Min Range', 'Avg Range', 'Max Range']
                range_amounts = [
                    range_data.get('min_range', 0),
                    range_data.get('average_range', 0),
                    range_data.get('max_range', 0)
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=range_values,
                        y=range_amounts,
                        marker_color=self.colors['secondary'],
                        name="Range"
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Pattern Analysis Dashboard",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern chart: {e}")
            return self._create_empty_chart("Error creating pattern chart")
    
    def create_prediction_chart(self, predictions: Dict[str, Any]) -> go.Figure:
        """Create visualization for predictions"""
        try:
            if not predictions:
                return self._create_empty_chart("No prediction data available")
            
            underdue = predictions.get('underdue_numbers', [])
            balanced = predictions.get('balanced_numbers', [])
            
            if not underdue and not balanced:
                return self._create_empty_chart("No prediction numbers available")
            
            fig = go.Figure()
            
            if underdue:
                fig.add_trace(go.Bar(
                    x=underdue[:10],
                    y=[1] * len(underdue[:10]),
                    name='Underdue Numbers',
                    marker_color=self.colors['warning'],
                    yaxis='y1'
                ))
            
            if balanced:
                fig.add_trace(go.Bar(
                    x=balanced[:10],
                    y=[0.5] * len(balanced[:10]),
                    name='Balanced Numbers',
                    marker_color=self.colors['info'],
                    yaxis='y1'
                ))
            
            fig.update_layout(
                title="Statistical Number Analysis",
                xaxis_title="Numbers",
                yaxis_title="Relative Weight",
                height=400,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction chart: {e}")
            return self._create_empty_chart("Error creating prediction chart")
    
    def display_metrics_cards(self, analysis_results: Any):
        """Display key metrics in card format"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Confidence Score",
                    f"{analysis_results.confidence_score:.1%}",
                    delta=None
                )
            
            with col2:
                hot_count = len(analysis_results.hot_numbers)
                st.metric(
                    "Hot Numbers",
                    hot_count,
                    delta=None
                )
            
            with col3:
                cold_count = len(analysis_results.cold_numbers)
                st.metric(
                    "Cold Numbers", 
                    cold_count,
                    delta=None
                )
            
            with col4:
                insight_count = len(analysis_results.insights)
                st.metric(
                    "Insights Generated",
                    insight_count,
                    delta=None
                )
                
        except Exception as e:
            logger.error(f"Error displaying metrics: {e}")
            st.error("Error displaying metrics")
    
    def display_insights_section(self, insights: List[str]):
        """Display insights in an organized section"""
        try:
            if not insights:
                st.info("No insights available for this analysis.")
                return
            
            st.subheader("📊 Key Insights")
            
            for i, insight in enumerate(insights, 1):
                with st.container():
                    st.markdown(f"**{i}.** {insight}")
                    
        except Exception as e:
            logger.error(f"Error displaying insights: {e}")
            st.error("Error displaying insights")
    
    def display_number_grids(self, hot_numbers: List[int], cold_numbers: List[int], game_range: tuple = (1, 69)):
        """Display numbers in an interactive grid format"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Hot Numbers")
                if hot_numbers:
                    # Create a grid display for hot numbers
                    hot_display = []
                    for i in range(0, len(hot_numbers), 5):
                        row = hot_numbers[i:i+5]
                        hot_display.append(row)
                    
                    for row in hot_display:
                        cols = st.columns(5)
                        for j, num in enumerate(row):
                            with cols[j]:
                                st.button(f"{num}", key=f"hot_{num}", help="Hot number")
                else:
                    st.info("No hot numbers identified")
            
            with col2:
                st.subheader("🧊 Cold Numbers")
                if cold_numbers:
                    # Create a grid display for cold numbers
                    cold_display = []
                    for i in range(0, len(cold_numbers), 5):
                        row = cold_numbers[i:i+5]
                        cold_display.append(row)
                    
                    for row in cold_display:
                        cols = st.columns(5)
                        for j, num in enumerate(row):
                            with cols[j]:
                                st.button(f"{num}", key=f"cold_{num}", help="Cold number")
                else:
                    st.info("No cold numbers identified")
                    
        except Exception as e:
            logger.error(f"Error displaying number grids: {e}")
            st.error("Error displaying number grids")
    
    def create_summary_table(self, results: List[Any]) -> pd.DataFrame:
        """Create a summary table of recent results"""
        try:
            if not results:
                return pd.DataFrame()
            
            data = []
            for result in results[-10:]:  # Last 10 results
                if hasattr(result, 'draw_date') and hasattr(result, 'numbers'):
                    row = {
                        'Date': result.draw_date.strftime('%Y-%m-%d') if result.draw_date else 'N/A',
                        'Numbers': ', '.join(map(str, result.numbers)) if result.numbers else 'N/A',
                        'Bonus': ', '.join(map(str, result.bonus_numbers)) if hasattr(result, 'bonus_numbers') and result.bonus_numbers else '',
                        'Jackpot': f"${result.jackpot_amount:,.0f}" if hasattr(result, 'jackpot_amount') and result.jackpot_amount else ''
                    }
                    data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error creating summary table: {e}")
            return pd.DataFrame()
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=400
        )
        return fig
    
    def export_analysis_report(self, analysis_results: Any, results: List[Any]) -> Dict[str, Any]:
        """Export analysis results as a structured report"""
        try:
            report = {
                'analysis_date': datetime.now().isoformat(),
                'game': analysis_results.game,
                'confidence_score': analysis_results.confidence_score,
                'summary': {
                    'total_draws_analyzed': len(results),
                    'hot_numbers': analysis_results.hot_numbers,
                    'cold_numbers': analysis_results.cold_numbers,
                    'key_insights': analysis_results.insights
                },
                'patterns': analysis_results.patterns,
                'predictions': analysis_results.predictions
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return {}
    
    def create_comparison_chart(self, multi_game_data: Dict[str, List[Any]]) -> go.Figure:
        """Create comparison chart across multiple games"""
        try:
            if not multi_game_data:
                return self._create_empty_chart("No comparison data available")
            
            fig = go.Figure()
            
            for game, results in multi_game_data.items():
                if not results:
                    continue
                    
                dates = []
                jackpots = []
                
                for result in results:
                    if hasattr(result, 'draw_date') and hasattr(result, 'jackpot_amount'):
                        dates.append(result.draw_date)
                        jackpots.append(result.jackpot_amount or 0)
                
                if dates and jackpots:
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=jackpots,
                        mode='lines+markers',
                        name=game.title(),
                        hovertemplate=f'{game}<br>Date: %{{x}}<br>Jackpot: $%{{y:,.0f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title="Multi-Game Jackpot Comparison",
                xaxis_title="Date",
                yaxis_title="Jackpot Amount ($)",
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return self._create_empty_chart("Error creating comparison chart")
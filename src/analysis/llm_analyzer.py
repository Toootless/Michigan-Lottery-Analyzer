"""
LLM-powered Lottery Analysis Module
Provides AI-powered analysis and insights for lottery data
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from collections import Counter, defaultdict
import statistics

# Simple logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. LLM features will be limited.")

@dataclass
class LotteryAnalysis:
    """Container for lottery analysis results"""
    game: str
    analysis_date: datetime
    hot_numbers: List[int]
    cold_numbers: List[int]
    patterns: Dict[str, Any]
    predictions: Dict[str, Any]
    insights: List[str]
    confidence_score: float
    
class LotteryLLMAnalyzer:
    """AI-powered lottery analysis using large language models"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                openai.api_key = self.api_key
                self.client = openai
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OpenAI client not available - using statistical analysis only")
        
        # Analysis parameters
        self.hot_number_threshold = 0.7  # Numbers appearing more than 70% of average
        self.cold_number_threshold = 0.3  # Numbers appearing less than 30% of average
        
    def is_ready(self) -> bool:
        """Check if the analyzer is ready for LLM operations"""
        return self.client is not None
    
    def analyze_lottery_data(self, lottery_results: List[Any], game: str) -> LotteryAnalysis:
        """
        Comprehensive analysis of lottery data
        
        Args:
            lottery_results: List of LotteryResult objects
            game: Name of the lottery game
            
        Returns:
            LotteryAnalysis object with insights
        """
        logger.info(f"Analyzing {len(lottery_results)} results for {game}")
        
        # Perform statistical analysis
        stats = self._calculate_statistics(lottery_results, game)
        
        # Identify hot and cold numbers
        hot_numbers, cold_numbers = self._identify_hot_cold_numbers(lottery_results, game)
        
        # Find patterns
        patterns = self._find_patterns(lottery_results, game)
        
        # Generate predictions
        predictions = self._generate_predictions(lottery_results, game, stats)
        
        # Get LLM insights if available
        insights = self._get_llm_insights(stats, patterns, game) if self.is_ready() else []
        
        # Add statistical insights
        insights.extend(self._get_statistical_insights(stats, hot_numbers, cold_numbers))
        
        # Calculate confidence score
        confidence = self._calculate_confidence(lottery_results, patterns)
        
        return LotteryAnalysis(
            game=game,
            analysis_date=datetime.now(),
            hot_numbers=hot_numbers,
            cold_numbers=cold_numbers,
            patterns=patterns,
            predictions=predictions,
            insights=insights,
            confidence_score=confidence
        )
    
    def _calculate_statistics(self, results: List[Any], game: str) -> Dict[str, Any]:
        """Calculate statistical metrics from lottery results"""
        if not results:
            return {}
        
        all_numbers = []
        draw_dates = []
        jackpots = []
        
        for result in results:
            if hasattr(result, 'numbers') and result.numbers:
                all_numbers.extend(result.numbers)
            if hasattr(result, 'draw_date'):
                draw_dates.append(result.draw_date)
            if hasattr(result, 'jackpot_amount') and result.jackpot_amount:
                jackpots.append(result.jackpot_amount)
        
        if not all_numbers:
            return {}
        
        number_frequency = Counter(all_numbers)
        
        stats = {
            'total_draws': len(results),
            'date_range': {
                'start': min(draw_dates).strftime('%Y-%m-%d') if draw_dates else None,
                'end': max(draw_dates).strftime('%Y-%m-%d') if draw_dates else None
            },
            'number_frequency': dict(number_frequency),
            'most_common_numbers': number_frequency.most_common(10),
            'least_common_numbers': number_frequency.most_common()[:-11:-1],
            'average_frequency': statistics.mean(number_frequency.values()) if number_frequency else 0,
            'frequency_std_dev': statistics.stdev(number_frequency.values()) if len(number_frequency) > 1 else 0
        }
        
        if jackpots:
            stats['jackpot_stats'] = {
                'average': statistics.mean(jackpots),
                'median': statistics.median(jackpots),
                'max': max(jackpots),
                'min': min(jackpots)
            }
        
        return stats
    
    def _identify_hot_cold_numbers(self, results: List[Any], game: str) -> Tuple[List[int], List[int]]:
        """Identify hot and cold numbers based on frequency"""
        if not results:
            return [], []
        
        all_numbers = []
        for result in results:
            if hasattr(result, 'numbers') and result.numbers:
                all_numbers.extend(result.numbers)
        
        if not all_numbers:
            return [], []
        
        number_frequency = Counter(all_numbers)
        average_frequency = statistics.mean(number_frequency.values())
        
        hot_numbers = [
            num for num, freq in number_frequency.items()
            if freq >= average_frequency * self.hot_number_threshold
        ]
        
        cold_numbers = [
            num for num, freq in number_frequency.items()
            if freq <= average_frequency * self.cold_number_threshold
        ]
        
        return sorted(hot_numbers), sorted(cold_numbers)
    
    def _find_patterns(self, results: List[Any], game: str) -> Dict[str, Any]:
        """Find patterns in lottery results"""
        patterns = {
            'consecutive_numbers': self._find_consecutive_patterns(results),
            'sum_patterns': self._analyze_sum_patterns(results),
            'odd_even_patterns': self._analyze_odd_even_patterns(results),
            'range_patterns': self._analyze_range_patterns(results)
        }
        
        return patterns
    
    def _find_consecutive_patterns(self, results: List[Any]) -> Dict[str, Any]:
        """Find consecutive number patterns"""
        consecutive_counts = []
        
        for result in results:
            if not hasattr(result, 'numbers') or not result.numbers:
                continue
                
            numbers = sorted(result.numbers)
            consecutive = 0
            max_consecutive = 0
            
            for i in range(1, len(numbers)):
                if numbers[i] == numbers[i-1] + 1:
                    consecutive += 1
                else:
                    max_consecutive = max(max_consecutive, consecutive + 1)
                    consecutive = 0
            
            max_consecutive = max(max_consecutive, consecutive + 1)
            consecutive_counts.append(max_consecutive)
        
        return {
            'average_consecutive': statistics.mean(consecutive_counts) if consecutive_counts else 0,
            'max_consecutive': max(consecutive_counts) if consecutive_counts else 0,
            'consecutive_frequency': Counter(consecutive_counts)
        }
    
    def _analyze_sum_patterns(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze sum patterns of drawn numbers"""
        sums = []
        
        for result in results:
            if hasattr(result, 'numbers') and result.numbers:
                sums.append(sum(result.numbers))
        
        if not sums:
            return {}
        
        return {
            'average_sum': statistics.mean(sums),
            'median_sum': statistics.median(sums),
            'min_sum': min(sums),
            'max_sum': max(sums),
            'std_dev': statistics.stdev(sums) if len(sums) > 1 else 0
        }
    
    def _analyze_odd_even_patterns(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze odd/even number patterns"""
        odd_counts = []
        even_counts = []
        
        for result in results:
            if not hasattr(result, 'numbers') or not result.numbers:
                continue
                
            odd_count = sum(1 for num in result.numbers if num % 2 == 1)
            even_count = len(result.numbers) - odd_count
            
            odd_counts.append(odd_count)
            even_counts.append(even_count)
        
        return {
            'average_odd': statistics.mean(odd_counts) if odd_counts else 0,
            'average_even': statistics.mean(even_counts) if even_counts else 0,
            'odd_distribution': Counter(odd_counts),
            'even_distribution': Counter(even_counts)
        }
    
    def _analyze_range_patterns(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze number range patterns"""
        ranges = []
        
        for result in results:
            if hasattr(result, 'numbers') and result.numbers:
                number_range = max(result.numbers) - min(result.numbers)
                ranges.append(number_range)
        
        if not ranges:
            return {}
        
        return {
            'average_range': statistics.mean(ranges),
            'median_range': statistics.median(ranges),
            'min_range': min(ranges),
            'max_range': max(ranges)
        }
    
    def _generate_predictions(self, results: List[Any], game: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical predictions"""
        if not results or not stats:
            return {}
        
        # Simple prediction based on frequency analysis
        number_frequency = stats.get('number_frequency', {})
        
        if not number_frequency:
            return {}
        
        # Predict numbers based on inverse frequency (less frequent numbers might be "due")
        sorted_by_freq = sorted(number_frequency.items(), key=lambda x: x[1])
        
        predictions = {
            'underdue_numbers': [num for num, freq in sorted_by_freq[:10]],
            'balanced_numbers': [num for num, freq in sorted_by_freq[len(sorted_by_freq)//3:2*len(sorted_by_freq)//3]],
            'prediction_method': 'frequency_analysis',
            'confidence_level': 'low'  # Statistical predictions have inherently low confidence
        }
        
        return predictions
    
    def _get_llm_insights(self, stats: Dict[str, Any], patterns: Dict[str, Any], game: str) -> List[str]:
        """Generate insights using LLM"""
        if not self.client:
            return []
        
        try:
            prompt = self._build_analysis_prompt(stats, patterns, game)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a statistical analyst specializing in lottery data analysis. Provide factual, statistical insights without making gambling recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            insights_text = response.choices[0].message.content
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip()]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            return []
    
    def _build_analysis_prompt(self, stats: Dict[str, Any], patterns: Dict[str, Any], game: str) -> str:
        """Build prompt for LLM analysis"""
        prompt = f"""
        Analyze the following lottery statistics for {game}:
        
        Statistics:
        - Total draws analyzed: {stats.get('total_draws', 0)}
        - Date range: {stats.get('date_range', {})}
        - Average number frequency: {stats.get('average_frequency', 0):.2f}
        - Most common numbers: {stats.get('most_common_numbers', [])}
        
        Patterns:
        - Average consecutive numbers: {patterns.get('consecutive_numbers', {}).get('average_consecutive', 0):.2f}
        - Average sum: {patterns.get('sum_patterns', {}).get('average_sum', 0):.2f}
        - Average odd numbers: {patterns.get('odd_even_patterns', {}).get('average_odd', 0):.2f}
        
        Provide 3-5 statistical insights about these patterns. Focus on factual observations without making predictions or gambling advice.
        """
        
        return prompt
    
    def _get_statistical_insights(self, stats: Dict[str, Any], hot_numbers: List[int], cold_numbers: List[int]) -> List[str]:
        """Generate statistical insights without LLM"""
        insights = []
        
        if stats.get('total_draws', 0) > 0:
            insights.append(f"Analysis based on {stats['total_draws']} lottery draws")
        
        if hot_numbers:
            insights.append(f"Hot numbers (appearing frequently): {', '.join(map(str, hot_numbers[:5]))}")
        
        if cold_numbers:
            insights.append(f"Cold numbers (appearing less frequently): {', '.join(map(str, cold_numbers[:5]))}")
        
        avg_freq = stats.get('average_frequency', 0)
        if avg_freq > 0:
            insights.append(f"Average number appearance frequency: {avg_freq:.2f}")
        
        # Add jackpot insights if available
        jackpot_stats = stats.get('jackpot_stats')
        if jackpot_stats:
            avg_jackpot = jackpot_stats.get('average', 0)
            max_jackpot = jackpot_stats.get('max', 0)
            insights.append(f"Average jackpot: ${avg_jackpot:,.0f}, Maximum: ${max_jackpot:,.0f}")
        
        return insights
    
    def _calculate_confidence(self, results: List[Any], patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis (0-1 scale)"""
        if not results:
            return 0.0
        
        # Base confidence on data quantity
        data_quantity_score = min(len(results) / 100, 1.0)  # Full confidence at 100+ draws
        
        # Factor in pattern consistency (simplified)
        pattern_score = 0.5  # Default moderate confidence for patterns
        
        # Overall confidence is average of factors
        confidence = (data_quantity_score + pattern_score) / 2
        
        return round(confidence, 2)
    
    def chat_about_results(self, question: str, analysis: LotteryAnalysis) -> str:
        """
        Answer questions about lottery analysis results
        
        Args:
            question: User question about the analysis
            analysis: LotteryAnalysis object
            
        Returns:
            Response string
        """
        if not self.is_ready():
            return self._answer_question_statistically(question, analysis)
        
        try:
            context = self._build_context_from_analysis(analysis)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about lottery statistics. Provide factual information without giving gambling advice or making predictions about future outcomes."},
                    {"role": "assistant", "content": f"I have analyzed lottery data for {analysis.game}. Here's what I found: {context}"},
                    {"role": "user", "content": question}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            return self._answer_question_statistically(question, analysis)
    
    def _build_context_from_analysis(self, analysis: LotteryAnalysis) -> str:
        """Build context string from analysis results"""
        context_parts = [
            f"Game: {analysis.game}",
            f"Hot numbers: {', '.join(map(str, analysis.hot_numbers[:5]))}",
            f"Cold numbers: {', '.join(map(str, analysis.cold_numbers[:5]))}",
            f"Key insights: {'; '.join(analysis.insights[:3])}"
        ]
        
        return " | ".join(context_parts)
    
    def _answer_question_statistically(self, question: str, analysis: LotteryAnalysis) -> str:
        """Answer questions using statistical analysis only"""
        question_lower = question.lower()
        
        if 'hot' in question_lower or 'frequent' in question_lower:
            return f"The most frequent numbers in {analysis.game} are: {', '.join(map(str, analysis.hot_numbers[:10]))}"
        
        elif 'cold' in question_lower or 'rare' in question_lower:
            return f"The least frequent numbers in {analysis.game} are: {', '.join(map(str, analysis.cold_numbers[:10]))}"
        
        elif 'pattern' in question_lower:
            patterns = analysis.patterns
            return f"Key patterns found: {list(patterns.keys())}"
        
        elif 'confidence' in question_lower:
            return f"Analysis confidence score: {analysis.confidence_score:.1%}"
        
        else:
            return f"Based on the analysis of {analysis.game}, here are the key insights: {' '.join(analysis.insights[:2])}"
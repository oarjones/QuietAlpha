"""
LSTM Enhanced Portfolio Manager

This module extends the base PortfolioManager with LSTM prediction capabilities.
It uses the LSTM integration to enhance symbol selection, prioritize training,
and incorporate model predictions into the portfolio decision process.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import datetime
import time
from threading import Lock

from portfolio_manager.base import PortfolioManager
from portfolio_manager.lstm_integration import PortfolioLSTMIntegration
from utils.data_utils import load_config

logger = logging.getLogger(__name__)

class LSTMEnhancedPortfolioManager(PortfolioManager):
    """
    Enhanced Portfolio Manager that incorporates LSTM predictions into the 
    portfolio selection and optimization process.
    """
    
    def __init__(self, config_path: str = None, ibkr_interface = None):
        """
        Initialize LSTM Enhanced Portfolio Manager.
        
        Args:
            config_path (str): Path to configuration file
            ibkr_interface: IBKR interface instance
        """
        # Initialize base portfolio manager
        super().__init__(config_path, ibkr_interface)
        
        # Load LSTM-specific configuration
        self.lstm_config = self.portfolio_config.get('lstm', {})
        
        # Initialize LSTM integration
        self.lstm_integration = PortfolioLSTMIntegration(ibkr_interface=ibkr_interface)
        
        # LSTM weighting in portfolio decisions
        self.lstm_weight = self.lstm_config.get('weight', 0.3)
        
        # Maximum models to schedule for training per update
        self.max_training_models = self.lstm_config.get('max_training_models', 5)
        
        # Cache for LSTM analysis to avoid redundant processing
        self.lstm_analysis_cache = {}
        
        logger.info("LSTM Enhanced Portfolio Manager initialized")
    
    def scan_market(self) -> List[str]:
        """
        Scan the market with LSTM enhancement.
        
        This overrides the base method to incorporate LSTM insights into the
        initial market scanning process.
        
        Returns:
            List[str]: List of symbols to analyze
        """
        # Get base symbols from parent implementation
        base_symbols = super().scan_market()
        
        # If no symbols were found, return empty list
        if not base_symbols:
            return []
        
        # Limit to a reasonable number for LSTM analysis
        # (to avoid overloading the system with too many requests)
        max_initial_symbols = self.lstm_config.get('max_initial_symbols', 50)
        analysis_symbols = base_symbols[:max_initial_symbols]
        
        # Schedule LSTM model training for these symbols
        # This is done in the background and won't affect current selection
        self.lstm_integration.create_training_schedule(
            analysis_symbols, 
            {s: 50 for s in analysis_symbols},  # Default equal scores
            max_models=self.max_training_models
        )
        
        # Return the original list - we're just scheduling training here
        # The actual LSTM analysis will happen in analyze_symbol
        return base_symbols
    
    def fetch_symbol_data(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        This overrides the base method to ensure the data fetched is also suitable
        for LSTM analysis.
        
        Args:
            symbol (str): Symbol to fetch data for
            lookback_days (int): Number of days of history to fetch
            
        Returns:
            pd.DataFrame: Historical data with technical indicators
        """
        # Use base implementation to fetch data
        df = super().fetch_symbol_data(symbol, lookback_days)
        
        # Ensure we have enough data for LSTM (at least 60 periods)
        if not df.empty and len(df) < 60:
            # If we don't have enough data, try to fetch more
            logger.info(f"Not enough data for LSTM analysis of {symbol}, fetching more...")
            df = super().fetch_symbol_data(symbol, lookback_days * 2)
        
        return df
    
    def analyze_symbol(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Analyze a symbol with LSTM enhancement.
        
        This overrides the base method to incorporate LSTM predictions into the
        symbol analysis process.
        
        Args:
            symbol (str): Symbol to analyze
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            
        Returns:
            Dict: Enhanced analysis results
        """
        # Use base implementation for initial analysis
        base_analysis = super().analyze_symbol(symbol, data)
        
        # If base analysis failed or has insufficient data, return as is
        if base_analysis.get('recommendation') == 'avoid' and base_analysis.get('reason') == 'insufficient_data':
            return base_analysis
        
        try:
            # Get LSTM prediction for this symbol
            # Use the same data that was used for base analysis if available
            if data is None and 'data' in base_analysis:
                data = base_analysis['data']
            
            # Get LSTM analysis
            lstm_result = self.lstm_integration.get_lstm_prediction(symbol, data)
            
            # If LSTM prediction successful, integrate it
            if lstm_result.get('status') == 'success':
                # Extract useful information from LSTM prediction
                direction = lstm_result.get('predicted_direction', 'neutral')
                confidence = lstm_result.get('confidence', 0)
                price_change_pct = lstm_result.get('price_change_pct', 0)
                
                # Convert direction and confidence to a score adjustment
                lstm_score_adj = 0
                if direction == 'up':
                    lstm_score_adj = confidence * 30  # Up to +30 points for high confidence up
                elif direction == 'down':
                    lstm_score_adj = -confidence * 30  # Up to -30 points for high confidence down
                
                # Adjust base score with LSTM insights
                base_score = base_analysis.get('score', 50)
                adjusted_score = max(0, min(100, base_score + lstm_score_adj))
                
                # Update recommendation based on new score
                recommendation = 'neutral'
                if adjusted_score >= 75:
                    recommendation = 'strong_buy'
                elif adjusted_score >= 60:
                    recommendation = 'buy'
                elif adjusted_score <= 25:
                    recommendation = 'avoid'
                elif adjusted_score <= 40:
                    recommendation = 'reduce'
                
                # Include LSTM insights in the analysis
                enhanced_analysis = base_analysis.copy()
                enhanced_analysis.update({
                    'score': adjusted_score,
                    'recommendation': recommendation,
                    'lstm': {
                        'direction': direction,
                        'confidence': confidence,
                        'price_change_pct': price_change_pct,
                        'model_type': lstm_result.get('model_type', 'unknown'),
                        'contribution': lstm_score_adj
                    }
                })
                
                # Add LSTM as a reason if it was significant
                if abs(lstm_score_adj) > 10:
                    reasons = enhanced_analysis.get('reasons', [])
                    if direction == 'up' and lstm_score_adj > 0:
                        reasons.append(f'lstm_bullish_{confidence:.2f}')
                    elif direction == 'down' and lstm_score_adj < 0:
                        reasons.append(f'lstm_bearish_{confidence:.2f}')
                    enhanced_analysis['reasons'] = reasons
                
                return enhanced_analysis
            else:
                # If LSTM prediction failed, use base analysis but log warning
                logger.warning(f"LSTM prediction failed for {symbol}: {lstm_result.get('message', 'Unknown error')}")
                return base_analysis
                
        except Exception as e:
            logger.error(f"Error in LSTM-enhanced analysis for {symbol}: {e}")
            # Fall back to base analysis
            return base_analysis
    
    def select_portfolio(self, num_symbols: int = None) -> Dict[str, float]:
        """
        Select portfolio with LSTM enhancement.
        
        This overrides the base method to incorporate LSTM predictions into the
        portfolio selection process, potentially improving the quality of selected
        symbols.
        
        Args:
            num_symbols (int, optional): Number of symbols to select
            
        Returns:
            Dict[str, float]: Symbol -> allocation mapping
        """
        if num_symbols is None:
            num_symbols = self.max_symbols
        
        logger.info(f"Selecting LSTM-enhanced portfolio with up to {num_symbols} symbols")
        
        # Get candidate symbols - use twice as many as needed to allow for filtering
        candidate_symbols = self.scan_market()[:num_symbols * 2]
        
        if not candidate_symbols:
            logger.warning("No candidate symbols found")
            return {}
        
        # Analyze each symbol with both traditional and LSTM methods
        symbol_analyses = {}
        lstm_analysis_results = {}
        
        # First, fetch data for all symbols to avoid redundant fetches
        symbol_data = {}
        for symbol in candidate_symbols:
            data = self.fetch_symbol_data(symbol)
            if not data.empty:
                symbol_data[symbol] = data
        
        # Get LSTM analysis for all symbols
        lstm_batch_analysis = self.lstm_integration.analyze_symbols_with_lstm(
            [s for s in candidate_symbols if s in symbol_data],
            symbol_data
        )
        
        # Analyze each symbol incorporating LSTM insights
        for symbol in candidate_symbols:
            if symbol in symbol_data:
                analysis = self.analyze_symbol(symbol, symbol_data[symbol])
                symbol_analyses[symbol] = analysis
                
                # Also save LSTM analysis separately
                if symbol in lstm_batch_analysis:
                    lstm_analysis_results[symbol] = lstm_batch_analysis[symbol]
            else:
                logger.warning(f"No data available for {symbol}, skipping")
        
        # Filter symbols with positive recommendation
        positive_symbols = {
            s: a for s, a in symbol_analyses.items() 
            if a.get('recommendation') in ['buy', 'strong_buy']
        }
        
        # If not enough positive symbols, include neutral ones
        if len(positive_symbols) < num_symbols:
            neutral_symbols = {
                s: a for s, a in symbol_analyses.items()
                if a.get('recommendation') == 'neutral'
                and s not in positive_symbols
            }
            
            additional_needed = num_symbols - len(positive_symbols)
            if neutral_symbols:
                neutral_sorted = sorted(
                    neutral_symbols.items(),
                    key=lambda x: x[1].get('score', 0),
                    reverse=True
                )
                
                for s, a in neutral_sorted[:additional_needed]:
                    positive_symbols[s] = a
        
        # If still not enough, use all we have
        selected_analyses = positive_symbols if positive_symbols else symbol_analyses
        
        # Limit to requested number and sort by score
        selected_symbols = dict(
            sorted(selected_analyses.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:num_symbols]
        )
        
        if not selected_symbols:
            logger.warning("No suitable symbols found for portfolio")
            return {}
        
        # Calculate score-weighted allocations
        total_score = sum(a.get('score', 0) for a in selected_symbols.values())
        
        if total_score == 0:
            # Equal weight if all scores are 0
            allocation_per_symbol = 1.0 / len(selected_symbols)
            portfolio = {symbol: allocation_per_symbol for symbol in selected_symbols}
        else:
            # Weighted allocation based on scores
            portfolio = {
                symbol: analysis.get('score', 0) / total_score 
                for symbol, analysis in selected_symbols.items()
            }
        
        # Schedule training for the selected symbols (high priority)
        self.lstm_integration.request_model_training(
            list(portfolio.keys()),
            {s: 30 for s in portfolio.keys()}  # Higher priority (30 instead of default 50)
        )
        
        logger.info(f"Selected {len(portfolio)} symbols for LSTM-enhanced portfolio")
        return portfolio
    
    def update_portfolio(self) -> Dict:
        """
        Update portfolio with LSTM enhancement.
        
        This overrides the base method to incorporate LSTM predictions into the
        portfolio update process and ensure LSTM models are available for selected
        symbols.
        
        Returns:
            Dict: Update results
        """
        with self.portfolio_lock:
            try:
                # Use base implementation for core update logic
                update_result = super().update_portfolio()
                
                if update_result.get('status') != 'success':
                    return update_result
                
                # Get updated portfolio plan
                update_plan = update_result.get('update_plan', {})
                allocations = update_plan.get('allocations', {})
                
                # Nothing to enhance if no allocations
                if not allocations:
                    return update_result
                
                # Schedule training for all symbols in the new portfolio (high priority)
                training_result = self.lstm_integration.create_training_schedule(
                    list(allocations.keys()),
                    {s: a * 100 for s, a in allocations.items()},  # Convert allocations to scores
                    max_models=self.max_training_models
                )
                
                # Add LSTM training info to the result
                update_result['lstm_training'] = training_result
                
                return update_result
                
            except Exception as e:
                logger.error(f"Error updating portfolio with LSTM enhancement: {e}")
                return {
                    'status': 'error',
                    'message': str(e)
                }
    
    def get_portfolio_stats(self) -> Dict:
        """
        Get portfolio statistics with LSTM insights.
        
        This overrides the base method to include LSTM prediction information
        in the portfolio statistics.
        
        Returns:
            Dict: Enhanced portfolio statistics
        """
        # Get base statistics
        base_stats = super().get_portfolio_stats()
        
        # If no active portfolio, return base stats
        if not self.current_portfolio:
            return base_stats
        
        try:
            # Get LSTM predictions for current portfolio symbols
            lstm_predictions = {}
            
            for symbol in self.current_portfolio.keys():
                prediction = self.lstm_integration.get_lstm_prediction(symbol)
                if prediction.get('status') == 'success':
                    lstm_predictions[symbol] = {
                        'direction': prediction.get('predicted_direction', 'unknown'),
                        'confidence': prediction.get('confidence', 0),
                        'price_change_pct': prediction.get('price_change_pct', 0),
                        'model_type': prediction.get('model_type', 'unknown')
                    }
            
            # Calculate portfolio-level LSTM outlook
            if lstm_predictions:
                # Weighted average of predicted price changes
                weighted_change = 0
                total_weight = 0
                
                for symbol, pred in lstm_predictions.items():
                    allocation = self.current_portfolio.get(symbol, 0)
                    confidence = pred.get('confidence', 0)
                    price_change = pred.get('price_change_pct', 0)
                    
                    # Weight by both allocation and confidence
                    weight = allocation * confidence
                    weighted_change += price_change * weight
                    total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    portfolio_outlook = weighted_change / total_weight
                else:
                    portfolio_outlook = 0
                
                # Add LSTM insights to stats
                base_stats['lstm_insights'] = {
                    'predictions': lstm_predictions,
                    'portfolio_outlook': portfolio_outlook,
                    'symbols_with_predictions': len(lstm_predictions),
                    'timestamp': datetime.datetime.now().isoformat()
                }
            
            return base_stats
            
        except Exception as e:
            logger.error(f"Error getting LSTM-enhanced portfolio stats: {e}")
            # Return base stats if there's an error
            return base_stats
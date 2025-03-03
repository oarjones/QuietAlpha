"""
LSTM Integration for Portfolio Manager

This module provides integration between the Portfolio Manager and the LSTM prediction system.
It allows the Portfolio Manager to:
1. Request model training for symbols it's considering
2. Use LSTM predictions as part of the symbol selection process
3. Prioritize training based on portfolio needs

It acts as a bridge between the portfolio selection logic and the LSTM prediction capabilities.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

# Import LSTM service and utilities
from lstm.integration import get_lstm_service, predict_with_lstm, request_model_training, get_model_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioLSTMIntegration:
    """
    Integration between Portfolio Manager and LSTM prediction system.
    
    This class manages the interaction between portfolio selection and LSTM models,
    providing methods to request training, get predictions, and incorporate LSTM
    insights into the portfolio decision process.
    """
    
    def __init__(self, ibkr_interface=None):
        """
        Initialize the integration.
        
        Args:
            ibkr_interface: IBKR interface for data fetching (optional)
        """
        self.ibkr = ibkr_interface
        self.lstm_service = get_lstm_service(ibkr_interface=ibkr_interface)
        self.training_requested = set()  # Symbols that have been requested for training
        self.prediction_cache = {}  # Cache for recent predictions to avoid redundant API calls
        self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        
        logger.info("Portfolio LSTM Integration initialized")
    
    def request_model_training(self, symbols: List[str], priority_override: Dict[str, int] = None) -> Dict:
        """
        Request training for multiple symbols with customized priorities.
        
        Args:
            symbols: List of symbols to train
            priority_override: Dictionary mapping symbols to custom priorities
            
        Returns:
            dict: Training request results
        """
        if not symbols:
            return {'status': 'error', 'message': 'No symbols provided'}
        
        # Initialize results
        results = {
            'requested': 0,
            'already_training': 0,
            'details': {}
        }
        
        # Calculate default priorities based on symbol positions in the list
        # Earlier symbols in the list get higher priority (lower number)
        base_priority = 50
        default_priorities = {}
        for i, symbol in enumerate(symbols):
            # Adjust priority by position - earlier symbols get higher priority
            # But ensure it stays within 0-100 range
            position_adjustment = min(30, 5 * i)  # Max adjustment of 30 points
            default_priorities[symbol] = base_priority + position_adjustment
        
        # Process each symbol
        for symbol in symbols:
            # Skip if already requested in this session (avoid duplicate requests)
            if symbol in self.training_requested:
                results['already_training'] += 1
                results['details'][symbol] = 'already_requested'
                continue
            
            # Get model status to check if training is needed
            status = get_model_status(symbol)
            
            # If already training or scheduled, skip
            if status.get('training_status') in ['training', 'scheduled']:
                results['already_training'] += 1
                results['details'][symbol] = 'already_in_progress'
                continue
            
            # Determine priority - use override if provided, otherwise use default
            priority = priority_override.get(symbol, default_priorities[symbol]) if priority_override else default_priorities[symbol]
            
            # Check if model exists and is fresh - if so, use lower priority
            if status.get('status') == 'available' and status.get('is_fresh', False):
                # Lower priority for fresh models (they're less urgent)
                priority = min(100, priority + 20)
                
                # If reliability is good, may not need training at all
                if status.get('reliability', 0) > 0.7:
                    results['details'][symbol] = 'skipped_reliable_model'
                    continue
            
            # Special case: if not available and not in training, this is high priority
            if status.get('status') == 'not_available':
                # Higher priority (lower number) for missing models
                priority = max(10, priority - 20)
            
            # Request training with determined priority
            success = request_model_training(symbol, priority)
            
            if success:
                results['requested'] += 1
                results['details'][symbol] = f'requested_priority_{priority}'
                self.training_requested.add(symbol)
            else:
                results['details'][symbol] = 'request_failed'
        
        logger.info(f"Requested training for {results['requested']} symbols, {results['already_training']} were already training")
        return results
    
    def get_lstm_prediction(self, symbol: str, data: pd.DataFrame = None, force_refresh: bool = False) -> Dict:
        """
        Get LSTM prediction for a symbol with caching.
        
        Args:
            symbol: Symbol to predict for
            data: Optional price data (will fetch if not provided)
            force_refresh: Whether to force refresh the prediction
            
        Returns:
            dict: Prediction result
        """
        current_time = datetime.now().timestamp()
        
        # Check cache if not forcing refresh
        if not force_refresh and symbol in self.prediction_cache:
            cached_prediction, timestamp = self.prediction_cache[symbol]
            
            # If cache is still valid (not expired)
            if current_time - timestamp < self.cache_expiry:
                return cached_prediction
        
        # Get fresh prediction
        prediction = predict_with_lstm(symbol, data, self.ibkr)
        
        # Cache the prediction
        if prediction.get('status') == 'success':
            self.prediction_cache[symbol] = (prediction, current_time)
        
        return prediction
    
    def analyze_symbols_with_lstm(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame] = None) -> Dict[str, Dict]:
        """
        Analyze multiple symbols with LSTM and return enhanced analysis.
        
        Args:
            symbols: List of symbols to analyze
            data_dict: Optional dictionary of pre-loaded data frames
            
        Returns:
            dict: Symbol -> Analysis results mapping
        """
        results = {}
        missing_models = []
        
        # Process each symbol
        for symbol in symbols:
            # Get data if not provided
            data = None
            if data_dict and symbol in data_dict:
                data = data_dict[symbol]
            
            # Get prediction
            prediction = self.get_lstm_prediction(symbol, data)
            
            # If prediction failed
            if prediction.get('status') != 'success':
                results[symbol] = {
                    'status': 'error',
                    'lstm_available': False,
                    'message': prediction.get('message', 'Unknown error')
                }
                missing_models.append(symbol)
                continue
            
            # Extract useful information
            direction = prediction.get('predicted_direction', 'unknown')
            confidence = prediction.get('confidence', 0)
            price_change_pct = prediction.get('price_change_pct', 0)
            model_type = prediction.get('model_type', 'unknown')
            reliability = prediction.get('reliability_index', 0)
            
            # Normalize score to 0-100 range for consistency with other metrics
            if direction == 'up':
                lstm_score = 50 + (confidence * 50)  # 50-100 for bullish
            elif direction == 'down':
                lstm_score = 50 - (confidence * 50)  # 0-50 for bearish
            else:
                lstm_score = 50  # Neutral
            
            # Prepare result
            results[symbol] = {
                'status': 'success',
                'lstm_available': True,
                'lstm_score': lstm_score,
                'lstm_direction': direction,
                'lstm_confidence': confidence,
                'lstm_price_change_pct': price_change_pct,
                'lstm_model_type': model_type,
                'lstm_reliability': reliability
            }
            
            # If using universal model, might want to train a specific model
            if model_type == 'universal':
                missing_models.append(symbol)
        
        # Request training for missing or universal models
        if missing_models:
            # Only for symbols we haven't already requested
            new_requests = [s for s in missing_models if s not in self.training_requested]
            if new_requests:
                self.request_model_training(new_requests)
        
        return results
    
    def get_priority_ranking(self, symbols: List[str], scores: Dict[str, float]) -> List[str]:
        """
        Rank symbols for training priority based on scores and model status.
        
        Args:
            symbols: List of symbols to rank
            scores: Dictionary of symbol scores from portfolio analysis
            
        Returns:
            list: Symbols ordered by training priority
        """
        if not symbols:
            return []
        
        # Get model status for all symbols
        model_status = {}
        for symbol in symbols:
            status = get_model_status(symbol)
            model_status[symbol] = status
        
        # Calculate a priority score for each symbol
        priority_scores = {}
        for symbol in symbols:
            base_score = scores.get(symbol, 50)  # Default score if not provided
            
            # Get model status information
            status = model_status.get(symbol, {})
            
            # Start with base score from portfolio analysis
            priority = base_score
            
            # Adjust based on model status
            if status.get('status') == 'not_available':
                # Higher priority (lower final rank) for missing models
                priority += 30
            elif status.get('status') == 'outdated':
                # Higher priority for outdated models
                priority += 20
            elif status.get('status') == 'available':
                # Adjust based on reliability
                reliability = status.get('reliability', 0)
                if reliability < 0.5:
                    # Higher priority for unreliable models
                    priority += 15
                else:
                    # Lower priority for reliable models
                    priority -= reliability * 20
            
            # Adjust for symbols already in training/scheduled
            if status.get('training_status') in ['training', 'scheduled']:
                # Much lower priority for models already in training
                priority -= 40
            
            priority_scores[symbol] = priority
        
        # Sort symbols by priority score (descending)
        ranked_symbols = sorted(symbols, key=lambda s: priority_scores.get(s, 0), reverse=True)
        
        return ranked_symbols
    
    def create_training_schedule(self, symbols: List[str], scores: Dict[str, float], 
                               max_models: int = 5) -> Dict:
        """
        Create an optimal model training schedule based on portfolio needs.
        
        Args:
            symbols: List of symbols to consider
            scores: Dictionary of symbol scores from portfolio analysis
            max_models: Maximum models to schedule for training
            
        Returns:
            dict: Training schedule result
        """
        # Get prioritized list of symbols
        ranked_symbols = self.get_priority_ranking(symbols, scores)
        
        # Take top N symbols for training
        training_candidates = ranked_symbols[:max_models]
        
        # Calculate priorities based on rank
        priorities = {}
        for i, symbol in enumerate(training_candidates):
            # Higher positions get higher priority (lower priority number)
            priorities[symbol] = max(10, 50 - (40 * (len(training_candidates) - i) / len(training_candidates)))
        
        # Request training with calculated priorities
        if training_candidates:
            result = self.request_model_training(training_candidates, priorities)
            result['training_candidates'] = training_candidates
            result['priorities'] = priorities
        else:
            result = {
                'status': 'info',
                'message': 'No training candidates identified',
                'training_candidates': [],
                'priorities': {}
            }
        
        return result
    
    def integrate_lstm_scores(self, base_scores: Dict[str, float], lstm_analysis: Dict[str, Dict], 
                            lstm_weight: float = 0.3) -> Dict[str, float]:
        """
        Integrate LSTM prediction scores with base portfolio scores.
        
        Args:
            base_scores: Dictionary of base scores from portfolio analysis
            lstm_analysis: Dictionary of LSTM analysis results
            lstm_weight: Weight to give to LSTM scores (0-1)
            
        Returns:
            dict: Integrated scores
        """
        integrated_scores = {}
        
        for symbol, base_score in base_scores.items():
            # Get LSTM analysis
            lstm_result = lstm_analysis.get(symbol, {})
            
            if lstm_result.get('lstm_available', False):
                # Get LSTM score
                lstm_score = lstm_result.get('lstm_score', 50)
                
                # Adjust weight by reliability
                reliability = lstm_result.get('lstm_reliability', 0.5)
                adjusted_weight = lstm_weight * reliability
                
                # Ensure weight is not extreme
                adjusted_weight = max(0.1, min(0.5, adjusted_weight))
                
                # Integrate scores
                integrated_score = (base_score * (1 - adjusted_weight)) + (lstm_score * adjusted_weight)
            else:
                # If no LSTM data, use base score
                integrated_score = base_score
            
            integrated_scores[symbol] = integrated_score
        
        return integrated_scores
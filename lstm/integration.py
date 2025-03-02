"""
LSTM Integration Module

This module provides utilities to integrate the LSTMTrainer with the rest of the system,
allowing other components like TradingManager to request model training and access
trained models transparently.

It includes:
- A singleton LSTMService for central access to models
- Helper functions to get predictions using the appropriate model
- Integration with the Portfolio Manager for requesting model training
"""

import os
import logging
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import project modules
from lstm.trainer import LSTMTrainer
from lstm.model_manager import LSTMModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMService:
    """
    Singleton service for central access to LSTM models and training functionality.
    
    This service is designed to be accessed by other components in the system,
    providing them with a unified interface to interact with LSTM models.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LSTMService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: str = None, ibkr_interface = None):
        """Initialize the LSTM service."""
        with self._lock:
            if self._initialized:
                return
                
            # Initialize trainer and model manager
            self.trainer = LSTMTrainer(config_path, ibkr_interface)
            self.model_manager = LSTMModelManager()
            
            # Start training queue in background
            self.trainer.start()
            
            # Flag as initialized
            self._initialized = True
            logger.info("LSTM Service initialized")
    
    def get_model_for_symbol(self, symbol: str) -> tuple:
        """
        Get the appropriate model for a symbol.
        
        This will first try to get a symbol-specific model, and if not available,
        will fall back to the universal model.
        
        Args:
            symbol: Symbol to get model for
            
        Returns:
            tuple: (model, scaler, metadata)
        """
        # First try to get symbol-specific model
        if self.model_manager.model_exists(symbol) and self.model_manager.model_is_fresh(symbol):
            model, scaler = self.model_manager.load_model_and_scaler(symbol)
            metadata = self.model_manager.get_model_metadata(symbol)
            logger.info(f"Using symbol-specific model for {symbol}")
            return model, scaler, metadata
        
        # Fall back to universal model
        logger.info(f"No suitable symbol-specific model for {symbol}, using universal model")
        model, scaler = self.model_manager.load_universal_model_and_scaler()
        
        # Get universal metadata
        universal_path = os.path.join('models', 'lstm', 'universal', 'metadata.json')
        metadata = {}
        if os.path.exists(universal_path):
            try:
                import json
                with open(universal_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading universal model metadata: {e}")
        
        return model, scaler, metadata
    
    def request_model_training(self, symbol: str, priority: int = 50) -> bool:
        """
        Request training for a model.
        
        Args:
            symbol: Symbol to train model for
            priority: Training priority (0-100, lower is higher priority)
            
        Returns:
            bool: True if training was requested, False otherwise
        """
        return self.trainer.request_training(symbol, priority)
    
    def request_batch_training(self, symbols: List[str], auto_prioritize: bool = True) -> Dict:
        """
        Request training for a batch of symbols.
        
        Args:
            symbols: List of symbols to train
            auto_prioritize: Whether to automatically prioritize symbols
            
        Returns:
            dict: Batch training status
        """
        return self.trainer.run_parallel_batch_training(symbols, max_models=None)
    
    def get_model_status(self, symbol: str = None) -> Dict:
        """
        Get status of one or all models.
        
        Args:
            symbol: Symbol to get status for, or None for all
            
        Returns:
            dict: Model status
        """
        if symbol:
            if symbol.upper() == "UNIVERSAL":
                # Get universal model status
                universal_path = os.path.join('models', 'lstm', 'universal', 'metadata.json')
                if os.path.exists(universal_path):
                    try:
                        import json
                        with open(universal_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Calculate days old
                        trained_date = datetime.fromisoformat(metadata.get('trained_date', '2000-01-01T00:00:00'))
                        days_old = (datetime.now() - trained_date).days
                        
                        return {
                            'status': 'available',
                            'is_fresh': days_old <= 30,
                            'days_old': days_old,
                            'reliability': metadata.get('reliability_index', 0),
                            'metrics': metadata.get('metrics', {})
                        }
                    except Exception as e:
                        logger.error(f"Error getting universal model status: {e}")
                
                return {'status': 'not_available'}
            else:
                # Get specific model status
                if self.model_manager.model_exists(symbol):
                    is_fresh = self.model_manager.model_is_fresh(symbol)
                    metadata = self.model_manager.get_model_metadata(symbol)
                    
                    # Get training status
                    training_status = self.trainer.get_training_status(symbol)
                    
                    return {
                        'status': 'available' if is_fresh else 'outdated',
                        'is_fresh': is_fresh,
                        'reliability': metadata.get('reliability_index', 0),
                        'metrics': metadata.get('metrics', {}),
                        'training_status': training_status.get('status', 'unknown')
                    }
                else:
                    # Check if it's in training queue
                    training_status = self.trainer.get_training_status(symbol)
                    if training_status and training_status.get('status') in ['training', 'scheduled']:
                        return {
                            'status': 'training',
                            'training_status': training_status
                        }
                    
                    return {'status': 'not_available'}
        else:
            # Get all models status
            models_status = self.trainer.get_trained_models_status()
            
            # Get universal model status
            universal_status = self.get_model_status("UNIVERSAL")
            
            # Combine
            combined_status = {
                'models': models_status,
                'universal': universal_status,
                'queue_status': self.trainer.get_queue_status()
            }
            
            return combined_status
    
    def predict_price(self, symbol: str, data: pd.DataFrame = None, ibkr_interface = None) -> Dict:
        """
        Predict future price using the appropriate model.
        
        This method abstracts away the model selection logic, automatically choosing 
        between symbol-specific and universal models based on availability and freshness.
        
        Args:
            symbol: Symbol to predict for
            data: Price data (optional, will fetch if None)
            ibkr_interface: IBKR interface (optional, only needed if data is None)
            
        Returns:
            dict: Prediction result
        """
        # If no data provided, we need to fetch it
        if data is None and ibkr_interface is not None:
            from lstm.model import fetch_historical_data
            data = fetch_historical_data(ibkr_interface, symbol, lookback_days=60)
            
            if data.empty:
                return {
                    'status': 'error',
                    'message': f'No data available for {symbol}'
                }
        elif data is None:
            return {
                'status': 'error',
                'message': 'Either data or ibkr_interface must be provided'
            }
        
        # Process data if needed
        from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators
        
        if 'RSI_14' not in data.columns:
            data = calculate_technical_indicators(data, include_all=True)
            data = calculate_trend_indicator(data)
            data = normalize_indicators(data)
        
        # Get model
        model, scaler, metadata = self.get_model_for_symbol(symbol)
        
        if model is None:
            return {
                'status': 'error',
                'message': 'No suitable model available'
            }
        
        # Get input shape from model
        input_shape = model.input_shape
        seq_length = input_shape[1]
        num_features = input_shape[2]
        
        # Extract normalized features
        feature_cols = [col for col in data.columns if col.endswith('_norm')]
        
        # Add close_norm if not already included
        if 'close_norm' not in feature_cols and 'close_norm' in data.columns:
            feature_cols.append('close_norm')
        
        # Check if we have enough data
        if len(data) < seq_length:
            return {
                'status': 'error',
                'message': f'Not enough data: need at least {seq_length} periods'
            }
        
        # Handle feature count mismatch
        if len(feature_cols) != num_features:
            logger.warning(f"Feature count mismatch: model expects {num_features}, got {len(feature_cols)}")
            
            if len(feature_cols) < num_features:
                # Add placeholder features
                for i in range(len(feature_cols), num_features):
                    placeholder_name = f"placeholder_{i}_norm"
                    data[placeholder_name] = 0.5
                    feature_cols.append(placeholder_name)
            
            # Select only the first num_features if we have too many
            feature_cols = feature_cols[:num_features]
        
        # Extract sequence
        sequence = data[feature_cols].values[-seq_length:]
        
        # Reshape for prediction
        X = np.array([sequence])
        
        # Make prediction
        prediction = model.predict(X, verbose=0)[0][0]
        
        # Get current price information
        current_close_norm = data['close_norm'].iloc[-1]
        current_close = data['close'].iloc[-1]
        
        # Determine direction and confidence
        if prediction > current_close_norm:
            direction = 'up'
            confidence = min(1.0, (prediction - current_close_norm) * 10)
        elif prediction < current_close_norm:
            direction = 'down'
            confidence = min(1.0, (current_close_norm - prediction) * 10)
        else:
            direction = 'neutral'
            confidence = 0.0
        
        # Calculate predicted price using volatility-aware approach
        atr = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else current_close * 0.01
        price_volatility = atr / current_close
        
        volatility_factor = 5
        expected_pct_change = confidence * price_volatility * volatility_factor
        
        # Limit to reasonable change
        max_pct_change = 0.02
        expected_pct_change = min(expected_pct_change, max_pct_change)
        
        # Calculate predicted price
        if direction == 'up':
            predicted_price = current_close * (1 + expected_pct_change)
        elif direction == 'down':
            predicted_price = current_close * (1 - expected_pct_change)
        else:
            predicted_price = current_close
        
        # Prepare result
        result = {
            'status': 'success',
            'symbol': symbol,
            'predicted_direction': direction,
            'predicted_value_norm': float(prediction),
            'current_value_norm': float(current_close_norm),
            'confidence': float(confidence),
            'current_price': float(current_close),
            'predicted_price': float(predicted_price),
            'price_change': float(predicted_price - current_close),
            'price_change_pct': float((predicted_price / current_close - 1) * 100),
            'model_type': 'symbol_specific' if self.model_manager.model_exists(symbol) else 'universal',
            'reliability_index': metadata.get('reliability_index', 0.5),
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# Helper functions for easy access to LSTM service

def get_lstm_service(config_path: str = None, ibkr_interface = None) -> LSTMService:
    """
    Get the LSTM service singleton instance.
    
    Args:
        config_path: Configuration path
        ibkr_interface: IBKR interface
        
    Returns:
        LSTMService: LSTM service instance
    """
    return LSTMService(config_path, ibkr_interface)

def predict_with_lstm(symbol: str, data: pd.DataFrame = None, ibkr_interface = None) -> Dict:
    """
    Make a price prediction using the appropriate LSTM model.
    
    This is a convenience function that abstracts away the service instantiation.
    
    Args:
        symbol: Symbol to predict for
        data: Price data (optional, will fetch if None)
        ibkr_interface: IBKR interface (optional, only needed if data is None)
        
    Returns:
        dict: Prediction result
    """
    service = get_lstm_service(ibkr_interface=ibkr_interface)
    return service.predict_price(symbol, data, ibkr_interface)

def request_model_training(symbol: str, priority: int = 50) -> bool:
    """
    Request training for a model.
    
    Args:
        symbol: Symbol to train model for
        priority: Training priority (0-100, lower is higher priority)
        
    Returns:
        bool: True if training was requested, False otherwise
    """
    service = get_lstm_service()
    return service.request_model_training(symbol, priority)

def get_model_status(symbol: str = None) -> Dict:
    """
    Get status of one or all models.
    
    Args:
        symbol: Symbol to get status for, or None for all
        
    Returns:
        dict: Model status
    """
    service = get_lstm_service()
    return service.get_model_status(symbol)
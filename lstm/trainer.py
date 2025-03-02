"""
LSTM Training System

This module implements the Phase 2 of the LSTM training system, including:
- Symbol prioritization for training
- Incremental training for existing models
- Parallel training for multiple symbols
- Integration with LSTMModelManager

The system automatically decides whether to train models from scratch or update existing ones,
and efficiently manages computational resources through parallel processing.
"""

import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ke
import joblib
import concurrent.futures
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
import queue

# Project imports
from lstm.model_manager import LSTMModelManager
from lstm.model import (
    fetch_historical_data, 
    preprocess_data, 
    build_lstm_model, 
    train_model, 
    evaluate_model
)
from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators
from utils.data_utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMTrainer:
    """
    System for training LSTM models with priority queue, incremental training and parallelization.
    """
    
    def __init__(self, config_path: str = None, ibkr_interface: IBKRInterface = None):
        """
        Initialize the LSTM Trainer.
        
        Args:
            config_path: Path to configuration file
            ibkr_interface: Interface to Interactive Brokers (optional)
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else {}
        self.trainer_config = self.config.get('lstm_trainer', {})
        
        # Setup IBKR interface
        self.ibkr = ibkr_interface
        if self.ibkr is None:
            host = self.config.get('ibkr', {}).get('host', '127.0.0.1')
            port = self.config.get('ibkr', {}).get('port', 7497)
            client_id = self.config.get('ibkr', {}).get('client_id', 1)
            self.ibkr = IBKRInterface(host=host, port=port, client_id=client_id)
        
        # Initialize model manager
        self.model_manager = LSTMModelManager()
        
        # Training parameters
        self.max_parallel_trainings = self.trainer_config.get('max_parallel_trainings', 2)
        self.min_training_data_days = self.trainer_config.get('min_training_data_days', 90)
        self.max_training_attempts = self.trainer_config.get('max_training_attempts', 3)
        self.incremental_update_epochs = self.trainer_config.get('incremental_update_epochs', 10)
        
        # Symbol priority queue and training status
        self.training_queue = queue.PriorityQueue()
        self.training_status = {}  # symbol -> status info
        self.active_trainings = {}  # symbol -> future
        
        # Metrics cache for symbols
        self.symbol_metrics = {}  # symbol -> metrics
        
        # Threading and execution management
        self.queue_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_trainings)
        self.is_running = False
        self.worker_thread = None
        
        # Default training configuration for LSTM models
        self.lstm_config = {
            'seq_length': 60,
            'epochs': 50,
            'batch_size': 128,
            'units': 200,
            'dropout': 0.4,
            'learning_rate': 0.0001,
            'patience': 10,
            'num_layers': 2,
            'bidirectional': True,
            'l2_reg': 0,
            'lr_factor': 0.3,
            'min_lr': 1e-05
        }
        
        # Initialize data directory
        self.data_dir = os.path.join("data")
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        
        logger.info("LSTM Trainer initialized")
    
    def start(self):
        """Start the training queue processing."""
        if self.is_running:
            logger.warning("Training queue is already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("LSTM Trainer queue processing started")
    
    def stop(self):
        """Stop the training queue processing."""
        if not self.is_running:
            logger.warning("Training queue is not running")
            return
        
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("LSTM Trainer queue processing stopped")
    
    def _process_queue(self):
        """Process the training queue in a background thread."""
        logger.info("Starting training queue processing")
        
        while self.is_running:
            try:
                # Check if we can start new trainings
                if len(self.active_trainings) < self.max_parallel_trainings:
                    # Try to get a symbol from the queue (non-blocking)
                    try:
                        priority, symbol, config = self.training_queue.get(block=False)
                        
                        # Start training in a separate thread
                        self._start_training(symbol, config)
                        
                        # Mark task as done
                        self.training_queue.task_done()
                        
                    except queue.Empty:
                        # No items in queue, sleep briefly
                        time.sleep(1)
                else:
                    # Check if any active trainings have completed
                    completed = []
                    
                    for symbol, future in list(self.active_trainings.items()):
                        if future.done():
                            try:
                                # Get the result (will raise exception if the task failed)
                                result = future.result()
                                logger.info(f"Training completed for {symbol}: {result['status']}")
                                
                                # Update training status
                                self.training_status[symbol] = {
                                    'status': result['status'],
                                    'completed_at': datetime.now().isoformat(),
                                    'metrics': result.get('metrics', {}),
                                    'message': result.get('message', '')
                                }
                                
                                # Update metrics cache
                                if result['status'] == 'success' and 'metrics' in result:
                                    self.symbol_metrics[symbol] = result['metrics']
                                
                            except Exception as e:
                                logger.error(f"Error processing training result for {symbol}: {e}")
                                self.training_status[symbol] = {
                                    'status': 'error',
                                    'completed_at': datetime.now().isoformat(),
                                    'message': str(e)
                                }
                            
                            # Remove from active trainings
                            completed.append(symbol)
                    
                    # Remove completed trainings
                    for symbol in completed:
                        del self.active_trainings[symbol]
                    
                    # If we removed any, continue immediately to start new trainings
                    if not completed:
                        # Otherwise, sleep briefly
                        time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in training queue processing: {e}")
                time.sleep(5)  # Sleep longer on error
        
        logger.info("Training queue processing stopped")
    
    def _start_training(self, symbol: str, config: Dict):
        """
        Start training for a symbol in a separate thread.
        
        Args:
            symbol: Symbol to train
            config: Training configuration
        """
        try:
            logger.info(f"Starting training for {symbol}")
            
            # Submit the training task to the executor
            future = self.executor.submit(self.train_lstm_model, symbol, config)
            
            # Store the future
            self.active_trainings[symbol] = future
            
            # Update training status
            self.training_status[symbol] = {
                'status': 'training',
                'started_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting training for {symbol}: {e}")
            self.training_status[symbol] = {
                'status': 'error',
                'message': str(e)
            }
    
    def request_training(self, symbol: str, priority: int = 50, config: Dict = None):
        """
        Request training for a symbol with specified priority.
        Lower priority value means higher priority (will be trained sooner).
        
        Args:
            symbol: Symbol to train
            priority: Priority level (0-100, lower is higher priority)
            config: Optional custom configuration for training
        
        Returns:
            bool: True if training was requested, False otherwise
        """
        try:
            # Validate priority
            priority = max(0, min(100, priority))
            
            # Check if symbol is already in queue or being trained
            with self.queue_lock:
                # If already training, don't add again
                if symbol in self.active_trainings:
                    logger.info(f"Symbol {symbol} is already being trained")
                    return False
                
                # Check training status to avoid retraining failed symbols too frequently
                if symbol in self.training_status:
                    status = self.training_status[symbol]
                    
                    # If recently failed, check if we should retry
                    if status.get('status') == 'error' and 'completed_at' in status:
                        completed_at = datetime.fromisoformat(status['completed_at'])
                        hours_since_failure = (datetime.now() - completed_at).total_seconds() / 3600
                        
                        # If failed recently and has attempted too many times, don't retry yet
                        attempt_count = status.get('attempt_count', 0)
                        if hours_since_failure < 6 * (attempt_count or 1) and attempt_count >= self.max_training_attempts:
                            logger.info(f"Symbol {symbol} has failed training {attempt_count} times recently, skipping")
                            return False
                        
                        # Increment attempt count
                        status['attempt_count'] = attempt_count + 1
                
                # Use default config if none provided
                if config is None:
                    config = self.lstm_config.copy()
                
                # Add symbol to queue
                self.training_queue.put((priority, symbol, config))
                logger.info(f"Training requested for {symbol} with priority {priority}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error requesting training for {symbol}: {e}")
            return False
    
    def prioritize_symbols(self, symbols: List[str]) -> List[Tuple[int, str]]:
        """
        Prioritize symbols for training based on various factors.
        
        Args:
            symbols: List of symbols to prioritize
            
        Returns:
            List of tuples (priority, symbol) sorted by priority
        """
        try:
            # If no symbols, return empty list
            if not symbols:
                return []
            
            # Connect to IBKR if needed
            if not self.ibkr.connected:
                self.ibkr.connect()
            
            prioritized_symbols = []
            
            for symbol in symbols:
                # Start with base priority (50)
                priority = 50
                
                # 1. Check if model exists
                model_exists = self.model_manager.model_exists(symbol)
                if not model_exists:
                    # Higher priority for symbols without models
                    priority -= 20
                
                # 2. Check model freshness
                model_fresh = self.model_manager.model_is_fresh(symbol)
                if not model_fresh and model_exists:
                    # Higher priority for stale models
                    priority -= 15
                
                # 3. Check model performance (if available)
                model_metadata = self.model_manager.get_model_metadata(symbol)
                if model_metadata and 'metrics' in model_metadata:
                    metrics = model_metadata['metrics']
                    reliability = model_metadata.get('reliability_index', 0)
                    
                    # Lower priority for models with good performance
                    if reliability > 0.7:
                        priority += 10
                
                # 4. Check recent trading activity
                # This requires data from the trading manager, which we don't have direct access to
                # Could be extended in future versions
                
                # 5. Check trading volume and market data
                try:
                    # Get recent market data
                    data = self.ibkr.get_historical_data(
                        symbol=symbol,
                        duration="5 D",
                        bar_size="1 day",
                        what_to_show="TRADES"
                    )
                    
                    if not data.empty and 'volume' in data.columns:
                        # Average daily volume
                        avg_volume = data['volume'].mean()
                        
                        # Higher priority for higher volume symbols
                        if avg_volume > 10000000:  # Very high volume
                            priority -= 10
                        elif avg_volume > 5000000:  # High volume
                            priority -= 5
                        elif avg_volume < 1000000:  # Low volume
                            priority += 5
                        
                        # Volatility
                        if 'high' in data.columns and 'low' in data.columns:
                            avg_range_pct = ((data['high'] - data['low']) / data['low']).mean() * 100
                            
                            # Higher priority for more volatile symbols
                            if avg_range_pct > 5:  # Very volatile
                                priority -= 8
                            elif avg_range_pct > 3:  # Moderately volatile
                                priority -= 4
                except Exception as e:
                    logger.warning(f"Error fetching market data for {symbol}: {e}")
                
                # 6. Adjust priority based on previous training results
                if symbol in self.training_status:
                    status = self.training_status[symbol]
                    
                    # Lower priority for recently succeeded trainings
                    if status.get('status') == 'success' and 'completed_at' in status:
                        completed_at = datetime.fromisoformat(status['completed_at'])
                        days_since_training = (datetime.now() - completed_at).days
                        
                        if days_since_training < 7:  # Trained within last week
                            priority += 15
                    
                    # Higher priority for older failed trainings that have sufficient cooling off
                    if status.get('status') == 'error' and 'completed_at' in status:
                        completed_at = datetime.fromisoformat(status['completed_at'])
                        days_since_failure = (datetime.now() - completed_at).days
                        
                        if days_since_failure > 1:  # Failed more than a day ago
                            priority -= 5
                
                # Ensure priority stays within bounds
                priority = max(0, min(100, priority))
                
                # Add to list
                prioritized_symbols.append((priority, symbol))
            
            # Sort by priority (ascending)
            prioritized_symbols.sort()
            
            return prioritized_symbols
            
        except Exception as e:
            logger.error(f"Error prioritizing symbols: {e}")
            return [(50, symbol) for symbol in symbols]  # Default equal priority
    
    def schedule_training_batch(self, symbols: List[str], auto_prioritize: bool = True):
        """
        Schedule training for a batch of symbols.
        
        Args:
            symbols: List of symbols to train
            auto_prioritize: Whether to automatically prioritize symbols
            
        Returns:
            int: Number of symbols scheduled for training
        """
        if auto_prioritize:
            # Prioritize symbols
            prioritized_symbols = self.prioritize_symbols(symbols)
        else:
            # Use default priority
            prioritized_symbols = [(50, symbol) for symbol in symbols]
        
        scheduled_count = 0
        
        # Schedule training for each symbol
        for priority, symbol in prioritized_symbols:
            scheduled = self.request_training(symbol, priority)
            if scheduled:
                scheduled_count += 1
        
        logger.info(f"Scheduled training for {scheduled_count}/{len(symbols)} symbols")
        return scheduled_count
    
    def train_lstm_model(self, symbol: str, config: Dict = None) -> Dict:
        """
        Train LSTM model for a specific symbol.
        This is the main training function that handles both fresh and incremental training.
        
        Args:
            symbol: Symbol to train
            config: Training configuration
            
        Returns:
            dict: Training results
        """
        try:
            logger.info(f"Training LSTM model for {symbol}")
            
            # Ensure we're connected to IBKR
            if not self.ibkr.connected:
                success = self.ibkr.connect()
                if not success:
                    return {
                        'status': 'error',
                        'message': 'Failed to connect to IBKR'
                    }
            
            # Use default config if none provided
            if config is None:
                config = self.lstm_config.copy()
            
            # Set symbol specific configuration
            config['symbol'] = symbol
            config['label'] = f"model_{symbol}"
            
            # Check if we should do incremental training or fresh training
            model_exists = self.model_manager.model_exists(symbol)
            
            if model_exists:
                logger.info(f"Existing model found for {symbol}, checking performance")
                
                # Check model metrics
                metadata = self.model_manager.get_model_metadata(symbol)
                reliability = metadata.get('reliability_index', 0)
                
                # If model is reliable, do incremental training
                if reliability > 0.5:
                    logger.info(f"Model for {symbol} is reliable, doing incremental training")
                    return self._incremental_training(symbol, config)
                else:
                    logger.info(f"Model for {symbol} is not reliable, doing fresh training")
                    return self._fresh_training(symbol, config)
            else:
                logger.info(f"No existing model for {symbol}, doing fresh training")
                return self._fresh_training(symbol, config)
                
        except Exception as e:
            logger.error(f"Error in train_lstm_model for {symbol}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _fresh_training(self, symbol: str, config: Dict) -> Dict:
        """
        Train a new LSTM model from scratch.
        
        Args:
            symbol: Symbol to train
            config: Training configuration
            
        Returns:
            dict: Training results
        """
        try:
            logger.info(f"Starting fresh training for {symbol}")
            
            # Fetch historical data
            df = fetch_historical_data(self.ibkr, symbol, lookback_days=self.min_training_data_days * 2)
            
            if df.empty:
                return {
                    'status': 'error',
                    'message': f'No data fetched for {symbol}'
                }
            
            # Check if we have enough data
            if len(df) < self.min_training_data_days * 5:  # Assuming ~5 bars per day for 1h timeframe
                return {
                    'status': 'error',
                    'message': f'Insufficient data for {symbol}: {len(df)} bars, need at least {self.min_training_data_days * 5}'
                }
            
            # Preprocess data
            symbol_dir = self.model_manager.get_symbol_dir(symbol)
            processed_dir = os.path.join(self.data_dir, "processed")
            
            X_train, y_train, X_test, y_test, scaler, feature_cols, train_df, test_df = preprocess_data(
                df, 
                symbol, 
                seq_length=config['seq_length'],
                train_split=0.8,
                output_dir=processed_dir
            )
            
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_lstm_model(input_shape, config)
            
            # Train model
            history = train_model(model, X_train, y_train, X_test, y_test, config)
            
            # Evaluate model
            rmse, mae, r2, y_test_true, y_test_pred = evaluate_model(
                model, X_test, y_test, scaler, X_test
            )
            
            # Calculate reliability index (0-1)
            # Higher is better
            price_accuracy = max(0, min(1, 1 - (rmse / df['close'].mean())))
            direction_accuracy = self._calculate_direction_accuracy(y_test_true, y_test_pred)
            reliability_index = (price_accuracy * 0.7) + (direction_accuracy * 0.3)
            
            # Prepare metrics
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy),
                'price_accuracy': float(price_accuracy),
                'train_size': len(train_df),
                'test_size': len(test_df)
            }
            
            # Save model and scaler
            model_path = self.model_manager.get_model_path(symbol)
            scaler_path = self.model_manager.get_scaler_path(symbol)
            
            # Save model
            ke.saving.save_model(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'trained_date': datetime.now().isoformat(),
                'symbol': symbol,
                'metrics': metrics,
                'reliability_index': reliability_index,
                'feature_cols': feature_cols,
                'config': {k: v for k, v in config.items() if k != 'label'},
                'training_data': {
                    'start_date': train_df['datetime'].min().isoformat() if 'datetime' in train_df else '',
                    'end_date': test_df['datetime'].max().isoformat() if 'datetime' in test_df else '',
                    'rows': len(df)
                }
            }
            
            self.model_manager.save_model_metadata(symbol, metadata)
            
            logger.info(f"Fresh training completed for {symbol}: RMSE={rmse:.4f}, R²={r2:.4f}, Reliability={reliability_index:.4f}")
            
            return {
                'status': 'success',
                'method': 'fresh',
                'metrics': metrics,
                'reliability_index': reliability_index,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': self.model_manager.get_metadata_path(symbol)
            }
            
        except Exception as e:
            logger.error(f"Error in fresh training for {symbol}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _incremental_training(self, symbol: str, config: Dict) -> Dict:
        """
        Incrementally update an existing LSTM model with new data.
        
        Args:
            symbol: Symbol to train
            config: Training configuration
            
        Returns:
            dict: Training results
        """
        try:
            logger.info(f"Starting incremental training for {symbol}")
            
            # Load existing model and metadata
            model, scaler = self.model_manager.load_model_and_scaler(symbol)
            metadata = self.model_manager.get_model_metadata(symbol)
            
            if model is None or scaler is None:
                logger.warning(f"Could not load existing model for {symbol}, falling back to fresh training")
                return self._fresh_training(symbol, config)
            
            # Get the last training date
            last_trained = datetime.fromisoformat(metadata.get('trained_date', '2000-01-01T00:00:00'))
            days_since_training = (datetime.now() - last_trained).days
            
            # Fetch new data since last training (with some overlap)
            fetch_days = max(30, days_since_training + 15)  # At least 30 days, with 15 days overlap
            df = fetch_historical_data(self.ibkr, symbol, lookback_days=fetch_days)
            
            if df.empty:
                return {
                    'status': 'error',
                    'message': f'No new data fetched for {symbol}'
                }
            
            # Preprocess new data
            processed_dir = os.path.join(self.data_dir, "processed")
            
            # Get input shape from model
            input_shape = model.input_shape
            seq_length = input_shape[1]
            
            X_new, y_new, X_val, y_val, _, feature_cols, train_df, test_df = preprocess_data(
                df, 
                symbol, 
                seq_length=seq_length,
                train_split=0.8,
                output_dir=processed_dir
            )
            
            # Check feature compatibility
            if X_new.shape[2] != input_shape[2]:
                logger.warning(f"Feature mismatch in incremental training for {symbol}: {X_new.shape[2]} vs {input_shape[2]}")
                logger.warning("Falling back to fresh training")
                return self._fresh_training(symbol, config)
            
            # Configure fewer epochs for incremental training
            incremental_config = config.copy()
            incremental_config['epochs'] = min(config.get('epochs', 50), self.incremental_update_epochs)
            
            # Train model incrementally
            history = train_model(model, X_new, y_new, X_val, y_val, incremental_config)
            
            # Evaluate updated model
            rmse, mae, r2, y_val_true, y_val_pred = evaluate_model(
                model, X_val, y_val, scaler, X_val
            )
            
            # Calculate reliability index
            # We weight the new performance more heavily for incremental updates
            price_accuracy = max(0, min(1, 1 - (rmse / df['close'].mean())))
            direction_accuracy = self._calculate_direction_accuracy(y_val_true, y_val_pred)
            
            # Blend new reliability with existing
            old_reliability = metadata.get('reliability_index', 0.5)
            reliability_index = (old_reliability * 0.3) + (price_accuracy * 0.5) + (direction_accuracy * 0.2)
            
            # Prepare metrics
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy),
                'price_accuracy': float(price_accuracy),
                'train_size': len(train_df),
                'test_size': len(test_df)
            }
            
            # Save updated model and metadata
            model_path = self.model_manager.get_model_path(symbol)
            ke.saving.save_model(model, model_path)
            
            # Update metadata
            metadata.update({
                'trained_date': datetime.now().isoformat(),
                'metrics': metrics,
                'reliability_index': reliability_index,
                'incremental_updates': metadata.get('incremental_updates', 0) + 1,
                'training_data': {
                    'start_date': train_df['datetime'].min().isoformat() if 'datetime' in train_df else '',
                    'end_date': test_df['datetime'].max().isoformat() if 'datetime' in test_df else '',
                    'rows': len(df)
                }
            })
            
            self.model_manager.save_model_metadata(symbol, metadata)
            
            logger.info(f"Incremental training completed for {symbol}: RMSE={rmse:.4f}, R²={r2:.4f}, Reliability={reliability_index:.4f}")
            
            return {
                'status': 'success',
                'method': 'incremental',
                'metrics': metrics,
                'reliability_index': reliability_index,
                'model_path': model_path,
                'metadata_path': self.model_manager.get_metadata_path(symbol)
            }
            
        except Exception as e:
            logger.error(f"Error in incremental training for {symbol}: {e}")
            # If incremental fails, try fresh training as fallback
            logger.warning(f"Falling back to fresh training for {symbol}")
            return self._fresh_training(symbol, config)
    
    def _calculate_direction_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (percentage of correct up/down predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: Direction accuracy (0-1)
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.5  # Default value if not enough data
        
        # Calculate direction (1 for up, 0 for down)
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        direction_matches = true_direction == pred_direction
        accuracy = np.mean(direction_matches)
        
        return accuracy
    
    def get_training_status(self, symbol: str = None) -> Dict:
        """
        Get training status for one or all symbols.
        
        Args:
            symbol: Symbol to get status for, or None for all
            
        Returns:
            dict: Training status
        """
        if symbol:
            return self.training_status.get(symbol, {'status': 'unknown'})
        else:
            return self.training_status.copy()
    
    def get_queue_status(self) -> Dict:
        """
        Get status of the training queue.
        
        Returns:
            dict: Queue status
        """
        return {
            'queue_size': self.training_queue.qsize(),
            'active_trainings': list(self.active_trainings.keys()),
            'is_running': self.is_running
        }
    
    def get_trained_models_status(self) -> Dict:
        """
        Get status of all trained models.
        
        Returns:
            dict: Models status
        """
        models = self.model_manager.get_available_models()
        result = {}
        
        for symbol in models:
            metadata = self.model_manager.get_model_metadata(symbol)
            is_fresh = self.model_manager.model_is_fresh(symbol)
            
            # Extract key information
            trained_date = metadata.get('trained_date', 'unknown')
            if trained_date != 'unknown':
                try:
                    trained_dt = datetime.fromisoformat(trained_date)
                    days_old = (datetime.now() - trained_dt).days
                except:
                    days_old = None
            else:
                days_old = None
            
            reliability = metadata.get('reliability_index', 0)
            
            result[symbol] = {
                'trained_date': trained_date,
                'days_old': days_old,
                'is_fresh': is_fresh,
                'reliability': reliability,
                'metrics': metadata.get('metrics', {}),
                'needs_retraining': not is_fresh or reliability < 0.5
            }
        
        return result
    
    def run_parallel_batch_training(self, symbols: List[str], max_models: int = None) -> Dict:
        """
        Run parallel batch training for multiple symbols.
        
        This is a higher-level method that manages the entire training process:
        1. Prioritizes symbols
        2. Schedules training for the highest priority ones
        3. Starts the training queue if not running
        4. Returns status information
        
        Args:
            symbols: List of symbols to train
            max_models: Maximum number of models to train (default: train all)
            
        Returns:
            dict: Batch training status
        """
        try:
            # Prioritize symbols
            prioritized_symbols = self.prioritize_symbols(symbols)
            
            # Limit to max_models if specified
            if max_models is not None:
                prioritized_symbols = prioritized_symbols[:max_models]
            
            # Schedule training
            scheduled_symbols = []
            for priority, symbol in prioritized_symbols:
                scheduled = self.request_training(symbol, priority)
                if scheduled:
                    scheduled_symbols.append(symbol)
            
            # Start training queue if not running
            if not self.is_running:
                self.start()
            
            # Return status
            return {
                'status': 'success',
                'scheduled_symbols': scheduled_symbols,
                'total_scheduled': len(scheduled_symbols),
                'queue_status': self.get_queue_status()
            }
            
        except Exception as e:
            logger.error(f"Error in run_parallel_batch_training: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def train_universal_model(self, base_symbols: List[str] = None) -> Dict:
        """
        Train a universal LSTM model using data from multiple symbols.
        
        Args:
            base_symbols: List of symbols to use for training (default: use a predefined set)
            
        Returns:
            dict: Training result
        """
        try:
            # Use default symbols if not provided
            if not base_symbols:
                base_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            logger.info(f"Training universal model using symbols: {base_symbols}")
            
            # Connect to IBKR if needed
            if not self.ibkr.connected:
                success = self.ibkr.connect()
                if not success:
                    return {
                        'status': 'error',
                        'message': 'Failed to connect to IBKR'
                    }
            
            # Collect data from all symbols
            all_data = pd.DataFrame()
            
            for symbol in base_symbols:
                # Fetch data
                df = fetch_historical_data(self.ibkr, symbol, lookback_days=365)  # 1 year
                
                if df.empty:
                    logger.warning(f"No data fetched for {symbol}, skipping")
                    continue
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Append to all data
                all_data = pd.concat([all_data, df], ignore_index=True)
            
            if all_data.empty:
                return {
                    'status': 'error',
                    'message': 'No data fetched for any symbol'
                }
            
            # Process the combined data
            logger.info(f"Collected {len(all_data)} rows of data from {len(base_symbols)} symbols")
            
            # Preprocess data - we'll use a special function for universal model
            processed_dir = os.path.join(self.data_dir, "processed")
            universal_dir = os.path.join("models", "lstm", "universal")
            os.makedirs(universal_dir, exist_ok=True)
            
            # Since we're combining data from multiple symbols, we need to do preprocessing differently
            # First, calculate indicators for each symbol separately
            symbols_data = {}
            for symbol in base_symbols:
                symbol_data = all_data[all_data['symbol'] == symbol].copy()
                if len(symbol_data) > 0:
                    # Calculate indicators
                    symbol_data = calculate_technical_indicators(symbol_data, include_all=True)
                    symbol_data = calculate_trend_indicator(symbol_data)
                    symbol_data = normalize_indicators(symbol_data)
                    symbols_data[symbol] = symbol_data
            
            # Combine processed data
            processed_data = pd.concat(list(symbols_data.values()), ignore_index=True)
            
            # Handle sequences for each symbol separately, then combine
            seq_length = self.lstm_config['seq_length']
            feature_cols = [col for col in processed_data.columns if col.endswith('_norm')]
            
            # Ensure close_norm is included
            if "close_norm" not in feature_cols and "close_norm" in processed_data.columns:
                feature_cols.append("close_norm")
            
            # Generate sequences from processed data
            X_sequences = []
            y_values = []
            
            for symbol, df in symbols_data.items():
                # Create sequences
                for i in range(len(df) - seq_length):
                    # Extract sequence
                    X_seq = df[feature_cols].iloc[i:i+seq_length].values
                    # Target is close at the next step
                    y_val = df['close_norm'].iloc[i+seq_length]
                    
                    X_sequences.append(X_seq)
                    y_values.append(y_val)
            
            # Convert to arrays
            X = np.array(X_sequences)
            y = np.array(y_values)
            
            # Split into train and test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build universal model
            input_shape = (X_train.shape[1], X_train.shape[2])
            config = self.lstm_config.copy()
            config['label'] = "universal_model"
            
            model = build_lstm_model(input_shape, config)
            
            # Train model
            history = train_model(model, X_train, y_train, X_test, y_test, config)
            
            # Create a combined scaler for original price ranges
            scaler = MinMaxScaler()
            # Fit scaler on close prices from all symbols
            close_data = np.array([df['close'].values for df in symbols_data.values() if 'close' in df.columns])
            close_data = close_data.flatten().reshape(-1, 1)
            scaler.fit(close_data)
            
            # Evaluate model
            # Since we're using a universal scaler, evaluation is approximate
            dummy_data = X_test.copy()  # Just for structure
            rmse, mae, r2, _, _ = evaluate_model(model, X_test, y_test, None, None)
            
            # Save model and scaler
            model_path = os.path.join(universal_dir, "model.keras")
            scaler_path = os.path.join(universal_dir, "scaler.pkl")
            
            # Save model
            ke.saving.save_model(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Calculate reliability metrics
            # For universal model, we use a simpler approach
            reliability_index = max(0, min(1, 0.7 + (r2 * 0.3)))  # Base reliability + r2 contribution
            
            # Save metadata
            metadata = {
                'trained_date': datetime.now().isoformat(),
                'symbol': 'universal',
                'metrics': {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2)
                },
                'reliability_index': reliability_index,
                'feature_cols': feature_cols,
                'training_data': {
                    'symbols': base_symbols,
                    'rows': len(all_data),
                    'start_date': all_data['datetime'].min().isoformat() if 'datetime' in all_data else '',
                    'end_date': all_data['datetime'].max().isoformat() if 'datetime' in all_data else ''
                }
            }
            
            # Save metadata
            metadata_path = os.path.join(universal_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Universal model training completed: RMSE={rmse:.4f}, R²={r2:.4f}, Reliability={reliability_index:.4f}")
            
            return {
                'status': 'success',
                'metrics': metadata['metrics'],
                'reliability_index': reliability_index,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'training_symbols': base_symbols
            }
            
        except Exception as e:
            logger.error(f"Error training universal model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }


# Add MinMaxScaler import at the top of the file
from sklearn.preprocessing import MinMaxScaler
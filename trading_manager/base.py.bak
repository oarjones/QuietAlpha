"""
Trading Manager Base Class

This module provides the base implementation for the Trading Manager component,
which is responsible for executing trades on symbols selected by the Portfolio Manager.
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras as ke
from typing import List, Dict, Optional, Tuple, Union
import datetime
import time
from threading import Lock

from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, normalize_indicators, calculate_trend_indicator
from utils.data_utils import load_config

logger = logging.getLogger(__name__)

class TradingManager:
    """
    Trading Manager is responsible for executing and managing trades
    on symbols selected by the Portfolio Manager.
    """
    
    def __init__(self, config_path: str = None, ibkr_interface = None):
        """
        Initialize Trading Manager.
        
        Args:
            config_path (str): Path to configuration file
            ibkr_interface: IBKR interface instance
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else {}
        self.trading_config = self.config.get('trading_manager', {})
        
        # Set up IBKR interface
        self.ibkr = ibkr_interface
        if self.ibkr is None:
            host = self.config.get('ibkr', {}).get('host', '127.0.0.1')
            port = self.config.get('ibkr', {}).get('port', 7497)
            client_id = self.config.get('ibkr', {}).get('client_id', 1)
            self.ibkr = IBKRInterface(host=host, port=port, client_id=client_id)
        
        # Trading parameters
        self.timeframe = self.trading_config.get('timeframe', '1h')
        self.max_day_trades = self.trading_config.get('max_day_trades_per_week', 3)
        self.stop_loss_atr_mult = self.trading_config.get('stop_loss_atr_multiplier', 2.0)
        self.take_profit_atr_mult = self.trading_config.get('take_profit_atr_multiplier', 3.0)
        self.max_drawdown_pct = self.trading_config.get('max_drawdown_percentage', 5.0)
        
        # Active trades tracking
        self.active_trades = {}  # symbol -> trade_details
        self.trade_history = []  # list of completed trades
        self.day_trades_count = 0  # count of day trades in current 5-day period
        self.day_trades_dates = []  # dates of day trades for rolling 5-day window
        
        # Models
        self.models = {}
        self.model_path = os.path.join('models', 'trading_manager')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load pre-trained LSTM model if available
        self.lstm_models = {}  # symbol -> model
        self.lstm_scalers = {}  # symbol -> scaler
        
        # Concurrency control
        self.trade_lock = Lock()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load ML models for trading decisions."""
        # Try to load pre-trained LSTM models
        lstm_model_dir = os.path.join('models', 'lstm')
        if os.path.exists(lstm_model_dir):
            for model_file in os.listdir(lstm_model_dir):
                if model_file.endswith('.keras'):
                    symbol = model_file.split('_')[2].split('.')[0]  # Extract symbol from filename
                    model_path = os.path.join(lstm_model_dir, model_file)
                    
                    try:
                        self.lstm_models[symbol] = ke.models.load_model(model_path)
                        logger.info(f"Loaded LSTM model for {symbol}")
                        
                        # Check for corresponding scaler
                        scaler_file = f"scaler_{symbol}.pkl"
                        scaler_path = os.path.join(lstm_model_dir, scaler_file)
                        if os.path.exists(scaler_path):
                            self.lstm_scalers[symbol] = joblib.load(scaler_path)
                            logger.info(f"Loaded scaler for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading LSTM model for {symbol}: {e}")
        
        # Try to load RL agent if available
        rl_model_path = os.path.join(self.model_path, 'rl_agent.pkl')
        if os.path.exists(rl_model_path):
            try:
                self.models['rl_agent'] = joblib.load(rl_model_path)
                logger.info("Loaded RL agent model")
            except Exception as e:
                logger.error(f"Error loading RL agent: {e}")
    
    def connect_to_broker(self) -> bool:
        """
        Connect to Interactive Brokers.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        if not self.ibkr.connected:
            return self.ibkr.connect()
        return True
    
    def get_market_data(self, symbol: str, lookback_periods: int = 200) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol (str): Symbol to get data for
            lookback_periods (int): Number of periods to look back
            
        Returns:
            pd.DataFrame: Historical market data
        """
        timeframe_map = {
            '1min': '1 min',
            '5min': '5 mins',
            '15min': '15 mins',
            '30min': '30 mins',
            '1h': '1 hour',
            '2h': '2 hours',
            '4h': '4 hours',
            '1d': '1 day'
        }
        
        # Convert timeframe to IBKR format
        bar_size = timeframe_map.get(self.timeframe, '1 hour')
        
        # Determine duration string based on lookback periods and timeframe
        if self.timeframe in ['1min', '5min', '15min', '30min']:
            # For intraday timeframes, convert periods to days (approximately)
            days = max(1, int(lookback_periods * int(self.timeframe.replace('min', '')) / (60 * 6.5)))
            duration = f"{days} D"
        elif self.timeframe in ['1h', '2h', '4h']:
            # For hourly timeframes, convert to days
            hours_per_day = 6.5  # Approximate trading hours per day
            days = max(1, int(lookback_periods * int(self.timeframe.replace('h', '')) / hours_per_day))
            duration = f"{days} D"
        elif self.timeframe == '1d':
            # For daily timeframe, use days directly
            duration = f"{lookback_periods} D"
        else:
            # Default fallback
            duration = "5 D"
        
        # Fetch data from IBKR
        df = self.ibkr.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES",
            use_rth=True
        )
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Handle datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            logger.warning("No datetime column found, creating one")
            df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
            
        # Sort by datetime
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, include_all=True)
        df = calculate_trend_indicator(df)
        df = normalize_indicators(df)

        # Remove NaN values
        df.dropna(inplace=True)
        
        return df
    
    
    
    def predict_price_with_lstm(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Use LSTM model to predict future price.
        
        This version supports both universal and symbol-specific models,
        and uses a percentage-based approach for price prediction.
        
        Args:
            symbol (str): Symbol to predict for
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.get_market_data(symbol)
                    
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return {'predicted_direction': 'unknown', 'confidence': 0}
            
            # Try to get symbol-specific model first
            from lstm.model_manager import LSTMModelManager
            model_manager = LSTMModelManager()
            
            # Check if symbol-specific model exists
            symbol_specific = False
            if model_manager.model_exists(symbol) and model_manager.model_is_fresh(symbol):
                # Load symbol-specific model
                model, scaler = model_manager.load_model_and_scaler(symbol)
                if model is not None:
                    logger.info(f"Using symbol-specific LSTM model for {symbol}")
                    symbol_specific = True
            
            # Fall back to universal model if no symbol-specific model
            if not symbol_specific:
                if not hasattr(self, 'universal_lstm_model') or self.universal_lstm_model is None:
                    # Load universal model
                    if model_manager is not None:
                        model, scaler = model_manager.load_universal_model_and_scaler()
                    else:
                        # Legacy path for loading universal model
                        model_path = os.path.join('models', 'lstm', 'lstm_model_universal.keras')
                        scaler_path = os.path.join('models', 'lstm', 'scaler_universal.pkl')
                        
                        try:
                            model = ke.models.load_model(model_path)
                            scaler = joblib.load(scaler_path)
                            logger.info(f"Loaded universal LSTM model")
                        except Exception as e:
                            logger.warning(f"Could not load universal LSTM model: {e}")
                            return {'predicted_direction': 'unknown', 'confidence': 0}
                    
                    # Cache for future use
                    self.universal_lstm_model = model
                    self.universal_lstm_scaler = scaler
                else:
                    # Use cached universal model
                    model = self.universal_lstm_model
                    scaler = self.universal_lstm_scaler
                
                logger.info(f"Using universal LSTM model for {symbol}")
            
            # Prepare data for prediction
            # 1. Ensure we have all needed technical indicators
            if 'RSI_14' not in data.columns:
                data = calculate_technical_indicators(data, include_all=True)
                data = calculate_trend_indicator(data)
                data = normalize_indicators(data)
            
            # 2. Get normalized features
            feature_cols = [col for col in data.columns if col.endswith('_norm')]
            
            # 3. Ensure close_norm is included
            if 'close_norm' not in feature_cols and 'close_norm' in data.columns:
                feature_cols.append('close_norm')
            
            # 4. Check if we have enough data
            seq_length = 60  # This should match the training seq_length
            if len(data) < seq_length:
                logger.warning(f"Not enough data for {symbol}, need at least {seq_length} periods")
                return {'predicted_direction': 'unknown', 'confidence': 0}
            
            # 5. Check feature count compatibility
            expected_features = 41  # The number of features used during training
            current_features = len(feature_cols)
            
            # Log feature counts for debugging
            logger.info(f"Model expects {expected_features} features, found {current_features} features")
            
            # If we don't have the exact number of features needed, adjust our features
            if current_features != expected_features:
                logger.warning(f"Feature count mismatch: expected {expected_features}, got {current_features}")
                
                # Option 1: Add missing features with default values if we have fewer than expected
                if current_features < expected_features:
                    missing_count = expected_features - current_features
                    logger.info(f"Adding {missing_count} placeholder features with default values")
                    # Add placeholder columns with 0.5 (neutral value)
                    for i in range(missing_count):
                        placeholder_name = f"placeholder_{i}_norm"
                        data[placeholder_name] = 0.5
                        feature_cols.append(placeholder_name)
                
                # Option 2: Select only the most important features if we have more than expected
                elif current_features > expected_features:
                    logger.info(f"Selecting {expected_features} most important features from {current_features}")
                    # Ensure we keep close_norm as it's the primary target
                    essential_features = ['close_norm']
                    # Add other important features 
                    important_features = [
                        'TrendStrength_norm', 'RSI_14_norm', 'EMA_8_norm', 'EMA_21_norm', 'SMA_34_norm',
                        'MACD_diff_norm', 'MACD_line_norm', 'MACD_signal_norm', 'ATR_14_norm',
                        'Stoch_k_norm', 'Stoch_d_norm', 'BB_Width_norm', 'ADX_14_norm',
                        'volume_norm', 'OBV_norm'
                    ]
                    
                    # Ensure these features exist in the data
                    available_important = [f for f in important_features if f in feature_cols]
                    
                    # Combine essential and important features
                    selected_features = essential_features + available_important
                    
                    # If we still need more features, add remaining ones until we reach expected count
                    remaining_features = [f for f in feature_cols if f not in selected_features]
                    needed_count = expected_features - len(selected_features)
                    if needed_count > 0 and len(remaining_features) > 0:
                        additional_features = remaining_features[:needed_count]
                        selected_features.extend(additional_features)
                    
                    # If we still don't have enough, add placeholder features
                    if len(selected_features) < expected_features:
                        missing_count = expected_features - len(selected_features)
                        for i in range(missing_count):
                            placeholder_name = f"placeholder_{i}_norm"
                            data[placeholder_name] = 0.5
                            selected_features.append(placeholder_name)
                    
                    # Trim to expected count if we have too many
                    feature_cols = selected_features[:expected_features]
                    
                logger.info(f"Adjusted feature count: {len(feature_cols)}")
            
            # 6. Extract the last sequence
            sequence = data[feature_cols].values[-seq_length:]
            
            # 7. Reshape for model input
            X = np.array([sequence])
            
            # 8. Make prediction
            prediction = model.predict(X, verbose=0)[0][0]
            
            # 9. Get the current normalized close and determine if prediction is higher or lower
            current_close_norm = data['close_norm'].iloc[-1]
            current_close = data['close'].iloc[-1]
            
            # 10. Determine direction and confidence
            if prediction > current_close_norm:
                direction = 'up'
                confidence = min(1.0, (prediction - current_close_norm) * 10)  # Scale difference to confidence
            elif prediction < current_close_norm:
                direction = 'down'
                confidence = min(1.0, (current_close_norm - prediction) * 10)  # Scale difference to confidence
            else:
                direction = 'neutral'
                confidence = 0.0
            
            # 11. Calculate predicted price using percentage-based approach
            # Get recent volatility as a guide
            atr = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else current_close * 0.01
            price_volatility = atr / current_close  # Normalized volatility
            
            # Base expected change on confidence and volatility
            volatility_factor = 5  # Multiplier for volatility (can be tuned)
            expected_pct_change = confidence * price_volatility * volatility_factor
            
            # Limit to reasonable change (e.g., maximum 2% per period)
            max_pct_change = 0.02
            expected_pct_change = min(expected_pct_change, max_pct_change)
            
            # Calculate predicted price
            if direction == 'up':
                predicted_price = current_close * (1 + expected_pct_change)
            elif direction == 'down':
                predicted_price = current_close * (1 - expected_pct_change)
            else:
                predicted_price = current_close
            
            # 12. Prepare result
            result = {
                'predicted_direction': direction,
                'predicted_value_norm': float(prediction),
                'current_value_norm': float(current_close_norm),
                'confidence': float(confidence),
                'current_price': float(current_close),
                'predicted_price': float(predicted_price),
                'price_change': float(predicted_price - current_close),
                'price_change_pct': float((predicted_price / current_close - 1) * 100),
                'model_type': 'symbol_specific' if symbol_specific else 'universal',
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Error predicting with LSTM for {symbol}: {e}")
            return {'predicted_direction': 'error', 'confidence': 0, 'error': str(e)}

        

    
    """
    Modified generate_trading_signals method for TradingManager

    This code should replace or modify the generate_trading_signals method in the TradingManager
    class to better integrate the universal LSTM model's predictions with technical analysis.
    """

    def generate_trading_signals(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Generate trading signals based on technical analysis and universal LSTM predictions.
        
        Args:
            symbol (str): Symbol to generate signals for
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            
        Returns:
            Dict: Trading signals
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.get_market_data(symbol)
                    
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return {'signal': 'neutral', 'strength': 0}
            
            # Get LSTM prediction
            lstm_prediction = self.predict_price_with_lstm(symbol, data)
            
            # Extract technical indicators from most recent data
            latest = data.iloc[-1]
            
            # Calculate signal based on technical indicators
            signal_strength = 0
            signal_reasons = []
            
            # Trend strength and direction
            trend_strength = latest.get('TrendStrength_norm', 0.5)
            if trend_strength > 0.7:
                signal_strength += 30
                signal_reasons.append('strong_trend')
            elif trend_strength > 0.5:
                signal_strength += 15
                signal_reasons.append('moderate_trend')
            
            # Moving averages
            if 'close' in latest and 'EMA_8' in latest and 'EMA_21' in latest:
                if latest['close'] > latest['EMA_8'] > latest['EMA_21']:
                    signal_strength += 20
                    signal_reasons.append('bullish_ma_alignment')
                elif latest['close'] < latest['EMA_8'] < latest['EMA_21']:
                    signal_strength -= 20
                    signal_reasons.append('bearish_ma_alignment')
            
            # RSI
            if 'RSI_14' in latest:
                rsi = latest['RSI_14']
                if rsi > 70:
                    signal_strength -= 15
                    signal_reasons.append('overbought')
                elif rsi < 30:
                    signal_strength += 15
                    signal_reasons.append('oversold')
            
            # MACD
            if 'MACD_diff' in latest:
                macd_diff = latest['MACD_diff']
                if macd_diff > 0:
                    signal_strength += 10
                    signal_reasons.append('bullish_macd')
                elif macd_diff < 0:
                    signal_strength -= 10
                    signal_reasons.append('bearish_macd')
            
            # Bollinger Bands
            if all(x in latest for x in ['close', 'BB_High', 'BB_Low']):
                bb_width = (latest['BB_High'] - latest['BB_Low']) / latest['close']
                bb_position = (latest['close'] - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low']) if latest['BB_High'] != latest['BB_Low'] else 0.5
                
                if bb_position > 0.8:
                    signal_strength -= 10
                    signal_reasons.append('near_upper_bollinger')
                elif bb_position < 0.2:
                    signal_strength += 10
                    signal_reasons.append('near_lower_bollinger')
                
                if bb_width < 0.03:  # Narrow bands (3% of price)
                    signal_strength += 5
                    signal_reasons.append('tight_consolidation')
            
            # Enhanced LSTM prediction integration
            if lstm_prediction['predicted_direction'] == 'up':
                # Scale the impact based on confidence and price change percentage
                lstm_weight = 25
                if 'price_change_pct' in lstm_prediction:
                    # Increase weight for larger predicted price movements
                    price_change_pct = abs(lstm_prediction['price_change_pct'])
                    if price_change_pct > 1.0:  # If predicting more than 1% change
                        lstm_weight = 35
                    elif price_change_pct > 0.5:  # If predicting more than 0.5% change
                        lstm_weight = 30
                
                lstm_impact = lstm_weight * lstm_prediction['confidence']
                signal_strength += lstm_impact
                signal_reasons.append(f'lstm_bullish_{lstm_prediction["confidence"]:.2f}')
                
                # Add extra weight if LSTM and technical analysis agree
                if signal_strength > 0:
                    signal_strength += 10
                    signal_reasons.append('lstm_technical_agreement')
            elif lstm_prediction['predicted_direction'] == 'down':
                # Scale the impact based on confidence and price change percentage
                lstm_weight = 25
                if 'price_change_pct' in lstm_prediction:
                    # Increase weight for larger predicted price movements
                    price_change_pct = abs(lstm_prediction['price_change_pct'])
                    if price_change_pct > 1.0:  # If predicting more than 1% change
                        lstm_weight = 35
                    elif price_change_pct > 0.5:  # If predicting more than 0.5% change
                        lstm_weight = 30
                
                lstm_impact = lstm_weight * lstm_prediction['confidence']
                signal_strength -= lstm_impact
                signal_reasons.append(f'lstm_bearish_{lstm_prediction["confidence"]:.2f}')
                
                # Add extra weight if LSTM and technical analysis agree
                if signal_strength < 0:
                    signal_strength -= 10
                    signal_reasons.append('lstm_technical_agreement')
            
            # Determine final signal
            if signal_strength > 30:
                signal = 'buy'
            elif signal_strength < -30:
                signal = 'sell'
            else:
                signal = 'neutral'
            
            return {
                'symbol': symbol,
                'signal': signal,
                'strength': abs(signal_strength),
                'reasons': signal_reasons,
                'indicators': {
                    'trend_strength': float(trend_strength),
                    'rsi': float(latest.get('RSI_14', 50)),
                    'macd': float(latest.get('MACD_diff', 0)),
                },
                'lstm_prediction': lstm_prediction,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return {'signal': 'error', 'strength': 0, 'error': str(e)}

    
    
    def check_day_trade_limit(self) -> bool:
        """
        Check if we've reached the day trading limit.
        
        Returns:
            bool: True if we can make another day trade, False otherwise
        """
        # Update IBKR day trades count
        if self.ibkr.connected:
            self.day_trades_count = self.ibkr.day_trades_count
        
        # Clean up old day trades (older than 5 trading days)
        current_date = datetime.datetime.now().date()
        self.day_trades_dates = [date for date in self.day_trades_dates 
                               if (current_date - date).days <= 5]
        
        # Check if we can make another day trade
        account_summary = self.ibkr.get_account_summary() if self.ibkr.connected else {}
        account_value = account_summary.get('NetLiquidation', 0)
        
        # If account value is over $25,000, no PDT rule applies
        if account_value >= 25000:
            return True
        
        # Otherwise, check if we've made too many day trades
        return len(self.day_trades_dates) < self.max_day_trades
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                               risk_per_trade: float = 0.02) -> Dict:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            symbol (str): Symbol to trade
            signal_strength (float): Strength of the trading signal (0-100)
            risk_per_trade (float): Maximum risk per trade as fraction of account
            
        Returns:
            Dict: Position sizing details
        """
        try:
            # Get account value
            account_summary = self.ibkr.get_account_summary() if self.ibkr.connected else {}
            account_value = account_summary.get('NetLiquidation', 0)
            
            if account_value <= 0:
                logger.warning("Account value not available, using default")
                account_value = self.config.get('initial_capital', 2500)
            
            # Get current market data for volatility assessment
            data = self.get_market_data(symbol)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}, using default position size")
                return {
                    'symbol': symbol,
                    'position_size': 0,
                    'risk_amount': 0,
                    'account_value': account_value
                }
            
            # Calculate ATR for stop loss placement
            atr = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else data['close'].iloc[-1] * 0.02
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Calculate stop loss distance based on ATR and signal strength
            # For stronger signals, we can use wider stops
            stop_distance = self.stop_loss_atr_mult * atr
            
            # Adjust stop distance based on signal strength
            signal_factor = min(1.5, max(0.5, signal_strength / 50))  # Scale from 0.5 to 1.5
            adjusted_stop = stop_distance * signal_factor
            
            # Calculate dollar risk (account value * risk percentage)
            dollar_risk = account_value * risk_per_trade
            
            # Calculate position size based on risk and stop distance
            shares = int(dollar_risk / adjusted_stop)
            
            # Limit position size to a maximum percentage of account
            max_position_size = account_value * self.config.get('risk_management', {}).get('max_position_size_percentage', 0.1)
            position_value = shares * current_price
            
            if position_value > max_position_size:
                shares = int(max_position_size / current_price)
            
            return {
                'symbol': symbol,
                'position_size': shares,
                'position_value': shares * current_price,
                'stop_loss_price': current_price - adjusted_stop if shares > 0 else current_price + adjusted_stop,
                'take_profit_price': current_price + (adjusted_stop * self.take_profit_atr_mult / self.stop_loss_atr_mult) if shares > 0 else current_price - (adjusted_stop * self.take_profit_atr_mult / self.stop_loss_atr_mult),
                'risk_amount': dollar_risk,
                'risk_per_share': adjusted_stop,
                'account_value': account_value,
                'atr': atr,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return {
                'symbol': symbol,
                'position_size': 0,
                'risk_amount': 0,
                'error': str(e)
            }
    
    def execute_trade(self, symbol: str, action: str, position_details: Dict) -> Dict:
        """
        Execute a trade with proper risk management.
        
        Args:
            symbol (str): Symbol to trade
            action (str): Trade action (buy/sell)
            position_details (Dict): Position sizing details
            
        Returns:
            Dict: Trade execution results
        """
        with self.trade_lock:
            try:
                if not self.ibkr.connected:
                    if not self.connect_to_broker():
                        return {'status': 'error', 'message': 'Not connected to broker'}
                
                # Check if this would be a day trade and if we're allowed to make it
                is_day_trade = False
                positions = self.ibkr.get_positions()
                for pos in positions:
                    if pos['symbol'] == symbol and (
                        (action == 'sell' and pos['position'] > 0) or
                        (action == 'buy' and pos['position'] < 0)
                    ):
                        is_day_trade = True
                        break
                
                if is_day_trade and not self.check_day_trade_limit():
                    logger.warning(f"Cannot execute day trade for {symbol}, reached PDT limit")
                    return {'status': 'error', 'message': 'PDT limit reached'}
                
                # Get position size and prices
                position_size = position_details.get('position_size', 0)
                if position_size <= 0:
                    logger.warning(f"Invalid position size for {symbol}: {position_size}")
                    return {'status': 'error', 'message': 'Invalid position size'}
                
                stop_loss_price = position_details.get('stop_loss_price')
                take_profit_price = position_details.get('take_profit_price')
                current_price = position_details.get('current_price')
                
                # Execute bracket order
                trade_result = self.ibkr.place_bracket_order(
                    symbol=symbol,
                    action='BUY' if action == 'buy' else 'SELL',
                    quantity=position_size,
                    entry_price=current_price,
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                    as_market=True  # Use market order for entry
                )
                
                if trade_result.get('status') == 'submitted':
                    logger.info(f"Trade executed for {symbol}: {action}, size: {position_size}")
                    
                    # Record day trade if applicable
                    if is_day_trade:
                        self.day_trades_dates.append(datetime.datetime.now().date())
                        self.day_trades_count += 1
                    
                    # Record active trade
                    self.active_trades[symbol] = {
                        'symbol': symbol,
                        'action': action,
                        'entry_price': current_price,
                        'entry_time': datetime.datetime.now().isoformat(),
                        'position_size': position_size,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'order_ids': trade_result.get('parent_order_id', 0),
                        'is_day_trade': is_day_trade
                    }
                    
                    return {
                        'status': 'success',
                        'symbol': symbol,
                        'action': action,
                        'position_size': position_size,
                        'entry_price': current_price,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'is_day_trade': is_day_trade,
                        'order_details': trade_result
                    }
                else:
                    logger.error(f"Trade execution failed for {symbol}: {trade_result}")
                    return {
                        'status': 'error',
                        'message': f"Trade execution failed: {trade_result.get('message', 'Unknown error')}",
                        'details': trade_result
                    }
                
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
                return {'status': 'error', 'message': str(e)}
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close an open position.
        
        Args:
            symbol (str): Symbol to close position for
            
        Returns:
            Dict: Position closing results
        """
        with self.trade_lock:
            try:
                if not self.ibkr.connected:
                    if not self.connect_to_broker():
                        return {'status': 'error', 'message': 'Not connected to broker'}
                
                # Check if we have an active trade for this symbol
                if symbol not in self.active_trades:
                    logger.warning(f"No active trade found for {symbol}")
                    return {'status': 'warning', 'message': 'No active trade found'}
                
                # Get current position from IBKR
                positions = self.ibkr.get_positions()
                position = None
                for pos in positions:
                    if pos['symbol'] == symbol:
                        position = pos
                        break
                
                if not position or position['position'] == 0:
                    logger.warning(f"No open position found for {symbol}")
                    # Clean up active trade record anyway
                    trade_record = self.active_trades.pop(symbol)
                    return {'status': 'warning', 'message': 'No open position found', 'trade_record': trade_record}
                
                # Get position details
                position_size = abs(position['position'])
                position_direction = 'long' if position['position'] > 0 else 'short'
                
                # Execute market order to close position
                trade_result = self.ibkr.place_market_order(
                    symbol=symbol,
                    action='SELL' if position_direction == 'long' else 'BUY',
                    quantity=position_size
                )
                
                if trade_result.get('status') == 'submitted':
                    logger.info(f"Position closed for {symbol}: {position_direction}, size: {position_size}")
                    
                    # Record closed trade
                    trade_record = self.active_trades.pop(symbol)
                    trade_record['exit_price'] = trade_result.get('order_status', {}).get('avgFillPrice', position['marketPrice'])
                    trade_record['exit_time'] = datetime.datetime.now().isoformat()
                    
                    # Calculate P&L
                    if position_direction == 'long':
                        pl_points = trade_record['exit_price'] - trade_record['entry_price']
                    else:
                        pl_points = trade_record['entry_price'] - trade_record['exit_price']
                        
                    pl_amount = pl_points * position_size
                    trade_record['pl_points'] = pl_points
                    trade_record['pl_amount'] = pl_amount
                    
                    # Add to trade history
                    self.trade_history.append(trade_record)
                    
                    return {
                        'status': 'success',
                        'symbol': symbol,
                        'position_size': position_size,
                        'position_direction': position_direction,
                        'entry_price': trade_record['entry_price'],
                        'exit_price': trade_record['exit_price'],
                        'pl_points': pl_points,
                        'pl_amount': pl_amount,
                        'order_details': trade_result
                    }
                else:
                    logger.error(f"Position closing failed for {symbol}: {trade_result}")
                    return {
                        'status': 'error',
                        'message': f"Position closing failed: {trade_result.get('message', 'Unknown error')}",
                        'details': trade_result
                    }
                
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                return {'status': 'error', 'message': str(e)}
    
    def process_trading_signals(self, symbol: str) -> Dict:
        """
        Process trading signals and execute trades if appropriate.
        
        Args:
            symbol (str): Symbol to process
            
        Returns:
            Dict: Processing results
        """
        try:
            # Generate trading signals
            signals = self.generate_trading_signals(symbol)
            
            if signals.get('signal') == 'error':
                logger.error(f"Error generating signals for {symbol}: {signals.get('error')}")
                return {'status': 'error', 'message': f"Signal error: {signals.get('error')}"}
            
            # Check if we already have an active trade for this symbol
            has_active_trade = symbol in self.active_trades
            
            # Decide what to do based on signals and active trades
            if has_active_trade:
                active_trade = self.active_trades[symbol]
                active_direction = active_trade['action']
                
                # Check if signal is opposite to our position (exit signal)
                opposite_signal = (active_direction == 'buy' and signals['signal'] == 'sell') or \
                                (active_direction == 'sell' and signals['signal'] == 'buy')
                
                if opposite_signal and signals['strength'] > 50:
                    # Strong opposite signal, close position
                    logger.info(f"Closing position for {symbol} due to opposite signal")
                    return self.close_position(symbol)
                else:
                    # No action needed, maintain position
                    return {
                        'status': 'info',
                        'message': f"Maintaining {active_direction} position for {symbol}",
                        'signal': signals['signal'],
                        'strength': signals['strength']
                    }
            else:
                # No active trade, check if we should enter a new one
                if signals['signal'] in ['buy', 'sell'] and signals['strength'] > 70:
                    # Strong signal to enter a new trade
                    
                    # Calculate position size
                    position_details = self.calculate_position_size(symbol, signals['strength'])
                    
                    if position_details.get('position_size', 0) <= 0:
                        logger.warning(f"Calculated zero position size for {symbol}")
                        return {
                            'status': 'warning',
                            'message': 'Zero position size calculated',
                            'signal': signals['signal'],
                            'strength': signals['strength']
                        }
                    
                    # Check if this would be a day trade and if we can make it
                    if not self.check_day_trade_limit():
                        # If we can't make a day trade, consider alternatives
                        current_time = self.ibkr.ib.reqCurrentTime() if self.ibkr.connected else None
                        
                        # If it's late in the trading day, skip the trade to avoid PDT violations
                        if current_time and current_time.hour >= 15:  # After 3 PM
                            logger.info(f"Skipping {symbol} trade to avoid PDT violation late in the day")
                            return {
                                'status': 'skipped',
                                'message': 'Trade skipped to avoid PDT violation (late in day)',
                                'signal': signals['signal'],
                                'strength': signals['strength']
                            }
                        
                        # Otherwise, consider a swing trade instead of a day trade
                        logger.info(f"Converting to swing trade for {symbol} to avoid PDT violation")
                        # Adjust position size for swing trade (typically smaller)
                        position_details['position_size'] = max(1, position_details['position_size'] // 2)
                        
                        # Adjust stop loss and take profit for swing trade (wider)
                        atr = position_details.get('atr', 0)
                        if atr > 0:
                            if signals['signal'] == 'buy':
                                position_details['stop_loss_price'] = position_details['current_price'] - (atr * 3.0)
                                position_details['take_profit_price'] = position_details['current_price'] + (atr * 4.5)
                            else:  # sell
                                position_details['stop_loss_price'] = position_details['current_price'] + (atr * 3.0)
                                position_details['take_profit_price'] = position_details['current_price'] - (atr * 4.5)
                    
                    # Execute the trade
                    logger.info(f"Executing {signals['signal']} trade for {symbol} with strength {signals['strength']}")
                    return self.execute_trade(symbol, signals['signal'], position_details)
                else:
                    # Signal not strong enough to enter a trade
                    return {
                        'status': 'info',
                        'message': f"No trade: {signals['signal']} signal with strength {signals['strength']} insufficient",
                        'signal': signals['signal'],
                        'strength': signals['strength']
                    }
        
        except Exception as e:
            logger.error(f"Error processing signals for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}

    def manage_open_positions(self) -> Dict:
        """
        Manage all open positions, updating stops and evaluating exits.
        
        Returns:
            Dict: Results of position management
        """
        results = {
            'monitored': 0,
            'updated': 0,
            'closed': 0,
            'errors': 0,
            'details': []
        }
        
        try:
            # Check connection to broker
            if not self.ibkr.connected:
                if not self.connect_to_broker():
                    return {'status': 'error', 'message': 'Not connected to broker'}
            
            # Get all current positions
            positions = self.ibkr.get_positions()
            
            # Process each position
            for position in positions:
                symbol = position['symbol']
                position_size = position['position']
                
                if position_size == 0:
                    continue
                
                results['monitored'] += 1
                
                try:
                    # Get current market data
                    data = self.get_market_data(symbol)
                    
                    if data.empty:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Generate current signals
                    signals = self.generate_trading_signals(symbol)
                    
                    # Determine if we should exit the position
                    should_exit = False
                    exit_reason = ""
                    
                    # 1. Check for exit signals
                    is_long = position_size > 0
                    if (is_long and signals['signal'] == 'sell' and signals['strength'] > 60) or \
                    (not is_long and signals['signal'] == 'buy' and signals['strength'] > 60):
                        should_exit = True
                        exit_reason = "opposite_signal"
                    
                    # 2. Check technical exit conditions
                    current_price = data['close'].iloc[-1]
                    
                    # For long positions
                    if is_long:
                        # Trail stop on significant moves
                        if symbol in self.active_trades:
                            trade = self.active_trades[symbol]
                            entry_price = trade.get('entry_price', current_price)
                            stop_price = trade.get('stop_loss_price', 0)
                            
                            # If price moved up significantly, update stop loss
                            price_change_pct = (current_price - entry_price) / entry_price
                            if price_change_pct > 0.02:  # 2% move
                                # Calculate new trailing stop
                                atr = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else current_price * 0.02
                                new_stop = max(stop_price, current_price - (atr * 2.0))
                                
                                if new_stop > stop_price:
                                    # Update stop loss in active trades record
                                    self.active_trades[symbol]['stop_loss_price'] = new_stop
                                    
                                    # Update actual stop order (in a real system)
                                    # For simplicity, we're just updating our internal record here
                                    # In a production system, you would modify the actual order
                                    
                                    results['updated'] += 1
                                    results['details'].append({
                                        'symbol': symbol,
                                        'action': 'update_stop',
                                        'old_stop': stop_price,
                                        'new_stop': new_stop
                                    })
                                    
                                    logger.info(f"Updated trailing stop for {symbol} from {stop_price} to {new_stop}")
                    
                    # 3. Check time-based exits for swing trades
                    if symbol in self.active_trades:
                        trade = self.active_trades[symbol]
                        if not trade.get('is_day_trade', False):
                            # For swing trades, check if we've held for target duration
                            import datetime
                            entry_time = datetime.datetime.fromisoformat(trade['entry_time'])
                            current_time = datetime.datetime.now()
                            
                            # Exit swing trades after a certain number of days if profitable
                            days_held = (current_time - entry_time).days
                            
                            # Calculate current P&L
                            if is_long:
                                pl_pct = (current_price - trade['entry_price']) / trade['entry_price']
                            else:
                                pl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                            
                            # Exit if held for 5+ days and profitable
                            if days_held >= 5 and pl_pct > 0:
                                should_exit = True
                                exit_reason = "time_based_profit_take"
                            
                            # Exit if held for 10+ days regardless
                            elif days_held >= 10:
                                should_exit = True
                                exit_reason = "max_hold_time"
                    
                    # Execute exit if conditions met
                    if should_exit:
                        exit_result = self.close_position(symbol)
                        
                        if exit_result.get('status') == 'success':
                            results['closed'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'action': 'close_position',
                                'reason': exit_reason,
                                'pl_amount': exit_result.get('pl_amount', 0)
                            })
                            
                            logger.info(f"Closed position for {symbol} due to {exit_reason}")
                        else:
                            results['errors'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'action': 'close_position_failed',
                                'reason': exit_reason,
                                'error': exit_result.get('message', 'Unknown error')
                            })
                            
                            logger.error(f"Failed to close position for {symbol}: {exit_result}")
                
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'action': 'error',
                        'error': str(e)
                    })
                    
                    logger.error(f"Error managing position for {symbol}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error managing open positions: {e}")
            return {'status': 'error', 'message': str(e)}

    def run_trading_cycle(self, symbols: list) -> Dict:
        """
        Run a complete trading cycle for a list of symbols.
        
        Args:
            symbols (list): List of symbols to process
            
        Returns:
            Dict: Results of the trading cycle
        """
        cycle_results = {
            'processed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'details': []
        }
        
        try:
            logger.info(f"Starting trading cycle for {len(symbols)} symbols")
            
            # Ensure connection to broker
            if not self.ibkr.connected:
                if not self.connect_to_broker():
                    return {'status': 'error', 'message': 'Not connected to broker'}
            
            # First manage existing positions
            position_results = self.manage_open_positions()
            cycle_results['positions_managed'] = position_results.get('monitored', 0)
            cycle_results['positions_updated'] = position_results.get('updated', 0)
            cycle_results['positions_closed'] = position_results.get('closed', 0)
            
            # Process each symbol
            for symbol in symbols:
                cycle_results['processed'] += 1
                
                try:
                    # Process trading signals
                    signal_result = self.process_trading_signals(symbol)
                    cycle_results['signals_generated'] += 1
                    
                    # Track successful trades
                    if signal_result.get('status') == 'success' and 'execute' in str(signal_result.get('message', '')):
                        cycle_results['trades_executed'] += 1
                        
                    cycle_results['details'].append({
                        'symbol': symbol,
                        'status': signal_result.get('status'),
                        'action': signal_result.get('message'),
                        'signal': signal_result.get('signal'),
                        'strength': signal_result.get('strength', 0)
                    })
                    
                except Exception as e:
                    cycle_results['errors'] += 1
                    cycle_results['details'].append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
                    
                    logger.error(f"Error processing {symbol} in trading cycle: {e}")
            
            logger.info(f"Completed trading cycle: processed {cycle_results['processed']} symbols, "
                    f"executed {cycle_results['trades_executed']} trades")
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the trading strategy.
        
        Returns:
            Dict: Performance metrics
        """
        try:
            # Calculate metrics from trade history
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'average_profit': 0,
                    'average_loss': 0,
                    'profit_factor': 0,
                    'total_pnl': 0
                }
            
            total_trades = len(self.trade_history)
            profitable_trades = [t for t in self.trade_history if t.get('pl_amount', 0) > 0]
            losing_trades = [t for t in self.trade_history if t.get('pl_amount', 0) <= 0]
            
            win_count = len(profitable_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            total_profit = sum(t.get('pl_amount', 0) for t in profitable_trades)
            total_loss = sum(abs(t.get('pl_amount', 0)) for t in losing_trades)
            
            average_profit = total_profit / win_count if win_count > 0 else 0
            average_loss = total_loss / loss_count if loss_count > 0 else 0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            total_pnl = total_profit - total_loss
            
            # Calculate Sharpe ratio if we have enough data
            if len(self.trade_history) >= 30:
                daily_returns = []
                current_day = None
                day_pnl = 0
                
                # Group trades by day
                sorted_trades = sorted(self.trade_history, key=lambda x: x.get('exit_time', ''))
                
                for trade in sorted_trades:
                    if not trade.get('exit_time'):
                        continue
                    
                    exit_time = trade.get('exit_time', '')[:10]  # Get just the date part
                    
                    if current_day is None:
                        current_day = exit_time
                        day_pnl = trade.get('pl_amount', 0)
                    elif current_day == exit_time:
                        day_pnl += trade.get('pl_amount', 0)
                    else:
                        daily_returns.append(day_pnl)
                        current_day = exit_time
                        day_pnl = trade.get('pl_amount', 0)
                
                # Add the last day
                if current_day is not None:
                    daily_returns.append(day_pnl)
                
                # Calculate Sharpe ratio
                if daily_returns:
                    mean_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns) if len(daily_returns) > 1 else 1
                    risk_free_rate = self.config.get('risk_management', {}).get('risk_free_rate', 0.03) / 252  # Daily risk-free rate
                    
                    sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            cumulative_pnl = 0
            peak = 0
            drawdown = 0
            max_drawdown = 0
            
            for trade in self.trade_history:
                pnl = trade.get('pl_amount', 0)
                cumulative_pnl += pnl
                
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                    drawdown = 0
                else:
                    drawdown = peak - cumulative_pnl
                    max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'average_profit': average_profit,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'risk_reward_ratio': average_profit / average_loss if average_loss > 0 else float('inf')
            }
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'status': 'error', 'message': str(e)}
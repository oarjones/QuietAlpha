"""
XGBoost Portfolio Manager Implementation

This module provides a concrete implementation of Portfolio Manager using XGBoost
for market scanning and symbol selection.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import datetime
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from portfolio_manager.base import PortfolioManager
from data.processing import calculate_technical_indicators, normalize_indicators
from utils.data_utils import load_config

logger = logging.getLogger(__name__)

class XGBoostPortfolioManager(PortfolioManager):
    """
    Portfolio Manager implementation using XGBoost for market scanning and symbol selection.
    This implementation extends the base PortfolioManager with specialized ML models.
    """
    
    def __init__(self, config_path: str = None, ibkr_interface = None):
        """
        Initialize XGBoost Portfolio Manager.
        
        Args:
            config_path (str): Path to configuration file
            ibkr_interface: IBKR interface instance
        """
        super().__init__(config_path, ibkr_interface)
        
        # XGBoost specific configurations
        self.xgb_config = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Override base model path
        self.model_path = os.path.join('models', 'xgboost_portfolio')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize feature scaling
        self.scaler = StandardScaler()
        self.feature_scaler_path = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Load scaler if available
        self._load_scaler()
        
    def _load_scaler(self):
        """Load feature scaler if available."""
        try:
            if os.path.exists(self.feature_scaler_path):
                self.scaler = joblib.load(self.feature_scaler_path)
                logger.info("Loaded feature scaler")
        except Exception as e:
            logger.warning(f"Error loading feature scaler: {e}")
    
    def train_xgboost_models(self, symbols: List[str] = None, lookback_days: int = 180) -> Dict:
        """
        Train XGBoost models for symbol selection.
        
        Args:
            symbols (List[str], optional): Symbols to use for training
            lookback_days (int): Days of historical data to use
            
        Returns:
            Dict: Training results
        """
        try:
            logger.info("Starting XGBoost model training for Portfolio Manager")
            
            # Use default symbols or scan market if not provided
            if not symbols:
                symbols = self.config.get('data', {}).get('symbols', [])
                if not symbols:
                    symbols = self.scan_market()[:30]  # Use top 30 symbols
            
            if not symbols:
                logger.error("No symbols available for training")
                return {'status': 'error', 'message': 'No symbols available for training'}
            
            # Collect features and labels
            features_list = []
            labels_list = []
            
            for symbol in symbols:
                logger.info(f"Processing training data for {symbol}")
                
                # Fetch historical data
                df = self.fetch_symbol_data(symbol, lookback_days=lookback_days)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # Process data into training examples
                # For each day, we'll create a training example with features
                # and a label based on future performance
                for i in range(20, len(df) - 10):  # Need history for features and future for labels
                    # Extract features
                    features = self._extract_advanced_features(df.iloc[:i])
                    
                    # Create label based on future 5-day return
                    future_return = df.iloc[i+5]['close'] / df.iloc[i]['close'] - 1
                    
                    # Binary classification: 1 if return > 1.5%, 0 otherwise
                    # This threshold can be adjusted based on strategy
                    label = 1 if future_return > 0.015 else 0
                    
                    features_list.append(features)
                    labels_list.append(label)
            
            if not features_list:
                logger.error("No training examples collected")
                return {'status': 'error', 'message': 'No training examples collected'}
                
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Save scaler
            joblib.dump(self.scaler, self.feature_scaler_path)
            
            # Prepare DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            
            # Train model
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
            
            xgb_model = xgb.train(
                params=self.xgb_config,
                dtrain=dtrain,
                num_boost_round=200,
                evals=watchlist,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Save the model
            model_path = os.path.join(self.model_path, 'market_scanner.model')
            xgb_model.save_model(model_path)
            self.models['market_scanner'] = xgb_model
            
            # Get feature importance
            importance = xgb_model.get_score(importance_type='gain')
            total = sum(importance.values())
            normalized_importance = {k: v/total for k, v in importance.items()}
            self.feature_importance = normalized_importance
            
            # Evaluate on validation set
            y_pred = xgb_model.predict(dval)
            y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
            
            # Calculate accuracy
            accuracy = sum(y_pred_binary[i] == y_val[i] for i in range(len(y_val))) / len(y_val)
            
            # Log results
            logger.info(f"XGBoost model trained with accuracy: {accuracy:.4f}")
            logger.info(f"Class distribution - Positives: {sum(y)}, Negatives: {len(y) - sum(y)}")
            
            # Return training results
            return {
                'status': 'success',
                'accuracy': accuracy,
                'samples': len(X),
                'positive_samples': int(sum(y)),
                'negative_samples': int(len(y) - sum(y)),
                'top_features': sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost models: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _extract_advanced_features(self, data: pd.DataFrame) -> List:
        """
        Extract advanced features for XGBoost model.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            List: Feature values
        """
        # Use only the most recent data for features
        window = data.iloc[-20:]  # Use last 20 periods
        
        if len(window) < 20:
            # Pad with duplicate of first row if not enough data
            padding = pd.concat([window.iloc[[0]] for _ in range(20 - len(window))], ignore_index=True)
            window = pd.concat([padding, window], ignore_index=True)
        
        # Get latest row for current values
        latest = window.iloc[-1]
        
        features = []
        
        # 1. Price-based features
        # Normalized closing price (compared to recent range)
        price_min = window['close'].min()
        price_max = window['close'].max()
        price_range = price_max - price_min
        if price_range > 0:
            norm_price = (latest['close'] - price_min) / price_range
        else:
            norm_price = 0.5
        features.append(norm_price)
        
        # Price relative to moving averages
        features.append(latest['close'] / latest['SMA_34'] if 'SMA_34' in latest else 1.0)
        features.append(latest['close'] / latest['EMA_21'] if 'EMA_21' in latest else 1.0)
        features.append(latest['close'] / latest['EMA_8'] if 'EMA_8' in latest else 1.0)
        
        # 2. Trend-based features
        features.append(latest.get('TrendStrength_norm', 0.5))
        features.append(latest.get('ADX_14_norm', 0.5))
        
        # Direction of moving averages
        ema8_trend = window['EMA_8'].iloc[-1] / window['EMA_8'].iloc[-5] - 1 if 'EMA_8' in window else 0
        ema21_trend = window['EMA_21'].iloc[-1] / window['EMA_21'].iloc[-5] - 1 if 'EMA_21' in window else 0
        features.append(ema8_trend)
        features.append(ema21_trend)
        
        # 3. Momentum features
        features.append(latest.get('RSI_14_norm', 0.5))
        features.append(latest.get('MACD_line', 0))
        features.append(latest.get('MACD_signal', 0))
        features.append(latest.get('MACD_diff', 0))
        
        # Stochastic
        features.append(latest.get('Stoch_k', 50) / 100)
        features.append(latest.get('Stoch_d', 50) / 100)
        
        # 4. Volatility features
        features.append(latest.get('ATR_14_norm', 0.5))
        features.append(latest.get('BB_Width_norm', 0.5))
        
        # Bollinger position
        if all(x in latest for x in ['close', 'BB_Low', 'BB_High']):
            bb_position = (latest['close'] - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low']) if latest['BB_High'] != latest['BB_Low'] else 0.5
            features.append(bb_position)
        else:
            features.append(0.5)
        
        # 5. Volume features
        features.append(latest.get('volume_norm', 0.5))
        
        # Volume trend
        vol_trend = window['volume'].iloc[-5:].mean() / window['volume'].iloc[-10:-5].mean() if 'volume' in window else 1.0
        features.append(vol_trend)
        
        # 6. Price patterns
        # Support/resistance proximity
        if 'high_20d' in latest and 'low_20d' in latest:
            dist_to_high = (latest['high_20d'] - latest['close']) / latest['close'] if latest['close'] > 0 else 0
            dist_to_low = (latest['close'] - latest['low_20d']) / latest['close'] if latest['close'] > 0 else 0
            features.append(dist_to_high)
            features.append(dist_to_low)
        else:
            features.append(0)
            features.append(0)
        
        # 7. Return-based features
        # Recent returns
        returns_1d = window['close'].iloc[-1] / window['close'].iloc[-2] - 1 if len(window) > 1 else 0
        returns_5d = window['close'].iloc[-1] / window['close'].iloc[-6] - 1 if len(window) > 5 else 0
        returns_10d = window['close'].iloc[-1] / window['close'].iloc[-11] - 1 if len(window) > 10 else 0
        features.append(returns_1d)
        features.append(returns_5d)
        features.append(returns_10d)
        
        return features
    
    def _predict_symbol_potential(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Predict the potential of a symbol using XGBoost model.
        
        Args:
            symbol (str): Symbol to analyze
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.fetch_symbol_data(symbol)
                
            if data.empty:
                return {
                    'symbol': symbol,
                    'score': 0,
                    'probability': 0,
                    'recommendation': 'insufficient_data'
                }
            
            # Check if model is available
            if 'market_scanner' not in self.models:
                logger.warning("XGBoost model not available, falling back to technical rules")
                return super().analyze_symbol(symbol, data)
            
            # Extract features
            features = self._extract_advanced_features(data)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict with model
            dtest = xgb.DMatrix(features_scaled)
            probability = float(self.models['market_scanner'].predict(dtest)[0])
            
            # Calculate score (0-100)
            score = int(probability * 100)
            
            # Determine recommendation
            recommendation = 'neutral'
            if score >= 75:
                recommendation = 'strong_buy'
            elif score >= 60:
                recommendation = 'buy'
            elif score <= 25:
                recommendation = 'avoid'
            elif score <= 40:
                recommendation = 'reduce'
            
            return {
                'symbol': symbol,
                'score': score,
                'probability': probability,
                'recommendation': recommendation,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting potential for {symbol}: {e}")
            # Fall back to base analysis
            return super().analyze_symbol(symbol, data)
    
    def analyze_symbol(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Analyze a symbol using XGBoost prediction model.
        
        This overrides the base implementation to use our XGBoost model.
        
        Args:
            symbol (str): Symbol to analyze
            data (pd.DataFrame): Historical data (optional)
            
        Returns:
            Dict: Analysis results
        """
        return self._predict_symbol_potential(symbol, data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the XGBoost model.
        
        Returns:
            Dict[str, float]: Feature importance
        """
        return self.feature_importance
    
    def adaptive_market_scan(self, min_symbols: int = 10, max_symbols: int = 30) -> List[str]:
        """
        Perform an adaptive market scan based on current conditions.
        
        This method goes beyond the basic scan by adapting criteria to market conditions.
        
        Args:
            min_symbols (int): Minimum symbols to return
            max_symbols (int): Maximum symbols to return
            
        Returns:
            List[str]: Selected symbols
        """
        try:
            logger.info("Performing adaptive market scan")
            
            # Get market sectors performance to guide our scan
            sector_performance = self.get_market_sectors_performance()
            
            # Determine top and bottom sectors
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            top_sectors = [s[0] for s in sorted_sectors[:3]]
            
            logger.info(f"Top sectors for scanning: {top_sectors}")
            
            # Perform basic scan to get initial candidates
            base_candidates = self.scan_market()
            
            if not base_candidates:
                logger.warning("No candidates from base scan")
                return []
            
            # Analyze each candidate symbol
            analyzed_symbols = []
            
            for symbol in base_candidates:
                analysis = self.analyze_symbol(symbol)
                
                # Add sector information if available (in real implementation, this would be fetched)
                # For now, we'll assign random sectors for demonstration
                import random
                analysis['sector'] = random.choice(list(sector_performance.keys()))
                
                analyzed_symbols.append(analysis)
            
            # Boost scores for symbols in top sectors
            for analysis in analyzed_symbols:
                if analysis['sector'] in top_sectors:
                    analysis['score'] = min(100, analysis['score'] * 1.2)
            
            # Sort by score
            sorted_symbols = sorted(analyzed_symbols, key=lambda x: x.get('score', 0), reverse=True)
            
            # Get top symbols, ensuring diversity
            selected_symbols = []
            selected_sectors = set()
            
            # First, add top symbols regardless of sector (up to 50% of max)
            top_count = max_symbols // 2
            for analysis in sorted_symbols[:top_count]:
                selected_symbols.append(analysis['symbol'])
                selected_sectors.add(analysis['sector'])
            
            # Then, add more symbols while ensuring sector diversity
            remaining = set(sorted_symbols[top_count:])
            
            # Try to include at least one from each sector
            all_sectors = set(sector_performance.keys())
            missing_sectors = all_sectors - selected_sectors
            
            for sector in missing_sectors:
                sector_symbols = [s for s in remaining if s['sector'] == sector and s['score'] >= 50]
                if sector_symbols:
                    best_symbol = max(sector_symbols, key=lambda x: x.get('score', 0))
                    selected_symbols.append(best_symbol['symbol'])
                    remaining.remove(best_symbol)
                    
                    if len(selected_symbols) >= max_symbols:
                        break
            
            # Fill any remaining slots with best remaining symbols
            sorted_remaining = sorted(remaining, key=lambda x: x.get('score', 0), reverse=True)
            for s in sorted_remaining:
                if len(selected_symbols) >= max_symbols:
                    break
                selected_symbols.append(s['symbol'])
            
            logger.info(f"Adaptive scan selected {len(selected_symbols)} symbols")
            return selected_symbols
            
        except Exception as e:
            logger.error(f"Error in adaptive market scan: {e}")
            # Fall back to basic scan
            return self.scan_market()[:max_symbols]
    
    def select_portfolio(self, num_symbols: int = None) -> Dict[str, float]:
        """
        Select symbols for the portfolio using the XGBoost model and advanced allocation.
        
        This overrides the base implementation to use adaptive scanning and smarter allocation.
        
        Args:
            num_symbols (int, optional): Number of symbols to select
            
        Returns:
            Dict[str, float]: Symbol -> allocation mapping
        """
        try:
            if num_symbols is None:
                num_symbols = self.max_symbols
            
            logger.info(f"Selecting portfolio with up to {num_symbols} symbols using XGBoost")
            
            # Use adaptive market scan
            candidate_symbols = self.adaptive_market_scan(
                min_symbols=num_symbols, 
                max_symbols=num_symbols * 2
            )
            
            if not candidate_symbols:
                logger.warning("No candidate symbols found")
                return {}
            
            # Analyze each symbol
            symbol_analyses = {}
            for symbol in candidate_symbols:
                analysis = self.analyze_symbol(symbol)
                symbol_analyses[symbol] = analysis
            
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
            
            # Limit to requested number
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
            
            logger.info(f"Selected {len(portfolio)} symbols for portfolio using XGBoost")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error selecting portfolio with XGBoost: {e}")
            # Fall back to base implementation
            return super().select_portfolio(num_symbols)
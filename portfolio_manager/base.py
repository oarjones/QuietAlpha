"""
Portfolio Manager Base Class

This module provides the base implementation for the Portfolio Manager component, 
which is responsible for scanning the market and selecting the best symbols to invest in.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import datetime
import time
import joblib
from threading import Lock

from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators
from utils.data_utils import load_config

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Portfolio Manager is responsible for scanning the market and selecting
    the best symbols to invest in based on risk/reward profiles.
    """
    
    def __init__(self, config_path: str = None, ibkr_interface: IBKRInterface = None):
        """
        Initialize Portfolio Manager.
        
        Args:
            config_path (str): Path to configuration file
            ibkr_interface (IBKRInterface): IBKR interface instance
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else {}
        self.portfolio_config = self.config.get('portfolio_manager', {})
        
        # Set up IBKR interface
        self.ibkr = ibkr_interface
        if self.ibkr is None:
            host = self.config.get('ibkr', {}).get('host', '127.0.0.1')
            port = self.config.get('ibkr', {}).get('port', 7497)
            client_id = self.config.get('ibkr', {}).get('client_id', 1)
            self.ibkr = IBKRInterface(host=host, port=port, client_id=client_id)
            
        # Portfolio properties
        self.max_symbols = self.portfolio_config.get('max_symbols', 5)
        self.max_sector_exposure = self.portfolio_config.get('max_sector_exposure', 0.3)
        self.risk_allocation = self.portfolio_config.get('risk_allocation', 0.02)
        self.market_scan_interval = self.portfolio_config.get('market_scan_interval', 3600)
        
        # Market universe - these could be sectors, indices, or specific stock lists
        self.market_universes = self.portfolio_config.get('market_universes', ['SP500', 'NASDAQ100'])
        
        # Portfolio state
        self.current_portfolio = {}  # Symbol -> allocation
        self.symbol_metadata = {}    # Symbol -> metadata (sector, beta, etc.)
        self.symbol_rankings = {}    # Symbol -> rank score
        
        # ML models
        self.models = {}
        self.model_path = os.path.join('models', 'portfolio_manager')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Concurrency control
        self.portfolio_lock = Lock()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load ML models for symbol selection."""
        # Try to load pre-trained models if they exist
        trend_model_path = os.path.join(self.model_path, 'trend_classifier.pkl')
        ranking_model_path = os.path.join(self.model_path, 'symbol_ranker.pkl')
        
        try:
            if os.path.exists(trend_model_path):
                self.models['trend_classifier'] = joblib.load(trend_model_path)
                logger.info("Loaded trend classifier model")
            
            if os.path.exists(ranking_model_path):
                self.models['symbol_ranker'] = joblib.load(ranking_model_path)
                logger.info("Loaded symbol ranker model")
        except Exception as e:
            logger.warning(f"Error loading models: {e}. Will train new models when needed.")
    
    def scan_market(self) -> List[str]:
        """
        Scan the market to find potential symbols to analyze.
        
        Returns:
            List[str]: List of symbols to analyze
        """
        if not self.ibkr.connected:
            if not self.ibkr.connect():
                logger.error("Failed to connect to IBKR for market scan")
                return []
        
        all_symbols = []
        
        try:
            logger.info("Scanning market for potential symbols...")
            
            # Scan different market segments based on configuration
            for universe in self.market_universes:
                # Use IBKR's market scanner to find candidates
                # This could be enhanced with more sophisticated criteria
                
                if universe == 'SP500':
                    # Example: Scan S&P 500 stocks with momentum
                    scanner_params = {
                        'instrument': 'STK',
                        'locationCode': 'STK.US.MAJOR',
                        'scanCode': 'TOP_PERC_GAIN',
                        'numberOfRows': 50,
                    }
                    
                elif universe == 'NASDAQ100':
                    # Example: Scan NASDAQ 100 stocks with high volume
                    scanner_params = {
                        'instrument': 'STK',
                        'locationCode': 'STK.US.NASDAQ',
                        'scanCode': 'HIGH_VS_13W_HL',
                        'numberOfRows': 50,
                    }
                    
                else:
                    # Custom universe
                    scanner_params = {
                        'instrument': 'STK',
                        'locationCode': 'STK.US',
                        'scanCode': 'HOT_BY_VOLUME',
                        'numberOfRows': 30,
                    }
                
                # In a real implementation, use self.ibkr.ib.reqScannerData()
                # Here we'll simulate scanner results for simplicity
                # This would be replaced with actual API calls
                
                # Simulated scan results for development purposes
                import random
                sample_symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'WMT',
                    'JNJ', 'JPM', 'BAC', 'DIS', 'NFLX', 'INTC', 'AMD', 'COST', 'PEP',
                    'KO', 'PFE', 'T', 'VZ', 'CSCO', 'IBM', 'ORCL', 'ADBE', 'PYPL'
                ]
                
                universe_symbols = random.sample(sample_symbols, min(15, len(sample_symbols)))
                logger.info(f"Found {len(universe_symbols)} symbols in {universe}")
                
                all_symbols.extend(universe_symbols)
            
            # Remove duplicates and limit to a reasonable number to analyze
            all_symbols = list(set(all_symbols))
            
            # In a real implementation, you would fetch additional metadata
            # like sector, market cap, etc. for filtering
            
            logger.info(f"Total unique symbols found: {len(all_symbols)}")
            return all_symbols
            
        except Exception as e:
            logger.error(f"Error during market scan: {e}")
            return []
    
    def fetch_symbol_data(self, symbol: str, lookback_days: int = 90) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol (str): Symbol to fetch data for
            lookback_days (int): Number of days of history to fetch
            
        Returns:
            pd.DataFrame: Historical data with technical indicators
        """
        try:
            logger.info(f"Fetching data for {symbol}, lookback: {lookback_days} days")
            
            # Determine duration string based on lookback
            if lookback_days <= 7:
                duration = f"{lookback_days} D"
            elif lookback_days <= 30:
                duration = f"{lookback_days // 7 + 1} W"
            else:
                duration = f"{lookback_days // 30 + 1} M"
            
            # Fetch hourly data
            df = self.ibkr.get_historical_data(
                symbol=symbol,
                duration=duration,
                bar_size="1 hour",
                what_to_show="TRADES",
                use_rth=True
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df, include_all=True)
            
            # Calculate trend indicator
            df = calculate_trend_indicator(df)
            
            # Normalize indicators
            df = normalize_indicators(df)
            
            logger.info(f"Successfully processed data for {symbol}, rows: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching and processing data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Analyze a symbol to determine if it should be included in the portfolio.
        
        Args:
            symbol (str): Symbol to analyze
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.fetch_symbol_data(symbol)
                
            if data.empty:
                return {
                    'symbol': symbol,
                    'score': 0,
                    'recommendation': 'avoid',
                    'reason': 'insufficient_data'
                }
            
            # Extract features for ML prediction
            features = self._extract_features(data)
            
            # If we don't have a trained model yet, use technical rules
            if 'trend_classifier' not in self.models:
                return self._analyze_technical_rules(symbol, data, features)
            
            # Otherwise use ML model for prediction
            return self._analyze_ml_prediction(symbol, data, features)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'score': 0,
                'recommendation': 'avoid',
                'reason': 'analysis_error',
                'error': str(e)
            }
    
    def _extract_features(self, data: pd.DataFrame) -> Dict:
        """
        Extract features for ML model from data.
        
        Args:
            data (pd.DataFrame): Processed data with indicators
            
        Returns:
            Dict: Features dictionary
        """
        # Use the most recent data point (last row)
        latest = data.iloc[-1]
        
        # Extract key features (focusing on normalized versions)
        features = {}
        
        # Trend indicators
        features['trend_strength'] = latest.get('TrendStrength_norm', 0.5)
        features['adx'] = latest.get('ADX_14_norm', 0.5)
        features['macd_diff'] = latest.get('MACD_diff_norm', 0)
        
        # Momentum indicators
        features['rsi'] = latest.get('RSI_14_norm', 0.5)
        features['stoch_k'] = latest.get('Stoch_k_norm', 0.5)
        features['stoch_d'] = latest.get('Stoch_d_norm', 0.5)
        
        # Volatility indicators
        features['atr'] = latest.get('ATR_14_norm', 0.5)
        features['bb_width'] = latest.get('BB_Width_norm', 0.5)
        
        # Price position indicators
        features['price_vs_sma50'] = 1 if latest.get('close') > latest.get('SMA_50', 0) else 0
        features['price_vs_ema21'] = 1 if latest.get('close') > latest.get('EMA_21', 0) else 0
        
        # Calculate recent returns
        if len(data) >= 5:
            features['return_5d'] = (latest.get('close') / data.iloc[-5]['close'] - 1)
        else:
            features['return_5d'] = 0
            
        if len(data) >= 20:
            features['return_20d'] = (latest.get('close') / data.iloc[-20]['close'] - 1)
        else:
            features['return_20d'] = 0
        
        # Volume indicators
        features['obv_norm'] = latest.get('OBV_norm', 0.5)
        features['volume_norm'] = latest.get('volume_norm', 0.5)
        
        # Add market relative metrics if available
        # This would require market index data to be calculated
        
        return features
    
    def _analyze_technical_rules(self, symbol: str, data: pd.DataFrame, features: Dict) -> Dict:
        """
        Analyze symbol using technical rules (when ML model is not available).
        
        Args:
            symbol (str): Symbol being analyzed
            data (pd.DataFrame): Historical data
            features (Dict): Extracted features
            
        Returns:
            Dict: Analysis results
        """
        # Implement a rule-based approach
        score = 0
        reasons = []
        
        # Trend strength (0-1)
        trend_strength = features['trend_strength']
        if trend_strength > 0.7:
            score += 30
            reasons.append('strong_trend')
        elif trend_strength > 0.5:
            score += 15
            reasons.append('moderate_trend')
        
        # Momentum
        rsi = features['rsi'] * 100  # Convert from normalized
        if 40 < rsi < 60:
            score += 10
            reasons.append('neutral_momentum')
        elif 60 <= rsi < 70:
            score += 15
            reasons.append('positive_momentum')
        elif 30 <= rsi <= 40:
            score += 5
            reasons.append('recovering_momentum')
        elif rsi >= 70:
            score -= 5
            reasons.append('overbought')
        elif rsi <= 30:
            score -= 5
            reasons.append('oversold')
        
        # Moving average alignment
        if features['price_vs_sma50'] and features['price_vs_ema21']:
            score += 20
            reasons.append('price_above_ma')
        elif not features['price_vs_sma50'] and not features['price_vs_ema21']:
            score -= 10
            reasons.append('price_below_ma')
        
        # Recent performance
        if features['return_5d'] > 0.03:  # 3% in 5 days
            score += 10
            reasons.append('strong_recent_performance')
        elif features['return_5d'] < -0.03:
            score -= 5
            reasons.append('weak_recent_performance')
        
        # Volatility
        if features['atr'] < 0.4:  # Low volatility
            score += 5
            reasons.append('low_volatility')
        elif features['atr'] > 0.7:  # High volatility
            score -= 5
            reasons.append('high_volatility')
        
        # Normalize score to 0-100
        score = max(0, min(100, score + 50))  # Base of 50, capped at 0-100
        
        # Determine recommendation
        recommendation = 'neutral'
        if score >= 70:
            recommendation = 'strong_buy'
        elif score >= 60:
            recommendation = 'buy'
        elif score <= 30:
            recommendation = 'avoid'
        elif score <= 40:
            recommendation = 'reduce'
        
        return {
            'symbol': symbol,
            'score': score,
            'recommendation': recommendation,
            'reasons': reasons,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def _analyze_ml_prediction(self, symbol: str, data: pd.DataFrame, features: Dict) -> Dict:
        """
        Analyze symbol using ML prediction model.
        
        Args:
            symbol (str): Symbol being analyzed
            data (pd.DataFrame): Historical data
            features (Dict): Extracted features
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Prepare features for model
            feature_list = list(features.values())
            feature_array = np.array(feature_list).reshape(1, -1)
            
            # Make prediction
            model = self.models['trend_classifier']
            prediction_probas = model.predict_proba(feature_array)[0]
            prediction_class = model.predict(feature_array)[0]
            
            # Get class mapping and confidence
            class_mapping = {0: 'downtrend', 1: 'neutral', 2: 'uptrend'}
            predicted_trend = class_mapping.get(prediction_class, 'unknown')
            
            # Calculate confidence score (0-100)
            confidence = int(np.max(prediction_probas) * 100)
            
            # Calculate overall score
            if predicted_trend == 'uptrend':
                score = confidence
            elif predicted_trend == 'neutral':
                score = 50
            else:  # downtrend
                score = 100 - confidence
            
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
            
            # Include feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = list(features.keys())
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    feature_importance[name] = float(importance)
            
            return {
                'symbol': symbol,
                'score': score,
                'prediction': predicted_trend,
                'confidence': confidence,
                'recommendation': recommendation,
                'feature_importance': feature_importance,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ML prediction for {symbol}: {e}")
            # Fall back to technical rules if ML fails
            return self._analyze_technical_rules(symbol, data, features)
    
    def train_models(self, symbols: List[str] = None) -> Dict:
        """
        Train ML models for symbol selection and ranking.
        
        Args:
            symbols (List[str], optional): Symbols to use for training
            
        Returns:
            Dict: Training results
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
        import xgboost as xgb
        
        try:
            logger.info("Starting model training for Portfolio Manager")
            
            # If no symbols provided, use a default list or scan the market
            if not symbols:
                symbols = self.config.get('data', {}).get('symbols', [])
                if not symbols:
                    symbols = self.scan_market()[:20]  # Use top 20 symbols
            
            if not symbols:
                logger.error("No symbols available for training")
                return {'status': 'error', 'message': 'No symbols available for training'}
            
            # Collect training data
            training_data = []
            labels = []
            
            for symbol in symbols:
                # Fetch and process historical data
                data = self.fetch_symbol_data(symbol, lookback_days=180)
                
                if data.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # For each data point, extract features and create label
                for i in range(20, len(data)):
                    # Extract features from historical data
                    window = data.iloc[i-20:i]
                    features = self._extract_features_for_training(window)
                    
                    # Create label: -1 (down), 0 (sideways), 1 (up) based on future performance
                    if i+5 < len(data):  # Ensure we have future data
                        future_return = data.iloc[i+5]['close'] / data.iloc[i]['close'] - 1
                        
                        if future_return > 0.02:  # 2% gain = uptrend
                            label = 2  # uptrend
                        elif future_return < -0.02:  # 2% loss = downtrend
                            label = 0  # downtrend
                        else:
                            label = 1  # neutral/sideways
                        
                        training_data.append(features)
                        labels.append(label)
            
            if not training_data:
                logger.error("No training data collected")
                return {'status': 'error', 'message': 'No training data collected'}
            
            # Convert to numpy arrays
            X = np.array(training_data)
            y = np.array(labels)
            
            logger.info(f"Collected {len(X)} training samples with {X.shape[1]} features")
            
            # Train trend classifier (RandomForest)
            trend_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            trend_classifier.fit(X, y)
            classifier_accuracy = trend_classifier.score(X, y)
            logger.info(f"Trend classifier trained with accuracy: {classifier_accuracy:.4f}")
            
            # Save the model
            self.models['trend_classifier'] = trend_classifier
            joblib.dump(trend_classifier, os.path.join(self.model_path, 'trend_classifier.pkl'))
            
            # Train symbol ranker (XGBoost)
            # For ranking, we convert class labels to numeric scores
            ranking_labels = np.where(y == 2, 1.0,  # uptrend -> 1.0
                              np.where(y == 1, 0.5,  # neutral -> 0.5
                                       0.0))  # downtrend -> 0.0
            
            ranker = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            ranker.fit(X, ranking_labels)
            ranker_mse = np.mean((ranker.predict(X) - ranking_labels) ** 2)
            logger.info(f"Symbol ranker trained with MSE: {ranker_mse:.4f}")
            
            # Save the model
            self.models['symbol_ranker'] = ranker
            joblib.dump(ranker, os.path.join(self.model_path, 'symbol_ranker.pkl'))
            
            return {
                'status': 'success',
                'classifier_accuracy': classifier_accuracy,
                'ranker_mse': ranker_mse,
                'samples': len(X),
                'features': X.shape[1],
                'class_distribution': {
                    'downtrend': int(np.sum(y == 0)),
                    'neutral': int(np.sum(y == 1)),
                    'uptrend': int(np.sum(y == 2))
                }
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _extract_features_for_training(self, data: pd.DataFrame) -> List:
        """
        Extract features for training from a data window.
        
        Args:
            data (pd.DataFrame): Window of historical data
            
        Returns:
            List: Feature values
        """
        # Use the last row for current values
        latest = data.iloc[-1]
        
        # Create feature list in consistent order for ML
        features = []
        
        # Trend indicators
        features.append(latest.get('TrendStrength_norm', 0.5))
        features.append(latest.get('ADX_14_norm', 0.5))
        features.append(latest.get('MACD_diff_norm', 0))
        
        # Momentum indicators
        features.append(latest.get('RSI_14_norm', 0.5))
        features.append(latest.get('Stoch_k_norm', 0.5))
        features.append(latest.get('Stoch_d_norm', 0.5))
        
        # Volatility indicators
        features.append(latest.get('ATR_14_norm', 0.5))
        features.append(latest.get('BB_Width_norm', 0.5))
        
        # Price position indicators
        features.append(1 if latest.get('close') > latest.get('SMA_50', 0) else 0)
        features.append(1 if latest.get('close') > latest.get('EMA_21', 0) else 0)
        
        # Recent returns
        if len(data) >= 5:
            features.append(latest.get('close') / data.iloc[0]['close'] - 1)
        else:
            features.append(0)
        
        # Volume indicators
        features.append(latest.get('OBV_norm', 0.5))
        features.append(latest.get('volume_norm', 0.5))
        
        # Add price pattern features
        if 'dist_to_resistance_norm' in latest:
            features.append(latest['dist_to_resistance_norm'])
            features.append(latest['dist_to_support_norm'])
            features.append(latest['channel_position'])
        else:
            features.extend([0.5, 0.5, 0.5])  # Default values
        
        # Ensure consistent length
        expected_length = 15  # Adjust as needed
        if len(features) < expected_length:
            features.extend([0] * (expected_length - len(features)))
        
        return features
    
    def select_portfolio(self, num_symbols: int = None) -> Dict[str, float]:
        """
        Select symbols for the portfolio and assign allocations.
        
        Args:
            num_symbols (int, optional): Number of symbols to select
            
        Returns:
            Dict[str, float]: Symbol -> allocation mapping
        """
        try:
            if num_symbols is None:
                num_symbols = self.max_symbols
            
            logger.info(f"Selecting portfolio with up to {num_symbols} symbols")
            
            # Scan the market for potential symbols
            candidate_symbols = self.scan_market()
            
            if not candidate_symbols:
                logger.warning("No candidate symbols found")
                return {}
            
            # Analyze each symbol and rank them
            symbol_analyses = {}
            for symbol in candidate_symbols:
                analysis = self.analyze_symbol(symbol)
                symbol_analyses[symbol] = analysis
                logger.info(f"Analysis for {symbol}: score={analysis.get('score', 0)}, "
                           f"recommendation={analysis.get('recommendation', 'unknown')}")
            
            # Filter symbols with positive recommendation
            positive_symbols = [
                s for s, a in symbol_analyses.items() 
                if a.get('recommendation') in ['buy', 'strong_buy']
            ]
            
            # If not enough positive symbols, include neutral ones
            if len(positive_symbols) < num_symbols:
                neutral_symbols = [
                    s for s, a in symbol_analyses.items()
                    if a.get('recommendation') == 'neutral'
                    and s not in positive_symbols
                ]
                
                additional_needed = num_symbols - len(positive_symbols)
                if neutral_symbols:
                    neutral_sorted = sorted(
                        neutral_symbols,
                        key=lambda s: symbol_analyses[s].get('score', 0),
                        reverse=True
                    )
                    positive_symbols.extend(neutral_sorted[:additional_needed])
            
            # Limit to requested number
            selected_symbols = positive_symbols[:num_symbols]
            
            if not selected_symbols:
                logger.warning("No suitable symbols found for portfolio")
                return {}
            
            # Calculate allocations (simple equal weighting for now)
            allocation_per_symbol = 1.0 / len(selected_symbols)
            
            portfolio = {symbol: allocation_per_symbol for symbol in selected_symbols}
            
            logger.info(f"Selected {len(portfolio)} symbols for portfolio")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error selecting portfolio: {e}")
            return {}
    
    def run_market_scan(self, interval: int = None) -> None:
        """
        Run continuous market scanning in a loop.
        
        Args:
            interval (int, optional): Scan interval in seconds
        """
        if interval is None:
            interval = self.market_scan_interval
            
        logger.info(f"Starting market scan loop with interval {interval} seconds")
        
        while True:
            try:
                update_result = self.update_portfolio()
                logger.info(f"Portfolio update: {update_result['status']}")
                
                # Sleep for the specified interval
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Market scan loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in market scan loop: {e}")
                # Sleep briefly before retrying
                time.sleep(60)
    
    def get_market_sectors_performance(self) -> Dict[str, float]:
        """
        Get performance of different market sectors.
        
        Returns:
            Dict[str, float]: Sector -> performance score mapping
        """
        try:
            logger.info("Getting market sectors performance")
            
            # In a production implementation, this would query sector ETFs or indices
            # through IBKR API to get relative performance
            
            # For demonstration, return simulated sector performance
            sectors = [
                'Technology', 'Healthcare', 'Finance', 'Consumer', 
                'Energy', 'Materials', 'Utilities', 'Real Estate',
                'Communication', 'Industrials'
            ]
            
            import random
            sector_performance = {}
            
            for sector in sectors:
                # Simulate sector performance between -10% and +10%
                performance = (random.random() * 20) - 10
                sector_performance[sector] = performance
            
            # Log top and bottom sectors
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            top_sectors = sorted_sectors[:3]
            bottom_sectors = sorted_sectors[-3:]
            
            logger.info(f"Top performing sectors: {top_sectors}")
            logger.info(f"Bottom performing sectors: {bottom_sectors}")
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return {}
    
    def get_portfolio_stats(self) -> Dict:
        """
        Get statistics about the current portfolio.
        
        Returns:
            Dict: Portfolio statistics
        """
        try:
            if not self.current_portfolio:
                return {
                    'status': 'warning',
                    'message': 'No active portfolio',
                    'symbols': 0,
                    'allocation': {}
                }
            
            # Gather portfolio statistics
            stats = {
                'status': 'success',
                'symbols': len(self.current_portfolio),
                'allocation': self.current_portfolio,
                'sectors': {},
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Sector allocation (using metadata if available)
            sector_allocation = {}
            for symbol, allocation in self.current_portfolio.items():
                # Get sector from metadata or use 'Unknown'
                sector = self.symbol_metadata.get(symbol, {}).get('sector', 'Unknown')
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                    
                sector_allocation[sector] += allocation
            
            stats['sectors'] = sector_allocation
            
            # Calculate diversification metrics
            # Herfindahl-Hirschman Index (HHI) - lower is more diversified
            hhi = sum([alloc ** 2 for alloc in self.current_portfolio.values()])
            stats['diversification_score'] = 1 - min(1, hhi)  # 0 = concentrated, 1 = diversified
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def update_portfolio(self) -> Dict:
        """
        Update the portfolio based on current market conditions.
        
        Returns:
            Dict: Update results
        """
        with self.portfolio_lock:
            try:
                logger.info("Updating portfolio...")
                
                # Get account summary
                account_summary = self.ibkr.get_account_summary()
                
                if not account_summary:
                    logger.error("Could not retrieve account summary")
                    return {
                        'status': 'error',
                        'message': 'Could not retrieve account summary'
                    }
                
                # Get current positions
                current_positions = self.ibkr.get_positions()
                current_symbols = {p['symbol']: p for p in current_positions}
                
                # Select optimal portfolio
                optimal_portfolio = self.select_portfolio()
                
                if not optimal_portfolio:
                    logger.warning("No optimal portfolio could be determined")
                    return {
                        'status': 'warning',
                        'message': 'No optimal portfolio could be determined'
                    }
                
                # Calculate positions to add, remove, or adjust
                symbols_to_add = [s for s in optimal_portfolio if s not in current_symbols]
                symbols_to_remove = [s for s in current_symbols if s not in optimal_portfolio]
                symbols_to_adjust = [s for s in optimal_portfolio if s in current_symbols]
                
                # Create portfolio update plan
                update_plan = {
                    'add': symbols_to_add,
                    'remove': symbols_to_remove,
                    'adjust': symbols_to_adjust,
                    'allocations': optimal_portfolio
                }
                
                # Update internal portfolio state
                self.current_portfolio = optimal_portfolio
                
                # Save metadata for symbols in portfolio
                for symbol in optimal_portfolio:
                    if symbol not in self.symbol_metadata:
                        # Fetch metadata like sector, market cap, etc.
                        # In a real implementation, this would pull data from IBKR
                        # Here we'll just use placeholder data
                        self.symbol_metadata[symbol] = {
                            'sector': 'Unknown',  # This would come from fundamental data
                            'market_cap': 0,
                            'beta': 1.0,
                            'last_analyzed': datetime.datetime.now().isoformat()
                        }
                
                logger.info(f"Portfolio update plan created: "
                           f"add {len(symbols_to_add)}, remove {len(symbols_to_remove)}, "
                           f"adjust {len(symbols_to_adjust)}")
                
                return {
                    'status': 'success',
                    'update_plan': update_plan,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error updating portfolio: {e}")
                return {
                    'status': 'error',
                    'message': str(e)
                }
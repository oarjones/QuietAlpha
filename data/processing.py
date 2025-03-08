"""
Data Processing Module

This module handles data preprocessing, including technical indicator calculation,
normalization, and feature engineering for machine learning models.
"""

import pandas as pd
import numpy as np
import os
import logging
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
from typing import List, Dict, Tuple, Union, Optional

logger = logging.getLogger(__name__)

def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a series using min-max scaling, values in [0,1].
    
    Args:
        series (pd.Series): Series to normalize
        
    Returns:
        pd.Series: Normalized series
    """
    if series.max() == series.min():
        return series - series.min()  # If constant
    return (series - series.min()) / (series.max() - series.min())

def calculate_technical_indicators(
    df: pd.DataFrame,
    date_col: str = 'datetime',
    close_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    open_col: str = 'open',
    volume_col: str = 'volume',
    include_all: bool = False
) -> pd.DataFrame:
    """
    Calculate technical indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        date_col (str): Date column name
        close_col (str): Close price column name
        high_col (str): High price column name
        low_col (str): Low price column name
        open_col (str): Open price column name
        volume_col (str): Volume column name
        include_all (bool): Whether to include all indicators or just the essential ones
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Verify required columns are present
    required_cols = [close_col, high_col, low_col, volume_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Required columns are missing: {missing}")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Make sure the index is properly set if date column exists
    if date_col in result.columns and not isinstance(result.index, pd.DatetimeIndex):
        result[date_col] = pd.to_datetime(result[date_col])
        result.set_index(date_col, inplace=True)
    
    # --- Essential Trend Indicators ---
    # Moving Averages
    result['SMA_34'] = ta.trend.sma_indicator(result[close_col], window=34)
    result['EMA_21'] = ta.trend.ema_indicator(result[close_col], window=21)
    result['EMA_8'] = ta.trend.ema_indicator(result[close_col], window=8)
    
    # MACD adapted for 1h timeframe
    result['MACD_line'] = ta.trend.macd(result[close_col], window_slow=21, window_fast=8)
    result['MACD_signal'] = ta.trend.macd_signal(result[close_col], window_slow=21, window_fast=8, window_sign=5)
    result['MACD_diff'] = ta.trend.macd_diff(result[close_col], window_slow=21, window_fast=8, window_sign=5)
    
    # ADX for trend strength
    result['ADX_14'] = ta.trend.adx(result[high_col], result[low_col], result[close_col], window=14)
    
    # --- Essential Momentum Indicators ---
    result['RSI_14'] = ta.momentum.rsi(result[close_col], window=14)
    
    stoch = ta.momentum.StochasticOscillator(
        result[high_col], 
        result[low_col], 
        result[close_col], 
        window=14, 
        smooth_window=3
    )
    result['Stoch_k'] = stoch.stoch()
    result['Stoch_d'] = stoch.stoch_signal()
    
    # --- Essential Volatility Indicators ---
    result['ATR_14'] = ta.volatility.average_true_range(
        result[high_col], result[low_col], result[close_col], window=14
    )
    
    bollinger = ta.volatility.BollingerBands(result[close_col], window=20, window_dev=2)
    result['BB_High'] = bollinger.bollinger_hband()
    result['BB_Low'] = bollinger.bollinger_lband()
    result['BB_Width'] = bollinger.bollinger_wband()
    
    # --- Essential Volume Indicators ---
    result['OBV'] = ta.volume.on_balance_volume(result[close_col], result[volume_col])
    
    # If include_all is True, add additional indicators
    if include_all:
        # --- Additional Trend Indicators ---
        result['SMA_50'] = ta.trend.sma_indicator(result[close_col], window=50)
        result['SMA_200'] = ta.trend.sma_indicator(result[close_col], window=200)
        result['EMA_50'] = ta.trend.ema_indicator(result[close_col], window=50)
        result['EMA_200'] = ta.trend.ema_indicator(result[close_col], window=200)
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(
            result[high_col], result[low_col], 
            window1=9, window2=26, window3=52
        )
        result['Ichimoku_a'] = ichimoku.ichimoku_a()
        result['Ichimoku_b'] = ichimoku.ichimoku_b()
        result['Ichimoku_base'] = ichimoku.ichimoku_base_line()
        result['Ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # --- Additional Momentum Indicators ---
        result['MOM_10'] = ta.momentum.roc(result[close_col], window=10)
        result['MOM_14'] = ta.momentum.roc(result[close_col], window=14)
        
        # Williams %R
        result['Williams_R_14'] = ta.momentum.williams_r(
            result[high_col], result[low_col], result[close_col], lbp=14
        )
        
        # --- Additional Volatility Indicators ---
        result['ATR_5'] = ta.volatility.average_true_range(
            result[high_col], result[low_col], result[close_col], window=5
        )
        
        keltner = ta.volatility.KeltnerChannel(
            result[high_col], result[low_col], result[close_col], window=20, window_atr=10
        )
        result['KC_High'] = keltner.keltner_channel_hband()
        result['KC_Low'] = keltner.keltner_channel_lband()
        result['KC_Width'] = keltner.keltner_channel_wband()
        
        # --- Additional Volume Indicators ---
        result['VWAP'] = ta.volume.volume_weighted_average_price(
            result[high_col], result[low_col], result[close_col], result[volume_col], window=14
        )
        
        result['CMF'] = ta.volume.chaikin_money_flow(
            result[high_col], result[low_col], result[close_col], result[volume_col], window=20
        )
        
        # Force Index
        result['Force_Index_13'] = ta.volume.force_index(
            result[close_col], result[volume_col], window=13
        )
    
    # Reset index if we set it earlier
    if date_col in df.columns and result.index.name == date_col:
        result.reset_index(inplace=True)
    
    # Fill NaN values that might be introduced by calculations
    result.bfill(inplace=True)
    
    return result

def calculate_trend_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a continuous trend indicator normalized between 0 and 1.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
    
    Returns:
        pd.DataFrame with new column 'TrendStrength_norm' where:
        - Values close to 1: Strong bullish trend
        - Values close to 0: Strong bearish trend
        - Values close to 0.5: Stable/sideways market
    """
    # Validate required columns exist
    required_cols = [
        'RSI_14', 'Stoch_k', 'EMA_8', 'EMA_21', 'SMA_34', 
        'BB_Low', 'BB_High', 'BB_Width', 'MACD_diff', 'ADX_14', 'close'
    ]
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for trend calculation: {missing}")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Directional Components
    trend_direction = (
        # Momentum
        0.3 * (result['RSI_14'] / 100) +  # Normalized RSI
        0.2 * (result['Stoch_k'] / 100) +  # Normalized Stochastic
        0.2 * np.clip(result['MACD_diff'], -1, 1) +  # Normalized MACD diff
        0.3 * np.sign(result['close'] - result['EMA_21'])  # Price vs EMA21
    )
    
    # Trend Strength
    trend_strength = (
        result['ADX_14'] / 100 +  # Normalized ADX
        result['BB_Width'] / result['BB_Width'].rolling(window=20).mean()  # Relative Bollinger width
    ) / 2
    
    # Price Position
    price_position = (
        # Position relative to moving averages
        0.4 * np.sign(result['close'] - result['EMA_8']) + 
        0.3 * np.sign(result['close'] - result['EMA_21']) +
        0.3 * np.sign(result['close'] - result['SMA_34'])
    )
    
    # Combine components
    raw_trend = (
        0.4 * trend_direction +
        0.3 * np.sign(price_position) * trend_strength +
        0.3 * (result['close'] - result['BB_Low']) / (result['BB_High'] - result['BB_Low'])  # Position in Bollinger
    )
    
    # Final normalization using sigmoid smoothed
    result['TrendStrength_norm'] = 1 / (1 + np.exp(-4 * raw_trend))  # Factor 4 for better distribution
    
    # Smoothing to reduce noise
    window_size = 3
    result['TrendStrength_norm'] = result['TrendStrength_norm'].rolling(
        window=window_size, center=True
    ).mean()
    
    # Fill NaN values
    result['TrendStrength_norm'] = result['TrendStrength_norm'].ffill().bfill()
    
    # Final validation
    result['TrendStrength_norm'] = np.clip(result['TrendStrength_norm'], 0, 1)
    
    return result

def normalize_indicators(
    df: pd.DataFrame, 
    exclude_cols: List[str] = None
) -> pd.DataFrame:
    """
    Normalize all numeric columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with indicators
        exclude_cols (list): Columns to exclude from normalization
        
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    # Default excluded columns
    if exclude_cols is None:
        exclude_cols = ['date', 'datetime', 'TrendStrength_norm']
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Normalize all numeric columns except excluded ones
    for col in result.columns:
        if result[col].dtype in ['float64', 'int64'] and col not in exclude_cols:
            # Skip columns that are already normalized (end with _norm)
            if not col.endswith('_norm'):
                result[f"{col}_norm"] = normalize_series(result[col])
    
    return result

def create_sequences(
    data: np.ndarray, 
    target: np.ndarray, 
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series models like LSTM.
    
    Args:
        data (np.ndarray): Input features
        target (np.ndarray): Target values
        seq_length (int): Sequence length
        
    Returns:
        tuple: (X sequences, Y targets)
    """
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)

def preprocess_data(
    df: pd.DataFrame,
    symbol: str = None,
    date_col: str = 'datetime',
    seq_length: int = 60,
    train_split: float = 0.8,
    output_dir: str = None
) -> Dict:
    """
    Preprocess data for machine learning models.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        symbol (str): Symbol name for saving
        date_col (str): Date column name
        seq_length (int): Sequence length for time series models
        train_split (float): Proportion of data to use for training
        output_dir (str): Directory to save processed data
        
    Returns:
        dict: Dictionary with preprocessed data and metadata
    """
    logger.info(f"Preprocessing data for {symbol if symbol else 'unknown symbol'}")
    
    # 1. Date handling
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        raise ValueError(f"Date column '{date_col}' not found")
    
    # 2. Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # 3. Calculate trend indicator
    df = calculate_trend_indicator(df)
    
    # 4. Normalize indicators
    df = normalize_indicators(df)
    
    # 5. Remove NaN values
    df.dropna(inplace=True)
    
    # 6. Save processed DataFrame if output_dir is specified
    if output_dir and symbol:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
    
    # 7. Create train/test split
    train_size = int(len(df) * train_split)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # 8. Create sequences for time series models
    feature_cols = [col for col in df.columns if col.endswith('_norm')]
    
    # Ensure close_norm is included
    if "close_norm" not in feature_cols and "close_norm" in df.columns:
        feature_cols.append("close_norm")
    
    X_train, y_train = create_sequences(
        train_df[feature_cols].values, 
        train_df['close_norm'].values,
        seq_length
    )
    
    X_test, y_test = create_sequences(
        test_df[feature_cols].values,
        test_df['close_norm'].values,
        seq_length
    )
    
    # 9. Return preprocessed data and metadata
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df,
        'feature_cols': feature_cols,
        'date_col': date_col,
        'symbol': symbol,
        'seq_length': seq_length
    }
    
    logger.info(f"Preprocessing complete. X_train shape: {X_train.shape}")
    return result

def load_and_preprocess(
    symbol: str,
    data_path: str,
    seq_length: int = 60,
    train_split: float = 0.8,
    output_dir: str = None
) -> Dict:
    """
    Load data from CSV and preprocess it.
    
    Args:
        symbol (str): Symbol to load
        data_path (str): Path to data file or directory
        seq_length (int): Sequence length for time series models
        train_split (float): Proportion of data to use for training
        output_dir (str): Directory to save processed data
        
    Returns:
        dict: Dictionary with preprocessed data and metadata
    """
    # Determine the file path
    if os.path.isdir(data_path):
        file_path = os.path.join(data_path, f"{symbol}.csv")
    else:
        file_path = data_path
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Preprocess data
    return preprocess_data(
        df=df,
        symbol=symbol,
        seq_length=seq_length,
        train_split=train_split,
        output_dir=output_dir
    )

# Additional utility functions for feature engineering
def add_price_derivatives(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Add price derivative features like returns and momentum.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        price_col (str): Price column name
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    result = df.copy()
    
    # Returns
    result['return_1d'] = result[price_col].pct_change(1)
    result['return_5d'] = result[price_col].pct_change(5)
    result['return_10d'] = result[price_col].pct_change(10)
    
    # Momentum (normalized)
    result['momentum_5d'] = result[price_col] / result[price_col].shift(5) - 1
    result['momentum_10d'] = result[price_col] / result[price_col].shift(10) - 1
    result['momentum_20d'] = result[price_col] / result[price_col].shift(20) - 1
    
    # Volatility
    result['volatility_5d'] = result['return_1d'].rolling(window=5).std()
    result['volatility_10d'] = result['return_1d'].rolling(window=10).std()
    result['volatility_20d'] = result['return_1d'].rolling(window=20).std()
    
    return result

def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price pattern features like support/resistance and trends.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    result = df.copy()
    
    # 20-day high/low
    result['high_20d'] = result['high'].rolling(window=20).max()
    result['low_20d'] = result['low'].rolling(window=20).min()
    
    # Distance to support/resistance
    result['dist_to_resistance'] = (result['high_20d'] - result['close']) / result['close']
    result['dist_to_support'] = (result['close'] - result['low_20d']) / result['close']
    
    # Price channels
    result['upper_channel'] = result['high'].rolling(window=20).max()
    result['lower_channel'] = result['low'].rolling(window=20).min()
    result['channel_width'] = (result['upper_channel'] - result['lower_channel']) / result['close']
    
    # Price position within channel (0-1)
    result['channel_position'] = (result['close'] - result['lower_channel']) / (result['upper_channel'] - result['lower_channel'])
    
    return result

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    result = df.copy()
    
    # Relative volume
    result['rel_volume_5d'] = result['volume'] / result['volume'].rolling(window=5).mean()
    result['rel_volume_10d'] = result['volume'] / result['volume'].rolling(window=10).mean()
    
    # Volume trend
    result['volume_trend_5d'] = result['volume'].rolling(window=5).mean() / result['volume'].rolling(window=10).mean()
    
    # Price-volume correlation
    result['price_volume_corr_5d'] = result['close'].rolling(window=5).corr(result['volume'])
    
    # Volume-based buy/sell pressure
    result['up_volume'] = result['volume'] * (result['close'] > result['open']).astype(int)
    result['down_volume'] = result['volume'] * (result['close'] < result['open']).astype(int)
    result['buy_sell_volume_ratio'] = result['up_volume'].rolling(window=5).sum() / (
        result['down_volume'].rolling(window=5).sum() + 1
    )  # Add 1 to avoid division by zero
    
    return result


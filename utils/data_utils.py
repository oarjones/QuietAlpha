"""
Data utility functions

This module provides utility functions for loading and manipulating data.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str, optional): Path to configuration file.
            If None, loads the default configuration.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Use default config path if none provided
    if config_path is None:
        config_path = os.path.join('config', 'base_config.json')
    
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Using empty configuration")
            return {}
        
        # Load JSON configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save JSON configuration with pretty formatting
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Override configuration
    
    Returns:
        Dict[str, Any]: Merged configuration
    """
    # Start with a copy of the base config
    merged = base_config.copy()
    
    # Recursively update with override values
    for key, value in override_config.items():
        # If both values are dictionaries, merge them recursively
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            # Otherwise, just override the value
            merged[key] = value
    
    return merged

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file (CSV, Excel, etc.).
    
    Args:
        file_path (str): Path to data file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
        
        # Determine file type from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return pd.DataFrame()
        
        logger.info(f"Loaded data from {file_path}, shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, file_path: str) -> bool:
    """
    Save data to a file (CSV, Excel, etc.).
    
    Args:
        df (pd.DataFrame): Data to save
        file_path (str): Path to save data
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Determine file type from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df.to_csv(file_path, index=False)
        elif file_ext in ['.xls', '.xlsx']:
            df.to_excel(file_path, index=False)
        elif file_ext == '.json':
            df.to_json(file_path, orient='records')
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return False
        
        logger.info(f"Saved data to {file_path}, shape: {df.shape}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        return False

def convert_timeframe(df: pd.DataFrame, source_timeframe: str, 
                     target_timeframe: str, date_col: str = 'datetime') -> pd.DataFrame:
    """
    Convert between timeframes (e.g., from 15min to 1h).
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        source_timeframe (str): Source timeframe ('1min', '5min', '1h', etc.)
        target_timeframe (str): Target timeframe
        date_col (str): Date column name
    
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    try:
        # Make sure the date column is datetime
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        else:
            logger.error(f"Date column '{date_col}' not found")
            return df
        
        # Set date column as index
        df = df.set_index(date_col)
        
        # Convert timeframe strings to pandas frequency strings
        freq_map = {
            '1min': '1T', '5min': '5T', '15min': '15T', '30min': '30T',
            '1h': 'H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': 'D', '1w': 'W', '1M': 'M'
        }
        
        target_freq = freq_map.get(target_timeframe)
        if not target_freq:
            logger.error(f"Unsupported target timeframe: {target_timeframe}")
            return df.reset_index()
        
        # Resample data
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Reset index to get the date column back
        resampled = resampled.reset_index()
        
        logger.info(f"Converted timeframe from {source_timeframe} to {target_timeframe}")
        return resampled
    
    except Exception as e:
        logger.error(f"Error converting timeframe: {e}")
        # Return original dataframe if conversion fails
        if date_col in df.columns and df.index.name == date_col:
            return df.reset_index()
        return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling missing values, outliers, etc.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Make a copy to avoid modifying the original
        cleaned = df.copy()
        
        # Handle missing values
        # For OHLC data, forward fill is often appropriate
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna(method='ffill')
        
        # For volume, fill with zeros
        if 'volume' in cleaned.columns:
            cleaned['volume'] = cleaned['volume'].fillna(0)
        
        # Remove rows where essential values are still missing
        essential_cols = [c for c in ['open', 'high', 'low', 'close'] if c in cleaned.columns]
        if essential_cols:
            cleaned = cleaned.dropna(subset=essential_cols)
        
        # Handle outliers in price data
        for col in ['high', 'low', 'close']:
            if col in cleaned.columns:
                # Detect outliers using 3 standard deviations from the mean
                mean = cleaned[col].mean()
                std = cleaned[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # Replace outliers with last valid value
                outliers = (cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)
                if outliers.any():
                    cleaned.loc[outliers, col] = cleaned[col].shift(1)
        
        logger.info(f"Cleaned data, removed {len(df) - len(cleaned)} rows")
        return cleaned
    
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return df

def get_available_symbols(data_dir: str = None) -> List[str]:
    """
    Get list of available symbols from data directory.
    
    Args:
        data_dir (str, optional): Data directory path.
            If None, uses default 'data/raw' directory.
    
    Returns:
        List[str]: List of available symbols
    """
    if data_dir is None:
        data_dir = os.path.join('data', 'raw')
    
    try:
        # Check if directory exists
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory not found: {data_dir}")
            return []
        
        # Get all CSV files
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        # Extract symbol names from filenames
        symbols = [os.path.splitext(f)[0] for f in files]
        
        logger.info(f"Found {len(symbols)} symbols in {data_dir}")
        return symbols
    
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        return []
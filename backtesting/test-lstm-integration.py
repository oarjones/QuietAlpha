"""
Test script to verify the integration of the universal LSTM model with the Trading Manager.

Run this script to test if the trading signals are correctly generated using LSTM predictions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from ibkr_api.interface import IBKRInterface
from trading_manager.base import TradingManager
from utils.data_utils import load_config

def test_trading_signals():
    """Test trading signals generation with LSTM predictions."""
    try:
        # Load configuration
        config = load_config()
        
        # Connect to IBKR
        ibkr = IBKRInterface()
        connected = ibkr.connect()
        
        if not connected:
            logger.error("Failed to connect to IBKR")
            return
        
        # Initialize Trading Manager
        trading_manager = TradingManager(ibkr_interface=ibkr)
        
        # Define test symbols
        test_symbols = ['AMZN', 'AAPL', 'MSFT', 'TSLA', 'GOOGL']
        
        # Test results
        results = {}
        
        # Generate signals for each symbol
        for symbol in test_symbols:
            logger.info(f"Generating signals for {symbol}")
            
            # Get market data
            data = trading_manager.get_market_data(symbol)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            # Generate signals
            signals = trading_manager.generate_trading_signals(symbol, data)
            
            # Store results
            results[symbol] = signals
            
            # Print summary
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Signal: {signals['signal']}")
            logger.info(f"Strength: {signals['strength']}")
            logger.info(f"Reasons: {signals['reasons']}")
            
            # Print LSTM prediction details
            lstm_pred = signals.get('lstm_prediction', {})
            logger.info(f"LSTM Direction: {lstm_pred.get('predicted_direction', 'unknown')}")
            logger.info(f"LSTM Confidence: {lstm_pred.get('confidence', 0):.4f}")
            
            if 'predicted_price' in lstm_pred:
                current_price = lstm_pred.get('current_price', 0)
                predicted_price = lstm_pred.get('predicted_price', 0)
                price_change_pct = lstm_pred.get('price_change_pct', 0)
                
                logger.info(f"Current Price: ${current_price:.2f}")
                logger.info(f"Predicted Price: ${predicted_price:.2f}")
                logger.info(f"Price Change: {price_change_pct:.2f}%")
            
            logger.info("-" * 50)
        
        # Disconnect from IBKR
        ibkr.disconnect()
        
        return results
    
    except Exception as e:
        logger.error(f"Error in test_trading_signals: {e}")
        # Ensure IBKR is disconnected
        try:
            ibkr.disconnect()
        except:
            pass
        return None

if __name__ == "__main__":
    logger.info("Starting LSTM integration test")
    results = test_trading_signals()
    logger.info("Test completed")
"""
Script to initialize the LSTM model directory structure and setup initial metadata.

This script:
1. Creates the necessary directory structure for symbol-specific models
2. Copies the universal model to the universal folder
3. Creates initial metadata files
"""

import os
import json
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_lstm_structure():
    """Initialize the LSTM model directory structure."""
    
    # Create base directories
    lstm_dir = os.path.join('models', 'lstm')
    symbols_dir = os.path.join(lstm_dir, 'symbols')
    universal_dir = os.path.join(lstm_dir, 'universal')
    
    os.makedirs(lstm_dir, exist_ok=True)
    os.makedirs(symbols_dir, exist_ok=True)
    os.makedirs(universal_dir, exist_ok=True)
    
    logger.info(f"Created directory structure: {lstm_dir}")
    
    # Check if universal model exists in root directory
    universal_model_src = os.path.join(lstm_dir, 'lstm_model_universal.keras')
    universal_scaler_src = os.path.join(lstm_dir, 'scaler_universal.pkl')
    
    universal_model_dst = os.path.join(universal_dir, 'model.keras')
    universal_scaler_dst = os.path.join(universal_dir, 'scaler.pkl')
    
    # Copy universal model if it exists
    if os.path.exists(universal_model_src):
        shutil.copy2(universal_model_src, universal_model_dst)
        logger.info(f"Copied universal model to {universal_model_dst}")
    else:
        logger.warning(f"Universal model not found at {universal_model_src}")
    
    # Copy universal scaler if it exists
    if os.path.exists(universal_scaler_src):
        shutil.copy2(universal_scaler_src, universal_scaler_dst)
        logger.info(f"Copied universal scaler to {universal_scaler_dst}")
    else:
        logger.warning(f"Universal scaler not found at {universal_scaler_src}")
    
    # Create metadata for universal model
    if os.path.exists(universal_model_dst):
        metadata = {
            'trained_date': datetime.now().isoformat(),
            'symbol': 'universal',
            'metrics': {
                'rmse': 0.0,  # Placeholder - should be updated with actual values
                'mae': 0.0,
                'r2': 0.0
            },
            'reliability_index': 0.7,  # Initial default value
            'training_data': {
                'symbols': ['AMZN'],  # Update with actual symbols used
                'periods': 0,
                'start_date': '2022-01-01T00:00:00',
                'end_date': '2025-02-28T00:00:00'
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(universal_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created metadata for universal model at {metadata_path}")
    
    # Setup example directories for a few common symbols
    example_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    for symbol in example_symbols:
        symbol_dir = os.path.join(symbols_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Create placeholder metadata
        metadata = {
            'trained_date': '2000-01-01T00:00:00',  # Old date to trigger retraining
            'symbol': symbol,
            'metrics': {
                'rmse': 0.0,
                'mae': 0.0,
                'r2': 0.0
            },
            'reliability_index': 0.0,
            'status': 'pending',
            'message': 'Initial placeholder metadata'
        }
        
        # Save metadata
        metadata_path = os.path.join(symbol_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created directory and metadata for {symbol}")
    
    logger.info("LSTM structure initialization complete")

if __name__ == "__main__":
    logger.info("Starting LSTM structure initialization")
    initialize_lstm_structure()
    logger.info("LSTM structure initialization finished")
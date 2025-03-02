"""
LSTM System Initialization

This script initializes the LSTM model system, including:
1. Setting up directory structure
2. Checking for model availability
3. Ensuring universal model is available
4. Starting the training service
5. Applying integration patches

Run this script before starting the main application to ensure all LSTM components
are properly initialized.
"""

import os
import sys
import logging
import argparse
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_directory_structure():
    """Setup required directories for the LSTM system."""
    directories = [
        'logs',
        'models',
        os.path.join('models', 'lstm'),
        os.path.join('models', 'lstm', 'symbols'),
        os.path.join('models', 'lstm', 'universal'),
        'data',
        os.path.join('data', 'raw'),
        os.path.join('data', 'processed')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
    
    return True

def check_universal_model():
    """
    Check if universal model exists, train if needed.
    
    Returns:
        bool: True if model is available or training started, False otherwise
    """
    # Import model manager
    from lstm.model_manager import LSTMModelManager
    
    model_manager = LSTMModelManager()
    
    # Check for universal model
    model_path = os.path.join('models', 'lstm', 'universal', 'model.keras')
    metadata_path = os.path.join('models', 'lstm', 'universal', 'metadata.json')
    
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        logger.info("Universal model found")
        
        # Check model age
        import json
        from datetime import datetime
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            trained_date = datetime.fromisoformat(metadata.get('trained_date', '2000-01-01T00:00:00'))
            days_old = (datetime.now() - trained_date).days
            
            if days_old > 30:
                logger.warning(f"Universal model is {days_old} days old, consider retraining")
            else:
                logger.info(f"Universal model is {days_old} days old")
        except Exception as e:
            logger.error(f"Error checking universal model age: {e}")
        
        return True
    else:
        logger.warning("Universal model not found")
        return False

def start_lstm_service():
    """
    Start the LSTM service.
    
    Returns:
        bool: True if service started successfully, False otherwise
    """
    try:
        # Import service
        from lstm.integration import get_lstm_service
        
        # Start service
        service = get_lstm_service()
        
        # Check if service is running
        if service._initialized:
            logger.info("LSTM service started successfully")
            return True
        else:
            logger.error("LSTM service failed to initialize properly")
            return False
            
    except Exception as e:
        logger.error(f"Error starting LSTM service: {e}")
        return False

def update_trading_manager():
    """
    Update TradingManager with LSTM integration.
    
    Returns:
        bool: True if update applied successfully, False otherwise
    """
    try:
        # Import update script
        import trading_manager.update_trading_manager as updater
        
        # Apply update
        result = updater.update_trading_manager()
        
        if result:
            logger.info("Trading Manager updated successfully")
        else:
            logger.warning("Failed to update Trading Manager")
        
        return result
        
    except Exception as e:
        logger.error(f"Error updating Trading Manager: {e}")
        return False

def train_initial_models(symbols=None, train_universal=True):
    """
    Train initial models if needed.
    
    Args:
        symbols: List of symbols to train
        train_universal: Whether to train universal model
        
    Returns:
        dict: Training results
    """
    # Import trainer
    from lstm.trainer import LSTMTrainer
    
    # Check if symbols are provided
    if not symbols:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Create trainer
    trainer = LSTMTrainer()
    
    results = {}
    
    # Train universal model if requested and not exists
    if train_universal and not check_universal_model():
        logger.info(f"Training universal model with symbols: {symbols}")
        universal_result = trainer.train_universal_model(symbols)
        results['universal'] = universal_result
        
        if universal_result['status'] == 'success':
            logger.info("Universal model trained successfully")
        else:
            logger.warning(f"Universal model training failed: {universal_result.get('message', 'Unknown error')}")
    
    # Check for symbol-specific models
    missing_models = []
    
    for symbol in symbols:
        if not trainer.model_manager.model_exists(symbol):
            missing_models.append(symbol)
    
    # Schedule training for missing models
    if missing_models:
        logger.info(f"Scheduling training for missing models: {missing_models}")
        
        # Start the trainer if not running
        if not trainer.is_running:
            trainer.start()
        
        # Schedule models with priority
        for symbol in missing_models:
            trainer.request_training(symbol, priority=50)
        
        results['scheduled'] = missing_models
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Initialize LSTM System')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                      help='Symbols to initialize')
    parser.add_argument('--train-universal', action='store_true',
                      help='Train universal model if not exists')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training initial models')
    parser.add_argument('--check-only', action='store_true',
                      help='Only check system status without making changes')
    
    args = parser.parse_args()
    
    # Check only mode
    if args.check_only:
        logger.info("Running in check-only mode")
        
        # Check directory structure
        setup_directory_structure()
        
        # Check universal model
        has_universal = check_universal_model()
        
        # Check symbol-specific models
        from lstm.model_manager import LSTMModelManager
        model_manager = LSTMModelManager()
        
        available_models = model_manager.get_available_models()
        missing_models = [symbol for symbol in args.symbols if symbol not in available_models]
        
        logger.info(f"Available models: {available_models}")
        logger.info(f"Missing models: {missing_models}")
        
        # Exit without changing anything
        return 0
    
    # Setup directory structure
    setup_directory_structure()
    
    # Train initial models if needed
    if not args.skip_training:
        train_results = train_initial_models(args.symbols, args.train_universal)
        logger.info(f"Training results: {train_results}")
    
    # Start LSTM service
    service_started = start_lstm_service()
    
    # Update Trading Manager
    tm_updated = update_trading_manager()
    
    # Report status
    if service_started and tm_updated:
        logger.info("LSTM system initialized successfully")
        return 0
    else:
        logger.warning("LSTM system initialization completed with warnings")
        return 1

if __name__ == "__main__":
    sys.exit(main())
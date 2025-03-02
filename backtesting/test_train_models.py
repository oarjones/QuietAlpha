"""
LSTM Model Training Script

This script demonstrates how to use the LSTMTrainer for:
1. Training individual symbol models
2. Batch training with priority
3. Training the universal model

Usage:
    python -m lstm.train_models --mode [single|batch|universal] --symbols AAPL MSFT ...
"""

import argparse
import logging
import time
import sys
import os
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'lstm_training.log'))
    ]
)
logger = logging.getLogger(__name__)

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trainer
from lstm.trainer import LSTMTrainer
from ibkr_api.interface import IBKRInterface

def train_single_model(symbols: List[str], wait_completion: bool = True) -> Dict:
    """
    Train a single model for each symbol in the list.
    
    Args:
        symbols: List of symbols to train
        wait_completion: Whether to wait for training to complete
        
    Returns:
        dict: Training results
    """
    # Create trainer
    trainer = LSTMTrainer()
    
    results = {}
    
    # Train each symbol
    for symbol in symbols:
        logger.info(f"Training model for {symbol}")
        
        # Train directly
        result = trainer.train_lstm_model(symbol)
        results[symbol] = result
        
        logger.info(f"Training result for {symbol}: {result['status']}")
    
    return {
        'status': 'completed',
        'results': results
    }

def train_batch_with_priority(symbols: List[str], max_parallel: int = 2) -> Dict:
    """
    Train a batch of models in parallel with priority.
    
    Args:
        symbols: List of symbols to train
        max_parallel: Maximum number of parallel trainings
        
    Returns:
        dict: Training results
    """
    # Create trainer with specified max parallel trainings
    trainer = LSTMTrainer()
    trainer.max_parallel_trainings = max_parallel
    
    # Run batch training
    result = trainer.run_parallel_batch_training(symbols)
    
    if result['status'] != 'success':
        logger.error(f"Error starting batch training: {result}")
        return result
    
    # Wait for training to complete
    logger.info(f"Started batch training for {len(result['scheduled_symbols'])} symbols")
    logger.info("Training is running in the background.")
    logger.info("Press Ctrl+C to stop monitoring (training will continue)")
    
    try:
        while True:
            # Get queue status
            queue_status = trainer.get_queue_status()
            active_count = len(queue_status['active_trainings'])
            queue_size = queue_status['queue_size']
            
            logger.info(f"Active trainings: {active_count}, Queue size: {queue_size}")
            
            # Get training status
            for symbol, status in trainer.get_training_status().items():
                if status.get('status') == 'success':
                    metrics = status.get('metrics', {})
                    rmse = metrics.get('rmse', 'N/A')
                    r2 = metrics.get('r2', 'N/A')
                    logger.info(f"Completed {symbol}: RMSE={rmse}, R²={r2}")
            
            # If no active trainings and queue is empty, we're done
            if active_count == 0 and queue_size == 0:
                logger.info("All trainings completed!")
                break
            
            # Sleep
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped. Training continues in background.")
    
    # Return final status
    return {
        'status': 'monitoring_complete',
        'queue_status': trainer.get_queue_status(),
        'training_status': trainer.get_training_status()
    }

def train_universal_model(base_symbols: List[str] = None) -> Dict:
    """
    Train the universal model.
    
    Args:
        base_symbols: List of symbols to use for training
        
    Returns:
        dict: Training results
    """
    # Create trainer
    trainer = LSTMTrainer()
    
    # Use default symbols if not provided
    if not base_symbols:
        base_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Train universal model
    logger.info(f"Training universal model with symbols: {base_symbols}")
    result = trainer.train_universal_model(base_symbols)
    
    if result['status'] == 'success':
        logger.info(f"Universal model training completed!")
        metrics = result.get('metrics', {})
        rmse = metrics.get('rmse', 'N/A')
        r2 = metrics.get('r2', 'N/A')
        reliability = result.get('reliability_index', 'N/A')
        logger.info(f"RMSE={rmse}, R²={r2}, Reliability={reliability}")
    else:
        logger.error(f"Error training universal model: {result}")
    
    return result

def main():
    """Main entry point for the script."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='LSTM Model Training')
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'batch', 'universal'],
                      help='Training mode: single, batch, or universal')
    parser.add_argument('--symbols', type=str, nargs='+', default=[],
                      help='Symbols to train')
    parser.add_argument('--max-parallel', type=int, default=2,
                      help='Maximum parallel trainings for batch mode')
    
    args = parser.parse_args()
    
    # Check if symbols are provided when needed
    if args.mode in ['single', 'batch'] and not args.symbols:
        parser.error(f"{args.mode} mode requires at least one symbol (--symbols)")
    
    try:
        # Run the appropriate training function
        if args.mode == 'single':
            logger.info(f"Starting single model training for {len(args.symbols)} symbols")
            result = train_single_model(args.symbols)
        elif args.mode == 'batch':
            logger.info(f"Starting batch training for {len(args.symbols)} symbols with max {args.max_parallel} parallel")
            result = train_batch_with_priority(args.symbols, args.max_parallel)
        elif args.mode == 'universal':
            logger.info("Starting universal model training")
            result = train_universal_model(args.symbols if args.symbols else None)
        
        logger.info(f"Training completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
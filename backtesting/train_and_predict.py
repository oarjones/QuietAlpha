"""
Example script demonstrating how to:
1. Train a universal LSTM model
2. Train multiple symbol-specific models in parallel
3. Predict prices using both universal and symbol-specific models
4. Compare the performance of different model approaches
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from lstm.trainer import LSTMTrainer
from lstm.integration import get_lstm_service, predict_with_lstm
from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators

def setup_environment():
    """Setup directories and environment."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('models', 'lstm'), exist_ok=True)
    os.makedirs(os.path.join('models', 'lstm', 'symbols'), exist_ok=True)
    os.makedirs(os.path.join('models', 'lstm', 'universal'), exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def train_example_models(symbols, train_universal=True, parallel=True):
    """
    Train example LSTM models for the specified symbols.
    
    Args:
        symbols: List of symbols to train
        train_universal: Whether to train a universal model
        parallel: Whether to train models in parallel
        
    Returns:
        dict: Training results
    """
    # Create trainer
    trainer = LSTMTrainer()
    
    results = {}
    
    # First, train universal model if requested
    if train_universal:
        logger.info(f"Training universal model with symbols: {symbols}")
        universal_result = trainer.train_universal_model(symbols)
        results['universal'] = universal_result
        logger.info(f"Universal model training result: {universal_result['status']}")
    
    # Then train symbol-specific models
    if parallel:
        # Train in parallel
        logger.info(f"Training symbol-specific models in parallel for: {symbols}")
        batch_result = trainer.run_parallel_batch_training(symbols)
        results['parallel'] = batch_result
        
        # Start queue if not already running
        if not trainer.is_running:
            trainer.start()
        
        # Monitor training progress
        logger.info("Monitoring training progress...")
        
        try:
            complete = False
            while not complete:
                # Get queue status
                queue_status = trainer.get_queue_status()
                active_count = len(queue_status['active_trainings'])
                queue_size = queue_status['queue_size']
                
                if active_count > 0:
                    active_symbols = ', '.join(queue_status['active_trainings'])
                    logger.info(f"Active trainings: {active_symbols}, Queue: {queue_size}")
                else:
                    logger.info(f"No active trainings, Queue: {queue_size}")
                
                # Check if all scheduled trainings have completed
                scheduled_count = batch_result.get('total_scheduled', 0)
                trained_count = 0
                
                for symbol in batch_result.get('scheduled_symbols', []):
                    status = trainer.get_training_status(symbol)
                    if status.get('status') in ['success', 'error']:
                        trained_count += 1
                
                logger.info(f"Progress: {trained_count}/{scheduled_count} models completed")
                
                # If all scheduled trainings have completed or no active trainings and empty queue
                if trained_count == scheduled_count or (active_count == 0 and queue_size == 0 and scheduled_count > 0):
                    complete = True
                    logger.info("All trainings completed!")
                else:
                    # Sleep before checking again
                    time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
    else:
        # Train sequentially
        symbol_results = {}
        for symbol in symbols:
            logger.info(f"Training model for {symbol}")
            result = trainer.train_lstm_model(symbol)
            symbol_results[symbol] = result
            logger.info(f"Training result for {symbol}: {result['status']}")
        
        results['sequential'] = symbol_results
    
    # Get final training status
    final_status = {}
    for symbol in symbols:
        status = trainer.get_training_status(symbol)
        if status:
            metrics = status.get('metrics', {})
            message = status.get('message', '')
            final_status[symbol] = {
                'status': status.get('status', 'unknown'),
                'rmse': metrics.get('rmse', 'N/A'),
                'r2': metrics.get('r2', 'N/A'),
                'message': message
            }
    
    results['final_status'] = final_status
    return results

def predict_and_compare(symbols, ibkr_interface):
    """
    Make predictions using trained models and compare results.
    
    Args:
        symbols: List of symbols to predict for
        ibkr_interface: IBKR interface for fetching data
        
    Returns:
        dict: Prediction results
    """
    lstm_service = get_lstm_service(ibkr_interface=ibkr_interface)
    
    results = {}
    
    for symbol in symbols:
        logger.info(f"Making predictions for {symbol}")
        
        # Get historical data
        from lstm.model import fetch_historical_data
        data = fetch_historical_data(ibkr_interface, symbol, lookback_days=60)
        
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            results[symbol] = {'status': 'error', 'message': 'No data available'}
            continue
        
        # Process data
        data = calculate_technical_indicators(data, include_all=True)
        data = calculate_trend_indicator(data)
        data = normalize_indicators(data)
        
        # Get model status
        model_status = lstm_service.get_model_status(symbol)
        
        # Make prediction
        prediction = lstm_service.predict_price(symbol, data)
        
        results[symbol] = {
            'model_status': model_status,
            'prediction': prediction
        }
        
        # Log prediction result
        if prediction['status'] == 'success':
            logger.info(f"Prediction for {symbol}:")
            logger.info(f"  Direction: {prediction['predicted_direction']}")
            logger.info(f"  Confidence: {prediction['confidence']:.4f}")
            logger.info(f"  Current price: {prediction['current_price']:.2f}")
            logger.info(f"  Predicted price: {prediction['predicted_price']:.2f}")
            logger.info(f"  Change: {prediction['price_change_pct']:.2f}%")
            logger.info(f"  Model type: {prediction['model_type']}")
        else:
            logger.warning(f"Prediction failed for {symbol}: {prediction.get('message', 'Unknown error')}")
    
    return results

def visualize_predictions(predictions, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        predictions: Dictionary of prediction results
        save_path: Path to save the plot
    """
    successful_predictions = {
        symbol: pred['prediction'] for symbol, pred in predictions.items()
        if pred.get('prediction', {}).get('status') == 'success'
    }
    
    if not successful_predictions:
        logger.warning("No successful predictions to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot predicted price changes
    symbols = list(successful_predictions.keys())
    changes = [pred['price_change_pct'] for pred in successful_predictions.values()]
    directions = [pred['predicted_direction'] for pred in successful_predictions.values()]
    confidences = [pred['confidence'] for pred in successful_predictions.values()]
    model_types = [pred['model_type'] for pred in successful_predictions.values()]
    
    # Color bars by direction
    colors = []
    for direction in directions:
        if direction == 'up':
            colors.append('green')
        elif direction == 'down':
            colors.append('red')
        else:
            colors.append('grey')
    
    # Plot bars
    bars = plt.bar(symbols, changes, color=colors, alpha=0.7)
    
    # Add confidence annotations
    for i, (bar, confidence) in enumerate(zip(bars, confidences)):
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -0.3
        else:
            va = 'bottom'
            offset = 0.3
        plt.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{confidence:.2f}', ha='center', va=va, fontweight='bold')
    
    # Add model type markers
    for i, (bar, model_type) in enumerate(zip(bars, model_types)):
        marker = '*' if model_type == 'symbol_specific' else 'o'
        plt.plot(bar.get_x() + bar.get_width()/2, 0, marker, color='black', 
                ms=10, markeredgecolor='white')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Universal Model'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12, label='Symbol-specific Model'),
        plt.Rectangle((0,0), 1, 1, fc='green', alpha=0.7, label='Bullish'),
        plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.7, label='Bearish'),
        plt.Rectangle((0,0), 1, 1, fc='grey', alpha=0.7, label='Neutral')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Add labels and title
    plt.xlabel('Symbol')
    plt.ylabel('Predicted Price Change (%)')
    plt.title('LSTM Model Predictions')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.figtext(0.5, 0.01, f'Generated: {timestamp}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='LSTM Training and Prediction Example')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                      help='Symbols to use')
    parser.add_argument('--universal', action='store_true',
                      help='Train universal model')
    parser.add_argument('--parallel', action='store_true',
                      help='Train models in parallel')
    parser.add_argument('--predict-only', action='store_true',
                      help='Skip training and only make predictions')
    parser.add_argument('--plot', action='store_true',
                      help='Generate visualization plot')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Connect to IBKR
    ibkr = IBKRInterface()
    connected = ibkr.connect()
    
    if not connected:
        logger.error("Failed to connect to IBKR")
        return 1
    
    try:
        # Train models if not predict-only
        if not args.predict_only:
            logger.info(f"Training models for symbols: {args.symbols}")
            training_results = train_example_models(
                args.symbols,
                train_universal=args.universal,
                parallel=args.parallel
            )
            logger.info("Training completed")
        
        # Make predictions
        logger.info("Making predictions")
        prediction_results = predict_and_compare(args.symbols, ibkr)
        
        # Visualize if requested
        if args.plot:
            logger.info("Generating visualization")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join('plots', f'predictions_{timestamp}.png')
            visualize_predictions(prediction_results, save_path)
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)
        return 1
    finally:
        # Disconnect from IBKR
        ibkr.disconnect()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
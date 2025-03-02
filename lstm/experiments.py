"""
LSTM Experiments for Price Prediction

This module provides functions to run various LSTM experiments using the
universal LSTM model with data from IBKR.

Main functions:
- train_universal_model: Train a universal LSTM model with optimal hyperparameters
- evaluate_multiple_symbols: Evaluate the universal model on multiple symbols
- random_search: Perform random search for hyperparameter optimization
"""

import os
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime

from model import run_lstm_training, connect_to_ibkr, load_model, predict_future_prices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def random_search(base_config, n_iter=10, ibkr=None):
    """
    Perform random search for hyperparameters.
    
    Args:
        base_config (dict): Base configuration
        n_iter (int): Number of iterations
        ibkr (IBKRInterface): IBKR interface
        
    Returns:
        tuple: (best_config, best_score)
    """
    best_config = None
    best_score = float('inf')
    
    logger.info(f"Starting random search with {n_iter} iterations")
    
    # Connect to IBKR if not provided
    if ibkr is None:
        ibkr = connect_to_ibkr()
        if ibkr is None:
            logger.error("Failed to connect to IBKR. Exiting.")
            return None, float('inf')
    
    # Track all results
    all_results = []
    
    for i in range(n_iter):
        # Create a new configuration with random parameters
        config = base_config.copy()
        config['units'] = random.choice([50, 100, 150, 200])
        config['dropout'] = random.uniform(0.2, 0.5)
        config['learning_rate'] = random.choice([0.001, 0.0005, 0.0001])
        config['num_layers'] = random.choice([2, 3, 4])
        config['batch_size'] = random.choice([64, 128, 256])
        config['bidirectional'] = random.choice([True, False])
        config['l2_reg'] = random.choice([0, 0.01, 0.001])
        config['lr_factor'] = random.choice([0.1, 0.2, 0.3])
        
        # Update label for this specific experiment
        config['label'] = f"{base_config['label']}_rs_{i+1}"
        
        logger.info(f"Running iteration {i+1}/{n_iter} with config: {config['label']}")
        
        # Run training
        result = run_lstm_training(config, ibkr)
        
        # Track result
        if result["status"] == "success":
            score = result['rmse']  # Use RMSE as score
            all_results.append({
                'iteration': i+1,
                'config': config,
                'rmse': score,
                'r2': result.get('r2', 0)
            })
            
            # Update best if improved
            if score < best_score:
                best_score = score
                best_config = config
                logger.info(f"New best config found with RMSE: {best_score:.4f}")
    
    # Save all results
    results_df = pd.DataFrame([
        {
            'iteration': r['iteration'],
            'label': r['config']['label'],
            'units': r['config']['units'],
            'dropout': r['config']['dropout'],
            'learning_rate': r['config']['learning_rate'],
            'num_layers': r['config']['num_layers'],
            'batch_size': r['config']['batch_size'],
            'bidirectional': r['config']['bidirectional'],
            'l2_reg': r['config']['l2_reg'],
            'lr_factor': r['config']['lr_factor'],
            'rmse': r['rmse'],
            'r2': r['r2']
        } for r in all_results
    ])
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/random_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    
    logger.info(f"Random search completed. Best config: {best_config}")
    logger.info(f"All results saved to {csv_path}")
    
    return best_config, best_score

def train_universal_model(symbols=None):
    """
    Train a universal LSTM model with optimal hyperparameters.
    
    Args:
        symbols (list): List of symbols to train on (defaults to ['AMZN'])
        
    Returns:
        dict: Training results
    """
    # Use AMZN if no symbols provided
    if not symbols:
        symbols = ['AMZN']
    
    logger.info(f"Training universal model with symbols: {symbols}")
    
    # Connect to IBKR
    ibkr = connect_to_ibkr()
    if ibkr is None:
        logger.error("Failed to connect to IBKR. Exiting.")
        return {"status": "error", "message": "Failed to connect to IBKR"}
    
    # Train on each symbol sequentially, with last one being the final model
    results = {}
    
    # Define the optimal configuration (based on provided hyperparameters)
    universal_config = {
        'label': 'universal_model',
        'symbol': 'AMZN',  # Will be updated in the loop
        'seq_length': 60,
        'epochs': 50,
        'batch_size': 128,
        'units': 200,
        'dropout': 0.40973999440459574,
        'learning_rate': 0.0001,
        'patience': 10,
        'num_layers': 2,
        'bidirectional': True,
        'l2_reg': 0,
        'lr_factor': 0.3,
        'min_lr': 1e-05
    }
    
    for symbol in symbols:
        logger.info(f"Training on {symbol}...")
        
        # Update config for this symbol
        config = universal_config.copy()
        config['symbol'] = symbol
        config['label'] = f"universal_model_{symbol}"
        
        # Run training
        result = run_lstm_training(config, ibkr)
        results[symbol] = result
        
        # Log result
        if result["status"] == "success":
            logger.info(f"Training on {symbol} successful. RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}")
        else:
            logger.error(f"Training on {symbol} failed: {result['message']}")
    
    # Disconnect from IBKR
    ibkr.disconnect()
    
    return {
        "status": "success",
        "results": results,
        "model_path": "models/lstm/lstm_model_universal.keras",
        "scaler_path": "models/lstm/scaler_universal.pkl"
    }

def evaluate_multiple_symbols(symbols, use_existing_model=True):
    """
    Evaluate the universal model on multiple symbols.
    
    Args:
        symbols (list): List of symbols to evaluate
        use_existing_model (bool): Whether to use existing model or train new one
        
    Returns:
        dict: Evaluation results
    """
    logger.info(f"Evaluating universal model on symbols: {symbols}")
    
    # Connect to IBKR
    ibkr = connect_to_ibkr()
    if ibkr is None:
        logger.error("Failed to connect to IBKR. Exiting.")
        return {"status": "error", "message": "Failed to connect to IBKR"}
    
    # Load or train model
    if use_existing_model:
        model, scaler = load_model()
        if model is None or scaler is None:
            logger.warning("Failed to load existing model. Training new model...")
            train_result = train_universal_model(['AMZN'])
            if train_result["status"] != "success":
                ibkr.disconnect()
                return {"status": "error", "message": "Failed to train new model"}
            model, scaler = load_model()
    else:
        train_result = train_universal_model(symbols[:1])  # Train on first symbol
        if train_result["status"] != "success":
            ibkr.disconnect()
            return {"status": "error", "message": "Failed to train model"}
        model, scaler = load_model()
    
    # Evaluate on each symbol
    results = {}
    
    for symbol in symbols:
        logger.info(f"Evaluating on {symbol}...")
        
        # Create config for evaluation
        config = {
            'label': f"eval_{symbol}",
            'symbol': symbol,
            'seq_length': 60
        }
        
        # Run evaluation (simplified version of training without actual training)
        from model import fetch_historical_data, preprocess_data, evaluate_model
        
        # Fetch data
        df = fetch_historical_data(ibkr, symbol, lookback_days=365)  # 1 year for evaluation
        if df.empty:
            logger.error(f"No data fetched for {symbol}. Skipping.")
            results[symbol] = {"status": "error", "message": f"No data fetched for {symbol}"}
            continue
        
        # Preprocess data
        try:
            _, _, X_test, y_test, _, _, _, _ = preprocess_data(
                df, 
                symbol, 
                seq_length=60,
                train_split=0.8
            )
            
            # Evaluate model
            rmse, mae, r2, _, _ = evaluate_model(model, X_test, y_test, scaler, X_test)
            
            # Store results
            results[symbol] = {
                "status": "success",
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            logger.info(f"Evaluation on {symbol} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {e}")
            results[symbol] = {"status": "error", "message": str(e)}
    
    # Disconnect from IBKR
    ibkr.disconnect()
    
    # Compile summary
    summary = {
        "status": "success",
        "symbols": len(symbols),
        "successful": sum(1 for s in results if results[s]["status"] == "success"),
        "average_rmse": np.mean([results[s]["rmse"] for s in results if results[s]["status"] == "success"]),
        "average_r2": np.mean([results[s]["r2"] for s in results if results[s]["status"] == "success"]),
        "results": results
    }
    
    return summary

# Main execution code
if __name__ == "__main__":
    import argparse

    

    
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="LSTM Experiments")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "evaluate", "random_search"],
                        help="Mode of operation")
    parser.add_argument("--symbols", type=str, nargs="+", 
                        default=["BAC"],
                        help="List of symbols to use (e.g., AAPL MSFT GOOGL)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for random search")
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == "train":
        logger.info(f"Training universal model on symbols: {args.symbols}")
        result = train_universal_model(args.symbols)
        
        if result["status"] == "success":
            logger.info("Training complete. Model saved.")
        else:
            logger.error(f"Training failed: {result.get('message', 'Unknown error')}")
    
    elif args.mode == "evaluate":
        logger.info(f"Evaluating model on symbols: {args.symbols}")
        result = evaluate_multiple_symbols(args.symbols)
        
        if result["status"] == "success":
            logger.info(f"Evaluation complete. Average RMSE: {result['average_rmse']:.4f}, Average R²: {result['average_r2']:.4f}")
        else:
            logger.error(f"Evaluation failed: {result.get('message', 'Unknown error')}")
    
    elif args.mode == "random_search":
        logger.info(f"Performing random search with {args.iterations} iterations")
        
        # Base config for random search
        base_config = {
            'label': 'rs_experiment',
            'symbol': args.symbols[0],  # Use first symbol
            'seq_length': 60,
            'epochs': 30,  # Reduced epochs for faster search
            'batch_size': 128,
            'patience': 5,  # Reduced patience for faster search
        }
        
        best_config, best_score = random_search(base_config, args.iterations)
        
        if best_config:
            logger.info(f"Random search complete. Best RMSE: {best_score:.4f}")
            logger.info(f"Best config: {best_config}")
        else:
            logger.error("Random search failed")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
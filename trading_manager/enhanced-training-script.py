#!/usr/bin/env python
"""
Enhanced RL Trading Training Script

This script implements the enhanced training process for RL trading models with:
1. Improved hyperparameter optimization
2. Enhanced reward function
3. Better neural network architecture
4. Comprehensive evaluation metrics

Usage:
    python train_enhanced_rl.py --symbol MSFT --n-trials 30 --enable-optuna
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

# Import project modules
from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, normalize_indicators, calculate_trend_indicator
from utils.data_utils import load_config, save_config, merge_configs
from rl_trading_stable import RLTradingStableManager, EnhancedTradingEnv
from integration_helpers import (
    make_enhanced_env, 
    enhance_trading_manager, 
    create_enhanced_config,
    update_config_file,
    run_enhanced_optimization
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'enhanced_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_and_process_data(ibkr: IBKRInterface, symbol: str, lookback_days: int = 730) -> pd.DataFrame:
    """
    Fetch and process data for a symbol.
    
    Args:
        ibkr: IBKR interface
        symbol: Symbol to fetch data for
        lookback_days: Number of days to look back
        
    Returns:
        pd.DataFrame: Processed data
    """
    try:
        logger.info(f"Fetching data for {symbol}, lookback: {lookback_days} days")
        
        # Determine duration string based on lookback
        if lookback_days <= 30:
            duration = f"{lookback_days} D"
        elif lookback_days <= 180:
            duration = f"{lookback_days // 30 + 1} M"
        else:
            duration = f"{lookback_days // 365 + 1} Y"
        
        # Fetch hourly data
        df = ibkr.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size="1 hour",
            what_to_show="TRADES",
            use_rth=True
        )
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Process data
        logger.info(f"Processing data for {symbol} ({len(df)} rows)")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, include_all=True)
        
        # Calculate trend indicator
        df = calculate_trend_indicator(df)
        
        # Normalize indicators
        df = normalize_indicators(df)
        
        # Remove the first 22 rows due to incorrect values from indicator windows
        df = df.iloc[22:]
        
        # Remove NaN values
        df.dropna(inplace=True)

        # Save processed data for reference
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(f'data/processed/{symbol}_processed.csv', index=False)
        
        logger.info(f"Processed data for {symbol}: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching and processing data for {symbol}: {e}")
        return pd.DataFrame()

def load_data_from_file(filepath: str) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        filepath: Path to data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading data from {filepath}")
        
        # Determine file type from extension
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.pkl', '.pickle']:
            df = pd.read_pickle(filepath)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return pd.DataFrame()
        
        # Convert datetime column if exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from file: {e}")
        return pd.DataFrame()

def simple_rl_training(
    data: pd.DataFrame, 
    config: dict, 
    symbol: str, 
    timesteps: int = 200000,
    output_dir: str = "models/rl/enhanced"
) -> dict:
    """
    Train a simple RL model without optimization.
    
    Args:
        data: Historical price data
        config: Environment configuration
        symbol: Symbol being trained
        timesteps: Number of training timesteps
        output_dir: Output directory
        
    Returns:
        dict: Training results
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environments
    env = make_enhanced_env(data, config)
    eval_env = make_enhanced_env(data, config)
    
    # Create model
    policy_kwargs = {
        'net_arch': [256, 128, 64]  # Default architecture
    }
    
    # Check for custom architecture in config
    if 'net_arch' in config:
        if isinstance(config['net_arch'], list):
            policy_kwargs['net_arch'] = config['net_arch']
        elif config['net_arch'] == 'large':
            policy_kwargs['net_arch'] = [512, 256, 128]
        elif config['net_arch'] == 'medium':
            policy_kwargs['net_arch'] = [256, 128, 64]
        elif config['net_arch'] == 'small':
            policy_kwargs['net_arch'] = [128, 64, 32]
    
    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "logs"),
        policy_kwargs=policy_kwargs,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        gamma=config.get('gamma', 0.99),
        ent_coef=config.get('ent_coef', 0.01)
    )
    
    # Create callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "logs"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train model
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    training_time = time.time() - start_time
    
    # Save final model
    model_save_path = os.path.join(output_dir, f"final_model_{symbol}")
    model.save(model_save_path)
    
    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    # Get environment metrics
    env_unwrapped = env.envs[0].unwrapped
    info = env_unwrapped._get_info()
    
    # Prepare results
    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'win_rate': float(info['win_rate']),
        'max_drawdown': float(info['max_drawdown']),
        'portfolio_change': float(info['portfolio_change']),
        'total_trades': int(info['total_trades']),
        'sharpe_ratio': float(info.get('sharpe_ratio', 0)),
        'sortino_ratio': float(info.get('sortino_ratio', 0)),
        'calmar_ratio': float(info.get('calmar_ratio', 0)),
        'training_time': float(training_time),
        'training_time_formatted': time.strftime("%H:%M:%S", time.gmtime(training_time)),
        'model_path': model_save_path
    }
    
    # Save results
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    with open(os.path.join(output_dir, "results", f"{symbol}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Training completed for {symbol}: mean_reward={mean_reward:.2f}, win_rate={info['win_rate']:.2f}")
    
    return results

def train_with_trading_manager(
    data: pd.DataFrame, 
    symbol: str, 
    config_path: str,
    timesteps: int = 200000,
    enhanced: bool = True
) -> dict:
    """
    Train a model using the RLTradingStableManager, optionally enhanced.
    
    Args:
        data: Historical price data
        symbol: Symbol being trained
        config_path: Path to configuration file
        timesteps: Number of training timesteps
        enhanced: Whether to use enhanced training
        
    Returns:
        dict: Training results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create trading manager
    manager = RLTradingStableManager(config_path=config_path)
    
    # Enhance manager if requested
    if enhanced:
        manager = enhance_trading_manager(manager)
    
    # Initialize with data
    success = manager._initialize_with_data(data)
    if not success:
        logger.error("Failed to initialize trading manager with data")
        return {'status': 'error', 'message': 'Failed to initialize trading manager'}
    
    # Train model
    start_time = time.time()
    training_result = manager.train_model(
        symbol=symbol,
        data=data,
        epochs=timesteps,
        save_path=os.path.join("models/rl", "enhanced" if enhanced else "standard")
    )
    training_time = time.time() - start_time
    
    # Add training time
    training_result['training_time'] = training_time
    training_result['training_time_formatted'] = time.strftime("%H:%M:%S", time.gmtime(training_time))
    
    return training_result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced RL Trading Training')
    parser.add_argument('--symbol', type=str, default='MSFT', help='Symbol to train on')
    parser.add_argument('--config', type=str, default='config/lstm_config.json', help='Path to configuration file')
    parser.add_argument('--data-file', type=str, default='data/processed/MSFT_processed.csv', help='Path to data file (optional)')
    parser.add_argument('--lookback-days', type=int, default=730, help='Number of days to look back for data')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps for training')
    parser.add_argument('--enable-optuna', action='store_true', default=True, help='Enable Optuna optimization')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs for Optuna')
    parser.add_argument('--use-manager', action='store_true', help='Use trading manager for training')
    parser.add_argument('--output-dir', type=str, default='models/rl/enhanced', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/rl/enhanced', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create enhanced configuration
    enhanced_config = create_enhanced_config(config)
    
    # Update configuration file with enhanced settings
    if not os.path.exists(args.config + '.original'):
        # Make backup of original config
        import shutil
        shutil.copy(args.config, args.config + '.original')
    
    # Update config file with enhanced settings
    update_config_file(args.config, enhanced_config)
    
    # Load data
    data = None
    if args.data_file:
        # Load from file
        data = load_data_from_file(args.data_file)
    else:
        # Initialize IBKR
        ibkr_config = config.get('ibkr', {})
        ibkr = IBKRInterface(
            host=ibkr_config.get('host', '127.0.0.1'),
            port=ibkr_config.get('port', 4002),
            client_id=ibkr_config.get('client_id', 1)
        )
        
        # Connect to IBKR
        connected = ibkr.connect()
        if not connected:
            logger.error("Failed to connect to IBKR")
            return
        
        logger.info(f"Connected to IBKR at {ibkr.host}:{ibkr.port}")
        
        # Fetch and process data
        data = fetch_and_process_data(ibkr, args.symbol, args.lookback_days)
        
        # Disconnect from IBKR when done
        ibkr.disconnect()
    
    if data is None or data.empty:
        logger.error(f"No data available for {args.symbol}")
        return
    
    # Get environment configuration from enhanced config
    rl_config = enhanced_config.get('trading_manager', {}).get('rl', {})
    
    # Create a clean environment config
    env_config = {
        'initial_balance': rl_config.get('initial_balance', 10000.0),
        'max_position': rl_config.get('max_position', 100),
        'transaction_fee': rl_config.get('transaction_fee', 0.001),
        'reward_scaling': rl_config.get('reward_scaling', 1.0),
        'window_size': rl_config.get('window_size', 60),
        'reward_strategy': rl_config.get('reward_strategy', 'balanced'),
        'risk_free_rate': rl_config.get('risk_free_rate', 0.0),
        'risk_aversion': rl_config.get('risk_aversion', 1.0),
        'drawdown_penalty_factor': rl_config.get('drawdown_penalty_factor', 15.0),
        'holding_penalty_factor': rl_config.get('holding_penalty_factor', 0.1),
        'inactive_penalty_factor': rl_config.get('inactive_penalty_factor', 0.05),
        'consistency_reward_factor': rl_config.get('consistency_reward_factor', 0.2),
        'trend_following_factor': rl_config.get('trend_following_factor', 0.3),
        'win_streak_factor': rl_config.get('win_streak_factor', 0.1),
        'seed': args.seed
    }
    
    # Add network architecture
    if 'net_arch' in rl_config:
        env_config['net_arch'] = rl_config['net_arch']
    
    # Add training parameters
    for param in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 
                  'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm']:
        if param in rl_config:
            env_config[param] = rl_config[param]
    
    # Run training
    start_time = time.time()
    
    if args.enable_optuna:
        # Run Optuna optimization
        logger.info(f"Starting Optuna optimization with {args.n_trials} trials")
        result = run_enhanced_optimization(
            data=data,
            config=env_config,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            n_timesteps=args.timesteps // 2,  # Use half timesteps for trials
            final_timesteps=args.timesteps,
            n_jobs=args.n_jobs
        )
        
        # Extract and save best parameters to config file
        best_params = result['best_params']
        
        # Update environment parameters in config
        rl_config.update(best_params['env_params'])
        
        # Update PPO parameters in config
        for key, value in best_params['ppo_params'].items():
            rl_config[key] = value
        
        # Update network architecture in config
        rl_config['net_arch'] = best_params['policy_kwargs']['net_arch']
        
        # Save updated config
        update_config_file(args.config, enhanced_config)
        
        # Print summary
        print("\n===== Enhanced Optimization Results =====")
        print(f"Symbol: {args.symbol}")
        print(f"Total Trials: {args.n_trials}")
        print("\nBest Model Performance:")
        final_results = result['final_results']
        print(f"  Mean Reward: {final_results['mean_reward']:.4f}")
        print(f"  Win Rate: {final_results['win_rate'] * 100:.2f}%")
        print(f"  Portfolio Change: {final_results['portfolio_change'] * 100:.2f}%")
        print(f"  Sharpe Ratio: {final_results['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio: {final_results['sortino_ratio']:.4f}")
        print(f"  Max Drawdown: {final_results['max_drawdown'] * 100:.2f}%")
        print(f"  Total Trades: {final_results['total_trades']}")
        print(f"  Training Time: {final_results['training_time_formatted']}")
        print("============================================\n")
        
    elif args.use_manager:
        # Train with trading manager
        logger.info(f"Training with enhanced trading manager for {args.timesteps} timesteps")
        result = train_with_trading_manager(
            data=data,
            symbol=args.symbol,
            config_path=args.config,
            timesteps=args.timesteps,
            enhanced=True
        )
        
        # Print summary
        print("\n===== Trading Manager Training Results =====")
        print(f"Symbol: {args.symbol}")
        print(f"Status: {result.get('status', 'unknown')}")
        if 'evaluation' in result:
            eval_results = result['evaluation']
            print(f"  Mean Reward: {eval_results.get('avg_return', 0):.4f}")
            print(f"  Win Rate: {eval_results.get('avg_win_rate', 0) * 100:.2f}%")
            print(f"  Portfolio Change: {eval_results.get('avg_portfolio_change', 0) * 100:.2f}%")
        print(f"  Training Time: {result.get('training_time_formatted', 'unknown')}")
        print("=============================================\n")
        
    else:
        # Train without optimization
        logger.info(f"Training without optimization for {args.timesteps} timesteps")
        result = simple_rl_training(
            data=data,
            config=env_config,
            symbol=args.symbol,
            timesteps=args.timesteps,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n===== Simple Training Results =====")
        print(f"Symbol: {args.symbol}")
        print(f"Mean Reward: {result['mean_reward']:.4f}")
        print(f"Win Rate: {result['win_rate'] * 100:.2f}%")
        print(f"Portfolio Change: {result['portfolio_change'] * 100:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {result['sortino_ratio']:.4f}")
        print(f"Max Drawdown: {result['max_drawdown'] * 100:.2f}%")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Training Time: {result['training_time_formatted']}")
        print("===================================\n")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
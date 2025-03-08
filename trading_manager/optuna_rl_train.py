"""
RL Training with Hyperparameter Optimization using Optuna

This script trains a PPO agent for trading with automated hyperparameter optimization.
It uses Optuna to find the best hyperparameters and implements early stopping.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from trading_manager.rl_trading_stable import SimpleTradingEnv, RLTradingStableManager
from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, normalize_indicators, calculate_trend_indicator
from utils.data_utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'train_rl_optuna.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrialEvalCallback(EvalCallback):
    """
    Callback for evaluating and saving best model during optimization.
    Extends the EvalCallback to add trial pruning functionality.
    """
    
    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=10000,
        deterministic=True,
        verbose=0,
        best_model_save_path=None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
        )
        self.trial = trial
        self.is_pruned = False
        self.eval_idx = 0
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Call original callback implementation
            continue_training = super()._on_step()
            
            # Get reward from last evaluation
            self.eval_idx += 1
            
            # Report current reward to Optuna for pruning check
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
                
            return continue_training
        
        return True


def fetch_and_process_data(ibkr: IBKRInterface, symbol: str, lookback_days: int = 365) -> pd.DataFrame:
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
        
        # Remove NaN values
        df.dropna(inplace=True)
        
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


def make_env(data, config):
    """
    Create a trading environment wrapped in DummyVecEnv.
    
    Args:
        data: Historical price data
        config: Environment configuration
        
    Returns:
        DummyVecEnv: Vectorized environment
    """
    def _init():
        # Create and wrap the environment
        env = SimpleTradingEnv(
            data=data, 
            initial_balance=config.get('initial_balance', 10000.0),
            max_position=config.get('max_position', 100),
            transaction_fee=config.get('transaction_fee', 0.001)
        )
        # Wrap with Monitor for training metrics
        env = Monitor(env)
        return env
    
    # Create vectorized environment
    return DummyVecEnv([_init])


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """
    Sample PPO hyperparameters from the search space.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Sampled hyperparameters
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 1024, 8192, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    n_epochs = trial.suggest_int("n_epochs", 5, 30)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    
    # Build neural network architecture
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
        "large": [dict(pi=[256, 256], vf=[256, 256])],
    }[trial.suggest_categorical("net_arch", ["small", "medium", "large"])]
    
    # Activation function
    activation_fn_name = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu"]
    )
    activation_fn = {
        "tanh": stable_baselines3.common.torch_layers.nn.Tanh,
        "relu": stable_baselines3.common.torch_layers.nn.ReLU,
        "elu": stable_baselines3.common.torch_layers.nn.ELU,
    }[activation_fn_name]
    
    # Return hyperparameters
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


def objective(trial: optuna.Trial, data: pd.DataFrame, config: dict, n_timesteps: int) -> float:
    """
    Optuna objective function to minimize.
    
    Args:
        trial: Optuna trial object
        data: Historical price data
        config: Environment configuration
        n_timesteps: Number of timesteps for training
        
    Returns:
        float: Negative mean reward (to be minimized)
    """
    # Sample hyperparameters
    kwargs = sample_ppo_params(trial)
    
    # Create environments
    env = make_env(data, config)
    eval_env = make_env(data, config)
    
    # Create logger dir
    log_dir = f"logs/optuna/{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Add early stopping callback
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=5,
        verbose=1
    )
    
    # Create evaluation callback
    eval_callback = TrialEvalCallback(
        eval_env=eval_env,
        trial=trial,
        best_model_save_path=f"models/rl/optuna/trial_{trial.number}",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    
    # Create model
    model = PPO("MlpPolicy", env, verbose=1, **kwargs)
    
    try:
        # Train model
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_callback,
        )
        
        # Evaluate model after training
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        # Check if pruned
        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
        
        # Get trade statistics
        env_unwrapped = env.envs[0].unwrapped
        win_rate = env_unwrapped.winning_trades / max(1, env_unwrapped.total_trades)
        max_drawdown = env_unwrapped.max_drawdown
        portfolio_change = (env_unwrapped.portfolio_value - env_unwrapped.initial_balance) / env_unwrapped.initial_balance
        
        # Log results
        logger.info(f"Trial {trial.number}: mean_reward={mean_reward:.2f}, "
                   f"win_rate={win_rate:.2f}, portfolio_change={portfolio_change:.2f}")
        
        # Record additional metrics
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("portfolio_change", portfolio_change)
        trial.set_user_attr("std_reward", std_reward)
        trial.set_user_attr("total_trades", env_unwrapped.total_trades)
        
        # For optimization, we use a combined metric: mean_reward + win_rate bonus - drawdown penalty
        # This encourages models with good rewards, high win rates, and low drawdowns
        combined_score = mean_reward + (win_rate * 5) - (max_drawdown * 10)
        
        return combined_score
        
    except (AssertionError, optuna.exceptions.TrialPruned) as e:
        # Return a poor score if trial fails
        logger.error(f"Error in trial {trial.number}: {e}")
        return float("-inf")


def optimize_agent(
    data: pd.DataFrame,
    config: dict,
    n_trials: int = 50,
    n_timesteps: int = 100000,
    n_jobs: int = 1
) -> dict:
    """
    Run hyperparameter optimization.
    
    Args:
        data: Historical price data
        config: Environment configuration
        n_trials: Number of trials
        n_timesteps: Number of timesteps for training
        n_jobs: Number of parallel jobs
        
    Returns:
        dict: Best hyperparameters
    """
    # Create study directory
    study_name = f"ppo_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_storage = f"sqlite:///models/rl/optuna/{study_name}.db"
    os.makedirs("models/rl/optuna", exist_ok=True)
    
    # Define the sampler and pruner
    sampler = TPESampler(n_startup_trials=5, seed=config.get('seed', 42))
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        storage=study_storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, data, config, n_timesteps),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    
    # Save study results
    os.makedirs("results/optuna", exist_ok=True)
    results_file = f"results/optuna/{study_name}_results.json"
    
    # Extract trials data
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            params = {**trial.params}
            
            # Handle non-serializable objects
            if "policy_kwargs" in params and "activation_fn" in params["policy_kwargs"]:
                params["policy_kwargs"]["activation_fn"] = params["policy_kwargs"]["activation_fn"].__name__
                
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": params,
                "user_attrs": trial.user_attrs,
            }
            trials_data.append(trial_data)
    
    # Save results
    with open(results_file, "w") as f:
        json.dump({
            "study_name": study_name,
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_params,
                "user_attrs": study.best_trial.user_attrs,
            },
            "trials": trials_data,
        }, f, indent=2)
    
    logger.info(f"Study results saved to {results_file}")
    
    # Plot optimization results
    try:
        # Plot optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(f"results/optuna/{study_name}_history.png")
        
        # Plot parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(f"results/optuna/{study_name}_importances.png")
        
        # Plot parallel coordinate
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_image(f"results/optuna/{study_name}_parallel.png")
        
        logger.info("Created visualization plots")
    except Exception as e:
        logger.warning(f"Could not create some plots: {e}")
    
    return study.best_params


def train_with_best_params(data: pd.DataFrame, best_params: dict, config: dict, n_timesteps: int = 200000) -> dict:
    """
    Train the final model with the best hyperparameters.
    
    Args:
        data: Historical price data
        best_params: Best hyperparameters
        config: Environment configuration
        n_timesteps: Number of timesteps for training
        
    Returns:
        dict: Training results
    """
    # Create environments
    env = make_env(data, config)
    eval_env = make_env(data, config)
    
    # Create model with best parameters
    model = PPO("MlpPolicy", env, verbose=1, **best_params)
    
    # Fix activation_fn if it's a string
    if isinstance(model.policy_kwargs.get("activation_fn", None), str):
        activation_fn = {
            "tanh": stable_baselines3.common.torch_layers.nn.Tanh,
            "relu": stable_baselines3.common.torch_layers.nn.ReLU,
            "elu": stable_baselines3.common.torch_layers.nn.ELU,
        }[model.policy_kwargs["activation_fn"]]
        model.policy_kwargs["activation_fn"] = activation_fn
    
    # Create best model directory
    best_model_dir = "models/rl/best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Create callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path="logs/best_model",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    # Train model
    start_time = time.time()
    model.learn(total_timesteps=n_timesteps, callback=eval_callback)
    training_time = time.time() - start_time
    
    # Save final model
    model.save(os.path.join(best_model_dir, "final_model"))
    
    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    
    # Get trade statistics
    env_unwrapped = env.envs[0].unwrapped
    win_rate = env_unwrapped.winning_trades / max(1, env_unwrapped.total_trades)
    max_drawdown = env_unwrapped.max_drawdown
    portfolio_change = (env_unwrapped.portfolio_value - env_unwrapped.initial_balance) / env_unwrapped.initial_balance
    
    # Log results
    logger.info(f"Final model trained with mean_reward={mean_reward:.2f}, "
               f"win_rate={win_rate:.2f}, portfolio_change={portfolio_change:.2f}")
    
    # Save training results
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "portfolio_change": float(portfolio_change),
        "total_trades": int(env_unwrapped.total_trades),
        "training_time": float(training_time),
        "training_time_formatted": time.strftime("%H:%M:%S", time.gmtime(training_time)),
        "n_timesteps": int(n_timesteps),
        "best_params": best_params,
    }
    
    # Save results
    os.makedirs("results/best_model", exist_ok=True)
    with open(f"results/best_model/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main function to run hyperparameter optimization and training."""
    parser = argparse.ArgumentParser(description='RL Trading with Optuna Hyperparameter Optimization')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to train on')
    parser.add_argument('--config', type=str, default='config/lstm_config.json', help='Path to configuration file')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials for optimization')
    parser.add_argument('--n-timesteps', type=int, default=100000, help='Number of timesteps for training')
    parser.add_argument('--final-timesteps', type=int, default=200000, help='Number of timesteps for final training')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--lookback-days', type=int, default=365, help='Number of days to look back for data')
    parser.add_argument('--data-file', type=str, default=None, help='Path to data file (optional)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/optuna', exist_ok=True)
    os.makedirs('models/rl/optuna', exist_ok=True)
    os.makedirs('results/optuna', exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Initialize IBKR (only if we need to fetch data)
        ibkr = None
        data = None
        
        if args.data_file:
            # Load data from file
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
        
        if data is None or data.empty:
            logger.error(f"No data available for {args.symbol}")
            if ibkr and ibkr.connected:
                ibkr.disconnect()
            return
        
        # Get trading environment configuration
        env_config = {
            'initial_balance': config.get('trading_manager', {}).get('rl', {}).get('initial_balance', 10000.0),
            'max_position': config.get('trading_manager', {}).get('rl', {}).get('max_position', 100),
            'transaction_fee': config.get('trading_manager', {}).get('rl', {}).get('transaction_fee', 0.001),
            'seed': args.seed
        }
        
        # Run hyperparameter optimization
        """ Executed code below to avoid running optimization
        logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
        best_params = optimize_agent(
            data=data,
            config=env_config,
            n_trials=args.n_trials,
            n_timesteps=args.n_timesteps,
            n_jobs=args.n_jobs
        )
        """
        best_params = {
            "learning_rate": 5.984253246234144e-05,
            "n_steps": 2171,
            "batch_size": 503,
            "n_epochs": 19,
            "gamma": 0.9914553983193832,
            "gae_lambda": 0.9491142291825448,
            "clip_range": 0.27625776550096276,
            "ent_coef": 0.06304257572865528,
            "vf_coef": 0.7194378115811361,
            "max_grad_norm": 1.6223447963637094,
            "net_arch": "large",
            "activation_fn": "relu"
        }

        # Train final model with best parameters
        logger.info(f"Training final model with best parameters")
        final_results = train_with_best_params(
            data=data,
            best_params=best_params,
            config=env_config,
            n_timesteps=args.final_timesteps
        )
        
        # Print summary
        print("\n===== Optimization and Training Summary =====")
        print(f"Symbol: {args.symbol}")
        print(f"Total Trials: {args.n_trials}")
        print(f"Best Parameters: {json.dumps(best_params, indent=2)}")
        print("\nFinal Model Performance:")
        print(f"  Mean Reward: {final_results['mean_reward']:.4f}")
        print(f"  Win Rate: {final_results['win_rate'] * 100:.2f}%")
        print(f"  Portfolio Change: {final_results['portfolio_change'] * 100:.2f}%")
        print(f"  Max Drawdown: {final_results['max_drawdown'] * 100:.2f}%")
        print(f"  Total Trades: {final_results['total_trades']}")
        print(f"  Training Time: {final_results['training_time_formatted']}")
        print("==============================================\n")
        
        # Disconnect from IBKR if connected
        if ibkr and ibkr.connected:
            ibkr.disconnect()
            logger.info("Disconnected from IBKR")
        
    except Exception as e:
        logger.error(f"Error in optimization/training: {e}")
        if 'ibkr' in locals() and ibkr and ibkr.connected:
            ibkr.disconnect()
            logger.info("Disconnected from IBKR")


if __name__ == "__main__":
    main()
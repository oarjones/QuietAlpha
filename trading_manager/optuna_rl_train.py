"""
Enhanced RL Training with Optimized Hyperparameter Optimization

This module improves the existing optuna_rl_train.py with:
1. Enhanced hyperparameter search space
2. Refined objective function incorporating financial metrics
3. Better architecture search
4. Improved reward function integration
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
        logging.FileHandler(os.path.join('logs', 'enhanced_rl_optuna.log')),
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


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe ratio of a series of returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (defaults to 0)
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns (over risk-free rate)
    excess_returns = returns - risk_free_rate
    
    # Calculate Sharpe ratio
    sharpe_ratio = 0.0
    if np.std(excess_returns) > 0:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    
    # Annualize Sharpe ratio (assuming daily returns)
    sharpe_ratio = sharpe_ratio * np.sqrt(252)
    
    return sharpe_ratio


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sortino ratio (variant of Sharpe that only penalizes downside risk).
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (defaults to 0)
        
    Returns:
        float: Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate
    
    # Calculate downside returns (negative returns only)
    downside_returns = excess_returns[excess_returns < 0]
    
    # Calculate Sortino ratio
    sortino_ratio = 0.0
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = np.mean(excess_returns) / np.std(downside_returns)
    
    # Annualize Sortino ratio (assuming daily returns)
    sortino_ratio = sortino_ratio * np.sqrt(252)
    
    return sortino_ratio


def calculate_calmar_ratio(returns, max_drawdown):
    """
    Calculate the Calmar ratio (annualized return divided by maximum drawdown).
    
    Args:
        returns: Array of daily returns
        max_drawdown: Maximum drawdown (as a positive fraction)
        
    Returns:
        float: Calmar ratio
    """
    if max_drawdown <= 0 or len(returns) == 0:
        return 0.0
    
    # Calculate annualized return
    annualized_return = np.mean(returns) * 252
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown
    
    return calmar_ratio


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


def process_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Process a dataframe with price data.
    
    Args:
        df: Pandas Dataframe
        
    Returns:
        pd.DataFrame: Processed data
    """
    try:
        if df.empty:
            logger.warning(f"No data")
            return pd.DataFrame()
        
        # Process data
        logger.info(f"Processing data ({len(df)} rows)")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, include_all=True)
        
        # Calculate trend indicator
        df = calculate_trend_indicator(df)
        
        # Normalize indicators
        df = normalize_indicators(df)
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Processed data: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
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


def sample_enhanced_ppo_params(trial: optuna.Trial) -> dict:
    """
    Enhanced hyperparameter sampling with expanded search space.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Sampled hyperparameters
    """
    # Learning rate - expanded range and log scale for better exploration
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-3, log=True)
    
    # PPO buffer size and steps
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    
    # Batch size - ensure it's smaller than n_steps
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    batch_size = min(batch_size, n_steps)  # Ensure batch_size <= n_steps
    
    # Number of epochs
    n_epochs = trial.suggest_int("n_epochs", 3, 30)
    
    # Discount factor - finer control for financial time horizons
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    
    # GAE lambda - generalized advantage estimation
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    
    # PPO clip range
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # Entropy coefficient - expanded for more exploration options
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.2)
    
    # Value function coefficient - increased upper bound
    vf_coef = trial.suggest_float("vf_coef", 0.4, 1.5)
    
    # Max gradient norm - expanded range
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 10.0, log=True)
    
    # ENHANCED: Neural network architecture exploration
    # Network width (number of neurons per layer)
    net_width = trial.suggest_categorical("net_width", [64, 128, 256, 512])
    
    # Network depth (number of layers)
    net_depth = trial.suggest_int("net_depth", 1, 4) 
    
    # Build architecture based on width and depth
    net_arch = []
    for _ in range(net_depth):
        net_arch.append(net_width)
    
    # ENHANCED: Activation function selection
    activation_fn_name = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    
    activation_fn = {
        "tanh": stable_baselines3.common.torch_layers.nn.Tanh,
        "relu": stable_baselines3.common.torch_layers.nn.ReLU,
        "elu": stable_baselines3.common.torch_layers.nn.ELU,
        "leaky_relu": stable_baselines3.common.torch_layers.nn.LeakyReLU,
    }[activation_fn_name]
    
    # ENHANCED: Layer normalization option
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])
    
    # Build complete policy kwargs
    policy_kwargs = {
        "net_arch": [dict(pi=net_arch, vf=net_arch)],
        "activation_fn": activation_fn,
    }
    
    # Add layer normalization if selected
    if use_layer_norm:
        policy_kwargs["normalize_layers"] = True
    
    # Return complete hyperparameter dictionary
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
        "policy_kwargs": policy_kwargs,
    }


def enhanced_objective(trial: optuna.Trial, data: pd.DataFrame, config: dict, n_timesteps: int) -> float:
    """
    Enhanced Optuna objective function with comprehensive financial metrics evaluation.
    
    Args:
        trial: Optuna trial object
        data: Historical price data
        config: Environment configuration
        n_timesteps: Number of timesteps for training
        
    Returns:
        float: Combined performance score (to be maximized)
    """
    # Sample hyperparameters
    kwargs = sample_enhanced_ppo_params(trial)
    
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
        
        # Evaluate model after training with more episodes for better statistics
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        # Check if pruned
        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
        
        # Get environment object for financial metrics
        env_unwrapped = env.envs[0].unwrapped
        
        # Collect trade statistics
        win_rate = env_unwrapped.winning_trades / max(1, env_unwrapped.total_trades)
        max_drawdown = env_unwrapped.max_drawdown
        portfolio_change = (env_unwrapped.portfolio_value - env_unwrapped.initial_balance) / env_unwrapped.initial_balance
        total_trades = env_unwrapped.total_trades
        
        # Extract daily returns from trade history for financial metrics
        # Construct a series of daily returns from portfolio value changes
        daily_returns = []
        if hasattr(env_unwrapped, 'trades') and len(env_unwrapped.trades) > 1:
            # Sort trades by step (time)
            sorted_trades = sorted(env_unwrapped.trades, key=lambda x: x['step'])
            
            # Group trades by day and calculate daily returns
            current_day = None
            day_start_value = env_unwrapped.initial_balance
            
            for trade in sorted_trades:
                # Assuming each 24 steps is a day (for hourly data)
                trade_day = trade['step'] // 24
                
                if current_day is None:
                    current_day = trade_day
                
                elif current_day != trade_day:
                    # Calculate return for the previous day
                    if 'portfolio_value' in trade:
                        daily_return = (trade['portfolio_value'] - day_start_value) / day_start_value
                        daily_returns.append(daily_return)
                        day_start_value = trade['portfolio_value']
                    
                    current_day = trade_day
        
        # Calculate financial metrics
        sharpe_ratio = calculate_sharpe_ratio(np.array(daily_returns)) if daily_returns else 0
        sortino_ratio = calculate_sortino_ratio(np.array(daily_returns)) if daily_returns else 0
        calmar_ratio = calculate_calmar_ratio(np.array(daily_returns), max_drawdown) if daily_returns and max_drawdown > 0 else 0
        
        # Log results
        logger.info(f"Trial {trial.number}: mean_reward={mean_reward:.2f}, "
                   f"win_rate={win_rate:.2f}, portfolio_change={portfolio_change:.2f}, "
                   f"sharpe={sharpe_ratio:.2f}, sortino={sortino_ratio:.2f}, calmar={calmar_ratio:.2f}")
        
        # Record additional metrics
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("portfolio_change", portfolio_change)
        trial.set_user_attr("std_reward", std_reward)
        trial.set_user_attr("total_trades", total_trades)
        trial.set_user_attr("sharpe_ratio", sharpe_ratio)
        trial.set_user_attr("sortino_ratio", sortino_ratio)
        trial.set_user_attr("calmar_ratio", calmar_ratio)
        
        # ENHANCED: Comprehensive scoring that balances multiple financial objectives
        # This score combines reward, risk-adjusted metrics, and trading activity
        
        # Base component: Portfolio performance
        performance_score = mean_reward + (portfolio_change * 5)  # Reward raw returns
        
        # Risk-adjusted component: Higher is better for all these ratios
        risk_adjusted_score = (
            sharpe_ratio * 2.0 +    # Sharpe: return per unit of risk
            sortino_ratio * 1.5 +   # Sortino: return per unit of downside risk  
            calmar_ratio * 1.0      # Calmar: return per unit of max drawdown
        )
        
        # Trading activity component: Reward systems that make enough trades
        # but penalize excessive trading or very low win rates
        activity_score = 0
        if total_trades >= 5:  # Minimum trades to be meaningful
            # Reward higher win rates, but penalize excessive trading
            activity_score = (win_rate * 5.0) - (min(1.0, total_trades / 200) * 2.0)
        else:
            # Penalize too few trades
            activity_score = -5.0
        
        # Drawdown penalty: Heavily penalize large drawdowns
        drawdown_penalty = max_drawdown * 15.0  # Scaling factor to make drawdown important
        
        # Combine components into final score
        combined_score = (
            performance_score +     # Raw performance
            risk_adjusted_score +   # Risk-adjusted metrics
            activity_score -        # Trading behavior
            drawdown_penalty        # Risk control
        )
        
        logger.info(f"Trial {trial.number} score components: "
                   f"performance={performance_score:.2f}, "
                   f"risk_adjusted={risk_adjusted_score:.2f}, "
                   f"activity={activity_score:.2f}, "
                   f"drawdown_penalty={drawdown_penalty:.2f}, "
                   f"final={combined_score:.2f}")
        
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
    Run enhanced hyperparameter optimization with improved study configuration.
    
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
    study_name = f"enhanced_ppo_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_storage = f"sqlite:///models/rl/optuna/{study_name}.db"
    os.makedirs("models/rl/optuna", exist_ok=True)
    
    # Define the sampler and pruner
    # ENHANCED: Improved sampler configuration
    sampler = TPESampler(
        n_startup_trials=10,  # More exploration before exploitation
        seed=config.get('seed', 42),
        multivariate=True,  # Consider parameter correlations
        constant_liar=True  # Better parallel optimization
    )
    
    # ENHANCED: More patient pruner
    pruner = MedianPruner(
        n_startup_trials=10,  # Wait longer before pruning
        n_warmup_steps=5,
        interval_steps=2
    )
    
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
            lambda trial: enhanced_objective(trial, data, config, n_timesteps),
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
                "params": {k: v for k, v in study.best_params.items() 
                           if k != "policy_kwargs"},  # Handle policy_kwargs separately
                "policy_kwargs": {
                    k: (v.__name__ if k == "activation_fn" else v) 
                    for k, v in study.best_params.get("policy_kwargs", {}).items()
                },
                "user_attrs": study.best_trial.user_attrs,
            },
            "trials": trials_data,
        }, f, indent=2)
    
    logger.info(f"Study results saved to {results_file}")
    
    # ENHANCED: Create more comprehensive visualizations
    try:
        import optuna.visualization as vis
        
        os.makedirs(f"results/optuna/{study_name}", exist_ok=True)
        
        # Plot optimization history
        fig1 = vis.plot_optimization_history(study)
        fig1.write_image(f"results/optuna/{study_name}/history.png")
        
        # Plot parameter importances
        fig2 = vis.plot_param_importances(study)
        fig2.write_image(f"results/optuna/{study_name}/importances.png")
        
        # Plot parallel coordinate
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_image(f"results/optuna/{study_name}/parallel.png")
        
        # Plot slice plots for important parameters
        param_importances = optuna.importance.get_param_importances(study)
        top_params = list(param_importances.keys())[:5]  # Top 5 important params
        
        for param in top_params:
            try:
                fig = vis.plot_slice(study, params=[param])
                fig.write_image(f"results/optuna/{study_name}/slice_{param}.png")
            except:
                logger.warning(f"Could not create slice plot for {param}")
        
        # Plot contour plots for top parameter combinations
        if len(top_params) >= 2:
            for i in range(min(len(top_params), 3)):
                for j in range(i+1, min(len(top_params), 4)):
                    try:
                        fig = vis.plot_contour(study, params=[top_params[i], top_params[j]])
                        fig.write_image(f"results/optuna/{study_name}/contour_{top_params[i]}_{top_params[j]}.png")
                    except:
                        pass
        
        logger.info("Created enhanced visualization plots")
    except Exception as e:
        logger.warning(f"Could not create all plots: {e}")
    
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
    
    # Process the best parameters to make them compatible with PPO
    processed_params = best_params.copy()
    
    # Convert activation function name to actual function if needed
    if ("policy_kwargs" in processed_params and 
        "activation_fn" in processed_params["policy_kwargs"] and 
        isinstance(processed_params["policy_kwargs"]["activation_fn"], str)):
        
        activation_map = {
            "tanh": stable_baselines3.common.torch_layers.nn.Tanh,
            "relu": stable_baselines3.common.torch_layers.nn.ReLU,
            "elu": stable_baselines3.common.torch_layers.nn.ELU,
            "leaky_relu": stable_baselines3.common.torch_layers.nn.LeakyReLU,
        }
        
        activation_fn_name = processed_params["policy_kwargs"]["activation_fn"]
        processed_params["policy_kwargs"]["activation_fn"] = activation_map.get(
            activation_fn_name, stable_baselines3.common.torch_layers.nn.ReLU
        )
    
    # Log the processed parameters
    logger.info(f"Using processed parameters: {processed_params}")
    
    # Create model with processed parameters
    model = PPO("MlpPolicy", env, verbose=1, **processed_params)
    
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
    
    # ENHANCED: More comprehensive evaluation
    # Evaluate model with more episodes for better statistics
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    
    # Get environment unwrapped for additional metrics
    env_unwrapped = env.envs[0].unwrapped
    
    # Get trade statistics
    win_rate = env_unwrapped.winning_trades / max(1, env_unwrapped.total_trades)
    max_drawdown = env_unwrapped.max_drawdown
    portfolio_change = (env_unwrapped.portfolio_value - env_unwrapped.initial_balance) / env_unwrapped.initial_balance
    total_trades = env_unwrapped.total_trades
    
    # Extract daily returns for financial metrics
    daily_returns = []
    if hasattr(env_unwrapped, 'trades') and len(env_unwrapped.trades) > 1:
        # Sort trades by step (time)
        sorted_trades = sorted(env_unwrapped.trades, key=lambda x: x['step'])
        
        # Group trades by day and calculate daily returns
        current_day = None
        day_start_value = env_unwrapped.initial_balance
        
        for trade in sorted_trades:
            # Assuming each 24 steps is a day (for hourly data)
            trade_day = trade['step'] // 24
            
            if current_day is None:
                current_day = trade_day
            
            elif current_day != trade_day:
                # Calculate return for the previous day
                if 'portfolio_value' in trade:
                    daily_return = (trade['portfolio_value'] - day_start_value) / day_start_value
                    daily_returns.append(daily_return)
                    day_start_value = trade['portfolio_value']
                
                current_day = trade_day
    
    # Calculate financial metrics
    sharpe_ratio = calculate_sharpe_ratio(np.array(daily_returns)) if daily_returns else 0
    sortino_ratio = calculate_sortino_ratio(np.array(daily_returns)) if daily_returns else 0
    calmar_ratio = calculate_calmar_ratio(np.array(daily_returns), max_drawdown) if daily_returns and max_drawdown > 0 else 0
    
    # Log results
    logger.info(f"Final model trained with mean_reward={mean_reward:.2f}, "
               f"win_rate={win_rate:.2f}, portfolio_change={portfolio_change:.2f}, "
               f"sharpe={sharpe_ratio:.2f}, sortino={sortino_ratio:.2f}")
    
    # Save detailed results
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "portfolio_change": float(portfolio_change),
        "total_trades": int(total_trades),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "training_time": float(training_time),
        "training_time_formatted": time.strftime("%H:%M:%S", time.gmtime(training_time)),
        "n_timesteps": int(n_timesteps),
        "best_params": best_params,
    }
    
    # Save results
    os.makedirs("results/best_model", exist_ok=True)
    with open(f"results/best_model/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create detailed performance plots
    try:
        plot_trading_performance(env_unwrapped, best_model_dir)
    except Exception as e:
        logger.error(f"Error creating performance plots: {e}")
    
    return results


def plot_trading_performance(env_unwrapped, output_dir):
    """
    Create comprehensive performance plots for the trading model.
    
    Args:
        env_unwrapped: Unwrapped environment with trade history
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Portfolio Value Over Time
    if hasattr(env_unwrapped, 'trades') and len(env_unwrapped.trades) > 0:
        # Sort trades by step
        sorted_trades = sorted(env_unwrapped.trades, key=lambda x: x['step'])
        
        # Extract steps and portfolio values
        steps = [0] + [t['step'] for t in sorted_trades]
        portfolio_values = [env_unwrapped.initial_balance]
        
        # Calculate portfolio value at each trade
        for trade in sorted_trades:
            if 'portfolio_value' in trade:
                portfolio_values.append(trade['portfolio_value'])
            else:
                # Estimate if not directly available
                last_value = portfolio_values[-1]
                portfolio_values.append(last_value)
        
        # Plot portfolio value
        plt.figure(figsize=(14, 7))
        plt.plot(steps, portfolio_values, marker='o', markersize=3, linestyle='-', linewidth=1)
        plt.title('Portfolio Value Over Time', fontsize=16)
        plt.xlabel('Trading Steps', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for initial balance
        plt.axhline(y=env_unwrapped.initial_balance, color='r', linestyle='--', alpha=0.5, 
                    label=f'Initial Balance (${env_unwrapped.initial_balance:.2f})')
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'portfolio_value.png'), dpi=300)
        plt.close()
        
        # 2. Trade Analysis: Wins vs. Losses
        if hasattr(env_unwrapped, 'trades'):
            # Extract profit/loss from trades
            trade_pnls = []
            for trade in sorted_trades:
                if 'profit_loss' in trade and trade['type'] == 'sell':
                    trade_pnls.append(trade['profit_loss'])
            
            if trade_pnls:
                # Split into winning and losing trades
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]
                
                # Create histogram of trade P&Ls
                plt.figure(figsize=(14, 7))
                
                if winning_trades:
                    plt.hist(winning_trades, bins=20, alpha=0.7, color='green', label='Winning Trades')
                
                if losing_trades:
                    plt.hist(losing_trades, bins=20, alpha=0.7, color='red', label='Losing Trades')
                
                plt.title('Distribution of Trade Profits/Losses', fontsize=16)
                plt.xlabel('Profit/Loss ($)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Format x-axis as currency
                plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'trade_distribution.png'), dpi=300)
                plt.close()
                
                # 3. Cumulative P&L
                cumulative_pnl = np.cumsum(trade_pnls)
                
                plt.figure(figsize=(14, 7))
                plt.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
                plt.title('Cumulative Profit/Loss', fontsize=16)
                plt.xlabel('Trade Number', fontsize=12)
                plt.ylabel('Cumulative P&L ($)', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Format y-axis as currency
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'cumulative_pnl.png'), dpi=300)
                plt.close()
                
                # 4. Drawdown chart
                if hasattr(env_unwrapped, 'portfolio_values') and len(env_unwrapped.portfolio_values) > 0:
                    portfolio_values = env_unwrapped.portfolio_values
                    
                    # Calculate drawdown
                    peak = np.maximum.accumulate(portfolio_values)
                    drawdown = (peak - portfolio_values) / peak
                    
                    plt.figure(figsize=(14, 7))
                    plt.plot(range(len(drawdown)), drawdown * 100, linewidth=2)
                    plt.title('Portfolio Drawdown', fontsize=16)
                    plt.xlabel('Steps', fontsize=12)
                    plt.ylabel('Drawdown (%)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # Format y-axis as percentage
                    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}%'))
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'drawdown.png'), dpi=300)
                    plt.close()


def main():
    """Main function to run enhanced RL training with Optuna optimization."""
    parser = argparse.ArgumentParser(description='Enhanced RL Trading with Optuna Hyperparameter Optimization')
    parser.add_argument('--symbol', type=str, default='MSFT', help='Symbol to train on')
    parser.add_argument('--config', type=str, default='config/lstm_config.json', help='Path to configuration file')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials for optimization')
    parser.add_argument('--n-timesteps', type=int, default=100000, help='Number of timesteps for training')
    parser.add_argument('--final-timesteps', type=int, default=200000, help='Number of timesteps for final training')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--lookback-days', type=int, default=730, help='Number of days to look back for data')
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
            data = process_data(data)
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
        logger.info(f"Starting enhanced hyperparameter optimization with {args.n_trials} trials")
        best_params = optimize_agent(
            data=data,
            config=env_config,
            n_trials=args.n_trials,
            n_timesteps=args.n_timesteps,
            n_jobs=args.n_jobs
        )
        
        # Train final model with best parameters
        logger.info(f"Training final model with best parameters")
        final_results = train_with_best_params(
            data=data,
            best_params=best_params,
            config=env_config,
            n_timesteps=args.final_timesteps
        )
        
        # Print summary
        print("\n===== Enhanced Optimization and Training Summary =====")
        print(f"Symbol: {args.symbol}")
        print(f"Total Trials: {args.n_trials}")
        print(f"Best Parameters: {json.dumps(best_params, indent=2, default=str)}")
        print("\nFinal Model Performance:")
        print(f"  Mean Reward: {final_results['mean_reward']:.4f}")
        print(f"  Win Rate: {final_results['win_rate'] * 100:.2f}%")
        print(f"  Portfolio Change: {final_results['portfolio_change'] * 100:.2f}%")
        print(f"  Sharpe Ratio: {final_results['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio: {final_results['sortino_ratio']:.4f}")
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
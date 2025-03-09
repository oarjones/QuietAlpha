"""
Integration Helpers for Enhanced Trading Environment

This module provides helper functions to integrate the EnhancedTradingEnv
with the existing trading system in RLTradingStableManager.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from trading_manager.rl_trading_stable import EnhancedTradingEnv, RLTradingStableManager

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from utils.data_utils import load_config

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement

from stable_baselines3.common.evaluation import evaluate_policy
import time
import json
from datetime import datetime, timedelta

import logging
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)


class DetailedEvalCallback(EvalCallback):
    """
    Versión corregida de DetailedEvalCallback que ejecuta episodios completos
    para obtener métricas financieras precisas.
    """
    
    def __init__(
        self,
        eval_env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        render=False,
        verbose=1,
        warn=True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.start_time = time.time()
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Call parent evaluation method
            parent_continue = super()._on_step()
            
            # Run full episode evaluation to get accurate metrics
            if hasattr(self.eval_env, 'run_full_episode'):
                try:
                    metrics = self.eval_env.run_full_episode(self.model, self.deterministic, reset_env=False)
                    
                    # Extract financial metrics
                    win_rate = metrics.get('win_rate', 0) * 100
                    portfolio_change = metrics.get('portfolio_change', 0) * 100
                    max_drawdown = metrics.get('max_drawdown', 0) * 100
                    total_trades = metrics.get('total_trades', 0)
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    sortino_ratio = metrics.get('sortino_ratio', 0)
                    
                    # Calculate elapsed time
                    elapsed = time.time() - self.start_time
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    
                    # Log more detailed information
                    if self.verbose > 0:
                        print(f"\n{'=' * 50}")
                        print(f"Evaluation at {self.num_timesteps} timesteps ({elapsed_str})")
                        print(f"Mean reward: {self.last_mean_reward:.2f}")
                        print(f"Win rate: {win_rate:.2f}%")
                        print(f"Portfolio change: {portfolio_change:.2f}%")
                        print(f"Max drawdown: {max_drawdown:.2f}%")
                        print(f"Total trades: {total_trades}")
                        print(f"Sharpe ratio: {sharpe_ratio:.4f}")
                        print(f"Sortino ratio: {sortino_ratio:.4f}")
                        print(f"{'=' * 50}\n")
                
                except Exception as e:
                    # If failed to get additional metrics, just show a message
                    logger.error(f"Error obteniendo métricas detalladas: {e}")
                    if self.verbose > 0:
                        print(f"Error obteniendo métricas detalladas: {e}")
            
            return parent_continue
        
        return True


class DetailedTrialEvalCallback(EvalCallback):
    """
    Versión mejorada del callback de evaluación para pruebas de Optuna que ejecuta
    correctamente un episodio completo para obtener métricas financieras precisas.
    """
    
    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        verbose=1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.is_pruned = False
        self.eval_idx = 0
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        
        # Print header for evaluations
        if self.verbose > 0:
            print("\nEVALUATION METRICS DURING OPTIMIZATION")
            print(f"{'Timesteps':<10} {'Reward':<10} {'Win Rate':<10} {'Profit %':<10} {'Drawdown':<10} {'Trades':<8} {'Sharpe':<8}")
            print("-" * 70)
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Call original evaluation for reward calculation
            continue_training = super()._on_step()
            
            # Report reward to Optuna for pruning
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Get financial metrics by running a complete episode
            # El parámetro reset_env=False es crucial para no perder métricas acumuladas
            if hasattr(self.eval_env, 'run_full_episode'):
                try:
                    # Ejecutar episodio sin resetear el entorno para mantener métricas
                    metrics = self.eval_env.run_full_episode(
                        self.model, 
                        self.deterministic, 
                        reset_env=False  # No resetear para mantener métricas acumuladas
                    )
                    
                    # Extract financial metrics
                    win_rate = metrics.get('win_rate', 0) * 100
                    portfolio_change = metrics.get('portfolio_change', 0) * 100
                    max_drawdown = metrics.get('max_drawdown', 0) * 100
                    total_trades = metrics.get('total_trades', 0)
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    sortino_ratio = metrics.get('sortino_ratio', 0)
                    
                    # Set additional attributes for the trial
                    self.trial.set_user_attr('win_rate', metrics.get('win_rate', 0))
                    self.trial.set_user_attr('portfolio_change', metrics.get('portfolio_change', 0))
                    self.trial.set_user_attr('max_drawdown', metrics.get('max_drawdown', 0))
                    self.trial.set_user_attr('total_trades', metrics.get('total_trades', 0))
                    self.trial.set_user_attr('sharpe_ratio', sharpe_ratio)
                    self.trial.set_user_attr('sortino_ratio', sortino_ratio)                           

                    # Print metrics
                    if self.verbose > 0:
                        print(f"{self.num_timesteps:<10} {self.last_mean_reward:<10.2f} {win_rate:<10.2f} {portfolio_change:<10.2f} {max_drawdown:<10.2f} {total_trades:<8} {sharpe_ratio:<8.2f}")
                
                except Exception as e:
                    # If metrics extraction fails, just print basic info with zeros
                    if self.verbose > 0:
                        logger.error(f"Error obteniendo métricas detalladas: {e}")
                        print(f"{self.num_timesteps:<10} {self.last_mean_reward:<10.2f} 0.00       0.00       0.00       0        0.00")
            else:
                # If run_full_episode is not available, print basic info with zeros
                if self.verbose > 0:
                    print(f"{self.num_timesteps:<10} {self.last_mean_reward:<10.2f} 0.00       0.00       0.00       0        0.00")
            
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                if self.verbose > 0:
                    print(f"Trial pruned at {self.num_timesteps} timesteps with reward {self.last_mean_reward:.2f}")
                return False
                
            return continue_training
        
        return True


class TrainingProgressCallback(BaseCallback):
    """
    Callback for tracking and displaying training progress.
    
    This callback shows a progress bar and statistics during training,
    providing a better user experience than the default logging.
    """
    
    def __init__(self, total_timesteps: int, update_interval: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_timesteps = 0
        
        # Initialize tables for progress display
        if self.verbose > 0:
            print("\nTRAINING PROGRESS")
            print("-" * 80)
            print(f"{'Timesteps':<15} {'Progress':<10} {'Steps/s':<10} {'Elapsed':<15} {'ETA':<15}")
            print("-" * 80)
    
    def _on_step(self) -> bool:
        # Check if it's time to update the display
        current_time = time.time()
        if (self.num_timesteps - self.last_timesteps >= self.update_interval or 
            current_time - self.last_update_time >= 30):  # At least every 30 seconds
            
            # Calculate progress percentage
            progress = self.num_timesteps / self.total_timesteps * 100
            
            # Calculate steps per second
            elapsed = current_time - self.start_time
            steps_per_second = self.num_timesteps / max(elapsed, 1e-6)
            
            # Calculate ETA
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta = remaining_steps / max(steps_per_second, 1e-6)
            
            # Format times
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            eta_str = str(timedelta(seconds=int(eta)))
            
            # Print progress
            if self.verbose > 0:
                print(f"{self.num_timesteps}/{self.total_timesteps:<8} {progress:<9.1f}% {steps_per_second:<10.1f} {elapsed_str:<15} {eta_str:<15}")
                
            # Update last update time and timesteps
            self.last_update_time = current_time
            self.last_timesteps = self.num_timesteps
            
        return True
    
    def _on_training_end(self) -> None:
        # Final progress update
        if self.verbose > 0:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            steps_per_second = self.num_timesteps / max(elapsed, 1e-6)
            
            print("-" * 80)
            print(f"Training completed: {self.num_timesteps}/{self.total_timesteps} steps in {elapsed_str} ({steps_per_second:.1f} steps/s)")
            print("-" * 80)


class IncrementalTrialCallback:
    """
    Callback to track progress across multiple trials in an optimization.
    
    This callback is designed to be passed to optuna.study.optimize() to
    provide global progress tracking across all trials.
    """
    
    def __init__(self, n_trials: int, update_interval: int = 1):
        self.n_trials = n_trials
        self.completed_trials = 0
        self.best_value = float('-inf')
        self.best_trial_number = None
        self.start_time = time.time()
        self.last_update = time.time()
        self.update_interval = update_interval
        
        # Print header
        print("\n" + "=" * 80)
        print(f"OPTUNA OPTIMIZATION PROGRESS ({n_trials} trials)")
        print("=" * 80)
        print(f"{'Trial':<8} {'Value':<10} {'Win Rate':<10} {'Profit %':<10} {'Drawdown':<10} {'Sharpe':<8} {'Time':<10}")
        print("-" * 80)
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> None:
        # Update completion count
        self.completed_trials += 1
        
        # Get trial attributes
        value = trial.value if trial.value is not None else float('nan')
        win_rate = trial.user_attrs.get('win_rate', 0) * 100
        portfolio_change = trial.user_attrs.get('portfolio_change', 0) * 100
        max_drawdown = trial.user_attrs.get('max_drawdown', 0) * 100
        sharpe_ratio = trial.user_attrs.get('sharpe_ratio', 0)
        
        # Calculate elapsed time for this trial
        elapsed = time.time() - self.last_update
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Update best value
        is_best = False
        if value is not None and value > self.best_value:
            self.best_value = value
            self.best_trial_number = trial.number
            is_best = True
        
        # Print progress
        best_marker = " ⭐" if is_best else ""
        print(f"{trial.number+1}/{self.n_trials:<3} {value:<10.4f} {win_rate:<10.2f} {portfolio_change:<10.2f} {max_drawdown:<10.2f} {sharpe_ratio:<8.2f} {elapsed_str:<10}{best_marker}")
        
        # Update last update time
        self.last_update = time.time()
    
    def finalize(self) -> None:
        # Print final summary
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print("-" * 80)
        print(f"Optimization completed: {self.completed_trials}/{self.n_trials} trials in {elapsed_str}")
        if self.best_trial_number is not None:
            print(f"Best trial: #{self.best_trial_number+1} with value {self.best_value:.4f}")
        print("=" * 80 + "\n")

def make_enhanced_env(data, config, isEval=False):
    """
    Versión corregida de make_enhanced_env que mejora el manejo de estados
    y asegura que la evaluación pueda realizarse correctamente.
    """
    from trading_manager.rl_trading_stable import EnhancedTradingEnv
    
    def _init():
        # Create enhanced trading environment with configuration
        env = EnhancedTradingEnv(
            data=data,
            initial_balance=config.get('initial_balance', 10000.0),
            max_position=config.get('max_position', 100),
            transaction_fee=config.get('transaction_fee', 0.001),
            reward_scaling=config.get('reward_scaling', 1.0),
            window_size=config.get('window_size', 60),
            reward_strategy=config.get('reward_strategy', 'balanced'),
            risk_free_rate=config.get('risk_free_rate', 0.0),
            risk_aversion=config.get('risk_aversion', 1.0),
            drawdown_penalty_factor=config.get('drawdown_penalty_factor', 15.0),
            holding_penalty_factor=config.get('holding_penalty_factor', 0.1),
            inactive_penalty_factor=config.get('inactive_penalty_factor', 0.05),
            consistency_reward_factor=config.get('consistency_reward_factor', 0.2),
            trend_following_factor=config.get('trend_following_factor', 0.3),
            win_streak_factor=config.get('win_streak_factor', 0.1)
        )

        filename = None
        if isEval:
            filename="./eval_logs"
        else:
            filename="./training_logs"
            
        # Wrap environment with Monitor for logging
        env = Monitor(env, filename=filename)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([_init])
    
    # Inicializar métricas acumulativas para evaluación
    if not hasattr(env, 'evaluation_metrics'):
        env.evaluation_metrics = {
            'episodes_run': 0,
            'winning_trades': 0,
            'total_trades': 0,
            'portfolio_changes': [],
            'max_drawdown_seen': 0.0
        }
    
    # Add methods to access unwrapped environment directly
    def get_unwrapped_env():
        """Get the unwrapped environment instance."""
        return env.envs[0].unwrapped
    
    # Define a method to run a complete evaluation episode correctly
    def run_full_episode(model, deterministic=True, reset_env=False):
        """
        Run a complete episode and return metrics
        
        Args:
            model: The model to evaluate
            deterministic: Whether to use deterministic actions
            reset_env: Whether to reset environment state before evaluation
        """
        # Get unwrapped environment for direct access
        unwrapped = get_unwrapped_env()
        
        # Guardar el estado actual para poder restaurarlo después
        if hasattr(unwrapped, 'get_state'):
            original_state = unwrapped.get_state()

        tradingEnv: EnhancedTradingEnv = env.envs[0].unwrapped
        
        # Reset environment to specific state if requested
        if reset_env:
            obs = env.reset()
        else:            
            # Otherwise, capture current observation without resetting
            obs = tradingEnv.current_observation
            if obs is None:  # Si no hay observación actual, debemos resetear
                obs = env.reset()
        
        # Ejecutar un episodio completo (hasta que done=True)
        # Almacenamos las recompensas para cálculo de métricas
        done = False
        total_reward = 0
        episode_rewards = []
        trade_results = []
        initial_portfolio = unwrapped.portfolio_value
        
        step_count = 0
        max_steps = len(unwrapped.data) - unwrapped.current_step - 1
        max_steps = min(max_steps, 1000)  # Limitar a 1000 pasos máximo
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = tradingEnv.step(action)
            total_reward += reward
            episode_rewards.append(reward)
            
            # Registrar operaciones completadas para análisis
            if hasattr(unwrapped, 'trades') and len(unwrapped.trades) > 0:
                latest_trade = unwrapped.trades[-1]
                if latest_trade.get('type') == 'sell':  # Solo operaciones completadas
                    trade_results.append(latest_trade.get('profit_loss', 0))
            
            step_count += 1
        
        # Obtener métricas del episodio
        episode_info = unwrapped._get_info()
        
        # Actualizar métricas acumulativas para evaluación
        env.evaluation_metrics['episodes_run'] += 1
        env.evaluation_metrics['winning_trades'] += episode_info.get('winning_trades', 0)
        env.evaluation_metrics['total_trades'] += episode_info.get('total_trades', 0)
        
        # Calcular cambio porcentual en portafolio
        final_portfolio = unwrapped.portfolio_value
        portfolio_change = (final_portfolio - initial_portfolio) / initial_portfolio
        env.evaluation_metrics['portfolio_changes'].append(portfolio_change)
        
        # Actualizar drawdown máximo visto
        current_drawdown = episode_info.get('max_drawdown', 0)
        env.evaluation_metrics['max_drawdown_seen'] = max(
            env.evaluation_metrics['max_drawdown_seen'], current_drawdown
        )
        
        # Calcular métricas acumulativas
        accumulated_metrics = {
            'win_rate': env.evaluation_metrics['winning_trades'] / max(1, env.evaluation_metrics['total_trades']),
            'portfolio_change': sum(env.evaluation_metrics['portfolio_changes']) / max(1, len(env.evaluation_metrics['portfolio_changes'])),
            'max_drawdown': env.evaluation_metrics['max_drawdown_seen'],
            'total_trades': env.evaluation_metrics['total_trades'],
            'sharpe_ratio': episode_info.get('sharpe_ratio', 0),
            'sortino_ratio': episode_info.get('sortino_ratio', 0),
            'calmar_ratio': episode_info.get('calmar_ratio', 0)
        }
        
        # Restaurar el estado original si es necesario
        if not reset_env and hasattr(unwrapped, 'set_state') and 'original_state' in locals():
            unwrapped.set_state(original_state)
        
        return {
            'total_reward': float(total_reward),
            'episode_length': step_count,
            'win_rate': float(accumulated_metrics['win_rate']),
            'portfolio_change': float(accumulated_metrics['portfolio_change']),
            'max_drawdown': float(accumulated_metrics['max_drawdown']),
            'total_trades': int(accumulated_metrics['total_trades']),
            'sharpe_ratio': float(accumulated_metrics['sharpe_ratio']),
            'sortino_ratio': float(accumulated_metrics['sortino_ratio']),
            'calmar_ratio': float(accumulated_metrics['calmar_ratio'])
        }
    
    # Add the method to the environment
    env.run_full_episode = run_full_episode
    env.get_unwrapped_env = get_unwrapped_env
    
    return env


def enhance_trading_manager(manager: RLTradingStableManager):
    """
    Enhance an existing RLTradingStableManager to use the improved environment.
    
    This function modifies the _initialize_with_data method of a trading manager
    to use the enhanced environment.
    
    Args:
        manager: The trading manager instance to enhance
    """
    # Store the original method for fallback
    original_initialize = manager._initialize_with_data
    
    # Define the enhanced version of _initialize_with_data
    def enhanced_initialize_with_data(data: pd.DataFrame):
        if data is None or data.empty:
            logger.error("Cannot initialize RL with empty data")
            return False
        
        try:
            # Log data info
            logger.info(f"Initializing enhanced RL with data: {len(data)} samples, {len(data.columns)} features")
            
            # Get environment configuration
            env_config = {
                'initial_balance': manager.rl_config.get('initial_balance', 10000.0),
                'max_position': manager.rl_config.get('max_position', 100),
                'transaction_fee': manager.rl_config.get('transaction_fee', 0.0005),
                # Enhanced parameters
                'reward_scaling': manager.rl_config.get('reward_scaling', 1.0),
                'window_size': manager.rl_config.get('window_size', 60),
                'reward_strategy': manager.rl_config.get('reward_strategy', 'balanced'),
                'risk_free_rate': manager.rl_config.get('risk_free_rate', 0.0),
                'risk_aversion': manager.rl_config.get('risk_aversion', 1.0),
                'drawdown_penalty_factor': manager.rl_config.get('drawdown_penalty_factor', 15.0),
                'holding_penalty_factor': manager.rl_config.get('holding_penalty_factor', 0.1),
                'inactive_penalty_factor': manager.rl_config.get('inactive_penalty_factor', 0.05),
                'consistency_reward_factor': manager.rl_config.get('consistency_reward_factor', 0.2),
                'trend_following_factor': manager.rl_config.get('trend_following_factor', 0.3),
                'win_streak_factor': manager.rl_config.get('win_streak_factor', 0.1)
            }
            
            # Create enhanced environment
            try:
                manager.env = make_enhanced_env(data, env_config)
                logger.info("Created enhanced trading environment")
            except Exception as e:
                logger.error(f"Failed to create enhanced environment: {e}")
                # Fall back to original environment
                logger.info("Falling back to original environment")
                
                # Call the original method to create the environment
                return original_initialize(data)
            
            # Try to load existing model
            try:
                model_path = os.path.join(manager.model_path, 'ppo_trading')
                if os.path.exists(model_path + '.zip'):
                    manager.model = PPO.load(model_path, env=manager.env)
                    logger.info(f"Loaded existing model from {model_path}")
                else:
                    # Create new model with enhanced network architecture
                    policy_kwargs = {
                        'net_arch': [256, 128, 64]  # Default architecture
                    }
                    
                    # Check for custom architecture in config
                    if 'net_arch' in manager.rl_config:
                        if isinstance(manager.rl_config['net_arch'], list):
                            policy_kwargs['net_arch'] = manager.rl_config['net_arch']
                        elif manager.rl_config['net_arch'] == 'large':
                            policy_kwargs['net_arch'] = [512, 256, 128]
                        elif manager.rl_config['net_arch'] == 'medium':
                            policy_kwargs['net_arch'] = [256, 128, 64]
                        elif manager.rl_config['net_arch'] == 'small':
                            policy_kwargs['net_arch'] = [128, 64, 32]
                    
                    # Create new model
                    manager.model = PPO(
                        'MlpPolicy',
                        manager.env,
                        verbose=1,
                        tensorboard_log=os.path.join(manager.model_path, 'logs'),
                        policy_kwargs=policy_kwargs,
                        learning_rate=manager.rl_config.get('learning_rate', 3e-4),
                        n_steps=manager.rl_config.get('n_steps', 2048),
                        batch_size=manager.rl_config.get('batch_size', 64),
                        gamma=manager.rl_config.get('gamma', 0.99),
                        ent_coef=manager.rl_config.get('ent_coef', 0.01)
                    )
                    logger.info("Created new PPO model with enhanced environment")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Create new model
                manager.model = PPO(
                    'MlpPolicy',
                    manager.env,
                    verbose=1,
                    tensorboard_log=os.path.join(manager.model_path, 'logs'),
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    gamma=0.99,
                    ent_coef=0.01
                )
                logger.info("Created new PPO model with enhanced environment (fallback configuration)")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing enhanced RL components: {e}")
            # Try the original initialization as fallback
            return original_initialize(data)
    
    # Replace the method with our enhanced version
    manager._initialize_with_data = enhanced_initialize_with_data
    
    # Add a marker attribute to indicate this manager has been enhanced
    manager.is_enhanced = True
    
    return manager

def create_enhanced_config(base_config: Dict) -> Dict:
    """
    Create an enhanced configuration dictionary by adding
    the necessary parameters for the enhanced trading environment.
    
    Args:
        base_config: Base configuration dictionary
        
    Returns:
        Dict: Enhanced configuration dictionary
    """
    # Start with a copy of the base config
    enhanced_config = base_config.copy()
    
    # Ensure the 'trading_manager' and 'rl' sections exist
    if 'trading_manager' not in enhanced_config:
        enhanced_config['trading_manager'] = {}
        
    if 'rl' not in enhanced_config['trading_manager']:
        enhanced_config['trading_manager']['rl'] = {}
    
    # Get the rl config section for easier access
    rl_config = enhanced_config['trading_manager']['rl']
    
    # Add enhanced environment parameters with default values if not present
    if 'reward_strategy' not in rl_config:
        rl_config['reward_strategy'] = 'balanced'  # 'simple', 'sharpe', 'sortino', or 'balanced'
    
    if 'reward_scaling' not in rl_config:
        rl_config['reward_scaling'] = 1.0
    
    if 'window_size' not in rl_config:
        rl_config['window_size'] = 60
    
    if 'risk_free_rate' not in rl_config:
        rl_config['risk_free_rate'] = 0.03  # 3% annual risk-free rate
    
    if 'risk_aversion' not in rl_config:
        rl_config['risk_aversion'] = 1.0  # Neutral risk aversion
    
    # Add enhanced reward component parameters
    reward_params = {
        'drawdown_penalty_factor': 15.0,
        'holding_penalty_factor': 0.1,
        'inactive_penalty_factor': 0.05,
        'consistency_reward_factor': 0.2,
        'trend_following_factor': 0.3,
        'win_streak_factor': 0.1
    }
    
    for param, default_value in reward_params.items():
        if param not in rl_config:
            rl_config[param] = default_value
    
    # Add optimized neural network architecture if not present
    if 'net_arch' not in rl_config:
        rl_config['net_arch'] = 'medium'  # Default to medium size
    
    # Set default training parameters if not present
    training_params = {
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    }
    
    for param, default_value in training_params.items():
        if param not in rl_config:
            rl_config[param] = default_value
    
    return enhanced_config

def update_config_file(config_path: str, enhanced_config: Dict) -> bool:
    """
    Update the configuration file with the enhanced configuration.
    
    Args:
        config_path: Path to the configuration file
        enhanced_config: Enhanced configuration dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    import json
    
    try:
        # Save the enhanced configuration to file
        with open(config_path, 'w') as f:
            json.dump(enhanced_config, f, indent=4)
        
        logger.info(f"Updated configuration file at {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating configuration file: {e}")
        return False

def run_enhanced_optimization(
    data: pd.DataFrame, 
    config: Dict, 
    output_dir: str = "models/rl/enhanced",
    n_trials: int = 20,
    n_timesteps: int = 100000,
    final_timesteps: int = 200000,
    n_jobs: int = 1
) -> Dict:
    """
    Run an enhanced optimization process using Optuna and the EnhancedTradingEnv.
    
    Args:
        data: Historical price data
        config: Environment configuration
        output_dir: Directory to save results
        n_trials: Number of trials for optimization
        n_timesteps: Number of timesteps for training during optimization
        final_timesteps: Number of timesteps for final model training
        n_jobs: Number of parallel jobs
        
    Returns:
        Dict: Optimization results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Define TrialEvalCallback for Optuna
    class TrialEvalCallback(EvalCallback):
        """Callback for evaluating and pruning trials in Optuna."""
        
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
                # Call original implementation
                continue_training = super()._on_step()
                
                # Report to Optuna for pruning
                self.eval_idx += 1
                self.trial.report(self.last_mean_reward, self.eval_idx)
                
                # Prune trial if needed
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
                    
                return continue_training
            
            return True
    
    # Define function to calculate financial metrics
    def calculate_financial_metrics(env):
        """Calculate financial metrics from environment."""
        env_unwrapped = env.envs[0].unwrapped
        
        # Basic metrics
        win_rate = env_unwrapped.winning_trades / max(1, env_unwrapped.total_trades)
        max_drawdown = env_unwrapped.max_drawdown
        portfolio_change = (env_unwrapped.portfolio_value - env_unwrapped.initial_balance) / env_unwrapped.initial_balance
        total_trades = env_unwrapped.total_trades
        
        # Calculate Sharpe ratio if available
        sharpe_ratio = env_unwrapped._get_info().get('sharpe_ratio', 0)
        
        # Calculate Sortino ratio if available
        sortino_ratio = env_unwrapped._get_info().get('sortino_ratio', 0)
        
        # Calculate Calmar ratio if available
        calmar_ratio = env_unwrapped._get_info().get('calmar_ratio', 0)
        
        return {
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'portfolio_change': portfolio_change,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    # Define sample_params function for Optuna
    def sample_params(trial):
        """Sample hyperparameters for Optuna trial."""
        # Environment parameters
        env_params = {
            'reward_strategy': trial.suggest_categorical(
                'reward_strategy', 
                ['balanced', 'sharpe', 'sortino']
            ),
            'risk_aversion': trial.suggest_float('risk_aversion', 0.5, 2.0),
            'reward_scaling': trial.suggest_float('reward_scaling', 0.5, 2.0),
            
            # Reward component factors
            'drawdown_penalty_factor': trial.suggest_float('drawdown_penalty_factor', 5.0, 25.0),
            'holding_penalty_factor': trial.suggest_float('holding_penalty_factor', 0.05, 0.2),
            'inactive_penalty_factor': trial.suggest_float('inactive_penalty_factor', 0.01, 0.1),
            'consistency_reward_factor': trial.suggest_float('consistency_reward_factor', 0.1, 0.4),
            'trend_following_factor': trial.suggest_float('trend_following_factor', 0.1, 0.5),
            'win_streak_factor': trial.suggest_float('win_streak_factor', 0.05, 0.2)
        }
        
        # PPO parameters
        ppo_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.999),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
        }
        
        # Network architecture
        net_width = trial.suggest_categorical('net_width', [64, 128, 256, 512])
        net_depth = trial.suggest_int('net_depth', 1, 4)
        net_arch = []
        for _ in range(net_depth):
            net_arch.append(net_width)
        
        activation_fn_name = trial.suggest_categorical(
            'activation_fn', ['tanh', 'relu', 'elu']
        )
        activation_fn = {
            'tanh': 'tanh',
            'relu': 'relu',
            'elu': 'elu'
        }[activation_fn_name]
        
        # Policy kwargs
        policy_kwargs = {
            'net_arch': net_arch,
            'activation_fn': activation_fn
        }
        
        # Combine all parameters
        params = {
            'env_params': env_params,
            'ppo_params': ppo_params,
            'policy_kwargs': policy_kwargs
        }
        
        return params
    
    # Define objective function for Optuna
    def objective(trial):
        """Objective function for Optuna optimization."""
        # Sample parameters
        params = sample_params(trial)
        
        # Update environment configuration with sampled parameters
        env_config = config.copy()
        env_config.update(params['env_params'])
        
        # Create environments 
        env = make_enhanced_env(data, env_config)
        eval_env = make_enhanced_env(data, env_config)
        
        # Import necessary activation functions properly
        import torch.nn as nn
        
        # Map string activation function names to actual PyTorch activation functions
        activation_fn_map = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU
        }
        
        # Get activation function from map
        activation_fn_name = params['policy_kwargs']['activation_fn']
        activation_fn = activation_fn_map[activation_fn_name]
        
        # Create model
        try:
            model = PPO(
                'MlpPolicy', 
                env, 
                verbose=0, 
                **params['ppo_params'],
                policy_kwargs={
                    'net_arch': params['policy_kwargs']['net_arch'],
                    'activation_fn': activation_fn
                }
            )
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return float("-inf")
        
        # Create callback for evaluation and pruning
        eval_callback = DetailedTrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            best_model_save_path=os.path.join(output_dir, f"models/trial_{trial.number}"),
            eval_freq=5000,
            n_eval_episodes=3,
            deterministic=True,
            verbose=1
        )
        
        # Train model
        try:
            model.learn(total_timesteps=n_timesteps, callback=eval_callback)
            
            # Check if pruned
            if eval_callback.is_pruned:
                raise optuna.exceptions.TrialPruned()
            
            # Check best reward
            if eval_callback.best_mean_reward < 0:
                # If reward is negative, return it directly
                return eval_callback.best_mean_reward
            
            # Run a full evaluation episode to get metrics
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)
            
            # Get metrics from the environment
            metrics = calculate_financial_metrics(eval_env)
            
            # Record metrics as attributes
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            
            # Create a score that balances reward with financial metrics
            # Higher values should be better overall performers
            if metrics['portfolio_change'] > 0 and metrics['win_rate'] > 0:
                # For good performers, promote those with good metrics
                portfolio_score = metrics['portfolio_change'] * 2
                risk_score = -metrics['max_drawdown'] * 10  # Penalize drawdowns
                trade_quality = metrics['win_rate'] * 2  # Reward win rate
                sharpe_bonus = metrics['sharpe_ratio']
                
                # Create combined score that balances multiple factors
                score = (
                    eval_callback.best_mean_reward * 0.3 +  # Base on reward
                    portfolio_score +                       # Reward good returns
                    risk_score +                            # Penalize drawdowns
                    trade_quality +                         # Reward high win rates
                    sharpe_bonus                            # Reward good risk-adjusted returns
                )
            else:
                # For poor performers, just use the reward
                score = eval_callback.best_mean_reward
            
            return score
            
        except Exception as e:
            # Return poor score on error
            logger.error(f"Error in trial: {e}")
            return float("-inf")
    
    # Create study
    sampler = TPESampler(n_startup_trials=5, seed=config.get('seed', 42))
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=f"enhanced_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Create a progress callback
    progress_callback = IncrementalTrialCallback(n_trials=n_trials)
    
    # Optimize
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[progress_callback])
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        progress_callback.finalize()
    
    # Get best parameters
    try:
        best_params = sample_params(study.best_trial)
    except Exception as e:
        logger.error(f"Error getting best parameters: {e}")
        return {'status': 'error', 'message': str(e)}
    
    # Save best parameters
    try:
        with open(os.path.join(output_dir, "results/best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"Error saving best parameters: {e}")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    
    # Update environment configuration with best parameters
    final_env_config = config.copy()
    final_env_config.update(best_params['env_params'])
    
    # Create environments
    final_env = make_enhanced_env(data, final_env_config)
    final_eval_env = make_enhanced_env(data, final_env_config)
    
    # Import necessary activation functions properly
    import torch.nn as nn
    
    # Map string activation function names to actual PyTorch activation functions
    activation_fn_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }
    
    # Get activation function
    activation_fn_name = best_params['policy_kwargs']['activation_fn']
    activation_fn = activation_fn_map[activation_fn_name]
    
    # Create final model
    try:
        final_model = PPO(
            'MlpPolicy', 
            final_env, 
            verbose=1, 
            **best_params['ppo_params'],
            policy_kwargs={
                'net_arch': best_params['policy_kwargs']['net_arch'],
                'activation_fn': activation_fn
            }
        )
    except Exception as e:
        logger.error(f"Error creating final model: {e}")
        return {
            'status': 'error',
            'message': f"Failed to create final model: {e}",
            'best_params': best_params
        }
    
    # Create callback for evaluation
    final_eval_callback = DetailedEvalCallback(
        eval_env=final_eval_env,
        best_model_save_path=os.path.join(output_dir, "models/final"),
        log_path=os.path.join(output_dir, "logs/final"),
        eval_freq=10000,
        deterministic=True,
        verbose=1
    )
    
    # Create training progress callback
    training_progress = TrainingProgressCallback(
        total_timesteps=final_timesteps,
        update_interval=5000,
        verbose=1
    )
    
    # Train final model
    start_time = time.time()
    try:
        final_model.learn(
            total_timesteps=final_timesteps, 
            callback=[final_eval_callback, training_progress]
        )
    except Exception as e:
        logger.error(f"Error training final model: {e}")
        return {
            'status': 'error',
            'message': f"Failed to train final model: {e}",
            'best_params': best_params,
            'training_time': time.time() - start_time
        }
    
    training_time = time.time() - start_time
    
    # Save final model
    try:
        final_model.save(os.path.join(output_dir, "models/final/model"))
    except Exception as e:
        logger.error(f"Error saving final model: {e}")
    
    # Evaluate final model on a full episode to get accurate metrics
    try:
        # Reset environment
        obs, _ = final_eval_env.reset()
        done = False
        
        # Run through one episode
        while not done:
            action, _ = final_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = final_eval_env.step(action)
        
        # Get metrics
        metrics = calculate_financial_metrics(final_eval_env)
        
        # Also get reward from standard evaluation
        mean_reward, std_reward = evaluate_policy(
            final_model, 
            final_eval_env, 
            n_eval_episodes=10,
            deterministic=True
        )
    except Exception as e:
        logger.error(f"Error evaluating final model: {e}")
        metrics = {}
        mean_reward = -float('inf')
        std_reward = 0
    
    # Save results
    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'win_rate': float(metrics.get('win_rate', 0)),
        'max_drawdown': float(metrics.get('max_drawdown', 0)),
        'portfolio_change': float(metrics.get('portfolio_change', 0)),
        'total_trades': int(metrics.get('total_trades', 0)),
        'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
        'sortino_ratio': float(metrics.get('sortino_ratio', 0)),
        'calmar_ratio': float(metrics.get('calmar_ratio', 0)),
        'training_time': float(training_time),
        'training_time_formatted': time.strftime("%H:%M:%S", time.gmtime(training_time)),
        'best_params': best_params
    }
    
    # Save results to file
    try:
        with open(os.path.join(output_dir, "results/final_results.json"), "w") as f:
            json.dump(results, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Return results
    return {
        'status': 'success',
        'best_params': best_params,
        'final_results': results,
        'study': study
    }

def custom_run_enhanced_optimization(
    data: pd.DataFrame, 
    config: Dict, 
    output_dir: str = "models/rl/enhanced",
    n_trials: int = 20,
    n_timesteps: int = 100000,
    final_timesteps: int = 200000,
    n_jobs: int = 1
) -> Dict:
        """
        Run an enhanced optimization process with better progress reporting using improved callbacks.
        
        Args:
            data: Historical price data
            config: Environment configuration
            output_dir: Directory to save results
            n_trials: Number of trials for optimization
            n_timesteps: Number of timesteps for training during optimization
            final_timesteps: Number of timesteps for final model training
            n_jobs: Number of parallel jobs
            
        Returns:
            Dict: Optimization results
        """
        # Create progress callback using the improved version
        progress_callback = IncrementalTrialCallback(n_trials=n_trials)
            
        
        # Define improved objective function that uses DetailedTrialEvalCallback
        def enhanced_objective(trial):
            """Enhanced objective function using improved callbacks."""
            from stable_baselines3 import PPO
            import torch.nn as nn
            
            # Sample parameters (same as in run_enhanced_optimization)
            env_params = {
                'reward_strategy': trial.suggest_categorical(
                    'reward_strategy', 
                    ['balanced', 'sharpe', 'sortino']
                ),
                'risk_aversion': trial.suggest_float('risk_aversion', 0.5, 2.0),
                'reward_scaling': trial.suggest_float('reward_scaling', 0.5, 2.0),
                'drawdown_penalty_factor': trial.suggest_float('drawdown_penalty_factor', 5.0, 25.0),
                'holding_penalty_factor': trial.suggest_float('holding_penalty_factor', 0.05, 0.2),
                'inactive_penalty_factor': trial.suggest_float('inactive_penalty_factor', 0.01, 0.1),
                'consistency_reward_factor': trial.suggest_float('consistency_reward_factor', 0.1, 0.4),
                'trend_following_factor': trial.suggest_float('trend_following_factor', 0.1, 0.5),
                'win_streak_factor': trial.suggest_float('win_streak_factor', 0.05, 0.2)
            }
            
            # PPO hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            n_epochs = trial.suggest_int("n_epochs", 5, 20)
            gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
            gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
            clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
            ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
            vf_coef = trial.suggest_float("vf_coef", 0.4, 1.0)
            
            # Network architecture
            net_width = trial.suggest_categorical("net_width", [64, 128, 256, 512])
            net_depth = trial.suggest_int("net_depth", 1, 4)
            net_arch = [net_width for _ in range(net_depth)]
            
            # Activation function
            activation_fn_name = trial.suggest_categorical(
                "activation_fn", ["tanh", "relu", "elu"]
            )
            activation_fn = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "elu": nn.ELU
            }[activation_fn_name]
            
            # Update environment configuration with sampled parameters
            trial_config = config.copy()
            trial_config.update(env_params)
            
            # Create environments
            env = make_enhanced_env(data, trial_config)
            eval_env = make_enhanced_env(data, trial_config)
            
            # Create policy kwargs
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn
            }
            
            # Create model
            ppo_params = {
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "policy_kwargs": policy_kwargs
            }
            
            model = PPO("MlpPolicy", env, verbose=0, **ppo_params)
            
            # Use DetailedTrialEvalCallback for better evaluation and pruning
            eval_callback = DetailedTrialEvalCallback(
                eval_env=eval_env,
                trial=trial,
                n_eval_episodes=5,
                eval_freq=5000,
                log_path=os.path.join(output_dir, "logs", f"trial_{trial.number}"),
                best_model_save_path=os.path.join(output_dir, "models", f"trial_{trial.number}"),
                deterministic=True,
                verbose=1
            )
            
            try:
                # Train the model
                model.learn(total_timesteps=n_timesteps, callback=eval_callback)
                
                # If the trial was pruned, raise a pruned exception
                if eval_callback.is_pruned:
                    raise optuna.exceptions.TrialPruned()
                
                # Return best reward as the objective value
                return eval_callback.best_mean_reward
                
            except (optuna.exceptions.TrialPruned, Exception) as e:
                # Handle pruning and other exceptions
                if isinstance(e, optuna.exceptions.TrialPruned):
                    print_progress(f"Trial {trial.number} pruned.")
                else:
                    print_progress(f"Error in trial {trial.number}: {e}")
                
                # Re-raise TrialPruned but convert other exceptions to TrialPruned
                if isinstance(e, optuna.exceptions.TrialPruned):
                    raise e
                return float('-inf')
        
        # Create study with improved sampling and pruning
        sampler = TPESampler(n_startup_trials=5, seed=config.get('seed', 42))
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=f"enhanced_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        try:
            # Run optimization with improved callback
            study.optimize(
                enhanced_objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                callbacks=[progress_callback],
                show_progress_bar=True
            )
            
            # Finalize progress report
            progress_callback.finalize()
            
            # Get best trial and parameters
            best_trial = study.best_trial
            best_params = best_trial.params
            
            # Convert best parameters back to format expected by run_enhanced_optimization
            activation_fn_name = best_params.pop("activation_fn")
            net_width = best_params.pop("net_width")
            net_depth = best_params.pop("net_depth")
            net_arch = [net_width for _ in range(net_depth)]
            
            # Separate environment parameters
            env_params = {
                'reward_strategy': best_params.pop('reward_strategy'),
                'risk_aversion': best_params.pop('risk_aversion'),
                'reward_scaling': best_params.pop('reward_scaling'),
                'drawdown_penalty_factor': best_params.pop('drawdown_penalty_factor'),
                'holding_penalty_factor': best_params.pop('holding_penalty_factor'),
                'inactive_penalty_factor': best_params.pop('inactive_penalty_factor'),
                'consistency_reward_factor': best_params.pop('consistency_reward_factor'),
                'trend_following_factor': best_params.pop('trend_following_factor'),
                'win_streak_factor': best_params.pop('win_streak_factor')
            }
            
            # Create policy kwargs
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn_name
            }
            
            # Bundle remaining parameters as ppo_params
            ppo_params = {
                key: best_params[key] for key in [
                    "learning_rate", "n_steps", "batch_size", "n_epochs",
                    "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef"
                ]
            }
            
            # Organize results in expected format
            best_params = {
                'env_params': env_params,
                'policy_kwargs': policy_kwargs,
                'ppo_params': ppo_params
            }
            
            # Train final model with best parameters
            print_progress(f"\nTraining final model with best parameters...")
            
            # Create environment with best parameters
            final_config = config.copy()
            final_config.update(env_params)
            
            final_env = make_enhanced_env(data, final_config)
            final_eval_env = make_enhanced_env(data, final_config)
            
            # Import activation function properly
            import torch.nn as nn
            activation_fn = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "elu": nn.ELU
            }[activation_fn_name]
            
            # Create final model
            final_policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn
            }
            
            final_model = PPO(
                "MlpPolicy",
                final_env,
                verbose=1,
                policy_kwargs=final_policy_kwargs,
                **ppo_params
            )
            
            final_eval_callback = DetailedEvalCallback(
                eval_env=final_eval_env,
                best_model_save_path=os.path.join(output_dir, "final_model"),
                log_path=os.path.join(output_dir, "logs"),
                eval_freq=10000,
                deterministic=True,
                verbose=1
            )
            
            final_progress_callback = TrainingProgressCallback(
                total_timesteps=final_timesteps,
                update_interval=5000,
                verbose=1
            )
            
            # Train final model
            start_time = time.time()
            final_model.learn(
                total_timesteps=final_timesteps,
                callback=[final_eval_callback, final_progress_callback]
            )
            training_time = time.time() - start_time
            
            # Save final model
            final_model_path = os.path.join(output_dir, "final_model", "model")
            final_model.save(final_model_path)
            
            # Evaluate final model
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(
                final_model, final_eval_env, n_eval_episodes=10
            )
            
            # Get environment metrics
            env_unwrapped = final_env.envs[0].unwrapped
            info = env_unwrapped._get_info()
            
            # Extract metrics
            win_rate = info.get('win_rate', 0)
            portfolio_change = info.get('portfolio_change', 0)
            max_drawdown = info.get('max_drawdown', 0)
            total_trades = info.get('total_trades', 0)
            sharpe_ratio = info.get('sharpe_ratio', 0)
            sortino_ratio = info.get('sortino_ratio', 0)
            calmar_ratio = info.get('calmar_ratio', 0)
            
            # Create results dict
            final_results = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'win_rate': float(win_rate),
                'portfolio_change': float(portfolio_change),
                'max_drawdown': float(max_drawdown),
                'total_trades': int(total_trades),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'training_time': float(training_time),
                'training_time_formatted': time.strftime("%H:%M:%S", time.gmtime(training_time))
            }
            
            # Return results in expected format
            return {
                'best_params': best_params,
                'final_results': final_results,
                'study': study
            }
                
        except KeyboardInterrupt:
            print_progress("\n⚠️ Optimization interrupted by user")
            # Collect partial results
            if hasattr(study, 'best_trial'):
                print_progress(f"Best trial so far: #{study.best_trial.number}, value: {study.best_value:.4f}")
            return {'status': 'interrupted'}


def simple_rl_training(
    data: pd.DataFrame, 
    config: dict, 
    symbol: str, 
    timesteps: int = 200000,
    output_dir: str = "models/rl/enhanced"
) -> dict:
    """
    Train a simple RL model without optimization, using improved callbacks
    
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
    from stable_baselines3.common.evaluation import evaluate_policy
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print_progress(f"Setting up environments for simple training...")
    
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
    
    print_progress(f"Creating PPO model with architecture: {policy_kwargs['net_arch']}...")
    
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
    
    # Crear DetailedEvalCallback en lugar del EvalCallback estándar
    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=10000,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "logs"),
        deterministic=True,
        verbose=1
    )
    
    # Crear TrainingProgressCallback en lugar del ProgressTracker personalizado
    progress_callback = TrainingProgressCallback(
        total_timesteps=timesteps,
        update_interval=5000,
        verbose=1
    )
    
    # Train model
    print_progress(f"Starting training for {timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    training_time = time.time() - start_time
    
    # Save final model
    model_save_path = os.path.join(output_dir, f"final_model_{symbol}")
    model.save(model_save_path)
    
    print_progress(f"Evaluating model performance...")
    
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
    
    print_progress(f"✅ Training completed for {symbol}: mean_reward={mean_reward:.2f}, win_rate={info['win_rate']:.2f}")
    
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
    
    print_progress(f"Creating {'enhanced' if enhanced else 'standard'} trading manager...")
    
    # Create trading manager
    manager = RLTradingStableManager(config_path=config_path)
    
    # Enhance manager if requested
    if enhanced:
        manager = enhance_trading_manager(manager)
    
    # Initialize with data
    print_progress(f"Initializing trading manager with data...")
    success = manager._initialize_with_data(data)
    if not success:
        print_progress(f"❌ Failed to initialize trading manager with data")
        return {'status': 'error', 'message': 'Failed to initialize trading manager'}
    
    # Train model
    print_progress(f"Starting training for {timesteps} timesteps...")
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
    
    print_progress(f"✅ Training completed in {training_result['training_time_formatted']}")
    
    return training_result

# Función para imprimir mensajes de progreso
def print_progress(message, end='\n'):
    """Print progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", end=end)
    import sys
    sys.stdout.flush()
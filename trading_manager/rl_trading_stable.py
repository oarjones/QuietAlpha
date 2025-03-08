"""
RL Trading Manager using stable-baselines3

This module implements a Trading Manager using Proximal Policy Optimization (PPO)
from stable-baselines3 for making trading decisions with a simple action space:
buy, sell, or hold.
"""

import os
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional, Tuple, Union
import datetime
import time
from threading import Lock

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from trading_manager.base import TradingManager
from data.processing import calculate_technical_indicators, normalize_indicators
from utils.data_utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTradingEnv(gym.Env):
    """
    Simple trading environment with three actions: buy, sell, hold.
    
    This environment simulates a basic trading scenario where an agent can
    make simple decisions and is rewarded based on profit/loss.
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                max_position: int = 20, transaction_fee: float = 0.001):
        """
        Initialize the trading environment.
        
        Args:
            data (pd.DataFrame): Historical price data with technical indicators
            initial_balance (float): Starting account balance
            max_position (int): Maximum position size
            transaction_fee (float): Fee per transaction as a fraction of trade value
        """
        super(SimpleTradingEnv, self).__init__()
        
        # Store configuration
        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_fee = transaction_fee
        
        # State variables
        self.balance = initial_balance
        self.position = 0  # Current position (number of shares)
        self.current_step = 0
        self.current_price = 0
        self.avg_entry_price = 0  # Track average entry price for P&L calculation
        self.trades = []
        self.portfolio_value = initial_balance
        self.done = False
        
        # Extract feature columns (indicators)
        # self.features = [col for col in data.columns 
        #                 if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        self.features = [
            'EMA_200_norm',        # Contexto tendencia a largo plazo
            'EMA_21_norm',         # Contexto tendencia intermedia
            'MACD_diff_norm',      # Señal cambio dirección
            'RSI_14_norm',         # Fuerza relativa del movimiento
            'ATR_14_norm',         # Volatilidad absoluta
            'BB_Width_norm',       # Volatilidad relativa
            'OBV_norm',            # Volumen acumulado
            'Force_Index_13_norm', # Fuerza inmediata volumen-precio
            'ATR_14_norm',         # Riesgo y tamaño posiciones
            'TrendStrength_norm'   # Indicación general del estado de tendencia
        ]

        
        # Define observation space (state space)
        # Includes technical indicators + position + balance
        self.obs_dim = len(self.features) + 2  # +2 for position and balance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Trading session stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = initial_balance
        self.trade_history = []
        
        # Performance tracking
        self.episode_returns = []
        
        # Reset on init
        self.reset()
        
        logger.info(f"SimpleTradingEnv initialized with {len(data)} data points, "
                   f"initial balance: {initial_balance}, max position: {max_position}")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.current_price = self.data['close'].iloc[0]
        self.avg_entry_price = 0
        self.trades = []
        self.portfolio_value = self.initial_balance
        self.done = False
        
        # Reset statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_balance
        self.trade_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: 0 = hold, 1 = buy, 2 = sell
            
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Ensure current step is valid
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        next_data = self.data.iloc[self.current_step + 1]
        
        # Update current price
        self.current_price = current_data['close']
        next_price = next_data['close']
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.position * self.current_price
        
        # Initialize components of reward for better tracking
        action_reward = 0
        holding_penalty = 0
        invalid_action_penalty = 0
        portfolio_performance_reward = 0
        drawdown_penalty = 0
        
        # Track consecutive holds
        if not hasattr(self, 'consecutive_holds'):
            self.consecutive_holds = 0
        
        # Process the action
        action_taken = False  # Flag to track if a meaningful action was taken
        
        # --- HOLD ACTION ---
        if action == 0:  # Hold
            # Increment consecutive holds counter
            self.consecutive_holds += 1
            
            # Apply mild penalty for excessive holding, more severe after longer periods
            if self.consecutive_holds > 10:
                # Sigmoid-like scaling of penalty - increases slowly then more rapidly
                hold_factor = min((self.consecutive_holds - 10) / 30, 1.0)
                holding_penalty = 0.1 * hold_factor
                
            # Extra penalty if market is moving significantly but we're not acting
            price_change_pct = abs(next_price - self.current_price) / self.current_price
            if price_change_pct > 0.01:  # 1% price move
                volatility_penalty = 0.05 * min(price_change_pct / 0.01, 3.0)  # Scale with volatility, cap at 3x
                holding_penalty += volatility_penalty
                
            # If holding cash for too long (no position), apply additional mild penalty
            if self.position == 0 and self.consecutive_holds > 20:
                cash_penalty = 0.02 * min((self.consecutive_holds - 20) / 20, 1.0)
                holding_penalty += cash_penalty
        else:
            # Reset consecutive holds counter on any other action
            self.consecutive_holds = 0
        
        # --- BUY ACTION ---
        if action == 1:  # Buy
            # Calculate how many shares we can buy
            max_new_shares = int(self.balance / (self.current_price * (1 + self.transaction_fee)))
            
            # Limit position size
            available_capacity = self.max_position - self.position
            position_size = min(max_new_shares, available_capacity)
            
            if position_size > 0:
                # Calculate trade cost
                trade_cost = position_size * self.current_price * self.transaction_fee
                trade_value = position_size * self.current_price
                
                # Update average entry price if we already have a position
                if self.position > 0:
                    # Calculate weighted average price
                    total_position = self.position + position_size
                    self.avg_entry_price = (
                        (self.position * self.avg_entry_price + position_size * self.current_price) /
                        total_position
                    )
                else:
                    # First purchase
                    self.avg_entry_price = self.current_price
                
                # Execute trade
                self.balance -= (trade_value + trade_cost)
                self.position += position_size
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': self.current_price,
                    'quantity': position_size,
                    'cost': trade_cost,
                    'avg_entry_price': self.avg_entry_price
                })
                
                self.total_trades += 1
                action_taken = True
                
                # Small positive reward for taking action after long holding period
                if self.consecutive_holds > 15:
                    action_reward += 0.1
                
                logger.debug(f"Buy: {position_size} shares at {self.current_price:.2f}, " +
                            f"new total position: {self.position}, avg price: {self.avg_entry_price:.2f}")
            else:
                # Tried to buy but couldn't (no funds or at max position)
                # Small penalty for attempting invalid action
                invalid_action_penalty = 0.05
        
        # --- SELL ACTION ---
        elif action == 2:  # Sell
            # Only sell if we have a position
            if self.position > 0:
                # Calculate trade cost
                trade_cost = self.position * self.current_price * self.transaction_fee
                
                # Calculate profit/loss based on average entry price
                position_value = self.position * self.current_price
                entry_value = self.position * self.avg_entry_price
                profit_loss = position_value - entry_value - trade_cost
                
                # Execute trade
                self.balance += position_value - trade_cost
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': self.current_price,
                    'quantity': self.position,
                    'cost': trade_cost,
                    'profit_loss': profit_loss
                })
                
                # Update win/loss stats
                if profit_loss > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Calculate reward based on profit/loss relative to investment
                position_cost = entry_value + trade_cost
                if position_cost > 0:  # Avoid division by zero
                    # Scale the profit/loss reward to keep it in a reasonable range
                    # Apply tanh to limit extreme values while preserving sign
                    profit_loss_pct = profit_loss / position_cost
                    action_reward = np.tanh(profit_loss_pct * 5) * 0.5  # Scale to [-0.5, 0.5] range
                
                # Small positive reward for taking action after long holding period
                if self.consecutive_holds > 15:
                    action_reward += 0.1
                
                action_taken = True
                logger.debug(f"Sell: {self.position} shares at {self.current_price:.2f}, P&L: {profit_loss:.2f}")
                
                # Reset position
                self.position = 0
                self.avg_entry_price = 0
                
                self.total_trades += 1
            else:
                # Tried to sell but no position
                # Small penalty for attempting invalid action
                invalid_action_penalty = 0.05
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate portfolio value after action and step
        self.portfolio_value = self.balance + self.position * next_price
        
        # Update peak portfolio value and calculate drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        current_drawdown = 0
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Penalize drawdowns more explicitly but with a reasonable scale
        if current_drawdown > 0.05:  # More than 5% drawdown
            # Apply sigmoid-like scaling to avoid extreme penalties
            drawdown_penalty = 0.2 * min(current_drawdown / 0.05, 2.0)
        
        # Calculate portfolio performance reward if no explicit reward from selling
        if action_reward == 0:
            # Calculate portfolio value change
            portfolio_value_after = self.balance + self.position * next_price
            portfolio_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
            
            # Scale reward to a reasonable range using tanh
            portfolio_performance_reward = np.tanh(portfolio_return * 5) * 0.2  # Scale to [-0.2, 0.2] range
            
            # Add small bonus/penalty for being in the market during price movements
            if self.position > 0:
                if portfolio_return > 0:
                    portfolio_performance_reward += 0.02  # Small bonus for catching uptrend
                elif portfolio_return < 0:
                    portfolio_performance_reward -= 0.02  # Small penalty for getting caught in downtrend
        
        # Apply inactivity penalty for long periods without trades
        inactivity_penalty = 0
        
        if not action_taken and not hasattr(self, 'last_trade_step'):
            self.last_trade_step = 0
            
        if action_taken:
            self.last_trade_step = self.current_step
        elif hasattr(self, 'last_trade_step'):
            # Increasing penalty for long periods without trading
            inactivity_duration = self.current_step - self.last_trade_step
            if inactivity_duration > 50:  # Arbitrary threshold
                inactivity_factor = min((inactivity_duration - 50) / 100, 1.0)
                inactivity_penalty = 0.1 * inactivity_factor
        
        # Combine all reward components with careful weighting
        reward = (
            action_reward * 1.0 +                   # Strongest weight for direct trade results
            portfolio_performance_reward * 0.8 -    # Strong weight for portfolio performance
            holding_penalty * 0.5 -                 # Moderate penalty for excessive holding
            invalid_action_penalty * 0.3 -          # Mild penalty for invalid actions
            drawdown_penalty * 0.7 -                # Significant penalty for drawdowns
            inactivity_penalty * 0.4                # Moderate penalty for inactivity
        )
        
        # Apply final scaling to keep reward in a reasonable range
        reward = np.clip(reward, -1.0, 1.0)
        
        # Check if we're done
        self.done = self.current_step >= len(self.data) - 1
        
        # Return results
        next_observation = self._get_observation()
        info = self._get_info()
        
        # Add detailed reward components to info for debugging
        info.update({
            'consecutive_holds': self.consecutive_holds,
            'reward_components': {
                'action_reward': action_reward,
                'portfolio_performance': portfolio_performance_reward,
                'holding_penalty': holding_penalty,
                'invalid_action_penalty': invalid_action_penalty,
                'drawdown_penalty': drawdown_penalty,
                'inactivity_penalty': inactivity_penalty,
                'total_reward': reward
            }
        })
        
        return next_observation, reward, self.done, False, info
    
    def _get_observation(self):
        """
        Get current observation (state).
        
        Returns:
            numpy.ndarray: Current state representation
        """
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Get current feature values
        features = self.data.iloc[self.current_step][self.features].values
        
        # Normalize position to [-1, 1]
        normalized_position = self.position / self.max_position
        
        # Normalize balance relative to initial balance
        normalized_balance = self.balance / self.initial_balance - 1.0
        
        # Combine features with position and balance
        obs = np.hstack([features, [normalized_position, normalized_balance]])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """
        Get additional information about the environment.
        
        Returns:
            dict: Information about the current state
        """
        # Calculate win rate
        win_rate = 0
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
        
        # Calculate portfolio change
        portfolio_change = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        return {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'current_price': self.current_price,
            'current_step': self.current_step,
            'total_trades': max(1, self.total_trades),  # Avoid division by zero
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'portfolio_change': portfolio_change,
            'avg_entry_price': self.avg_entry_price
        }


class TensorboardCallback(BaseCallback):
    """
    Callback for logging metrics during training.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Get environment info
        info = self.training_env.get_attr('_get_info')[0]()
        
        # Log metrics
        self.logger.record('portfolio_value', info['portfolio_value'])
        self.logger.record('win_rate', info['win_rate'])
        self.logger.record('total_trades', info['total_trades'])
        self.logger.record('portfolio_change', info['portfolio_change'])
        self.logger.record('max_drawdown', info['max_drawdown'])
        
        return True


class RLTradingStableManager(TradingManager):
    """
    Trading Manager implementation using stable-baselines3 PPO.
    
    This class extends the base TradingManager to use PPO from stable-baselines3
    for making simple trading decisions (buy, sell, hold).
    """
    
    def __init__(self, config_path: str = None, ibkr_interface = None):
        """
        Initialize RL Trading Manager.
        
        Args:
            config_path (str): Path to configuration file
            ibkr_interface: IBKR interface instance
        """
        # Initialize base trading manager
        super().__init__(config_path, ibkr_interface)
        
        # RL-specific configuration
        self.rl_config = self.trading_config.get('rl', {})
        
        # Model parameters
        self.model_path = os.path.join('models', 'rl', 'stable_trading')
        os.makedirs(self.model_path, exist_ok=True)
        
        # RL components
        self.env = None
        self.model = None
        
        # Trading parameters
        self.min_confidence = self.rl_config.get('min_confidence', 0.6)
        self.use_lstm_predictions = self.rl_config.get('use_lstm_predictions', True)
        
        # LSTM integration
        self.lstm_enabled = False
        try:
            from lstm.integration import predict_with_lstm
            self.predict_with_lstm = predict_with_lstm
            self.lstm_enabled = True
            logger.info("LSTM integration enabled for RL Trading Manager")
        except ImportError:
            logger.warning("LSTM integration not available for RL Trading Manager")
        
        # Initialize RL components
        self._initialize_rl()
        
        logger.info("RL Trading Stable Manager initialized")
    
    def _initialize_rl(self):
        """Initialize the RL environment and agent."""
        # We defer actual initialization until we have data
        logger.info("RL components will be initialized when data is available")
    
    def _initialize_with_data(self, data: pd.DataFrame):
        """
        Initialize RL environment and agent with data.
        
        Args:
            data (pd.DataFrame): Historical data with technical indicators
            
        Returns:
            bool: Success status
        """
        if data is None or data.empty:
            logger.error("Cannot initialize RL with empty data")
            return False
        
        try:
            # Log data info
            logger.info(f"Initializing RL with data: {len(data)} samples, {len(data.columns)} features")
            
            # Initialize environment
            initial_balance = self.rl_config.get('initial_balance', 10000.0)
            max_position = self.rl_config.get('max_position', 100)
            transaction_fee = self.rl_config.get('transaction_fee', 0.0005)
            
            # Create simple trading environment
            def make_env():
                env = SimpleTradingEnv(
                    data=data,
                    initial_balance=initial_balance,
                    max_position=max_position,
                    transaction_fee=transaction_fee
                )
                # Wrap environment with Monitor for logging
                env = Monitor(env)
                return env
            
            # Create vectorized environment
            self.env = DummyVecEnv([make_env])
            
            # Initialize model
            policy_kwargs = {
                'net_arch': [256, 128, 64]  # Network architecture
            }
            
            # Try to load existing model
            try:
                model_path = os.path.join(self.model_path, 'ppo_trading')
                if os.path.exists(model_path + '.zip'):
                    self.model = PPO.load(model_path, env=self.env)
                    logger.info(f"Loaded existing model from {model_path}")
                else:
                    # Create new model
                    self.model = PPO(
                        'MlpPolicy',
                        self.env,
                        verbose=1,
                        tensorboard_log=os.path.join(self.model_path, 'logs'),
                        policy_kwargs=policy_kwargs,
                        learning_rate=3e-4,
                        n_steps=2048,
                        batch_size=64,
                        gamma=0.99,
                        ent_coef=0.01
                    )
                    logger.info("Created new PPO model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Create new model
                self.model = PPO(
                    'MlpPolicy',
                    self.env,
                    verbose=1,
                    tensorboard_log=os.path.join(self.model_path, 'logs'),
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    gamma=0.99,
                    ent_coef=0.01
                )
                logger.info("Created new PPO model")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing RL components: {e}")
            return False
    
    def train_model(self, symbol: str, data: pd.DataFrame = None, epochs: int = None,
                   save_path: str = None) -> Dict:
        """
        Train the RL model on historical data.
        
        Args:
            symbol (str): Symbol to train on
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            epochs (int): Number of training timesteps (optional)
            save_path (str): Path to save model (optional)
            
        Returns:
            Dict: Training results
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.get_market_data(symbol, lookback_periods=500)
                
            if data.empty:
                return {
                    'status': 'error',
                    'message': f'No data available for {symbol}'
                }
            
            # Initialize RL components if not already done
            if self.env is None or self.model is None:
                success = self._initialize_with_data(data)
                if not success:
                    return {
                        'status': 'error',
                        'message': 'Failed to initialize RL components'
                    }
            
            # Use provided path or default
            save_dir = save_path or self.model_path
            model_save_path = os.path.join(save_dir, 'ppo_trading')
            
            # Training timesteps
            total_timesteps = epochs if epochs is not None else 100000
            
            # Create callback for logging
            callback = TensorboardCallback()
            
            # Create eval environment
            eval_env = self.env
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, 'best_model'),
                log_path=os.path.join(save_dir, 'logs'),
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            
            # Train the model
            logger.info(f"Training model for {symbol} with {total_timesteps} timesteps")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[callback, eval_callback],
                tb_log_name=f"ppo_{symbol}"
            )
            
            # Save the model
            self.model.save(model_save_path)
            logger.info(f"Model saved to {model_save_path}")
            
            # Evaluate the model
            eval_results = self.evaluate_model(symbol, data, episodes=5)
            
            return {
                'status': 'success',
                'message': f'Model trained for {total_timesteps} timesteps',
                'model_path': model_save_path,
                'evaluation': eval_results
            }
            
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def evaluate_model(self, symbol: str, data: pd.DataFrame = None, episodes: int = 10) -> Dict:
        """
        Evaluate the RL model on historical data.
        
        Args:
            symbol (str): Symbol to evaluate on
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            episodes (int): Number of evaluation episodes
            
        Returns:
            Dict: Evaluation results
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.get_market_data(symbol, lookback_periods=250)
                
            if data.empty:
                return {
                    'status': 'error',
                    'message': f'No data available for {symbol}'
                }
            
            # Initialize RL components if not already done
            if self.env is None or self.model is None:
                success = self._initialize_with_data(data)
                if not success:
                    return {
                        'status': 'error',
                        'message': 'Failed to initialize RL components'
                    }
            
            # Evaluation metrics
            metrics = {
                'episode_returns': [],
                'portfolio_values': [],
                'win_rates': [],
                'total_trades': [],
                'max_drawdowns': []
            }
            
            logger.info(f"Evaluating model on {symbol} for {episodes} episodes")
            
            # Evaluation loop
            for episode in range(episodes):
                # Versión corregida que funciona con cualquier formato de retorno
                result = self.env.reset()
                if isinstance(result, tuple) and len(result) == 2:
                    obs, _ = result
                else:
                    obs = result
                    
                done = False
                episode_reward = 0
                
                while not done:
                    # Get action from model
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Take step in environment
                    obs, reward, done, _, info = self.env.step(action)
                    
                    # Accumulate reward
                    episode_reward += reward[0]
                
                # Get final info
                final_info = info[0]
                
                # Record metrics
                metrics['episode_returns'].append(float(episode_reward))
                metrics['portfolio_values'].append(float(final_info['portfolio_value']))
                metrics['win_rates'].append(float(final_info['win_rate']))
                metrics['total_trades'].append(int(final_info['total_trades']))
                metrics['max_drawdowns'].append(float(final_info['max_drawdown']))
                
                # Log progress
                logger.info(f"Eval Episode {episode+1}/{episodes}: "
                          f"Return={episode_reward:.2f}, "
                          f"Portfolio Value={final_info['portfolio_value']:.2f}, "
                          f"Win Rate={final_info['win_rate']:.2f}, "
                          f"Trades={final_info['total_trades']}")
            
            # Calculate average metrics
            avg_return = np.mean(metrics['episode_returns'])
            avg_portfolio_value = np.mean(metrics['portfolio_values'])
            avg_win_rate = np.mean(metrics['win_rates'])
            avg_trades = np.mean(metrics['total_trades'])
            avg_drawdown = np.mean(metrics['max_drawdowns'])
            
            # Average portfolio change
            initial_balance = self.env.get_attr('initial_balance')[0]
            avg_portfolio_change = (avg_portfolio_value - initial_balance) / initial_balance
            
            logger.info(f"Evaluation results: "
                       f"Avg Return={avg_return:.2f}, "
                       f"Avg Portfolio Value={avg_portfolio_value:.2f} ({avg_portfolio_change:.2%}), "
                       f"Avg Win Rate={avg_win_rate:.2f}")
            
            return {
                'status': 'success',
                'avg_return': float(avg_return),
                'avg_portfolio_value': float(avg_portfolio_value),
                'avg_portfolio_change': float(avg_portfolio_change),
                'avg_win_rate': float(avg_win_rate),
                'avg_trades': float(avg_trades),
                'avg_max_drawdown': float(avg_drawdown),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating RL model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def generate_trading_signals(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Generate trading signals based on the RL model.
        
        Args:
            symbol (str): Symbol to generate signals for
            data (pd.DataFrame): Historical data (optional, will fetch if None)
            
        Returns:
            Dict: Trading signals
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.get_market_data(symbol)
                    
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return {'signal': 'neutral', 'strength': 0}
            
            # Initialize RL components if not already done
            if self.env is None or self.model is None:
                success = self._initialize_with_data(data)
                if not success:
                    logger.warning("Failed to initialize RL components, falling back to base method")
                    return super().generate_trading_signals(symbol, data)
            
            # Prepare observation
            # Extract features
            features = [col for col in data.columns 
                       if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
            current_data = data.iloc[-1]
            
            # Get position information
            position = 0
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                if trade['action'] == 'buy':
                    position = trade['position_size']
                elif trade['action'] == 'sell':
                    position = -trade['position_size']
            
            # Create observation
            # Get feature values
            feature_values = current_data[features].values
            
            # Normalize position
            max_position = self.rl_config.get('max_position', 100)
            normalized_position = position / max_position
            
            # Normalize balance (placeholder)
            normalized_balance = 0.0
            
            # Combine features with position and balance
            obs = np.hstack([feature_values, [normalized_position, normalized_balance]])
            obs = obs.reshape(1, -1)  # Reshape for model input
            
            # Get LSTM prediction for additional context
            lstm_prediction = None
            if self.lstm_enabled and self.use_lstm_predictions:
                try:
                    lstm_prediction = self.predict_with_lstm(symbol, data)
                except Exception as e:
                    logger.warning(f"Error getting LSTM prediction: {e}")
            
            # Get action from the RL model
            action, states = self.model.predict(obs, deterministic=True)
            action = action[0]  # Get scalar action
            
            # Map action to signal
            signal = 'neutral'
            if action == 1:
                signal = 'buy'
            elif action == 2:
                signal = 'sell'
            
            # For strength, use action probabilities if available
            # If not, use a default value
            strength = 70  # Default medium-high strength
            
            # Get action probabilities if possible
            try:
                action_probs = self.model.policy.get_distribution(states).distribution.probs
                if action_probs is not None:
                    strength = int(action_probs[0][action].numpy() * 100)
            except:
                # If we can't get probabilities, use default strength
                pass
            
            # Add reasons based on the action
            reasons = [f'rl_decision_{action}']
            
            # If LSTM prediction is available, add it as a reason
            if lstm_prediction and lstm_prediction.get('status') == 'success':
                lstm_direction = lstm_prediction.get('predicted_direction', 'neutral')
                lstm_confidence = lstm_prediction.get('confidence', 0)
                
                # Check if LSTM agrees with RL
                if (signal == 'buy' and lstm_direction == 'up') or \
                   (signal == 'sell' and lstm_direction == 'down'):
                    reasons.append(f'lstm_confirms_{lstm_confidence:.2f}')
                    strength = min(100, strength + int(lstm_confidence * 10))
                else:
                    reasons.append(f'lstm_disagrees_{lstm_direction}')
                    strength = max(20, strength - int(lstm_confidence * 10))
            
            # Prepare result
            result = {
                'symbol': symbol,
                'signal': signal,
                'strength': strength,
                'reasons': reasons,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Add LSTM prediction if available
            if lstm_prediction and lstm_prediction.get('status') == 'success':
                result['lstm_prediction'] = {
                    'direction': lstm_prediction.get('predicted_direction', 'neutral'),
                    'confidence': lstm_prediction.get('confidence', 0),
                    'price_change_pct': lstm_prediction.get('price_change_pct', 0)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating RL trading signals for {symbol}: {e}")
            # Fall back to base method
            logger.warning(f"Falling back to base method for signal generation")
            return super().generate_trading_signals(symbol, data)
    
    def process_trading_signals(self, symbol: str) -> Dict:
        """
        Process trading signals and execute trades if appropriate.
        
        Args:
            symbol (str): Symbol to process
            
        Returns:
            Dict: Processing results
        """
        try:
            # Generate signals
            signals = self.generate_trading_signals(symbol)
            
            if signals.get('status') == 'error':
                logger.error(f"Error generating signals for {symbol}: {signals.get('error')}")
                return {'status': 'error', 'message': f"Signal error: {signals.get('error')}"}
            
            # Check if we already have an active trade for this symbol
            has_active_trade = symbol in self.active_trades
            
            # Decide what to do based on signals and active trades
            if has_active_trade:
                active_trade = self.active_trades[symbol]
                active_direction = active_trade['action']
                
                # Check if signal is opposite to our position (exit signal)
                opposite_signal = (active_direction == 'buy' and signals['signal'] == 'sell') or \
                                 (active_direction == 'sell' and signals['signal'] == 'buy')
                
                if opposite_signal and signals['strength'] > 50:
                    # Close position
                    logger.info(f"Closing position for {symbol} due to opposite signal")
                    return self.close_position(symbol)
                else:
                    # No action needed, maintain position
                    return {
                        'status': 'info',
                        'message': f"Maintaining {active_direction} position for {symbol}",
                        'signal': signals['signal'],
                        'strength': signals['strength']
                    }
            else:
                # No active trade, check if we should enter a new one
                if signals['signal'] in ['buy', 'sell'] and signals['strength'] > 60:
                    # Strong signal to enter a new trade
                    
                    # Calculate position size
                    position_details = self.calculate_position_size(symbol, signals['strength'])
                    
                    if position_details.get('position_size', 0) <= 0:
                        logger.warning(f"Calculated zero position size for {symbol}")
                        return {
                            'status': 'warning',
                            'message': 'Zero position size calculated',
                            'signal': signals['signal'],
                            'strength': signals['strength']
                        }
                    
                    # Check if we can make a day trade
                    if not self.check_day_trade_limit():
                        logger.info(f"Cannot make day trade for {symbol}, reached PDT limit")
                        return {
                            'status': 'warning',
                            'message': 'PDT limit reached',
                            'signal': signals['signal'],
                            'strength': signals['strength']
                        }
                    
                    # Execute the trade
                    logger.info(f"Executing {signals['signal']} trade for {symbol} with strength {signals['strength']}")
                    return self.execute_trade(symbol, signals['signal'], position_details)
                else:
                    # Signal not strong enough to enter a trade
                    return {
                        'status': 'info',
                        'message': f"No trade: {signals['signal']} signal with strength {signals['strength']} insufficient",
                        'signal': signals['signal'],
                        'strength': signals['strength']
                    }
        
        except Exception as e:
            logger.error(f"Error processing signals for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_trading_cycle(self, symbols: list) -> Dict:
        """
        Run a complete trading cycle for a list of symbols.
        
        Args:
            symbols (list): List of symbols to process
            
        Returns:
            Dict: Results of the trading cycle
        """
        cycle_results = {
            'processed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'details': []
        }
        
        try:
            logger.info(f"Starting trading cycle for {len(symbols)} symbols")
            
            # Ensure connection to broker
            if not self.ibkr.connected:
                if not self.connect_to_broker():
                    return {'status': 'error', 'message': 'Not connected to broker'}
            
            # First manage existing positions
            position_results = self.manage_open_positions()
            cycle_results['positions_managed'] = position_results.get('monitored', 0)
            cycle_results['positions_updated'] = position_results.get('updated', 0)
            cycle_results['positions_closed'] = position_results.get('closed', 0)
            
            # Process each symbol
            for symbol in symbols:
                cycle_results['processed'] += 1
                
                try:
                    # Process trading signals
                    signal_result = self.process_trading_signals(symbol)
                    cycle_results['signals_generated'] += 1
                    
                    # Track successful trades
                    if signal_result.get('status') == 'success':
                        cycle_results['trades_executed'] += 1
                        
                    cycle_results['details'].append({
                        'symbol': symbol,
                        'status': signal_result.get('status'),
                        'action': signal_result.get('message'),
                        'signal': signal_result.get('signal'),
                        'strength': signal_result.get('strength', 0)
                    })
                    
                except Exception as e:
                    cycle_results['errors'] += 1
                    cycle_results['details'].append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
                    
                    logger.error(f"Error processing {symbol} in trading cycle: {e}")
            
            logger.info(f"Completed trading cycle: processed {cycle_results['processed']} symbols, "
                      f"executed {cycle_results['trades_executed']} trades")
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the trading strategy.
        
        Returns:
            Dict: Performance metrics
        """
        try:
            # First get base metrics
            base_metrics = super().get_performance_metrics()
            
            # Add RL-specific metrics
            rl_metrics = {
                'model_path': self.model_path,
                'model_loaded': self.model is not None,
                'environment_initialized': self.env is not None
            }
            
            # Combine metrics
            metrics = {
                **base_metrics,
                'rl_metrics': rl_metrics,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'status': 'error', 'message': str(e)}
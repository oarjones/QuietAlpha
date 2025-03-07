"""
RL Trading Manager Implementation

This module implements a Trading Manager using Reinforcement Learning (RL)
to make trading decisions. It uses a Proximal Policy Optimization (PPO)
agent to learn optimal trading strategies from market data and execution results.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional, Tuple, Union
import datetime
import time
from threading import Lock

from trading_manager.base import TradingManager
from data.processing import calculate_technical_indicators, normalize_indicators, calculate_trend_indicator
from utils.data_utils import load_config

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for trading using RL.
    
    This environment simulates a trading scenario where an agent can
    take long or short positions, and is evaluated based on returns and risk.
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                max_position: int = 100, transaction_fee: float = 0.001):
        """
        Initialize the trading environment.
        
        Args:
            data (pd.DataFrame): Historical price data with technical indicators
            initial_balance (float): Starting account balance
            max_position (int): Maximum position size (positive or negative)
            transaction_fee (float): Fee per transaction as a fraction of trade value
        """
        super(TradingEnvironment, self).__init__()
        
        # Store configuration
        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_fee = transaction_fee
        
        # State variables
        self.balance = initial_balance
        self.position = 0
        self.current_step = 0
        self.current_price = 0
        self.last_trade_price = 0
        self.trades = []
        self.portfolio_value = initial_balance
        self.done = False
        
        # Extract feature columns (indicators)
        self.features = [col for col in data.columns 
                        if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Trading costs calculation
        self.trade_cost_pct = transaction_fee
        
        # Define observation space (state space)
        # Includes technical indicators + position + balance
        self.obs_dim = len(self.features) + 2  # +2 for position and balance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Define action space
        # Action: (position_delta_pct, stop_loss_pct, take_profit_pct)
        # position_delta_pct: How much to change position (-1 to 1)
        # stop_loss_pct: Stop loss distance in percent (0 to 0.1)
        # take_profit_pct: Take profit distance in percent (0 to 0.1)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), 
            high=np.array([1.0, 0.1, 0.1]),
            dtype=np.float32
        )
        
        # Trading session stats
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = initial_balance
        
        # Reset on init
        self.reset()
    
    def reset(self, seed=None):
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
        self.last_trade_price = 0
        self.trades = []
        self.portfolio_value = self.initial_balance
        self.done = False
        
        # Reset statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_balance
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: [position_delta_pct, stop_loss_pct, take_profit_pct]
            
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
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.position * self.current_price
        
        # Process the action to determine position delta and risk parameters
        position_delta_pct = action[0]  # Between -1 and 1
        stop_loss_pct = action[1]       # Between 0 and 0.1
        take_profit_pct = action[2]     # Between 0 and 0.1
        
        # Calculate target position change
        position_delta = int(position_delta_pct * self.max_position)
        
        # Calculate new desired position
        new_position = self.position + position_delta
        
        # Clip position to allowed range
        new_position = max(-self.max_position, min(self.max_position, new_position))
        
        # Calculate actual position delta after clipping
        actual_position_delta = new_position - self.position
        
        # If position changes, execute trade
        if actual_position_delta != 0:
            # Calculate trade cost
            trade_cost = abs(actual_position_delta) * self.current_price * self.trade_cost_pct
            
            # Calculate trade value
            trade_value = actual_position_delta * self.current_price
            
            # Check if we can afford this trade
            if self.balance - trade_value - trade_cost < 0 and actual_position_delta > 0:
                # If buying, ensure we have enough balance
                actual_position_delta = max(0, int((self.balance - trade_cost) / self.current_price))
                new_position = self.position + actual_position_delta
                trade_value = actual_position_delta * self.current_price
                trade_cost = abs(actual_position_delta) * self.current_price * self.trade_cost_pct
            
            # Execute trade if there's any position change
            if actual_position_delta != 0:
                # Update balance
                self.balance -= trade_value + trade_cost
                
                # Update position
                self.position = new_position
                
                # Record trade
                self.last_trade_price = self.current_price
                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'position_delta': actual_position_delta,
                    'position': self.position,
                    'cost': trade_cost,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct
                })

                # Log progress every 50 steps
                if self.current_step % 50 == 0:
                    logger.info(f"Step {self.current_step}: "
                                f"price={self.current_price:.2f}, "
                                f"position_delta={actual_position_delta:.2f}, " 
                                f"position={self.position}, "
                                f"cost={trade_cost:.2f}, "
                                f"stop_loss_pct={stop_loss_pct:.2f}, "
                                f"take_profit_pct={take_profit_pct:.2f}, "
                                f"cumulative_reward={self.portfolio_value - self.initial_balance:.2f}"
                                )
                
                
                self.total_trades += 1
        
        # Move to the next time step
        self.current_step += 1
        
        # Get new price after time step
        next_price = next_data['close']
        
        # Check if we hit stop loss or take profit based on previous trade
        hit_stop_loss = False
        hit_take_profit = False
        
        if self.position != 0 and self.last_trade_price > 0:
            # Calculate price change since last trade
            price_change_pct = (next_price - self.last_trade_price) / self.last_trade_price
            
            # Check for stop loss (for both long and short positions)
            if (self.position > 0 and price_change_pct <= -stop_loss_pct) or \
            (self.position < 0 and price_change_pct >= stop_loss_pct):
                hit_stop_loss = True
            
            # Check for take profit (for both long and short positions)
            if (self.position > 0 and price_change_pct >= take_profit_pct) or \
            (self.position < 0 and price_change_pct <= -take_profit_pct):
                hit_take_profit = True
        
        # If we hit stop loss or take profit, close the position
        if (hit_stop_loss or hit_take_profit) and self.position != 0:
            # Calculate trade cost
            trade_cost = abs(self.position) * next_price * self.trade_cost_pct
            
            # Calculate trade value
            trade_value = -self.position * next_price
            
            # Execute trade
            self.balance -= trade_value + trade_cost
            
            # Calculate P&L for this trade
            if self.last_trade_price > 0:
                price_change = next_price - self.last_trade_price
                
                # Para posiciones largas, ganamos si el precio sube
                if (self.position > 0 and price_change > 0) or \
                (self.position < 0 and price_change < 0):
                    self.winning_trades += 1
                    
            # Record trade
            exit_reason = 'stop_loss' if hit_stop_loss else 'take_profit'
            self.trades.append({
                'step': self.current_step,
                'price': next_price,
                'position_delta': -self.position,
                'position': 0,
                'cost': trade_cost,
                'reason': exit_reason
            })
            
            # Reset position
            self.position = 0
            self.last_trade_price = 0
            
            self.total_trades += 1
        
        # Calculate portfolio value after action
        self.portfolio_value = self.balance + self.position * next_price
        
        # Update peak portfolio value and calculate drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_value_before)
        
        # Check if we're done
        self.done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, self.done, False, self._get_info()
    
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
        
        # Normalize balance
        normalized_balance = self.balance / self.initial_balance - 1.0
        
        # Combine features with position and balance
        obs = np.hstack([features, [normalized_position, normalized_balance]])
        
        return obs.astype(np.float32)
    
    
    def _get_info(self):
        """
        Get additional information about the environment.
        """
        return {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'current_price': self.current_price,
            'current_step': self.current_step,
            'total_trades': max(1, self.total_trades),  # Evitar división por cero
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'max_drawdown': self.max_drawdown
        }
    
    def _calculate_reward(self, previous_portfolio_value):
        # Recompensa principal: cambio porcentual en el portfolio
        portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Escalar el retorno para que sea significativo, pero no aplastando los retornos negativos
        if portfolio_return > 0:
            scaled_return = portfolio_return * 10.0  # Valores más razonables
        else:
            scaled_return = portfolio_return * 5.0   # Menos penalización por pérdidas
        
        # Bonificación por mantener efectivo en mercados bajistas
        current_trend = 0
        if hasattr(self, 'data') and 'TrendStrength_norm' in self.data.columns:
            current_idx = min(self.current_step, len(self.data) - 1)
            current_trend = self.data['TrendStrength_norm'].iloc[current_idx]
        
        # Bonificaciones más pequeñas para no dominar el retorno del portfolio
        cash_bonus = 0.1 if current_trend < 0.4 and self.position == 0 else 0
        position_bonus = 0.1 if current_trend > 0.6 and self.position > 0 else 0
        
        # Penalización menos severa por drawdown
        drawdown_penalty = max(0, (self.max_drawdown - 0.1) * 5.0)
        
        # Recompensa final
        reward = scaled_return + cash_bonus + position_bonus - drawdown_penalty
        
        # Pequeña bonificación adicional por portfolio positivo
        if self.portfolio_value > self.initial_balance:
            reward += 0.1
        
        return reward


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for trading.
    
    This class implements a PPO agent specifically designed for the trading environment.
    It builds, trains, and uses a policy network to make trading decisions.
    """
    
    def __init__(self, state_dim, action_dim, save_path='models/rl/ppo_trading'):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            save_path (str): Path to save model checkpoints
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_path = save_path

        
        
        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)
        
        # Hyperparameters
        self.actor_lr = 0.0005
        self.critic_lr = 0.001
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_update_epochs = 10
        
        self.initial_entropy_coef = 0.1
        self.final_entropy_coef = 0.01
        self.entropy_coef = self.initial_entropy_coef
        self.entropy_decay_episodes = 50
        
        # Build networks
        self.actor, self.critic = self._build_networks()
        
        # Optimizer
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        
        # Episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        # Training metrics
        self.metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'episode_returns': [],
            'episode_lengths': []
        }
    
    def _build_networks(self):
        # Actor network - policy network
        actor_inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(512, activation='relu')(actor_inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Actor outputs mean values for each action dimension
        action_means = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        
        # Actor also outputs log standard deviations
        log_stds = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        log_stds = tf.keras.layers.Lambda(lambda x: x * 0.5 - 2.0)(log_stds)  # Valores iniciales más pequeños
        
        actor = tf.keras.Model(inputs=actor_inputs, outputs=[action_means, log_stds])
        
        # Critic network - value function
        critic_inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(512, activation='relu')(critic_inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        critic_outputs = tf.keras.layers.Dense(1)(x)
        
        critic = tf.keras.Model(inputs=critic_inputs, outputs=critic_outputs)
        
        return actor, critic
    
    def choose_action(self, state, training=True):
        """
        Choose an action using the policy network.
        
        Args:
            state: Current state
            training: Whether to add exploration noise
            
        Returns:
            numpy.ndarray: Selected action
            float: Log probability of selected action
        """
        state = np.array([state], dtype=np.float32)
        
        # Get action mean and log_std from actor network
        action_mean, log_std = self.actor(state)
        
        if not training:
            # During evaluation, use the mean action (no exploration)
            return action_mean[0].numpy(), 0.0
        
        # Convert log_std to std
        std = tf.exp(log_std)
        
        # Sample action from normal distribution
        normal_dist = tf.random.normal(shape=action_mean.shape)
        action = action_mean + normal_dist * std
        
        # Apply tanh to ensure actions are within [-1, 1]
        action = tf.tanh(action)
        
        # Compute log probability
        # Using a simplified version for now
        log_prob = -0.5 * tf.reduce_sum(tf.square((action - action_mean) / (std + 1e-8)))
        
        return action[0].numpy(), log_prob.numpy()
    
    def get_value(self, state):
        """
        Get state value using the critic network.
        
        Args:
            state: Current state
            
        Returns:
            float: State value
        """
        state = np.array([state], dtype=np.float32)
        
        return self.critic(state)[0, 0].numpy()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """
        Store a transition in the episode buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value of state
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear_buffer(self):
        """Clear the episode buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def compute_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute advantage estimates using Generalized Advantage Estimation (GAE).
        
        Args:
            last_value: Value of the last state
            gamma: Discount factor
            lam: GAE parameter
            
        Returns:
            tuple: (returns, advantages)
        """
        values = np.append(self.values, last_value)
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
        
        return np.array(returns), np.array(advantages)
    
    @tf.function
    def actor_loss(self, states, actions, old_log_probs, advantages, clip_ratio):
        """
        Compute the actor (policy) loss.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            advantages: Batch of advantages
            clip_ratio: PPO clipping parameter
            
        Returns:
            tuple: (actor_loss, entropy)
        """
        with tf.GradientTape() as tape:
            action_means, log_stds = self.actor(states)
            stds = tf.exp(log_stds)
            
            # Compute log probabilities of actions
            normal_dist = tf.compat.v1.distributions.Normal(loc=action_means, scale=stds)
            new_log_probs = tf.reduce_sum(normal_dist.log_prob(actions), axis=1)
            
            # Compute ratio and clipped ratio
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            
            # Compute surrogate losses
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            
            # Compute entropy for more exploration
            entropy = tf.reduce_mean(normal_dist.entropy())
            
            # Actor loss
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2)) - self.entropy_coef * entropy
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return actor_loss, entropy
    
    @tf.function
    def critic_loss(self, states, returns):
        """
        Compute the critic (value) loss.
        
        Args:
            states: Batch of states
            returns: Batch of returns
            
        Returns:
            float: Critic loss
        """
        with tf.GradientTape() as tape:
            predicted_values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - predicted_values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return critic_loss
    
    def train(self, batch_size=64, episode=0):
        """
        Train the PPO model using the collected experience.
        
        Args:
            batch_size: Training batch size
            
        Returns:
            dict: Training metrics
        """
        # Convert buffer data to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)

        # Actualizar coeficiente de entropía
        if episode < self.entropy_decay_episodes:
            self.entropy_coef = self.initial_entropy_coef - (self.initial_entropy_coef - self.final_entropy_coef) * (episode / self.entropy_decay_episodes)
        else:
            self.entropy_coef = self.final_entropy_coef
        
        # Compute returns and advantages
        returns, advantages = self.compute_advantages(
            self.get_value(self.next_states[-1]),
            gamma=self.gamma,
            lam=self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Create dataset for mini-batch training
        dataset = tf.data.Dataset.from_tensor_slices(
            (states_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor)
        )
        dataset = dataset.shuffle(buffer_size=len(states)).batch(batch_size)
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        entropies = []
        
        
        # Multiple epochs of training
        for _ in range(self.policy_update_epochs):
            for batch in dataset:
                states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch = batch
                
                # Update actor network
                actor_loss, entropy = self.actor_loss(
                    states_batch, actions_batch, old_log_probs_batch, advantages_batch, self.clip_ratio
                )
                
                # Update critic network
                critic_loss = self.critic_loss(states_batch, returns_batch)
                
                # Store metrics
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                entropies.append(entropy.numpy())
        
        # Calculate average metrics
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        avg_entropy = np.mean(entropies)
        
        # Store metrics for tracking
        self.metrics['actor_loss'].append(avg_actor_loss)
        self.metrics['critic_loss'].append(avg_critic_loss)
        self.metrics['entropy'].append(avg_entropy)
        
        # Clear buffer after training
        self.clear_buffer()
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy
        }
    
    def save_model(self, episode=None):
        """
        Save the model weights.
        
        Args:
            episode: Current episode number (optional)
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            
            # Define filenames with correct .weights.h5 extension
            if episode is not None:
                actor_path = os.path.join(self.save_path, f'actor_episode_{episode}.weights.h5')
                critic_path = os.path.join(self.save_path, f'critic_episode_{episode}.weights.h5')
            else:
                actor_path = os.path.join(self.save_path, 'actor.weights.h5')
                critic_path = os.path.join(self.save_path, 'critic.weights.h5')
            
            # Save weights
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            
            logger.info(f"Model weights saved to {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            return False
    
    def load_model(self, actor_path=None, critic_path=None):
        """
        Load model weights.
        
        Args:
            actor_path: Path to actor weights (optional)
            critic_path: Path to critic weights (optional)
            
        Returns:
            bool: Whether the model was loaded successfully
        """
        try:
            if actor_path is None:
                actor_path = os.path.join(self.save_path, 'actor.weights.h5')
            if critic_path is None:
                critic_path = os.path.join(self.save_path, 'critic.weights.h5')
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.actor.load_weights(actor_path)
                self.critic.load_weights(critic_path)
                logger.info(f"Model weights loaded from {self.save_path}")
                return True
            else:
                logger.warning(f"Model weight files not found at {self.save_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            return False


class RLTradingManager(TradingManager):
    """
    Trading Manager implementation using Reinforcement Learning.
    
    This class extends the base TradingManager to use a PPO agent for making trading decisions.
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
        self.model_path = os.path.join('models', 'rl', 'trading_manager')
        os.makedirs(self.model_path, exist_ok=True)
        
        # RL agent
        self.env = None
        self.agent = None
        
        # Training parameters
        self.num_episodes = self.rl_config.get('num_episodes', 100)
        self.max_steps_per_episode = self.rl_config.get('max_steps_per_episode', 1000)
        self.save_freq = self.rl_config.get('save_freq', 10)
        self.eval_freq = self.rl_config.get('eval_freq', 5)
        self.warmup_episodes = self.rl_config.get('warmup_episodes', 5)
        
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
    
    def _initialize_rl(self):
        """Initialize the RL environment and agent."""
        # We defer actual initialization until we have data
        logger.info("RL components will be initialized when data is available")
    
    def _initialize_with_data(self, data: pd.DataFrame):
        """
        Initialize RL environment and agent with data.
        
        Args:
            data (pd.DataFrame): Historical data with technical indicators
        """
        if data is None or data.empty:
            logger.error("Cannot initialize RL with empty data")
            return False
        
        try:
            # Initialize environment
            self.env = TradingEnvironment(
                data=data,
                initial_balance=self.rl_config.get('initial_balance', 10000.0),
                max_position=self.rl_config.get('max_position', 100),
                transaction_fee=self.rl_config.get('transaction_fee', 0.0005)
            )
            
            # Get state and action dimensions
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0]
            
            # Initialize agent
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                save_path=self.model_path
            )
            
            # Try to load pre-trained model
            if not self.agent.load_model():
                logger.warning("No pre-trained model found, agent will start from scratch")
            else:
                logger.info("Pre-trained model loaded successfully")
            
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
            epochs (int): Number of training episodes (optional)
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
            if self.env is None or self.agent is None:
                success = self._initialize_with_data(data)
                if not success:
                    return {
                        'status': 'error',
                        'message': 'Failed to initialize RL components'
                    }
            
            # Number of episodes
            num_episodes = epochs if epochs is not None else self.num_episodes
            
            # Training metrics
            metrics = {
                'episode_returns': [],
                'episode_lengths': [],
                'win_rates': [],
                'max_drawdowns': [],
                'final_portfolio_values': [],
                'cumulative_rewards': []  # Nueva métrica para recompensas acumuladas
            }
            
            # Training loop
            for episode in range(num_episodes):
                # Reset environment
                state, info = self.env.reset()
                
                episode_return = 0
                episode_length = 0
                cumulative_reward = 0  # Inicializar recompensa acumulada
                
                # Episode loop
                done = False
                while not done and episode_length < self.max_steps_per_episode:
                    # Choose action
                    action, log_prob = self.agent.choose_action(state)
                    
                    # Get state value
                    value = self.agent.get_value(state)
                    
                    # Take step in environment
                    next_state, reward, done, _, info = self.env.step(action)
                    
                    # Acumular recompensa
                    cumulative_reward += reward
                    
                    # Store transition
                    self.agent.store_transition(state, action, reward, next_state, done, log_prob, value)
                    
                    # Update state
                    state = next_state
                    
                    # Update metrics
                    episode_return += reward
                    episode_length += 1
                
                # Train agent after each episode
                train_metrics = self.agent.train(batch_size=min(64, episode_length), episode=episode)
                
                # Record metrics
                metrics['episode_returns'].append(episode_return)
                metrics['episode_lengths'].append(episode_length)
                metrics['win_rates'].append(info['win_rate'])
                metrics['max_drawdowns'].append(info['max_drawdown'])
                metrics['final_portfolio_values'].append(info['portfolio_value'])
                metrics['cumulative_rewards'].append(cumulative_reward)  # Guardar recompensa acumulada
                
                # Log progress with cumulative reward
                logger.info(f"Episode {episode+1}/{num_episodes}: "
                        f"Return={episode_return:.2f}, "
                        f"Cumulative Reward={cumulative_reward:.2f}, "  # Mostrar recompensa acumulada
                        f"Length={episode_length}, "
                        f"Win Rate={info['win_rate']:.2f}, "
                        f"Portfolio Value={info['portfolio_value']:.2f}")
                
                # Save model periodically
                if (episode + 1) % self.save_freq == 0:
                    if save_path:
                        self.agent.save_model(episode=episode)
                    else:
                        self.agent.save_model()
            
            # Save final model
            if save_path:
                self.agent.save_model()
            else:
                self.agent.save_model()
            
            return {
                'status': 'success',
                'message': f'Model trained for {num_episodes} episodes',
                'metrics': metrics,
                'final_portfolio_value': metrics['final_portfolio_values'][-1],
                'final_win_rate': metrics['win_rates'][-1],
                'final_cumulative_reward': metrics['cumulative_rewards'][-1]  # Devolver recompensa acumulada final
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
            if self.env is None or self.agent is None:
                success = self._initialize_with_data(data)
                if not success:
                    return {
                        'status': 'error',
                        'message': 'Failed to initialize RL components'
                    }
            
            # Evaluation metrics
            metrics = {
                'episode_returns': [],
                'episode_lengths': [],
                'win_rates': [],
                'max_drawdowns': [],
                'final_portfolio_values': [],
                'trades': []
            }
            
            # Evaluation loop
            for episode in range(episodes):
                # Reset environment
                state, info = self.env.reset()
                
                episode_return = 0
                episode_length = 0
                
                # Episode loop
                done = False
                while not done and episode_length < self.max_steps_per_episode:
                    # Choose action without exploration
                    action, _ = self.agent.choose_action(state, training=False)
                    
                    # Take step in environment
                    next_state, reward, done, _, info = self.env.step(action)
                    
                    # Update state
                    state = next_state
                    
                    # Update metrics
                    episode_return += reward
                    episode_length += 1
                
                # Record metrics
                metrics['episode_returns'].append(episode_return)
                metrics['episode_lengths'].append(episode_length)
                metrics['win_rates'].append(info['win_rate'])
                metrics['max_drawdowns'].append(info['max_drawdown'])
                metrics['final_portfolio_values'].append(info['portfolio_value'])
                metrics['trades'].append(self.env.trades.copy())
                
                # Log progress
                logger.info(f"Eval Episode {episode+1}/{episodes}: "
                          f"Return={episode_return:.2f}, "
                          f"Win Rate={info['win_rate']:.2f}, "
                          f"Portfolio Value={info['portfolio_value']:.2f}")
            
            # Calculate average metrics
            avg_return = np.mean(metrics['episode_returns'])
            avg_win_rate = np.mean(metrics['win_rates'])
            avg_max_drawdown = np.mean(metrics['max_drawdowns'])
            avg_portfolio_value = np.mean(metrics['final_portfolio_values'])
            
            return {
                'status': 'success',
                'message': f'Model evaluated for {episodes} episodes',
                'avg_return': float(avg_return),
                'avg_win_rate': float(avg_win_rate),
                'avg_max_drawdown': float(avg_max_drawdown),
                'avg_portfolio_value': float(avg_portfolio_value),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating RL model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_action(self, state: np.ndarray, confidence_threshold: float = None) -> Tuple[np.ndarray, float]:
        """
        Predict action using the RL model.
        
        Args:
            state: Current state
            confidence_threshold: Minimum confidence threshold (optional)
            
        Returns:
            tuple: (action, confidence)
        """
        if self.agent is None:
            logger.error("RL agent not initialized")
            return np.zeros(3), 0.0
        
        # Get confidence threshold
        if confidence_threshold is None:
            confidence_threshold = self.min_confidence
        
        # Predict action
        action, _ = self.agent.choose_action(state, training=False)
        
        # Calculate confidence based on action values
        # For position_delta_pct, higher absolute value means more confidence
        confidence = abs(action[0])
        
        # If confidence is below threshold, adjust action to do nothing
        if confidence < confidence_threshold:
            action[0] = 0.0  # No position change
        
        return action, confidence
    
    def generate_trading_signals(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Generate trading signals based on the RL model.
        
        This overrides the base method to use the RL model for signal generation.
        
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
            if self.env is None or self.agent is None:
                success = self._initialize_with_data(data)
                if not success:
                    logger.warning("Failed to initialize RL components, falling back to base method")
                    return super().generate_trading_signals(symbol, data)
            
            # Extract state features from data
            features = [col for col in data.columns 
                      if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Get current state
            current_data = data.iloc[-1]
            state_features = current_data[features].values
            
            # Get current position and balance (based on current active trades)
            position = 0
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                if trade['action'] == 'buy':
                    position = trade['position_size']
                elif trade['action'] == 'sell':
                    position = -trade['position_size']
            
            # Normalize position
            max_position = self.rl_config.get('max_position', 100)
            normalized_position = position / max_position
            
            # Normalize balance (using a placeholder since we don't track balance directly)
            normalized_balance = 0.0
            
            # Combine features with position and balance
            state = np.hstack([state_features, [normalized_position, normalized_balance]])
            
            # Get LSTM prediction if available
            lstm_prediction = None
            if self.lstm_enabled and self.use_lstm_predictions:
                try:
                    lstm_prediction = self.predict_with_lstm(symbol, data)
                    
                    if lstm_prediction.get('status') == 'success':
                        # Add LSTM prediction to state
                        lstm_direction = lstm_prediction.get('predicted_direction', 'neutral')
                        lstm_confidence = lstm_prediction.get('confidence', 0)
                        
                        # Convert direction to numeric value
                        lstm_value = 0.0
                        if lstm_direction == 'up':
                            lstm_value = lstm_confidence
                        elif lstm_direction == 'down':
                            lstm_value = -lstm_confidence
                        
                        # Append to state
                        state = np.append(state, lstm_value)
                except Exception as e:
                    logger.warning(f"Error getting LSTM prediction: {e}")
            
            # Predict action
            action, confidence = self.predict_action(state)
            
            # Parse the action
            position_delta_pct = action[0]  # Between -1 and 1
            stop_loss_pct = action[1]       # Between 0 and 0.1
            take_profit_pct = action[2]     # Between 0 and 0.1
            
            # Determine signal based on position delta
            signal = 'neutral'
            if position_delta_pct > 0.2:
                signal = 'buy'
            elif position_delta_pct < -0.2:
                signal = 'sell'
            
            # Calculate signal strength (0-100)
            strength = int(abs(position_delta_pct) * 100)
            
            # Determine reasons for signal
            reasons = []
            if signal != 'neutral':
                reasons.append(f'rl_decision_{position_delta_pct:.2f}')
                reasons.append(f'sl_{stop_loss_pct:.3f}_tp_{take_profit_pct:.3f}')
                
                if lstm_prediction and lstm_prediction.get('status') == 'success':
                    lstm_direction = lstm_prediction.get('predicted_direction', 'neutral')
                    lstm_confidence = lstm_prediction.get('confidence', 0)
                    
                    # Check if LSTM agrees with RL
                    if (signal == 'buy' and lstm_direction == 'up') or \
                       (signal == 'sell' and lstm_direction == 'down'):
                        reasons.append(f'lstm_confirms_{lstm_confidence:.2f}')
                    else:
                        reasons.append(f'lstm_disagrees_{lstm_direction}')
            
            # Prepare response
            result = {
                'symbol': symbol,
                'signal': signal,
                'strength': strength,
                'reasons': reasons,
                'rl_action': {
                    'position_delta_pct': float(position_delta_pct),
                    'stop_loss_pct': float(stop_loss_pct),
                    'take_profit_pct': float(take_profit_pct),
                    'confidence': float(confidence)
                },
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
    
    def execute_trade(self, symbol: str, action: str, position_details: Dict) -> Dict:
        """
        Execute a trade with RL-based risk management.
        
        This overrides the base method to use RL-determined stop loss and take profit levels.
        
        Args:
            symbol (str): Symbol to trade
            action (str): Trade action (buy/sell)
            position_details (Dict): Position sizing details
            
        Returns:
            Dict: Trade execution results
        """
        # Get RL signal
        data = self.get_market_data(symbol)
        rl_signal = self.generate_trading_signals(symbol, data)
        
        # If we have valid RL signals, use them for risk management
        if 'rl_action' in rl_signal:
            rl_action = rl_signal['rl_action']
            
            # Override stop loss and take profit with RL values if they exist
            if 'stop_loss_pct' in rl_action and rl_action['stop_loss_pct'] > 0:
                # Calculate stop loss price based on percentage
                current_price = position_details.get('current_price', 0)
                if current_price > 0:
                    if action == 'buy':
                        stop_loss_price = current_price * (1 - rl_action['stop_loss_pct'])
                    else:  # sell
                        stop_loss_price = current_price * (1 + rl_action['stop_loss_pct'])
                    
                    position_details['stop_loss_price'] = stop_loss_price
            
            # Same for take profit
            if 'take_profit_pct' in rl_action and rl_action['take_profit_pct'] > 0:
                current_price = position_details.get('current_price', 0)
                if current_price > 0:
                    if action == 'buy':
                        take_profit_price = current_price * (1 + rl_action['take_profit_pct'])
                    else:  # sell
                        take_profit_price = current_price * (1 - rl_action['take_profit_pct'])
                    
                    position_details['take_profit_price'] = take_profit_price
        
        # Execute trade using base method with possibly updated position details
        return super().execute_trade(symbol, action, position_details)
    
    def process_trading_signals(self, symbol: str) -> Dict:
        """
        Process trading signals and execute trades if appropriate.
        
        This overrides the base method to use the RL-based signal generator.
        
        Args:
            symbol (str): Symbol to process
            
        Returns:
            Dict: Processing results
        """
        try:
            # Generate RL-based trading signals
            signals = self.generate_trading_signals(symbol)
            
            if 'error' in signals:
                logger.error(f"Error generating signals for {symbol}: {signals.get('error')}")
                return {'status': 'error', 'message': f"Signal error: {signals.get('error')}"}
            
            # Check if we already have an active trade for this symbol
            has_active_trade = symbol in self.active_trades
            
            # Get the RL action for position sizing
            rl_action = signals.get('rl_action', {})
            position_delta_pct = rl_action.get('position_delta_pct', 0)
            
            # Decide what to do based on signals and active trades
            if has_active_trade:
                active_trade = self.active_trades[symbol]
                active_direction = active_trade['action']
                
                # Check if signal is opposite to our position (exit signal)
                opposite_signal = (active_direction == 'buy' and signals['signal'] == 'sell') or \
                                 (active_direction == 'sell' and signals['signal'] == 'buy')
                
                # Also check if RL suggests closing position
                close_position = abs(position_delta_pct) < 0.1
                
                if opposite_signal or close_position:
                    # Close position
                    logger.info(f"Closing position for {symbol} based on RL signal")
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
                if signals['signal'] in ['buy', 'sell'] and signals['strength'] > 50:
                    # Strong signal to enter a new trade
                    
                    # Calculate position size
                    position_details = self.calculate_position_size(symbol, signals['strength'])
                    
                    # Adjust position size based on RL confidence
                    rl_confidence = rl_action.get('confidence', 0.5)
                    adjusted_size = int(position_details.get('position_size', 0) * max(0.2, rl_confidence))
                    position_details['position_size'] = adjusted_size
                    
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
        
        This overrides the base method to include RL-specific behavior.
        
        Args:
            symbols (list): List of symbols to process
            
        Returns:
            Dict: Results of the trading cycle
        """
        # Use the base implementation with our RL-enhanced methods
        return super().run_trading_cycle(symbols)
"""
Reinforcement Learning Trading Manager

This module provides a specialized implementation of the Trading Manager
that uses Reinforcement Learning to optimize trading decisions.
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras as ke
from typing import List, Dict, Optional, Tuple, Union
import datetime
import random
from collections import deque

from trading_manager.base import TradingManager
from data.processing import calculate_technical_indicators, normalize_indicators

logger = logging.getLogger(__name__)

class RLTradingManager(TradingManager):
    """
    Trading Manager implementation that uses Reinforcement Learning
    to optimize trading decisions.
    """
    
    def __init__(self, config_path: str = None, ibkr_interface = None):
        """
        Initialize RL Trading Manager.
        
        Args:
            config_path (str): Path to configuration file
            ibkr_interface: IBKR interface instance
        """
        super().__init__(config_path, ibkr_interface)
        
        # RL specific configurations
        self.rl_config = {
            'learning_rate': 0.001,
            'gamma': 0.95,  # Discount factor
            'epsilon': 1.0,  # Exploration rate
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'memory_size': 10000,  # Replay memory size
            'batch_size': 32,
            'train_interval': 100,  # Train every N steps
            'target_update_interval': 500,  # Copy weights every N steps
        }
        
        # Override model path
        self.model_path = os.path.join('models', 'rl_trading')
        os.makedirs(self.model_path, exist_ok=True)
        
        # RL components
        self.memory = deque(maxlen=self.rl_config['memory_size'])
        self.state_size = None  # Will be set when creating the model
        self.action_size = 3  # hold, buy, sell
        self.q_network = None
        self.target_network = None
        self.step_count = 0
        
        # Trading state
        self.current_state = {}  # Symbol -> state
        
        # Initialize RL model
        self._initialize_rl()
    
    def _initialize_rl(self):
        """Initialize the Reinforcement Learning model."""
        # Try to load pre-trained model
        rl_model_path = os.path.join(self.model_path, 'q_network.keras')
        if os.path.exists(rl_model_path):
            try:
                self.q_network = ke.models.load_model(rl_model_path)
                self.state_size = self.q_network.input_shape[1]
                self.target_network = ke.models.clone_model(self.q_network)
                self.target_network.set_weights(self.q_network.get_weights())
                logger.info(f"Loaded pre-trained RL model with state size {self.state_size}")
            except Exception as e:
                logger.error(f"Error loading RL model: {e}")
                self._build_rl_model()
        else:
            self._build_rl_model()
    
    def _build_rl_model(self):
        """Build the Q-network model."""
        # Define state size based on features
        self.state_size = 20  # Default size, will be updated when extracting features
        
        # Create Q-network
        self.q_network = ke.Sequential([
            ke.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            ke.layers.Dense(64, activation='relu'),
            ke.layers.Dense(self.action_size, activation='linear')
        ])
        
        self.q_network.compile(
            loss='mse',
            optimizer=ke.optimizers.Adam(learning_rate=self.rl_config['learning_rate'])
        )
        
        # Create target network (same architecture)
        self.target_network = ke.models.clone_model(self.q_network)
        self.target_network.set_weights(self.q_network.get_weights())
        
        logger.info(f"Built RL model with state size {self.state_size}")
    
    def save_rl_model(self):
        """Save the RL model to disk."""
        if self.q_network:
            try:
                model_path = os.path.join(self.model_path, 'q_network.keras')
                self.q_network.save(model_path)
                logger.info(f"Saved RL model to {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving RL model: {e}")
                return False
        return False
    
    def extract_state(self, symbol: str, data: pd.DataFrame) -> np.ndarray:
        """
        Extract state representation from market data.
        
        Args:
            symbol (str): Symbol to extract state for
            data (pd.DataFrame): Market data
            
        Returns:
            np.ndarray: State representation
        """
        if data.empty:
            # Return zero state if no data
            return np.zeros(self.state_size)
        
        # Use the most recent data point
        latest = data.iloc[-1]
        
        # Create state vector
        state = []
        
        # 1. Price-based features
        for col in ['close_norm', 'open_norm', 'high_norm', 'low_norm']:
            if col in latest:
                state.append(latest[col])
            else:
                state.append(0.5)  # Default value
        
        # 2. Trend indicators
        state.append(latest.get('TrendStrength_norm', 0.5))
        state.append(latest.get('ADX_14_norm', 0.5))
        
        # 3. Moving average indicators
        if all(col in latest for col in ['close', 'EMA_8', 'EMA_21', 'SMA_34']):
            state.append(latest['close'] / latest['EMA_8'] - 1)  # Normalized distance from EMA_8
            state.append(latest['close'] / latest['EMA_21'] - 1)  # Normalized distance from EMA_21
            state.append(latest['close'] / latest['SMA_34'] - 1)  # Normalized distance from SMA_34
        else:
            state.extend([0, 0, 0])
        
        # 4. Momentum indicators
        state.append(latest.get('RSI_14_norm', 0.5))
        state.append(latest.get('Stoch_k_norm', 0.5))
        state.append(latest.get('Stoch_d_norm', 0.5))
        
        # 5. Volatility indicators
        state.append(latest.get('ATR_14_norm', 0.5))
        state.append(latest.get('BB_Width_norm', 0.5))
        
        # 6. Volume indicators
        state.append(latest.get('volume_norm', 0.5))
        state.append(latest.get('OBV_norm', 0.5))
        
        # 7. LSTM prediction if available
        lstm_prediction = self.predict_price_with_lstm(symbol, data)
        if lstm_prediction['predicted_direction'] == 'up':
            state.append(0.5 + lstm_prediction['confidence'] / 2)
        elif lstm_prediction['predicted_direction'] == 'down':
            state.append(0.5 - lstm_prediction['confidence'] / 2)
        else:
            state.append(0.5)
        
        # 8. Position flags (0: no position, 1: long, -1: short)
        position_flag = 0
        if symbol in self.active_trades:
            position_flag = 1 if self.active_trades[symbol]['action'] == 'buy' else -1
        state.append(position_flag)
        
        # 9. Market regime indicators (if available)
        # This could be added in a more sophisticated implementation
        
        # Ensure state vector has the right size
        state = np.array(state, dtype=np.float32)
        
        # Update state size if this is the first state we're creating
        if self.state_size != len(state):
            self.state_size = len(state)
            logger.info(f"Updated state size to {self.state_size}")
            self._build_rl_model()  # Rebuild model with new state size
        
        return state
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Get action from Q-network with epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            explore (bool): Whether to explore or exploit
            
        Returns:
            int: Action index (0: hold, 1: buy, 2: sell)
        """
        if explore and random.random() < self.rl_config['epsilon']:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action from Q-network
            q_values = self.q_network.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the Q-network from replay memory."""
        if len(self.memory) < self.rl_config['batch_size']:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.rl_config['batch_size'])
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        targets = self.q_network.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values for actions taken
        for i in range(self.rl_config['batch_size']):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.rl_config['gamma'] * np.max(next_q_values[i])
        
        # Train the Q-network
        self.q_network.train_on_batch(states, targets)
        
        # Decay epsilon
        if self.rl_config['epsilon'] > self.rl_config['epsilon_min']:
            self.rl_config['epsilon'] *= self.rl_config['epsilon_decay']
    
    def update_target_network(self):
        """Update the target network with Q-network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
        logger.info("Updated target network weights")
    
    def calculate_reward(self, symbol: str, action: int, 
                     prev_price: float, current_price: float,
                     position: int = 0, trade_cost: float = 0.001) -> float:
        """
        Calculate reward for reinforcement learning.
        
        Args:
            symbol (str): Symbol being traded
            action (int): Action taken (0: hold, 1: buy, 2: sell)
            prev_price (float): Previous price
            current_price (float): Current price
            position (int): Current position (0: none, 1: long, -1: short)
            trade_cost (float): Trading cost as fraction of price
            
        Returns:
            float: Calculated reward
        """
        # Calculate price change
        price_change = (current_price - prev_price) / prev_price
        
        # Initialize reward
        reward = 0.0
        
        # Calculate reward based on action and position
        if action == 0:  # Hold
            if position == 1:  # Long position
                reward = price_change
            elif position == -1:  # Short position
                reward = -price_change
            else:  # No position
                reward = 0
        
        elif action == 1:  # Buy
            if position == 0:  # No position -> Long
                reward = price_change - trade_cost
            elif position == -1:  # Short -> No position
                reward = -price_change - trade_cost
            else:  # Already long, no change
                reward = price_change - trade_cost  # Penalize for unnecessary trade
        
        elif action == 2:  # Sell
            if position == 0:  # No position -> Short
                reward = -price_change - trade_cost
            elif position == 1:  # Long -> No position
                reward = price_change - trade_cost
            else:  # Already short, no change
                reward = -price_change - trade_cost  # Penalize for unnecessary trade
        
        # Apply additional rewards/penalties based on market conditions
        # For example, reward for trading with trend, penalize for trading against it
        try:
            if symbol in self.current_state:
                state = self.current_state[symbol]
                trend_strength = state[4]  # Assuming trend strength is at index 4
                
                # Reward for trading with trend
                if trend_strength > 0.7:  # Strong bullish trend
                    if action == 1:  # Buy
                        reward += 0.001
                    elif action == 2 and position == 1:  # Sell long position
                        reward -= 0.001
                elif trend_strength < 0.3:  # Strong bearish trend
                    if action == 2:  # Sell
                        reward += 0.001
                    elif action == 1 and position == -1:  # Buy short position
                        reward -= 0.001
        except (IndexError, KeyError) as e:
            logger.warning(f"Error accessing trend strength in state: {e}")
        
        return reward

    def process_rl_trading(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """
        Process trading using reinforcement learning.
        
        Args:
            symbol (str): Symbol to trade
            data (pd.DataFrame): Market data (optional, will fetch if None)
            
        Returns:
            Dict: Trading results
        """
        try:
            # Fetch data if not provided
            if data is None or data.empty:
                data = self.get_market_data(symbol)
                
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return {'status': 'error', 'message': 'No data available'}
            
            # Extract current state
            current_state = self.extract_state(symbol, data)
            
            # Get current position
            position = 0
            if symbol in self.active_trades:
                position = 1 if self.active_trades[symbol]['action'] == 'buy' else -1
            
            # Get current and previous prices
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
            
            # Check if we have a previous state for this symbol
            have_previous_state = symbol in self.current_state
            
            # If we have a previous state, we can calculate reward and remember experience
            if have_previous_state:
                previous_state = self.current_state[symbol]
                previous_action = self.active_trades.get(symbol, {}).get('rl_action', 0)
                
                # Calculate reward
                reward = self.calculate_reward(
                    symbol=symbol,
                    action=previous_action,
                    prev_price=prev_price,
                    current_price=current_price,
                    position=position
                )
                
                # Remember experience
                done = False  # Not applicable in continuous trading
                self.remember(previous_state, previous_action, reward, current_state, done)
                
                # Train model periodically
                self.step_count += 1
                if self.step_count % self.rl_config['train_interval'] == 0:
                    self.replay()
                
                # Update target network periodically
                if self.step_count % self.rl_config['target_update_interval'] == 0:
                    self.update_target_network()
            
            # Save current state
            self.current_state[symbol] = current_state
            
            # Get action from model
            # In production, we may want to disable exploration
            use_exploration = self.config.get('rl_exploration', True)
            action = self.get_action(current_state, explore=use_exploration)
            
            # Convert RL action to trading action
            if action == 0:  # Hold
                trading_action = 'hold'
            elif action == 1:  # Buy
                trading_action = 'buy'
            else:  # Sell
                trading_action = 'sell'
            
            # Record the RL action for future reward calculation
            if symbol in self.active_trades:
                self.active_trades[symbol]['rl_action'] = action
            
            # Execute trading action if not holding
            if trading_action != 'hold':
                # Generate traditional signals for comparison
                traditional_signals = self.generate_trading_signals(symbol, data)
                
                # Combine RL decision with traditional signals
                # Only take action if RL and traditional signals align, or if RL has high confidence
                combined_action = trading_action
                
                # Calculate position size based on confidence
                q_values = self.q_network.predict(np.expand_dims(current_state, axis=0), verbose=0)[0]
                action_confidence = q_values[action] - np.mean(q_values)  # Relative confidence
                
                # Scale confidence to [0, 1]
                confidence = min(1.0, max(0.0, action_confidence / 5.0))
                
                # Only execute if confidence is high or traditional signals agree
                should_execute = (confidence > 0.7) or (traditional_signals['signal'] == trading_action)
                
                if should_execute:
                    # Calculate position size based on confidence and traditional signals
                    signal_strength = traditional_signals.get('strength', 50)
                    combined_strength = (confidence * 100 + signal_strength) / 2
                    
                    position_details = self.calculate_position_size(symbol, combined_strength)
                    
                    # Check if this would be a day trade and if we can make it
                    if self.check_day_trade_limit():
                        # Execute the trade
                        logger.info(f"RL executing {trading_action} for {symbol} with confidence {confidence:.2f}")
                        execution_result = self.execute_trade(symbol, trading_action, position_details)
                        
                        return {
                            'status': 'success',
                            'action': trading_action,
                            'confidence': confidence,
                            'position_size': position_details.get('position_size', 0),
                            'execution_result': execution_result
                        }
                    else:
                        logger.info(f"Skipping {trading_action} for {symbol} due to PDT limit")
                        return {
                            'status': 'skipped',
                            'action': trading_action,
                            'reason': 'PDT limit'
                        }
                else:
                    logger.info(f"RL suggested {trading_action} for {symbol} but confidence too low: {confidence:.2f}")
                    return {
                        'status': 'skipped',
                        'action': trading_action,
                        'confidence': confidence,
                        'reason': 'low confidence',
                        'traditional_signal': traditional_signals['signal']
                    }
            else:
                # Holding, no action needed
                return {
                    'status': 'hold',
                    'action': 'hold',
                    'position': position
                }
                
        except Exception as e:
            logger.error(f"Error in RL trading for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}

    def train_rl_model(self, symbols: list = None, episodes: int = 100, 
                    data_lookback_days: int = 90) -> Dict:
        """
        Train the RL model on historical data.
        
        Args:
            symbols (list): List of symbols to train on
            episodes (int): Number of episodes to train
            data_lookback_days (int): Days of historical data to use
            
        Returns:
            Dict: Training results
        """
        try:
            logger.info(f"Starting RL model training with {episodes} episodes")
            
            # Use default symbols if none provided
            if not symbols:
                symbols = self.config.get('data', {}).get('symbols', [])
                if not symbols:
                    return {'status': 'error', 'message': 'No symbols specified for training'}
            
            # Training metrics
            total_rewards = []
            episode_rewards = []
            
            # Main training loop
            for episode in range(episodes):
                logger.info(f"Episode {episode+1}/{episodes}")
                episode_reward = 0
                
                # Shuffle symbols to avoid bias
                random.shuffle(symbols)
                
                for symbol in symbols:
                    # Get historical data
                    data = self.get_market_data(symbol, lookback_periods=60)  # Reduced for training speed
                    
                    if data.empty:
                        continue
                    
                    # Reset symbol state
                    if symbol in self.current_state:
                        del self.current_state[symbol]
                    
                    # Simulate trading through the data
                    position = 0  # No position initially
                    
                    for i in range(10, len(data) - 1):  # Start after warmup period
                        # Extract state from window of data
                        window = data.iloc[:i+1]
                        state = self.extract_state(symbol, window)
                        
                        # Get action from model with exploration
                        action = self.get_action(state, explore=True)
                        
                        # Execute action and calculate reward
                        current_price = data['close'].iloc[i]
                        next_price = data['close'].iloc[i+1]
                        
                        # Update position based on action
                        next_position = position
                        if action == 1:  # Buy
                            next_position = 1
                        elif action == 2:  # Sell
                            next_position = -1
                        
                        # Calculate reward
                        reward = self.calculate_reward(
                            symbol=symbol,
                            action=action,
                            prev_price=current_price,
                            current_price=next_price,
                            position=position
                        )
                        
                        # Extract next state
                        next_window = data.iloc[:i+2]
                        next_state = self.extract_state(symbol, next_window)
                        
                        # Store experience
                        done = (i == len(data) - 2)  # End of data
                        self.remember(state, action, reward, next_state, done)
                        
                        # Update running reward
                        episode_reward += reward
                        
                        # Update position for next iteration
                        position = next_position
                        
                        # Train after each step
                        if len(self.memory) > self.rl_config['batch_size']:
                            self.replay()
                    
                    # Update target network after each symbol
                    self.update_target_network()
                
                # Track episode reward
                episode_rewards.append(episode_reward)
                total_rewards.append(sum(episode_rewards))
                
                # Decay epsilon
                if self.rl_config['epsilon'] > self.rl_config['epsilon_min']:
                    self.rl_config['epsilon'] *= self.rl_config['epsilon_decay']
                
                # Log progress
                logger.info(f"Episode {episode+1} completed with reward: {episode_reward:.4f}, "
                        f"epsilon: {self.rl_config['epsilon']:.4f}")
            
            # Save the trained model
            self.save_rl_model()
            
            # Return training metrics
            return {
                'status': 'success',
                'episodes': episodes,
                'final_reward': episode_rewards[-1],
                'avg_reward': sum(episode_rewards) / len(episode_rewards),
                'epsilon': self.rl_config['epsilon']
            }
            
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            return {'status': 'error', 'message': str(e)}

    def run_trading_cycle_rl(self, symbols: list) -> Dict:
        """
        Run a complete trading cycle using RL model for decision making.
        
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
            logger.info(f"Starting RL trading cycle for {len(symbols)} symbols")
            
            # Ensure connection to broker
            if not self.ibkr.connected:
                if not self.connect_to_broker():
                    return {'status': 'error', 'message': 'Not connected to broker'}
            
            # First manage existing positions using standard methods
            position_results = self.manage_open_positions()
            cycle_results['positions_managed'] = position_results.get('monitored', 0)
            cycle_results['positions_updated'] = position_results.get('updated', 0)
            cycle_results['positions_closed'] = position_results.get('closed', 0)
            
            # Process each symbol with RL
            for symbol in symbols:
                cycle_results['processed'] += 1
                
                try:
                    # Process with RL
                    rl_result = self.process_rl_trading(symbol)
                    cycle_results['signals_generated'] += 1
                    
                    # Track successful trades
                    if rl_result.get('status') == 'success' and rl_result.get('action') in ['buy', 'sell']:
                        cycle_results['trades_executed'] += 1
                        
                    cycle_results['details'].append({
                        'symbol': symbol,
                        'status': rl_result.get('status'),
                        'action': rl_result.get('action'),
                        'confidence': rl_result.get('confidence', 0),
                        'reason': rl_result.get('reason', '')
                    })
                    
                except Exception as e:
                    cycle_results['errors'] += 1
                    cycle_results['details'].append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
                    
                    logger.error(f"Error processing {symbol} in RL trading cycle: {e}")
            
            logger.info(f"Completed RL trading cycle: processed {cycle_results['processed']} symbols, "
                    f"executed {cycle_results['trades_executed']} trades")
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in RL trading cycle: {e}")
            return {'status': 'error', 'message': str(e)}
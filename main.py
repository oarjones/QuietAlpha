"""
Main application entry point with LSTM integration

This module updates the main entry point to support LSTM-enhanced portfolio management.
It incorporates the LSTM portfolio manager when enabled in configuration.
"""

import os
import logging
import argparse
import time
import datetime
import json
from typing import Dict, List

# Import original components
from portfolio_manager.base import PortfolioManager
from portfolio_manager.xgboost_manager import XGBoostPortfolioManager
from trading_manager.base import TradingManager

# Import LSTM-enhanced components
from portfolio_manager.lstm_enhanced_manager import LSTMEnhancedPortfolioManager
from lstm.integration import get_lstm_service

# Import utilities
from ibkr_api.interface import IBKRInterface
from utils.data_utils import load_config, merge_configs, save_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'quietalpha.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class QuietAlphaTradingBot:
    """
    Main trading bot class with LSTM integration support.
    """
    
    def __init__(self, config_path: str = None, lstm_config_path: str = None):
        """
        Initialize trading bot with optional LSTM configuration.
        
        Args:
            config_path (str): Path to main configuration file
            lstm_config_path (str): Path to LSTM-specific configuration file
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration files
        self.config_path = config_path or os.path.join('config', 'base_config.json')
        self.config = load_config(self.config_path)
        
        # Load and merge LSTM config if provided
        if lstm_config_path:
            lstm_config = load_config(lstm_config_path)
            self.config = merge_configs(self.config, lstm_config)
        
        # Initialize IBKR interface
        self.ibkr = self._init_ibkr_interface()
        
        # Check if LSTM is enabled
        self.lstm_enabled = self.config.get('portfolio_manager', {}).get('lstm', {}).get('enabled', False)
        
        # Initialize LSTM service if enabled
        if self.lstm_enabled:
            logger.info("LSTM integration enabled, initializing LSTM service")
            self.lstm_service = get_lstm_service(ibkr_interface=self.ibkr)
        else:
            self.lstm_service = None
        
        # Initialize Portfolio Manager
        self.portfolio_manager = self._init_portfolio_manager()
        
        # Initialize Trading Manager
        self.trading_manager = self._init_trading_manager()
        
        # Trading state
        self.is_running = False
        self.last_portfolio_update = None
        self.last_trading_cycle = None
        self.current_portfolio = {}
        
        logger.info("QuietAlpha Trading Bot initialized")
        if self.lstm_enabled:
            logger.info("LSTM integration active")
    
    def _init_ibkr_interface(self) -> IBKRInterface:
        """Initialize IBKR interface."""
        ibkr_config = self.config.get('ibkr', {})
        host = ibkr_config.get('host', '127.0.0.1')
        port = ibkr_config.get('port', 7497)
        client_id = ibkr_config.get('client_id', 1)
        
        ibkr = IBKRInterface(host=host, port=port, client_id=client_id)
        logger.info(f"IBKR interface initialized with host={host}, port={port}")
        
        return ibkr
    
    def _init_portfolio_manager(self) -> PortfolioManager:
        """Initialize Portfolio Manager with LSTM support if enabled."""
        pm_type = self.config.get('portfolio_manager_type', 'xgboost')
        
        # Handle LSTM-enhanced case
        if self.lstm_enabled:
            logger.info("Initializing LSTM-Enhanced Portfolio Manager")
            if pm_type == 'xgboost':
                logger.info("Using XGBoost as base for LSTM-enhanced Portfolio Manager")
                # In this case, we still use the LSTM enhanced manager
                # The XGBoost functionality is already included via inheritance
                return LSTMEnhancedPortfolioManager(config_path=self.config_path, ibkr_interface=self.ibkr)
            else:
                logger.info("Using base implementation for LSTM-enhanced Portfolio Manager")
                return LSTMEnhancedPortfolioManager(config_path=self.config_path, ibkr_interface=self.ibkr)
        
        # Regular (non-LSTM) portfolio managers
        if pm_type == 'xgboost':
            logger.info("Initializing XGBoost Portfolio Manager")
            return XGBoostPortfolioManager(config_path=self.config_path, ibkr_interface=self.ibkr)
        else:
            logger.info("Initializing Base Portfolio Manager")
            return PortfolioManager(config_path=self.config_path, ibkr_interface=self.ibkr)
    
    def _init_trading_manager(self) -> TradingManager:
        """Initialize Trading Manager."""
        tm_type = self.config.get('trading_manager_type', 'base')
        
        if tm_type == 'rl':
            logger.info("RL Trading Manager not implemented")
        else:
            logger.info("Initializing Base Trading Manager")
            return TradingManager(config_path=self.config_path, ibkr_interface=self.ibkr)
    
    def connect(self) -> bool:
        """
        Connect to IBKR.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        success = self.ibkr.connect()
        if success:
            logger.info("Connected to IBKR")
        else:
            logger.error("Failed to connect to IBKR")
        
        return success
    
    def disconnect(self):
        """Disconnect from IBKR."""
        self.ibkr.disconnect()
        logger.info("Disconnected from IBKR")
    
    def update_portfolio(self) -> Dict:
        """
        Update portfolio based on market conditions.
        
        Returns:
            Dict: Update results
        """
        try:
            logger.info("Updating portfolio")
            
            # Use Portfolio Manager to update portfolio
            update_result = self.portfolio_manager.update_portfolio()
            
            if update_result.get('status') == 'success':
                self.current_portfolio = update_result.get('update_plan', {}).get('allocations', {})
                self.last_portfolio_update = datetime.datetime.now()
                
                # Log portfolio update
                logger.info(f"Portfolio updated with {len(self.current_portfolio)} symbols")
                logger.info(f"Symbols: {list(self.current_portfolio.keys())}")
                
                # If LSTM is enabled, log additional info
                if self.lstm_enabled and 'lstm_training' in update_result:
                    lstm_training = update_result['lstm_training']
                    logger.info(f"LSTM training requested for {lstm_training.get('requested', 0)} symbols")
            else:
                logger.warning(f"Portfolio update failed: {update_result}")
            
            return update_result
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_trading_cycle(self) -> Dict:
        """
        Run a complete trading cycle.
        
        Returns:
            Dict: Trading cycle results
        """
        try:
            logger.info("Running trading cycle")
            
            # Check if we have current portfolio
            if not self.current_portfolio:
                logger.warning("No current portfolio. Updating portfolio first.")
                self.update_portfolio()
            
            # Get symbols from portfolio
            symbols = list(self.current_portfolio.keys())
            
            if not symbols:
                logger.warning("No symbols in portfolio to trade")
                return {'status': 'warning', 'message': 'No symbols in portfolio'}
            
            # Use Trading Manager to run trading cycle
            cycle_result = self.trading_manager.run_trading_cycle(symbols)
            
            self.last_trading_cycle = datetime.datetime.now()
            
            # Log trading cycle results
            logger.info(f"Trading cycle completed: processed {cycle_result.get('processed', 0)} symbols, "
                      f"executed {cycle_result.get('trades_executed', 0)} trades")
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run(self, interval: int = 3600):
        """
        Run the trading bot continuously.
        
        Args:
            interval (int): Interval between trading cycles in seconds
        """
        try:
            logger.info(f"Starting QuietAlpha Trading Bot with interval {interval} seconds")
            
            # Connect to IBKR
            if not self.connect():
                logger.error("Failed to connect to IBKR. Exiting.")
                return
            
            # Set running flag
            self.is_running = True
            
            # Initial portfolio update
            self.update_portfolio()
            
            # Main loop
            while self.is_running:
                try:
                    # Check connection
                    if not self.ibkr.connected:
                        logger.warning("Lost connection to IBKR. Reconnecting...")
                        self.connect()
                    
                    # Run trading cycle
                    self.run_trading_cycle()
                    
                    # Check if we need to update portfolio
                    # Update every 4 hours by default
                    portfolio_update_interval = self.config.get('portfolio_update_interval', 14400)
                    if (self.last_portfolio_update is None) or (
                        datetime.datetime.now() - self.last_portfolio_update).total_seconds() > portfolio_update_interval:
                        self.update_portfolio()
                    
                    # Sleep until next cycle
                    logger.info(f"Sleeping for {interval} seconds")
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("Trading bot interrupted by user")
                    self.is_running = False
                    break
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    # Sleep for a short time before retrying
                    time.sleep(60)
            
            # Disconnect from IBKR
            self.disconnect()
            
        except Exception as e:
            logger.error(f"Fatal error in trading bot: {e}")
            self.disconnect()
    
    def train_models(self) -> Dict:
        """
        Train Portfolio Manager and Trading Manager models.
        
        Returns:
            Dict: Training results
        """
        results = {}
        
        try:
            logger.info("Starting model training")
            
            # Train Portfolio Manager
            if isinstance(self.portfolio_manager, XGBoostPortfolioManager):
                logger.info("Training XGBoost Portfolio Manager")
                pm_result = self.portfolio_manager.train_xgboost_models()
            else:
                logger.info("Training Portfolio Manager")
                pm_result = self.portfolio_manager.train_models()
            
            results['portfolio_manager'] = pm_result
            
            # Train Trading Manager if it's RL-based (not implemented)
            
            
            logger.info("Model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the trading bot.
        
        Returns:
            Dict: Performance metrics
        """
        try:
            # Get Trading Manager metrics
            trading_metrics = self.trading_manager.get_performance_metrics()
            
            # Get Portfolio Manager metrics
            portfolio_stats = self.portfolio_manager.get_portfolio_stats()
            
            # Combine metrics
            combined_metrics = {
                'trading': trading_metrics,
                'portfolio': portfolio_stats,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def save_performance_report(self, file_path: str = None) -> bool:
        """
        Save performance report to file.
        
        Args:
            file_path (str): Path to save report
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Get metrics
            metrics = self.get_performance_metrics()
            
            # Default file path
            if file_path is None:
                reports_dir = os.path.join('reports')
                os.makedirs(reports_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = os.path.join(reports_dir, f'performance_report_{timestamp}.json')
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Performance report saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return False

def main():
    """Main entry point for running the trading bot with LSTM support."""
    parser = argparse.ArgumentParser(description='QuietAlpha Trading Bot with LSTM')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--lstm-config', type=str, help='Path to LSTM configuration file')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--interval', type=int, default=3600, help='Trading cycle interval in seconds')
    parser.add_argument('--backtest', action='store_true', help='Run backtest instead of live trading')
    parser.add_argument('--enable-lstm', action='store_true', help='Enable LSTM integration')
    
    args = parser.parse_args()
    
    # If LSTM config is not specified but enable-lstm flag is set, use default
    if args.enable_lstm and not args.lstm_config:
        args.lstm_config = os.path.join('config', 'lstm_config.json')
    
    # Create bot instance
    bot = QuietAlphaTradingBot(config_path=args.config, lstm_config_path=args.lstm_config)
    
    # Train models if requested
    if args.train:
        logger.info("Training models...")
        bot.train_models()
        logger.info("Training completed")
    
    # Run backtest if requested
    if args.backtest:
        logger.info("Backtesting not implemented yet")
        return
    
    # Run the bot
    bot.run(interval=args.interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        logger.info("QuietAlpha Trading Bot shutting down")
"""
Trading Manager module initialization

This module provides classes for executing trades based on signals
from technical analysis and machine learning models.
"""

from trading_manager.base import TradingManager
from trading_manager.rl_trading_stable import RLTradingStableManager

__all__ = ['TradingManager', 'RLTradingStableManager']
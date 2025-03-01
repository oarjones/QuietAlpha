"""
Trading Signal Processing

This module provides functions for processing trading signals,
managing open positions, and executing trading cycles.
"""

import logging
import os
import numpy as np
import pandas as pd
import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

def process_trading_signals(self, symbol: str) -> Dict:
    """
    Process trading signals and execute trades if appropriate.
    
    Args:
        symbol (str): Symbol to process
        
    Returns:
        Dict: Processing results
    """
    try:
        # Generate trading signals
        signals = self.generate_trading_signals(symbol)
        
        if signals.get('signal') == 'error':
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
                # Strong opposite signal, close position
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
            if signals['signal'] in ['buy', 'sell'] and signals['strength'] > 70:
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
                
                # Check if this would be a day trade and if we can make it
                if not self.check_day_trade_limit():
                    # If we can't make a day trade, consider alternatives
                    current_time = self.ibkr.ib.reqCurrentTime() if self.ibkr.connected else None
                    
                    # If it's late in the trading day, skip the trade to avoid PDT violations
                    if current_time and current_time.hour >= 15:  # After 3 PM
                        logger.info(f"Skipping {symbol} trade to avoid PDT violation late in the day")
                        return {
                            'status': 'skipped',
                            'message': 'Trade skipped to avoid PDT violation (late in day)',
                            'signal': signals['signal'],
                            'strength': signals['strength']
                        }
                    
                    # Otherwise, consider a swing trade instead of a day trade
                    logger.info(f"Converting to swing trade for {symbol} to avoid PDT violation")
                    # Adjust position size for swing trade (typically smaller)
                    position_details['position_size'] = max(1, position_details['position_size'] // 2)
                    
                    # Adjust stop loss and take profit for swing trade (wider)
                    atr = position_details.get('atr', 0)
                    if atr > 0:
                        if signals['signal'] == 'buy':
                            position_details['stop_loss_price'] = position_details['current_price'] - (atr * 3.0)
                            position_details['take_profit_price'] = position_details['current_price'] + (atr * 4.5)
                        else:  # sell
                            position_details['stop_loss_price'] = position_details['current_price'] + (atr * 3.0)
                            position_details['take_profit_price'] = position_details['current_price'] - (atr * 4.5)
                
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

def manage_open_positions(self) -> Dict:
    """
    Manage all open positions, updating stops and evaluating exits.
    
    Returns:
        Dict: Results of position management
    """
    results = {
        'monitored': 0,
        'updated': 0,
        'closed': 0,
        'errors': 0,
        'details': []
    }
    
    try:
        # Check connection to broker
        if not self.ibkr.connected:
            if not self.connect_to_broker():
                return {'status': 'error', 'message': 'Not connected to broker'}
        
        # Get all current positions
        positions = self.ibkr.get_positions()
        
        # Process each position
        for position in positions:
            symbol = position['symbol']
            position_size = position['position']
            
            if position_size == 0:
                continue
            
            results['monitored'] += 1
            
            try:
                # Get current market data
                data = self.get_market_data(symbol)
                
                if data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Generate current signals
                signals = self.generate_trading_signals(symbol)
                
                # Determine if we should exit the position
                should_exit = False
                exit_reason = ""
                
                # 1. Check for exit signals
                is_long = position_size > 0
                if (is_long and signals['signal'] == 'sell' and signals['strength'] > 60) or \
                   (not is_long and signals['signal'] == 'buy' and signals['strength'] > 60):
                    should_exit = True
                    exit_reason = "opposite_signal"
                
                # 2. Check technical exit conditions
                current_price = data['close'].iloc[-1]
                
                # For long positions
                if is_long:
                    # Trail stop on significant moves
                    if symbol in self.active_trades:
                        trade = self.active_trades[symbol]
                        entry_price = trade.get('entry_price', current_price)
                        stop_price = trade.get('stop_loss_price', 0)
                        
                        # If price moved up significantly, update stop loss
                        price_change_pct = (current_price - entry_price) / entry_price
                        if price_change_pct > 0.02:  # 2% move
                            # Calculate new trailing stop
                            atr = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else current_price * 0.02
                            new_stop = max(stop_price, current_price - (atr * 2.0))
                            
                            if new_stop > stop_price:
                                # Update stop loss in active trades record
                                self.active_trades[symbol]['stop_loss_price'] = new_stop
                                
                                # Update actual stop order (in a real system)
                                # For simplicity, we're just updating our internal record here
                                # In a production system, you would modify the actual order
                                
                                results['updated'] += 1
                                results['details'].append({
                                    'symbol': symbol,
                                    'action': 'update_stop',
                                    'old_stop': stop_price,
                                    'new_stop': new_stop
                                })
                                
                                logger.info(f"Updated trailing stop for {symbol} from {stop_price} to {new_stop}")
                
                # 3. Check time-based exits for swing trades
                if symbol in self.active_trades:
                    trade = self.active_trades[symbol]
                    if not trade.get('is_day_trade', False):
                        # For swing trades, check if we've held for target duration
                        import datetime
                        entry_time = datetime.datetime.fromisoformat(trade['entry_time'])
                        current_time = datetime.datetime.now()
                        
                        # Exit swing trades after a certain number of days if profitable
                        days_held = (current_time - entry_time).days
                        
                        # Calculate current P&L
                        if is_long:
                            pl_pct = (current_price - trade['entry_price']) / trade['entry_price']
                        else:
                            pl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                        
                        # Exit if held for 5+ days and profitable
                        if days_held >= 5 and pl_pct > 0:
                            should_exit = True
                            exit_reason = "time_based_profit_take"
                        
                        # Exit if held for 10+ days regardless
                        elif days_held >= 10:
                            should_exit = True
                            exit_reason = "max_hold_time"
                
                # Execute exit if conditions met
                if should_exit:
                    exit_result = self.close_position(symbol)
                    
                    if exit_result.get('status') == 'success':
                        results['closed'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'action': 'close_position',
                            'reason': exit_reason,
                            'pl_amount': exit_result.get('pl_amount', 0)
                        })
                        
                        logger.info(f"Closed position for {symbol} due to {exit_reason}")
                    else:
                        results['errors'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'action': 'close_position_failed',
                            'reason': exit_reason,
                            'error': exit_result.get('message', 'Unknown error')
                        })
                        
                        logger.error(f"Failed to close position for {symbol}: {exit_result}")
            
            except Exception as e:
                results['errors'] += 1
                results['details'].append({
                    'symbol': symbol,
                    'action': 'error',
                    'error': str(e)
                })
                
                logger.error(f"Error managing position for {symbol}: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error managing open positions: {e}")
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
                if signal_result.get('status') == 'success' and 'execute' in str(signal_result.get('message', '')):
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
    Calculate performance metrics for the trading strategy.
    
    Returns:
        Dict: Performance metrics
    """
    try:
        # Calculate metrics from trade history
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'average_profit': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        total_trades = len(self.trade_history)
        profitable_trades = [t for t in self.trade_history if t.get('pl_amount', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pl_amount', 0) <= 0]
        
        win_count = len(profitable_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('pl_amount', 0) for t in profitable_trades)
        total_loss = sum(abs(t.get('pl_amount', 0)) for t in losing_trades)
        
        average_profit = total_profit / win_count if win_count > 0 else 0
        average_loss = total_loss / loss_count if loss_count > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        total_pnl = total_profit - total_loss
        
        # Calculate Sharpe ratio if we have enough data
        if len(self.trade_history) >= 30:
            daily_returns = []
            current_day = None
            day_pnl = 0
            
            # Group trades by day
            sorted_trades = sorted(self.trade_history, key=lambda x: x.get('exit_time', ''))
            
            for trade in sorted_trades:
                if not trade.get('exit_time'):
                    continue
                
                exit_time = trade.get('exit_time', '')[:10]  # Get just the date part
                
                if current_day is None:
                    current_day = exit_time
                    day_pnl = trade.get('pl_amount', 0)
                elif current_day == exit_time:
                    day_pnl += trade.get('pl_amount', 0)
                else:
                    daily_returns.append(day_pnl)
                    current_day = exit_time
                    day_pnl = trade.get('pl_amount', 0)
            
            # Add the last day
            if current_day is not None:
                daily_returns.append(day_pnl)
            
            # Calculate Sharpe ratio
            if daily_returns:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns) if len(daily_returns) > 1 else 1
                risk_free_rate = self.config.get('risk_management', {}).get('risk_free_rate', 0.03) / 252  # Daily risk-free rate
                
                sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_pnl = 0
        peak = 0
        drawdown = 0
        max_drawdown = 0
        
        for trade in self.trade_history:
            pnl = trade.get('pl_amount', 0)
            cumulative_pnl += pnl
            
            if cumulative_pnl > peak:
                peak = cumulative_pnl
                drawdown = 0
            else:
                drawdown = peak - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'risk_reward_ratio': average_profit / average_loss if average_loss > 0 else float('inf')
        }
    
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {'status': 'error', 'message': str(e)}
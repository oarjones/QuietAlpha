"""
Interactive Brokers API Interface

Provides a complete interface to interact with Interactive Brokers via ib_insync.
Handles connection, data retrieval, account information, and order execution.
"""

from ib_insync import IB, Stock, Contract, Order, util
import pandas as pd
import numpy as np
import datetime
import logging
import os
import time
from typing import List, Dict, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class IBKRInterface:
    """Interface for Interactive Brokers API."""
    
    def __init__(self, host='127.0.0.1', port=4002, client_id=1, is_paper=True):
        """
        Initialize IBKR Interface.
        
        Args:
            host (str): TWS/Gateway host address
            port (int): TWS/Gateway port (7496/7497 for TWS, 4001/4002 for Gateway)
            client_id (int): Unique client ID
            is_paper (bool): Whether using paper trading account
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.is_paper = is_paper
        self.ib = IB()
        self.connected = False
        self.day_trades_count = 0
        self.max_day_trades = 3  # Default PDT limit
        
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = self.ib.isConnected()
            if self.connected:
                logger.info(f"Connected to IBKR at {self.host}:{self.port}")
                # Initialize day trades count
                self._update_day_trades_count()
            else:
                logger.error("Failed to connect to IBKR")
            return self.connected
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers TWS/Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def _check_connection(self) -> bool:
        """
        Check if connected to IBKR, attempt reconnect if not.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.connected or not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Attempting to reconnect...")
            return self.connect()
        return True
    
    def get_contract(self, symbol: str, sec_type: str = 'STK', 
                    exchange: str = 'SMART', currency: str = 'USD',
                    primary_exchange: str = None) -> Contract:
        """
        Create a contract object for the specified symbol.
        
        Args:
            symbol (str): Ticker symbol
            sec_type (str): Security type (STK, FUT, OPT, etc.)
            exchange (str): Exchange to route order to
            currency (str): Currency of the security
            primary_exchange (str): Primary exchange where security is listed
            
        Returns:
            Contract: IBKR contract object
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        if primary_exchange:
            contract.primaryExchange = primary_exchange
            
        return contract
    
    def get_historical_data(self, symbol: str, duration: str = "1 Y", 
                           bar_size: str = "1 hour", what_to_show: str = "TRADES",
                           use_rth: bool = True, end_datetime: str = '',
                           format_date: int = 1) -> pd.DataFrame:
        """
        Retrieve historical market data for a symbol.
        
        Args:
            symbol (str): Ticker symbol
            duration (str): Time duration to retrieve data for (e.g. "1 Y", "6 M", "5 D")
            bar_size (str): Size of each bar ("1 min", "5 mins", "1 hour", "1 day", etc.)
            what_to_show (str): Type of data ("TRADES", "MIDPOINT", "BID", "ASK", etc.)
            use_rth (bool): Use regular trading hours only
            end_datetime (str): End date and time for data retrieval (empty for now)
            format_date (int): Format of returned dates
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        if not self._check_connection():
            return pd.DataFrame()
        
        # Create contract
        contract = self.get_contract(symbol)
        
        # Use current time if end_datetime not specified
        if not end_datetime:
            end_datetime = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
        
        try:
            logger.info(f"Fetching {bar_size} data for {symbol}, duration {duration}")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date
            )
            
            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = util.df(bars)
            
            # Rename 'date' to 'datetime' if needed
            if 'date' in df.columns:
                df.rename(columns={'date': 'datetime'}, inplace=True)
            
            # Ensure datetime column is actual datetime type
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_historical_data(self, symbols: List[str], output_dir: str = None,
                           **kwargs) -> Dict[str, str]:
        """
        Fetch and save historical data for multiple symbols.
        
        Args:
            symbols (list): List of symbols to fetch data for
            output_dir (str): Directory to save data to (default: data/raw)
            **kwargs: Additional arguments to pass to get_historical_data
            
        Returns:
            dict: Dictionary mapping symbols to output file paths
        """
        if output_dir is None:
            output_dir = os.path.join("data", "raw")
            
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, **kwargs)
                
                if not df.empty:
                    output_file = os.path.join(output_dir, f"{symbol}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Data saved to {output_file}")
                    results[symbol] = output_file
                else:
                    logger.warning(f"No data to save for {symbol}")
                    results[symbol] = None
            
            except Exception as e:
                logger.error(f"Error saving data for {symbol}: {e}")
                results[symbol] = None
                
        return results
    
    def get_account_summary(self) -> Dict[str, float]:
        """
        Get account summary information.
        
        Returns:
            dict: Account summary with key metrics
        """
        if not self._check_connection():
            return {}
            
        try:
            # Request account summary
            account_values = self.ib.accountSummary()
            
            # Convert to dictionary
            summary = {}
            for val in account_values:
                # Try to convert to float if possible
                try:
                    summary[val.tag] = float(val.value)
                except ValueError:
                    summary[val.tag] = val.value
            
            # Add some useful derived metrics
            if 'NetLiquidation' in summary and 'GrossPositionValue' in summary:
                summary['AvailableFunds'] = summary['NetLiquidation'] - summary['GrossPositionValue']
            
            logger.info(f"Retrieved account summary with {len(summary)} values")
            return summary
            
        except Exception as e:
            logger.error(f"Error fetching account summary: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """
        Get current portfolio positions.
        
        Returns:
            list: List of positions with details
        """
        if not self._check_connection():
            return []
            
        try:
            # Request portfolio items
            portfolio = self.ib.portfolio()
            
            # Convert to list of dictionaries
            positions = []
            for item in portfolio:
                pos = {
                    'symbol': item.contract.symbol,
                    'secType': item.contract.secType,
                    'exchange': item.contract.exchange,
                    'currency': item.contract.currency,
                    'position': item.position,
                    'marketPrice': item.marketPrice,
                    'marketValue': item.marketValue,
                    'averageCost': item.averageCost,
                    'unrealizedPNL': item.unrealizedPNL,
                    'realizedPNL': item.realizedPNL,
                    'account': item.account
                }
                positions.append(pos)
            
            logger.info(f"Retrieved {len(positions)} positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def _update_day_trades_count(self):
        """Update the count of day trades used this week."""
        if not self._check_connection():
            return
            
        try:
            # Get account values
            account_values = self.ib.accountValues()
            
            # Find day trades remaining
            for val in account_values:
                if val.tag == 'DayTradesRemaining' and val.currency == 'DayTradesRemaining':
                    try:
                        # This usually returns how many are left before hitting PDT
                        remaining = float(val.value)
                        self.day_trades_count = self.max_day_trades - remaining
                        logger.info(f"Day trades used this week: {self.day_trades_count}")
                        return
                    except ValueError:
                        logger.error(f"Could not parse day trades remaining: {val.value}")
            
            logger.warning("Could not find day trades remaining in account values")
            
        except Exception as e:
            logger.error(f"Error updating day trades count: {e}")
    
    def can_day_trade(self) -> bool:
        """
        Check if a day trade can be made without violating PDT rule.
        
        Returns:
            bool: True if day trade is allowed, False otherwise
        """
        # Update the day trades count first
        self._update_day_trades_count()
        
        # For accounts > $25k, PDT doesn't apply
        account_summary = self.get_account_summary()
        if account_summary.get('NetLiquidation', 0) >= 25000:
            return True
            
        # Otherwise, check if we've used all day trades
        return self.day_trades_count < self.max_day_trades
    
    def place_market_order(self, symbol: str, action: str, quantity: float) -> Dict:
        """
        Place a market order.
        
        Args:
            symbol (str): Symbol to trade
            action (str): Action to take ('BUY' or 'SELL')
            quantity (float): Quantity to trade
            
        Returns:
            dict: Order details
        """
        if not self._check_connection():
            return {"status": "error", "message": "Not connected to IBKR"}
            
        # Validate input
        if action not in ['BUY', 'SELL']:
            return {"status": "error", "message": f"Invalid action: {action}"}
            
        if quantity <= 0:
            return {"status": "error", "message": f"Invalid quantity: {quantity}"}
            
        try:
            # Create contract
            contract = self.get_contract(symbol)
            
            # Create order
            order = Order()
            order.action = action
            order.orderType = 'MKT'
            order.totalQuantity = quantity
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait briefly for order to be processed
            self.ib.sleep(1)
            
            # Get order status
            order_status = {
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avgFillPrice": trade.orderStatus.avgFillPrice,
                "lastFillPrice": trade.orderStatus.lastFillPrice,
                "whyHeld": trade.orderStatus.whyHeld,
                "order_id": trade.order.orderId
            }
            
            logger.info(f"Placed {action} order for {quantity} {symbol}: {order_status}")
            
            # If this is a day trade, update counter
            # This is simplified and would need to be properly implemented
            # to track actual day trades based on positions
            if action == 'SELL':
                positions = self.get_positions()
                for pos in positions:
                    if pos['symbol'] == symbol and pos['position'] > 0:
                        # If we sold a position we bought today, it's a day trade
                        self.day_trades_count += 1
                        break
            
            return {
                "status": "submitted",
                "order_status": order_status,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": "MKT"
            }
            
        except Exception as e:
            logger.error(f"Error placing market order for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def place_limit_order(self, symbol: str, action: str, quantity: float,
                         limit_price: float) -> Dict:
        """
        Place a limit order.
        
        Args:
            symbol (str): Symbol to trade
            action (str): Action to take ('BUY' or 'SELL')
            quantity (float): Quantity to trade
            limit_price (float): Limit price
            
        Returns:
            dict: Order details
        """
        if not self._check_connection():
            return {"status": "error", "message": "Not connected to IBKR"}
            
        # Validate input
        if action not in ['BUY', 'SELL']:
            return {"status": "error", "message": f"Invalid action: {action}"}
            
        if quantity <= 0:
            return {"status": "error", "message": f"Invalid quantity: {quantity}"}
            
        try:
            # Create contract
            contract = self.get_contract(symbol)
            
            # Create order
            order = Order()
            order.action = action
            order.orderType = 'LMT'
            order.totalQuantity = quantity
            order.lmtPrice = limit_price
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait briefly for order to be processed
            self.ib.sleep(1)
            
            # Get order status
            order_status = {
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avgFillPrice": trade.orderStatus.avgFillPrice,
                "lastFillPrice": trade.orderStatus.lastFillPrice,
                "whyHeld": trade.orderStatus.whyHeld,
                "order_id": trade.order.orderId
            }
            
            logger.info(f"Placed {action} limit order for {quantity} {symbol} at {limit_price}: {order_status}")
            
            return {
                "status": "submitted",
                "order_status": order_status,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "limit_price": limit_price,
                "order_type": "LMT"
            }
            
        except Exception as e:
            logger.error(f"Error placing limit order for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def place_bracket_order(self, symbol: str, action: str, quantity: float,
                          entry_price: float, take_profit: float, 
                          stop_loss: float, as_market: bool = False) -> Dict:
        """
        Place a bracket order (entry + take profit + stop loss).
        
        Args:
            symbol (str): Symbol to trade
            action (str): Action to take ('BUY' or 'SELL')
            quantity (float): Quantity to trade
            entry_price (float): Entry price (limit)
            take_profit (float): Take profit price
            stop_loss (float): Stop loss price
            as_market (bool): Whether to use market order for entry
            
        Returns:
            dict: Order details
        """
        if not self._check_connection():
            return {"status": "error", "message": "Not connected to IBKR"}
            
        # Validate input
        if action not in ['BUY', 'SELL']:
            return {"status": "error", "message": f"Invalid action: {action}"}
            
        if quantity <= 0:
            return {"status": "error", "message": f"Invalid quantity: {quantity}"}
            
        try:
            # Create contract
            contract = self.get_contract(symbol)
            
            # Create parent order
            parent = Order()
            parent.action = action
            parent.totalQuantity = quantity
            
            if as_market:
                parent.orderType = 'MKT'
            else:
                parent.orderType = 'LMT'
                parent.lmtPrice = entry_price
                
            parent.transmit = False
            
            # Create take profit order
            take_profit_order = Order()
            take_profit_order.action = 'SELL' if action == 'BUY' else 'BUY'
            take_profit_order.totalQuantity = quantity
            take_profit_order.orderType = 'LMT'
            take_profit_order.lmtPrice = take_profit
            take_profit_order.parentId = parent.orderId
            take_profit_order.transmit = False
            
            # Create stop loss order
            stop_loss_order = Order()
            stop_loss_order.action = 'SELL' if action == 'BUY' else 'BUY'
            stop_loss_order.totalQuantity = quantity
            stop_loss_order.orderType = 'STP'
            stop_loss_order.auxPrice = stop_loss
            stop_loss_order.parentId = parent.orderId
            stop_loss_order.transmit = True
            
            # Place orders
            parent_trade = self.ib.placeOrder(contract, parent)
            tp_trade = self.ib.placeOrder(contract, take_profit_order)
            sl_trade = self.ib.placeOrder(contract, stop_loss_order)
            
            # Wait briefly for orders to be processed
            self.ib.sleep(1)
            
            logger.info(f"Placed bracket order for {quantity} {symbol}: "
                       f"Entry: {parent.orderType} at {entry_price if not as_market else 'market'}, "
                       f"TP: {take_profit}, SL: {stop_loss}")
            
            return {
                "status": "submitted",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "entry_type": parent.orderType,
                "entry_price": entry_price if not as_market else None,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "parent_order_id": parent_trade.order.orderId,
                "tp_order_id": tp_trade.order.orderId,
                "sl_order_id": sl_trade.order.orderId
            }
            
        except Exception as e:
            logger.error(f"Error placing bracket order for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders.
        
        Returns:
            list: List of open orders
        """
        if not self._check_connection():
            return []
            
        try:
            # Request open orders
            orders = self.ib.openOrders()
            
            # Convert to list of dictionaries
            open_orders = []
            for order in orders:
                order_details = {
                    "order_id": order.orderId,
                    "symbol": order.contract.symbol,
                    "secType": order.contract.secType,
                    "action": order.action,
                    "orderType": order.orderType,
                    "quantity": order.totalQuantity,
                    "status": order.orderStatus.status,
                    "filled": order.orderStatus.filled,
                    "remaining": order.orderStatus.remaining,
                    "lmtPrice": order.lmtPrice if hasattr(order, 'lmtPrice') else None,
                    "auxPrice": order.auxPrice if hasattr(order, 'auxPrice') else None,
                    "parentId": order.parentId if hasattr(order, 'parentId') else None,
                }
                open_orders.append(order_details)
            
            logger.info(f"Retrieved {len(open_orders)} open orders")
            return open_orders
            
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    def cancel_order(self, order_id: int) -> Dict:
        """
        Cancel an order by ID.
        
        Args:
            order_id (int): Order ID to cancel
            
        Returns:
            dict: Cancellation result
        """
        if not self._check_connection():
            return {"status": "error", "message": "Not connected to IBKR"}
            
        try:
            # Find the order
            open_orders = self.ib.openOrders()
            target_order = None
            
            for order in open_orders:
                if order.orderId == order_id:
                    target_order = order
                    break
                    
            if not target_order:
                return {"status": "error", "message": f"Order ID {order_id} not found"}
                
            # Cancel the order
            self.ib.cancelOrder(target_order)
            
            # Wait briefly for cancellation to process
            self.ib.sleep(1)
            
            logger.info(f"Cancelled order {order_id}")
            
            return {
                "status": "cancelled",
                "order_id": order_id
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get real-time market data for a symbol.
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            dict: Current market data
        """
        if not self._check_connection():
            return {}
            
        try:
            # Create contract
            contract = self.get_contract(symbol)
            
            # Request market data
            self.ib.reqMktData(contract)
            
            # Wait briefly for data to arrive
            self.ib.sleep(1)
            
            # Get ticker
            ticker = self.ib.ticker(contract)
            
            # Extract relevant data
            market_data = {
                "symbol": symbol,
                "last": ticker.last,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "close": ticker.close,
                "open": ticker.open,
                "high": ticker.high,
                "low": ticker.low,
                "volume": ticker.volume,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"Retrieved market data for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {}
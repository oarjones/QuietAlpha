# QuietAlpha Trading Bot

A low-risk trading bot using AI for portfolio management and trading decisions.

## Overview

QuietAlpha is designed as a two-module system:

1. **Portfolio Manager**: Scans the market to select the best symbols to invest in based on risk/reward profiles.
2. **Trading Manager**: Executes trades on the selected symbols while adhering to risk management rules and avoiding pattern day trading.

## Features

- AI-driven portfolio selection
- Reinforcement learning for optimized trade execution
- Low-risk strategy prioritization
- Pattern Day Trading (PDT) avoidance
- IBKR (Interactive Brokers) integration
- 1-hour timeframe trading

## Project Structure

```
quietalpha/
 |
 +-- portfolio_manager/       # Symbol selection and portfolio optimization
 +-- trading_manager/         # Trade execution and management
 +-- risk_management/         # Risk assessment and position sizing
 +-- data/                    # Data handling and preprocessing
 +-- ibkr_api/                # Interactive Brokers API interface
 +-- utils/                   # Utility functions
 +-- visualization/           # Visualization tools
 +-- backtesting/             # Backtesting framework
 +-- config/                  # Configuration settings
```

## Installation

```bash
pip install -e .
```

## Usage

TBD

## License

MIT
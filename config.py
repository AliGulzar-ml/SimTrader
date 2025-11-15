# Global configuration for the trading project

# Universe of tickers to train and trade on
TICKERS = ["AAPL", "MSFT", "SPY", "QQQ"]

# Start date for historical data
START_DATE = "2015-01-01"

# Saved model path
MODEL_PATH = "model.pkl"

# Initial cash for backtesting
INITIAL_CASH = 100_000.0

# Trading thresholds
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4

# Max fraction of portfolio in a single ticker
MAX_POSITION_PCT = 0.10

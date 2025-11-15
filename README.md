# Stock Trader ML (Mac-Optimized)

This project trains a gradient-boosted tree model to predict next-day stock direction,
backtests the strategy on historical data, and generates daily BUY/SELL/HOLD signals
that you can execute in the Investopedia Stock Simulator.

The code is optimized for a MacBook Pro with an Apple Silicon chip (like the M4 Pro)
by:

- Using vectorized `pandas` operations.
- Using `HistGradientBoostingClassifier`, which is implemented in C and runs very
  efficiently on multi-core CPUs.
- Avoiding heavy deep learning frameworks to keep installs and runtime light.

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # or: source .venv/bin/activate.fish
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the full pipeline (train, backtest, generate signals):

```bash
python main.py
```

This will:

- Download data for the tickers in `config.py`.
- Engineer features and train a model.
- Backtest the strategy.
- Print and save today's trading signals to `today_signals.csv`.

If you only want fresh signals (after you already trained a model):

```bash
python live_trader.py
```

Then log into your Investopedia simulator and manually enter the suggested
BUY/SELL/HOLD orders.

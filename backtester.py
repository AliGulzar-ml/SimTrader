import joblib
import pandas as pd
from config import MODEL_PATH, INITIAL_CASH, BUY_THRESHOLD, SELL_THRESHOLD


def backtest(df: pd.DataFrame,
             model_path: str = MODEL_PATH,
             buy_threshold: float = BUY_THRESHOLD,
             sell_threshold: float = SELL_THRESHOLD) -> pd.DataFrame:
    """
    Simple daily close-to-close backtest:
    - Long or flat only
    - Single aggregated portfolio (no leverage)
    - One share count for each ticker proportional to cash when entering.
    """
    model, feature_cols = joblib.load(model_path)
    df = df.copy()

    X = df[feature_cols].values.astype("float32")
    prob_up = model.predict_proba(X)[:, 1]
    df["prob_up"] = prob_up

    cash = INITIAL_CASH
    position = 0  # number of shares of a synthetic aggregated asset
    entry_price = 0.0
    portfolio_values = []

    prices = df["Adj Close"].values

    for price, p in zip(prices, prob_up):
        # BUY
        if p > buy_threshold and position == 0:
            position = int(cash // price)
            if position > 0:
                cash -= position * price
                entry_price = price
        # SELL
        elif p < sell_threshold and position > 0:
            cash += position * price
            position = 0

        portfolio_values.append(cash + position * price)

    df["portfolio_value"] = portfolio_values
    print(f"Final portfolio value: {portfolio_values[-1]:.2f}")
    return df

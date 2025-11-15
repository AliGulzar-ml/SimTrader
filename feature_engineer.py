import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features and next-day direction target to a price DataFrame."""
    df = df.copy()
    adj = df["Adj Close"]

    # Returns
    df["ret_1"] = adj.pct_change(1)
    df["ret_5"] = adj.pct_change(5)
    df["ret_10"] = adj.pct_change(10)

    # Moving averages and ratios
    for win in (5, 10, 20):
        ma_col = f"ma_{win}"
        ratio_col = f"ma_ratio_{win}"
        df[ma_col] = adj.rolling(win, min_periods=win).mean()
        df[ratio_col] = adj / df[ma_col] - 1

    # Volatility
    df["vol_5"] = df["ret_1"].rolling(5, min_periods=5).std()
    df["vol_10"] = df["ret_1"].rolling(10, min_periods=10).std()

    # Target = next-day return > 0
    df["future_ret"] = adj.pct_change(-1)
    df["target"] = (df["future_ret"] > 0).astype(int)

    df = df.dropna()
    return df

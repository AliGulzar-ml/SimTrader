import joblib
import pandas as pd
from config import TICKERS, MODEL_PATH, BUY_THRESHOLD, SELL_THRESHOLD
from data_loader import load_stock
from feature_engineer import add_features


def generate_signals(tickers=None, model_path: str = MODEL_PATH):
    """
    Generate today's BUY/SELL/HOLD signals for each ticker.
    Output is printed and saved to today_signals.csv.
    """
    if tickers is None:
        tickers = TICKERS

    model, feature_cols = joblib.load(model_path)
    rows = []

    for t in tickers:
        df = load_stock(t)
        df = add_features(df)
        latest = df.iloc[-1]
        X = latest[feature_cols].values.astype("float32").reshape(1, -1)
        prob_up = float(model.predict_proba(X)[0, 1])
        price = float(latest["Adj Close"])

        if prob_up > BUY_THRESHOLD:
            action = "BUY"
        elif prob_up < SELL_THRESHOLD:
            action = "SELL"
        else:
            action = "HOLD"

        rows.append({
            "Ticker": t,
            "Date": latest.name.strftime("%Y-%m-%d"),
            "Price": price,
            "Prob_Up": prob_up,
            "Action": action,
        })

    signals = pd.DataFrame(rows)
    print("\nToday's signals:\n")
    print(signals.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    signals.to_csv("today_signals.csv", index=False)
    print("\nSaved to today_signals.csv\n")
    return signals

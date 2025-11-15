import pandas as pd
from config import TICKERS
from data_loader import load_stock
from feature_engineer import add_features
from model_trainer import train_model
from backtester import backtest
from live_trader import generate_signals


def build_merged_dataframe(tickers):
    frames = []
    for t in tickers:
        df = load_stock(t)
        df = add_features(df)
        df["ticker"] = t
        frames.append(df)
    merged = pd.concat(frames).sort_index()
    return merged


def main():
    print("\n1) Loading and featurizing data...")
    df = build_merged_dataframe(TICKERS)

    print("\n2) Training model on merged universe...")
    train_model(df)

    print("\n3) Backtesting strategy on merged data...")
    backtest(df)

    print("\n4) Generating today's signals for each ticker...\n")
    generate_signals(TICKERS)


if __name__ == "__main__":
    main()

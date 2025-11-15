import yfinance as yf
from datetime import datetime
from config import START_DATE


def load_stock(ticker: str, start: str = START_DATE, end: str | None = None):
    """Download daily OHLCV data for a ticker using yfinance."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df = df.dropna()
    return df

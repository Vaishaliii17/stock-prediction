import yfinance as yf
import pandas as pd

def download_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data reliably from Yahoo Finance."""

    
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {ticker}.")

    df.index = pd.to_datetime(df.index)

  
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    
    df.rename(columns={'Close': 'Adj_Close'}, inplace=True)

    return df
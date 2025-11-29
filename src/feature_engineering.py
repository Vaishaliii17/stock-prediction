import pandas as pd
import numpy as np

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()

    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['return'] = df['Adj_Close'].pct_change()
    df['log_return'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1))

    df['sma_5'] = sma(df['Adj_Close'], 5)
    df['sma_10'] = sma(df['Adj_Close'], 10)
    df['sma_20'] = sma(df['Adj_Close'], 20)
    df['ema_10'] = ema(df['Adj_Close'], 10)

    df['vol_10'] = df['log_return'].rolling(10).std()

    df['rsi_14'] = rsi(df['Adj_Close'], 14)

    df['target'] = df['Adj_Close'].shift(-1)

    return df.dropna()
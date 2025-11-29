import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def download_stock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.dropna()
    return df


def prepare_data(df, window=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler


def build_lstm(window=60):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def main(ticker, start, end, window=60):
    df = download_stock(ticker, start, end)
    X, y, scaler = prepare_data(df, window)

    model = build_lstm(window)

    if not os.path.exists("models"):
        os.makedirs("models")

    checkpoint = ModelCheckpoint(
        f"models/{ticker.lower()}_lstm.keras",
        save_best_only=True,
        monitor="loss",
        mode="min"
    )

    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint, es],
        verbose=1
    )

    print("Model saved:", f"models/{ticker.lower()}_lstm.keras")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()

    main(args.ticker, args.start, args.end, args.window)
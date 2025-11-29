import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def prepare_data(df, window=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler


def forecast_future(model, df, scaler, window=60, days=30):
    last_window = df[['Close']].values[-window:]
    scaled_window = scaler.transform(last_window)

    predictions = []
    current = scaled_window

    for _ in range(days):
        pred = model.predict(current.reshape(1, window, 1), verbose=0)
        predictions.append(pred[0][0])
        current = np.vstack([current[1:], pred])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


def predict_stock(ticker="AAPL", start="2015-01-01", end="2024-12-31", window=60):
    df = yf.download(ticker, start=start, end=end, progress=False).dropna()

    X, y, scaler = prepare_data(df, window)

    model = load_model(f"models/{ticker.lower()}_lstm.keras")

    # Predict whole dataset
    preds = model.predict(X)
    actual = scaler.inverse_transform(y)
    predicted = scaler.inverse_transform(preds)

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price")
    plt.plot(predicted, label="Predicted Price")
    plt.title(f"{ticker} Actual vs Predicted Price")
    plt.legend()
    plt.show()

    # Future prediction
    forecast_7 = forecast_future(model, df, scaler, window, 7)
    forecast_15 = forecast_future(model, df, scaler, window, 15)
    forecast_30 = forecast_future(model, df, scaler, window, 30)

    print("\nNext 7 days:", forecast_7.flatten())
    print("\nNext 15 days:", forecast_15.flatten())
    print("\nNext 30 days:", forecast_30.flatten())

    # Plot 30-day forecast
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_30, label="30-Day Forecast")
    plt.title(f"{ticker} 30-Day Forecast")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    predict_stock("AAPL")

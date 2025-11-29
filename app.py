import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# -------------------------
# Data Preparation Function
# -------------------------
def prepare_data(df, window=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    X = []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])

    return np.array(X), scaler


# -------------------------
# Future Forecast Function
# -------------------------
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


# -------------------------
# Streamlit App UI
# -------------------------
st.title("📈 Stock Price Prediction App (LSTM Model)")
st.write("Predict future prices using your trained LSTM model.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
days = st.slider("Days to Forecast:", 7, 60, 30)

if st.button("Predict"):
    st.write(f"### Loading Data for **{ticker}**...")

    df = yf.download(ticker, start="2015-01-01", end="2024-12-31").dropna()
    st.line_chart(df["Close"])

    try:
        model_path = f"models/{ticker.lower()}_lstm.keras"
        model = load_model(model_path)
        st.success("Model Loaded Successfully!")
    except:
        st.error(f"Model file not found: {model_path}")
        st.stop()

    window = 60
    X, scaler = prepare_data(df, window)

    preds = model.predict(X)
    preds = scaler.inverse_transform(preds)

    # Plot historical vs predicted
    st.write("### 📉 Actual vs Predicted (Training Data)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Close'].values[window:], label="Actual")
    ax.plot(preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Future forecast
    st.write(f"### 🔮 Forecast for Next {days} Days")
    forecast = forecast_future(model, df, scaler, window, days)

    forecast_df = pd.DataFrame({
        "Day": np.arange(1, days + 1),
        "Predicted Price": forecast.flatten()
    })

    st.table(forecast_df)

    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(forecast_df["Predicted Price"])
    ax2.set_title(f"{ticker} - {days}-Day Forecast")
    st.pyplot(fig2)
    
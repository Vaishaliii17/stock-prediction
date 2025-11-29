# 📈 Stock Price Prediction Using LSTM (Deep Learning)

A complete end-to-end stock price prediction system using **LSTM neural networks**, built with **TensorFlow/Keras**, **yFinance**, **NumPy**, **Pandas**, and an interactive **Streamlit web app** for future forecasting.

---

## 🚀 Features

- ✔ Train an LSTM model on historical stock data  
- ✔ Predict future prices (7, 15, 30+ days)  
- ✔ Plot Actual vs Predicted prices  
- ✔ Interactive Streamlit Web App  
- ✔ Clean modular code structure  
- ✔ Works with any stock ticker  

---

## 📂 Project Structurestock-prediction/
│
├── src/
│ ├── train.py
│ ├── predict.py
│
├── models/
│ └── (saved models appear here)
│
├── app.py # Streamlit app
├── requirements.txt
├── README.md
## 🧠 Model Overview

This project uses a **Long Short-Term Memory (LSTM)** neural network to learn stock price patterns and forecast future values.

Model architecture:

- LSTM(64) → Dropout  
- LSTM(32) → Dropout  
- Dense(1) for output  
- Loss: `mean_squared_error`  
- Optimizer: `adam`

---

## 🛠️ Installation

### 1. Clone the Repository
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt
python -m src.train --ticker AAPL --start 2015-01-01 --end 2024-12-31
models/aapl_lstm.keras
python -m src.predict
Outputs:

Actual vs Predicted graph

7-, 15-, 30-day future forecast

Future forecast graph
streamlit run app.py
Features:

Enter any stock ticker (AAPL, TSLA, MSFT…)

View historical closing price chart

Predict next 7–60 days

Interactive comparison charts

Forecast table with future prices

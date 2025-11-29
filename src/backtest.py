import pandas as pd

def backtest_signals(dates, prices, preds):
    """Simple backtest: long when predicted price > current price."""
    df = pd.DataFrame({"date": dates, "price": prices, "pred": preds})
    df = df.set_index("date")

    df["signal"] = (df["pred"] > df["price"]).astype(int)

    df["next_ret"] = df["price"].pct_change().shift(-1).fillna(0)
    df["strategy_ret"] = df["signal"] * df["next_ret"]

    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    df["buy_hold"] = (1 + df["price"].pct_change().fillna(0)).cumprod()

    return df
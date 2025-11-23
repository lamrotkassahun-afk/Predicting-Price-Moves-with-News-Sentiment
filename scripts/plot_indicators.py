# scripts/plot_indicators.py

import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "../outputs"

def load_indicated_stock(symbol: str) -> pd.DataFrame:
    """
    Loads the CSV file with indicators added.
    """
    filepath = os.path.join(OUTPUT_DIR, f"{symbol}_with_indicators.csv")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return df


def plot_stock_indicators(symbol: str):
    """
    Plots Close price, SMA, RSI, and MACD for a given stock.
    """
    df = load_indicated_stock(symbol)

    # ---- PRICE + SMA ----
    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Close"], label="Close Price")
    plt.plot(df["Date"], df["SMA_20"], label="SMA 20")
    plt.plot(df["Date"], df["SMA_50"], label="SMA 50")
    plt.title(f"{symbol} Price + SMA")
    plt.legend()
    plt.grid()
    plt.show()

    # ---- RSI ----
    plt.figure(figsize=(14, 4))
    plt.plot(df["Date"], df["RSI_14"], label="RSI 14", color="purple")
    plt.axhline(70, color="red", linestyle="--")
    plt.axhline(30, color="green", linestyle="--")
    plt.title(f"{symbol} RSI 14")
    plt.grid()
    plt.show()

    # ---- MACD ----
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["MACD"], label="MACD", color="blue")
    plt.plot(df["Date"], df["MACD_SIGNAL"], label="Signal", color="orange")
    plt.bar(df["Date"], df["MACD_HIST"], label="MACD Histogram")
    plt.title(f"{symbol} MACD")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    symbols = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]

    for sym in symbols:
        print(f"Plotting {sym} indicators...")
        plot_stock_indicators(sym)

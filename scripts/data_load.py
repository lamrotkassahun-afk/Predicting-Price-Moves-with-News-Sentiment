# scripts/data_load.py

import os
import pandas as pd

DATA_DIR = "../data"

def load_stock(symbol: str) -> pd.DataFrame:
    """
    Loads a single stock CSV file from the data folder.
    """
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df


def load_all_stocks(symbols=None):
    """
    Loads all stock CSV files and returns a dictionary of DataFrames.
    """
    if symbols is None:
        symbols = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]

    stock_data = {}
    for sym in symbols:
        print(f"Loading {sym}...")
        stock_data[sym] = load_stock(sym)

    print("✔ All stocks loaded successfully.")
    return stock_data


if __name__ == "__main__":
    load_all_stocks()


import pandas as pd
import ta 
import sys
import os # Included os and sys for potential testing needs

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds SMA, EMA, RSI, and MACD to the DataFrame.
    Assumes the DataFrame has a 'Close' column.
    """
    df = df.copy()

    # --- Moving Averages (SMA & EMA) ---
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # --- Relative Strength Index (RSI) ---
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    # --- Moving Average Convergence Divergence (MACD) ---
    macd_data = ta.trend.macd(df['Close'])
    df['MACD_Line'] = macd_data
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    
    return df

# The following code only runs if you execute indicators.py directly, 
# not when it's imported by another script/notebook.
if __name__ == "__main__":
    # This block is safe to keep for testing purposes
    print("Running indicators.py as main script for testing...")
    # Add testing code here, using absolute paths if needed.
    # NOTE: REMOVE the code that failed in the traceback from the main body of the file!
    pass




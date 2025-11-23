import matplotlib.pyplot as plt
import pandas as pd
# Assume you have already run calculate_financial_metrics and have the DataFrame: df_metrics

def visualize_task2_plots(df: pd.DataFrame, ticker: str):
    """
    Creates the required visualizations for Task 2 metrics.
    """
    
    # Use a clean version of the Ticker for titles
    ticker_name = ticker.upper().replace('.CSV', '') 

    # --- Plot 1: Closing Price with SMA/EMA (Already Completed, but good to include) ---
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1.5)
    plt.plot(df.index, df['SMA_20'], label='SMA (20-day)', color='red', linestyle='--', linewidth=1)
    plt.plot(df.index, df['EMA_20'], label='EMA (20-day)', color='green', linestyle=':', linewidth=1)
    plt.title(f'{ticker_name} Closing Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    #  # Optional: for a cleaner visual

    # --- Plot 2: Daily Returns (New Metric Plot) ---
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['Daily_Return'] * 100, label='Daily Returns (%)', color='purple', alpha=0.7)
    plt.title(f'{ticker_name} Daily Percentage Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.axhline(0, color='red', linestyle='-', linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Plot 3: Drawdown (New Metric Plot) ---
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['Drawdown'] * 100, label='Drawdown (%)', color='orange', linewidth=2)
    plt.title(f'{ticker_name} Drawdown over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.fill_between(df.index, df['Drawdown'] * 100, color='orange', alpha=0.3)
    plt.axhline(df['Drawdown'].min() * 100, color='red', linestyle='--', label=f'Max Drawdown: {df["Drawdown"].min()*100:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # --- Plot 4: Rolling Sharpe Ratio (New Metric Plot) ---
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['Rolling_Sharpe'], label='Rolling Sharpe Ratio (20-day)', color='teal', linewidth=1.5)
    plt.title(f'{ticker_name} Rolling Sharpe Ratio (Risk-Adjusted Performance)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.axhline(0, color='red', linestyle='-', linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
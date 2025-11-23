import pandas as pd
import numpy as np

# We'll calculate core metrics using standard Pandas/NumPy
# as PyNance functions often require specific object types or full environment setup.

def calculate_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates essential financial metrics including Daily Returns, Volatility, 
    Cumulative Returns, and Drawdown (required for PyNance analysis).
    Assumes the DataFrame has a 'Close' column.
    """
    df = df.copy()

    # 1. Daily Returns (Required for Task 3 correlation analysis [cite: 159])
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 2. Cumulative Returns (for Drawdown calculation)
    # The '1 +' is crucial to handle percentage change correctly
    df['Cumulative_Returns'] = (1 + df['Daily_Return']).cumprod()
    
    # 3. Peak/Rolling Maximum (Required for Drawdown)
    df['Peak'] = df['Cumulative_Returns'].cummax()
    
    # 4. Drawdown (Required PyNance Metric)
    df['Drawdown'] = (df['Cumulative_Returns'] - df['Peak']) / df['Peak']
    
    # --- Other key metrics (calculated over the entire period) ---
    
    # Annualized Volatility (assuming 252 trading days)
    # Volatility is the standard deviation of daily returns
    daily_volatility = df['Daily_Return'].std()
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Annualized Returns (Mean of daily returns * 252 days)
    annual_returns = df['Daily_Return'].mean() * 252
    
    # Sharpe Ratio (Assuming 0% risk-free rate for simplicity)
    # Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
    sharpe_ratio = annual_returns / annual_volatility
    
    # Adding summary metrics to the DataFrame for inspection (will be constant rows)
    df['Annualized_Volatility'] = annual_volatility
    df['Sharpe_Ratio'] = sharpe_ratio
    df['Max_Drawdown'] = df['Drawdown'].min()
    
    return df

# If you run this file directly, nothing happens.
if __name__ == "__main__":
    pass
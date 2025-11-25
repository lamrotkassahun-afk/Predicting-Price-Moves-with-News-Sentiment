import os
import sys
import pandas as pd
from textblob import TextBlob
import numpy as np
from pathlib import Path

# --- Configuration and Path Setup ---
# Define the absolute path to your project root.
# NOTE: You MUST verify this path is correct for your local environment.
PROJECT_ROOT = Path(r'C:\Users\natna\Downloads\KAIM WEEK-1 CHALLENGE\Predicting-Price-Moves-with-News-Sentiment')

# Insert the root directory into the system path for modular imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NVDA'] 
DATA_DIR = PROJECT_ROOT / 'data' 
OUTPUT_DIR = PROJECT_ROOT / 'outputs' 
OUTPUT_DIR.mkdir(exist_ok=True) 

# --- CRITICAL CONFIGURATION ADJUSTMENT ---
# 1. NEWS FILE NAME (Updated based on user input)
NEWS_FILE_NAME = 'raw_analyst_ratings.csv' 

# 2. TICKER COLUMN NAME (Updated based on traceback output: 'stock' is correct)
TICKER_COLUMN_NAME = 'stock' 
# 3. TEXT COLUMN NAME (Updated based on traceback output: 'headline' is correct)
TEXT_COLUMN_NAME = 'headline'
# 4. DATE COLUMN NAME (Updated based on traceback output: 'date' is correct)
DATE_COLUMN_NAME = 'date'
# --- END CRITICAL CONFIGURATION ---

# --- 1. Core Functions ---

def calculate_headline_sentiment(text: str) -> float:
    """
    Calculates the sentiment polarity (-1.0 to +1.0) of a single text using TextBlob.
    -1.0: Very negative; +1.0: Very positive.
    """
    if pd.isna(text):
        return 0.0
    # TextBlob provides sentiment as a named tuple (polarity, subjectivity)
    return TextBlob(text).sentiment.polarity

def align_and_correlate(news_df: pd.DataFrame, stock_df: pd.DataFrame, ticker: str) -> float:
    """
    Performs data alignment, sentiment aggregation, and correlation calculation.

    Steps:
    1. Calculate Polarity for each headline.
    2. Map news to the NEXT trading day's stock return (Date Alignment).
    3. Aggregate sentiment per trading day (using the mean).
    4. Merge sentiment with stock returns.
    5. Calculate Pearson Correlation.
    """
    
    # 1. Calculate Polarity
    # Use the configured text column name
    try:
        news_df['Polarity'] = news_df[TEXT_COLUMN_NAME].apply(calculate_headline_sentiment)
    except KeyError:
        print(f"    -> ERROR: '{TEXT_COLUMN_NAME}' column not found. Please verify the column name for the news text.")
        return 0.0

    print(f"    -> Sentiment calculated for {len(news_df)} headlines.")

    # 2. Date Normalization and Mapping
    # Use the configured date column name
    try:
        # --- FIX: Use 'mixed' format with 'errors=coerce' for robust date parsing ---
        # This handles inconsistent formats in the raw data by coercing unparsable dates to NaT.
        news_df['Date_UTC'] = pd.to_datetime(news_df[DATE_COLUMN_NAME], format='mixed', errors='coerce').dt.normalize()
    except KeyError:
        print(f"    -> ERROR: '{DATE_COLUMN_NAME}' column not found. Please verify the column name for the news date.")
        return 0.0
    except Exception as e:
        print(f"    -> ERROR: Failed to parse dates using format 'mixed'. Full error: {e}")
        return 0.0
        
    # Drop rows where date parsing failed (Date_UTC is NaT)
    news_df.dropna(subset=['Date_UTC'], inplace=True)
    
    if news_df.empty:
        print("    -> WARNING: All news dates were unparsable after applying robust parsing. Skipping.")
        return 0.0

    # Stock returns are indexed by the CLOSE date (t). 
    # News published at time t affects the return observed at the CLOSE of t+1.
    
    # Get all unique trading days (dates) from the stock data
    trading_days = stock_df.index.normalize().unique().sort_values()
    
    # Function to find the next valid trading day *after* the news date
    def find_next_trading_day(date):
        # Filter for trading days strictly greater than the news date
        next_day_index = trading_days[trading_days > date]
        return next_day_index.min() if not next_day_index.empty else pd.NaT

    # Map each news item to the next trading day's return
    news_df['Next_Trading_Day'] = news_df['Date_UTC'].apply(find_next_trading_day)
    
    # Clean up news data by dropping rows where no next trading day could be found
    news_df.dropna(subset=['Next_Trading_Day'], inplace=True)
    
    if news_df.empty:
        print(f"    -> WARNING: No news articles for {ticker} could be mapped to a future trading day.")
        return 0.0
        
    # 3. Aggregate Sentiment per Trading Day
    # Calculate the average sentiment score for all news mapped to the same trading day
    daily_sentiment = news_df.groupby('Next_Trading_Day')['Polarity'].mean().reset_index()
    daily_sentiment.rename(columns={'Next_Trading_Day': 'date', 'Polarity': 'Avg_Daily_Sentiment'}, inplace=True)
    daily_sentiment.set_index('date', inplace=True)
    
    print(f"    -> Aggregated sentiment into {len(daily_sentiment)} daily scores.")

    # 4. Merge Sentiment with Stock Returns
    # Stock data already has 'Daily_Return' and is indexed by 'date'
    merged_df = stock_df[['Daily_Return']].merge(
        daily_sentiment, 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    merged_df.dropna(inplace=True)
    
    if merged_df.empty:
        print(f"    -> WARNING: No overlapping sentiment and return data for {ticker}. Check dates.")
        return 0.0
    
    # 5. Calculate Pearson Correlation
    # We calculate the Pearson correlation coefficient between the two aligned series.
    correlation = merged_df['Avg_Daily_Sentiment'].corr(merged_df['Daily_Return'], method='pearson')
    
    # Save merged data for inspection
    merged_df.to_csv(OUTPUT_DIR / f'{ticker}_sentiment_correlation_data.csv')
    print(f"    -> Merged sentiment and return data saved to {OUTPUT_DIR / f'{ticker}_sentiment_correlation_data.csv'}.")
    
    return correlation

# --- 2. Main Execution ---

def run_task3_correlation_analysis():
    """
    Main function to orchestrate the sentiment and correlation analysis for all tickers.
    """
    print("\n--- Starting Task 3: Correlation Analysis (Sentiment vs. Returns) ---")
    
    results = {}
    
    # Load the comprehensive news dataset
    news_file = DATA_DIR / NEWS_FILE_NAME
    
    try:
        global_news_df = pd.read_csv(news_file)
        print(f"Successfully loaded global news data ({len(global_news_df)} articles).")
    except FileNotFoundError:
        print(f"FATAL ERROR: News data file not found at {news_file}. Please ensure it is present and named '{NEWS_FILE_NAME}'.")
        return

    # --- CRITICAL FIX: CLEAN UP COLUMN NAMES ---
    # Strip any leading/trailing whitespace from column names to prevent KeyError
    global_news_df.columns = global_news_df.columns.str.strip()
    print("Column names cleaned (whitespace stripped).")
    # --- END CRITICAL FIX ---
    
    # Test if the Ticker column is present *after* cleaning
    if TICKER_COLUMN_NAME not in global_news_df.columns:
        print(f"\nFATAL ERROR: The required ticker column '{TICKER_COLUMN_NAME}' is still not found in the news file headers.")
        print("Available columns (after cleanup):", global_news_df.columns.tolist())
        print("Please verify the column name for the stock ticker symbols is correct.")
        return

    for ticker in TICKERS:
        print(f"\n[Processing {ticker}]")
        
        # Load the output from Task 2 (which contains Daily_Return)
        stock_file_path = OUTPUT_DIR / f'{ticker}_with_analysis.csv'
        
        try:
            stock_df = pd.read_csv(stock_file_path, parse_dates=['date'])
            stock_df.set_index('date', inplace=True)
        except FileNotFoundError:
            print(f"    -> ERROR: Task 2 output not found for {ticker} at {stock_file_path}. Skipping. Run Task 2 first.")
            continue
        
        # Filter news for the current ticker using the configured column name
        ticker_news_df = global_news_df[global_news_df[TICKER_COLUMN_NAME] == ticker].copy()
        
        if ticker_news_df.empty:
            print(f"    -> WARNING: No news articles found for {ticker} using column '{TICKER_COLUMN_NAME}'. Recording NaN.")
            results[ticker] = np.nan
            continue

        # Perform the alignment and correlation calculation
        correlation_score = align_and_correlate(ticker_news_df, stock_df, ticker)
        results[ticker] = correlation_score
        
        # Only print if a valid number was returned
        if not np.isnan(correlation_score):
             print(f"    -> Final Pearson Correlation (Sentiment vs. Daily Return): {correlation_score:.4f}")
        
    # --- 3. Final Results Summary ---
    
    print("\n--- Correlation Summary (Task 3 Results) ---")
    
    correlation_df = pd.Series(results, name='Pearson_Correlation').to_frame()
    correlation_df['Interpretation'] = correlation_df['Pearson_Correlation'].apply(
        lambda x: 'Strong Positive' if x > 0.7 else 
                  'Moderate Positive' if x > 0.3 else
                  'Weak/No Correlation' if x >= -0.3 else
                  'Moderate Negative' if x < -0.3 else 
                  'Strong Negative'
    )
    print(correlation_df.to_markdown())
    
    # Save Final Correlation Table
    final_output_file = OUTPUT_DIR / 'task3_final_correlation_results.csv'
    correlation_df.to_csv(final_output_file)
    print(f"\nFinal Correlation results saved to: {final_output_file}")


if __name__ == "__main__":
    # Ensure TextBlob data is downloaded (often required in new environments)
    try:
        import nltk
        # Attempt to download TextBlob's required corpus if not already present
        # nltk.download('punkt') 
        # nltk.download('averaged_perceptron_tagger')
    except Exception:
        # Fails silently if nltk is not installed or import fails
        pass

    run_task3_correlation_analysis()
    print("\nTask 3 execution complete.")
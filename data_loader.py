import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker='EURUSD=X', start='2010-01-01', end='2024-12-31'):
    print(f"Loading data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    
    # Handle MultiIndex columns if present (yfinance update)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Calculate Returns
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    
    print(f"Data loaded: {len(data)} rows")
    return data

if __name__ == "__main__":
    df = load_data()
    print(df.head())

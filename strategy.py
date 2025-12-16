import pandas as pd
import numpy as np
import ta

class MACDStrategy:
    def __init__(self, data):
        self.data = data.copy()
        
    def generate_signals(self, fast_window, slow_window, signal_window):
        """
        Generates buy/sell signals based on MACD Crossover.
        Returns the dataframe with 'Signal' and 'Strategy_Returns' columns.
        """
        # Ensure parameters are integers
        fast_window = int(fast_window)
        slow_window = int(slow_window)
        signal_window = int(signal_window)
        
        # Validation
        if fast_window >= slow_window:
            return None 

        df = self.data.copy()
        
        # Calculate MACD
        # MACD Line = Fast EMA - Slow EMA
        # Signal Line = EMA(MACD Line, signal_window)
        
        # Using pandas ewm for speed and control or ta library
        # Let's use pandas directly for performance in GA loops
        # df['EMA_Fast'] = df['Close'].ewm(span=fast_window, adjust=False).mean()
        # df['EMA_Slow'] = df['Close'].ewm(span=slow_window, adjust=False).mean()
        # df['MACD_Line'] = df['EMA_Fast'] - df['EMA_Slow']
        # df['Signal_Line'] = df['MACD_Line'].ewm(span=signal_window, adjust=False).mean()
        
        # Or using TA library which is robust
        macd = ta.trend.MACD(
            close=df['Close'], 
            window_slow=slow_window, 
            window_fast=fast_window, 
            window_sign=signal_window
        )
        df['MACD_Line'] = macd.macd()
        df['Signal_Line'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff() # Histogram
        
        # Signal Logic:
        # Buy (1) when MACD Line crosses ABOVE Signal Line
        # Sell (-1) when MACD Line crosses BELOW Signal Line
        
        df['Signal'] = 0
        df['Signal'] = np.where(df['MACD_Line'] > df['Signal_Line'], 1, -1)
        
        # Shift signal to apply to next day returns
        df['Position'] = df['Signal'].shift(1)
        df['Strategy_Returns'] = df['Position'] * df['Returns']
        
        df.dropna(inplace=True)
        return df

    def evaluate(self, fast_window, slow_window, signal_window):
        """
        Fitness Function: Returns Cumulative Return
        """
        res_df = self.generate_signals(fast_window, slow_window, signal_window)
        
        if res_df is None or len(res_df) == 0:
            return -9999.0
            
        cumulative_return = (1 + res_df['Strategy_Returns']).prod() - 1
        return cumulative_return

import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_loader import load_data
from strategy import MACDStrategy

# 1. Configuration (Best Params from GA - Update manually or fetch dynamically)
# From previous run: Fast=18, Slow=92, Signal=46
FAST = 18
SLOW = 92
SIGNAL = 46

# 2. Load Data & Generate Signals
data = load_data()
strategy = MACDStrategy(data)
df_signals = strategy.generate_signals(FAST, SLOW, SIGNAL)

# 3. Feature Engineering
df = df_signals.copy()

# Add indicators
df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

# Add MACD specific features
# Normalize MACD values by price to handle different price regimes
df['macd_norm'] = df['MACD_Line'] / df['Close']
df['signal_norm'] = df['Signal_Line'] / df['Close']
df['macd_hist_norm'] = df['MACD_Diff'] / df['Close']

# Slope of MACD Line
df['macd_slope'] = df['MACD_Line'].diff() 

df.dropna(inplace=True)

# 4. Prepare Dataset for ML
# Filter for Entry Points (when Signal changes)
df['Prev_Signal'] = df['Signal'].shift(1)
trade_entries = df[df['Signal'] != df['Prev_Signal']].copy()
# Filter out same signal continuation just in case (e.g. 1 -> 1) 
# Logic above handles it (0 -> 1 or 1 -> -1) if pure binary.
# Our Strategy class outputs 1 or -1. So changes are always flips.

trade_entries.dropna(inplace=True)

# Labeling: Profitability of the trade
trades = []
current_entry_price = None
current_entry_idx = None
current_signal = 0

for idx, row in df.iterrows():
    sig = row['Signal']
    price = row['Close']
    
    if sig != current_signal:
        if current_entry_price is not None:
             # Calculate return of the closed trade
             if current_signal == 1:
                 ret = (price - current_entry_price) / current_entry_price
             else:
                 ret = (current_entry_price - price) / current_entry_price
             
             trades.append({
                 'Entry_Idx': current_entry_idx,
                 'Return': ret,
                 'Label': 1 if ret > 0 else 0
             })
        
        current_entry_price = price
        current_entry_idx = idx
        current_signal = sig

trades_df = pd.DataFrame(trades)
trades_df.set_index('Entry_Idx', inplace=True)

# Join features
features = ['rsi', 'atr', 'adx', 'macd_norm', 'signal_norm', 'macd_hist_norm', 'macd_slope']
dataset = trades_df.join(df[features])
dataset.dropna(inplace=True)

print(f"Total Trades: {len(dataset)}")
print(f"Profitable: {dataset['Label'].sum()} ({(dataset['Label'].sum()/len(dataset)*100):.2f}%)")

# 5. Train Random Forest
X = dataset[features]
y = dataset['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
print("\n--- ML Classification Report ---")
preds = rf.predict(X_test)
print(classification_report(y_test, preds))

# Feature Importance
print("\n--- Feature Importance ---")
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

import pandas as pd
import numpy as np
import ta
import tensorflow as pd_tf # Just alias to avoid confusion if needed, but standard is tf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from data_loader import load_data
from strategy import MACDStrategy

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# 1. Config
# From previous run: Fast=18, Slow=92, Signal=46
FAST = 18
SLOW = 92
SIGNAL = 46
SEQ_LEN = 10 # Lookback window for LSTM

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

# MACD Norm
df['macd_norm'] = df['MACD_Line'] / df['Close']
df['signal_norm'] = df['Signal_Line'] / df['Close']
df['macd_hist_norm'] = df['MACD_Diff'] / df['Close']
df['macd_slope'] = df['MACD_Line'].diff() 

df.dropna(inplace=True)

# 4. Prepare Dataset for LSTM
# LSTM needs [Samples, TimeSteps, Features]
# We want to predict the Label of a Trade based on the Sequence LEADING UP TO the Trade Entry.

features = ['rsi', 'atr', 'adx', 'macd_norm', 'signal_norm', 'macd_hist_norm', 'macd_slope']

# Scale features first (Critical for LSTM/Neural Nets)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Identify Trade Entries
df['Prev_Signal'] = df['Signal'].shift(1)
trade_entries_mask = df['Signal'] != df['Prev_Signal']
# We only care about entries, but we need the index to look back
trade_indices = df.index[trade_entries_mask]

# Label Logic (Same as before)
trades = []
current_entry_price = None
current_entry_idx = None
current_signal = 0

# Re-loop to find labels (using unscaled price if possible, but price is not in features so it's fine)
# Note: df still has 'Close' column unscaled? Yes, we only scaled 'features' list.
# Wait, typical scaler usage modifies in place if assigned back.

# Let's be careful. Re-load or keep raw price.
raw_close = df['Close'].copy()

labels_dict = {} # Key: Index, Val: Label

for idx, row in df.iterrows():
    sig = row['Signal']
    price = raw_close[idx] # Use raw price
    
    if sig != current_signal:
        if current_entry_price is not None:
             if current_signal == 1:
                 ret = (price - current_entry_price) / current_entry_price
             else:
                 ret = (current_entry_price - price) / current_entry_price
             
             # Store label for the Entry Index
             labels_dict[current_entry_idx] = 1 if ret > 0 else 0
        
        current_entry_price = price
        current_entry_idx = idx
        current_signal = sig

# Construct Sequences
X = []
y = []

# We iterate through identified trade indices
# For each trade entry at 't', we want [t-SEQ_LEN : t] features.
# df index is datetime. We need integer location.
# It's safer to work with numpy array of features.

feature_data = df[features].values
# Create map from index (datetime) to integer loc
idx_map = {idx: i for i, idx in enumerate(df.index)}

for entry_idx in labels_dict: # Only valid closed trades
    if entry_idx not in idx_map: continue
    
    i = idx_map[entry_idx]
    
    if i < SEQ_LEN: continue # Not enough history
    
    # Extract sequence: from i-SEQ_LEN to i (non-inclusive of i? usually include i as "current state")
    # Let's say we use data UP TO the decision point.
    seq = feature_data[i-SEQ_LEN+1 : i+1] 
    
    if len(seq) == SEQ_LEN:
        X.append(seq)
        y.append(labels_dict[entry_idx])

X = np.array(X)
y = np.array(y)

print(f"LSTM Dataset Shape: X={X.shape}, y={y.shape}")
print(f"Profitable: {sum(y)} ({(sum(y)/len(y)*100):.2f}%)")

if len(X) == 0:
    print("Not enough data for LSTM!")
    exit()

# 5. Build & Train LSTM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(SEQ_LEN, len(features)), return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Binary Classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Training LSTM ---")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# 6. Evaluate
print("\n--- ML Classification Report ---")
preds_prob = model.predict(X_test)
preds = (preds_prob > 0.5).astype(int)

print(classification_report(y_test, preds, zero_division=0))

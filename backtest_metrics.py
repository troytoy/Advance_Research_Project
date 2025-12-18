import pandas as pd
import numpy as np

def calculate_metrics(returns_series, trades_df=None):
    """
    Calculates key performance metrics for a trading strategy.
    
    Args:
        returns_series (pd.Series): Daily returns of the strategy.
        trades_df (pd.DataFrame): DataFrame containing trade details (Entry, Exit, Return).
    
    Returns:
        dict: A dictionary of performance metrics.
    """
    
    metrics = {}
    
    # 1. Total Return
    total_return = (1 + returns_series).prod() - 1
    metrics['Total Return'] = f"{total_return * 100:.2f}%"
    
    # 2. Annualized Return (Assuming 252 trading days)
    days = len(returns_series)
    if days > 0:
        annualized_return = (1 + total_return) ** (252 / days) - 1
        metrics['Annualized Return'] = f"{annualized_return * 100:.2f}%"
    else:
        metrics['Annualized Return'] = "N/A"

    # 3. Volatility (Annualized)
    volatility = returns_series.std() * np.sqrt(252)
    metrics['Annualized Volatility'] = f"{volatility * 100:.2f}%"
    
    # 4. Sharpe Ratio (assuming Risk-Free Rate = 0 for simplicity)
    if volatility > 0:
        sharpe_ratio = (annualized_return / volatility)
        metrics['Sharpe Ratio'] = f"{sharpe_ratio:.2f}"
    else:
        metrics['Sharpe Ratio'] = "0.00"
        
    # 5. Max Drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    metrics['Max Drawdown'] = f"{max_drawdown * 100:.2f}%"
    
    # 6. Trade Statistics (if trades_df is provided)
    if trades_df is not None and not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['Label'] == 1]
        losing_trades = trades_df[trades_df['Label'] == 0]
        
        win_rate = len(winning_trades) / total_trades
        metrics['Win Rate'] = f"{win_rate * 100:.2f}%"
        
        avg_gain = winning_trades['Return'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['Return'].mean()) if len(losing_trades) > 0 else 0
        
        if avg_loss > 0:
            profit_factor = (len(winning_trades) * avg_gain) / (len(losing_trades) * avg_loss)
            metrics['Profit Factor'] = f"{profit_factor:.2f}"
        else:
            metrics['Profit Factor'] = "Inf"
            
        metrics['Total Trades'] = total_trades
    else:
        metrics['Win Rate'] = "N/A"
        metrics['Total Trades'] = 0

    return metrics

def print_metrics(metrics):
    print("\n" + "="*30)
    print("   STRATEGY PERFORMANCE")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:<20}: {v}")
    print("="*30 + "\n")

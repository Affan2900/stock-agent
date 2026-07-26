import numpy as np
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and stationary features to the DataFrame.
    
    All indicators are strictly computed on past information to eliminate lookahead leakage.
    
    Args:
        df: DataFrame containing ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
        
    Returns:
        DataFrame augmented with technical indicator columns, with NaN rows dropped.
    """
    data = df.copy()
    
    # 1. Lagged log returns
    for lag in range(1, 6):
        data[f"return_lag_{lag}"] = data["log_return"].shift(lag)
        
    # 2. Rolling Volatility 
    for window in [5, 10, 20]:
        data[f"volatility_{window}d"] = data["log_return"].rolling(window=window).std()
        
    # 3. SMA Price Ratios: Close / SMA(Close, N) - 1 (Stationary)
    for window in [5, 20, 50]:
        sma = data["Close"].rolling(window=window).mean()
        data[f"close_sma_ratio_{window}d"] = (data["Close"] / sma) - 1.0
        
    # 4. Exponential Moving Average (EMA) Ratio
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["ema_ratio_12_26"] = (ema_12 / ema_26) - 1.0
    
    # 5. MACD normalized by Close price
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    data["macd_norm"] = macd_line / data["Close"]
    data["macd_hist_norm"] = (macd_line - macd_signal) / data["Close"]
    
    # 6. Relative Strength Index (RSI - 14 day)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data["rsi_14"] = 100 - (100 / (1 + rs))
    # Rescale RSI to [0, 1] range for neural network stability
    data["rsi_14_norm"] = data["rsi_14"] / 100.0
    
    # 7. Log Volume Change
    data["vol_log_change"] = np.log((data["Volume"] + 1) / (data["Volume"].shift(1) + 1))
    
    # Drop rows with NaNs introduced by rolling windows and lags
    data = data.dropna()
    return data

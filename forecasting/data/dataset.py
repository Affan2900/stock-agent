from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

def create_forecasting_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log_return",
    lookback: int = 60,
    horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Construct rolling window inputs X and multi-horizon target returns Y.
    
    Args:
        df: Input DataFrame indexed by Date/Timestamp containing feature and target columns.
        feature_cols: List of feature column names to include in X.
        target_col: Column name of target log returns (default 'log_return').
        lookback: Number of historical time steps in input sequence X (default 60).
        horizon: Multi-step forecast horizon H (default 5).
        
    Returns:
        X: numpy array of shape (N, lookback, n_features)
        Y: numpy array of shape (N, horizon) containing future log returns [r_{t+1}, ..., r_{t+H}]
        dates: List of pd.Timestamp corresponding to the origin index t (the last observed day of context window X_i)
    """
    if len(df) < lookback + horizon:
        raise ValueError(
            f"DataFrame length ({len(df)}) insufficient for lookback={lookback} + horizon={horizon}."
        )
        
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
            
    feature_data = df[feature_cols].values
    target_data = df[target_col].values
    time_index = df.index
    
    N = len(df) - lookback - horizon + 1
    n_features = len(feature_cols)
    
    X = np.zeros((N, lookback, n_features), dtype=np.float32)
    Y = np.zeros((N, horizon), dtype=np.float32)
    origin_dates: List[pd.Timestamp] = []
    
    for i in range(N):
        # Input context window 
        X[i] = feature_data[i : i + lookback]
        
        # Target returns Y_i
        Y[i] = target_data[i + lookback : i + lookback + horizon]
        
        # Origin date t is the date at index (i + lookback - 1)
        origin_dates.append(time_index[i + lookback - 1])
        
    return X, Y, origin_dates

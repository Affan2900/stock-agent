import logging
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_ticker_data(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    use_adj_close: bool = True
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from yfinance and compute stationary log returns.
    
    Args:
        ticker: Ticker symbol (e.g. 'SPY', 'AAPL', 'MSFT').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD), or None for current date.
        use_adj_close: Whether to use Adjusted Close for price calculations.
        
    Returns:
        pd.DataFrame indexed by Date with columns:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date or 'latest'}")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' from yfinance.")
        
    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    price_col = "Adj Close" if (use_adj_close and "Adj Close" in df.columns) else "Close"
    df["Close"] = df[price_col]
    
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in ticker data.")
            
    df = df[required_cols].copy()
    df.sort_index(inplace=True)
    df = df.dropna()
    
    # Calculate daily log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Drop initial NaN from shift
    df = df.dropna()
    
    return df

def reconstruct_prices(initial_price: float, log_returns: np.ndarray) -> np.ndarray:
    """
    Reconstruct price series from an initial base price and a sequence of predicted/actual log returns.
    
    Args:
        initial_price: Price C_0 at baseline.
        log_returns: 1D or 2D array of log returns. If 2D (samples x horizon), cumsum across horizon axis.
        
    Returns:
        np.ndarray: Reconstructed price level(s).
    """
    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        cum_returns = np.cumsum(log_returns)
        return initial_price * np.exp(cum_returns)
    elif log_returns.ndim == 2:
        cum_returns = np.cumsum(log_returns, axis=1)
        return initial_price * np.exp(cum_returns)
    else:
        raise ValueError(f"Expected 1D or 2D array for log_returns, got ndim={log_returns.ndim}")

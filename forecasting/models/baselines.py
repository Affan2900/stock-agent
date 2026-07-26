from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class RandomWalkBaseline:
    """
    Random Walk (Naive) Baseline Forecaster.
    
    Assumes expected future log return is 0 (or historical mean return).
    Interval bounds expand with sqrt(horizon step) based on training return volatility.
    """
    
    def __init__(self, use_mean_return: bool = False):
        self.use_mean_return = use_mean_return
        self.mean_return: float = 0.0
        self.std_return: float = 0.01

    def fit(self, train_returns: np.ndarray) -> "RandomWalkBaseline":
        """
        Fit baseline on historical training log returns.
        
        Args:
            train_returns: 1D array of historical log returns.
        """
        train_returns = np.asarray(train_returns).ravel()
        if self.use_mean_return:
            self.mean_return = float(np.mean(train_returns))
        else:
            self.mean_return = 0.0
        self.std_return = float(np.std(train_returns, ddof=1)) + 1e-8
        return self

    def predict(
        self,
        n_samples: int,
        horizon: int = 5,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)
    ) -> Dict[float, np.ndarray]:
        """
        Predict return forecasts and quantile bands across horizon.
        
        Args:
            n_samples: Number of forecast evaluation instances (samples) N.
            horizon: Forecast horizon H (default 5).
            quantiles: Quantiles to output (default 0.10, 0.50, 0.90).
            
        Returns:
            Dict mapping quantile q -> numpy array of shape (n_samples, horizon)
        """
        outputs = {}
        # Horizon step scaling factor sqrt(1..H)
        h_scaling = np.sqrt(np.arange(1, horizon + 1, dtype=np.float32)) # (H,)
        
        for q in quantiles:
            z_score = float(stats.norm.ppf(q))
            # Shape (horizon,) -> (1, horizon) -> broadcast to (n_samples, horizon)
            q_step = self.mean_return + z_score * self.std_return * h_scaling
            outputs[q] = np.tile(q_step, (n_samples, 1))
            
        return outputs


class SeasonalNaiveBaseline:
    """
    Seasonal Naive Baseline Forecaster.
    
    Forecasts log return at horizon step h equal to log return observed h days ago (5-day cycle).
    """
    
    def __init__(self, season_length: int = 5):
        self.season_length = season_length
        self.train_residuals: np.ndarray = np.array([])
        
    def fit(self, train_returns: np.ndarray) -> "SeasonalNaiveBaseline":
        train_returns = np.asarray(train_returns).ravel()
        if len(train_returns) > self.season_length:
            # Residuals of seasonal naive forecast on training set
            naive_pred = train_returns[:-self.season_length]
            actual = train_returns[self.season_length:]
            self.train_residuals = actual - naive_pred
        else:
            self.train_residuals = train_returns
        return self

    def predict(
        self,
        recent_returns: np.ndarray,
        horizon: int = 5,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)
    ) -> Dict[float, np.ndarray]:
        """
        Args:
            recent_returns: Array of shape (n_samples, lookback_window) containing past returns.
            horizon: Forecast horizon H (default 5).
            quantiles: Quantiles to output.
            
        Returns:
            Dict mapping quantile q -> numpy array of shape (n_samples, horizon)
        """
        recent_returns = np.asarray(recent_returns)
        if recent_returns.ndim == 1:
            recent_returns = recent_returns.reshape(1, -1)
            
        n_samples = len(recent_returns)
        # Take last season_length returns from context window
        last_season = recent_returns[:, -self.season_length:] # (n_samples, season_length)
        
        if last_season.shape[1] < horizon:
            # Tile if horizon > season_length
            repeats = (horizon // last_season.shape[1]) + 1
            median_pred = np.tile(last_season, (1, repeats))[:, :horizon]
        else:
            median_pred = last_season[:, :horizon]
            
        outputs = {}
        for q in quantiles:
            if q == 0.50:
                outputs[q] = median_pred.copy()
            else:
                q_res = float(np.quantile(self.train_residuals, q)) if len(self.train_residuals) > 0 else 0.0
                outputs[q] = median_pred + q_res
                
        return outputs


class ARIMABaseline:
    """
    ARIMA Statistical Baseline Forecaster.
    
    Fits an ARIMA(p, d, q) model on train fold returns and projects point & quantile forecasts.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1)):
        self.order = order
        self.model_fit = None
        self.std_err: float = 0.01

    def fit(self, train_returns: np.ndarray) -> "ARIMABaseline":
        train_returns = np.asarray(train_returns).ravel()
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_returns, order=self.order)
            self.model_fit = model.fit()
            self.std_err = float(np.std(self.model_fit.resid, ddof=1)) + 1e-8
        except Exception as e:
            logger.warning(f"ARIMA fit failed ({e}). Falling back to simple AR(1) estimate.")
            self.model_fit = None
            self.std_err = float(np.std(train_returns, ddof=1)) + 1e-8
        return self

    def predict(
        self,
        n_samples: int,
        horizon: int = 5,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)
    ) -> Dict[float, np.ndarray]:
        """
        Forecast H steps ahead.
        """
        if self.model_fit is not None:
            try:
                forecast_res = self.model_fit.get_forecast(steps=horizon)
                mean_fc = forecast_res.predicted_mean # (horizon,)
            except Exception:
                mean_fc = np.zeros(horizon, dtype=np.float32)
        else:
            mean_fc = np.zeros(horizon, dtype=np.float32)
            
        outputs = {}
        h_scaling = np.sqrt(np.arange(1, horizon + 1, dtype=np.float32))
        
        for q in quantiles:
            z_score = float(stats.norm.ppf(q))
            q_fc = mean_fc + z_score * self.std_err * h_scaling
            outputs[q] = np.tile(q_fc, (n_samples, 1))
            
        return outputs

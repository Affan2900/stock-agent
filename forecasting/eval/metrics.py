from typing import Dict, Tuple, Union
import numpy as np
from scipy import stats

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error across samples and horizon."""
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error across samples and horizon."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_baseline: np.ndarray
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    Scales model MAE by the in-sample naive baseline MAE.
    MASE < 1.0 indicates outperformance relative to naive baseline.
    """
    mae_model = calculate_mae(y_true, y_pred)
    # Naive baseline on train set: predicting zero return or previous return
    mae_train_naive = float(np.mean(np.abs(y_train_baseline)))
    if mae_train_naive < 1e-8:
        return 1.0
    return mae_model / mae_train_naive

def calculate_pinball_loss(
    y_true: np.ndarray,
    q_preds: Dict[float, np.ndarray]
) -> Dict[float, float]:
    """
    Compute pinball (quantile) loss per quantile.
    
    Args:
        y_true: Ground truth array of shape (N, H)
        q_preds: Dict mapping quantile q -> predicted array of shape (N, H)
        
    Returns:
        Dict mapping quantile q -> mean pinball loss float
    """
    losses = {}
    for q, pred in q_preds.items():
        err = y_true - pred
        loss = np.maximum(q * err, (q - 1.0) * err)
        losses[q] = float(np.mean(loss))
    return losses

def calculate_coverage_and_width(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray
) -> Tuple[float, float]:
    """
    Compute empirical coverage ratio and average width for a prediction interval.
    
    Args:
        y_true: Ground truth array of shape (N, H)
        q_lower: Lower quantile prediction (e.g. q=0.10) of shape (N, H)
        q_upper: Upper quantile prediction (e.g. q=0.90) of shape (N, H)
        
    Returns:
        coverage: Empirical coverage ratio (fraction of true values inside [q_lower, q_upper])
        avg_width: Average interval width (q_upper - q_lower)
    """
    inside = (y_true >= q_lower) & (y_true <= q_upper)
    coverage = float(np.mean(inside))
    avg_width = float(np.mean(q_upper - q_lower))
    return coverage, avg_width

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions matching ground truth sign."""
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    # Treat 0 sign matches as concordant
    match = (sign_true == sign_pred)
    return float(np.mean(match))

def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    criterion: str = "absolute"
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy of two forecasts.
    
    Args:
        y_true: Actual targets (N, H) or 1D (N,)
        y_pred1: Forecasts from model 1 (N, H) or 1D (N,)
        y_pred2: Forecasts from model 2 (N, H) or 1D (N,)
        h: Forecast horizon / autocorrelation lag parameter.
        criterion: Loss criterion ('absolute' or 'squared').
        
    Returns:
        dm_stat: DM test statistic (z-score)
        p_value: Two-tailed p-value
    """
    y_t = np.asarray(y_true).ravel()
    p1 = np.asarray(y_pred1).ravel()
    p2 = np.asarray(y_pred2).ravel()
    
    if criterion == "absolute":
        e1 = np.abs(y_t - p1)
        e2 = np.abs(y_t - p2)
    elif criterion == "squared":
        e1 = (y_t - p1) ** 2
        e2 = (y_t - p2) ** 2
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Use 'absolute' or 'squared'.")
        
    d = e1 - e2
    N = len(d)
    if N < 2:
        return 0.0, 1.0
        
    d_bar = np.mean(d)
    
    # Autocovariance calculation for lag truncation h-1
    def autocovariance(x, lag):
        n = len(x)
        x_mean = np.mean(x)
        if lag >= n:
            return 0.0
        return np.sum((x[:n - lag] - x_mean) * (x[lag:] - x_mean)) / n
        
    gamma_0 = autocovariance(d, 0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_sum += autocovariance(d, lag)
        
    variance = (gamma_0 + 2.0 * gamma_sum) / N
    if variance <= 1e-12:
        return 0.0, 1.0
        
    dm_stat = float(d_bar / np.sqrt(variance))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat))))
    
    return dm_stat, p_value

def evaluate_forecast_metrics(
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    y_train_returns: np.ndarray,
    q_preds: Dict[float, np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation metric aggregator.
    
    Args:
        y_true: Ground truth target returns (N, H)
        y_pred_median: Median/Point forecast returns (N, H)
        y_train_returns: In-sample train set returns for scaling MASE
        q_preds: Optional dict mapping quantile float -> predicted returns array
        
    Returns:
        Dict of metric names to scalar float values.
    """
    metrics = {
        "mae": calculate_mae(y_true, y_pred_median),
        "rmse": calculate_rmse(y_true, y_pred_median),
        "mase": calculate_mase(y_true, y_pred_median, y_train_returns),
        "directional_accuracy": calculate_directional_accuracy(y_true, y_pred_median)
    }
    
    if q_preds is not None:
        pinball_losses = calculate_pinball_loss(y_true, q_preds)
        for q, loss_val in pinball_losses.items():
            metrics[f"pinball_loss_q{q:.2f}"] = loss_val
        metrics["mean_pinball_loss"] = float(np.mean(list(pinball_losses.values())))
        
        if 0.10 in q_preds and 0.90 in q_preds:
            cov, width = calculate_coverage_and_width(y_true, q_preds[0.10], q_preds[0.90])
            metrics["empirical_coverage_80"] = cov
            metrics["interval_width_80"] = width
            
    return metrics

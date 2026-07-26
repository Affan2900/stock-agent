import numpy as np
from typing import List, Tuple

class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) for Online Coverage Tracking.
    
    Dynamically adjusts nominal error rate alpha_t in response to online coverage errors,
    handling regime shifts and non-exchangeable financial time-series.
    """
    
    def __init__(self, target_coverage: float = 0.80, gamma: float = 0.05):
        """
        Args:
            target_coverage: Desired nominal coverage ratio (default 0.80 -> alpha=0.20).
            gamma: Learning rate parameter for alpha updates.
        """
        self.target_coverage = target_coverage
        self.target_alpha = 1.0 - target_coverage
        self.gamma = gamma
        self.current_alpha = self.target_alpha
        
        self.alpha_history: List[float] = [self.current_alpha]
        self.error_history: List[int] = []

    def update(self, y_true: float, q_lower: float, q_upper: float) -> float:
        """
        Observe ground truth y_t for step t and update online alpha_{t+1}.
        
        Args:
            y_true: Ground truth scalar target return.
            q_lower: Calibrated lower prediction bound.
            q_upper: Calibrated upper prediction bound.
            
        Returns:
            Updated alpha_{t+1}
        """
        # err_t = 1 if outside interval, 0 if inside interval
        err_t = 1 if (y_true < q_lower or y_true > q_upper) else 0
        self.error_history.append(err_t)
        
        # ACI update rule: alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
        self.current_alpha = self.current_alpha + self.gamma * (self.target_alpha - err_t)
        # Clip alpha to valid probability range [0.01, 0.99]
        self.current_alpha = float(np.clip(self.current_alpha, 0.01, 0.99))
        self.alpha_history.append(self.current_alpha)
        
        return self.current_alpha

    def get_realized_coverage(self, window: int = 50) -> float:
        """
        Compute empirical coverage ratio over the last `window` resolved predictions.
        Exportable to Prometheus metric `forecast_coverage_ratio`.
        """
        if not self.error_history:
            return self.target_coverage
            
        recent_errors = self.error_history[-window:]
        coverage = 1.0 - (sum(recent_errors) / len(recent_errors))
        return float(coverage)

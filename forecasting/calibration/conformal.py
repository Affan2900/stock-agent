import numpy as np
from typing import Dict, Tuple, Optional

class SplitConformalCalibrator:
    """
    Split Conformal Calibrator for Multi-Horizon Prediction Intervals.
    
    """
    
    def __init__(self, target_coverage: float = 0.80):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.q_adjust: Optional[np.ndarray] = None # Shape (H,)

    def calibrate(
        self,
        y_true_cal: np.ndarray,
        q_lower_cal: np.ndarray,
        q_upper_cal: np.ndarray
    ) -> "SplitConformalCalibrator":
        """
        Compute nonconformity scores and calculate quantile adjustment per horizon step.
        
        Args:
            y_true_cal: Calibration targets (N_cal, H)
            q_lower_cal: Uncalibrated lower quantile predictions (N_cal, H)
            q_upper_cal: Uncalibrated upper quantile predictions (N_cal, H)
        """
        y_true_cal = np.asarray(y_true_cal)
        q_lower_cal = np.asarray(q_lower_cal)
        q_upper_cal = np.asarray(q_upper_cal)
        
        N_cal, H = y_true_cal.shape
        
        scores = np.maximum(q_lower_cal - y_true_cal, y_true_cal - q_upper_cal) # (N_cal, H)
        
        # Finite-sample inflation factor for quantile calculation
        quant_level = np.ceil((N_cal + 1) * self.target_coverage) / N_cal
        quant_level = float(np.clip(quant_level, 0.0, 1.0))
        
        # Compute quantile per horizon step h
        self.q_adjust = np.zeros(H, dtype=np.float32)
        for h in range(H):
            self.q_adjust[h] = float(np.quantile(scores[:, h], quant_level))
            
        return self

    def predict(
        self,
        q_lower: np.ndarray,
        q_median: np.ndarray,
        q_upper: np.ndarray
    ) -> Dict[float, np.ndarray]:
        """
        Apply conformal calibration adjustment to raw test prediction bounds.
        
        Args:
            q_lower: Raw lower quantile (N, H)
            q_median: Raw median prediction (N, H)
            q_upper: Raw upper quantile (N, H)
            
        Returns:
            Dict mapping quantile float -> calibrated predictions array of shape (N, H)
        """
        if self.q_adjust is None:
            raise RuntimeError("Calibrator has not been fitted! Call calibrate() first.")
            
        q_lower = np.asarray(q_lower)
        q_median = np.asarray(q_median)
        q_upper = np.asarray(q_upper)
        
        # Adjust lower and upper bounds by q_adjust array of shape (H,) broadcasted to (N, H)
        calibrated_lower = q_lower - self.q_adjust
        calibrated_upper = q_upper + self.q_adjust
        
        return {
            0.10: calibrated_lower,
            0.50: q_median,
            0.90: calibrated_upper
        }

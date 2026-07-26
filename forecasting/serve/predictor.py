from typing import Dict, Any, Optional, Tuple, List
import time
import torch
import numpy as np

from forecasting.models.lstm import QuantileLSTMForecaster
from forecasting.calibration.conformal import SplitConformalCalibrator
from forecasting.eval.promotion_gate import ModelPromotionGate
from forecasting.serve.fallback import FallbackPredictor
from forecasting.data.ingestion import reconstruct_prices

class ProductionPredictor:
    """
    Production Serving Layer.
    
    Orchestrates candidate model inference, conformal calibration, model promotion gate checks,
    and automatic fallback routing.
    """
    
    def __init__(
        self,
        model: Optional[QuantileLSTMForecaster] = None,
        calibrator: Optional[SplitConformalCalibrator] = None,
        gate: Optional[ModelPromotionGate] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.calibrator = calibrator
        self.gate = gate or ModelPromotionGate()
        self.device = device
        self.fallback_predictor = FallbackPredictor()
        self.is_promoted = False

    def promote_and_initialize(
        self,
        metrics: Dict[str, float],
        q_preds_val: Dict[float, np.ndarray],
        latencies_ms: List[float]
    ) -> Tuple[bool, List[str]]:
        """Run candidate model through promotion gate before serving."""
        passed, reasons = self.gate.evaluate_candidate(metrics, q_preds_val, latencies_ms)
        self.is_promoted = passed
        return passed, reasons

    def predict(
        self,
        ticker: str,
        features_x: np.ndarray,
        current_price: float,
        train_returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Serve forecast response.
        If promoted model exists, return calibrated prediction with fallback: False.
        Else, automatically route to FallbackPredictor with fallback: True.
        """
        if not self.is_promoted or self.model is None or self.calibrator is None:
            fb = FallbackPredictor(train_returns=train_returns)
            return fb.predict(
                ticker=ticker,
                current_price=current_price,
                reason="Model is not promoted or failed promotion gate requirements."
            )
            
        self.model.to(self.device)
        self.model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            x_tensor = torch.tensor(features_x, dtype=torch.float32).to(self.device)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0) # (1, T, D)
                
            out = self.model(x_tensor).cpu().numpy() # (1, H, 3)
            
        latency_ms = (time.time() - start_time) * 1000.0
        
        q_lower_raw = out[:, :, 0]
        q_median_raw = out[:, :, 1]
        q_upper_raw = out[:, :, 2]
        
        # Apply split conformal calibration adjustment
        calibrated_dict = self.calibrator.predict(q_lower_raw, q_median_raw, q_upper_raw)
        
        median_returns = calibrated_dict[0.50][0]
        reconstructed_price_path = reconstruct_prices(current_price, median_returns)
        
        return {
            "ticker": ticker,
            "fallback": False,
            "latency_ms": round(latency_ms, 2),
            "current_price": current_price,
            "horizon": len(median_returns),
            "quantiles": {
                "0.10": calibrated_dict[0.10][0].tolist(),
                "0.50": calibrated_dict[0.50][0].tolist(),
                "0.90": calibrated_dict[0.90][0].tolist()
            },
            "reconstructed_prices": reconstructed_price_path.tolist()
        }

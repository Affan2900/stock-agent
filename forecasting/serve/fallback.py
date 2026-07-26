from typing import Dict, Any, Optional
import numpy as np

from forecasting.models.baselines import RandomWalkBaseline
from forecasting.data.ingestion import reconstruct_prices

class FallbackPredictor:
    """
    Graceful Fallback Predictor.
    
    Serves calibrated baseline forecaster responses when a trained ML model fails the promotion gate
    or encounters runtime issues. Explicitly marks response with fallback: true.
    """
    
    def __init__(self, train_returns: Optional[np.ndarray] = None):
        self.baseline = RandomWalkBaseline(use_mean_return=False)
        if train_returns is not None:
            self.baseline.fit(train_returns)
            
    def predict(
        self,
        ticker: str,
        current_price: float,
        reason: str = "Candidate model failed promotion gate requirements.",
        horizon: int = 5
    ) -> Dict[str, Any]:
        """
        Generate calibrated fallback prediction response.
        """
        # Baseline forecast for 1 sample, H steps
        q_preds = self.baseline.predict(n_samples=1, horizon=horizon)
        
        median_returns = q_preds[0.50][0]
        reconstructed_price_path = reconstruct_prices(current_price, median_returns)
        
        return {
            "ticker": ticker,
            "fallback": True,
            "fallback_reason": reason,
            "current_price": current_price,
            "horizon": horizon,
            "quantiles": {
                "0.10": q_preds[0.10][0].tolist(),
                "0.50": q_preds[0.50][0].tolist(),
                "0.90": q_preds[0.90][0].tolist()
            },
            "reconstructed_prices": reconstructed_price_path.tolist()
        }

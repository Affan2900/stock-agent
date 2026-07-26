import time
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

class ModelPromotionGate:
    """
    MLflow Model Promotion Gate.
    
    Decides whether a trained candidate model is fit to be promoted from Staging to Production.
    If rejected, the serving system automatically falls back to FallbackPredictor.
    """
    
    def __init__(
        self,
        max_mase: float = 1.0,
        coverage_range: Tuple[float, float] = (0.75, 0.85),
        max_p95_latency_ms: float = 50.0
    ):
        self.max_mase = max_mase
        self.min_coverage, self.max_coverage = coverage_range
        self.max_p95_latency_ms = max_p95_latency_ms

    def evaluate_candidate(
        self,
        metrics: Dict[str, float],
        q_preds: Dict[float, np.ndarray],
        inference_latencies_ms: List[float]
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate candidate model outputs and metrics against gate thresholds.
        
        Args:
            metrics: Dict containing 'mase', 'empirical_coverage_80', etc.
            q_preds: Dict mapping quantile float -> numpy array of predictions.
            inference_latencies_ms: List of inference latency measurements in milliseconds.
            
        Returns:
            passed: True if candidate satisfies all 4 promotion criteria, else False.
            reasons: List of human-readable failure reason strings if rejected.
        """
        reasons: List[str] = []
        
        # 1. Skill Score Check (MASE < 1.0)
        mase = metrics.get("mase", 999.0)
        if mase >= self.max_mase:
            reasons.append(f"Skill score failure: MASE ({mase:.4f}) >= threshold ({self.max_mase:.4f}). Model does not beat Random Walk.")
            
        # 2. Empirical Coverage Check (0.75 <= coverage <= 0.85)
        coverage = metrics.get("empirical_coverage_80", 0.0)
        if not (self.min_coverage <= coverage <= self.max_coverage):
            reasons.append(
                f"Coverage calibration failure: Empirical coverage ({coverage*100:.1f}%) "
                f"outside target range [{self.min_coverage*100:.0f}%, {self.max_coverage*100:.0f}%]."
            )
            
        # 3. Output Sanity Check (NaN, constant, ordering)
        for q, preds in q_preds.items():
            if np.isnan(preds).any() or np.isinf(preds).any():
                reasons.append(f"Output sanity failure: NaN or Inf values detected in quantile {q} predictions.")
            if np.std(preds) < 1e-7:
                reasons.append(f"Output sanity failure: Degenerate constant predictions detected in quantile {q}.")
                
        if 0.10 in q_preds and 0.90 in q_preds:
            if (q_preds[0.10] > q_preds[0.90] + 1e-6).any():
                reasons.append("Output sanity failure: Quantile crossing detected (q_0.10 > q_0.90).")
                
        # 4. Latency Budget Check (p95 latency < 50ms)
        if inference_latencies_ms:
            p95_lat = float(np.percentile(inference_latencies_ms, 95))
            if p95_lat > self.max_p95_latency_ms:
                reasons.append(
                    f"Latency budget failure: p95 latency ({p95_lat:.2f}ms) exceeds budget ({self.max_p95_latency_ms:.2f}ms)."
                )
                
        passed = len(reasons) == 0
        if passed:
            logger.info("Model Promotion Gate: PASSED. Model approved for Production.")
        else:
            logger.warning(f"Model Promotion Gate: REJECTED. Reasons: {reasons}")
            
        return passed, reasons

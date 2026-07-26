from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

@dataclass
class PolicyDecision:
    stance: str # 'BULLISH', 'BEARISH', 'NEUTRAL', 'ABSTAIN'
    confidence: float # [0.0, 1.0]
    reason: str
    allow_llm_override: bool = False

class UncertaintyGatedPolicy:
    """
    Deterministic Uncertainty-Gated Stance Policy.
    
    Determines stance recommendation mathematically from interval width and median return and prevents LLM hallucinated recommendations and enforces first-class abstentions.
    """
    
    def __init__(
        self,
        min_coverage_threshold: float = 0.75,
        max_interval_width: float = 0.15,
        noise_margin_ratio: float = 0.50
    ):
        self.min_coverage_threshold = min_coverage_threshold
        self.max_interval_width = max_interval_width
        self.noise_margin_ratio = noise_margin_ratio

    def evaluate(
        self,
        median_return: float,
        interval_width: float,
        coverage_health: float = 0.80,
        data_freshness: bool = True,
        is_fallback: bool = False,
        grounding_retries: int = 0
    ) -> PolicyDecision:
        """
        Evaluate forecast metrics and return deterministic policy decision.
        """
        # 1. First-Class Abstention Trigger Checks
        if is_fallback:
            return PolicyDecision(
                stance="ABSTAIN",
                confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE: Served by fallback forecaster due to promotion gate failure."
            )
            
        if not data_freshness:
            return PolicyDecision(
                stance="ABSTAIN",
                confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE: Market data is stale."
            )
            
        if coverage_health < self.min_coverage_threshold:
            return PolicyDecision(
                stance="ABSTAIN",
                confidence=0.0,
                reason=f"INSUFFICIENT_EVIDENCE: Empirical coverage ({coverage_health*100:.1f}%) degraded below minimum threshold."
            )
            
        if interval_width > self.max_interval_width:
            return PolicyDecision(
                stance="ABSTAIN",
                confidence=0.0,
                reason=f"INSUFFICIENT_EVIDENCE: Interval width ({interval_width*100:.1f}%) exceeds maximum safety threshold ({self.max_interval_width*100:.1f}%)."
            )
            
        if grounding_retries >= 2:
            return PolicyDecision(
                stance="ABSTAIN",
                confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE: Numeric grounding validation failed maximum retry limit (2 retries)."
            )
            
        # 2. Uncertainty-Gated Stance Rules
        interval_half_width = 0.5 * interval_width
        signal_noise_ratio = abs(median_return) / max(interval_half_width, 1e-6)
        
        if abs(median_return) < self.noise_margin_ratio * interval_half_width:
            stance = "NEUTRAL"
            confidence = 0.50
            reason = f"Median 5-day return ({median_return*100:.2f}%) is within interval noise band (half-width = {interval_half_width*100:.2f}%)."
        elif median_return > 0:
            stance = "BULLISH"
            confidence = min(1.0, round(0.5 + 0.5 * min(1.0, signal_noise_ratio - 0.5), 2))
            reason = f"Positive median return ({median_return*100:.2f}%) exceeds interval noise threshold."
        else:
            stance = "BEARISH"
            confidence = min(1.0, round(0.5 + 0.5 * min(1.0, signal_noise_ratio - 0.5), 2))
            reason = f"Negative median return ({median_return*100:.2f}%) exceeds interval noise threshold."
            
        return PolicyDecision(
            stance=stance,
            confidence=confidence,
            reason=reason,
            allow_llm_override=False
        )

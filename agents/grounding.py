import re
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GroundingValidator:
    """
    Deterministic Pure-Python Numeric Grounding Validator.
    
    Verifies that numeric claims, percentage predictions, and stance assertions in an LLM draft report mechanically match ground-truth forecast tensors and policy decisions without using an LLM.
    """
    
    def __init__(self, pct_tolerance: float = 0.50):
        """
        Args:
            pct_tolerance: Maximum allowable percentage point error tolerance (default 0.50 percentage points, e.g. 1.25% vs 1.40% is ok, 12% is a violation).
        """
        self.pct_tolerance = pct_tolerance

    def validate(
        self,
        draft_text: str,
        ground_truth: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate draft text against ground truth forecast metadata.
        
        Args:
            draft_text: LLM generated report text.
            ground_truth: Dict containing keys:
                - 'policy_stance': Expected stance ('BULLISH', 'BEARISH', etc.)
                - 'median_return_pct': Expected median return in percent (e.g. 1.25)
                - 'lower_return_pct': Expected lower bound return in percent (e.g. -0.50)
                - 'upper_return_pct': Expected upper bound return in percent (e.g. 3.00)
                - 'current_price': Expected base price float
                
        Returns:
            passed: True if zero violations found, else False.
            violations: List of human-readable violation descriptions.
        """
        violations: List[str] = []
        
        expected_stance = ground_truth.get("policy_stance", "").upper()
        expected_median_pct = ground_truth.get("median_return_pct", None)
        expected_lower_pct = ground_truth.get("lower_return_pct", None)
        expected_upper_pct = ground_truth.get("upper_return_pct", None)
        
        # 1. Stance Conformity Verification
        stance_match = re.search(r"stance:\s*([A-Z_]+)", draft_text, re.IGNORECASE)
        if stance_match:
            claimed_stance = stance_match.group(1).upper()
            if claimed_stance != expected_stance:
                violations.append(
                    f"Stance mismatch: Report claims Stance '{claimed_stance}', but policy requires '{expected_stance}'."
                )
                
        # 2. Percentage Claim Extraction & Verification
        # Extract numbers followed by % or percent (e.g. "1.25%", "-0.50%", "12%")
        pct_claims = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", draft_text)
        
        if pct_claims and expected_median_pct is not None:
            valid_pct_targets = [expected_median_pct]
            if expected_lower_pct is not None:
                valid_pct_targets.append(expected_lower_pct)
            if expected_upper_pct is not None:
                valid_pct_targets.append(expected_upper_pct)
                
            for claim_str in pct_claims:
                try:
                    val = float(claim_str)
                    # Check if claim matches any of the ground truth percentage targets
                    min_diff = min([abs(val - target) for target in valid_pct_targets])
                    if min_diff > self.pct_tolerance:
                        violations.append(
                            f"Numeric claim violation: Report states '{val:.2f}%', but no forecast return matching target ({valid_pct_targets}) within tolerance."
                        )
                except ValueError:
                    continue
                    
        # 3. Directional Word Mismatch Verification
        if expected_stance in ["BEARISH", "ABSTAIN"]:
            bullish_keywords = ["skyrocket", "soar", "rally strongly", "surge 10%"]
            for kw in bullish_keywords:
                if kw in draft_text.lower():
                    violations.append(
                        f"Directional claim violation: Report contains bullish phrase '{kw}' while policy stance is '{expected_stance}'."
                    )
                    
        passed = len(violations) == 0
        if passed:
            logger.info("Grounding Validator: PASSED.")
        else:
            logger.warning(f"Grounding Validator: FAILED with {len(violations)} violations: {violations}")
            
        return passed, violations

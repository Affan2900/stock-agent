from typing import Dict, Any, List, Tuple
from agents.grounding import GroundingValidator

def evaluate_deterministic_assertions(
    fixture: Dict[str, Any],
    agent_output: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Run deterministic assertions on agent workflow execution output against fixture expectations.
    
    Returns:
        passed: True if all assertions hold, else False.
        failures: List of failure description strings.
    """
    failures: List[str] = []
    
    # 1. Schema Check
    if "final_report" not in agent_output or not agent_output["final_report"]:
        failures.append("Schema Failure: 'final_report' is missing or empty.")
        
    actual_stance = agent_output.get("policy_stance", "UNKNOWN")
    expected_stance = fixture.get("expected_stance", "UNKNOWN")
    
    # 2. Stance Policy Assertion
    if actual_stance != expected_stance:
        failures.append(f"Stance Policy Failure: Expected stance '{expected_stance}', got '{actual_stance}'.")
        
    # 3. Abstention Assertion
    if fixture.get("should_abstain", False) and actual_stance != "ABSTAIN":
        failures.append(f"Abstention Failure: Fixture requires ABSTAIN, but agent produced stance '{actual_stance}'.")
        
    # 4. Grounding Assertion (if not abstained)
    if actual_stance != "ABSTAIN":
        validator = GroundingValidator()
        gt = {
            "policy_stance": expected_stance,
            "median_return_pct": round(fixture["median_return"] * 100, 2),
            "lower_return_pct": round(fixture["lower_return"] * 100, 2),
            "upper_return_pct": round(fixture["upper_return"] * 100, 2)
        }
        passed_gr, violations = validator.validate(agent_output.get("final_report", ""), gt)
        if not passed_gr:
            failures.extend([f"Grounding Failure: {v}" for v in violations])
            
    passed = len(failures) == 0
    return passed, failures

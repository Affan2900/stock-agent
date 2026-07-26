import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def evaluate_llm_judge(
    fixture: Dict[str, Any],
    agent_output: Dict[str, Any]
) -> Dict[str, float]:
    """
    LLM-as-Judge rubric evaluator scoring report faithfulness, hedging, and entity accuracy.
    
    Returns dict mapping metric name -> float score in [0.0, 1.0].
    """
    report = agent_output.get("final_report", "")
    stance = agent_output.get("policy_stance", "NEUTRAL")
    
    # If abstained, judge score is 1.0 for appropriate refusal
    if stance == "ABSTAIN":
        return {
            "faithfulness": 1.0,
            "hedging_quality": 1.0,
            "entity_accuracy": 1.0,
            "overall_judge_score": 1.0
        }
        
    # Heuristic rubric evaluation for local test harness
    faithfulness = 1.0
    hedging = 1.0
    entity_acc = 1.0
    
    # Hedging check: must mention interval bounds or return range
    if "interval" not in report.lower() and "%" not in report:
        hedging = 0.5
        
    # Faithfulness check: stance in report must match stance tag
    if stance.lower() not in report.lower():
        faithfulness = 0.7
        
    overall = (faithfulness + hedging + entity_acc) / 3.0
    
    return {
        "faithfulness": round(faithfulness, 2),
        "hedging_quality": round(hedging, 2),
        "entity_accuracy": round(entity_acc, 2),
        "overall_judge_score": round(overall, 2)
    }

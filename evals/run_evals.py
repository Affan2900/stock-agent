import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph import GroundedAgentGraph, AgentState
from evals.deterministic.assertions import evaluate_deterministic_assertions
from evals.judge.judge import evaluate_llm_judge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CIEvalHarness")

def run_evaluation_harness() -> int:
    evals_dir = Path(__file__).resolve().parent
    fixtures_path = evals_dir / "fixtures" / "test_cases.json"
    baselines_path = evals_dir / "baseline_scores.json"
    
    if not fixtures_path.exists():
        logger.error(f"Fixtures file not found: {fixtures_path}")
        return 1
        
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures: List[Dict[str, Any]] = json.load(f)
        
    with open(baselines_path, "r", encoding="utf-8") as f:
        baseline_scores: Dict[str, float] = json.load(f)
        
    graph = GroundedAgentGraph()
    
    total_fixtures = len(fixtures)
    deterministic_passes = 0
    stance_matches = 0
    abstention_matches = 0
    grounding_passes = 0
    judge_scores: List[float] = []
    
    logger.info(f"=== Running Agent Evaluation Harness on {total_fixtures} Fixtures ===")
    
    for fix in fixtures:
        state: AgentState = {
            "ticker": fix["ticker"],
            "current_price": fix["current_price"],
            "median_return": fix["median_return"],
            "lower_return": fix["lower_return"],
            "upper_return": fix["upper_return"],
            "interval_width": fix["interval_width"],
            "coverage_health": fix["coverage_health"],
            "data_freshness": fix["data_freshness"],
            "is_fallback": fix["is_fallback"],
            "news_headlines": fix["news_headlines"]
        }
        
        output = graph.run(state)
        
        passed_det, failures = evaluate_deterministic_assertions(fix, output)
        if passed_det:
            deterministic_passes += 1
        else:
            logger.warning(f"Fixture '{fix['id']}' Failed Deterministic Assertions: {failures}")
            
        if output.get("policy_stance") == fix.get("expected_stance"):
            stance_matches += 1
            
        if fix.get("should_abstain") and output.get("policy_stance") == "ABSTAIN":
            abstention_matches += 1
            
        if output.get("grounding_passed", True):
            grounding_passes += 1
            
        judge_res = evaluate_llm_judge(fix, output)
        judge_scores.append(judge_res["overall_judge_score"])
        
    # Calculate aggregate scores
    det_pass_rate = round(deterministic_passes / total_fixtures, 4)
    stance_acc = round(stance_matches / total_fixtures, 4)
    abstention_count = sum([1 for f in fixtures if f.get("should_abstain")])
    abstention_acc = round(abstention_matches / max(1, abstention_count), 4)
    grounding_pass_rate = round(grounding_passes / total_fixtures, 4)
    mean_judge_score = round(float(sum(judge_scores) / max(1, len(judge_scores))), 4)
    
    results = {
        "min_deterministic_pass_rate": det_pass_rate,
        "min_stance_accuracy": stance_acc,
        "min_abstention_accuracy": abstention_acc,
        "min_grounding_pass_rate": grounding_pass_rate,
        "min_overall_judge_score": mean_judge_score
    }
    
    logger.info("\n=== Evaluation Suite Metric Results ===")
    logger.info(json.dumps(results, indent=2))
    
    # Check for regression against baseline_scores.json
    regressions: List[str] = []
    for k, baseline_val in baseline_scores.items():
        computed_val = results.get(k, 0.0)
        if computed_val < baseline_val:
            regressions.append(
                f"REGRESSION: Metric '{k}' score ({computed_val:.4f}) dropped below baseline ({baseline_val:.4f})."
            )
            
    if regressions:
        logger.error("\n❌ CI EVALUATION GATE FAILED WITH REGRESSIONS:")
        for r in regressions:
            logger.error(f"  - {r}")
        return 1
    else:
        logger.info("\n✅ CI EVALUATION GATE PASSED ALL BENCHMARK THRESHOLDS!")
        return 0

if __name__ == "__main__":
    sys.exit(run_evaluation_harness())

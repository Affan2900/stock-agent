import logging
from typing import TypedDict, List, Dict, Any, Optional

from agents.policy import UncertaintyGatedPolicy
from agents.grounding import GroundingValidator
from agents.llm import LLMProvider, get_default_llm_provider

logger = logging.getLogger(__name__)

class AgentState(TypedDict, total=False):
    ticker: str
    current_price: float
    median_return: float
    lower_return: float
    upper_return: float
    interval_width: float
    coverage_health: float
    data_freshness: bool
    is_fallback: bool
    news_headlines: List[str]
    
    analyst_summary: str
    policy_stance: str
    policy_confidence: float
    policy_reason: str
    market_summary: str
    draft_report: str
    
    grounding_passed: bool
    grounding_violations: List[str]
    retries: int
    final_report: str

def performance_analyst_node(state: AgentState) -> AgentState:
    """Analyze forecast tensor and compute deterministic policy stance."""
    median_r = state.get("median_return", 0.0)
    lower_r = state.get("lower_return", -0.01)
    upper_r = state.get("upper_return", 0.01)
    interval_w = state.get("interval_width", upper_r - lower_r)
    cov_health = state.get("coverage_health", 0.80)
    freshness = state.get("data_freshness", True)
    fallback = state.get("is_fallback", False)
    retries = state.get("retries", 0)
    
    policy = UncertaintyGatedPolicy()
    decision = policy.evaluate(
        median_return=median_r,
        interval_width=interval_w,
        coverage_health=cov_health,
        data_freshness=freshness,
        is_fallback=fallback,
        grounding_retries=retries
    )
    
    summary = (
        f"Forecast Analysis for {state.get('ticker', 'Ticker')}:\n"
        f"• Median 5-day return: {median_r*100:.2f}%\n"
        f"• 80% Conformal Interval: [{lower_r*100:.2f}%, {upper_r*100:.2f}%] (Width = {interval_w*100:.2f}%)\n"
        f"• Deterministic Policy Call: {decision.stance} (Confidence: {decision.confidence*100:.0f}%)\n"
        f"• Policy Rationale: {decision.reason}"
    )
    
    return {
        "analyst_summary": summary,
        "policy_stance": decision.stance,
        "policy_confidence": decision.confidence,
        "policy_reason": decision.reason
    }

def market_expert_node(state: AgentState) -> AgentState:
    """Summarize news headlines and sentiment context."""
    headlines = state.get("news_headlines", ["Market conditions remain steady."])
    summary = "Market Context:\n" + "\n".join([f"- {h}" for h in headlines])
    return {"market_summary": summary}

def generator_node(state: AgentState, llm: Optional[LLMProvider] = None) -> AgentState:
    """Call LLM provider to write narrative report adhering strictly to policy stance."""
    llm = llm or get_default_llm_provider()
    
    stance = state.get("policy_stance", "NEUTRAL")
    ticker = state.get("ticker", "Ticker")
    analyst_summary = state.get("analyst_summary", "")
    market_summary = state.get("market_summary", "")
    violations = state.get("grounding_violations", [])
    
    prompt = (
        f"Write a financial report for {ticker}.\n"
        f"CRITICAL CONSTRAINT: You MUST state 'Stance: {stance}' and adhere to this call.\n"
        f"Use the exact metrics below without fabricating numbers:\n\n"
        f"{analyst_summary}\n\n"
        f"{market_summary}\n"
    )
    
    if violations:
        prompt += f"\nCORRECTION REQUIRED: Previous draft failed with violations:\n" + "\n".join(violations) + "\n"
        
    draft = llm.generate(prompt=prompt)
    
    # Ensure Stance tag is present in draft
    if "Stance:" not in draft:
        draft = f"Stance: {stance}\n" + draft
        
    return {"draft_report": draft}

def critic_node(state: AgentState) -> AgentState:
    """Audit draft report using GroundingValidator."""
    draft = state.get("draft_report", "")
    stance = state.get("policy_stance", "NEUTRAL")
    median_pct = round(state.get("median_return", 0.0) * 100, 2)
    lower_pct = round(state.get("lower_return", 0.0) * 100, 2)
    upper_pct = round(state.get("upper_return", 0.0) * 100, 2)
    
    supplied_context = "\n".join([
        state.get("analyst_summary", ""),
        state.get("market_summary", "")
    ])

    validator = GroundingValidator()
    passed, violations = validator.validate(
        draft_text=draft,
        ground_truth={
            "policy_stance": stance,
            "median_return_pct": median_pct,
            "lower_return_pct": lower_pct,
            "upper_return_pct": upper_pct,
            "supplied_context": supplied_context
        }
    )
    
    return {
        "grounding_passed": passed,
        "grounding_violations": violations
    }

def revise_node(state: AgentState) -> AgentState:
    """Increment retry counter when grounding fails."""
    retries = state.get("retries", 0) + 1
    return {"retries": retries}

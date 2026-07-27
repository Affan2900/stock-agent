import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph import GroundedAgentGraph, AgentState
from agents.llm import MockLLMProvider
from agents.cache import DualCache, derive_cache_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4AgentDemo")

def run_agent_demonstration():
    logger.info("=== Phase 4 — Grounded Multi-Agent Reporting Layer Demonstration ===")
    
    graph = GroundedAgentGraph()
    
    # -------------------------------------------------------------------------
    # Scenario 1: Standard Grounded Forecast (Bullish Stance)
    # -------------------------------------------------------------------------
    logger.info("\n--- Scenario 1: Standard Grounded Forecast (AAPL) ---")
    state_bullish: AgentState = {
        "ticker": "AAPL",
        "current_price": 185.50,
        "median_return": 0.0125, # +1.25%
        "lower_return": -0.0050, # -0.50%
        "upper_return": 0.0300, # +3.00%
        "interval_width": 0.0350, # 3.50%
        "coverage_health": 0.81,
        "data_freshness": True,
        "is_fallback": False,
        "news_headlines": [
            "Apple announces new AI feature rollout for iOS.",
            "Analysts raise price targets following Q3 supply chain survey."
        ]
    }
    
    out_1 = graph.run(state_bullish)
    logger.info(f"Policy Decision: {out_1['policy_stance']} (Confidence: {out_1.get('policy_confidence', 0)*100:.0f}%)")
    logger.info(f"Grounding Passed: {out_1.get('grounding_passed')}")
    logger.info(f"Final Output:\n{out_1['final_report']}\n")
    
    # -------------------------------------------------------------------------
    # Scenario 2: Wide-Interval Uncertainty (Forced ABSTAIN)
    # -------------------------------------------------------------------------
    logger.info("\n--- Scenario 2: Wide-Interval Uncertainty (MSFT - Forced ABSTAIN) ---")
    state_wide: AgentState = {
        "ticker": "MSFT",
        "current_price": 420.00,
        "median_return": 0.0200, # +2.00%
        "lower_return": -0.0800, # -8.00%
        "upper_return": 0.1200, # +12.00%
        "interval_width": 0.2000, # 20.00% (> 15% max threshold)
        "coverage_health": 0.80,
        "data_freshness": True,
        "is_fallback": False,
        "news_headlines": ["High market volatility amidst tech earnings announcements."]
    }
    
    out_2 = graph.run(state_wide)
    logger.info(f"Policy Decision: {out_2['policy_stance']}")
    logger.info(f"Policy Reason: {out_2.get('policy_reason')}")
    logger.info(f"Final Output:\n{out_2['final_report']}\n")
    
    # -------------------------------------------------------------------------
    # Scenario 3: Grounding Violation & Conditional Retry Correction
    # -------------------------------------------------------------------------
    logger.info("\n--- Scenario 3: Grounding Violation & Conditional Retry Loop ---")
    
    # Custom Mock LLM that generates a fabricated 15.0% return on 1st call, then fixes it on retry
    class BadFirstCallLLM(MockLLMProvider):
        def __init__(self):
            super().__init__()
            self.call_count = 0
        def generate(self, prompt: str, system_prompt = None, temperature = 0.2) -> str:
            self.call_count += 1
            if self.call_count == 1:
                return "Stance: BULLISH\nForecast indicates an incredible surge of 15.0% over the next 5 days!"
            else:
                return "Stance: BULLISH\nForecast indicates a median return of 1.25% with 80% interval [-0.50%, 3.00%]."
                
    retry_graph = GroundedAgentGraph(llm=BadFirstCallLLM())
    out_3 = retry_graph.run(state_bullish)
    logger.info(f"Total Retries Executed: {out_3.get('retries')}")
    logger.info(f"Final Grounding Passed: {out_3.get('grounding_passed')}")
    logger.info(f"Final Output:\n{out_3['final_report']}\n")
    
    # -------------------------------------------------------------------------
    # Scenario 4: Two-Tiered Cache Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Scenario 4: Two-Tiered Cache Verification ---")
    cache = DualCache()
    key = derive_cache_key("AAPL", "fc_hash_aapl", "news_hash_aapl")
    cache.set_exact(key, {"final_report": out_1['final_report']})
    
    exact_hit = cache.get_exact(key)
    logger.info(f"Tier 1 Exact Hash Cache Hit: {'SUCCESS' if exact_hit else 'FAILED'}")
    
    news_text = "Apple expands AI datacenter investments and chip orders."
    cache.store_news_framing(news_text, {"framing": "ai_expansion"})
    
    similar_news = "Apple datacenter expansion and artificial intelligence chip orders"
    framing = cache.find_similar_news_framing(similar_news, similarity_threshold=0.50)
    logger.info(f"Tier 2 News Similarity Cache Hit: {'SUCCESS' if framing else 'FAILED'}")

if __name__ == "__main__":
    run_agent_demonstration()

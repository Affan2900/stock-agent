import logging
from typing import Dict, Any, Tuple, Optional

from agents.nodes import (
    AgentState,
    performance_analyst_node,
    market_expert_node,
    generator_node,
    critic_node,
    revise_node
)
from agents.llm import LLMProvider, get_default_llm_provider

logger = logging.getLogger(__name__)

class GroundedAgentGraph:
    """
    State-Graph Runner for Multi-Agent Grounded Reporting Layer.
    
    Orchestrates analyst, market, generator, critic, and conditional retry loop.
    """
    
    def __init__(self, llm: Optional[LLMProvider] = None, max_retries: int = 2):
        self.llm = llm or get_default_llm_provider()
        self.max_retries = max_retries

    def run(self, initial_state: AgentState) -> AgentState:
        """
        Execute agent pipeline trajectory.
        """
        state = dict(initial_state)
        state["retries"] = state.get("retries", 0)
        
        # Step 1: Performance Analyst Node
        analyst_out = performance_analyst_node(state)
        state.update(analyst_out)
        
        # Check if policy forced ABSTAIN early
        if state.get("policy_stance") == "ABSTAIN":
            logger.info("Policy forced ABSTAIN. Skipping generator loop.")
            state["final_report"] = (
                f"REPORT STATUS: ABSTAIN\n"
                f"Ticker: {state.get('ticker', 'Ticker')}\n"
                f"Reason: {state.get('policy_reason')}"
            )
            return state
            
        # Step 2: Market Expert Node
        market_out = market_expert_node(state)
        state.update(market_out)
        
        # Step 3: Generation & Grounding Retry Loop
        while state["retries"] <= self.max_retries:
            # Generator Node
            gen_out = generator_node(state, llm=self.llm)
            state.update(gen_out)
            
            # Critic Node
            critic_out = critic_node(state)
            state.update(critic_out)
            
            if state.get("grounding_passed", False):
                logger.info("Grounding Critic PASSED.")
                state["final_report"] = state["draft_report"]
                return state
            else:
                logger.warning(f"Grounding Critic FAILED (Retry {state['retries']}/{self.max_retries}).")
                if state["retries"] < self.max_retries:
                    # Revise Node
                    rev_out = revise_node(state)
                    state.update(rev_out)
                else:
                    logger.warning("Maximum retries reached. Forcing ABSTAIN response.")
                    state["policy_stance"] = "ABSTAIN"
                    state["final_report"] = (
                        f"REPORT STATUS: ABSTAIN\n"
                        f"Ticker: {state.get('ticker', 'Ticker')}\n"
                        f"Reason: INSUFFICIENT_EVIDENCE: Failed grounding validation after {self.max_retries} retries."
                    )
                    return state
                    
        return state

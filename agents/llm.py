import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LLMProvider:
    """Base interface for LLM completion providers."""

    last_backend: str = "unknown"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        raise NotImplementedError

class MockLLMProvider(LLMProvider):
    """Deterministic Mock LLM Provider for offline testing and fast demonstration."""

    def __init__(self, override_response: Optional[str] = None):
        self.override_response = override_response
        self.last_backend = "mock"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        self.last_backend = "mock"
        if self.override_response:
            return self.override_response
            
        # Parse context from prompt if possible
        if "BULLISH" in prompt:
            stance = "BULLISH"
        elif "BEARISH" in prompt:
            stance = "BEARISH"
        elif "ABSTAIN" in prompt:
            stance = "ABSTAIN"
        else:
            stance = "NEUTRAL"
            
        import re
        med_match = re.search(r"Median 5-day return:\s*([+-]?\d+(?:\.\d+)?)\s*%", prompt)
        median_str = (med_match.group(1) + "%") if med_match else "1.25%"
        
        int_match = re.search(r"80% Conformal Interval:\s*\[([+-]?\d+(?:\.\d+)?)\s*%,\s*([+-]?\d+(?:\.\d+)?)\s*%\]", prompt)
        if int_match:
            lower_str = int_match.group(1) + "%"
            upper_str = int_match.group(2) + "%"
        else:
            lower_str, upper_str = "-0.50%", "3.00%"
            
        return (
            f"Analysis Report\n"
            f"Stance: {stance}\n"
            f"The 5-trading-day forecast indicates a median return of {median_str} with an 80% confidence interval "
            f"spanning from {lower_str} to {upper_str}. Current price action reflects this outlook."
        )


class BedrockLLMProvider(LLMProvider):
    """Amazon Bedrock LLM Provider."""
    
    DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def __init__(self, model_id: Optional[str] = None, region_name: Optional[str] = None):
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", self.DEFAULT_MODEL_ID)
        region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.region_name = region_name
        self.client = None
        self.last_backend = "uninitialized"
        self.last_error: Optional[str] = None
        try:
            import boto3
            self.client = boto3.client("bedrock-runtime", region_name=region_name)
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Failed to initialize boto3 Bedrock client ({e}). Will fallback if called.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.client:
            logger.warning("Bedrock client uninitialized. Falling back to MockLLMProvider.")
            self.last_backend = "mock"
            return MockLLMProvider().generate(prompt, system_prompt, temperature)

        try:
            import json
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            })
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            response_body = json.loads(response.get("body").read())
            self.last_backend = f"bedrock:{self.model_id}"
            self.last_error = None
            return response_body["content"][0]["text"]
        except Exception as e:
            logger.error(f"Bedrock invocation failed ({e}). Falling back to MockLLMProvider.")
            self.last_backend = "mock"
            self.last_error = str(e)
            return MockLLMProvider().generate(prompt, system_prompt, temperature)


_PROVIDERS = {
    "bedrock": BedrockLLMProvider,
    "mock": MockLLMProvider,
}


def get_default_llm_provider() -> LLMProvider:
    """
    Factory helper returning available LLM provider.

    LLM_PROVIDER pins one explicitly; otherwise a configured AWS region selects
    Bedrock and its absence selects the mock. An unknown LLM_PROVIDER raises
    rather than defaulting, so a typo cannot silently downgrade a deployment to
    templated text that still looks like a report.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in _PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER={explicit!r}; expected one of {sorted(_PROVIDERS)}"
            )
        return _PROVIDERS[explicit]()

    if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        return BedrockLLMProvider()
    return MockLLMProvider()

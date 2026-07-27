"""Backend selection and, more importantly, honest reporting of degradation.

A provider that quietly falls back to templated text while still returning a
plausible-looking report is the failure mode these tests exist to prevent.
"""

import pytest

from agents.llm import (BedrockLLMProvider, MockLLMProvider,
                        get_default_llm_provider)

ENV_KEYS = ("AWS_REGION", "AWS_DEFAULT_REGION", "LLM_PROVIDER")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_no_aws_region_selects_mock():
    assert isinstance(get_default_llm_provider(), MockLLMProvider)


def test_aws_region_selects_bedrock(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert isinstance(get_default_llm_provider(), BedrockLLMProvider)


def test_llm_provider_pins_backend_over_inference(monkeypatch):
    """An explicit pin beats a configured region, so CI stays offline."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert isinstance(get_default_llm_provider(), MockLLMProvider)


def test_unknown_provider_raises_rather_than_defaulting(monkeypatch):
    """A typo must not silently downgrade a deployment to templated text."""
    monkeypatch.setenv("LLM_PROVIDER", "bedrok")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_default_llm_provider()


def test_mock_reports_itself_as_mock():
    p = MockLLMProvider()
    p.generate("Median 5-day return: 0.08 %")
    assert p.last_backend == "mock"


def test_mock_echoes_the_figures_it_was_given():
    """Keeps the mock grounded, so the eval harness is not scored against
    numbers no one supplied."""
    p = MockLLMProvider()
    out = p.generate(
        "Median 5-day return: 0.08 %\n"
        "80% Conformal Interval: [-1.01 %, 1.06 %]\nStance: NEUTRAL"
    )
    assert "0.08%" in out
    assert "-1.01%" in out and "1.06%" in out


def test_bedrock_failure_degrades_to_mock_and_says_so(monkeypatch):
    """The account-level Bedrock block made this path load-bearing: the report
    still renders, but last_backend must not claim Bedrock produced it."""
    p = BedrockLLMProvider(region_name="us-east-1")

    class Boom:
        def invoke_model(self, **_):
            raise RuntimeError("ValidationException: Operation not allowed")

    monkeypatch.setattr(p, "client", Boom())
    out = p.generate("Median 5-day return: 0.08 %\nStance: NEUTRAL")

    assert out.strip()
    assert p.last_backend == "mock"
    assert "Operation not allowed" in p.last_error


def test_bedrock_success_reports_the_model_id(monkeypatch):
    import io
    import json

    p = BedrockLLMProvider(model_id="test-model-id", region_name="us-east-1")

    class Fake:
        def invoke_model(self, **_):
            body = json.dumps({"content": [{"text": "Stance: NEUTRAL"}]})
            return {"body": io.BytesIO(body.encode())}

    monkeypatch.setattr(p, "client", Fake())
    out = p.generate("anything")

    assert out == "Stance: NEUTRAL"
    assert p.last_backend == "bedrock:test-model-id"
    assert p.last_error is None

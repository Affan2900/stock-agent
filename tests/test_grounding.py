"""The grounding validator is the only thing standing between generated prose and
an unverifiable numeric claim, so both what it catches and what it wrongly catches
are pinned here."""

import pytest

from agents.grounding import GroundingValidator

GROUND_TRUTH = {
    "policy_stance": "NEUTRAL",
    "median_return_pct": 0.08,
    "lower_return_pct": -1.01,
    "upper_return_pct": 1.06,
}


@pytest.fixture
def validator():
    return GroundingValidator()


def test_faithful_draft_passes(validator):
    draft = (
        "Stance: NEUTRAL\n"
        "Median 5-day return of 0.08% with an 80% interval of [-1.01%, 1.06%]."
    )
    passed, violations = validator.validate(draft, GROUND_TRUTH)
    assert passed, violations


def test_stance_contradiction_is_caught(validator):
    """The stance is computed upstream; prose claiming otherwise is a violation."""
    draft = "Stance: BULLISH\nMedian 5-day return of 0.08%."
    passed, violations = validator.validate(draft, GROUND_TRUTH)
    assert not passed
    assert any("stance" in v.lower() for v in violations)


def test_fabricated_number_is_caught(validator):
    """The core threat model: a figure that appears nowhere in the forecast."""
    draft = "Stance: NEUTRAL\nWe expect a median 5-day return of 12.40%."
    passed, violations = validator.validate(draft, GROUND_TRUTH)
    assert not passed
    assert any("12.40" in v for v in violations)


def test_nominal_coverage_mention_is_not_a_violation(validator):
    """'80% interval' is the coverage level, not a return claim."""
    draft = "Stance: NEUTRAL\nThe 80% conformal interval spans [-1.01%, 1.06%]."
    passed, violations = validator.validate(draft, GROUND_TRUTH)
    assert passed, violations


def test_small_rounding_drift_is_tolerated(validator):
    """0.50pp of slack, so 0.08% reported as 0.30% is not treated as invention."""
    draft = "Stance: NEUTRAL\nMedian 5-day return of 0.30%."
    passed, _ = validator.validate(draft, GROUND_TRUTH)
    assert passed


def test_bullish_language_under_abstention_is_caught(validator):
    draft = "Stance: ABSTAIN\nShares should skyrocket from here."
    passed, violations = validator.validate(
        draft, {**GROUND_TRUTH, "policy_stance": "ABSTAIN"}
    )
    assert not passed
    assert any("directional" in v.lower() for v in violations)


SUPPLIED_CONTEXT = (
    "• Median 5-day return: 0.08%\n"
    "• 80% Conformal Interval: [-1.01%, 1.06%] (Width = 2.08%)\n"
    "• Deterministic Policy Call: NEUTRAL (Confidence: 50%)\n"
    "- SPY 20-session realized volatility 11.6% annualized."
)


def test_pipeline_supplied_figures_are_not_fabrications(validator):
    """Every number below was handed to the model in its own prompt.

    Reproduces the live failure observed against a real backend: 50% is the policy
    confidence, 2.08% the interval width, 11.6% the realized volatility from
    MarketContext.factual_notes(). Scoring these as invention sent every report to
    ABSTAIN.
    """
    draft = (
        "Stance: NEUTRAL\n"
        "Median 5-day return of 0.08% with an 80% interval of [-1.01%, 1.06%].\n"
        "Interval width 2.08%. Policy confidence 50%.\n"
        "20-session realized volatility 11.6% annualized."
    )
    passed, violations = validator.validate(
        draft, {**GROUND_TRUTH, "supplied_context": SUPPLIED_CONTEXT}
    )
    assert passed, violations


def test_supplied_context_does_not_excuse_invention(validator):
    """Widening the accepted set must not disarm the check: a figure absent from
    the prompt is still a violation."""
    draft = "Stance: NEUTRAL\nWe expect a median 5-day return of 12.40%."
    passed, violations = validator.validate(
        draft, {**GROUND_TRUTH, "supplied_context": SUPPLIED_CONTEXT}
    )
    assert not passed
    assert any("12.40" in v for v in violations)


def test_absent_supplied_context_still_validates_the_quantiles(validator):
    """The key is optional; callers that omit it keep the old, narrower behaviour."""
    draft = "Stance: NEUTRAL\nInterval width 2.08%."
    passed, _ = validator.validate(draft, GROUND_TRUTH)
    assert not passed

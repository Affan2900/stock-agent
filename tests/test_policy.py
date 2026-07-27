"""The uncertainty gate is what makes abstention a first-class outcome rather
than a NEUTRAL call in disguise, so its trigger conditions are pinned here."""

import pytest

from agents.policy import UncertaintyGatedPolicy


@pytest.fixture
def policy():
    return UncertaintyGatedPolicy()


def test_fallback_forecaster_forces_abstention(policy):
    """A demoted model must never produce a directional call, however clean the
    numbers look."""
    d = policy.evaluate(median_return=0.05, interval_width=0.02, is_fallback=True)
    assert d.stance == "ABSTAIN"
    assert d.confidence == 0.0


def test_stale_data_forces_abstention(policy):
    d = policy.evaluate(median_return=0.05, interval_width=0.02, data_freshness=False)
    assert d.stance == "ABSTAIN"


def test_degraded_coverage_forces_abstention(policy):
    """Coverage below the floor means the intervals no longer mean what they say."""
    d = policy.evaluate(median_return=0.05, interval_width=0.02, coverage_health=0.60)
    assert d.stance == "ABSTAIN"
    assert "coverage" in d.reason.lower()


def test_coverage_just_above_floor_does_not_abstain(policy):
    """Guards the boundary: 0.75 is the threshold, not a failure."""
    d = policy.evaluate(median_return=0.05, interval_width=0.02, coverage_health=0.75)
    assert d.stance != "ABSTAIN"


def test_excessive_interval_width_forces_abstention(policy):
    d = policy.evaluate(median_return=0.05, interval_width=0.20)
    assert d.stance == "ABSTAIN"
    assert "width" in d.reason.lower()


def test_grounding_retry_exhaustion_forces_abstention(policy):
    """Two failed grounding repairs must end in refusal, not a third attempt."""
    d = policy.evaluate(median_return=0.05, interval_width=0.02, grounding_retries=2)
    assert d.stance == "ABSTAIN"


def test_signal_inside_noise_band_is_neutral(policy):
    """Median well inside half the interval carries no directional information."""
    d = policy.evaluate(median_return=0.0008, interval_width=0.0207)
    assert d.stance == "NEUTRAL"
    assert d.confidence == 0.50


@pytest.mark.parametrize(
    "median_return,expected",
    [(0.04, "BULLISH"), (-0.04, "BEARISH")],
)
def test_signal_outside_noise_band_takes_a_side(policy, median_return, expected):
    d = policy.evaluate(median_return=median_return, interval_width=0.02)
    assert d.stance == expected
    assert d.confidence > 0.50


def test_llm_can_never_override_the_policy(policy):
    """The stance is computed, not negotiated; the generator only narrates it."""
    d = policy.evaluate(median_return=0.04, interval_width=0.02)
    assert d.allow_llm_override is False

"""MarketContext.factual_notes() replaced fabricated news headlines as the agent's
market-context input. Its whole justification is that every line is arithmetic on
observed bars, so the arithmetic is checked here. Constructed directly — no
network, no yfinance."""

import numpy as np
import pytest

from forecasting.serve.features import MarketContext


def make_context(closes, ticker="SPY", as_of="2026-07-24"):
    closes = np.asarray(closes, dtype=float)
    return MarketContext(
        ticker=ticker,
        features_x=np.zeros((len(closes), 3)),
        current_price=float(closes[-1]),
        as_of=as_of,
        feature_cols=["a", "b", "c"],
        recent_closes=closes,
    )


def test_first_note_reports_the_real_last_close():
    ctx = make_context([100.0] * 29 + [738.93])
    assert ctx.factual_notes()[0] == "SPY last close 738.93 on 2026-07-24."


def test_session_changes_are_exact():
    """A flat series with a known step: 5-session change must be arithmetic, not
    an estimate."""
    closes = [100.0] * 25 + [110.0]
    notes = " ".join(make_context(closes).factual_notes())
    assert "5-session price change +10.00%." in notes


def test_negative_change_keeps_its_sign():
    closes = [100.0] * 25 + [95.0]
    notes = " ".join(make_context(closes).factual_notes())
    assert "5-session price change -5.00%." in notes


def test_volatility_of_a_flat_series_is_zero():
    notes = " ".join(make_context([100.0] * 30).factual_notes())
    assert "20-session realized volatility 0.0% annualized." in notes


def test_windows_are_omitted_when_history_is_too_short():
    """Better to say less than to compute a change over bars that do not exist."""
    notes = make_context([100.0, 101.0, 102.0]).factual_notes()
    assert len(notes) == 1
    assert not any("session" in n for n in notes)


def test_every_note_is_traceable_to_the_close_window():
    """Guards the property the grounding critic depends on: no note may introduce
    a claim that is not arithmetic on observed closes."""
    rng = np.random.default_rng(0)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 60))
    notes = make_context(closes).factual_notes()
    assert len(notes) == 4
    assert all(n.endswith(".") for n in notes)
    assert f"{closes[-1]:.2f}" in notes[0]


def test_to_meta_reports_real_shape():
    ctx = make_context([100.0] * 30)
    meta = ctx.to_meta()
    assert meta["lookback"] == 30
    assert meta["n_features"] == 3
    assert meta["ticker"] == "SPY"

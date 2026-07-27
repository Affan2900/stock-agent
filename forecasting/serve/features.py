"""
Live market feature construction for the serving path.

Replaces the `np.random.normal(0, 1, (60, 15))` placeholders that previously fed
the predictor. Builds the same feature matrix the model was trained on, from a
live yfinance pull, and returns the real last close as the anchor price for
price-path reconstruction.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecasting.data.indicators import add_technical_indicators
from forecasting.data.ingestion import fetch_ticker_data

logger = logging.getLogger(__name__)

# Columns that are price levels / raw volume rather than model features.
NON_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]


DEFAULT_HISTORY_DAYS = 500


@dataclass
class MarketContext:
    """A single ticker's serving-time inputs."""

    ticker: str
    features_x: np.ndarray  # (lookback, n_features)
    current_price: float
    as_of: str  # ISO date of the last observed bar
    feature_cols: List[str]
    recent_closes: np.ndarray  # trailing closes over the same window

    def to_meta(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "current_price": self.current_price,
            "n_features": len(self.feature_cols),
            "lookback": int(self.features_x.shape[0]),
        }

    def factual_notes(self) -> List[str]:
        """
        Descriptive statements computed from the observed closes.

        These feed the agent's market-context section. Every line is arithmetic on
        real bars — no sentiment, no headlines, nothing the critic cannot trace
        back to a number in this window.
        """
        closes = np.asarray(self.recent_closes, dtype=float)
        notes = [
            f"{self.ticker} last close {closes[-1]:.2f} on {self.as_of}."
        ]
        for window, label in ((5, "5-session"), (20, "20-session")):
            if len(closes) > window:
                pct = (closes[-1] / closes[-1 - window] - 1.0) * 100.0
                notes.append(f"{label} price change {pct:+.2f}%.")
        if len(closes) > 21:
            rets = np.diff(np.log(closes[-21:]))
            vol = float(np.std(rets, ddof=1) * np.sqrt(252) * 100.0)
            notes.append(f"20-session realized volatility {vol:.1f}% annualized.")
        return notes


class MarketFeatureProvider:
    """
    Fetches live OHLCV, computes indicators, and slices the trailing `lookback`
    window into the model's input matrix.

    Results are cached in-process with a TTL so that a burst of requests for the
    same ticker triggers one yfinance call, not one per request.
    """

    def __init__(
        self,
        lookback: int = 60,
        ttl_seconds: int = 900,
        history_days: int = DEFAULT_HISTORY_DAYS,
        feature_cols: Optional[List[str]] = None,
    ):
        self.lookback = lookback
        self.ttl_seconds = ttl_seconds
        self.history_days = history_days
        self.feature_cols = feature_cols
        self._cache: Dict[str, Tuple[float, MarketContext]] = {}
        self._lock = threading.Lock()

    def _resolve_feature_cols(self, df: pd.DataFrame) -> List[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Live features missing columns the model was trained on: {missing}"
                )
            return list(self.feature_cols)
        return [c for c in df.columns if c not in NON_FEATURE_COLS]

    def _build(self, ticker: str) -> MarketContext:
        start = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=self.history_days)).strftime(
            "%Y-%m-%d"
        )
        df = fetch_ticker_data(ticker, start_date=start)
        df_ind = add_technical_indicators(df)

        if len(df_ind) < self.lookback:
            raise ValueError(
                f"Only {len(df_ind)} usable rows for '{ticker}' after indicator warmup; "
                f"need at least {self.lookback}."
            )

        cols = self._resolve_feature_cols(df_ind)
        window = df_ind.iloc[-self.lookback :]

        features_x = window[cols].values.astype(np.float32)
        current_price = float(df_ind["Close"].iloc[-1])
        as_of = pd.Timestamp(df_ind.index[-1]).strftime("%Y-%m-%d")

        if not np.isfinite(features_x).all():
            raise ValueError(f"Non-finite values in live feature matrix for '{ticker}'.")

        logger.info(
            "Built live context for %s: %s rows x %s features, last close %.2f as of %s",
            ticker,
            features_x.shape[0],
            features_x.shape[1],
            current_price,
            as_of,
        )
        return MarketContext(
            ticker=ticker,
            features_x=features_x,
            current_price=current_price,
            as_of=as_of,
            feature_cols=cols,
            recent_closes=window["Close"].values.astype(np.float64),
        )

    def get_context(self, ticker: str, force_refresh: bool = False) -> MarketContext:
        """
        Return the live feature window and last close for `ticker`.

        Raises on fetch failure rather than silently substituting synthetic data —
        a stale or missing quote must surface as an error the caller can turn into
        an abstention, not as a plausible-looking forecast.
        """
        key = ticker.upper()
        now = time.time()

        with self._lock:
            hit = self._cache.get(key)
            if hit and not force_refresh and (now - hit[0]) < self.ttl_seconds:
                return hit[1]

        ctx = self._build(key)

        with self._lock:
            self._cache[key] = (time.time(), ctx)
        return ctx

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from forecasting.serve.bundle import DEFAULT_BUNDLE_PATH, try_load_model_bundle
from forecasting.serve.features import MarketContext, MarketFeatureProvider
from agents.graph import GroundedAgentGraph, AgentState
from api.metrics_exporter import metrics_exporter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Calibrated Stock Forecaster & Grounded Agent API",
    description="Production-shaped MLOps API delivering conformally-calibrated prediction intervals and numerically grounded agent reports.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", DEFAULT_BUNDLE_PATH)

# A missing or unpromoted bundle yields a predictor that serves the fallback
predictor, bundle_meta = try_load_model_bundle(BUNDLE_PATH)

feature_provider = MarketFeatureProvider(
    lookback=(bundle_meta or {}).get("provenance", {}).get("lookback", 60),
    feature_cols=(bundle_meta or {}).get("feature_cols"),
)

agent_graph = GroundedAgentGraph()

_MEASURED_COVERAGE = float(
    (bundle_meta or {}).get("metrics", {}).get("empirical_coverage_80", 0.80)
)

if bundle_meta is not None and "empirical_coverage_80" in bundle_meta["metrics"]:
    metrics_exporter.update_forecast(_MEASURED_COVERAGE)


def _load_context(ticker: str) -> MarketContext:
    """Fetch live features, or fail loudly — never substitute synthetic inputs."""
    try:
        return feature_provider.get_context(ticker)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as 503
        logger.error("Market data unavailable for %s: %s", ticker, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Market data unavailable for '{ticker.upper()}': {exc}",
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0", "model_promoted": predictor.is_promoted}


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """
    Provenance for whatever is actually loaded: gate verdict, held-out metrics,
    training window, and feature schema.
    """
    if bundle_meta is None:
        return {
            "loaded": False,
            "promoted": False,
            "bundle_path": BUNDLE_PATH,
            "note": "No model bundle found; serving the fallback forecaster.",
        }
    return {
        "loaded": True,
        "promoted": predictor.is_promoted,
        "bundle_path": BUNDLE_PATH,
        "gate": bundle_meta["gate"],
        "metrics": bundle_meta["metrics"],
        "provenance": bundle_meta["provenance"],
        "model_config": bundle_meta["model_config"],
        "n_features": len(bundle_meta["feature_cols"]),
    }


@app.get("/forecast/{ticker}")
def get_forecast(ticker: str, current_price: Optional[float] = None):
    """
    Return 5-day horizon log return forecast and conformally-calibrated prediction
    intervals, built from a live market data pull.

    `current_price` overrides the anchor used for price-path reconstruction; it
    defaults to the last observed close.
    """
    ctx = _load_context(ticker)
    res = predictor.predict(
        ticker=ctx.ticker,
        features_x=ctx.features_x,
        current_price=current_price if current_price is not None else ctx.current_price,
    )
    res["market_context"] = ctx.to_meta()
    return res


@app.post("/report/{ticker}")
def generate_report(ticker: str, current_price: Optional[float] = None):
    """
    Generate grounded agent narrative report with uncertainty-gated stance policy.
    """
    ctx = _load_context(ticker)
    anchor_price = current_price if current_price is not None else ctx.current_price

    fc = predictor.predict(
        ticker=ctx.ticker,
        features_x=ctx.features_x,
        current_price=anchor_price,
    )

    q_median = fc["quantiles"]["0.50"]
    q_lower = fc["quantiles"]["0.10"]
    q_upper = fc["quantiles"]["0.90"]

    median_return = float(q_median[0]) if len(q_median) > 0 else 0.0
    lower_return = float(q_lower[0]) if len(q_lower) > 0 else 0.0
    upper_return = float(q_upper[0]) if len(q_upper) > 0 else 0.0
    interval_width = upper_return - lower_return

    state: AgentState = {
        "ticker": ctx.ticker,
        "current_price": anchor_price,
        "median_return": median_return,
        "lower_return": lower_return,
        "upper_return": upper_return,
        "interval_width": interval_width,
        "coverage_health": _MEASURED_COVERAGE,
        "data_freshness": True,
        "is_fallback": fc.get("fallback", False),
        # Arithmetic on the observed window, not sourced headlines. The critic can
        # trace each line back to a real bar.
        "news_headlines": ctx.factual_notes(),
    }

    output = agent_graph.run(state)

    metrics_exporter.record_report_request(
        stance=output.get("policy_stance", "NEUTRAL"),
        retries=output.get("retries", 0),
        grounding_passed=output.get("grounding_passed", True)
    )

    return {
        "ticker": ctx.ticker,
        "current_price": anchor_price,
        "as_of": ctx.as_of,
        "stance": output.get("policy_stance"),
        "confidence": output.get("policy_confidence"),
        "reason": output.get("policy_reason"),
        "fallback": fc.get("fallback", False),
        "llm_backend": getattr(agent_graph.llm, "last_backend", "unknown"),
        "forecast": fc,
        "report": output.get("final_report")
    }


@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    """
    Expose Prometheus metrics endpoint.
    """
    return metrics_exporter.generate_prometheus_str()

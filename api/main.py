from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, List, Optional
import numpy as np

from forecasting.serve.predictor import ProductionPredictor
from agents.graph import GroundedAgentGraph, AgentState
from api.metrics_exporter import metrics_exporter

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

predictor = ProductionPredictor()
agent_graph = GroundedAgentGraph()

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/forecast/{ticker}")
def get_forecast(ticker: str = "AAPL", current_price: float = 180.0):
    """
    Return 5-day horizon log return forecast and conformally-calibrated prediction intervals.
    """
    # Sample features for demonstration
    dummy_features = np.random.normal(0, 1, (60, 15))
    res = predictor.predict(
        ticker=ticker.upper(),
        features_x=dummy_features,
        current_price=current_price
    )
    return res

@app.post("/report/{ticker}")
def generate_report(ticker: str = "AAPL", current_price: float = 180.0):
    """
    Generate grounded agent narrative report with uncertainty-gated stance policy.
    """
    ticker = ticker.upper()
    dummy_features = np.random.normal(0, 1, (60, 15))
    fc = predictor.predict(
        ticker=ticker,
        features_x=dummy_features,
        current_price=current_price
    )
    
    q_median = fc["quantiles"]["0.50"]
    q_lower = fc["quantiles"]["0.10"]
    q_upper = fc["quantiles"]["0.90"]
    
    median_return = float(q_median[0]) if len(q_median) > 0 else 0.01
    lower_return = float(q_lower[0]) if len(q_lower) > 0 else -0.01
    upper_return = float(q_upper[0]) if len(q_upper) > 0 else 0.03
    interval_width = upper_return - lower_return
    
    state: AgentState = {
        "ticker": ticker,
        "current_price": current_price,
        "median_return": median_return,
        "lower_return": lower_return,
        "upper_return": upper_return,
        "interval_width": interval_width,
        "coverage_health": 0.80,
        "data_freshness": True,
        "is_fallback": fc.get("fallback", False),
        "news_headlines": [
            f"Analyst consensus remains focused on {ticker} growth trajectory.",
            f"Trading volume steady across {ticker} sector."
        ]
    }
    
    output = agent_graph.run(state)
    
    metrics_exporter.record_report_request(
        stance=output.get("policy_stance", "NEUTRAL"),
        retries=output.get("retries", 0),
        grounding_passed=output.get("grounding_passed", True)
    )
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "stance": output.get("policy_stance"),
        "confidence": output.get("policy_confidence"),
        "reason": output.get("policy_reason"),
        "fallback": fc.get("fallback", False),
        "forecast": fc,
        "report": output.get("final_report")
    }

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    """
    Expose Prometheus metrics endpoint.
    """
    return metrics_exporter.generate_prometheus_str()

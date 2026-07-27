import os
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Page Configuration
st.set_page_config(
    page_title="Calibrated Stock Forecaster & Grounded Agent",
    page_icon="📈",
    layout="wide"
)


# Title & Framing
st.title("📈 Calibrated Stock Forecaster & Grounded Agent")
st.markdown(
    "**Probabilistic Time-Series Forecaster with Conformal Calibration & Numerically Grounded Agent Layer**"
)

# Sidebar Inputs
st.sidebar.header("Forecast Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
override_price = st.sidebar.checkbox("Override anchor price", value=False)
manual_price = st.sidebar.number_input(
    "Anchor Price ($)", value=180.0, min_value=1.0, disabled=not override_price
)

# Import local backend directly for Streamlit execution
from forecasting.serve.bundle import DEFAULT_BUNDLE_PATH, try_load_model_bundle
from forecasting.serve.features import MarketFeatureProvider
from agents.graph import GroundedAgentGraph, AgentState

BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", DEFAULT_BUNDLE_PATH)


@st.cache_resource
def get_backend():
    predictor, meta = try_load_model_bundle(BUNDLE_PATH)
    provider = MarketFeatureProvider(
        lookback=(meta or {}).get("provenance", {}).get("lookback", 60),
        feature_cols=(meta or {}).get("feature_cols"),
    )
    return predictor, meta, provider, GroundedAgentGraph()


predictor, bundle_meta, feature_provider, agent_graph = get_backend()

# Show what is actually loaded, so the numbers on screen are attributable.
with st.sidebar:
    st.divider()
    if bundle_meta is None:
        st.warning("No model bundle loaded — fallback forecaster only.")
    else:
        prov = bundle_meta["provenance"]
        m = bundle_meta["metrics"]
        st.caption(
            f"Model: {'PROMOTED' if predictor.is_promoted else 'NOT PROMOTED'} · "
            f"trained on {prov['ticker']} {prov['data_start']}→{prov['data_end']}"
        )
        st.caption(
            f"Held-out MASE {m['mase']:.3f} · "
            f"80% coverage {m['empirical_coverage_80']*100:.1f}%"
        )

if st.sidebar.button("Generate Forecast & Report", type="primary"):
    with st.spinner(f"Fetching market data and computing forecast for {ticker}..."):
        try:
            ctx = feature_provider.get_context(ticker)
        except Exception as exc:
            st.error(f"Market data unavailable for {ticker}: {exc}")
            st.stop()

        current_price = manual_price if override_price else ctx.current_price
        fc = predictor.predict(
            ticker=ctx.ticker, features_x=ctx.features_x, current_price=current_price
        )

        q_median = fc["quantiles"]["0.50"]
        q_lower = fc["quantiles"]["0.10"]
        q_upper = fc["quantiles"]["0.90"]
        reconstructed = fc["reconstructed_prices"]
        
        median_return = float(q_median[0])
        lower_return = float(q_lower[0])
        upper_return = float(q_upper[0])
        interval_width = upper_return - lower_return
        
        state: AgentState = {
            "ticker": ctx.ticker,
            "current_price": current_price,
            "median_return": median_return,
            "lower_return": lower_return,
            "upper_return": upper_return,
            "interval_width": interval_width,
            # Coverage the model actually achieved on its held-out fold.
            "coverage_health": float(
                (bundle_meta or {}).get("metrics", {}).get("empirical_coverage_80", 0.80)
            ),
            "data_freshness": True,
            "is_fallback": fc.get("fallback", False),
            # Arithmetic on the observed window — no sourced headlines, nothing the
            # grounding critic cannot trace back to a real bar.
            "news_headlines": ctx.factual_notes(),
        }

        agent_out = agent_graph.run(state)

    st.caption(
        f"Live data: {ctx.ticker} last close **${ctx.current_price:,.2f}** as of {ctx.as_of} "
        f"· {ctx.features_x.shape[0]}×{ctx.features_x.shape[1]} feature window"
    )

    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stance = agent_out.get("policy_stance", "NEUTRAL")
    confidence = agent_out.get("policy_confidence", 0.5)
    
    with col1:
        if stance == "BULLISH":
            st.success(f"### Stance: {stance}")
        elif stance == "BEARISH":
            st.error(f"### Stance: {stance}")
        elif stance == "NEUTRAL":
            st.info(f"### Stance: {stance}")
        else:
            st.warning(f"### Stance: {stance}")
            
    with col2:
        st.metric("Policy Confidence", f"{confidence*100:.0f}%")
        
    with col3:
        st.metric("5-Day Median Return", f"{median_return*100:+.2f}%")
        
    with col4:
        st.metric("80% Conformal Interval Width", f"{interval_width*100:.2f}%")
        
    if fc.get("fallback"):
        st.warning("⚠️ Served by Baseline Fallback Forecaster due to promotion gate rules.")
        
    st.divider()
    
    # 2. Interactive Chart (Reconstructed Price Path & Interval)
    st.subheader("5-Trading-Day Price Forecast & Conformal Band")
    
    days = [f"Day {i}" for i in range(1, 6)]
    
    # Calculate price bands
    lower_prices = current_price * np.exp(np.cumsum(q_lower))
    upper_prices = current_price * np.exp(np.cumsum(q_upper))
    
    df_chart = pd.DataFrame({
        "Day": days,
        "Median Forecast": reconstructed,
        "Lower 80% Bound": lower_prices,
        "Upper 80% Bound": upper_prices
    }).set_index("Day")
    
    st.line_chart(df_chart)
    
    # 3. Agent Report Box
    st.subheader("Grounded Agent Narrative Report")
    st.info(f"**Policy Rationale:** {agent_out.get('policy_reason')}")
    st.markdown(agent_out.get("final_report", ""))

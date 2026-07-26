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
current_price = st.sidebar.number_input("Current Price ($)", value=180.0, min_value=1.0)

# Import local backend directly for Streamlit execution
from forecasting.serve.predictor import ProductionPredictor
from agents.graph import GroundedAgentGraph, AgentState

@st.cache_resource
def get_predictor_and_graph():
    return ProductionPredictor(), GroundedAgentGraph()

predictor, agent_graph = get_predictor_and_graph()

if st.sidebar.button("Generate Forecast & Report", type="primary"):
    with st.spinner(f"Computing 5-day calibrated forecast and agent report for {ticker}..."):
        dummy_features = np.random.normal(0, 1, (60, 15))
        fc = predictor.predict(ticker=ticker, features_x=dummy_features, current_price=current_price)
        
        q_median = fc["quantiles"]["0.50"]
        q_lower = fc["quantiles"]["0.10"]
        q_upper = fc["quantiles"]["0.90"]
        reconstructed = fc["reconstructed_prices"]
        
        median_return = float(q_median[0])
        lower_return = float(q_lower[0])
        upper_return = float(q_upper[0])
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
                f"Market consensus focuses on {ticker} Q3 growth drivers.",
                f"Trading volume steady across {ticker} sector."
            ]
        }
        
        agent_out = agent_graph.run(state)
        
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

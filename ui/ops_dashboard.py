import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Ops & Drift Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=30)
def fetch_model_info():
    """Gate verdict, held-out metrics, and provenance for the loaded bundle."""
    with urllib.request.urlopen(f"{API_URL}/model/info", timeout=20) as r:
        return json.loads(r.read().decode())


@st.cache_data(ttl=30)
def fetch_runtime_metrics():
    """Parse the Prometheus exposition text into a flat name -> float map."""
    with urllib.request.urlopen(f"{API_URL}/metrics", timeout=20) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


st.title("🛡️ Ops & MLOps Monitoring Dashboard")
st.markdown(
    "**Model Performance, Conformal Coverage, Grounding Violations & Abstention Rates** — "
    f"read live from `{API_URL}`."
)

try:
    info = fetch_model_info()
    runtime = fetch_runtime_metrics()
except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
    st.error(f"Cannot reach the API at {API_URL}: {exc}")
    st.info("Every panel on this page is served by the API. Nothing is rendered without it.")
    st.stop()

if not info.get("loaded"):
    st.error("No model bundle is loaded — the API is serving the fallback forecaster.")
    st.caption(info.get("note", ""))
    st.stop()

metrics = info["metrics"]
gate = info["gate"]
prov = info["provenance"]

# Request-count context. The rates below are ratios over this denominator; with a
# denominator of zero they are undefined, not "healthy".
n_requests = int(runtime.get("report_requests_total", 0))

col1, col2, col3, col4 = st.columns(4)

with col1:
    cov = metrics["empirical_coverage_80"]
    st.metric(
        "Empirical 80% Coverage",
        f"{cov*100:.1f}%",
        delta=f"{(cov - 0.80)*100:+.1f}% vs nominal",
        delta_color="off",
    )

with col2:
    mase = metrics["mase"]
    st.metric(
        "MASE vs Random Walk",
        f"{mase:.4f}",
        delta=f"{mase - 1.0:+.4f} vs RW",
        delta_color="inverse",
    )

with col3:
    if n_requests == 0:
        st.metric("Grounding Violation Rate", "n/a", delta="no requests yet", delta_color="off")
    else:
        gvr = runtime.get("grounding_violation_rate", 0.0)
        st.metric(
            "Grounding Violation Rate",
            f"{gvr*100:.1f}%",
            delta=f"{int(runtime.get('grounding_violations_total', 0))} of {n_requests}",
            delta_color="off",
        )

with col4:
    if n_requests == 0:
        st.metric("Abstention Rate", "n/a", delta="no requests yet", delta_color="off")
    else:
        ar = runtime.get("abstention_rate", 0.0)
        st.metric(
            "Abstention Rate",
            f"{ar*100:.1f}%",
            delta=f"{int(runtime.get('report_abstentions_total', 0))} of {n_requests}",
            delta_color="off",
        )

if n_requests == 0:
    st.caption(
        "⚠️ No reports served since this API process started, so the grounding and "
        "abstention rates have no denominator. Generate a report to populate them."
    )

st.divider()

st.subheader("1. Model Promotion Gate — Audit")
st.dataframe(
    pd.DataFrame([{
        "Model": f"{prov['ticker']}_lstm_q_h{prov['horizon']}",
        "Trained": prov["trained_at"],
        "MASE": round(metrics["mase"], 4),
        "Coverage (80%)": f"{metrics['empirical_coverage_80']*100:.1f}%",
        "p95 Latency": f"{metrics.get('p95_latency_ms', float('nan')):.1f}ms",
        "Decision": "APPROVED (Production)" if gate["passed"] else "REJECTED (Fallback Active)",
    }]),
    use_container_width=True,
    hide_index=True,
)
if gate["reasons"]:
    for reason in gate["reasons"]:
        st.warning(reason)

st.subheader("2. Held-Out Test Metrics")
st.dataframe(
    pd.DataFrame(
        [{"Metric": k, "Value": round(v, 6)} for k, v in sorted(metrics.items())]
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("3. Training Provenance")
c1, c2 = st.columns(2)
with c1:
    st.write({
        "ticker": prov["ticker"],
        "data_start": prov["data_start"],
        "data_end": prov["data_end"],
        "n_samples": prov["n_samples"],
    })
with c2:
    st.write({
        "lookback": prov["lookback"],
        "horizon": prov["horizon"],
        "n_features": info["n_features"],
        "seed": prov["seed"],
    })

st.divider()
st.caption(
    "Metrics above are the deployed model's held-out test-fold results, recorded at "
    "training time and shipped inside the bundle. Runtime rates are in-process counters "
    "that reset on pod restart and are scraped from a single API replica.\n\n"
    "A conformal-coverage-over-time chart is deliberately absent: nothing in this system "
    "persists realized coverage at serving time, so there is no series to plot."
)

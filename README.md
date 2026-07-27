# Calibrated Forecasting + Grounded Agent MLOps System on AWS

> Built on the architecture of modern time-series forecasting reference systems, with the AWS deployment layer reproduced deliberately. The forecasting and agent layers are rebuilt: point forecasts replaced with **conformally-calibrated prediction intervals**, leaky random splits replaced with **purged walk-forward cross-validation** against explicit baselines, and the LLM critic replaced with a **deterministic grounding validator** plus an **uncertainty-gated stance policy** that can abstain.

---

## 🏛 Architecture Diagram

```
                 ┌──────────────┐        ┌──────────────────┐
                 │ Streamlit UI │        │  Ops Dashboard   │
                 └──────┬───────┘        └────────┬─────────┘
                        └──────────┬──────────────┘
                            ┌──────▼───────┐
                            │   FastAPI    │  rate limits, async jobs
                            └──┬────┬───┬──┘
              ┌────────────────┘    │   └────────────────┐
    ┌─────────▼─────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │ Forecast Service  │  │  Agent Service  │  │  Ops / Drift    │
    │ • LSTM+quantile   │  │ • LangGraph     │  │ • Evidently     │
    │ • conformal cal.  │  │ • grounding     │  │ • coverage      │
    │ • promotion gate  │  │ • stance policy │  │ • retrain trig. │
    │ • baseline fb.    │  │ • abstention    │  │                 │
    └────┬────────┬─────┘  └───┬─────────┬───┘  └────────┬────────┘
         │        │            │         │               │
     ┌───▼──┐ ┌───▼────┐   ┌───▼───┐ ┌───▼────┐    ┌─────▼──────┐
     │Redis │ │ MLflow │   │Bedrock│ │ Qdrant │    │ Prometheus │
     └──────┘ └────────┘   └───────┘ └────────┘    │  + Grafana │
                                                    └────────────┘
```

---

## ✨ Key Architectural Innovations over Reference Design

| Component | Reference Implementation Defect | Our Production Solution |
|---|---|---|
| **Multi-Step Horizon** | Autoregressive 1-step rollout (error compounds across horizon) | **Direct Multi-Horizon Head** `(H=5, Q=3)` emitting all steps in 1 forward pass |
| **Target Variable** | Raw close price levels ($R^2=0.98$ lag-1 copy, not skill) | Stationary **log returns** $r_t = \log(C_t / C_{t-1})$; prices reconstructed |
| **Uncertainty** | Uncalibrated point forecasts (no prediction interval) | **Split Conformal Calibration** & **ACI online tracking** for nominal 80% coverage |
| **Backtesting** | Leaky `train_test_split(shuffle=False)` without purge gaps | **Purged & Embargoed Walk-Forward CV** vs Random Walk & ARIMA baselines |
| **Agent Critic** | LLM checking another LLM (no ground truth, cannot reject) | **Deterministic Grounding Validator** parsing claims against forecast tensors |
| **Stance Policy** | LLM self-grading string matching ("BULLISH") | **Uncertainty-Gated Stance Policy** derived mathematically from interval width |
| **Deployment Gate** | Schema check disguised as evaluation | **Required CI Eval Gate** (`evals/run_evals.py`) blocking regressed PRs |
| **AWS Security** | Static IAM user access keys in CI/CD | **GitHub Actions OIDC Federation** + **IRSA** for pod Bedrock access |

---

## 🚀 Quick Start (Local Execution)

### 1. Run CI Agent Evaluation Suite
```bash
python evals/run_evals.py
```

### 2. Launch Local Docker Compose Stack
```bash
docker compose up --build
```
Access points:
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **User Financial Report UI**: [http://localhost:8501](http://localhost:8501)
- **Ops & MLOps Dashboard**: [http://localhost:8502](http://localhost:8502)

### 3. Run Individual Phase Scripts
```bash
# Phase 1: Baseline Backtesting (Random Walk, Seasonal Naive, ARIMA)
python scripts/run_phase1_baselines.py

# Phase 2: PyTorch Quantile LSTM & Transfer Learning
python scripts/run_phase2_training.py

# Phase 3: Conformal Calibration & Model Promotion Gate
python scripts/run_phase3_calibration_gate.py

# Phase 4: Multi-Agent Grounded Reporting Demonstration
python scripts/run_phase4_agent_demo.py
```

---



### Safety Rules:
1. **$10 Billing Alarm**: Configured in [infra/terraform/main.tf](file:///e:/Projects/stock-agent/infra/terraform/main.tf#L26) before any resource is applied.
2. **One-Command Teardown Verification**:
   ```bash
   # Tear down Terraform infrastructure
   bash scripts/nuke_aws.sh

   # Audit AWS CLI to confirm 0 surviving billable resources
   bash scripts/verify_teardown.sh
   ```

---

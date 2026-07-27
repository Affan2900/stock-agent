"""
Train a quantile LSTM on real market data, calibrate it, run the promotion gate,
and export a serving bundle.

The gate verdict is recorded as-is. If the model does not beat the random-walk
baseline the bundle is still written, marked unpromoted, and the API will serve
the fallback forecaster and label the response `fallback: true`. That is the
designed behaviour (plan.md §3.5), not a failure of this script.

Usage:
    python scripts/train_and_export.py --ticker SPY --epochs 40
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.calibration.conformal import SplitConformalCalibrator  # noqa: E402
from forecasting.data.dataset import create_forecasting_dataset  # noqa: E402
from forecasting.data.indicators import add_technical_indicators  # noqa: E402
from forecasting.data.ingestion import fetch_ticker_data  # noqa: E402
from forecasting.eval.metrics import evaluate_forecast_metrics  # noqa: E402
from forecasting.eval.promotion_gate import ModelPromotionGate  # noqa: E402
from forecasting.models.lstm import QuantileLSTMForecaster  # noqa: E402
from forecasting.models.trainer import create_dataloader, train_quantile_lstm  # noqa: E402
from forecasting.serve.bundle import save_model_bundle  # noqa: E402
from forecasting.serve.features import NON_FEATURE_COLS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_and_export")

LOOKBACK = 60
HORIZON = 5


def chronological_split(n: int, train_frac: float, cal_frac: float):
    """Index bounds for train / calibration / test, strictly in time order."""
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    return slice(0, n_train), slice(n_train, n_train + n_cal), slice(n_train + n_cal, n)


def raw_quantiles(model, X, device="cpu"):
    """Uncalibrated (lower, median, upper) arrays of shape (N, H)."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return out[:, :, 0], out[:, :, 1], out[:, :, 2]


def measure_latencies(model, X, n: int = 60, device: str = "cpu"):
    """Single-sample inference latencies in ms — what the gate budgets against."""
    model.eval()
    lats = []
    with torch.no_grad():
        for i in range(min(n, len(X))):
            x = torch.tensor(X[i : i + 1], dtype=torch.float32).to(device)
            t0 = time.perf_counter()
            model(x)
            lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "model_bundle.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Fetching %s from %s", args.ticker, args.start)
    df = fetch_ticker_data(args.ticker, start_date=args.start)
    df_ind = add_technical_indicators(df)
    feature_cols = [c for c in df_ind.columns if c not in NON_FEATURE_COLS]
    logger.info("%s rows, %s features: %s", len(df_ind), len(feature_cols), feature_cols)

    X, Y, dates = create_forecasting_dataset(
        df_ind, feature_cols=feature_cols, target_col="log_return",
        lookback=LOOKBACK, horizon=HORIZON,
    )
    tr, cal, te = chronological_split(len(X), 0.70, 0.15)
    logger.info(
        "Split -> train %s (%s..%s), cal %s, test %s (%s..%s)",
        len(X[tr]), dates[tr.start].date(), dates[tr.stop - 1].date(),
        len(X[cal]), len(X[te]), dates[te.start].date(), dates[te.stop - 1].date(),
    )

    model = QuantileLSTMForecaster(input_dim=len(feature_cols), horizon=HORIZON)
    model, history = train_quantile_lstm(
        model=model,
        train_loader=create_dataloader(X[tr], Y[tr], batch_size=32, shuffle=True),
        val_loader=create_dataloader(X[cal], Y[cal], batch_size=32, shuffle=False),
        epochs=args.epochs,
    )
    logger.info("Training done. Best val pinball loss: %.6f", min(history["val_loss"]))

    # Split conformal: fit offsets on the calibration fold only.
    lo_cal, _, hi_cal = raw_quantiles(model, X[cal])
    calibrator = SplitConformalCalibrator(target_coverage=0.80).calibrate(
        y_true_cal=Y[cal], q_lower_cal=lo_cal, q_upper_cal=hi_cal
    )
    logger.info("Conformal offsets per horizon: %s", np.round(calibrator.q_adjust, 5).tolist())

    # Evaluate calibrated intervals on the untouched test fold.
    lo_te, med_te, hi_te = raw_quantiles(model, X[te])
    cal_q = calibrator.predict(lo_te, med_te, hi_te)
    metrics = evaluate_forecast_metrics(
        y_true=Y[te],
        y_pred_median=cal_q[0.50],
        y_train_returns=Y[tr].ravel(),
        q_preds={0.10: cal_q[0.10], 0.50: cal_q[0.50], 0.90: cal_q[0.90]},
    )

    latencies = measure_latencies(model, X[te])
    metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))

    passed, reasons = ModelPromotionGate().evaluate_candidate(metrics, cal_q, latencies)

    print("\n" + "=" * 68)
    print(f"PROMOTION GATE: {'PASSED' if passed else 'REJECTED'}  ({args.ticker})")
    print("=" * 68)
    for k in ("mase", "empirical_coverage_80", "interval_width_80",
              "directional_accuracy", "mean_pinball_loss", "p95_latency_ms"):
        if k in metrics:
            print(f"  {k:<26} {metrics[k]:.4f}")
    for r in reasons:
        print(f"  ! {r}")
    print("=" * 68 + "\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        path=str(out),
        model=model, calibrator=calibrator, feature_cols=feature_cols,
        metrics=metrics, gate_passed=passed, gate_reasons=reasons,
        provenance={
            "ticker": args.ticker,
            "data_start": str(dates[0].date()),
            "data_end": str(dates[-1].date()),
            "n_samples": int(len(X)),
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": args.seed,
        },
    )
    print(json.dumps({"promoted": passed, "bundle": str(out),
                      "mase": round(metrics["mase"], 4),
                      "coverage_80": round(metrics["empirical_coverage_80"], 4)}, indent=2))


if __name__ == "__main__":
    main()

"""
Serialisation for a promoted model + its conformal calibrator.

The serving container has no MLflow connection, so the promotion decision made at
training time travels with the weights. `is_promoted` is read from the bundle and
never recomputed at serve time — the gate result is a training-time fact.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from forecasting.calibration.conformal import SplitConformalCalibrator
from forecasting.eval.promotion_gate import ModelPromotionGate
from forecasting.models.lstm import QuantileLSTMForecaster
from forecasting.serve.predictor import ProductionPredictor

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1
DEFAULT_BUNDLE_PATH = "artifacts/model_bundle.pt"


def save_model_bundle(
    path: str,
    model: QuantileLSTMForecaster,
    calibrator: SplitConformalCalibrator,
    feature_cols: List[str],
    metrics: Dict[str, float],
    gate_passed: bool,
    gate_reasons: List[str],
    provenance: Dict[str, Any],
) -> None:
    """Write weights, calibration offsets, feature schema, and the gate verdict."""
    if calibrator.q_adjust is None:
        raise ValueError("Refusing to save an uncalibrated calibrator.")

    payload = {
        "bundle_version": BUNDLE_VERSION,
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "horizon": model.horizon,
            "quantiles": list(model.quantiles),
        },
        "model_state": model.state_dict(),
        "calibration": {
            "q_adjust": np.asarray(calibrator.q_adjust).tolist(),
            "target_coverage": calibrator.target_coverage,
        },
        "feature_cols": list(feature_cols),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "gate": {"passed": bool(gate_passed), "reasons": list(gate_reasons)},
        "provenance": provenance,
    }
    torch.save(payload, path)
    logger.info("Wrote model bundle to %s (promoted=%s)", path, gate_passed)


def load_model_bundle(
    path: str = DEFAULT_BUNDLE_PATH,
    device: str = "cpu",
) -> Tuple[ProductionPredictor, Dict[str, Any]]:
    """
    Rebuild a ProductionPredictor from disk.

    Returns the predictor and the bundle metadata (metrics, gate verdict,
    provenance, feature schema).
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    cfg = payload["model_config"]
    model = QuantileLSTMForecaster(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        horizon=cfg["horizon"],
        quantiles=tuple(cfg["quantiles"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    calibrator = SplitConformalCalibrator(
        target_coverage=payload["calibration"]["target_coverage"]
    )
    calibrator.q_adjust = np.asarray(payload["calibration"]["q_adjust"], dtype=np.float32)

    predictor = ProductionPredictor(
        model=model,
        calibrator=calibrator,
        gate=ModelPromotionGate(),
        device=device,
    )
    # The gate already ran at training time against a held-out test fold.
    predictor.is_promoted = bool(payload["gate"]["passed"])

    meta = {
        "feature_cols": payload["feature_cols"],
        "metrics": payload["metrics"],
        "gate": payload["gate"],
        "provenance": payload["provenance"],
        "model_config": cfg,
    }
    return predictor, meta


def try_load_model_bundle(
    path: str = DEFAULT_BUNDLE_PATH,
    device: str = "cpu",
) -> Tuple[ProductionPredictor, Optional[Dict[str, Any]]]:
    """
    Load a bundle if one exists, otherwise return an unpromoted predictor.

    A missing or unreadable bundle degrades to the fallback forecaster rather than
    failing startup — same contract as a model that fails the gate.
    """
    try:
        return load_model_bundle(path, device=device)
    except FileNotFoundError:
        logger.warning("No model bundle at %s; serving fallback forecaster.", path)
    except Exception as exc:  # noqa: BLE001 - degrade rather than crash the pod
        logger.error("Failed to load model bundle at %s (%s); serving fallback.", path, exc)
    return ProductionPredictor(), None

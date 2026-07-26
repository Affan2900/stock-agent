import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class MLflowTracker:
    """
    MLflow tracking wrapper with graceful fallback if MLflow is uninitialized or offline.
    """
    
    def __init__(self, experiment_name: str = "stock-forecaster", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.active_run = None
        self._mlflow = None
        
        try:
            import mlflow
            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.enabled = True
        except Exception as e:
            logger.warning(f"MLflow initialization failed ({e}). Operating in dummy fallback mode.")
            self.enabled = False

    def start_run(self, run_name: str = "train_run") -> Any:
        if self.enabled and self._mlflow:
            try:
                self.active_run = self._mlflow.start_run(run_name=run_name)
                return self.active_run
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        return None

    def log_params(self, params: Dict[str, Any]):
        if self.enabled and self._mlflow and self.active_run:
            try:
                self._mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.enabled and self._mlflow and self.active_run:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def end_run(self):
        if self.enabled and self._mlflow and self.active_run:
            try:
                self._mlflow.end_run()
                self.active_run = None
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

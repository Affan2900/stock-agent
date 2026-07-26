import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Prometheus metrics state container with graceful dummy fallback if prometheus_client missing
class PrometheusMetrics:
    def __init__(self):
        self.coverage_ratio = 0.80
        self.grounding_violation_rate = 0.0
        self.abstention_rate = 0.0
        self.total_requests = 0
        self.total_abstentions = 0
        self.total_grounding_violations = 0
        self.agent_retries = 0

    def update_forecast(self, coverage: float):
        self.coverage_ratio = coverage

    def record_report_request(self, stance: str, retries: int, grounding_passed: bool):
        self.total_requests += 1
        self.agent_retries += retries
        if stance == "ABSTAIN":
            self.total_abstentions += 1
        if not grounding_passed:
            self.total_grounding_violations += 1
            
        self.abstention_rate = self.total_abstentions / max(1, self.total_requests)
        self.grounding_violation_rate = self.total_grounding_violations / max(1, self.total_requests)

    def generate_prometheus_str(self) -> str:
        return (
            f"# HELP forecast_coverage_ratio Empirical coverage ratio vs nominal 80%\n"
            f"# TYPE forecast_coverage_ratio gauge\n"
            f"forecast_coverage_ratio {self.coverage_ratio:.4f}\n"
            f"# HELP grounding_violation_rate Rate of draft reports failing grounding validator\n"
            f"# TYPE grounding_violation_rate gauge\n"
            f"grounding_violation_rate {self.grounding_violation_rate:.4f}\n"
            f"# HELP abstention_rate Rate of reports returning ABSTAIN\n"
            f"# TYPE abstention_rate gauge\n"
            f"abstention_rate {self.abstention_rate:.4f}\n"
            f"# HELP agent_retry_count Total grounding validator retry count\n"
            f"# TYPE agent_retry_count counter\n"
            f"agent_retry_count {self.agent_retries}\n"
        )

metrics_exporter = PrometheusMetrics()

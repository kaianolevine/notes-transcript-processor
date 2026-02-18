"""notes_processor package.

This repository is intended to be a thin orchestrator that relies on
kaiano-common-utils for Google and LLM integrations.

App-level utilities (like deterministic rendering and metrics) live here.
"""

from .metrics import MetricsLogger, RunMetrics, estimate_tokens, new_run_id

__all__ = [
    "MetricsLogger",
    "RunMetrics",
    "estimate_tokens",
    "new_run_id",
]

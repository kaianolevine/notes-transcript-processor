from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional


def estimate_tokens(text: str) -> int:
    """Rough token estimate suitable for cost/size instrumentation.

    Rule of thumb: ~4 characters per token for English-ish text.
    This is intentionally simple and deterministic.
    """
    if not text:
        return 0
    # Use max(1, ...) to avoid returning 0 for short non-empty strings.
    return max(1, int(round(len(text) / 4)))


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    stage: str

    file_id: Optional[str]
    file_name: Optional[str]

    char_count_input: int
    estimated_input_tokens: int

    model: Optional[str]
    provider: Optional[str]

    duration_s: float
    success: bool

    error: Optional[str] = None

    estimated_output_tokens: Optional[int] = None
    char_count_output: Optional[int] = None

    estimated_cost_usd: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


class MetricsLogger:
    """Lightweight metrics logger for GitHub Actions + local runs.

    - Emits a human-readable summary to stdout.
    - Also emits a single-line JSON record to stdout (machine-parsable).
    - Optionally appends JSONL to METRICS_PATH (useful for artifacts).
    """

    def __init__(self, metrics_path: Optional[str] = None):
        self.metrics_path = metrics_path or os.getenv("METRICS_PATH")

    def emit(self, metrics: RunMetrics) -> None:
        data = asdict(metrics)

        json_line = metrics.to_json()

        status = "OK" if data["success"] else "FAIL"

        print(f"METRICS | {status} | {data.get('model')} | {data.get('file_name')}")

        print(
            f"  tokens: {data.get('estimated_input_tokens')} → {data.get('estimated_output_tokens')}"
        )
        print(f"  duration: {round(data.get('duration_s', 0), 2)}s")

        if data.get("error"):
            print(f"  error: {data['error']}")

        print(f"  run: {data.get('run_id')}")

        # Machine-parsable single-line JSON (for log scrapers)
        print(f"METRICS_JSON {json_line}")

        # Optional JSONL artifact
        if self.metrics_path:
            try:
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
            except Exception as e:
                # Never fail the run due to metrics logging
                print(
                    f"METRICS_WARN failed to append metrics to {self.metrics_path}: {e}"
                )


class Timer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def new_run_id() -> str:
    return uuid.uuid4().hex

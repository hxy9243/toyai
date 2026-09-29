"""lm-evaluation-harness integration and quality score parsing."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from llm_serving.schemas import QualityConfig, QualityTask


@dataclass
class QualityMetricResult:
    task_name: str
    primary_metric_name: str
    primary_score: float
    stderr: Optional[float]
    metrics: Dict[str, Any]
    raw_results: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_results", None)
        return d


def build_lm_eval_command(
    model_id: str,
    task: QualityTask,
    base_url: str = "http://localhost:8000/v1/completions",
    output_dir: Optional[str] = None,
    max_length: Optional[int] = None,
) -> List[str]:
    """Builds lm_eval command targeting vLLM OpenAI-compatible local-completions endpoint."""
    model_args_parts = [
        f"model={model_id}",
        f"base_url={base_url}",
        "num_concurrent=8",
    ]
    if max_length is not None:
        model_args_parts.extend([f"max_length={max_length}", "truncate=true"])
    model_args = ",".join(model_args_parts)
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", task.name,
    ]
    if task.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(task.num_fewshot)])
    if task.limit is not None:
        cmd.extend(["--limit", str(task.limit)])
    if output_dir:
        cmd.extend(["--output_path", output_dir])
    if task.extra_args:
        cmd.extend(task.extra_args)
    return cmd


def parse_lm_eval_results(raw_json: Dict[str, Any]) -> List[QualityMetricResult]:
    """Extracts task metrics and primary scores from lm-evaluation-harness output."""
    results: List[QualityMetricResult] = []
    task_results = raw_json.get("results", {})

    def stderr_key(metric_key: str) -> str:
        """lm-eval inserts ``_stderr`` before a comma-qualified metric suffix."""
        if "," in metric_key:
            metric_name, metric_suffix = metric_key.split(",", 1)
            return f"{metric_name}_stderr,{metric_suffix}"
        return f"{metric_key}_stderr"

    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue

        # Determine primary metric
        primary_key = None
        primary_score = 0.0
        primary_stderr = None

        candidate_keys = [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
            "acc_norm,none",
            "acc,none",
            "acc_norm",
            "acc",
            "exact_match",
            "f1,none",
            "em,none",
            "f1",
            "em",
        ]

        for k in candidate_keys:
            if k in metrics:
                primary_key = k
                try:
                    primary_score = float(metrics[k])
                except (ValueError, TypeError):
                    primary_score = 0.0
                metric_stderr_key = stderr_key(k)
                if metric_stderr_key in metrics:
                    try:
                        primary_stderr = float(metrics[metric_stderr_key])
                    except (ValueError, TypeError):
                        pass
                break

        if primary_key is None:
            # Fall back to the first score-like numeric value.  lm-eval also
            # records metadata such as sample_len, which must never be
            # reported as an evaluation score.
            for k, v in metrics.items():
                if (
                    isinstance(v, (int, float))
                    and not k.endswith("_stderr")
                    and k not in {"sample_len", "version", "n-shot"}
                ):
                    primary_key = k
                    primary_score = float(v)
                    metric_stderr_key = stderr_key(k)
                    if metric_stderr_key in metrics:
                        primary_stderr = float(metrics[metric_stderr_key])
                    break

        if primary_key is None:
            primary_key = "score"
            primary_score = 0.0

        results.append(
            QualityMetricResult(
                task_name=task_name,
                primary_metric_name=primary_key,
                primary_score=primary_score,
                stderr=primary_stderr,
                metrics=metrics,
                raw_results=raw_json,
            )
        )

    return results

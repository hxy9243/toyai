"""Read and visualize saved experiment artifacts without modifying them.

The runner's ``manifest.json`` and ``summary.json`` remain the canonical source.
This module deliberately tolerates older, failed, and partially written runs so
notebooks can show what is known without manufacturing values for missing data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PERFORMANCE_METRICS = {
    "output_throughput_tok_per_s": ("Output throughput", "tokens/s", True),
    "request_throughput_req_per_s": ("Request throughput", "requests/s", True),
    "ttft_p50_ms": ("TTFT p50", "ms", False),
    "ttft_p95_ms": ("TTFT p95", "ms", False),
    "itl_p50_ms": ("ITL p50", "ms", False),
    "itl_p95_ms": ("ITL p95", "ms", False),
    "e2e_p50_ms": ("End-to-end p50", "ms", False),
}


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"invalid {path.name}: expected a JSON object"
    return value, None


def _read_profile_quality(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = run_dir / "profile.yaml"
    if not path.exists():
        return None, None
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"unreadable profile.yaml: {exc}"
    return value if isinstance(value, dict) else None, None


def load_run(run_dir: str | Path, *, source: str = "measured") -> dict[str, Any]:
    """Load one run directory, retaining warnings and partial information."""

    path = Path(run_dir)
    summary, summary_error = _read_json(path / "summary.json")
    manifest, manifest_error = _read_json(path / "manifest.json")
    notes_path = path / "experiment-notes.json"
    notes, notes_error = _read_json(notes_path) if notes_path.exists() else ({}, None)
    profile, profile_error = _read_profile_quality(path)
    warnings = [e for e in (summary_error, manifest_error, notes_error, profile_error) if e]

    summary = summary or {}
    manifest = manifest or {}
    notes = notes or {}
    if manifest.get("synthetic_fixture") is True:
        source = "synthetic fixture"
    run_id = summary.get("run_id") or manifest.get("run_id") or path.name
    status = summary.get("status") or manifest.get("status") or "UNKNOWN"
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []

    if status == "SUCCESS" and not cases:
        warnings.append("run reports SUCCESS but contains no cases")
    failed_cases = sum(1 for case in cases if case.get("status") != "SUCCESS")
    if failed_cases:
        warnings.append(f"{failed_cases} case(s) are incomplete or failed")

    return {
        "run_dir": path,
        "run_id": str(run_id),
        "profile_name": summary.get("profile_name") or manifest.get("profile_name"),
        "model_id": summary.get("model_id"),
        "status": status,
        "started_at": summary.get("started_at") or manifest.get("started_at"),
        "finished_at": summary.get("finished_at") or manifest.get("finished_at"),
        "duration_seconds": summary.get("duration_seconds") or manifest.get("duration_seconds"),
        "gpu_model": (summary.get("gpu_info") or manifest.get("gpu_info") or {}).get("model"),
        "summary": summary,
        "manifest": manifest,
        "profile": profile or {},
        "hypothesis": notes.get("hypothesis") or manifest.get("hypothesis"),
        "observations": notes.get("observations") or manifest.get("observations"),
        "source": source,
        "warnings": warnings,
        "complete": (
            status == "SUCCESS"
            and bool(cases)
            and not failed_cases
            and summary_error is None
            and manifest_error is None
        ),
    }


def write_run_notes(
    run_dir: str | Path,
    *,
    hypothesis: str,
    observations: str = "",
) -> Path:
    """Create notebook notes for a newly launched run without replacing a file."""

    path = Path(run_dir) / "experiment-notes.json"
    payload = {"hypothesis": hypothesis, "observations": observations}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def discover_runs(
    output_root: str | Path,
    *,
    profile_name: str | None = None,
    source: str = "measured",
) -> list[dict[str, Any]]:
    """Discover run directories below an output root, newest first."""

    root = Path(output_root)
    if not root.exists():
        return []
    candidates = {p.parent for p in root.rglob("summary.json")}
    candidates.update(p.parent for p in root.rglob("manifest.json"))
    runs = [load_run(path, source=source) for path in sorted(candidates)]
    if profile_name:
        runs = [run for run in runs if run["profile_name"] == profile_name]
    return sorted(runs, key=lambda run: (run.get("started_at") or "", run["run_id"]), reverse=True)


def history_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create compact rows for a notebook history table."""

    return [
        {
            "run_id": run.get("run_id"),
            "source": run.get("source"),
            "status": run.get("status"),
            "complete": run.get("complete"),
            "started_at": run.get("started_at"),
            "duration_seconds": run.get("duration_seconds"),
            "gpu_model": run.get("gpu_model"),
            "hypothesis": run.get("hypothesis"),
            "warnings": "; ".join(run.get("warnings", [])),
            "run_dir": str(run.get("run_dir")),
        }
        for run in runs
    ]


def _percentile(benchmark: Mapping[str, Any], family: str, percentile: str = "p50_ms") -> Any:
    values = benchmark.get(family)
    return values.get(percentile) if isinstance(values, Mapping) else None


def performance_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten performance cases while keeping every absent metric as ``None``."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        summary = run.get("summary") or {}
        for case in summary.get("cases") or []:
            config = f"CG {'on' if case.get('cuda_graphs') else 'off'} / CP {'on' if case.get('chunked_prefill') else 'off'}"
            for benchmark in case.get("benchmarks") or []:
                deltas = benchmark.get("deltas_vs_baseline") or {}
                rows.append(
                    {
                        "run_id": run.get("run_id"),
                        "source": run.get("source"),
                        "run_status": run.get("status"),
                        "case_id": case.get("case_id"),
                        "case_status": case.get("status"),
                        "config": config,
                        "is_baseline": bool(case.get("is_baseline")),
                        "repetition": case.get("repetition"),
                        "workload": benchmark.get("workload_name") or benchmark.get("workload_slug"),
                        "workload_slug": benchmark.get("workload_slug"),
                        "output_throughput_tok_per_s": benchmark.get("output_throughput_tok_per_s"),
                        "request_throughput_req_per_s": benchmark.get("request_throughput_req_per_s"),
                        "ttft_p50_ms": _percentile(benchmark, "ttft"),
                        "ttft_p95_ms": _percentile(benchmark, "ttft", "p95_ms"),
                        "itl_p50_ms": _percentile(benchmark, "itl"),
                        "itl_p95_ms": _percentile(benchmark, "itl", "p95_ms"),
                        "e2e_p50_ms": _percentile(benchmark, "e2e"),
                        "output_throughput_delta_pct": deltas.get("output_throughput_tok_per_s_pct"),
                        "ttft_delta_pct": deltas.get("ttft_p50_ms_pct"),
                        "ttft_p95_delta_pct": deltas.get("ttft_p95_ms_pct"),
                        "itl_delta_pct": deltas.get("itl_p50_ms_pct"),
                        "itl_p95_delta_pct": deltas.get("itl_p95_ms_pct"),
                        "e2e_delta_pct": deltas.get("e2e_p50_ms_pct"),
                    }
                )
    return rows


def _quality_signature(run: Mapping[str, Any], task_name: str) -> str | None:
    tasks = ((run.get("profile") or {}).get("quality") or {}).get("tasks") or []
    task = next((item for item in tasks if item.get("name") == task_name), None)
    if not task:
        return None
    comparable = {
        "name": task.get("name"),
        "num_fewshot": task.get("num_fewshot"),
        "limit": task.get("limit"),
        "extra_args": task.get("extra_args") or [],
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def quality_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten quality results and attach an evaluation-compatibility signature."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        for result in (run.get("summary") or {}).get("quality") or []:
            task_name = result.get("task_name") or result.get("task")
            rows.append(
                {
                    "run_id": run.get("run_id"),
                    "source": run.get("source"),
                    "task_name": task_name,
                    "metric": result.get("primary_metric_name") or result.get("metric"),
                    "score": result.get("primary_score") if "primary_score" in result else result.get("score"),
                    "stderr": result.get("stderr"),
                    "evaluation_signature": _quality_signature(run, task_name),
                }
            )
    return rows


def comparable_quality_groups(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    """Group rows with the same recorded task settings and named metric.

    This is a necessary compatibility check, not proof that unrecorded evaluator
    versions, model revisions, or prompt templates are identical.
    """

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        signature = row.get("evaluation_signature")
        task = row.get("task_name")
        metric = row.get("metric")
        if task and signature and metric:
            groups.setdefault((str(task), str(metric), str(signature)), []).append(row)
    return groups


def plot_performance(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = (
        "output_throughput_tok_per_s",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "itl_p50_ms",
        "itl_p95_ms",
        "e2e_p50_ms",
    ),
):
    """Plot available performance metrics and explicitly mark unavailable panels."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        title, unit, _ = PERFORMANCE_METRICS[metric]
        available = [row for row in rows if row.get(metric) is not None]
        if not available:
            ax.text(0.5, 0.5, "Unavailable in these artifacts", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax.set_title(title)
            continue
        labels = [f"{row.get('workload')}\n{row.get('config')}\n{row.get('run_id')}" for row in available]
        values = [row[metric] for row in available]
        colors = ["#3178c6" if row.get("is_baseline") else "#f29e4c" for row in available]
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_baseline_deltas(rows: Sequence[Mapping[str, Any]]):
    """Plot canonical within-run deltas already recorded by the harness."""

    import matplotlib.pyplot as plt

    fields = (
        ("output_throughput_delta_pct", "Output throughput"),
        ("ttft_delta_pct", "TTFT p50"),
        ("ttft_p95_delta_pct", "TTFT p95"),
        ("itl_delta_pct", "ITL p50"),
        ("itl_p95_delta_pct", "ITL p95"),
        ("e2e_delta_pct", "End-to-end p50"),
    )
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4), squeeze=False)
    non_baseline = [row for row in rows if not row.get("is_baseline")]
    for ax, (field, title) in zip(axes[0], fields):
        available = [row for row in non_baseline if row.get(field) is not None]
        if not available:
            ax.text(0.5, 0.5, "Unavailable in these artifacts", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax.set_title(title)
            continue
        labels = [f"{row.get('workload')}\n{row.get('config')}" for row in available]
        values = [row[field] for row in available]
        higher_is_better = field == "output_throughput_delta_pct"
        colors = [
            "#3aa76d" if ((value >= 0) == higher_is_better) else "#d95d5d"
            for value in values
        ]
        ax.bar(range(len(values)), values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% vs baseline")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_quality(rows: Sequence[Mapping[str, Any]]):
    """Plot quality scores separately, faceted by recorded task and metric."""

    import matplotlib.pyplot as plt

    groups = comparable_quality_groups(rows)
    if not groups:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No quality results with recorded task settings", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
    for ax, ((task, metric, _signature), group) in zip(axes[0], groups.items()):
        available = [row for row in group if row.get("score") is not None]
        if not available:
            ax.text(0.5, 0.5, "Scores unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax.set_title(f"{task}\n{metric}")
            continue
        labels = [f"{row.get('run_id')}\n{row.get('source')}" for row in available]
        scores = [row["score"] for row in available]
        errors = [row.get("stderr") if row.get("stderr") is not None else float("nan") for row in available]
        ax.bar(range(len(scores)), scores, yerr=errors, capsize=4, color="#7a6fd0")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_ylabel("score")
        ax.set_title(f"{task}\n{metric}")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig

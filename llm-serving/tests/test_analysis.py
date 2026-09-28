import json

import pytest

from llm_serving.analysis import (
    comparable_quality_groups,
    discover_runs,
    history_rows,
    load_run,
    performance_rows,
    quality_rows,
    write_run_notes,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_partial_artifacts_preserve_missing_metrics(tmp_path):
    run_dir = tmp_path / "qwen35-0.8b-smoke" / "partial-run"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "manifest.json",
        {"run_id": "partial-run", "profile_name": "qwen35-0.8b-smoke", "status": "ABORTED"},
    )
    write_json(
        run_dir / "summary.json",
        {
            "run_id": "partial-run",
            "profile_name": "qwen35-0.8b-smoke",
            "status": "ABORTED",
            "cases": [
                {
                    "case_id": "case-00-cg1-cp1-rep0",
                    "cuda_graphs": True,
                    "chunked_prefill": True,
                    "is_baseline": True,
                    "status": "SUCCESS",
                    "benchmarks": [
                        {
                            "workload_slug": "short-interactive",
                            "output_throughput_tok_per_s": 101.5,
                            "ttft": {"p50_ms": 12.0},
                            "itl": {"p50_ms": None},
                            "e2e": {},
                        }
                    ],
                },
                {"case_id": "case-01", "status": "FAILED", "benchmarks": []},
            ],
        },
    )

    run = load_run(run_dir)
    rows = performance_rows([run])

    assert run["complete"] is False
    assert "1 case(s) are incomplete or failed" in run["warnings"]
    assert rows[0]["output_throughput_tok_per_s"] == 101.5
    assert rows[0]["itl_p50_ms"] is None
    assert rows[0]["e2e_p50_ms"] is None
    assert rows[0]["output_throughput_delta_pct"] is None


def test_discovery_includes_manifest_only_failed_run_and_bad_summary(tmp_path):
    failed = tmp_path / "profile" / "failed"
    bad = tmp_path / "profile" / "bad"
    failed.mkdir(parents=True)
    bad.mkdir(parents=True)
    write_json(failed / "manifest.json", {"run_id": "failed", "profile_name": "profile", "status": "FAILED"})
    (bad / "summary.json").write_text("{broken", encoding="utf-8")

    runs = discover_runs(tmp_path)

    assert {run["run_id"] for run in runs} == {"failed", "bad"}
    assert next(run for run in runs if run["run_id"] == "bad")["warnings"]
    assert all(run["complete"] is False for run in runs)
    assert history_rows(runs)[0]["source"] == "measured"


def test_quality_comparison_requires_matching_recorded_settings(tmp_path):
    runs = []
    for run_id, limit in (("a", 20), ("b", 20), ("c", 100)):
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_json(
            run_dir / "summary.json",
            {
                "run_id": run_id,
                "status": "SUCCESS",
                "cases": [{}],
                "quality": [{"task_name": "gsm8k", "primary_metric_name": "exact_match", "primary_score": 0.5}],
            },
        )
        (run_dir / "profile.yaml").write_text(
            f"quality:\n  tasks:\n    - name: gsm8k\n      num_fewshot: 5\n      limit: {limit}\n",
            encoding="utf-8",
        )
        runs.append(load_run(run_dir))

    groups = comparable_quality_groups(quality_rows(runs))

    assert sorted(len(group) for group in groups.values()) == [1, 2]


def test_quality_comparison_separates_named_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "profile.yaml").write_text(
        "quality:\n  tasks:\n    - name: gsm8k\n      num_fewshot: 5\n      limit: 20\n",
        encoding="utf-8",
    )
    write_json(
        run_dir / "summary.json",
        {
            "run_id": "run",
            "status": "SUCCESS",
            "cases": [{"status": "SUCCESS"}],
            "quality": [
                {"task_name": "gsm8k", "primary_metric_name": "exact_match,strict-match", "primary_score": 0.5},
                {"task_name": "gsm8k", "primary_metric_name": "exact_match,flexible-extract", "primary_score": 0.6},
            ],
        },
    )
    write_json(run_dir / "manifest.json", {"run_id": "run", "status": "SUCCESS"})

    groups = comparable_quality_groups(quality_rows([load_run(run_dir)]))

    assert len(groups) == 2


def test_malformed_profile_is_visible_and_does_not_crash(tmp_path):
    write_json(tmp_path / "summary.json", {"run_id": "bad-yaml", "status": "SUCCESS", "cases": [{"status": "SUCCESS"}]})
    write_json(tmp_path / "manifest.json", {"run_id": "bad-yaml", "status": "SUCCESS"})
    (tmp_path / "profile.yaml").write_text("quality: [unterminated", encoding="utf-8")

    run = load_run(tmp_path)

    assert run["complete"] is True
    assert any("unreadable profile.yaml" in warning for warning in run["warnings"])


def test_run_notes_use_exclusive_create(tmp_path):
    path = write_run_notes(tmp_path, hypothesis="Graphs improve throughput", observations="")
    assert json.loads(path.read_text(encoding="utf-8"))["hypothesis"] == "Graphs improve throughput"
    with pytest.raises(FileExistsError):
        write_run_notes(tmp_path, hypothesis="replacement")


def test_synthetic_manifest_cannot_be_mislabeled_measured(tmp_path):
    write_json(tmp_path / "manifest.json", {"run_id": "fixture", "status": "FAILED", "synthetic_fixture": True})

    run = load_run(tmp_path, source="measured")

    assert run["source"] == "synthetic fixture"

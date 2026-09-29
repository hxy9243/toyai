import gzip
import json

import pytest
from pydantic import ValidationError

from llm_serving.analysis import performance_rows
from llm_serving.profiling import kernel_rows, trace_rows
from llm_serving.schemas import ProfilingConfig, ServerConfig
from llm_serving.workloads import WORKLOADS, build_bench_serve_command


@pytest.mark.parametrize('config', [
    {'max_iterations': 0}, {'delay_iterations': -1},
    {'workload': 'typo'}, {'benchmark_timeout_seconds': 0},
])
def test_invalid_windows(config):
    with pytest.raises(ValidationError):
        ProfilingConfig(**config)


@pytest.mark.parametrize('flag', ['--profiler-config', '--profiler-config.max_iterations=0'])
def test_cannot_bypass_managed_profiling(flag):
    with pytest.raises(ValidationError):
        ServerConfig(extra_args=[flag])


def test_benchmark_opt_in():
    assert '--profile' not in build_bench_serve_command('model', WORKLOADS[0])
    assert '--profile' in build_bench_serve_command('model', WORKLOADS[0], profile=True)


def test_trace_inventory_and_kernel_summary(tmp_path):
    directory = tmp_path / 'cases/case-00/profiling/attempt-1/decode-heavy'
    directory.mkdir(parents=True)
    path = directory / 'worker.pt.trace.json.gz'
    trace = {'traceEvents': [
        {'ph': 'X', 'cat': 'kernel', 'name': 'matmul', 'dur': 2000, 'args': {'device': 0}},
        {'ph': 'X', 'cat': 'kernel', 'name': 'matmul', 'dur': 1000, 'args': {'device': 0}},
        {'ph': 'X', 'cat': 'kernel', 'name': 'attention', 'dur': 1000, 'args': {'device': 0}},
        {'ph': 'X', 'cat': 'cpu_op', 'name': 'matmul', 'dur': 9000},
        {'ph': 'X', 'cat': 'kernel', 'name': 'other GPU', 'dur': 1000, 'args': {'device': 1}},
        {'ph': 'X', 'cat': 'kernel', 'name': 'invalid', 'dur': -1},
    ]}
    with gzip.open(path, 'wt') as handle:
        json.dump(trace, handle)
    inventory = trace_rows(tmp_path)
    assert len(inventory) == 1
    assert inventory[0]['attempt'] == 'attempt-1'
    assert inventory[0]['case_id'] == 'case-00'
    rows = kernel_rows(path)
    assert rows[0]['total_ms'] == 3
    assert rows[0]['calls'] == 2
    assert rows[0]['device_kernel_time_pct'] == 75
    assert next(r for r in rows if r['device'] == '1')['device_kernel_time_pct'] == 100
    assert kernel_rows(path, limit=1) == rows[:1]
    assert trace_rows(tmp_path / 'missing') == []


def test_profiled_metrics_require_explicit_opt_in():
    runs = [{'summary': {'cases': [{'benchmarks': [
        {'workload_slug': 'short-interactive'},
        {'workload_slug': 'decode-heavy', 'profiled': True},
    ]}]}}]
    assert len(performance_rows(runs)) == 1
    assert len(performance_rows(runs, include_profiled=True)) == 2

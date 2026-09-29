"""Offline coverage for the two-GPU concurrency experiment and metric boundaries."""
from pathlib import Path

import pytest
import yaml

from llm_serving.cache_metrics import cache_counter_delta
from llm_serving.flash import load_flash_config, build_create_kwargs
from llm_serving.matrix import expand_matrix
from llm_serving.schemas import ExperimentProfile, HostConfig, validate_profile_against_host
from llm_serving.workloads import expand_workloads, build_bench_serve_command, normalize_benchmark_output
from llm_serving.runner import ExperimentRunner
from llm_serving.transport import MockTransport, CommandResult

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return ExperimentProfile.model_validate(yaml.safe_load((ROOT / 'experiment_profiles/qwen38-27b-parallel.yaml').read_text()))


def snapshot(hits, queries):
    return f'vllm:prefix_cache_hits_total{{model_name="qwen",engine="0"}} {hits}\nvllm:prefix_cache_queries_total{{model_name="qwen",engine="0"}} {queries}\n'


def test_complete_matrix_and_provisioning_plan():
    p = profile()
    cases = expand_matrix(p)
    assert len(cases) == 12
    assert len({c.case_id for c in cases}) == 12
    assert {(c.tensor_parallel, c.pipeline_parallel) for c in cases} == {(2, 1), (1, 2)}
    for c in cases:
        assert c.cuda_graphs and c.chunked_prefill
        assert c.max_num_seqs in [4, 8, 16, 32, 64, 128]
        assert '--enforce-eager' not in c.vllm_args
        assert c.vllm_args[c.vllm_args.index('--max-model-len') + 1] == '262144'
        assert c.vllm_args[c.vllm_args.index('--pipeline-parallel-size') + 1] == str(c.pipeline_parallel)
        assert c.vllm_args[c.vllm_args.index('--max-num-seqs') + 1] == str(c.max_num_seqs)
    workloads = expand_workloads(p.benchmark)
    assert len(workloads) == 20
    for w in workloads:
        cmd = build_bench_serve_command(p.model.id, w)
        assert cmd[cmd.index('--request-rate') + 1] == 'inf'
        assert w.num_prompts >= w.max_concurrency
    config = load_flash_config(ROOT / 'runpod.qwen38-27b.yaml')
    kwargs = build_create_kwargs(config, 'test')
    assert kwargs['gpu_count'] == 2
    assert kwargs['gpu_type_id'] == 'NVIDIA A100-SXM4-80GB'
    assert kwargs['volume_in_gb'] == 100
    assert kwargs['volume_mount_path'] == '/workspace'
    host_data = dict(config['host'])
    host_data.pop('alias')
    host = HostConfig(ssh_target='unused.invalid', **host_data)
    validate_profile_against_host(p, host)
    # Also checks the sweep's PP configuration, independent of the base config.
    p.sweep.parallelism = [p.sweep.parallelism[1]]
    with pytest.raises(ValueError, match='requires 2 GPUs'):
        validate_profile_against_host(p, host.model_copy(update={'gpu_ids': [0]}))


def test_cache_deltas_and_missing_values():
    assert cache_counter_delta(snapshot(10, 100), snapshot(30, 150))['hit_rate'] == .4
    assert cache_counter_delta(snapshot(0, 0), snapshot(0, 10))['hit_rate'] == 0
    assert cache_counter_delta('', snapshot(0, 10))['hit_rate'] is None
    assert cache_counter_delta(snapshot(0, 0), snapshot(0, 0))['status'] == 'no_queries'
    assert cache_counter_delta(snapshot(20, 100), snapshot(0, 10))['status'] == 'counter_reset'
    assert cache_counter_delta(snapshot(0, 0), snapshot(20, 10))['status'] == 'invalid_delta'
    assert cache_counter_delta(snapshot(0, 0), snapshot('NaN', 10))['hit_rate'] is None
    multi_before = snapshot(0, 0) + snapshot(0, 0).replace('engine="0"', 'engine="1"')
    multi_after = snapshot(10, 20) + snapshot(0, 80).replace('engine="0"', 'engine="1"')
    assert cache_counter_delta(multi_before, multi_after)['hit_rate'] == .1
    assert normalize_benchmark_output(expand_workloads(profile().benchmark)[0], {'completed': 0}).completed_requests == 0


@pytest.mark.parametrize('mode', ['host', 'docker'])
def test_runner_captures_cache_metrics_for_every_concurrency(tmp_path, mode):
    p = profile()
    host = HostConfig(ssh_target='unused.invalid', execution_mode=mode, remote_root='/workspace/test',
                      gpu_ids=[0, 1], expected_gpu_model='A100', hf_cache_path='/workspace/hf',
                      vllm_cache_path='/workspace/vllm')
    class MetricsTransport(MockTransport):
        calls = 0
        def run_cmd(self, cmd, timeout=None):
            if '8000/metrics' in cmd:
                self.calls += 1
                return CommandResult(0, snapshot(0, 0) if self.calls % 2 else snapshot(10, 100), '')
            return super().run_cmd(cmd, timeout=timeout)
    transport = MetricsTransport(host)
    runner = ExperimentRunner(p, host, transport, ROOT, tmp_path)
    runner._wait_for_health = lambda _: (True, '')
    ok, error, results = runner._run_single_case(expand_matrix(p)[0])
    assert ok, error
    assert len(results) == 20
    assert all(b.cache_metrics['hit_rate'] == .1 for b in results)
    assert {b.max_concurrency for b in results} == {8, 16, 32, 64, 128}
    assert len(list(runner.local_run_dir.rglob('*.prom'))) == 40


@pytest.mark.parametrize('field,value', [('max_num_seqs', [0]), ('max_num_seqs', [4, 4])])
def test_invalid_sweep(field, value):
    data = profile().model_dump()
    data['sweep'][field] = value
    with pytest.raises(ValueError):
        ExperimentProfile.model_validate(data)


def test_reporting_keeps_sweep_and_cache_dimensions(tmp_path):
    import csv
    import io
    from llm_serving.reporting import generate_summary_data, export_summary_csv
    from llm_serving.analysis import performance_rows
    p = profile()
    host = HostConfig(ssh_target='unused.invalid', remote_root='/workspace/test',
                      gpu_ids=[0, 1], expected_gpu_model='A100', hf_cache_path='/workspace/hf',
                      vllm_cache_path='/workspace/vllm')
    benchmark = normalize_benchmark_output(expand_workloads(p.benchmark)[0], {'completed': 512})
    benchmark.cache_metrics = cache_counter_delta(snapshot(0, 0), snapshot(10, 100))
    case = dict(case_id='test', tensor_parallel=2, pipeline_parallel=1, max_num_seqs=4,
                cuda_graphs=True, chunked_prefill=True, status='SUCCESS', benchmarks=[benchmark.to_dict()])
    summary = generate_summary_data(p, host, {'run_id': 'test'}, [case])
    exported = next(csv.DictReader(io.StringIO(export_summary_csv(summary))))
    assert exported['cache_hit_rate'] == '0.1'
    assert exported['max_concurrency'] == '8'
    assert exported['max_num_seqs'] == '4'
    row = performance_rows([{'run_id': 'test', 'summary': summary}])[0]
    assert (row['tensor_parallel'], row['pipeline_parallel'], row['max_num_seqs']) == (2, 1, 4)
    assert row['cache_status'] == 'ok'

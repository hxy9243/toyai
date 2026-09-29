# Bounded PyTorch profiling

Profiling explains *where time went*. Keep a normal run as the performance baseline;
run a second, diagnostic capture with identical model, hardware, workload and serving
settings. Recording and exporting traces changes latency and throughput, even if
only a short portion of the benchmark is recorded.

## Enable a window

The harness targets vLLM **0.28.0**, including its worker delay/limit controls. Add
this top-level section to an experiment YAML (omitting it leaves profiling off):

```yaml
profiling:
  enabled: true
  workload: decode-heavy
  delay_iterations: 20
  max_iterations: 10
  with_stack: false
  record_shapes: false
  profile_memory: false
  benchmark_timeout_seconds: 1800
```

A ready example is `experiment_profiles/qwen35-0.8b-profile.yaml`. From `llm-serving/`:

```bash
uv run llm-serving plan qwen35-0.8b-profile --host YOUR_HOST --inventory hosts.local.yaml
uv run llm-serving run qwen35-0.8b-profile --host YOUR_HOST --inventory hosts.local.yaml
uv run --extra analysis jupyter lab notebooks/profiling.ipynb
```

The example retains the smoke profile's two CUDA-graph configurations, disables
quality evaluation, and records only the selected workload in each case. The other
performance workloads still run. Reduce the sweep to one case for a first capture.

The runner configures the **server workers**, then adds `--profile` only to the
selected `vllm bench serve` command. vLLM starts profiling after its test and warmup
requests, before its measured workload. Workers skip `delay_iterations` and record
up to `max_iterations`; the benchmark's stop call finalizes the profiler. The
frontend is excluded (`ignore_frontend: true`) because its profiler does not follow
worker iteration limits. Stacks, shapes and memory tracking are disabled by default
to keep the first trace small.

**An engine iteration is one worker execution step, not one HTTP request, one
second, or necessarily one token.** Batching and chunked prefill change its meaning.
With multiple workers the window is local to each worker, not a synchronized global
interval. A high delay can miss a short workload entirely; that is reported as a
failed capture if no nonempty trace is exported. Trace presence alone does not prove
that all requested iterations were reached. Inspect the timeline and server log.

Artifacts are fetched with the existing case outputs:

```text
cases/<case-id>/profiling/attempt-0/decode-heavy/<worker>.pt.trace.json.gz
cases/<case-id>/server.log
summary.json
report.md
```

Retries use separate attempt directories. Match the successful attempt with the
case's `retries` value in `summary.json`; earlier traces may be partial. In Docker,
traces are written under the mounted `/workspace/output` directory. Host mode uses
the case's absolute path. The benchmark timeout includes trace export, which can
be slow. On a benchmark failure the existing runner tears down the server and
retries; an interrupted export may leave incomplete traces. Do not treat those as
complete measurements.

## How torch.profiler works

The profiler collects CPU operator events and device activity such as CUDA kernels
and copies. These events include timing and correlation information, so a timeline
can connect CPU launches with GPU work. CPU time is not GPU time: CUDA launches are
asynchronous. `record_function` adds named regions in code you control. Optional
shape, stack and allocation collection adds detail and overhead.

In a Python loop you own, a typical capture looks like this:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=20, warmup=2, active=10, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace('trace.json'),
) as prof:
    for batch in batches:
        with torch.profiler.record_function('inference'):
            model(batch)
        prof.step()
```

`wait` leaves profiling inactive, `warmup` prepares it and discards those samples,
`active` saves events, and `prof.step()` advances the schedule. Profiler warmup is
separate from warming the model, compilation and CUDA graphs. The harness above
uses vLLM's delay/maximum controls, not this Python schedule, and does not add a
profiler warmup phase. Wrapping the remote benchmark client in `torch.profiler`
would profile the client, not the server GPU work.

## Presenting results

Use three complementary views:

1. **Benchmark context:** model/revision, GPU, runtime version, case, workload,
   capture window, and whether profiling was enabled. Keep TTFT/ITL/throughput
   comparisons from the unprofiled run. Reports/CSV mark captured benchmarks as
   `profiled`; their deltas are omitted. `performance_rows()` excludes them by
   default; `include_profiled=True` opts in for diagnostic inspection.
2. **Kernel table and bar chart:** the notebook ranks kernels in one selected worker
   trace by summed duration, with calls, average duration and per-device kernel-time
   share. Overlapping kernels mean those sums are **not** wall time, utilization or
   critical-path attribution. Do not sum ranks and call the result elapsed time.
3. **Interactive timeline:** open the `.pt.trace.json.gz` file in
   [Perfetto](https://ui.perfetto.dev/). Inspect GPU gaps, CPU launches, copies,
   attention/matmul kernels and synchronization. Use engine annotations when present
   to distinguish prefill from decode; duration alone cannot identify those phases.
   CUDA graphs and compiled/fused kernels can reduce per-operator attribution.

`notebooks/profiling.ipynb` reads saved files only. It does not invent sample timings
when no capture exists. Set `LLM_SERVING_PROFILE_RUN` to a run directory to select a
specific run, or `LLM_SERVING_OUTPUT_ROOT` to another artifact root. Trace JSON is
loaded fully for the kernel table, so start with small windows and select one worker.

## A window measured in seconds

The YAML window is iteration-based. For an approximate wall-clock window, launch a
server with `--profiler-config` and issue `POST /start_profile` while a separate
benchmark is already sending requests; after the desired interval issue
`POST /stop_profile`. Keep `ignore_frontend: true`, use `delay_iterations: 0` and
`max_iterations: 0` for endpoint-controlled timing, and **omit `--profile` on that
benchmark** so two controllers do not conflict. Put the stop request in a `finally`
block in any automation. Start/stop RPC latency and worker synchronization make
this approximate, and flushing may finish much later than the capture interval.
The harness does not yet automate a seconds-based window.

## Sources and validation scope

- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler)
- [vLLM profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling/)
- [Pinned vLLM 0.28 profiler configuration](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/profiler.py)
- [Pinned worker implementation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/profiler/wrapper.py)
- [Pinned benchmark start/stop flow](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/benchmarks/serve.py)

Local tests cover configuration, both launch modes through a mock transport,
retry isolation, missing exports, metric separation and trace analysis. An actual
GPU capture remains necessary to verify the runtime on the chosen host.

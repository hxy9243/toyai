# Qwen3.8-27B parallelism experiment

The initial delivery is **code only**. Do not allocate a Pod or volume, download
weights, or launch GPU benchmarks until the user explicitly requests a run.
This overrides the in-place measured-run requirement in `../AGENTS.md` for this preparation
commit: a notebook may show the offline plan and “no measured runs” output.
After a real run, execute the notebook in place against the saved artifacts.

## Experiment contract

- Profile: `experiment_profiles/qwen38-27b-parallel.yaml`.
- Model: `Qwen/Qwen3.8-27B`, BF16, text-only serving; source:
  [official model card](https://huggingface.co/Qwen/Qwen3.8-27B) and
  [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
  Native context is 262,144 tokens; keep `max_model_len` at that value.
- CUDA graphs enabled, chunked prefill enabled, prefix caching enabled.
  TP=2/PP=1 versus TP=1/PP=2; DP=1 throughout.
- Batch means vLLM `--max-num-seqs`: 4, 8, 16, 32, 64, 128.
  Client concurrency is independently 8, 16, 32, 64, 128, with unthrottled
  arrivals (`--request-rate inf`) and 512 requests per benchmark.
- 12 server cases × 5 concurrencies × 4 standard workloads = 240 benchmarks
  (122,880 measured requests plus warmups/probes), one repetition. Quality
  evaluation and profiler traces are disabled for this performance experiment.
- Request lengths remain 256/128, 8192/128, 15360/256, and 256/1024
  input/output tokens. The historical `near-limit-prefill` name describes the
  old 16K suite; this is **not a 256K prompt-load test**. The 256K server cap
  does not promise that 128 full-context requests fit in memory simultaneously.
- Hardware template: `runpod.qwen38-27b.yaml`, 2× NVIDIA A100-SXM4-80GB,
  100GB Pod volume mounted at `/workspace`; Hugging Face cache is
  `/workspace/.cache/huggingface`. It is allocated together with the Pod, not
  by an offline plan. The volume survives stops but not Pod deletion. For
  independently persistent storage, provision a 100GB network volume in a
  compatible data center, set `network_volume_id` and `volume_gb: 0` instead.

## Offline checks (safe now)

Run all commands from `llm-serving/`:

```bash
uv sync --extra dev --extra analysis
uv run pytest -q
uv run flash --config runpod.qwen38-27b.yaml plan
uv run jupyter execute --inplace notebooks/qwen38-27b-parallel.ipynb
```

The notebook only expands the plan and reads existing artifacts. It never
creates a machine, downloads weights, or starts a benchmark.

## Future run procedure (only after run authorization)

1. Review GPU type, image, SSH key, storage, and credentials as described in
   `docs/flash.md`. Copy the template to ignored `runpod.local.yaml` if local
   overrides are needed. Inspect the offline plan first, then:

   ```bash
   uv run flash --config runpod.qwen38-27b.yaml create --wait
   ```

2. Save the returned Pod ID and generated `.flash/hosts/POD_ID.yaml`.
   Connect using its SSH address/port/key. On the Pod, prepare the persistent
   runtime and check both GPUs and their interconnect:

   ```bash
   python3 -m venv /workspace/venvs/qwen38
   /workspace/venvs/qwen38/bin/python -m pip install 'vllm==0.28.0' ninja
   mkdir -p /workspace/toyai/llm-serving /workspace/.cache/huggingface /workspace/.cache/vllm
   nvidia-smi
   nvidia-smi topo -m
   df -h /workspace
   ```

   Confirm sufficient free space for roughly 54GB of BF16 weights plus runtime
   and caches before the first download. The harness uses the mounted HF cache
   automatically; do not download a duplicate copy elsewhere. Record topology
   and runtime version with the run notes. The pinned runtime's live Qwen3.8
   TP/PP and CUDA graph behavior has not yet been tested. If it fails, preserve
   logs; do not silently lower context, disable graphs, or change precision.

3. Locally validate, preview, then run:

   ```bash
   uv run llm-serving validate experiment_profiles/qwen38-27b-parallel.yaml --host runpod-qwen38 --inventory .flash/hosts/POD_ID.yaml
   uv run llm-serving plan experiment_profiles/qwen38-27b-parallel.yaml --host runpod-qwen38 --inventory .flash/hosts/POD_ID.yaml
   uv run llm-serving run experiment_profiles/qwen38-27b-parallel.yaml --host runpod-qwen38 --inventory .flash/hosts/POD_ID.yaml
   ```

   The current runner retries one failed case once, then aborts the sweep.
   Check failures and missing matrix rows; do not interpret partial runs as a
   complete comparison. Model revision `main` is mutable: pin a reviewed HF
   commit in the profile before a reproducibility-critical run.

4. Fetch/verify the harness artifacts, then stop the exact Pod even if a run
   failed. There is no automatic Pod stop:

   ```bash
   uv run flash --config runpod.qwen38-27b.yaml stop POD_ID
   uv run jupyter execute --inplace notebooks/qwen38-27b-parallel.ipynb
   ```

   Set `LLM_SERVING_OUTPUT_ROOT` for artifacts outside the default `output/`
   directory, and optional comma-separated `LLM_SERVING_RUN_IDS` to select runs.
   Stopped volumes retain storage charges. Preserve data before any deletion.

## Interpreting cache and performance measurements

Client throughput and TTFT/ITL/TPOT/end-to-end latency come from `vllm bench
serve` running on the same Pod over loopback; they exclude internet latency.
Client concurrency is an upper bound, not a guarantee that all requests stay
active throughout the finite benchmark.

Cache hit rate is `Δprefix_cache_hits / Δprefix_cache_queries`, weighted by
queried tokens across the reported engine series. It measures prefix reuse,
not KV memory occupancy. Each benchmark resets prefix cache before sampling;
the before/after window includes the benchmark's warmups and initial probe,
while its performance timer excludes those. Cache counters are asynchronous;
we allow 12 seconds outside performance timing for the default statistics
interval to flush at both boundaries. Do not raise the runtime's statistics
interval above that delay. Random prompts are not a controlled repeated-prefix
workload; low hit rates are valid observations, not evidence of broken caching.

Raw snapshots are stored under `cache-metrics/<case>/attempt-N/`. Summaries,
CSV, and the notebook retain concurrency, TP/PP, batch size, cache rate, and
cache status. Missing series, reset counters, invalid deltas, and no queries
produce a missing hit rate with an explicit status, never a fabricated zero.
Investigate any status other than `ok` before claiming cache coverage.
See [vLLM metric definitions](https://docs.vllm.ai/en/stable/usage/metrics/).

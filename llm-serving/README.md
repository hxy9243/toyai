# vLLM Serving Experiment Harness

Isolated, reproducible benchmark and evaluation harness for parameter sweeps over vLLM serving on remote GPU hosts.

## Features

- **Runtime choices**: Either an extended pinned Docker image (`vllm/vllm-openai:v0.28.0`) or a pre-installed remote Python environment.
- **Parameter Matrix**: Deterministic sweep over CUDA graphs (eager mode vs graph capture) × Chunked prefill on/off.
- **Reproducible Workloads**:
  - Short interactive (256 in / 128 out, 8 req/s, concurrency 32)
  - Long prefill (8192 in / 128 out, 1 req/s, concurrency 8)
  - Near-limit prefill (15360 in / 256 out, 0.5 req/s, concurrency 2)
  - Decode heavy (256 in / 1024 out, 2 req/s, concurrency 16)
- **Quality Evaluation**: Built-in lm-eval against the baseline serving configuration. By default it runs the full [LongBench v2](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/longbench2) suite, GSM8K (5-shot), and the [lm-eval v1 SuperGLUE](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/super_glue) suite.
- **Fail-Safe Retries & Artifact Preservation**: 1 automatic retry on failure; immediate abort on second failure while retrieving partial results and generating structured reports.
- **Reporting**: Full `manifest.json`, `summary.json`, `summary.csv`, and Markdown reports with delta comparisons vs baseline.

## Quick Start

### 1. Configure Host Inventory

Copy `hosts.example.yaml` to `hosts.local.yaml` (gitignored):

```bash
cp hosts.example.yaml hosts.local.yaml
```

Update your remote host SSH target, GPU IDs, and cache paths. Set `execution_mode: host` to run without Docker. The runner creates and reuses `<remote_root>/.llm-serving-venv` by default, using `python_executable` only as its bootstrap interpreter. It installs missing `vllm` and, when quality tasks are enabled, `lm-eval[api]` there. Docker remains the default.

```yaml
execution_mode: host
python_executable: python3  # bootstrap interpreter for .llm-serving-venv
environment:
  VLLM_USE_FLASHINFER_SAMPLER: "0"
```

In host mode the runner starts `vllm serve` directly, writes a PID and `server.log` for each case, and runs benchmarks through that same interpreter. Before the sweep, it installs missing required libraries and validates their imports. The optional `environment` map applies only to direct host processes.

### Model Metadata

Every experiment profile records an official model-card URL, the configuration URL used to read it, and its published `max_position_embeddings` under `model.model_card`. Treat this metadata as the source of truth for model capabilities when creating or revising a profile. The harness validates that the experiment's `server.max_model_len` does not exceed that published maximum.

`server.max_model_len` is an experiment-specific vLLM serving cap, not a claim about the model's architectural context window. It may be smaller when a benchmark intentionally targets a shorter context or the available hardware cannot host the full KV cache.

### Quality Evaluation Defaults

Profiles without `quality.tasks` run the standard suite once against the baseline:

- `longbench2` — the LongBench v2 task tag (503 long-context questions)
- `gsm8k` — five-shot grade-school math
- `super-glue-lm-eval-v1` — the full lm-eval v1 SuperGLUE tag

These are the task names accepted by lm-eval; they correspond to the requested LongBench v2 and SuperGLUE benchmarks. Specify `quality: {tasks: []}` to skip quality evaluation, or provide a `quality.tasks` list to replace the defaults. LongBench v2 includes contexts from 8K up to 2M words, so configure an appropriate `server.max_model_len` and model for the portion of that benchmark you intend to run.

### 2. Validate & Plan

```bash
# Validate profile schema and host compatibility
./launch.sh validate experiment_profiles/qwen35-0.8b-smoke.yaml --host local-mock

# Inspect expanded matrix without executing SSH
./launch.sh plan experiment_profiles/qwen35-27b-bf16.yaml --host h100-node1
```

### 3. Run Experiment Sweep

```bash
# Run sweep via launch.sh
./launch.sh experiment_profiles/qwen35-27b-bf16.yaml --host h100-node1
```

### 4. Regenerate Report

```bash
uv run python -m llm_serving.cli report output/qwen35-27b-bf16/20260902_120000/
```

## Qwen 0.8B notebook

The [Qwen smoke notebook](notebooks/qwen35-0.8b-smoke.ipynb) shows the profile,
serving matrix, saved experiment history, performance charts, and available
quality results. Run All defaults to local analysis; launching a remote experiment
requires explicitly enabling its run cell. When no compatible measured run is
available, it shows an explicit empty state instead of synthetic values.

From this directory:

```bash
uv sync --extra analysis
uv run --extra analysis jupyter lab notebooks/qwen35-0.8b-smoke.ipynb
```

See [notebook usage](docs/notebooks.md) for selecting output directories, configuring
a remote host, and attaching experiment notes.

## Runpod machines with `flash`

The local `flash` command manages Runpod Pods and exports a host configuration for
the existing experiment runner. By default it reads the API key from
`~/.runpod/config.toml`; `RUNPOD_API_KEY` is an explicit override for automation.

```bash
uv run flash --help
```

Follow the [Runpod driver guide](docs/flash.md) to configure and plan a machine,
create it, export its host configuration, and stop it when finished. Stopping
releases compute but may retain billable storage; this driver does not delete Pods.
Local provider configuration and `.flash/` state are excluded from Git and remote
experiment uploads.

## Profiling

See [bounded worker profiling](docs/profiling.md) for capture windows, the diagnostic example profile, and `notebooks/profiling.ipynb` for kernel summaries and Perfetto traces.

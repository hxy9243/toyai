# ToyAI — next experiment steps (temporary TODO)

Created: 2026-09-28. Status: first implementation slice in `codex/experiment-notebook`.

## Current slice

- [x] Create isolated `experiment-notebook` worktree and delegate driver/notebook
  development to separate Sol agents at medium effort.
- [x] Add the Qwen 0.8B notebook, shared results reader/plots, synthetic examples,
  and exclusive per-run hypothesis/observation notes.
- [x] Execute the notebook from a clean kernel against both synthetic examples
  and nine existing saved Qwen runs. Preserve missing measurements as unavailable.
- [x] Finish and independently review the local Runpod create/status/stop/start
  driver using REST API v2, explicit lifecycle targets, and mocked lifecycle tests.
- [x] Validate the combined slice: 61 tests pass, offline provisioning plan
  passes, notebook executes in both data modes, and formatting checks pass.
- [ ] Deliberately provision a real Pod and validate the combined live workflow.
  No paid resources have been created during implementation.

The original broader milestones below remain open where this first slice only
covers part of the scope. Gemma/27B notebooks, automatic run-and-cleanup
orchestration, Torch profiling, and expanded benchmarks remain future work.

Work through one numbered item at a time, review its output, then choose the next.
The order below is recommended, not a fixed schedule. All four items build on the
existing `llm-serving/` harness.

## Starting point

- Three YAML profiles exist: `qwen35-0.8b-smoke`, `qwen35-27b-bf16`, and `gemma3-27b-it`.
- The runner supports SSH hosts, Docker or direct host execution, parameter
  sweeps, retries, artifact collection, and JSON/CSV/Markdown reports.
- GSM8K already runs through lm-eval on the baseline configuration: 20 examples
  in the smoke profile and 250 in the larger profiles, all with 5 shots.
- There is no dedicated notebook per serving profile, Runpod lifecycle driver,
  or typed Torch profiling configuration in the inspected harness.

## Design direction

Keep YAML as the experiment configuration and saved run artifacts as the record.
Notebooks provide narrative, explicit launch cells, analysis, and visualizations;
shared Python helpers own loading and plotting so three notebooks do not become
three copies of the runner. Extend the existing CLI for execution.

Alternatives considered:

| Approach | Benefit | Tradeoff |
| --- | --- | --- |
| Minimal: notebooks that wrap the CLI and read current summaries | Quick first usable result | Logging and comparison logic can drift between notebooks |
| Recommended: thin notebooks plus shared artifact helpers and CLI integrations | Reuses the harness and gives every feature a common run record | Small shared-data foundation needed first |
| Hosted experiment tracker and orchestration service | Centralized multi-user history | Extra infrastructure beyond this local workflow; defer |

Assumptions to revisit when starting the relevant item: “profiles” means the
three serving YAML files; “gsm” means GSM8K initially; “flash” is a working name
for a local Pod driver, not a decision to adopt Runpod's separate Flash product.

## 1. Profile notebooks, experiment log, and visualizations

- [ ] Define a small, versioned run-record contract by extending existing
  `manifest.json` and `summary.json`, preserving older readable outputs.
- [ ] Record resolved profile/config hash, run ID, timestamps, status, Git
  revision/dirty state, model revision, hardware, runtime/package versions,
  workload/seed, and artifact paths. Add a per-run hypothesis and observations.
  Keep credentials and private host inventory out of published notebook outputs.
- [ ] Add shared readers/plotting helpers and a rebuildable experiment index
  derived from run directories. Support failed, partial, and older runs without
  silently converting missing results to zero.
- [ ] Create `llm-serving/notebooks/qwen35-0.8b-smoke.ipynb`,
  `qwen35-27b-bf16.ipynb`, and `gemma3-27b-it.ipynb` from one common structure.
- [ ] Each notebook shows the resolved profile and matrix, the question being
  tested, validation/plan output, an explicitly enabled run cell, a history
  table, results, and conclusions. Default execution is analysis-only and works
  without GPU access; demonstrate plotting with clearly labeled fixtures if
  real results are unavailable.
- [ ] Plot throughput, time to first token, inter-token latency, end-to-end
  latency and baseline deltas where recorded. Group by workload/configuration;
  show variability only where repeated measurements exist. Show quality scores
  separately and compare only runs with compatible evaluation settings.
- [ ] Add optional notebook/analysis dependencies and short usage instructions.

**Done when:** all three notebooks execute in analysis mode from a clean kernel;
the smoke notebook can read two compatible saved runs and display comparisons;
a partial run is visibly incomplete; repeating analysis does not launch jobs or
overwrite existing experiments. Check artifact compatibility with a focused
reader test and execute the notebooks with fixtures or existing artifacts.

**First slice to review:** the smoke notebook and shared artifact reader, before
applying the structure to the two larger profiles.

## 2. Local `flash` driver for Runpod lifecycle

- [ ] Settle the command name and local installation method. Proposed interface:
  `flash plan`, `flash run <profile>`, `flash status`, and `flash cleanup <run-id>`.
  Add a console entry point to the existing Python package; defer a public release.
- [ ] Keep provider settings separate from model/workload YAML: GPU constraints,
  image/template, disk/cache strategy, readiness timeout, maximum runtime,
  spending limits, and cleanup policy. Show the resolved plan without provisioning.
- [ ] Implement one provider adapter using the supported Runpod Pod API or CLI:
  provision/resume → wait for SSH and actual GPU readiness → produce temporary
  host configuration → invoke the existing runner → retrieve artifacts → cleanup.
  Prefer the existing direct-host execution mode in a suitable Pod image; verify
  runtime support rather than assuming Docker-in-Docker.
- [ ] Persist Pod ID, ownership, run ID, and lifecycle state immediately so
  interruption recovery can find resources. Retry/poll boundedly and avoid
  duplicate Pods after uncertain create responses.
- [ ] Make cleanup idempotent on success, failure, timeout, and Ctrl-C. Only
  manage resources created by this run or explicitly selected by the user.
  Provide reconciliation/cleanup after a crashed local process; a `finally`
  block alone cannot handle laptop shutdown or a network outage.
- [ ] Distinguish stop from terminate and select the policy before launching.
  Document retained-storage charges and data survival. Retrieve/verify results
  before termination; if retrieval fails, preserve recoverable data, report the
  outstanding Pod/storage, and give a concrete recovery command.
- [ ] Define timeout enforcement that can act independently of the local driver
  where supported, and clearly document residual cleanup/billing limits.
- [ ] Read API credentials from the environment or local secret configuration;
  exclude secrets from artifacts. Add lifecycle duration and estimated compute
  cost to the experiment record, labeled as estimates rather than billing totals.

**Done when:** mocked lifecycle tests cover successful runs, failed startup,
runner failure, interruption, artifact-copy failure, and repeated cleanup. Then
one deliberately provisioned smoke run returns its artifacts locally and has
its final Pod state verified. No paid resources are created during this planning task.

**Decision for this item:** reusable stopped Pod versus ephemeral Pod with
persistent cache storage; establish budget/GPU limits before the live smoke run.

## 3. Torch profiler support

- [ ] Add optional typed profiling settings, disabled by default: selected
  case/workload, warm-up, bounded capture window, CPU/CUDA activities, and
  optional shapes/stacks/memory capture.
- [ ] Instrument the remote vLLM workers using profiling controls supported by
  the actual runtime version. Check both the pinned Docker version and installed
  host version; profiling the local HTTP client will not expose GPU kernels.
- [ ] Capture a short diagnostic pass separately from ordinary performance
  measurements. Mark profiled runs clearly and exclude their latency/throughput
  from normal baseline comparisons by default.
- [ ] Stop/flush profiling before server teardown; collect trace files for all
  relevant workers/ranks, link them to case/run IDs, and preserve partial traces
  when possible. Bound trace duration and size.
- [ ] Add notebook trace links and operator CPU/CUDA-time summaries when the
  exported format supports them; document opening the timeline in Perfetto.

**Done when:** a small remote GPU workload produces a readable trace with CUDA
events, the trace can be opened locally, and an unprofiled comparison remains
unchanged. Test control sequencing and artifact handling without GPU access,
then verify one bounded GPU capture.

**Dependency:** item 1 supplies artifact viewing; item 2 is convenient but not
required because existing SSH hosts can run this slice.

## 4. Broader benchmarks, preferring lm-eval

- [ ] Strengthen the existing GSM8K path first: preserve named metrics and
  per-sample outputs, expose evaluation failures, and label smoke/subset results.
  Do not mistake the existing 20/250-example runs for full benchmark results.
- [ ] Pin a compatible lm-eval version and dataset/task revisions when this
  item starts. Record prompts/chat templates, few-shot count, seed, generation
  settings, sample count, endpoint/backend, and evaluator version.
- [ ] Make the primary metric explicit per task, preserve metric filters and
  standard errors, and fix parser assumptions against actual lm-eval fixtures.
  Missing/unrecognized metrics must be unavailable, never a fabricated zero.
- [ ] Add small smoke and fuller evaluation presets. Candidate initial tasks:
  GSM8K (including a separately labeled CoT variant), ARC-Challenge, and HellaSwag.
  Verify task IDs, licenses, model prompting, and endpoint capabilities in the
  pinned release before finalizing the set; likelihood tasks may require
  completion/logprob support unavailable through a chat-only backend.
- [ ] Investigate a tool-calling benchmark, with BFCL as the first candidate.
  Verify native support in the selected lm-eval release before promising it.
  Prefer a supported lm-eval task; otherwise use a small custom task only if it
  preserves benchmark semantics, or retain the official evaluator behind a
  separate adapter with the same artifact contract.
- [ ] For tool calling, start with a bounded static/single-turn subset; verify
  model-specific tool formatting, serving flags/parser, and deterministic
  scoring. Keep function choice, argument correctness, and irrelevant-tool
  behavior distinguishable where the benchmark provides those metrics.
  Report custom subsets as local evaluations, not official leaderboard scores.
- [ ] Keep quality evaluation on baseline by default; optionally select sweep
  cases when testing quality regressions. Store task outputs in separate paths
  and show quality alongside performance without pooling unrelated scores.

**Done when:** one math task, one additional reasoning/knowledge task, and one
tool-calling subset each produce inspectable sample outputs and named metrics.
Fixture tests cover metric parsing/failure cases; a small live evaluation matches
the chosen evaluator's raw scores. Full benchmark runs are a separate deliberate step.

**Dependency:** reuse item 1's records/notebooks; Runpod and profiler integration
are optional for benchmark development.

## References checked for planning

- [Runpod Pod lifecycle and storage behavior](https://docs.runpod.io/pods/manage-pods).
- [Profiling vLLM v0.28.0](https://docs.vllm.ai/en/v0.28.0/contributing/profiling/).
- [lm-evaluation-harness and API backends](https://github.com/EleutherAI/lm-evaluation-harness).
- [lm-eval task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md).
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).

Re-check version-specific interfaces when implementing each item. These links
inform the plan; no new integration has been executed or verified yet.

# Experiment notebook runs

When an `llm-serving` experiment has a corresponding notebook, execute that
notebook **in place** after the real run and before committing or handing off.
The checked-in notebook must retain the measured run history and analysis, not
only the notebook source.

- Execute it with `jupyter execute --inplace` (or the equivalent notebook UI)
  against the real artifact root. If the artifacts are outside the checkout,
  supply `LLM_SERVING_OUTPUT_ROOT` and, when needed, `LLM_SERVING_RUN_IDS`.
- Confirm the rendered notebook shows the selected run's performance analysis
  and quality results. Do not substitute synthetic output.
- Keep host inventories, credentials, and other temporary connection details
  outside Git; ensure they do not appear in notebook outputs before commit.

For experiment-specific setup, running instructions, and measurement details,
read the [Qwen3.8-27B notebook guide](notebooks/qwen38-27b-parallel.md).

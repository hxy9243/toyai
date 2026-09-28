# Local Runpod lifecycle driver

`flash` is a deliberately small local driver for Runpod Pods. It can show a
provisioning plan, create one Pod, wait until SSH and the expected GPU work,
emit an inventory for the existing `llm-serving` harness, inspect status, stop a
Pod, and restart it. It never terminates or deletes a Pod.

For notebooks, `create --wait` and `start --wait` export the ready machine to
`.flash/hosts/<pod-id>.yaml`; use that file and its host alias as the notebook's
`HOST_INVENTORY` and `HOST` inputs.

The command name is local to this repository and is unrelated to Runpod's
separate Flash product. Always invoke it as `uv run flash` from this directory
to avoid accidentally calling Runpod's official Flash CLI or another executable
with the same name.

## Prepare the Pod and local credentials

1. Add an SSH public key to your Runpod account. The matching private key must
   be available through your SSH agent/default key set, or set
   `host.ssh_key_path` in the local config.
2. Copy `runpod.example.yaml` to `runpod.local.yaml` and review the GPU type,
   storage, image, SSH key path, and expected GPU model. This initial slice
   supports image-based API v2 creates with one exact GPU type. The image must
   honor Runpod's `startSsh` setup, expose `22/tcp`, and accept the account key.
3. Export the API key in the shell that runs the driver:

   ```bash
   export RUNPOD_API_KEY=...
   ```

   The key is read only from the environment. It is never written to config,
   lifecycle state, a generated host inventory, or command output.
4. For `execution_mode: host`, make sure the Pod has Python 3.10 or newer plus
   the runtime expected by this repository. The current harness container pins
   vLLM 0.28.0 and lm-eval 0.4.12, so a reusable Pod image should install
   compatible versions. The official PyTorch image in the example provides a
   useful Pod/SSH base but does not guarantee those harness packages.

Install the local package and development dependencies once from `llm-serving/`:

```bash
uv sync --extra dev
```

`execution_mode: docker` is also accepted by `HostConfig`, but only choose it
for a Pod image where Docker and NVIDIA container access have been
tested. The driver does not assume Docker-in-Docker support.

## Plan and create

The plan is offline and does not require an API key:

```bash
uv run flash --config runpod.local.yaml plan
```

Create one Pod and wait for Runpod to report `RUNNING`, publish the TCP mapping
for port 22, accept SSH authentication, and return the expected GPU from
`nvidia-smi`:

```bash
uv run flash --config runpod.local.yaml create --wait
```

Create is intentionally never retried. Before the API request, the driver saves
a unique Pod name in `.flash/requests/<request-id>.json`. If the network fails
without a definite API response, inspect the Runpod console for that exact name
before trying another create. This prevents an automatic retry from silently
creating a second billable Pod.

As soon as create returns a Pod ID, the driver writes
`.flash/pods/<pod-id>.json`. With `--wait`, it also writes a strict inventory at
`.flash/hosts/<pod-id>.yaml`. Use the printed alias and file directly with the
existing harness, for example:

```bash
uv run llm-serving validate experiment_profiles/qwen35-0.8b-smoke.yaml \
  --host runpod-qwen08 --inventory .flash/hosts/POD_ID.yaml

uv run llm-serving run experiment_profiles/qwen35-0.8b-smoke.yaml \
  --host runpod-qwen08 --inventory .flash/hosts/POD_ID.yaml
```

The lifecycle driver does not install Python packages, stage the repository, or
invoke a benchmark. Those remain explicit harness/setup steps.

## Inspect, stop, and restart

`stop` and `start` require a tracked Pod ID so that an earlier status check
cannot silently change which paid resource a later command targets. `status`
accepts an ID and falls back to the most recently updated local state only when
the ID is omitted:

```bash
uv run flash --config runpod.local.yaml status POD_ID
uv run flash --config runpod.local.yaml stop POD_ID
uv run flash --config runpod.local.yaml start POD_ID --wait
```

`stop` is idempotent: it checks the Pod first and sends no stop request when the
Pod is already stopped, terminated, or absent. It only operates on IDs present
under `.flash/pods`, even when an ID is typed explicitly.

Stopping releases the GPU and clears container-disk data. Runpod documents that
the Pod volume mounted at `/workspace` is preserved, and its storage can keep
incurring charges while the Pod is stopped. Starting later can also yield zero
GPUs if capacity on the original machine has changed. A network volume persists
independently and is the safer cache/data choice for disposable Pods.

Termination is deliberately outside this first slice: deleting a Pod is a
distinct destructive operation and can permanently remove any data not stored
on a network volume. Use the Runpod console or its supported CLI only after
retrieving results and deciding that the retained data is no longer needed.

## Failure and recovery boundaries

- Readiness is bounded by `driver.readiness_timeout_seconds`. A timeout never
  stops or deletes the Pod automatically; the error includes the exact status
  and stop commands because the Pod can still be billable.
- API mutations are sent once. The driver does not conceal ambiguous outcomes
  with retries.
- Local state and generated inventories are private operator files under
  `.flash`. They contain Pod IDs and addresses but no Runpod API key.
- This implementation does not enforce a provider-side maximum runtime, estimate
  billing, select by current price, transfer artifacts, or delete Pods. Review
  the plan and Runpod console before every paid run.

The REST contract used here follows Runpod's current
[API v2 Pod create reference](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
[Pod lookup reference](https://docs.runpod.io/api-reference-v2/pods/get-a-pod), and
[state-transition reference](https://docs.runpod.io/api-reference-v2/pods/trigger-a-pod-state-transition):
create with `POST /v2/pods`, inspect with `GET /v2/pods/{id}`, start or stop with
`POST /v2/pods/{id}/action`, and resolve the harness connection from
`ssh.direct`. Runpod's
[zero-GPU restart guidance](https://docs.runpod.io/pods/troubleshooting/zero-gpus)
explains the capacity and storage behavior after a Pod is stopped.

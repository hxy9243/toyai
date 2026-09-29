"""Local Runpod Pod lifecycle driver for the llm-serving harness.

The intentionally small first slice provisions Pods, records their IDs locally,
waits for SSH/GPU readiness, and emits a HostInventory consumable by the existing
``llm-serving`` CLI.  It never terminates Pods; ``stop`` preserves the Pod volume.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml

from llm_serving.schemas import HostConfig, HostInventory

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


DEFAULT_READY_TIMEOUT = 900.0
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_RUNPOD_CREDENTIALS_PATH = Path.home() / ".runpod" / "config.toml"


class FlashError(RuntimeError):
    """An actionable lifecycle or configuration error."""


class RunpodAPIError(FlashError):
    """A Runpod SDK request failed."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def load_runpod_api_key(config_path: Path = DEFAULT_RUNPOD_CREDENTIALS_PATH) -> Optional[str]:
    """Read the default Runpod API key without exposing it in local state or output."""
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError:
        return None
    except tomllib.TOMLDecodeError as exc:
        raise FlashError(f"Invalid Runpod credentials TOML: {config_path}") from exc
    except OSError as exc:
        raise FlashError(f"Could not read Runpod credentials TOML: {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise FlashError(f"Runpod credentials TOML must contain a table: {config_path}")
    default_profile = config.get("default", {})
    if not isinstance(default_profile, dict):
        raise FlashError(f"Runpod credentials TOML [default] must be a table: {config_path}")
    api_key = default_profile.get("api_key", config.get("api_key"))
    return api_key.strip() if isinstance(api_key, str) and api_key.strip() else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(mapping: Mapping[str, Any], key: str, expected: type, context: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected) or (expected is str and not value.strip()):
        raise FlashError(f"{context}.{key} must be a non-empty {expected.__name__}")
    return value


def _positive_number(value: Any, name: str, *, integer: bool = False) -> float | int:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or value <= 0:
        kind = "integer" if integer else "number"
        raise FlashError(f"{name} must be a positive {kind}")
    return value


def load_flash_config(path: Path) -> dict[str, Any]:
    """Load and validate provider settings without reading credentials."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FlashError(f"Runpod config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise FlashError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FlashError(f"Runpod config must be a YAML mapping: {path}")
    if raw.get("schema_version") != 1:
        raise FlashError("schema_version must be 1")
    def contains_api_key(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in {"api_key", "runpod_api_key"} or contains_api_key(nested):
                    return True
        elif isinstance(value, list):
            return any(contains_api_key(item) for item in value)
        return False

    if contains_api_key(raw):
        raise FlashError(
            "API credentials must only be supplied through RUNPOD_API_KEY or "
            f"{DEFAULT_RUNPOD_CREDENTIALS_PATH}"
        )

    machine = _require(raw, "machine", dict, "config")
    host = _require(raw, "host", dict, "config")
    driver = raw.get("driver", {})
    if not isinstance(driver, dict):
        raise FlashError("config.driver must be a mapping")

    image = machine.get("image_name")
    if not isinstance(image, str) or not image.strip():
        raise FlashError("machine.image_name is required by this initial image-based driver")
    if machine.get("template_id"):
        raise FlashError("machine.template_id is not supported by this initial image-based driver")
    gpu_types = _require(machine, "gpu_type_ids", list, "machine")
    if len(gpu_types) != 1 or not isinstance(gpu_types[0], str) or not gpu_types[0].strip():
        raise FlashError("machine.gpu_type_ids must contain exactly one v2 GPU type ID")
    gpu_count = _positive_number(machine.get("gpu_count", 1), "machine.gpu_count", integer=True)
    _positive_number(machine.get("container_disk_gb", 50), "machine.container_disk_gb", integer=True)
    volume_gb = machine.get("volume_gb", 0)
    if isinstance(volume_gb, bool) or not isinstance(volume_gb, int) or (
        volume_gb != 0 and volume_gb < 10
    ):
        raise FlashError("machine.volume_gb must be zero or at least 10 GB")
    if volume_gb and machine.get("network_volume_id"):
        raise FlashError("machine.volume_gb and machine.network_volume_id are mutually exclusive")
    if machine.get("min_cuda_version") is not None:
        raise FlashError(
            "machine.min_cuda_version is not supported by Runpod's Python SDK; "
            "use machine.allowed_cuda_versions instead"
        )
    data_center_ids = machine.get("data_center_ids")
    if data_center_ids is not None and (
        not isinstance(data_center_ids, list)
        or len(data_center_ids) != 1
        or not isinstance(data_center_ids[0], str)
        or not data_center_ids[0].strip()
    ):
        raise FlashError(
            "machine.data_center_ids must contain exactly one data-center ID when using "
            "Runpod's Python SDK"
        )
    if machine.get("global_networking") is not None:
        raise FlashError("machine.global_networking is not supported by Runpod's Python SDK")
    ports = machine.get("ports", ["22/tcp"])
    if not isinstance(ports, list) or "22/tcp" not in ports:
        raise FlashError("machine.ports must include 22/tcp so the harness can connect")

    host_data = dict(host)
    host_data.pop("alias", None)
    host_data["ssh_target"] = "pending.invalid"
    host_data["ssh_port"] = 22
    if "gpu_ids" not in host_data:
        host_data["gpu_ids"] = list(range(int(gpu_count)))
    try:
        HostConfig.model_validate(host_data)
    except Exception as exc:
        raise FlashError(f"Invalid host settings: {exc}") from exc

    for name, default in (
        ("readiness_timeout_seconds", DEFAULT_READY_TIMEOUT),
        ("poll_interval_seconds", DEFAULT_POLL_INTERVAL),
    ):
        _positive_number(driver.get(name, default), f"driver.{name}")
    return raw


def build_create_kwargs(config: Mapping[str, Any], request_id: str) -> dict[str, Any]:
    """Translate local settings to the official ``runpod.create_pod`` arguments."""
    machine = config["machine"]
    base_name = str(machine.get("name", "toyai-flash")).strip()
    create_kwargs: dict[str, Any] = {
        "name": f"{base_name}-{request_id}",
        "image_name": machine["image_name"],
        "gpu_type_id": machine["gpu_type_ids"][0],
        "gpu_count": int(machine.get("gpu_count", 1)),
        "container_disk_in_gb": int(machine.get("container_disk_gb", 50)),
        "cloud_type": str(machine.get("cloud_type", "SECURE")).upper(),
        "ports": ",".join(str(port) for port in machine.get("ports", ["22/tcp"])),
        # The released SDK uses this to preserve an externally reachable SSH
        # endpoint. The local readiness check relies on that direct connection.
        "support_public_ip": True,
        "start_ssh": True,
    }
    optional = {
        "allowed_cuda_versions": "allowed_cuda_versions",
        "min_vcpu_per_gpu": "min_vcpu_count",
        "min_ram_per_gpu": "min_memory_in_gb",
    }
    for local_name, sdk_name in optional.items():
        if machine.get(local_name) is not None:
            create_kwargs[sdk_name] = machine[local_name]

    mount_path = str(machine.get("volume_mount_path", "/workspace"))
    if machine.get("network_volume_id"):
        create_kwargs["network_volume_id"] = machine["network_volume_id"]
        create_kwargs["volume_mount_path"] = mount_path
    elif machine.get("volume_gb", 0):
        create_kwargs["volume_in_gb"] = int(machine["volume_gb"])
        create_kwargs["volume_mount_path"] = mount_path
    if machine.get("data_center_ids") is not None:
        create_kwargs["data_center_id"] = machine["data_center_ids"][0]
    return create_kwargs


class RunpodClient:
    """Small adapter around the official Runpod Python SDK's Pod methods."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        sdk: Optional[Any] = None,
        credentials_path: Path = DEFAULT_RUNPOD_CREDENTIALS_PATH,
    ) -> None:
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY") or load_runpod_api_key(credentials_path)
        if not self.api_key:
            raise FlashError(
                "Runpod API key not found. Set RUNPOD_API_KEY or add default.api_key to "
                f"{credentials_path}"
            )
        if sdk is None:
            try:
                sdk = importlib.import_module("runpod")
            except ImportError as exc:  # pragma: no cover - enforced by package dependency
                raise FlashError("Runpod SDK is not installed; run `uv sync` from llm-serving/") from exc
        self._sdk = sdk
        # The official Pod SDK uses this module-level credential for lifecycle calls.
        self._sdk.api_key = self.api_key

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        for value in (getattr(exc, "status_code", None), getattr(exc, "status", None)):
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        return value if isinstance(value, int) else None

    def _call(
        self,
        operation: str,
        callable_: Callable[..., Any],
        *args: Any,
        missing_status_code: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            result = callable_(*args, **kwargs)
        except Exception as exc:
            raise RunpodAPIError(
                f"Runpod SDK {operation} failed: {exc}", status_code=self._status_code(exc)
            ) from exc
        if result is None and missing_status_code is not None:
            raise RunpodAPIError(
                f"Runpod SDK {operation} did not find the requested Pod",
                status_code=missing_status_code,
            )
        if not isinstance(result, Mapping):
            raise RunpodAPIError(
                f"Runpod SDK {operation} returned an invalid Pod response; reconcile in the Runpod console"
            )
        return self._normalize_pod(result)

    @staticmethod
    def _normalize_pod(pod: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize released-SDK GraphQL responses to the harness's Pod shape.

        The current PyPI SDK exposes ``desiredStatus`` and ``runtime.ports``;
        the newer Pod API exposes ``status`` and ``ssh.direct``. Preserve the
        latter when present, while deriving it from the former for a direct SSH
        port mapping. The configured host SSH user remains authoritative when
        the SDK response does not include one.
        """
        normalized = dict(pod)
        if not normalized.get("status") and normalized.get("desiredStatus"):
            normalized["status"] = normalized["desiredStatus"]
        ssh = normalized.get("ssh")
        direct = ssh.get("direct") if isinstance(ssh, Mapping) else None
        if isinstance(direct, Mapping):
            return normalized

        runtime = normalized.get("runtime")
        ports = runtime.get("ports") if isinstance(runtime, Mapping) else None
        if not isinstance(ports, list):
            return normalized
        for port in ports:
            if not isinstance(port, Mapping) or str(port.get("privatePort")) != "22":
                continue
            host = port.get("ip")
            public_port = port.get("publicPort")
            try:
                port_number = int(public_port)
            except (TypeError, ValueError):
                continue
            if isinstance(host, str) and host and port_number > 0:
                normalized["ssh"] = {"direct": {"host": host, "port": port_number}}
                break
        return normalized

    def create_pod(self, create_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        # Deliberately one attempt. A retry after an ambiguous network failure could
        # create a second billable Pod.
        return self._call("create_pod", self._sdk.create_pod, **dict(create_kwargs))

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        result = self._call("get_pod", self._sdk.get_pod, pod_id, missing_status_code=404)
        return result

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._call("stop_pod", self._sdk.stop_pod, pod_id)

    def start_pod(self, pod_id: str, gpu_count: int) -> dict[str, Any]:
        return self._call("resume_pod", self._sdk.resume_pod, pod_id, gpu_count)


class StateStore:
    """Private local lifecycle state under ``.flash``."""

    def __init__(self, root: Path):
        self.root = root
        self.requests_dir = root / "requests"
        self.pods_dir = root / "pods"
        self.hosts_dir = root / "hosts"
        for directory in (self.root, self.requests_dir, self.pods_dir, self.hosts_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.chmod(0o600)
        os.replace(temp, path)

    def write_request(self, request_id: str, value: Mapping[str, Any]) -> Path:
        path = self.requests_dir / f"{request_id}.json"
        self._write_json(path, value)
        return path

    def write_pod(self, pod_id: str, value: Mapping[str, Any]) -> Path:
        path = self.pods_dir / f"{pod_id}.json"
        self._write_json(path, value)
        return path

    def read_pod(self, pod_id: str) -> dict[str, Any]:
        path = self.pods_dir / f"{pod_id}.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FlashError(f"Pod {pod_id} is not tracked in {self.pods_dir}") from exc
        if not isinstance(result, dict):
            raise FlashError(f"Invalid Pod state file: {path}")
        return result

    def latest_pod_id(self) -> str:
        paths = sorted(self.pods_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not paths:
            raise FlashError(f"No tracked Pods found in {self.pods_dir}")
        return paths[0].stem

    def write_host_inventory(self, pod_id: str, inventory: HostInventory) -> Path:
        path = self.hosts_dir / f"{pod_id}.yaml"
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        temp.write_text(yaml.safe_dump(inventory.model_dump(), sort_keys=False), encoding="utf-8")
        temp.chmod(0o600)
        os.replace(temp, path)
        return path


def resolve_state_dir(config: Mapping[str, Any], config_path: Path) -> Path:
    configured = config.get("driver", {}).get("state_dir", ".flash")
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def pod_status(pod: Mapping[str, Any]) -> str:
    return str(
        pod.get("status") or pod.get("last_status") or pod.get("desiredStatus") or "UNKNOWN"
    ).upper()


def _validate_pod_id(pod_id: str) -> str:
    if not pod_id or not all(character.isalnum() or character in "_-" for character in pod_id):
        raise FlashError(f"Invalid Pod ID: {pod_id!r}")
    return pod_id


def resolve_host_config(config: Mapping[str, Any], pod: Mapping[str, Any]) -> tuple[str, HostConfig]:
    pod_id = str(pod.get("id") or "")
    ssh = pod.get("ssh") or {}
    direct = ssh.get("direct") if isinstance(ssh, dict) else None
    if not isinstance(direct, dict):
        raise FlashError(f"Pod {pod_id or '<unknown>'} does not have direct SSH details yet")
    ssh_host = direct.get("host")
    ssh_port = direct.get("port")
    ssh_user = direct.get("username")
    if not pod_id or not isinstance(ssh_host, str) or not ssh_host or not ssh_port:
        raise FlashError(f"Pod {pod_id or '<unknown>'} has incomplete direct SSH details")
    host_raw = dict(config["host"])
    alias = str(host_raw.pop("alias", "runpod"))
    host_raw["ssh_target"] = ssh_host
    host_raw["ssh_port"] = int(ssh_port)
    if isinstance(ssh_user, str) and ssh_user:
        host_raw["ssh_user"] = ssh_user
    if "gpu_ids" not in host_raw:
        host_raw["gpu_ids"] = list(range(int(config["machine"].get("gpu_count", 1))))
    try:
        return alias, HostConfig.model_validate(host_raw)
    except Exception as exc:
        raise FlashError(f"Runpod host mapping is invalid: {exc}") from exc


def probe_gpu(host: HostConfig, timeout: float) -> str:
    """Verify that SSH is usable and the expected GPU is visible."""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={max(1, int(timeout))}",
        "-p", str(host.ssh_port),
    ]
    if host.ssh_key_path:
        cmd.extend(["-i", host.ssh_key_path])
    target = f"{host.ssh_user}@{host.ssh_target}" if host.ssh_user else host.ssh_target
    cmd.extend([target, "nvidia-smi --query-gpu=name --format=csv,noheader"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FlashError(f"SSH/GPU readiness probe failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise FlashError(f"SSH/GPU readiness probe failed: {detail}")
    names = result.stdout.strip()
    if host.expected_gpu_model.lower() not in names.lower():
        raise FlashError(
            f"GPU readiness probe saw {names!r}, expected model containing {host.expected_gpu_model!r}"
        )
    return names


def wait_until_ready(
    client: RunpodClient,
    pod_id: str,
    config: Mapping[str, Any],
    *,
    timeout: float,
    poll_interval: float,
    probe: Callable[[HostConfig, float], str] = probe_gpu,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], str, HostConfig, str]:
    deadline = clock() + timeout
    last_detail = "Pod has not reported a state"
    while clock() < deadline:
        pod = client.get_pod(pod_id)
        status = pod_status(pod)
        if status in {"ERROR", "EXITED", "STOPPED", "TERMINATED"}:
            raise FlashError(f"Pod {pod_id} entered {status} before becoming ready")
        try:
            alias, host = resolve_host_config(config, pod)
            if status != "RUNNING":
                raise FlashError(f"Pod state is {status}, waiting for RUNNING")
            remaining = max(1.0, deadline - clock())
            gpu_names = probe(host, min(15.0, remaining))
            return pod, alias, host, gpu_names
        except FlashError as exc:
            last_detail = str(exc)
        remaining = deadline - clock()
        if remaining > 0:
            sleeper(min(poll_interval, remaining))
    raise FlashError(f"Pod {pod_id} was not SSH/GPU-ready within {timeout:g}s: {last_detail}")


@dataclass(frozen=True)
class CreateResult:
    pod_id: str
    state_path: Path
    host_inventory_path: Optional[Path]
    pod: Mapping[str, Any]


def create(
    config_path: Path,
    *,
    wait: bool = False,
    client: Optional[RunpodClient] = None,
    probe: Callable[[HostConfig, float], str] = probe_gpu,
) -> CreateResult:
    config_path = config_path.resolve()
    config = load_flash_config(config_path)
    driver = config.get("driver", {})
    store = StateStore(resolve_state_dir(config, config_path))
    request_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"
    create_kwargs = build_create_kwargs(config, request_id)
    request_state: dict[str, Any] = {
        "request_id": request_id,
        "created_at": _utc_now(),
        "outcome": "pending",
        "config_path": str(config_path),
        "pod_name": create_kwargs["name"],
        "create_kwargs": create_kwargs,
    }
    store.write_request(request_id, request_state)
    api = client or RunpodClient()
    try:
        pod = api.create_pod(create_kwargs)
    except RunpodAPIError as exc:
        # A provider error can arrive after a billable Pod was created. Timeouts
        # and server failures are ambiguous, even when surfaced by the SDK.
        definite_rejection = (
            exc.status_code is not None and 400 <= exc.status_code < 500
            and exc.status_code != 408
        )
        outcome = "rejected" if definite_rejection else "uncertain"
        request_state.update(outcome=outcome, updated_at=_utc_now(), error=str(exc))
        store.write_request(request_id, request_state)
        if not definite_rejection:
            raise RunpodAPIError(
                f"{exc}. Create outcome is uncertain; reconcile Pod name {create_kwargs['name']!r} "
                "in the Runpod console before another create request.",
                status_code=exc.status_code,
            ) from exc
        raise
    except Exception as exc:
        request_state.update(outcome="uncertain", updated_at=_utc_now(), error=str(exc))
        store.write_request(request_id, request_state)
        raise
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        request_state.update(outcome="invalid_response", updated_at=_utc_now())
        store.write_request(request_id, request_state)
        raise FlashError("Runpod created a resource but returned no Pod ID; reconcile by unique Pod name")
    _validate_pod_id(pod_id)

    request_state.update(outcome="confirmed", pod_id=pod_id, updated_at=_utc_now())
    try:
        store.write_request(request_id, request_state)
    except OSError as exc:
        raise FlashError(
            f"Runpod created Pod {pod_id} ({create_kwargs['name']}) but local request state could not be "
            f"updated: {exc}. Reconcile and stop it in the Runpod console; do not create another Pod."
        ) from exc

    state: dict[str, Any] = {
        "pod_id": pod_id,
        "owned": True,
        "request_id": request_id,
        "pod_name": create_kwargs["name"],
        "config_path": str(config_path),
        "created_at": _utc_now(),
        "last_observed_at": _utc_now(),
        "last_status": pod_status(pod),
        "host_inventory_path": None,
    }
    try:
        state_path = store.write_pod(pod_id, state)
    except OSError as exc:
        raise FlashError(
            f"Runpod created Pod {pod_id} ({create_kwargs['name']}) but local Pod state could not be "
            f"written: {exc}. Reconcile and stop it in the Runpod console; do not create another Pod."
        ) from exc

    host_path: Optional[Path] = None
    if wait:
        try:
            pod, alias, host, gpu_names = wait_until_ready(
                api,
                pod_id,
                config,
                timeout=float(driver.get("readiness_timeout_seconds", DEFAULT_READY_TIMEOUT)),
                poll_interval=float(driver.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)),
                probe=probe,
            )
            inventory = HostInventory(schema_version=1, hosts={alias: host})
            host_path = store.write_host_inventory(pod_id, inventory)
            state.update(
                last_observed_at=_utc_now(),
                last_status=pod_status(pod),
                gpu_names=gpu_names.splitlines(),
                host_inventory_path=str(host_path),
            )
        except Exception as exc:
            state.update(last_observed_at=_utc_now(), last_status="READINESS_FAILED")
            store.write_pod(pod_id, state)
            raise FlashError(
                f"Pod {pod_id} was created but did not become ready: {exc}. It may still be billable; "
                f"inspect with `uv run flash --config {shlex.quote(str(config_path))} status {pod_id}` "
                f"and stop with `uv run flash --config {shlex.quote(str(config_path))} stop {pod_id}`."
            ) from exc
        state_path = store.write_pod(pod_id, state)
    return CreateResult(pod_id, state_path, host_path, pod)


def refresh_status(
    config_path: Path,
    pod_id: Optional[str] = None,
    *,
    client: Optional[RunpodClient] = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = config_path.resolve()
    config = load_flash_config(config_path)
    store = StateStore(resolve_state_dir(config, config_path))
    selected = pod_id or store.latest_pod_id()
    _validate_pod_id(selected)
    state = store.read_pod(selected)
    api = client or RunpodClient()
    pod = api.get_pod(selected)
    state.update(last_observed_at=_utc_now(), last_status=pod_status(pod))
    path = store.write_pod(selected, state)
    return pod, state, path


def stop(
    config_path: Path,
    pod_id: str,
    *,
    client: Optional[RunpodClient] = None,
) -> tuple[dict[str, Any], bool]:
    config_path = config_path.resolve()
    config = load_flash_config(config_path)
    store = StateStore(resolve_state_dir(config, config_path))
    selected = pod_id
    _validate_pod_id(selected)
    state = store.read_pod(selected)
    api = client or RunpodClient()
    try:
        pod = api.get_pod(selected)
    except RunpodAPIError as exc:
        if exc.status_code == 404:
            state.update(last_observed_at=_utc_now(), last_status="NOT_FOUND")
            store.write_pod(selected, state)
            return state, False
        raise
    if pod_status(pod) in {"ERROR", "EXITED", "STOPPED", "TERMINATED"}:
        state.update(last_observed_at=_utc_now(), last_status=pod_status(pod))
        store.write_pod(selected, state)
        return pod, False
    try:
        result = api.stop_pod(selected)
    except RunpodAPIError as exc:
        if exc.status_code != 409:
            raise
        current = api.get_pod(selected)
        if pod_status(current) not in {"ERROR", "EXITED", "TERMINATED"}:
            raise
        state.update(last_observed_at=_utc_now(), last_status=pod_status(current))
        store.write_pod(selected, state)
        return current, False
    if not result or result.get("id") != selected or pod_status(result) == "UNKNOWN":
        state.update(last_observed_at=_utc_now(), last_status="STOP_OUTCOME_UNKNOWN")
        store.write_pod(selected, state)
        raise RunpodAPIError(
            f"Runpod SDK accepted the stop request for Pod {selected} but returned an invalid Pod "
            "response. Do not repeat the action blindly; run status to reconcile its state."
        )
    state.update(last_observed_at=_utc_now(), last_status=pod_status(result))
    store.write_pod(selected, state)
    return result, True


def start(
    config_path: Path,
    pod_id: str,
    *,
    wait: bool = False,
    client: Optional[RunpodClient] = None,
    probe: Callable[[HostConfig, float], str] = probe_gpu,
) -> tuple[dict[str, Any], Optional[Path], bool]:
    config_path = config_path.resolve()
    config = load_flash_config(config_path)
    driver = config.get("driver", {})
    store = StateStore(resolve_state_dir(config, config_path))
    selected = pod_id
    _validate_pod_id(selected)
    state = store.read_pod(selected)
    api = client or RunpodClient()
    pod = api.get_pod(selected)
    status = pod_status(pod)
    if status == "TERMINATED":
        raise FlashError(f"Pod {selected} is terminated and cannot be restarted")
    if status in {"EXITED", "ERROR"}:
        changed = True
    elif status in {"PROVISIONING", "STARTING", "RUNNING"}:
        changed = False
    else:
        raise FlashError(f"Pod {selected} has unknown state {status}; refusing to send start")
    if changed:
        pod = api.start_pod(selected, int(config["machine"].get("gpu_count", 1)))
    host_path: Optional[Path] = None
    if wait:
        try:
            pod, alias, host, gpu_names = wait_until_ready(
                api,
                selected,
                config,
                timeout=float(driver.get("readiness_timeout_seconds", DEFAULT_READY_TIMEOUT)),
                poll_interval=float(driver.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)),
                probe=probe,
            )
            host_path = store.write_host_inventory(
                selected, HostInventory(schema_version=1, hosts={alias: host})
            )
            state["gpu_names"] = gpu_names.splitlines()
            state["host_inventory_path"] = str(host_path)
        except Exception as exc:
            state.update(last_observed_at=_utc_now(), last_status="READINESS_FAILED")
            store.write_pod(selected, state)
            raise FlashError(
                f"Pod {selected} did not become ready after start: {exc}. It may still be billable; "
                f"inspect with `uv run flash --config {shlex.quote(str(config_path))} status {selected}` "
                f"and stop with `uv run flash --config {shlex.quote(str(config_path))} stop {selected}`."
            ) from exc
    state.update(last_observed_at=_utc_now(), last_status=pod_status(pod))
    store.write_pod(selected, state)
    return pod, host_path, changed


def _default_config_path() -> Path:
    return Path("runpod.local.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flash",
        description="Local Runpod Pod lifecycle driver for llm-serving (does not terminate Pods)",
    )
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Runpod YAML config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Print the resolved create request without provisioning")
    create_parser = sub.add_parser("create", help="Create one Pod exactly once")
    create_parser.add_argument("--wait", action="store_true", help="Wait for SSH and GPU readiness")
    status_parser = sub.add_parser("status", help="Refresh a tracked Pod's status")
    status_parser.add_argument("pod_id", nargs="?", help="Tracked Pod ID (defaults to most recent)")
    stop_parser = sub.add_parser("stop", help="Stop a tracked Pod; preserves volume data and may retain storage charges")
    stop_parser.add_argument("pod_id", help="Tracked Pod ID")
    start_parser = sub.add_parser("start", help="Start a tracked stopped Pod")
    start_parser.add_argument("pod_id", help="Tracked Pod ID")
    start_parser.add_argument("--wait", action="store_true", help="Wait for SSH and GPU readiness")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            config = load_flash_config(args.config)
            create_kwargs = build_create_kwargs(config, "<unique-request-id>")
            output = {
                "sdk": "runpod.create_pod",
                "create_kwargs": create_kwargs,
                "state_dir": str(resolve_state_dir(config, args.config.resolve())),
                "credentials": (
                    "RUNPOD_API_KEY override or ~/.runpod/config.toml [default].api_key "
                    "(not read during plan)"
                ),
            }
            print(yaml.safe_dump(output, sort_keys=False).rstrip())
            return 0
        if args.command == "create":
            result = create(args.config, wait=args.wait)
            print(f"Pod: {result.pod_id}")
            print(f"State: {result.state_path}")
            if result.host_inventory_path:
                print(f"Host inventory: {result.host_inventory_path}")
            else:
                print(f"Run `uv run flash --config {shlex.quote(str(args.config))} status {result.pod_id}` to inspect it.")
            return 0
        if args.command == "status":
            pod, state, path = refresh_status(args.config, args.pod_id)
            print(f"Pod: {state['pod_id']}")
            print(f"Status: {pod_status(pod)}")
            print(f"State: {path}")
            if state.get("host_inventory_path"):
                print(f"Host inventory: {state['host_inventory_path']}")
            return 0
        if args.command == "stop":
            pod, changed = stop(args.config, args.pod_id)
            print(f"Pod: {pod.get('id') or pod.get('pod_id') or args.pod_id}")
            print(f"Status: {pod_status(pod)}")
            print("Stop requested." if changed else "Already stopped, terminated, or absent; no stop request sent.")
            print("This command does not terminate the Pod. Volume storage can continue to incur charges.")
            return 0
        if args.command == "start":
            pod, host_path, changed = start(args.config, args.pod_id, wait=args.wait)
            print(f"Pod: {pod.get('id') or args.pod_id}")
            print(f"Status: {pod_status(pod)}")
            print("Start requested." if changed else f"Pod is already {pod_status(pod).lower()}; no start request sent.")
            if host_path:
                print(f"Host inventory: {host_path}")
            return 0
    except (FlashError, RunpodAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

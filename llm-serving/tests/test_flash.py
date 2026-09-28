import json
from pathlib import Path

import pytest
import yaml

from llm_serving.flash import (
    FlashError,
    RunpodAPIError,
    RunpodClient,
    StateStore,
    build_create_payload,
    create,
    load_flash_config,
    resolve_host_config,
    start,
    stop,
    wait_until_ready,
)


def write_config(tmp_path: Path, **driver_overrides) -> Path:
    config = {
        "schema_version": 1,
        "machine": {
            "name": "test-pod",
            "image_name": "runpod/pytorch:test",
            "gpu_type_ids": ["NVIDIA GeForce RTX 4090"],
            "gpu_count": 1,
            "container_disk_gb": 20,
            "volume_gb": 20,
            "ports": ["22/tcp"],
        },
        "host": {
            "alias": "test-runpod",
            "ssh_user": "root",
            "ssh_key_path": None,
            "execution_mode": "host",
            "python_executable": "python3",
            "environment": {},
            "remote_root": "/workspace/llm-serving",
            "gpu_ids": [0],
            "expected_gpu_model": "RTX 4090",
            "hf_cache_path": "/workspace/.cache/huggingface",
            "vllm_cache_path": "/workspace/.cache/vllm",
            "secret_env_file": None,
            "min_disk_space_gb": 1,
        },
        "driver": {"state_dir": ".flash", **driver_overrides},
    }
    path = tmp_path / "runpod.local.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class FakeClient:
    def __init__(self, pod=None, create_error=None):
        self.pod = pod or {"id": "pod-123", "status": "RUNNING"}
        self.create_error = create_error
        self.create_calls = 0
        self.stop_calls = 0
        self.start_calls = 0

    def create_pod(self, payload):
        self.create_calls += 1
        if self.create_error:
            raise self.create_error
        self.pod = {**self.pod, "name": payload["name"]}
        return dict(self.pod)

    def get_pod(self, pod_id):
        assert pod_id == self.pod["id"]
        return dict(self.pod)

    def stop_pod(self, pod_id):
        self.stop_calls += 1
        self.pod["status"] = "EXITED"
        return dict(self.pod)

    def start_pod(self, pod_id):
        self.start_calls += 1
        self.pod["status"] = "RUNNING"
        return dict(self.pod)


def ready_pod(status="RUNNING"):
    return {
        "id": "pod-123",
        "status": status,
        "ssh": {
            "direct": {"host": "203.0.113.10", "port": 10341, "username": "root"},
            "proxy": None,
        },
    }


def test_plan_payload_uses_documented_api_fields_and_no_credentials(tmp_path):
    path = write_config(tmp_path)
    config = load_flash_config(path)
    payload = build_create_payload(config, "request-1")

    assert payload["name"] == "test-pod-request-1"
    assert payload["image"] == "runpod/pytorch:test"
    assert payload["gpu"] == {"id": "NVIDIA GeForce RTX 4090", "count": 1}
    assert payload["mounts"] == {"persistent": {"size": 20, "path": "/workspace"}}
    assert payload["ports"] == ["22/tcp"]
    assert payload["startSsh"] is True
    assert not any("key" in key.lower() or "token" in key.lower() for key in payload)


def test_config_rejects_embedded_api_key(tmp_path):
    path = write_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["api_key"] = "secret"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(FlashError, match="RUNPOD_API_KEY"):
        load_flash_config(path)


def test_config_rejects_api_key_in_generated_host_environment(tmp_path):
    path = write_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["host"]["environment"]["RUNPOD_API_KEY"] = "must-not-be-persisted"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(FlashError, match="RUNPOD_API_KEY"):
        load_flash_config(path)


def test_create_wait_persists_id_and_emits_valid_host_inventory(tmp_path):
    path = write_config(tmp_path)
    client = FakeClient(ready_pod())

    result = create(path, wait=True, client=client, probe=lambda host, timeout: "NVIDIA GeForce RTX 4090")

    assert result.pod_id == "pod-123"
    assert client.create_calls == 1
    state = json.loads(result.state_path.read_text())
    assert state["owned"] is True
    assert state["last_status"] == "RUNNING"
    assert "api_key" not in json.dumps(state).lower()
    inventory = yaml.safe_load(result.host_inventory_path.read_text())
    host = inventory["hosts"]["test-runpod"]
    assert host["ssh_target"] == "203.0.113.10"
    assert host["ssh_port"] == 10341


def test_create_does_not_retry_uncertain_response_and_records_request(tmp_path):
    path = write_config(tmp_path)
    client = FakeClient(create_error=RunpodAPIError("uncertain"))

    with pytest.raises(RunpodAPIError, match="uncertain"):
        create(path, client=client)

    assert client.create_calls == 1
    request_files = list((tmp_path / ".flash" / "requests").glob("*.json"))
    assert len(request_files) == 1
    assert json.loads(request_files[0].read_text())["outcome"] == "uncertain"


def test_stop_is_idempotent_for_stopped_tracked_pod(tmp_path):
    path = write_config(tmp_path)
    client = FakeClient(ready_pod("EXITED"))
    store = StateStore(tmp_path / ".flash")
    store.write_pod("pod-123", {"pod_id": "pod-123", "owned": True})

    pod, changed = stop(path, "pod-123", client=client)

    assert changed is False
    assert pod["status"] == "EXITED"
    assert client.stop_calls == 0


@pytest.mark.parametrize("status_code,expected", [(500, "uncertain"), (502, "uncertain"), (408, "uncertain"), (400, "rejected")])
def test_create_failure_classification_does_not_invite_duplicate_pods(tmp_path, status_code, expected):
    path = write_config(tmp_path)
    client = FakeClient(create_error=RunpodAPIError("request failed", status_code=status_code))
    with pytest.raises(RunpodAPIError) as caught:
        create(path, client=client)
    assert client.create_calls == 1
    request_file = next((tmp_path / ".flash" / "requests").glob("*.json"))
    record = json.loads(request_file.read_text())
    assert record["outcome"] == expected
    if expected == "uncertain":
        assert record["pod_name"] in str(caught.value)
        assert "reconcile" in str(caught.value)


def test_stop_sends_one_request_for_running_tracked_pod(tmp_path):
    path = write_config(tmp_path)
    client = FakeClient(ready_pod())
    StateStore(tmp_path / ".flash").write_pod("pod-123", {"pod_id": "pod-123", "owned": True})

    pod, changed = stop(path, "pod-123", client=client)

    assert changed is True
    assert pod["status"] == "EXITED"
    assert client.stop_calls == 1


def test_stop_rejects_empty_v2_response_and_marks_outcome_unknown(tmp_path):
    path = write_config(tmp_path)
    client = FakeClient(ready_pod())
    client.stop_pod = lambda pod_id: {}
    StateStore(tmp_path / ".flash").write_pod("pod-123", {"pod_id": "pod-123", "owned": True})

    with pytest.raises(RunpodAPIError, match="invalid v2 Pod response"):
        stop(path, "pod-123", client=client)

    state = json.loads((tmp_path / ".flash" / "pods" / "pod-123.json").read_text())
    assert state["last_status"] == "STOP_OUTCOME_UNKNOWN"


def test_stop_refuses_untracked_pod(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(FlashError, match="not tracked"):
        stop(path, "someone-elses-pod", client=FakeClient())


def test_statusing_old_pod_cannot_change_explicit_stop_target(tmp_path):
    path = write_config(tmp_path)
    store = StateStore(tmp_path / ".flash")
    store.write_pod("pod-old", {"pod_id": "pod-old", "owned": True})
    store.write_pod("pod-new", {"pod_id": "pod-new", "owned": True})

    class TwoPodClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.stopped_id = None

        def get_pod(self, pod_id):
            return {"id": pod_id, "status": "RUNNING"}

        def stop_pod(self, pod_id):
            self.stopped_id = pod_id
            return {"id": pod_id, "status": "EXITED"}

    client = TwoPodClient()
    stop(path, "pod-new", client=client)
    assert client.stopped_id == "pod-new"


def test_start_wait_is_idempotent_and_refreshes_host_inventory(tmp_path):
    path = write_config(tmp_path)
    client = FakeClient(ready_pod("EXITED"))
    StateStore(tmp_path / ".flash").write_pod("pod-123", {"pod_id": "pod-123", "owned": True})

    pod, host_path, changed = start(
        path,
        "pod-123",
        wait=True,
        client=client,
        probe=lambda host, timeout: "NVIDIA GeForce RTX 4090",
    )

    assert changed is True
    assert client.start_calls == 1
    assert pod["status"] == "RUNNING"
    assert host_path.exists()


def test_create_reports_pod_id_if_local_state_write_fails(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    original = StateStore.write_pod

    def fail_write(self, pod_id, value):
        raise OSError("disk full")

    monkeypatch.setattr(StateStore, "write_pod", fail_write)
    with pytest.raises(FlashError, match=r"created Pod pod-123.*disk full.*Runpod console"):
        create(path, client=FakeClient(ready_pod("PROVISIONING")))
    monkeypatch.setattr(StateStore, "write_pod", original)

    request = json.loads(next((tmp_path / ".flash" / "requests").glob("*.json")).read_text())
    assert request["outcome"] == "confirmed"
    assert request["pod_id"] == "pod-123"


def test_wait_fails_fast_if_pod_exits(tmp_path):
    config = load_flash_config(write_config(tmp_path))
    with pytest.raises(FlashError, match="entered EXITED"):
        wait_until_ready(
            FakeClient(ready_pod("EXITED")),
            "pod-123",
            config,
            timeout=10,
            poll_interval=1,
            probe=lambda host, timeout: "unused",
        )


def test_host_resolution_requires_direct_ssh_details(tmp_path):
    config = load_flash_config(write_config(tmp_path))
    with pytest.raises(FlashError, match="direct SSH details"):
        resolve_host_config(config, {"id": "pod-123", "status": "RUNNING"})


def test_stdlib_client_sends_bearer_header_without_logging_key():
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"id":"pod-123"}'

    def opener(request, timeout):
        seen["authorization"] = request.headers["Authorization"]
        seen["url"] = request.full_url
        return Response()

    client = RunpodClient("top-secret", opener=opener)
    assert client.get_pod("pod-123") == {"id": "pod-123"}
    assert seen == {
        "authorization": "Bearer top-secret",
        "url": "https://api.runpod.io/v2/pods/pod-123",
    }


def test_stdlib_client_uses_v2_action_contract_for_stop():
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"id":"pod-123","status":"EXITED"}'

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.method
        seen["body"] = json.loads(request.data)
        return Response()

    client = RunpodClient("top-secret", opener=opener)
    assert client.stop_pod("pod-123")["status"] == "EXITED"
    assert seen == {
        "url": "https://api.runpod.io/v2/pods/pod-123/action",
        "method": "POST",
        "body": {"action": "stop"},
    }

import httpx
import pytest

from probing.client import ProbeClient
from probing.contracts import ErrorDetail, ErrorEnvelope
from probing.errors import RemoteProbeError
from test_service import fake_rank_spec


def test_http_error_preserves_daemon_error_code_and_exit_category(monkeypatch) -> None:
    response = httpx.Response(
        422,
        request=httpx.Request("POST", "http://test/api/v1/jobs"),
        json=ErrorEnvelope(
            error=ErrorDetail(code="capability_error", message="unsupported model")
        ).model_dump(mode="json"),
    )
    monkeypatch.setattr(httpx, "request", lambda *_args, **_kwargs: response)
    client = ProbeClient(endpoint="http://test")

    with pytest.raises(RemoteProbeError) as captured:
        client._request("POST", "/api/v1/jobs", json={})

    assert captured.value.exit_code == 3
    assert captured.value.as_detail().code == "capability_error"
    assert captured.value.as_detail().details["http_status"] == 422


def test_run_raises_when_stream_ends_with_a_failed_job(monkeypatch) -> None:
    client = ProbeClient(endpoint="http://test")
    monkeypatch.setattr(
        client, "_request", lambda *_args, **_kwargs: {"job_id": "job-1"}
    )
    monkeypatch.setattr(client, "watch_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        client,
        "job_status",
        lambda _job_id: {
            "state": "failed",
            "error": {
                "code": "model_policy_error",
                "message": "model unavailable",
                "retryable": False,
                "details": {},
            },
        },
    )

    with pytest.raises(RemoteProbeError) as captured:
        client.run(fake_rank_spec(), listener=lambda _event: None)

    assert captured.value.exit_code == 4
    assert captured.value.as_detail().code == "model_policy_error"


def test_run_transports_agent_request_id(monkeypatch) -> None:
    client = ProbeClient(endpoint="http://test")
    captured = {}

    def request(_method, _path, **kwargs):
        captured.update(kwargs)
        return {"job_id": "job-1"}

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "watch_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        client,
        "job_status",
        lambda _job_id: {"state": "completed", "run_id": "run-1"},
    )

    result = client.run(
        fake_rank_spec(),
        listener=lambda _event: None,
        request_id="agent-retry-key",
    )

    assert result["run_id"] == "run-1"
    assert captured["request_headers"] == {"X-Request-ID": "agent-retry-key"}

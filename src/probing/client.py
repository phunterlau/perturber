from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from .contracts import ErrorDetail, ErrorEnvelope, ExperimentSpec, JobEvent
from .errors import EndpointError, RemoteProbeError
from .events import EventListener


class ProbeClient:
    def __init__(
        self,
        *,
        endpoint: str,
        token_file: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token_file = token_file
        self.timeout = timeout

    @classmethod
    def from_context(cls, context: Any) -> "ProbeClient":
        token_file = context.token_file
        if token_file is None:
            candidate = Path(context.workspace) / "server.token"
            token_file = candidate if candidate.is_file() else None
        if context.endpoint is None:
            raise EndpointError("an explicit endpoint is required")
        return cls(endpoint=context.endpoint, token_file=token_file)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token_file is not None:
            try:
                token = self.token_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise EndpointError(f"could not read token file: {exc}") from exc
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        request_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.endpoint}{path}",
                headers={**self._headers(), **(request_headers or {})},
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise EndpointError(f"daemon request failed: {exc}") from exc
        if response.status_code >= 400:
            try:
                envelope = ErrorEnvelope.model_validate(response.json())
            except Exception:
                raise EndpointError(
                    f"daemon returned HTTP {response.status_code}: {response.text}",
                    details={"status_code": response.status_code},
                )
            raise RemoteProbeError(envelope.error, status_code=response.status_code)
        if not response.content:
            return None
        return response.json()

    def plan(self, spec: ExperimentSpec) -> dict[str, Any]:
        return self._request("POST", "/api/v1/plans", json=spec.model_dump(mode="json"))

    def preflight(self, spec: ExperimentSpec) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/preflight", json=spec.model_dump(mode="json")
        )

    def capabilities(self, spec: ExperimentSpec) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/capabilities", json=spec.model_dump(mode="json")
        )

    def run(
        self,
        spec: ExperimentSpec,
        *,
        listener: EventListener,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        status = self._request(
            "POST",
            "/api/v1/jobs",
            request_headers=(
                {"X-Request-ID": request_id} if request_id is not None else None
            ),
            json=spec.model_dump(mode="json"),
        )
        self.watch_job(status["job_id"], start_sequence=0, listener=listener)
        terminal = self.job_status(status["job_id"])
        if terminal["state"] != "completed":
            detail = terminal.get("error") or {
                "code": "runtime_error",
                "message": f"remote job ended in state {terminal['state']!r}",
            }
            raise RemoteProbeError(ErrorDetail.model_validate(detail))
        return terminal

    def watch_job(
        self,
        job_id: str,
        *,
        start_sequence: int,
        listener: EventListener,
    ) -> None:
        try:
            with httpx.stream(
                "GET",
                f"{self.endpoint}/api/v1/jobs/{job_id}/events",
                params={"start_sequence": start_sequence},
                headers={**self._headers(), "Accept": "application/x-ndjson"},
                timeout=None,
            ) as response:
                if response.status_code >= 400:
                    raw = response.read().decode("utf-8", errors="replace")
                    try:
                        envelope = ErrorEnvelope.model_validate_json(raw)
                    except Exception:
                        raise EndpointError(
                            f"event stream returned HTTP {response.status_code}: {raw}"
                        )
                    raise RemoteProbeError(
                        envelope.error, status_code=response.status_code
                    )
                for line in response.iter_lines():
                    if line:
                        listener(JobEvent.model_validate_json(line))
        except httpx.HTTPError as exc:
            raise EndpointError(f"event stream failed: {exc}") from exc

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def job_spec(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}/spec")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/jobs/{job_id}/cancel")

    def runs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/runs")

    def run_manifest(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/runs/{run_id}")

    def run_summary(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/runs/{run_id}/summary")

    def run_overview(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/runs/{run_id}/overview")

    def verify_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/runs/{run_id}/verify")

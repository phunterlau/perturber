from __future__ import annotations

from typing import Any

from .contracts import ErrorDetail


class ProbeError(Exception):
    exit_code = 5
    code = "runtime_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.hint = hint
        self.details = details or {}

    def as_detail(self) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            message=str(self),
            retryable=self.retryable,
            hint=self.hint,
            details=self.details,
        )


class SpecError(ProbeError):
    exit_code = 2
    code = "invalid_spec"


class CapabilityError(ProbeError):
    exit_code = 3
    code = "capability_error"


class BudgetError(ProbeError):
    exit_code = 3
    code = "budget_exceeded"


class ModelPolicyError(ProbeError):
    exit_code = 4
    code = "model_policy_error"


class ArtifactError(ProbeError):
    exit_code = 6
    code = "artifact_error"


class JobCancelled(ProbeError):
    exit_code = 7
    code = "job_cancelled"


class RequestConflictError(ProbeError):
    exit_code = 2
    code = "request_conflict"


class EndpointError(ProbeError):
    exit_code = 8
    code = "endpoint_error"
    retryable = True


_REMOTE_EXIT_CODES = {
    "invalid_spec": 2,
    "capability_error": 3,
    "budget_exceeded": 3,
    "model_policy_error": 4,
    "runtime_error": 5,
    "artifact_error": 6,
    "job_cancelled": 7,
    "request_conflict": 2,
    "job_interrupted": 5,
    "authentication_error": 8,
    "not_found": 8,
    "conflict": 8,
    "endpoint_error": 8,
}


class RemoteProbeError(ProbeError):
    """A daemon error whose machine-readable identity survives transport."""

    def __init__(self, detail: ErrorDetail, *, status_code: int | None = None) -> None:
        super().__init__(
            detail.message,
            hint=detail.hint,
            details={
                **detail.details,
                **({"http_status": status_code} if status_code is not None else {}),
            },
        )
        self.detail = detail.model_copy(update={"details": self.details})
        self.exit_code = _REMOTE_EXIT_CODES.get(detail.code, 5)

    def as_detail(self) -> ErrorDetail:
        return self.detail

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import threading
from typing import AsyncIterator, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .contracts import (
    ErrorDetail,
    ErrorEnvelope,
    ExperimentSpec,
    JobStatus,
    ResearchCaseCreate,
    ResearchCaseUpdate,
)
from .cases import (
    ResearchCaseRepository,
    agent_handoff,
    build_research_packet,
    finish_case_stage,
    plan_case,
    resolved_case_stage,
)
from .errors import ArtifactError, ProbeError, RequestConflictError, SpecError
from .reporting import build_overview
from .service import ResearchService
from .specs import canonical_json


TERMINAL_STATES = {"completed", "failed", "cancelled"}


class JobManager:
    def __init__(self, service: ResearchService) -> None:
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="probe-model")
        self.cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        spec: ExperimentSpec,
        request_id: str | None = None,
        *,
        before_execute: Callable[[JobStatus], None] | None = None,
        on_complete: Callable[[JobStatus], None] | None = None,
    ) -> JobStatus:
        with self._lock:
            existing = self.service.find_idempotent_job(spec, request_id)
            if existing is not None:
                return existing
            job = self.service.initialize_job(
                spec, request_id=request_id, state="queued"
            )
            cancel = threading.Event()
            self.cancel_events[job.job_id] = cancel

        if before_execute is not None:
            before_execute(job)

        def execute() -> None:
            try:
                self.service.execute(spec, job=job, cancel=cancel)
            except Exception:
                pass
            finally:
                with self._lock:
                    self.cancel_events.pop(job.job_id, None)
                if on_complete is not None:
                    try:
                        on_complete(self.service.repository.load_job(job.job_id))
                    except Exception:
                        # Scientific job state is authoritative; a case notebook can
                        # be refreshed from its durable job/run links after restart.
                        pass

        self.executor.submit(execute)
        return job

    def cancel(self, job_id: str) -> JobStatus:
        with self._lock:
            event = self.cancel_events.get(job_id)
        if event is None:
            status = self.service.repository.load_job(job_id)
            if status.state in TERMINAL_STATES:
                return status
            raise HTTPException(status_code=409, detail="job is not cancellable")
        event.set()
        return self.service.repository.load_job(job_id)

    def shutdown(self) -> None:
        with self._lock:
            for event in self.cancel_events.values():
                event.set()
        # Let queued callables observe their cancellation event so their durable
        # job status becomes "cancelled" instead of remaining "queued" forever.
        self.executor.shutdown(wait=False, cancel_futures=False)


def create_app(
    *,
    workspace: Path,
    cache_dir: Path,
    token: str,
    frontend_directory: Path | None = None,
    research_service: ResearchService | None = None,
    recover_interrupted_jobs: bool = False,
) -> FastAPI:
    service = research_service or ResearchService(
        workspace=workspace, cache_dir=cache_dir
    )
    manager = JobManager(service)
    cases = ResearchCaseRepository(workspace)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if recover_interrupted_jobs:
            service.repository.recover_interrupted_jobs()
        yield
        manager.shutdown()

    application = FastAPI(
        title="Perturbation Probing Research API",
        version="1.0.0",
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        lifespan=lifespan,
        responses={
            401: {"model": ErrorEnvelope, "description": "Authentication failed"},
            404: {"model": ErrorEnvelope, "description": "Resource not found"},
            409: {"model": ErrorEnvelope, "description": "Job state conflict"},
            422: {"model": ErrorEnvelope, "description": "Invalid or unsupported request"},
        },
    )

    def error_response(detail: ErrorDetail, status_code: int) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=ErrorEnvelope(error=detail).model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            ErrorDetail(
                code="invalid_spec",
                message="request validation failed",
                hint="Inspect the rank schema at /api/v1/openapi.json.",
                details={
                    "errors": [
                        {
                            "location": list(item["loc"]),
                            "message": item["msg"],
                            "type": item["type"],
                        }
                        for item in exc.errors()
                    ]
                },
            ),
            422,
        )

    @application.exception_handler(ProbeError)
    async def probe_error(_request: Request, exc: ProbeError) -> JSONResponse:
        status_code = (
            404
            if isinstance(exc, ArtifactError)
            else 409
            if isinstance(exc, RequestConflictError)
            else 422
        )
        return error_response(exc.as_detail(), status_code)

    @application.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        codes = {401: "authentication_error", 404: "not_found", 409: "conflict"}
        message = exc.detail if isinstance(exc.detail, str) else canonical_json(exc.detail)
        return error_response(
            ErrorDetail(code=codes.get(exc.status_code, "http_error"), message=message),
            exc.status_code,
        )

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

    auth = Depends(authenticate)

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": "probe.health/v1"}

    @application.post("/api/v1/plans", dependencies=[auth])
    def plan(spec: ExperimentSpec):
        return service.plan(spec)

    @application.post("/api/v1/preflight", dependencies=[auth])
    def preflight(spec: ExperimentSpec):
        return service.preflight(spec)

    @application.post("/api/v1/capabilities", dependencies=[auth])
    def capabilities(spec: ExperimentSpec):
        return service.capabilities(spec)

    @application.post("/api/v1/jobs", response_model=JobStatus, dependencies=[auth])
    def submit_job(
        spec: ExperimentSpec,
        x_request_id: str | None = Header(default=None),
    ) -> JobStatus:
        if x_request_id is not None:
            x_request_id = x_request_id.strip()
            if not x_request_id or len(x_request_id) > 256:
                raise SpecError(
                    "X-Request-ID must contain 1 to 256 non-whitespace characters"
                )
        existing = service.find_idempotent_job(spec, x_request_id)
        if existing is not None:
            return existing
        service.validate_execution(spec)
        return manager.submit(spec, request_id=x_request_id)

    @application.get(
        "/api/v1/jobs/{job_id}", response_model=JobStatus, dependencies=[auth]
    )
    def job_status(job_id: str) -> JobStatus:
        return service.repository.load_job(job_id)

    @application.get("/api/v1/jobs/{job_id}/spec", dependencies=[auth])
    def job_spec(job_id: str) -> ExperimentSpec:
        return service.repository.load_job_spec(job_id)

    @application.post(
        "/api/v1/jobs/{job_id}/cancel", response_model=JobStatus, dependencies=[auth]
    )
    def cancel_job(job_id: str) -> JobStatus:
        return manager.cancel(job_id)

    @application.get("/api/v1/jobs/{job_id}/events", dependencies=[auth])
    async def job_events(
        job_id: str,
        start_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        # Validate before constructing StreamingResponse so missing jobs retain
        # the versioned HTTP error contract instead of returning an error line
        # that is not a JobEvent.
        service.repository.load_job(job_id)

        async def stream() -> AsyncIterator[bytes]:
            sequence = start_sequence
            while True:
                events = service.repository.read_events(job_id, sequence)
                status = service.repository.load_job(job_id)
                for event in events:
                    yield (canonical_json(event) + "\n").encode("utf-8")
                    sequence = event.sequence + 1
                if status.state in TERMINAL_STATES and not events:
                    return
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @application.get("/api/v1/runs", dependencies=[auth])
    def runs():
        return service.repository.list_runs()

    @application.get("/api/v1/cases", dependencies=[auth])
    def list_cases():
        return [cases.refresh(item.case_id, service) for item in cases.list()]

    @application.post("/api/v1/cases", dependencies=[auth])
    def create_case(request: ResearchCaseCreate):
        created = cases.create(request)
        if request.rank_run_id is not None:
            manifest = service.repository.load_manifest(request.rank_run_id)
            if manifest.run_kind != "rank":
                raise SpecError("promoted run must be a rank run")
            saved = service.repository.load_run_spec(request.rank_run_id)
            if canonical_json(saved) != canonical_json(request.workflow.rank):
                raise SpecError("promoted rank run does not match the case rank specification")
            failures = service.repository.verify(request.rank_run_id)
            cases.update_stage(
                created.case_id,
                "rank",
                status="verified" if not failures else "failed",
                job_id=manifest.job_id,
                run_id=manifest.run_id,
                verification_failures=failures,
            )
        return cases.refresh(created.case_id, service)

    @application.get("/api/v1/cases/{case_id}", dependencies=[auth])
    def load_case(case_id: str):
        return cases.refresh(case_id, service)

    @application.put("/api/v1/cases/{case_id}", dependencies=[auth])
    def update_case(case_id: str, request: ResearchCaseUpdate):
        return cases.refresh(cases.update(case_id, request).case_id, service)

    @application.get("/api/v1/cases/{case_id}/plan", dependencies=[auth])
    def case_plan(case_id: str):
        return plan_case(cases, service, case_id)

    @application.post(
        "/api/v1/cases/{case_id}/stages/{stage_key}/preflight",
        dependencies=[auth],
    )
    def case_stage_preflight(case_id: str, stage_key: str):
        case = cases.refresh(case_id, service)
        try:
            return service.preflight(resolved_case_stage(case, stage_key))
        except ValueError as exc:
            raise SpecError(str(exc)) from exc

    @application.post(
        "/api/v1/cases/{case_id}/stages/{stage_key}/start",
        response_model=JobStatus,
        dependencies=[auth],
    )
    def start_case_stage(
        case_id: str,
        stage_key: str,
        x_request_id: str | None = Header(default=None),
    ) -> JobStatus:
        case = cases.refresh(case_id, service)
        spec = resolved_case_stage(case, stage_key)
        try:
            service.validate_execution(spec)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc

        def attach(job: JobStatus) -> None:
            cases.update_stage(
                case_id,
                stage_key,
                status="running",
                job_id=job.job_id,
                error=None,
            )

        def complete(status: JobStatus) -> None:
            finish_case_stage(cases, service, case_id, stage_key, status)

        return manager.submit(
            spec,
            request_id=x_request_id,
            before_execute=attach,
            on_complete=complete,
        )

    @application.get("/api/v1/cases/{case_id}/handoff", dependencies=[auth])
    def case_handoff(case_id: str):
        return agent_handoff(cases.refresh(case_id, service))

    @application.get("/api/v1/cases/{case_id}/packet", dependencies=[auth])
    def case_packet(case_id: str) -> FileResponse:
        packet = build_research_packet(cases, service, case_id)
        return FileResponse(
            packet,
            media_type="application/zip",
            filename=f"{case_id}-research-packet.zip",
        )

    @application.get(
        "/api/v1/cases/{case_id}/tokenize/{pair_id}", dependencies=[auth]
    )
    def tokenize_case_pair(case_id: str, pair_id: str):
        case = cases.load(case_id)
        pair = next((item for item in case.workflow.rank.pairs if item.id == pair_id), None)
        if pair is None:
            raise SpecError(f"pair {pair_id!r} was not found")
        if pair.original is None or pair.perturbed is None:
            raise SpecError("alignment preview currently requires text prompt pairs")
        snapshot = service.models.resolve_cached_snapshot(case.workflow.rank.model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )

        def tokens(text: str) -> dict[str, object]:
            if case.workflow.rank.model.chat_template:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=case.workflow.rank.model.enable_thinking,
                )
                if not isinstance(ids, (list, tuple)):
                    ids = ids["input_ids"]
            else:
                ids = tokenizer.encode(text, add_special_tokens=True)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            ids = [int(item) for item in ids]
            return {
                "ids": ids,
                "tokens": [tokenizer.decode([item]) for item in ids],
            }

        original = tokens(pair.original)
        perturbed = tokens(pair.perturbed)
        return {
            "schema_version": "probe.token-alignment-preview/v1",
            "pair_id": pair_id,
            "original": original,
            "perturbed": perturbed,
            "identity_eligible": len(original["ids"]) == len(perturbed["ids"]),
            "token_ids_identical": original["ids"] == perturbed["ids"],
        }

    @application.get("/api/v1/runs/{run_id}", dependencies=[auth])
    def run_manifest(run_id: str):
        return service.repository.load_manifest(run_id)

    @application.get("/api/v1/runs/{run_id}/spec", dependencies=[auth])
    def run_spec(run_id: str) -> ExperimentSpec:
        return service.repository.load_run_spec(run_id)

    @application.get("/api/v1/runs/{run_id}/summary", dependencies=[auth])
    def run_summary(run_id: str):
        return service.repository.load_summary(run_id)

    @application.get("/api/v1/runs/{run_id}/report", dependencies=[auth])
    def run_report(run_id: str):
        from .reporting import build_research_report

        return build_research_report(
            run_id=run_id,
            manifest=service.repository.load_manifest(run_id),
            summary=service.repository.load_summary(run_id),
        )

    @application.get("/api/v1/runs/{run_id}/overview", dependencies=[auth])
    def run_overview(run_id: str):
        manifest = service.repository.load_manifest(run_id)
        if manifest.run_kind != "rank":
            return service.repository.load_summary(run_id)
        return build_overview(
            run_id=run_id,
            summary=service.repository.load_summary(run_id),
        )

    @application.get("/api/v1/runs/{run_id}/verify", dependencies=[auth])
    def verify_run(run_id: str):
        from .contracts import VerificationReport

        failures = service.repository.verify(run_id)
        return VerificationReport(
            run_id=run_id,
            valid=not failures,
            failures=tuple(failures),
        )

    @application.get("/api/v1/runs/{run_id}/artifacts/{artifact_path:path}", dependencies=[auth])
    def run_artifact(run_id: str, artifact_path: str) -> FileResponse:
        manifest = service.repository.load_manifest(run_id)
        allowed = {item.path for item in manifest.artifacts} | {"manifest.json"}
        if artifact_path not in allowed:
            raise HTTPException(status_code=404, detail="artifact was not found")
        root = (service.repository.runs / run_id).resolve()
        candidate = (root / artifact_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(status_code=404, detail="artifact was not found")
        return FileResponse(candidate)

    @application.get("/probe-config.json")
    def probe_config() -> dict[str, str]:
        return {"token": token}

    if frontend_directory is not None and (frontend_directory / "index.html").is_file():
        application.mount(
            "/",
            StaticFiles(directory=frontend_directory, html=True),
            name="webui",
        )
    else:
        @application.get("/", response_class=HTMLResponse)
        def fallback() -> str:
            return (
                "<html><body><h1>Perturbation Probing API</h1>"
                "<p>WebUI has not been built. See <a href='/api/v1/docs'>API docs</a>.</p>"
                "</body></html>"
            )

    return application


def _write_runtime(workspace: Path, *, host: str, port: int, token: str) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    token_file = workspace / "server.token"
    token_file.write_text(token + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    runtime_file = workspace / "server.json"
    runtime_file.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "endpoint": f"http://{host}:{port}",
                "token_file": str(token_file),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_file.chmod(0o600)
    return runtime_file


def serve(*, workspace: Path, cache_dir: Path, host: str, port: int) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("v1 server must bind to a loopback host")
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as port_check:
        # Permit an immediate managed restart after the prior listener closes.
        # This does not permit a second active listener (SO_REUSEPORT would).
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            port_check.bind((host, port))
        except OSError as exc:
            raise ValueError(f"cannot bind probe server to {host}:{port}: {exc}") from exc
    token = secrets.token_urlsafe(32)
    runtime_file = _write_runtime(workspace, host=host, port=port, token=token)
    frontend = Path(__file__).resolve().parent / "web_dist"
    service = ResearchService(workspace=workspace, cache_dir=cache_dir)
    # Import Transformers after ResearchService has bound every Hugging Face
    # cache variable, but before the serialized model worker starts. On macOS,
    # first importing the Transformers/Torch stack inside that background
    # thread can stall before weight loading; model construction itself stays
    # lazy and follows the submitted spec.
    import transformers  # noqa: F401

    application = create_app(
        workspace=workspace,
        cache_dir=cache_dir,
        token=token,
        frontend_directory=frontend,
        research_service=service,
        recover_interrupted_jobs=True,
    )
    try:
        uvicorn.run(application, host=host, port=port, workers=1)
    finally:
        try:
            runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
            owns_runtime = int(runtime["pid"]) == os.getpid()
        except Exception:
            owns_runtime = False
        if owns_runtime:
            runtime_file.unlink(missing_ok=True)
            (workspace / "server.token").unlink(missing_ok=True)

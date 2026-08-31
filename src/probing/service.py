from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
import threading
import time
from typing import Any, TextIO
from uuid import uuid4

import torch
from safetensors.torch import load_file

from .aggregation import AggregateComputation, aggregate_analyses
from .artifacts import ArtifactRepository
from .contracts import (
    AttentionHeadInterventionRunSummary,
    AttentionHeadInterventionSpec,
    AttentionHeadRankRunSummary,
    AttentionHeadRankSpec,
    AttentionTraceRunSummary,
    AttentionTraceSpec,
    CapabilityReport,
    DirectionInjectionRunSummary,
    DirectionInjectionSpec,
    ErrorDetail,
    ExperimentSpec,
    ExperimentPlan,
    InterventionRunSummary,
    InterventionSpec,
    JobEvent,
    JobStatus,
    QualificationRunSummary,
    QualificationSpec,
    RankRunSummary,
    RankSpec,
    PreflightReport,
    RunManifest,
    TrajectoryRunSummary,
    TrajectorySpec,
)
from .domain import (
    ALGORITHM_VERSION,
    ATTENTION_ALGORITHM_VERSION,
    ObservableSpec,
    ProbeSpec,
    PromptPair,
)
from .engine import ProbeAnalysis, ProbeEngine
from .errors import (
    BudgetError,
    CapabilityError,
    JobCancelled,
    ProbeError,
    RequestConflictError,
)
from .events import EventEmitter, EventListener
from .models import ModelManager
from .interventions import intervention_plan_counts, run_intervention
from .attention import (
    attention_intervention_plan_counts,
    attention_rank_plan_counts,
    run_attention_intervention,
    run_attention_rank,
)
from .attention_trace import attention_trace_plan_counts, run_attention_trace
from .direction import direction_plan_counts, run_direction_injection
from .qualification_workflow import run_qualification
from .trajectory import run_trajectory, trajectory_plan_counts
from .reproducibility import seed_everything
from .specs import hash_value, request_hash, science_hash


EngineFactory = Callable[[RankSpec], ProbeEngine]


@dataclass(frozen=True)
class ExecutionOutcome:
    manifest: RunManifest
    run_directory: Path
    summary: (
        RankRunSummary
        | QualificationRunSummary
        | InterventionRunSummary
        | DirectionInjectionRunSummary
        | AttentionHeadRankRunSummary
        | AttentionHeadInterventionRunSummary
        | AttentionTraceRunSummary
        | TrajectoryRunSummary
    )


def _selected_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _selected_dtype(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


class ResearchService:
    def __init__(
        self,
        *,
        workspace: Path,
        cache_dir: Path,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.repository = ArtifactRepository(workspace)
        self.models = ModelManager(cache_dir)
        self._uses_managed_engine = engine_factory is None
        self.engine_factory = engine_factory or self._load_engine
        self._engine: ProbeEngine | None = None
        self._engine_key: tuple[Any, ...] | None = None
        self._engine_lock = threading.Lock()

    @property
    def workspace(self) -> Path:
        return self.repository.workspace

    def _load_engine(self, spec: RankSpec) -> ProbeEngine:
        request = spec.model
        key = self._engine_cache_key(spec)
        with self._engine_lock:
            if self._engine is not None and self._engine_key == key:
                return self._engine
            model_path = self.models.resolve_cached_snapshot(request)
            self._engine = ProbeEngine.from_pretrained(
                request.id,
                revision=request.revision,
                device=request.device,
                dtype=request.dtype,
                cache_dir=str(self.models.cache_dir),
                local_files_only=True,
                model_path=str(model_path),
            )
            self._engine_key = key
            return self._engine

    @staticmethod
    def _engine_cache_key(spec: RankSpec) -> tuple[Any, ...]:
        request = spec.model
        return (request.id, request.revision, request.device, request.dtype)

    def _will_reuse_engine(self, spec: RankSpec) -> bool:
        return (
            self._uses_managed_engine
            and self._engine is not None
            and self._engine_key == self._engine_cache_key(spec)
        )

    def _parent_rank(
        self,
        spec: (
            QualificationSpec
            | TrajectorySpec
            | InterventionSpec
            | DirectionInjectionSpec
            | AttentionHeadRankSpec
            | TrajectorySpec
        ),
    ) -> tuple[RunManifest, RankSpec, RankRunSummary]:
        manifest = self.repository.load_manifest(spec.parent_run_id)
        if manifest.run_kind != "rank":
            raise CapabilityError(
                "qualification, FFN causal, and attention-rank parents must be rank runs",
                details={
                    "parent_run_id": spec.parent_run_id,
                    "run_kind": manifest.run_kind,
                },
            )
        failures = self.repository.verify(spec.parent_run_id)
        if failures:
            raise CapabilityError(
                "parent rank artifacts failed integrity verification",
                details={"parent_run_id": spec.parent_run_id, "failures": failures},
            )
        parent_spec = self.repository.load_run_spec(spec.parent_run_id)
        if not isinstance(parent_spec, RankSpec):
            raise CapabilityError("parent run does not contain a rank spec")
        parent_summary = RankRunSummary.model_validate(
            self.repository.load_summary(spec.parent_run_id)
        )
        return manifest, parent_spec, parent_summary

    def _parent_attention_rank(
        self, spec: AttentionHeadInterventionSpec | AttentionTraceSpec
    ) -> tuple[RunManifest, AttentionHeadRankSpec, AttentionHeadRankRunSummary]:
        manifest = self.repository.load_manifest(spec.parent_run_id)
        if manifest.run_kind != "attention_rank":
            raise CapabilityError(
                "attention intervention and trace parents must be attention rank runs",
                details={
                    "parent_run_id": spec.parent_run_id,
                    "run_kind": manifest.run_kind,
                },
            )
        failures = self.repository.verify(spec.parent_run_id)
        if failures:
            raise CapabilityError(
                "parent attention rank artifacts failed integrity verification",
                details={"parent_run_id": spec.parent_run_id, "failures": failures},
            )
        parent_spec = self.repository.load_run_spec(spec.parent_run_id)
        if not isinstance(parent_spec, AttentionHeadRankSpec):
            raise CapabilityError("parent run does not contain an attention rank spec")
        parent_summary = AttentionHeadRankRunSummary.model_validate(
            self.repository.load_summary(spec.parent_run_id)
        )
        return manifest, parent_spec, parent_summary

    def _attention_intervention_parent(
        self, spec: AttentionTraceSpec
    ) -> tuple[
        RunManifest,
        AttentionHeadInterventionSpec,
        AttentionHeadInterventionRunSummary,
    ] | None:
        run_id = spec.parent_intervention_run_id
        if run_id is None:
            return None
        manifest = self.repository.load_manifest(run_id)
        if manifest.run_kind != "attention_intervention":
            raise CapabilityError(
                "parent_intervention_run_id must reference an attention intervention run",
                details={"run_id": run_id, "run_kind": manifest.run_kind},
            )
        failures = self.repository.verify(run_id)
        if failures:
            raise CapabilityError(
                "parent attention intervention failed integrity verification",
                details={"run_id": run_id, "failures": failures},
            )
        parent_spec = self.repository.load_run_spec(run_id)
        if not isinstance(parent_spec, AttentionHeadInterventionSpec):
            raise CapabilityError(
                "parent intervention run has an incompatible saved spec"
            )
        summary = AttentionHeadInterventionRunSummary.model_validate(
            self.repository.load_summary(run_id)
        )
        if summary.parent_run_id != spec.parent_run_id:
            raise CapabilityError(
                "attention trace and intervention must share the same attention rank",
                details={
                    "trace_parent_run_id": spec.parent_run_id,
                    "intervention_parent_run_id": summary.parent_run_id,
                },
            )
        return manifest, parent_spec, summary

    def _rank_context(self, spec: ExperimentSpec) -> RankSpec:
        if isinstance(spec, RankSpec):
            return spec
        if isinstance(
            spec,
            (
                QualificationSpec,
                InterventionSpec,
                DirectionInjectionSpec,
                AttentionHeadRankSpec,
                TrajectorySpec,
            ),
        ):
            return self._parent_rank(spec)[1]
        _manifest, attention_spec, _summary = self._parent_attention_rank(spec)
        return self._parent_rank(attention_spec)[1]

    def _qualification_statuses(
        self,
        spec: (
            InterventionSpec
            | DirectionInjectionSpec
            | AttentionHeadRankSpec
            | AttentionHeadInterventionSpec
        ),
    ) -> dict[str, str] | None:
        run_id = spec.qualification_run_id
        if run_id is None:
            return None
        manifest = self.repository.load_manifest(run_id)
        if manifest.run_kind != "qualify":
            raise CapabilityError(
                "qualification_run_id must reference a qualification run",
                details={"qualification_run_id": run_id, "run_kind": manifest.run_kind},
            )
        failures = self.repository.verify(run_id)
        if failures:
            raise CapabilityError(
                "qualification artifacts failed integrity verification",
                details={"qualification_run_id": run_id, "failures": failures},
            )
        summary = QualificationRunSummary.model_validate(
            self.repository.load_summary(run_id)
        )
        expected_parent = spec.parent_run_id
        if isinstance(spec, AttentionHeadInterventionSpec):
            _manifest, _attention_spec, attention_summary = self._parent_attention_rank(
                spec
            )
            expected_parent = attention_summary.parent_run_id
        if summary.parent_run_id != expected_parent:
            raise CapabilityError(
                "qualification and intervention must share the same parent rank run",
                details={
                    "qualification_parent_run_id": summary.parent_run_id,
                    "parent_run_id": expected_parent,
                },
            )
        return {item.pair_id: item.status for item in summary.pairs}

    def _attention_trace_qualification_statuses(
        self,
        *,
        attention_spec: AttentionHeadRankSpec,
        intervention_spec: AttentionHeadInterventionSpec | None,
    ) -> dict[str, str] | None:
        """Inherit the closest declared qualification gate in trace lineage."""

        source: AttentionHeadRankSpec | AttentionHeadInterventionSpec
        if (
            intervention_spec is not None
            and intervention_spec.qualification_run_id is not None
        ):
            source = intervention_spec
        else:
            source = attention_spec
        return self._qualification_statuses(source)

    def capabilities(self, spec: ExperimentSpec) -> CapabilityReport:
        rank_spec = self._rank_context(spec)
        request = rank_spec.model
        inspection = self.models.inspect_cached(request)
        model_types = set(inspection["model_types"])
        if request.adapter == "qwen3":
            architecture_supported = not model_types or model_types == {"qwen3"}
        else:
            architecture_supported = model_types == {"qwen3"} or (
                not model_types and "qwen3" in request.id.lower()
            )
        mps_backend = getattr(torch.backends, "mps", None)
        requested_device_available = not (
            (request.device == "mps" and not (mps_backend and mps_backend.is_available()))
            or (request.device == "cuda" and not torch.cuda.is_available())
        )
        supported = architecture_supported and requested_device_available
        limitations = [
            "Dense Qwen3 models only in the current live adapter.",
            "Binary first-token observables only.",
        ]
        if isinstance(spec, RankSpec):
            limitations.append(
                "Ranking is observational until qualified, replicated, and intervention-tested."
            )
        elif isinstance(spec, TrajectorySpec):
            limitations.append(
                "Native trajectory values measure intermediate decodability, not causal use."
            )
        elif isinstance(spec, QualificationSpec):
            limitations.append(
                "Generated-behavior validity depends on the declared evaluator."
            )
        elif isinstance(spec, InterventionSpec):
            limitations.append(
                "Intervention claims remain local to the parent prompts, observable, and doses."
            )
        elif isinstance(spec, DirectionInjectionSpec):
            limitations.append(
                "Direction injection tests controllability and does not by itself localize a circuit."
            )
        elif isinstance(spec, AttentionHeadRankSpec):
            limitations.append(
                "Direct-logit attention head ranking is observational until intervention-tested."
            )
        elif isinstance(spec, AttentionHeadInterventionSpec):
            limitations.append(
                "Head-output interventions are local to the parent prompts and first-token observable."
            )
        elif spec.trace_kind == "token_edges":
            limitations.append(
                "Token routes require eager attention and remain observational."
            )
        else:
            limitations.append(
                "Head path patching requires exact full-token alignment and tested endpoints."
            )
        if not architecture_supported:
            limitations.append(
                "The requested model is not identified as dense Qwen3; select a "
                "capability-gated adapter explicitly."
            )
        if not requested_device_available:
            limitations.append(f"Requested device {request.device!r} is unavailable.")
        selected_device = _selected_device(request.device)
        return CapabilityReport(
            model_id=request.id,
            supported=supported,
            adapter=(
                "qwen3-dense-swiglu-v1" if architecture_supported else "unresolved"
            ),
            device=selected_device,
            dtype=_selected_dtype(request.dtype, selected_device),
            activation="post_swiglu",
            interventions=(
                "ablate",
                "amplify",
                "patch",
                "restore",
                "direction_inject",
                "attention_head_ablate",
                "attention_head_amplify",
                "attention_head_patch",
                "attention_head_restore",
                "attention_token_edges",
                "attention_head_paths",
            ),
            limitations=tuple(limitations),
        )

    def preflight(self, spec: ExperimentSpec) -> PreflightReport:
        plan = self.plan(spec)
        capabilities = self.capabilities(spec)
        model_ready = plan.model_cached
        acquisition_required = not model_ready and spec.execution.allow_download
        warnings = list(plan.warnings)
        if not capabilities.supported:
            warnings.append("requested model/device capability is unsupported")
        if spec.execution.trust_remote_code:
            warnings.append("trust_remote_code is unsupported")
        executable = (
            plan.within_budget
            and capabilities.supported
            and (model_ready or acquisition_required)
            and not spec.execution.trust_remote_code
        )
        return PreflightReport(
            science_hash=plan.science_hash,
            request_hash=plan.request_hash,
            executable=executable,
            model_ready=model_ready,
            acquisition_required=acquisition_required,
            plan=plan,
            capabilities=capabilities,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def plan(self, spec: ExperimentSpec) -> ExperimentPlan:
        rank_spec = self._rank_context(spec)
        if isinstance(spec, RankSpec):
            pair_count = len(spec.pairs)
            required = 2 * pair_count
        elif isinstance(spec, TrajectorySpec):
            _manifest, _parent_spec, parent_summary = self._parent_rank(spec)
            try:
                pair_count, required = trajectory_plan_counts(
                    parent_summary=parent_summary, spec=spec
                )
            except ValueError as exc:
                raise CapabilityError(str(exc)) from exc
        elif isinstance(spec, QualificationSpec):
            _manifest, _parent_spec, parent_summary = self._parent_rank(spec)
            available = {item.pair_id for item in parent_summary.pairs}
            requested = spec.pair_ids or tuple(item.pair_id for item in parent_summary.pairs)
            unknown = sorted(set(requested) - available)
            if unknown:
                raise CapabilityError(
                    "qualification references unknown parent pairs",
                    details={"unknown_pair_ids": unknown},
                )
            pair_count = len(requested)
            required = 2 * pair_count
        elif isinstance(spec, InterventionSpec):
            _manifest, parent_spec, parent_summary = self._parent_rank(spec)
            pair_count, required = intervention_plan_counts(
                parent_spec=parent_spec,
                parent_summary=parent_summary,
                spec=spec,
                qualification_statuses=self._qualification_statuses(spec),
            )
        elif isinstance(spec, DirectionInjectionSpec):
            _manifest, parent_spec, parent_summary = self._parent_rank(spec)
            pair_count, required = direction_plan_counts(
                parent_spec=parent_spec,
                parent_summary=parent_summary,
                spec=spec,
                qualification_statuses=self._qualification_statuses(spec),
            )
        elif isinstance(spec, AttentionHeadRankSpec):
            _manifest, parent_spec, parent_summary = self._parent_rank(spec)
            pair_count, required = attention_rank_plan_counts(
                parent_spec=parent_spec,
                parent_summary=parent_summary,
                spec=spec,
                qualification_statuses=self._qualification_statuses(spec),
            )
        elif isinstance(spec, AttentionHeadInterventionSpec):
            _attention_manifest, attention_spec, attention_summary = (
                self._parent_attention_rank(spec)
            )
            _rank_manifest, parent_spec, parent_summary = self._parent_rank(
                attention_spec
            )
            pair_count, required = attention_intervention_plan_counts(
                rank_spec=parent_spec,
                rank_summary=parent_summary,
                attention_summary=attention_summary,
                spec=spec,
                qualification_statuses=self._qualification_statuses(spec),
            )
        else:
            _attention_manifest, attention_spec, attention_summary = (
                self._parent_attention_rank(spec)
            )
            _rank_manifest, parent_spec, parent_summary = self._parent_rank(
                attention_spec
            )
            intervention_parent = self._attention_intervention_parent(spec)
            intervention_summary = (
                intervention_parent[2] if intervention_parent is not None else None
            )
            pair_count, required = attention_trace_plan_counts(
                rank_spec=parent_spec,
                rank_summary=parent_summary,
                attention_summary=attention_summary,
                intervention_summary=intervention_summary,
                spec=spec,
                qualification_statuses=self._attention_trace_qualification_statuses(
                    attention_spec=attention_spec,
                    intervention_spec=(
                        intervention_parent[1]
                        if intervention_parent is not None
                        else None
                    ),
                ),
            )
        within_budget = required <= spec.execution.max_forward_passes
        warnings: list[str] = []
        if not within_budget:
            warnings.append(
                f"requires {required} forward passes but budget allows "
                f"{spec.execution.max_forward_passes}"
            )
        cached = self.models.is_cached(rank_spec.model)
        if not cached and not spec.execution.allow_download:
            warnings.append("model is not cached and downloads are disabled")
        if spec.execution.trust_remote_code:
            warnings.append("trust_remote_code is unsupported by the current Qwen adapter")
        return ExperimentPlan(
            science_hash=science_hash(spec),
            request_hash=request_hash(spec),
            kind=spec.kind,
            pair_count=pair_count,
            forward_passes=required,
            within_budget=within_budget,
            model_cached=cached,
            resolved_device=_selected_device(rank_spec.model.device),
            warnings=tuple(warnings),
        )

    def validate_execution(
        self, spec: ExperimentSpec
    ) -> tuple[ExperimentPlan, CapabilityReport]:
        """Perform all cheap checks required before a job is accepted."""
        plan = self.plan(spec)
        capability = self.capabilities(spec)
        if not capability.supported:
            raise CapabilityError(
                "the requested model/device combination is not supported",
                details=capability.model_dump(mode="json"),
            )
        if not plan.within_budget:
            raise BudgetError(
                plan.warnings[0],
                details={
                    "forward_passes": plan.forward_passes,
                    "max_forward_passes": spec.execution.max_forward_passes,
                },
            )
        if spec.execution.trust_remote_code:
            raise CapabilityError(
                "trust_remote_code is not supported by the current adapter",
                hint="Use a capability-gated adapter implementation instead.",
            )
        return plan, capability

    def initialize_job(
        self,
        spec: ExperimentSpec,
        *,
        job_id: str | None = None,
        request_id: str | None = None,
        state: str = "queued",
    ) -> JobStatus:
        now = datetime.now(timezone.utc)
        status = JobStatus(
            job_id=job_id or uuid4().hex,
            request_id=request_id or uuid4().hex,
            request_hash=request_hash(spec),
            science_hash=science_hash(spec),
            state=state,
            created_at=now,
            updated_at=now,
        )
        self.repository.create_job(status, spec)
        return status

    def find_idempotent_job(
        self, spec: ExperimentSpec, request_id: str | None
    ) -> JobStatus | None:
        if request_id is None:
            return None
        existing = self.repository.find_job_by_request_id(request_id)
        if existing is None:
            return None
        expected = request_hash(spec)
        if existing.request_hash != expected:
            raise RequestConflictError(
                f"request ID {request_id!r} is already bound to another request",
                details={
                    "request_id": request_id,
                    "existing_request_hash": existing.request_hash,
                    "submitted_request_hash": expected,
                    "existing_job_id": existing.job_id,
                },
            )
        return existing

    def execute(
        self,
        spec: ExperimentSpec,
        *,
        job: JobStatus | None = None,
        listener: EventListener | None = None,
        cancel: threading.Event | None = None,
        diagnostic_stream: TextIO | None = None,
    ) -> ExecutionOutcome:
        if isinstance(spec, QualificationSpec):
            return self._execute_qualification(
                spec,
                job=job,
                listener=listener,
                cancel=cancel,
                diagnostic_stream=diagnostic_stream,
            )
        if isinstance(spec, TrajectorySpec):
            return self._execute_child(
                spec,
                job=job,
                listener=listener,
                cancel=cancel,
                diagnostic_stream=diagnostic_stream,
            )
        if isinstance(spec, InterventionSpec):
            return self._execute_intervention(
                spec,
                job=job,
                listener=listener,
                cancel=cancel,
                diagnostic_stream=diagnostic_stream,
            )
        if isinstance(spec, DirectionInjectionSpec):
            return self._execute_direction(
                spec,
                job=job,
                listener=listener,
                cancel=cancel,
                diagnostic_stream=diagnostic_stream,
            )
        if isinstance(
            spec,
            (
                AttentionHeadRankSpec,
                AttentionHeadInterventionSpec,
                AttentionTraceSpec,
            ),
        ):
            return self._execute_child(
                spec,
                job=job,
                listener=listener,
                cancel=cancel,
                diagnostic_stream=diagnostic_stream,
            )
        return self._execute_rank(
            spec,
            job=job,
            listener=listener,
            cancel=cancel,
            diagnostic_stream=diagnostic_stream,
        )

    def _execute_rank(
        self,
        spec: RankSpec,
        *,
        job: JobStatus | None = None,
        listener: EventListener | None = None,
        cancel: threading.Event | None = None,
        diagnostic_stream: TextIO | None = None,
    ) -> ExecutionOutcome:
        job = job or self.initialize_job(spec)
        listeners: list[EventListener] = [self.repository.append_event]
        if listener is not None:
            listeners.append(listener)
        emitter = EventEmitter(
            job_id=job.job_id,
            request_id=job.request_id,
            science_hash=job.science_hash,
            listeners=tuple(listeners),
        )
        started_at = datetime.now(timezone.utc)
        running = job.model_copy(
            update={"state": "running", "updated_at": started_at}
        )
        self.repository.update_job(running)
        deadline = (
            time.monotonic() + spec.execution.max_wall_seconds
            if spec.execution.max_wall_seconds is not None
            else None
        )

        def check_cancelled() -> None:
            if cancel is not None and cancel.is_set():
                raise JobCancelled("job was cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetError(
                    "job exceeded max_wall_seconds between model operations",
                    details={"max_wall_seconds": spec.execution.max_wall_seconds},
                )

        try:
            plan, _capability = self.validate_execution(spec)
            emitter.emit("job.accepted", plan=plan.model_dump(mode="json"))
            check_cancelled()
            self.models.ensure_available(
                spec.model,
                allow_download=spec.execution.allow_download,
                max_download_bytes=spec.execution.max_download_bytes,
            )
            emitter.emit(
                "model.reused" if self._will_reuse_engine(spec) else "model.loading",
                model_id=spec.model.id,
            )
            seed_everything(spec.execution.seed)
            with (
                redirect_stdout(diagnostic_stream)
                if diagnostic_stream is not None
                else nullcontext()
            ):
                engine = self.engine_factory(spec)
            emitter.emit(
                "model.ready",
                model=asdict(engine.adapter.metadata),
            )

            analyses: list[ProbeAnalysis] = []
            for index, pair in enumerate(spec.pairs):
                check_cancelled()
                emitter.emit(
                    "pair.started",
                    pair_id=pair.id,
                    pair_index=index,
                    pair_count=len(spec.pairs),
                )
                probe_spec = ProbeSpec(
                    model_id=spec.model.id,
                    revision=spec.model.revision,
                    pair=PromptPair(
                        original=pair.original,
                        perturbed=pair.perturbed,
                        original_messages=tuple(
                            item.model_dump(mode="json", exclude_none=True)
                            for item in pair.original_messages
                        ),
                        perturbed_messages=tuple(
                            item.model_dump(mode="json", exclude_none=True)
                            for item in pair.perturbed_messages
                        ),
                        tools=pair.tools,
                    ),
                    observable=ObservableSpec(
                        name=spec.observable.name,
                        target_tokens=spec.observable.target_tokens,
                        control_tokens=spec.observable.control_tokens,
                    ),
                    chat_template=spec.model.chat_template,
                    enable_thinking=spec.model.enable_thinking,
                    capture_position=spec.capture.position,
                    top_k=spec.ranking.top_k,
                )
                with (
                    redirect_stdout(diagnostic_stream)
                    if diagnostic_stream is not None
                    else nullcontext()
                ):
                    analysis = engine.analyze_details(probe_spec)
                analyses.append(analysis)
                emitter.emit(
                    "pair.completed",
                    pair_id=pair.id,
                    measured_delta=analysis.result.measured_delta,
                    predicted_delta=analysis.result.predicted_delta,
                    elapsed_seconds=analysis.result.elapsed_seconds,
                    warnings=list(analysis.result.warnings),
                )
                check_cancelled()

            aggregate = aggregate_analyses(
                science_hash=job.science_hash,
                pair_ids=tuple(pair.id for pair in spec.pairs),
                analyses=tuple(analyses),
                top_k=spec.ranking.top_k,
                pair_splits=tuple(pair.split for pair in spec.pairs),
            )
            for warning in aggregate.summary.warnings:
                emitter.emit("warning", message=warning)

            first = analyses[0].result
            fingerprint = hash_value(
                {
                    "science_hash": job.science_hash,
                    "algorithm_version": ALGORITHM_VERSION,
                    "resolved_model": asdict(first.model),
                    "resolved_observable": asdict(first.observable),
                }
            )
            manifest, run_directory = self.repository.commit_rank(
                job_id=job.job_id,
                request_id=job.request_id,
                spec=spec,
                aggregate=aggregate,
                pair_results=tuple(item.result for item in analyses),
                science_hash=job.science_hash,
                run_fingerprint=fingerprint,
                algorithm_version=ALGORITHM_VERSION,
                created_at=started_at,
                max_artifact_bytes=spec.execution.max_artifact_bytes,
            )
            emitter.emit(
                "artifact.committed",
                run_id=manifest.run_id,
                run_directory=str(run_directory),
            )
            emitter.emit(
                "job.completed",
                run_id=manifest.run_id,
                evidence_stage=aggregate.summary.evidence_stage,
                pair_count=aggregate.summary.pair_count,
                measured_delta_mean=aggregate.summary.measured_delta_mean,
                predicted_delta_mean=aggregate.summary.predicted_delta_mean,
                ffn_skip_mean=aggregate.summary.ffn_skip_mean,
                warning_count=len(aggregate.summary.warnings),
            )
            completed = running.model_copy(
                update={
                    "state": "completed",
                    "updated_at": datetime.now(timezone.utc),
                    "run_id": manifest.run_id,
                }
            )
            self.repository.update_job(completed)
            return ExecutionOutcome(
                manifest=manifest,
                run_directory=run_directory,
                summary=aggregate.summary,
            )
        except Exception as exc:
            if isinstance(exc, ProbeError):
                detail = exc.as_detail()
                state = "cancelled" if isinstance(exc, JobCancelled) else "failed"
                event_name = "job.cancelled" if state == "cancelled" else "job.failed"
            else:
                detail = ErrorDetail(
                    code="runtime_error",
                    message=str(exc),
                    retryable=False,
                    details={"exception_type": type(exc).__name__},
                )
                state = "failed"
                event_name = "job.failed"
            emitter.emit(event_name, error=detail.model_dump(mode="json"))
            failed = running.model_copy(
                update={
                    "state": state,
                    "updated_at": datetime.now(timezone.utc),
                    "error": detail,
                }
            )
            self.repository.update_job(failed)
            raise

    def _execute_qualification(
        self,
        spec: QualificationSpec,
        *,
        job: JobStatus | None = None,
        listener: EventListener | None = None,
        cancel: threading.Event | None = None,
        diagnostic_stream: TextIO | None = None,
    ) -> ExecutionOutcome:
        return self._execute_child(
            spec,
            job=job,
            listener=listener,
            cancel=cancel,
            diagnostic_stream=diagnostic_stream,
        )

    def _execute_intervention(
        self,
        spec: InterventionSpec,
        *,
        job: JobStatus | None = None,
        listener: EventListener | None = None,
        cancel: threading.Event | None = None,
        diagnostic_stream: TextIO | None = None,
    ) -> ExecutionOutcome:
        return self._execute_child(
            spec,
            job=job,
            listener=listener,
            cancel=cancel,
            diagnostic_stream=diagnostic_stream,
        )

    def _execute_direction(
        self,
        spec: DirectionInjectionSpec,
        *,
        job: JobStatus | None = None,
        listener: EventListener | None = None,
        cancel: threading.Event | None = None,
        diagnostic_stream: TextIO | None = None,
    ) -> ExecutionOutcome:
        return self._execute_child(
            spec,
            job=job,
            listener=listener,
            cancel=cancel,
            diagnostic_stream=diagnostic_stream,
        )

    def _execute_child(
        self,
        spec: (
            QualificationSpec
            | InterventionSpec
            | DirectionInjectionSpec
            | AttentionHeadRankSpec
            | AttentionHeadInterventionSpec
            | AttentionTraceSpec
        ),
        *,
        job: JobStatus | None,
        listener: EventListener | None,
        cancel: threading.Event | None,
        diagnostic_stream: TextIO | None,
    ) -> ExecutionOutcome:
        job = job or self.initialize_job(spec)
        listeners: list[EventListener] = [self.repository.append_event]
        if listener is not None:
            listeners.append(listener)
        emitter = EventEmitter(
            job_id=job.job_id,
            request_id=job.request_id,
            science_hash=job.science_hash,
            listeners=tuple(listeners),
        )
        started_at = datetime.now(timezone.utc)
        running = job.model_copy(
            update={"state": "running", "updated_at": started_at}
        )
        self.repository.update_job(running)
        deadline = (
            time.monotonic() + spec.execution.max_wall_seconds
            if spec.execution.max_wall_seconds is not None
            else None
        )

        def check_cancelled() -> None:
            if cancel is not None and cancel.is_set():
                raise JobCancelled("job was cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetError(
                    "job exceeded max_wall_seconds between model operations",
                    details={"max_wall_seconds": spec.execution.max_wall_seconds},
                )

        try:
            plan, _capability = self.validate_execution(spec)
            emitter.emit("job.accepted", plan=plan.model_dump(mode="json"))
            check_cancelled()
            attention_summary: AttentionHeadRankRunSummary | None = None
            attention_tensors: dict[str, torch.Tensor] | None = None
            intervention_summary: AttentionHeadInterventionRunSummary | None = None
            trace_qualification_statuses: dict[str, str] | None = None
            if isinstance(spec, (AttentionHeadInterventionSpec, AttentionTraceSpec)):
                lineage_manifest, attention_parent_spec, attention_summary = (
                    self._parent_attention_rank(spec)
                )
                parent_manifest, parent_spec, parent_summary = self._parent_rank(
                    attention_parent_spec
                )
                if isinstance(spec, AttentionHeadInterventionSpec):
                    attention_tensors = load_file(
                        self.repository.runs
                        / spec.parent_run_id
                        / "attention-tensors.safetensors"
                    )
                else:
                    intervention_parent = self._attention_intervention_parent(spec)
                    intervention_summary = (
                        intervention_parent[2]
                        if intervention_parent is not None
                        else None
                    )
                    trace_qualification_statuses = (
                        self._attention_trace_qualification_statuses(
                            attention_spec=attention_parent_spec,
                            intervention_spec=(
                                intervention_parent[1]
                                if intervention_parent is not None
                                else None
                            ),
                        )
                    )
            else:
                parent_manifest, parent_spec, parent_summary = self._parent_rank(spec)
                lineage_manifest = parent_manifest
            self.models.ensure_available(
                parent_spec.model,
                allow_download=spec.execution.allow_download,
                max_download_bytes=spec.execution.max_download_bytes,
            )
            emitter.emit(
                "model.reused"
                if self._will_reuse_engine(parent_spec)
                else "model.loading",
                model_id=parent_spec.model.id,
            )
            seed_everything(spec.execution.seed)
            with (
                redirect_stdout(diagnostic_stream)
                if diagnostic_stream is not None
                else nullcontext()
            ):
                engine = self.engine_factory(parent_spec)
            resolved_parent = parent_manifest.resolved_model.get("resolved_revision")
            resolved_current = engine.adapter.metadata.resolved_revision
            if (
                resolved_parent is not None
                and resolved_current is not None
                and resolved_parent != resolved_current
            ):
                raise CapabilityError(
                    "loaded model revision differs from the immutable parent run",
                    details={
                        "parent_revision": resolved_parent,
                        "loaded_revision": resolved_current,
                    },
                )
            emitter.emit("model.ready", model=asdict(engine.adapter.metadata))
            emitter.emit(
                f"{spec.kind}.started",
                parent_run_id=spec.parent_run_id,
                pair_count=plan.pair_count,
            )
            with (
                redirect_stdout(diagnostic_stream)
                if diagnostic_stream is not None
                else nullcontext()
            ):
                if isinstance(spec, QualificationSpec):
                    summary = run_qualification(
                        engine=engine,
                        parent_spec=parent_spec,
                        parent_summary=parent_summary,
                        spec=spec,
                        science_hash=job.science_hash,
                    )
                elif isinstance(spec, TrajectorySpec):
                    summary = run_trajectory(
                        engine=engine,
                        parent_spec=parent_spec,
                        parent_summary=parent_summary,
                        spec=spec,
                        science_hash=job.science_hash,
                    )
                elif isinstance(spec, InterventionSpec):
                    tensor_path = (
                        self.repository.runs
                        / spec.parent_run_id
                        / "tensors.safetensors"
                    )
                    summary = run_intervention(
                        engine=engine,
                        parent_spec=parent_spec,
                        parent_summary=parent_summary,
                        parent_tensors=load_file(tensor_path),
                        spec=spec,
                        science_hash=job.science_hash,
                        qualification_statuses=self._qualification_statuses(spec),
                    )
                elif isinstance(spec, DirectionInjectionSpec):
                    summary = run_direction_injection(
                        engine=engine,
                        parent_spec=parent_spec,
                        parent_summary=parent_summary,
                        spec=spec,
                        science_hash=job.science_hash,
                        qualification_statuses=self._qualification_statuses(spec),
                    )
                elif isinstance(spec, AttentionHeadRankSpec):
                    attention_computation = run_attention_rank(
                        engine=engine,
                        parent_spec=parent_spec,
                        parent_summary=parent_summary,
                        spec=spec,
                        science_hash=job.science_hash,
                        qualification_statuses=self._qualification_statuses(spec),
                    )
                    summary = attention_computation.summary
                    attention_tensors = attention_computation.tensors
                elif isinstance(spec, AttentionHeadInterventionSpec):
                    assert attention_summary is not None
                    assert attention_tensors is not None
                    summary = run_attention_intervention(
                        engine=engine,
                        rank_spec=parent_spec,
                        rank_summary=parent_summary,
                        attention_summary=attention_summary,
                        attention_tensors=attention_tensors,
                        spec=spec,
                        science_hash=job.science_hash,
                        qualification_statuses=self._qualification_statuses(spec),
                    )
                else:
                    assert attention_summary is not None
                    summary = run_attention_trace(
                        engine=engine,
                        rank_spec=parent_spec,
                        rank_summary=parent_summary,
                        attention_summary=attention_summary,
                        intervention_summary=intervention_summary,
                        spec=spec,
                        science_hash=job.science_hash,
                        qualification_statuses=trace_qualification_statuses,
                    )
            check_cancelled()
            if summary.logical_forward_passes != plan.forward_passes:
                raise RuntimeError(
                    "executed model-call count does not match the preflight plan: "
                    f"planned={plan.forward_passes} actual={summary.logical_forward_passes}"
                )
            for warning in summary.warnings:
                emitter.emit("warning", message=warning)
            emitter.emit(
                f"{spec.kind}.completed",
                parent_run_id=spec.parent_run_id,
                logical_forward_passes=summary.logical_forward_passes,
                warning_count=len(summary.warnings),
            )
            algorithm_version = (
                ATTENTION_ALGORITHM_VERSION
                if isinstance(
                    spec,
                    (
                        AttentionHeadRankSpec,
                        AttentionHeadInterventionSpec,
                        AttentionTraceSpec,
                    ),
                )
                else ALGORITHM_VERSION
            )
            fingerprint = hash_value(
                {
                    "science_hash": job.science_hash,
                    "algorithm_version": algorithm_version,
                    "parent_fingerprint": lineage_manifest.run_fingerprint,
                    "resolved_model": asdict(engine.adapter.metadata),
                }
            )
            commit_arguments = {
                "job_id": job.job_id,
                "request_id": job.request_id,
                "spec": spec,
                "summary": summary,
                "requested_model": parent_spec.model,
                "resolved_model": asdict(engine.adapter.metadata),
                "science_hash": job.science_hash,
                "run_fingerprint": fingerprint,
                "algorithm_version": algorithm_version,
                "created_at": started_at,
                "max_artifact_bytes": spec.execution.max_artifact_bytes,
            }
            if isinstance(spec, QualificationSpec):
                assert isinstance(summary, QualificationRunSummary)
                manifest, run_directory = self.repository.commit_qualification(
                    **commit_arguments
                )
            elif isinstance(spec, TrajectorySpec):
                assert isinstance(summary, TrajectoryRunSummary)
                manifest, run_directory = self.repository.commit_trajectory(
                    **commit_arguments
                )
            elif isinstance(spec, InterventionSpec):
                assert isinstance(summary, InterventionRunSummary)
                manifest, run_directory = self.repository.commit_intervention(
                    **commit_arguments
                )
            elif isinstance(spec, DirectionInjectionSpec):
                assert isinstance(summary, DirectionInjectionRunSummary)
                manifest, run_directory = self.repository.commit_direction(
                    **commit_arguments
                )
            elif isinstance(spec, AttentionHeadRankSpec):
                assert isinstance(summary, AttentionHeadRankRunSummary)
                assert attention_tensors is not None
                manifest, run_directory = self.repository.commit_attention_rank(
                    **commit_arguments,
                    tensors=attention_tensors,
                )
            elif isinstance(spec, AttentionHeadInterventionSpec):
                assert isinstance(summary, AttentionHeadInterventionRunSummary)
                manifest, run_directory = (
                    self.repository.commit_attention_intervention(
                        **commit_arguments
                    )
                )
            else:
                assert isinstance(summary, AttentionTraceRunSummary)
                manifest, run_directory = self.repository.commit_attention_trace(
                    **commit_arguments
                )
            emitter.emit(
                "artifact.committed",
                run_id=manifest.run_id,
                run_directory=str(run_directory),
            )
            emitter.emit(
                "job.completed",
                run_id=manifest.run_id,
                evidence_stage=summary.evidence_stage,
                parent_run_id=spec.parent_run_id,
                warning_count=len(summary.warnings),
            )
            completed = running.model_copy(
                update={
                    "state": "completed",
                    "updated_at": datetime.now(timezone.utc),
                    "run_id": manifest.run_id,
                }
            )
            self.repository.update_job(completed)
            return ExecutionOutcome(
                manifest=manifest,
                run_directory=run_directory,
                summary=summary,
            )
        except Exception as exc:
            if isinstance(exc, ProbeError):
                detail = exc.as_detail()
                state = "cancelled" if isinstance(exc, JobCancelled) else "failed"
                event_name = "job.cancelled" if state == "cancelled" else "job.failed"
            else:
                detail = ErrorDetail(
                    code="runtime_error",
                    message=str(exc),
                    retryable=False,
                    details={"exception_type": type(exc).__name__},
                )
                state = "failed"
                event_name = "job.failed"
            emitter.emit(event_name, error=detail.model_dump(mode="json"))
            failed = running.model_copy(
                update={
                    "state": state,
                    "updated_at": datetime.now(timezone.utc),
                    "error": detail,
                }
            )
            self.repository.update_job(failed)
            raise

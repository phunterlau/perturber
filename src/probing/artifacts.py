from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any
import re

from safetensors.torch import save_file
import torch

from .aggregation import AggregateComputation
from .contracts import (
    AttentionHeadInterventionRunSummary,
    AttentionHeadInterventionSpec,
    AttentionHeadRankRunSummary,
    AttentionHeadRankSpec,
    AttentionTraceRunSummary,
    AttentionTraceSpec,
    ArtifactRef,
    DirectionInjectionRunSummary,
    DirectionInjectionSpec,
    ErrorDetail,
    ExperimentSpec,
    FFNCouplingRunSummary,
    FFNCouplingSpec,
    InterventionRunSummary,
    InterventionSpec,
    JobEvent,
    JobStatus,
    QualificationRunSummary,
    QualificationSpec,
    RankSpec,
    RunManifest,
    TrajectoryRunSummary,
    TrajectorySpec,
)
from .domain import ProbeResult
from .errors import ArtifactError
from .specs import canonical_json, hash_value, parse_spec_data


def _run_id(result: ProbeResult) -> str:
    digest_input = json.dumps(asdict(result.spec), sort_keys=True).encode("utf-8")
    digest = sha256(digest_input).hexdigest()[:10]
    timestamp = result.created_at.replace(":", "").replace("-", "")[:15]
    return f"{timestamp}-{digest}"


def export_result(result: ProbeResult, output_root: Path) -> Path:
    run_directory = output_root / _run_id(result)
    run_directory.mkdir(parents=True, exist_ok=True)

    manifest = asdict(result)
    manifest["environment"] = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "textual")
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (run_directory / "layers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.layers[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in result.layers)

    with (run_directory / "neurons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.neurons[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in result.neurons)

    return run_directory


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _artifact_ref(root: Path, path: Path, media_type: str) -> ArtifactRef:
    data = path.read_bytes()
    return ArtifactRef(
        path=str(path.relative_to(root)),
        media_type=media_type,
        sha256=sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _environment_metadata() -> dict[str, str]:
    values = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_deterministic_algorithms": str(
            torch.are_deterministic_algorithms_enabled()
        ).lower(),
        "pytorch_enable_mps_fallback": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK", "unset"
        ),
    }
    for distribution, key in (
        ("perturbation-probing-workbench", "probing"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("huggingface-hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
        ("pydantic", "pydantic"),
    ):
        try:
            values[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[key] = "unknown"
    return values


class ArtifactRepository:
    """Files-only job and immutable run repository."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.jobs = self.workspace / "jobs"
        self.runs = self.workspace / "runs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _component(value: str, kind: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ArtifactError(f"invalid {kind} identifier {value!r}")
        return value

    def create_job(self, status: JobStatus, spec: ExperimentSpec | None = None) -> Path:
        directory = self.jobs / self._component(status.job_id, "job")
        directory.mkdir(parents=False, exist_ok=False)
        _write_json(directory / "status.json", status)
        if spec is not None:
            _write_json(directory / "spec.json", spec)
        (directory / "events.jsonl").touch()
        return directory

    def update_job(self, status: JobStatus) -> None:
        directory = self.jobs / self._component(status.job_id, "job")
        if not directory.is_dir():
            raise ArtifactError(f"job {status.job_id!r} does not exist")
        _atomic_json(directory / "status.json", status)

    def append_event(self, event: JobEvent) -> None:
        path = self.jobs / self._component(event.job_id, "job") / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event))
            handle.write("\n")
            handle.flush()

    def load_job(self, job_id: str) -> JobStatus:
        path = self.jobs / self._component(job_id, "job") / "status.json"
        if not path.is_file():
            raise ArtifactError(f"job {job_id!r} was not found")
        try:
            return JobStatus.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactError(f"job {job_id!r} has invalid status metadata") from exc

    def read_events(self, job_id: str, start_sequence: int = 0) -> list[JobEvent]:
        path = self.jobs / self._component(job_id, "job") / "events.jsonl"
        if not path.is_file():
            raise ArtifactError(f"job {job_id!r} was not found")
        events: list[JobEvent] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                item = JobEvent.model_validate_json(line)
                if item.sequence >= start_sequence:
                    events.append(item)
        except Exception as exc:
            raise ArtifactError(f"job {job_id!r} has invalid event metadata") from exc
        return events

    def load_job_spec(self, job_id: str) -> ExperimentSpec:
        path = self.jobs / self._component(job_id, "job") / "spec.json"
        if not path.is_file():
            raise ArtifactError(f"job {job_id!r} has no saved spec")
        try:
            return parse_spec_data(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise ArtifactError(f"job {job_id!r} has invalid spec metadata") from exc

    def find_job_by_request_id(self, request_id: str) -> JobStatus | None:
        matches: list[JobStatus] = []
        for path in self.jobs.glob("*/status.json"):
            try:
                status = JobStatus.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if status.request_id == request_id:
                matches.append(status)
        return max(matches, key=lambda item: item.created_at) if matches else None

    def recover_interrupted_jobs(self) -> tuple[JobStatus, ...]:
        """Close durable queued/running jobs left by a previous daemon process."""
        recovered: list[JobStatus] = []
        for path in sorted(self.jobs.glob("*/status.json")):
            try:
                status = self.load_job(path.parent.name)
            except ArtifactError:
                continue
            if status.state not in {"queued", "running"}:
                continue
            try:
                events = self.read_events(status.job_id)
            except ArtifactError:
                events = []
            terminal = next(
                (
                    event
                    for event in reversed(events)
                    if event.event in {"job.completed", "job.failed", "job.cancelled"}
                ),
                None,
            )
            now = datetime.now(timezone.utc)
            if terminal is not None and terminal.event == "job.completed":
                updated = status.model_copy(
                    update={
                        "state": "completed",
                        "updated_at": now,
                        "run_id": terminal.payload.get("run_id"),
                    }
                )
            elif terminal is not None:
                detail = ErrorDetail.model_validate(terminal.payload["error"])
                updated = status.model_copy(
                    update={
                        "state": (
                            "cancelled"
                            if terminal.event == "job.cancelled"
                            else "failed"
                        ),
                        "updated_at": now,
                        "error": detail,
                    }
                )
            else:
                detail = ErrorDetail(
                    code="job_interrupted",
                    message="job was interrupted by a previous daemon process",
                    retryable=True,
                    hint="Inspect the saved job spec and submit it with a new request ID.",
                )
                sequence = max((event.sequence for event in events), default=-1) + 1
                self.append_event(
                    JobEvent(
                        event="job.failed",
                        sequence=sequence,
                        timestamp=now,
                        job_id=status.job_id,
                        request_id=status.request_id,
                        science_hash=status.science_hash,
                        payload={"error": detail.model_dump(mode="json")},
                    )
                )
                updated = status.model_copy(
                    update={"state": "failed", "updated_at": now, "error": detail}
                )
            self.update_job(updated)
            recovered.append(updated)
        return tuple(recovered)

    def commit_rank(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: RankSpec,
        aggregate: AggregateComputation,
        pair_results: tuple[ProbeResult, ...],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}-{run_fingerprint[:12]}"
        destination = self.runs / run_id
        if destination.exists():
            run_id = f"{run_id}-{job_id[:6]}"
            destination = self.runs / run_id

        staging = self.jobs / self._component(job_id, "job") / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()

        _write_json(staging / "spec.json", spec)
        _write_json(staging / "summary.json", aggregate.summary)
        with (staging / "pairs.jsonl").open("w", encoding="utf-8") as handle:
            for result in pair_results:
                handle.write(canonical_json(asdict(result)))
                handle.write("\n")

        with (staging / "layers.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            rows = [item.model_dump(mode="json") for item in aggregate.summary.layers]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (staging / "neurons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            rows = [item.model_dump(mode="json") for item in aggregate.summary.neurons]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_file(aggregate.tensors, staging / "tensors.safetensors")

        source_events = self.jobs / job_id / "events.jsonl"
        shutil.copy2(source_events, staging / "events.jsonl")
        media_types = {
            "spec.json": "application/json",
            "summary.json": "application/json",
            "pairs.jsonl": "application/x-ndjson",
            "layers.csv": "text/csv",
            "neurons.csv": "text/csv",
            "tensors.safetensors": "application/octet-stream",
            "events.jsonl": "application/x-ndjson",
        }
        refs = tuple(
            _artifact_ref(staging, staging / name, media_type)
            for name, media_type in media_types.items()
        )
        first_result = pair_results[0]
        manifest = RunManifest(
            run_id=run_id,
            job_id=job_id,
            request_id=request_id,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            created_at=created_at,
            completed_at=timestamp,
            evidence_stage=aggregate.summary.evidence_stage,
            algorithm_version=algorithm_version,
            requested_model=spec.model,
            resolved_model=first_result.model.__dict__,
            environment=_environment_metadata(),
            pair_count=len(pair_results),
            artifacts=refs,
            warnings=aggregate.summary.warnings,
        )
        _write_json(staging / "manifest.json", manifest)
        total_size = sum(item.size_bytes for item in refs) + (
            staging / "manifest.json"
        ).stat().st_size
        if total_size > max_artifact_bytes:
            shutil.rmtree(staging)
            raise ArtifactError(
                f"artifact size {total_size} exceeds max_artifact_bytes "
                f"{max_artifact_bytes}",
                details={"planned_bytes": total_size, "limit": max_artifact_bytes},
            )
        os.replace(staging, destination)
        return manifest, destination

    def commit_qualification(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: QualificationSpec,
        summary: QualificationRunSummary,
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        pair_rows = [item.model_dump(mode="json") for item in summary.pairs]
        return self._commit_child_run(
            job_id=job_id,
            request_id=request_id,
            spec=spec,
            summary=summary,
            requested_model=requested_model,
            resolved_model=resolved_model,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            algorithm_version=algorithm_version,
            created_at=created_at,
            evidence_stage="qualified_observable",
            run_kind="qualify",
            parent_run_ids=(spec.parent_run_id,),
            pair_count=len(summary.pairs),
            detail_name="qualified-pairs.jsonl",
            detail_rows=pair_rows,
            warnings=summary.warnings,
            max_artifact_bytes=max_artifact_bytes,
        )

    def commit_trajectory(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: TrajectorySpec,
        summary: TrajectoryRunSummary,
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        rows = [
            {
                "pair_id": pair.pair_id,
                "split": pair.split,
                **checkpoint.model_dump(mode="json"),
            }
            for pair in summary.pairs
            for checkpoint in pair.checkpoints
        ]
        return self._commit_child_run(
            job_id=job_id,
            request_id=request_id,
            spec=spec,
            summary=summary,
            requested_model=requested_model,
            resolved_model=resolved_model,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            algorithm_version=algorithm_version,
            created_at=created_at,
            evidence_stage="observational_trajectory",
            run_kind="trajectory",
            parent_run_ids=(spec.parent_run_id,),
            pair_count=len(summary.pairs),
            detail_name="trajectory-checkpoints.jsonl",
            detail_rows=rows,
            warnings=summary.warnings,
            max_artifact_bytes=max_artifact_bytes,
        )

    def commit_attention_rank(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: AttentionHeadRankSpec,
        summary: AttentionHeadRankRunSummary,
        tensors: dict[str, torch.Tensor],
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}-{run_fingerprint[:12]}"
        destination = self.runs / run_id
        if destination.exists():
            run_id = f"{run_id}-{job_id[:6]}"
            destination = self.runs / run_id
        staging = self.jobs / self._component(job_id, "job") / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        _write_json(staging / "spec.json", spec)
        _write_json(staging / "summary.json", summary)
        with (staging / "attention-pairs.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for item in summary.pairs:
                handle.write(canonical_json(item) + "\n")
        for filename, records in (
            ("attention-layers.csv", summary.layers),
            ("attention-heads.csv", summary.heads),
        ):
            rows = [item.model_dump(mode="json") for item in records]
            with (staging / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        save_file(tensors, staging / "attention-tensors.safetensors")
        source_events = self.jobs / self._component(job_id, "job") / "events.jsonl"
        shutil.copy2(source_events, staging / "events.jsonl")
        media_types = {
            "spec.json": "application/json",
            "summary.json": "application/json",
            "attention-pairs.jsonl": "application/x-ndjson",
            "attention-layers.csv": "text/csv",
            "attention-heads.csv": "text/csv",
            "attention-tensors.safetensors": "application/octet-stream",
            "events.jsonl": "application/x-ndjson",
        }
        refs = tuple(
            _artifact_ref(staging, staging / name, media_type)
            for name, media_type in media_types.items()
        )
        manifest = RunManifest(
            run_id=run_id,
            job_id=job_id,
            request_id=request_id,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            created_at=created_at,
            completed_at=timestamp,
            evidence_stage="attention_hypothesis",
            run_kind="attention_rank",
            parent_run_ids=tuple(
                value
                for value in (spec.parent_run_id, spec.qualification_run_id)
                if value is not None
            ),
            algorithm_version=algorithm_version,
            requested_model=requested_model,
            resolved_model=resolved_model,
            environment=_environment_metadata(),
            pair_count=len(summary.pairs),
            artifacts=refs,
            warnings=summary.warnings,
        )
        _write_json(staging / "manifest.json", manifest)
        total_size = sum(item.size_bytes for item in refs) + (
            staging / "manifest.json"
        ).stat().st_size
        if total_size > max_artifact_bytes:
            shutil.rmtree(staging)
            raise ArtifactError(
                f"artifact size {total_size} exceeds max_artifact_bytes "
                f"{max_artifact_bytes}",
                details={"planned_bytes": total_size, "limit": max_artifact_bytes},
            )
        os.replace(staging, destination)
        return manifest, destination

    def commit_ffn_coupling(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: FFNCouplingSpec,
        summary: FFNCouplingRunSummary,
        tensors: dict[str, torch.Tensor],
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}-{run_fingerprint[:12]}"
        destination = self.runs / run_id
        if destination.exists():
            run_id = f"{run_id}-{job_id[:6]}"
            destination = self.runs / run_id
        staging = self.jobs / self._component(job_id, "job") / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        _write_json(staging / "spec.json", spec)
        _write_json(staging / "summary.json", summary)
        with (staging / "ffn-coupling-pairs.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for item in summary.pairs:
                handle.write(canonical_json(item) + "\n")
        for filename, records in (
            ("ffn-coupling-layers.csv", summary.layers),
            ("ffn-coupling-neurons.csv", summary.neurons),
        ):
            rows = [item.model_dump(mode="json") for item in records]
            with (staging / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        save_file(tensors, staging / "ffn-coupling-tensors.safetensors")
        source_events = self.jobs / self._component(job_id, "job") / "events.jsonl"
        shutil.copy2(source_events, staging / "events.jsonl")
        media_types = {
            "spec.json": "application/json",
            "summary.json": "application/json",
            "ffn-coupling-pairs.jsonl": "application/x-ndjson",
            "ffn-coupling-layers.csv": "text/csv",
            "ffn-coupling-neurons.csv": "text/csv",
            "ffn-coupling-tensors.safetensors": "application/octet-stream",
            "events.jsonl": "application/x-ndjson",
        }
        refs = tuple(
            _artifact_ref(staging, staging / name, media_type)
            for name, media_type in media_types.items()
        )
        parent_ids = tuple(
            item
            for item in (spec.parent_run_id, spec.trajectory_run_id)
            if item is not None
        )
        manifest = RunManifest(
            run_id=run_id,
            job_id=job_id,
            request_id=request_id,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            created_at=created_at,
            completed_at=timestamp,
            evidence_stage="observational_ffn_coupling",
            run_kind="ffn_coupling",
            parent_run_ids=parent_ids,
            algorithm_version=algorithm_version,
            requested_model=requested_model,
            resolved_model=resolved_model,
            environment=_environment_metadata(),
            pair_count=len(summary.pairs),
            artifacts=refs,
            warnings=summary.warnings,
        )
        _write_json(staging / "manifest.json", manifest)
        total_size = sum(item.size_bytes for item in refs) + (
            staging / "manifest.json"
        ).stat().st_size
        if total_size > max_artifact_bytes:
            shutil.rmtree(staging)
            raise ArtifactError(
                f"artifact size {total_size} exceeds max_artifact_bytes {max_artifact_bytes}"
            )
        os.replace(staging, destination)
        return manifest, destination

    def commit_attention_intervention(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: AttentionHeadInterventionSpec,
        summary: AttentionHeadInterventionRunSummary,
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        return self._commit_child_run(
            job_id=job_id,
            request_id=request_id,
            spec=spec,
            summary=summary,
            requested_model=requested_model,
            resolved_model=resolved_model,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            algorithm_version=algorithm_version,
            created_at=created_at,
            evidence_stage="attention_causal_heads",
            run_kind="attention_intervention",
            parent_run_ids=tuple(
                value
                for value in (spec.parent_run_id, spec.qualification_run_id)
                if value is not None
            ),
            pair_count=len(summary.pairs),
            detail_name="attention-observations.jsonl",
            detail_rows=[item.model_dump(mode="json") for item in summary.observations],
            warnings=summary.warnings,
            max_artifact_bytes=max_artifact_bytes,
        )

    def commit_attention_trace(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: AttentionTraceSpec,
        summary: AttentionTraceRunSummary,
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        detail_rows = [
            item.model_dump(mode="json")
            for item in (
                summary.token_edges
                if summary.trace_kind == "token_edges"
                else summary.paths
            )
        ]
        return self._commit_child_run(
            job_id=job_id,
            request_id=request_id,
            spec=spec,
            summary=summary,
            requested_model=requested_model,
            resolved_model=resolved_model,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            algorithm_version=algorithm_version,
            created_at=created_at,
            evidence_stage=summary.evidence_stage,
            run_kind="attention_trace",
            parent_run_ids=tuple(
                value
                for value in (spec.parent_run_id, spec.parent_intervention_run_id)
                if value is not None
            ),
            pair_count=len(summary.pairs),
            detail_name=(
                "attention-token-edges.jsonl"
                if summary.trace_kind == "token_edges"
                else "attention-paths.jsonl"
            ),
            detail_rows=detail_rows,
            warnings=summary.warnings,
            max_artifact_bytes=max_artifact_bytes,
        )

    def commit_intervention(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: InterventionSpec,
        summary: InterventionRunSummary,
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        rows = [item.model_dump(mode="json") for item in summary.observations]
        return self._commit_child_run(
            job_id=job_id,
            request_id=request_id,
            spec=spec,
            summary=summary,
            requested_model=requested_model,
            resolved_model=resolved_model,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            algorithm_version=algorithm_version,
            created_at=created_at,
            evidence_stage="causal_intervention",
            run_kind="intervention",
            parent_run_ids=tuple(
                value
                for value in (spec.parent_run_id, spec.qualification_run_id)
                if value is not None
            ),
            pair_count=len(summary.pairs),
            detail_name="observations.jsonl",
            detail_rows=rows,
            warnings=summary.warnings,
            max_artifact_bytes=max_artifact_bytes,
        )

    def commit_direction(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: DirectionInjectionSpec,
        summary: DirectionInjectionRunSummary,
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        rows = [item.model_dump(mode="json") for item in summary.observations]
        return self._commit_child_run(
            job_id=job_id,
            request_id=request_id,
            spec=spec,
            summary=summary,
            requested_model=requested_model,
            resolved_model=resolved_model,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            algorithm_version=algorithm_version,
            created_at=created_at,
            evidence_stage="causal_intervention",
            run_kind="direction",
            parent_run_ids=tuple(
                value
                for value in (spec.parent_run_id, spec.qualification_run_id)
                if value is not None
            ),
            pair_count=len(summary.pairs),
            detail_name="direction-observations.jsonl",
            detail_rows=rows,
            warnings=summary.warnings,
            max_artifact_bytes=max_artifact_bytes,
        )

    def _commit_child_run(
        self,
        *,
        job_id: str,
        request_id: str,
        spec: (
            QualificationSpec
            | TrajectorySpec
            | InterventionSpec
            | DirectionInjectionSpec
            | AttentionHeadInterventionSpec
            | AttentionTraceSpec
        ),
        summary: (
            QualificationRunSummary
            | TrajectoryRunSummary
            | InterventionRunSummary
            | DirectionInjectionRunSummary
            | AttentionHeadInterventionRunSummary
            | AttentionTraceRunSummary
        ),
        requested_model: Any,
        resolved_model: dict[str, Any],
        science_hash: str,
        run_fingerprint: str,
        algorithm_version: str,
        created_at: datetime,
        evidence_stage: str,
        run_kind: str,
        parent_run_ids: tuple[str, ...],
        pair_count: int,
        detail_name: str,
        detail_rows: list[dict[str, Any]],
        warnings: tuple[str, ...],
        max_artifact_bytes: int,
    ) -> tuple[RunManifest, Path]:
        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}-{run_fingerprint[:12]}"
        destination = self.runs / run_id
        if destination.exists():
            run_id = f"{run_id}-{job_id[:6]}"
            destination = self.runs / run_id

        staging = self.jobs / self._component(job_id, "job") / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        _write_json(staging / "spec.json", spec)
        _write_json(staging / "summary.json", summary)
        with (staging / detail_name).open("w", encoding="utf-8") as handle:
            for row in detail_rows:
                handle.write(canonical_json(row) + "\n")
        source_events = self.jobs / self._component(job_id, "job") / "events.jsonl"
        shutil.copy2(source_events, staging / "events.jsonl")
        media_types = {
            "spec.json": "application/json",
            "summary.json": "application/json",
            detail_name: "application/x-ndjson",
            "events.jsonl": "application/x-ndjson",
        }
        refs = tuple(
            _artifact_ref(staging, staging / name, media_type)
            for name, media_type in media_types.items()
        )
        manifest = RunManifest(
            run_id=run_id,
            job_id=job_id,
            request_id=request_id,
            science_hash=science_hash,
            run_fingerprint=run_fingerprint,
            created_at=created_at,
            completed_at=timestamp,
            evidence_stage=evidence_stage,
            run_kind=run_kind,
            parent_run_ids=parent_run_ids,
            algorithm_version=algorithm_version,
            requested_model=requested_model,
            resolved_model=resolved_model,
            environment=_environment_metadata(),
            pair_count=pair_count,
            artifacts=refs,
            warnings=warnings,
        )
        _write_json(staging / "manifest.json", manifest)
        total_size = sum(item.size_bytes for item in refs) + (
            staging / "manifest.json"
        ).stat().st_size
        if total_size > max_artifact_bytes:
            shutil.rmtree(staging)
            raise ArtifactError(
                f"artifact size {total_size} exceeds max_artifact_bytes "
                f"{max_artifact_bytes}",
                details={"planned_bytes": total_size, "limit": max_artifact_bytes},
            )
        os.replace(staging, destination)
        return manifest, destination

    def list_runs(self) -> tuple[RunManifest, ...]:
        manifests: list[RunManifest] = []
        for path in sorted(self.runs.glob("*/manifest.json"), reverse=True):
            try:
                manifests.append(
                    RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        return tuple(manifests)

    def load_manifest(self, run_id: str) -> RunManifest:
        path = self.runs / self._component(run_id, "run") / "manifest.json"
        if not path.is_file():
            raise ArtifactError(f"run {run_id!r} was not found")
        try:
            return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactError(f"run {run_id!r} has invalid manifest metadata") from exc

    def load_summary(self, run_id: str) -> dict[str, Any]:
        path = self.runs / self._component(run_id, "run") / "summary.json"
        if not path.is_file():
            raise ArtifactError(f"run {run_id!r} has no summary")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactError(f"run {run_id!r} has an invalid summary") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"run {run_id!r} has an invalid summary")
        return value

    def load_run_spec(self, run_id: str) -> ExperimentSpec:
        path = self.runs / self._component(run_id, "run") / "spec.json"
        if not path.is_file():
            raise ArtifactError(f"run {run_id!r} has no saved spec")
        try:
            return parse_spec_data(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise ArtifactError(f"run {run_id!r} has invalid spec metadata") from exc

    def verify(self, run_id: str) -> tuple[str, ...]:
        manifest = self.load_manifest(run_id)
        directory = (self.runs / self._component(run_id, "run")).resolve()
        failures: list[str] = []
        for ref in manifest.artifacts:
            path = (directory / ref.path).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                failures.append(f"missing:{ref.path}")
                continue
            data = path.read_bytes()
            if len(data) != ref.size_bytes:
                failures.append(f"size:{ref.path}")
            if sha256(data).hexdigest() != ref.sha256:
                failures.append(f"sha256:{ref.path}")
        expected = {item.path for item in manifest.artifacts} | {"manifest.json"}
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        }
        failures.extend(f"untracked:{path}" for path in sorted(actual - expected))
        return tuple(failures)

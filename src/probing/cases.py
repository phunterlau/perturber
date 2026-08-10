from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any
import zipfile

import yaml

from .contracts import (
    AttentionTraceSpec,
    ClaimRecord,
    ErrorDetail,
    ExperimentPlan,
    ExperimentSpec,
    JobStatus,
    QualificationRunSummary,
    ResearchCase,
    ResearchCaseCreate,
    ResearchCasePlan,
    ResearchCaseStage,
    ResearchCaseStagePlan,
    ResearchCaseUpdate,
    ResearchWorkflowSpec,
)
from .errors import ArtifactError, RequestConflictError, SpecError
from .reporting import build_research_report
from .service import ResearchService
from .specs import canonical_json, hash_value
from .workflow import resolve_workflow_stage, workflow_stage_specs


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stage_records(spec: ResearchWorkflowSpec) -> tuple[ResearchCaseStage, ...]:
    return tuple(
        ResearchCaseStage(
            key=key,
            kind=child.kind,
            name=child.name,
            trace_kind=(child.trace_kind if isinstance(child, AttentionTraceSpec) else None),
            spec_hash=hash_value(child),
            status="ready" if key == "rank" else "not_configured",
        )
        for key, child in workflow_stage_specs(spec)
    )


class ResearchCaseRepository:
    """Mutable case notebooks that only point at immutable scientific runs."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.root = self.workspace / "cases"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _component(value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in value):
            raise ArtifactError(f"invalid research case identifier {value!r}")
        return value

    def _path(self, case_id: str) -> Path:
        return self.root / self._component(case_id) / "case.json"

    def create(self, request: ResearchCaseCreate) -> ResearchCase:
        now = datetime.now(timezone.utc)
        digest = hash_value(request)[:10]
        base = f"{now.strftime('%Y%m%dT%H%M%S')}-{digest}"
        with self._lock:
            case_id = base
            suffix = 1
            while (self.root / case_id).exists():
                suffix += 1
                case_id = f"{base}-{suffix}"
            directory = self.root / case_id
            directory.mkdir(parents=False)
            case = ResearchCase(
                case_id=case_id,
                revision=1,
                created_at=now,
                updated_at=now,
                intent=request.intent,
                workflow=request.workflow,
                stages=_stage_records(request.workflow),
            )
            _write_json(directory / "case.json", case)
            return case

    def load(self, case_id: str) -> ResearchCase:
        path = self._path(case_id)
        if not path.is_file():
            raise ArtifactError(f"research case {case_id!r} was not found")
        try:
            return ResearchCase.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactError(f"research case {case_id!r} is invalid") from exc

    def list(self) -> tuple[ResearchCase, ...]:
        values: list[ResearchCase] = []
        for path in sorted(self.root.glob("*/case.json"), reverse=True):
            try:
                values.append(ResearchCase.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return tuple(values)

    def _save(self, case: ResearchCase) -> ResearchCase:
        path = self._path(case.case_id)
        temporary = path.with_suffix(".json.tmp")
        _write_json(temporary, case)
        os.replace(temporary, path)
        return case

    def update(self, case_id: str, request: ResearchCaseUpdate) -> ResearchCase:
        with self._lock:
            current = self.load(case_id)
            if current.revision != request.revision:
                raise RequestConflictError(
                    "research case revision conflict",
                    details={"expected": current.revision, "submitted": request.revision},
                )
            existing = {stage.key: stage for stage in current.stages}
            replacement: list[ResearchCaseStage] = []
            for stage in _stage_records(request.workflow):
                prior = existing.get(stage.key)
                if prior is not None and prior.spec_hash == stage.spec_hash:
                    replacement.append(prior)
                elif prior is not None and (prior.job_id is not None or prior.run_id is not None):
                    raise RequestConflictError(
                        f"executed stage {stage.key!r} cannot be changed",
                        hint="Create a new case to change an executed scientific stage.",
                    )
                else:
                    replacement.append(stage)
            removed = set(existing) - {stage.key for stage in replacement}
            if any(existing[key].job_id or existing[key].run_id for key in removed):
                raise RequestConflictError("executed stages cannot be removed")
            updated = current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                    "intent": request.intent,
                    "workflow": request.workflow,
                    "stages": tuple(replacement),
                }
            )
            return self._save(updated)

    def update_stage(
        self,
        case_id: str,
        stage_key: str,
        **updates: Any,
    ) -> ResearchCase:
        with self._lock:
            current = self.load(case_id)
            stages = list(current.stages)
            try:
                index = next(i for i, item in enumerate(stages) if item.key == stage_key)
            except StopIteration as exc:
                raise SpecError(f"unknown research case stage {stage_key!r}") from exc
            stages[index] = stages[index].model_copy(update=updates)
            updated = current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                    "stages": tuple(stages),
                }
            )
            return self._save(updated)

    def refresh(self, case_id: str, service: ResearchService) -> ResearchCase:
        """Recalculate gates and the conservative evidence label from artifacts."""
        with self._lock:
            current = self.load(case_id)
            run_ids = {stage.key: stage.run_id for stage in current.stages if stage.run_id}
            qualification_ok = False
            qualification_stage = next((s for s in current.stages if s.key == "qualification"), None)
            if qualification_stage and qualification_stage.run_id and not qualification_stage.verification_failures:
                try:
                    qualification = QualificationRunSummary.model_validate(
                        service.repository.load_summary(qualification_stage.run_id)
                    )
                    qualification_ok = qualification.aggregate.claim_eligible
                except Exception:
                    qualification_ok = False

            refreshed: list[ResearchCaseStage] = []
            any_local_causal = False
            any_heldout_causal = False
            for stage in current.stages:
                if stage.run_id:
                    failures = service.repository.verify(stage.run_id)
                    summary = service.repository.load_summary(stage.run_id)
                    raw_claims = tuple(summary.get("claims", ()))
                    claims = tuple(ClaimRecord.model_validate(item) for item in raw_claims)
                    supported = any(item.status == "supported" for item in claims)
                    status = "verified" if not failures else "failed"
                    if status == "verified" and supported and stage.kind in {
                        "intervention", "attention_intervention", "attention_trace"
                    }:
                        any_local_causal = True
                        rows = summary.get("observations") or summary.get("paths") or []
                        any_heldout_causal = any_heldout_causal or any(
                            item.get("split") == "heldout" for item in rows
                        )
                    refreshed.append(
                        stage.model_copy(
                            update={
                                "status": status,
                                "verification_failures": failures,
                                "parent_run_ids": service.repository.load_manifest(stage.run_id).parent_run_ids,
                                "claims": claims,
                                "warnings": tuple(summary.get("warnings", ())),
                            }
                        )
                    )
                    continue
                if stage.status in {"running", "failed"}:
                    refreshed.append(stage)
                    continue
                try:
                    resolve_workflow_stage(current.workflow, stage.key, run_ids)
                    status = "ready"
                    if stage.key != "rank" and "$qualification" in canonical_json(
                        dict(workflow_stage_specs(current.workflow))[stage.key]
                    ) and qualification_stage and qualification_stage.run_id and not qualification_ok:
                        status = "gate_failed"
                    refreshed.append(stage.model_copy(update={"status": status}))
                except (KeyError, ValueError):
                    refreshed.append(stage.model_copy(update={"status": "not_configured"}))

            evidence = (
                "heldout_replicated"
                if any_heldout_causal
                else "locally_causal"
                if any_local_causal
                else "behaviorally_qualified"
                if qualification_ok
                else "observational"
            )
            updated = current.model_copy(
                update={
                    "updated_at": datetime.now(timezone.utc),
                    "stages": tuple(refreshed),
                    "evidence_label": evidence,
                }
            )
            return self._save(updated)


def plan_case(
    repository: ResearchCaseRepository,
    service: ResearchService,
    case_id: str,
) -> ResearchCasePlan:
    case = repository.refresh(case_id, service)
    run_ids = {stage.key: stage.run_id for stage in case.stages if stage.run_id}
    stages: list[ResearchCaseStagePlan] = []
    total = 0
    warnings: list[str] = []
    for stage in case.stages:
        if stage.status in {"verified", "running", "failed", "gate_failed"}:
            stages.append(ResearchCaseStagePlan(key=stage.key, status=stage.status))
            continue
        try:
            resolved = resolve_workflow_stage(case.workflow, stage.key, run_ids)
            plan = service.plan(resolved)
            total += plan.forward_passes
            warnings.extend(plan.warnings)
            stages.append(ResearchCaseStagePlan(key=stage.key, status=stage.status, plan=plan))
        except Exception as exc:
            stages.append(
                ResearchCaseStagePlan(
                    key=stage.key,
                    status="not_configured",
                    blocked_reason=str(exc),
                )
            )
    return ResearchCasePlan(
        case_id=case.case_id,
        evidence_label=case.evidence_label,
        stages=tuple(stages),
        total_forward_passes=total,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def resolved_case_stage(case: ResearchCase, stage_key: str) -> ExperimentSpec:
    stage = next((item for item in case.stages if item.key == stage_key), None)
    if stage is None:
        raise SpecError(f"unknown research case stage {stage_key!r}")
    if stage.status not in {"ready", "not_configured"}:
        raise RequestConflictError(
            f"stage {stage_key!r} is {stage.status} and cannot be started"
        )
    run_ids = {item.key: item.run_id for item in case.stages if item.run_id}
    try:
        return resolve_workflow_stage(case.workflow, stage_key, run_ids)
    except (KeyError, ValueError) as exc:
        raise SpecError(str(exc)) from exc


def finish_case_stage(
    repository: ResearchCaseRepository,
    service: ResearchService,
    case_id: str,
    stage_key: str,
    status: JobStatus,
) -> None:
    if status.state == "completed" and status.run_id:
        failures = service.repository.verify(status.run_id)
        repository.update_stage(
            case_id,
            stage_key,
            status="verified" if not failures else "failed",
            run_id=status.run_id,
            verification_failures=failures,
            error=None,
        )
    elif status.state in {"failed", "cancelled"}:
        repository.update_stage(
            case_id,
            stage_key,
            status="failed",
            error=status.error,
        )
    repository.refresh(case_id, service)


def build_research_packet(
    repository: ResearchCaseRepository,
    service: ResearchService,
    case_id: str,
) -> Path:
    case = repository.refresh(case_id, service)
    directory = repository.root / case.case_id
    destination = directory / "research-packet.zip"
    manifests = []
    reports = []
    verification: dict[str, list[str]] = {}
    claims: list[dict[str, Any]] = []
    for stage in case.stages:
        if not stage.run_id:
            continue
        manifest = service.repository.load_manifest(stage.run_id)
        manifests.append(manifest.model_dump(mode="json"))
        verification[stage.run_id] = list(service.repository.verify(stage.run_id))
        summary = service.repository.load_summary(stage.run_id)
        claims.extend(summary.get("claims", ()))
        reports.append(
            build_research_report(
                run_id=stage.run_id,
                manifest=manifest,
                summary=summary,
            ).model_dump(mode="json")
        )

    workflow_yaml = yaml.safe_dump(
        case.workflow.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    commands = [
        "# Agent continuation commands",
        "",
        f"CASE_ID={case.case_id}",
    ]
    for stage in case.stages:
        if stage.run_id:
            commands.extend(
                [
                    "",
                    f"## {stage.key}",
                    f"uv run --locked probe runs overview {stage.run_id}",
                    f"uv run --locked probe runs verify {stage.run_id}",
                    f"uv run --locked probe report {stage.run_id}",
                ]
            )
    commands.extend(
        [
            "",
            "## CLI-first diagnostics",
            "uv run --locked probe compare REFERENCE_RUN CANDIDATE_RUN --top-n 50",
            "uv run --locked probe stability RANK_RUN --top-n 50 --seed 0",
            "uv run --locked probe sensitivity RANK_RUN --metadata-key perturbation_family",
        ]
    )
    context = {
        "schema_version": "probe.research-packet/v1",
        "case_id": case.case_id,
        "evidence_label": case.evidence_label,
        "run_ids": {stage.key: stage.run_id for stage in case.stages if stage.run_id},
        "claims": claims,
        "verification": verification,
        "unresolved_stages": [
            stage.key for stage in case.stages if stage.status != "verified"
        ],
    }
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workflow.yaml", workflow_yaml)
        archive.writestr("case.json", json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
        archive.writestr("runs.json", json.dumps(manifests, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("claims.json", json.dumps(claims, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("verification.json", json.dumps(verification, indent=2) + "\n")
        archive.writestr("agent-context.json", json.dumps(context, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("reports.json", json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("COMMANDS.md", "\n".join(commands) + "\n")
    os.replace(temporary, destination)
    return destination


def agent_handoff(case: ResearchCase) -> dict[str, Any]:
    run_ids = {stage.key: stage.run_id for stage in case.stages if stage.run_id}
    next_stages = [stage.key for stage in case.stages if stage.status == "ready"]
    prompt = (
        f"Continue perturbation-probing research case {case.case_id}. "
        f"Current evidence: {case.evidence_label.replace('_', ' ')}. "
        f"Run IDs: {json.dumps(run_ids, sort_keys=True)}. "
        f"Ready stages: {', '.join(next_stages) or 'none'}. "
        "Read RESEARCH.md, verify immutable parents, and use the canonical workflow in the research packet."
    )
    return {
        "schema_version": "probe.agent-handoff/v1",
        "case_id": case.case_id,
        "prompt": prompt,
        "run_ids": run_ids,
        "ready_stages": next_stages,
    }


__all__ = [
    "ResearchCaseRepository",
    "agent_handoff",
    "build_research_packet",
    "finish_case_stage",
    "plan_case",
    "resolved_case_stage",
]

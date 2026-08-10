from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from .contracts import (
    AttentionHeadInterventionSpec,
    AttentionHeadRankSpec,
    AttentionTraceSpec,
    DirectionInjectionSpec,
    InterventionSpec,
    ResearchWorkflowOutcome,
    ResearchWorkflowSpec,
    WorkflowStageOutcome,
    ExperimentSpec,
)
from .events import EventListener
from .service import ExecutionOutcome, ResearchService
from .specs import hash_value


def _stage(outcome: ExecutionOutcome, name: str) -> WorkflowStageOutcome:
    summary = outcome.summary
    return WorkflowStageOutcome(
        kind=outcome.manifest.run_kind,
        name=name,
        run_id=outcome.manifest.run_id,
        parent_run_ids=outcome.manifest.parent_run_ids,
        evidence_stage=outcome.manifest.evidence_stage,
        logical_forward_passes=summary.logical_forward_passes,
        claims=summary.claims,
        warnings=summary.warnings,
    )


def _resolve_causal_stage(
    child: InterventionSpec | DirectionInjectionSpec,
    *,
    rank_run_id: str,
    qualification_run_id: str | None,
) -> InterventionSpec | DirectionInjectionSpec:
    updates: dict[str, str | None] = {"parent_run_id": rank_run_id}
    if child.qualification_run_id == "$qualification":
        if qualification_run_id is None:
            raise ValueError(
                "causal workflow stage requires a completed qualification run"
            )
        updates["qualification_run_id"] = qualification_run_id
    return child.model_copy(update=updates)


def _resolve_attention_rank_stage(
    child: AttentionHeadRankSpec,
    *,
    rank_run_id: str,
    qualification_run_id: str | None,
) -> AttentionHeadRankSpec:
    updates: dict[str, str | None] = {"parent_run_id": rank_run_id}
    if child.qualification_run_id == "$qualification":
        if qualification_run_id is None:
            raise ValueError("attention ranking requires completed qualification")
        updates["qualification_run_id"] = qualification_run_id
    return child.model_copy(update=updates)


def _resolve_attention_intervention_stage(
    child: AttentionHeadInterventionSpec,
    *,
    attention_rank_run_id: str,
    qualification_run_id: str | None,
) -> AttentionHeadInterventionSpec:
    updates: dict[str, str | None] = {"parent_run_id": attention_rank_run_id}
    if child.qualification_run_id == "$qualification":
        if qualification_run_id is None:
            raise ValueError("attention intervention requires completed qualification")
        updates["qualification_run_id"] = qualification_run_id
    return child.model_copy(update=updates)


def workflow_stage_specs(
    spec: ResearchWorkflowSpec,
) -> tuple[tuple[str, ExperimentSpec], ...]:
    """Return configured stages with stable keys shared by CLI and WebUI."""
    stages: list[tuple[str, ExperimentSpec]] = [("rank", spec.rank)]
    if spec.qualification is not None:
        stages.append(("qualification", spec.qualification))
    stages.extend(
        (f"intervention-{index}", child)
        for index, child in enumerate(spec.interventions, start=1)
    )
    stages.extend(
        (f"direction-{index}", child)
        for index, child in enumerate(spec.directions, start=1)
    )
    if spec.attention_rank is not None:
        stages.append(("attention-rank", spec.attention_rank))
    stages.extend(
        (f"attention-intervention-{index}", child)
        for index, child in enumerate(spec.attention_interventions, start=1)
    )
    trace_counts = {"token_edges": 0, "head_paths": 0}
    for child in spec.attention_traces:
        trace_counts[child.trace_kind] += 1
        prefix = (
            "attention-token-edges"
            if child.trace_kind == "token_edges"
            else "attention-head-paths"
        )
        stages.append((f"{prefix}-{trace_counts[child.trace_kind]}", child))
    return tuple(stages)


def resolve_workflow_stage(
    workflow: ResearchWorkflowSpec,
    stage_key: str,
    run_ids: dict[str, str],
) -> ExperimentSpec:
    """Resolve one symbolic workflow stage against completed immutable parents."""
    stages = dict(workflow_stage_specs(workflow))
    if stage_key not in stages:
        raise KeyError(f"unknown workflow stage {stage_key!r}")
    child = stages[stage_key]
    if stage_key == "rank":
        return child
    rank_run_id = run_ids.get("rank")
    if rank_run_id is None:
        raise ValueError("rank stage has not completed")
    qualification_run_id = run_ids.get("qualification")
    if stage_key == "qualification":
        return child.model_copy(update={"parent_run_id": rank_run_id})
    if isinstance(child, (InterventionSpec, DirectionInjectionSpec)):
        return _resolve_causal_stage(
            child,
            rank_run_id=rank_run_id,
            qualification_run_id=qualification_run_id,
        )
    if isinstance(child, AttentionHeadRankSpec):
        return _resolve_attention_rank_stage(
            child,
            rank_run_id=rank_run_id,
            qualification_run_id=qualification_run_id,
        )
    attention_rank_run_id = run_ids.get("attention-rank")
    if attention_rank_run_id is None:
        raise ValueError("attention rank stage has not completed")
    if isinstance(child, AttentionHeadInterventionSpec):
        return _resolve_attention_intervention_stage(
            child,
            attention_rank_run_id=attention_rank_run_id,
            qualification_run_id=qualification_run_id,
        )
    updates: dict[str, str | None] = {"parent_run_id": attention_rank_run_id}
    if child.parent_intervention_run_id == "$attention_intervention":
        intervention_keys = [
            key
            for key, _value in workflow_stage_specs(workflow)
            if key.startswith("attention-intervention-")
        ]
        completed = [run_ids[key] for key in intervention_keys if key in run_ids]
        if len(intervention_keys) != 1 or len(completed) != 1:
            raise ValueError(
                "attention path requires its single attention intervention parent"
            )
        updates["parent_intervention_run_id"] = completed[0]
    return child.model_copy(update=updates)


def run_workflow(
    *,
    service: ResearchService,
    spec: ResearchWorkflowSpec,
    listener: EventListener | None = None,
    diagnostic_stream=None,
) -> ResearchWorkflowOutcome:
    """Execute all stages in one process so a managed model can be reused."""

    stages: list[WorkflowStageOutcome] = []
    rank = service.execute(
        spec.rank,
        listener=listener,
        diagnostic_stream=diagnostic_stream,
    )
    rank_run_id = rank.manifest.run_id
    stages.append(_stage(rank, spec.rank.name))

    qualification_run_id = None
    if spec.qualification is not None:
        qualification_spec = spec.qualification.model_copy(
            update={"parent_run_id": rank_run_id}
        )
        qualification = service.execute(
            qualification_spec,
            listener=listener,
            diagnostic_stream=diagnostic_stream,
        )
        qualification_run_id = qualification.manifest.run_id
        stages.append(_stage(qualification, qualification_spec.name))

    intervention_run_ids: list[str] = []
    for child in spec.interventions:
        resolved = _resolve_causal_stage(
            child,
            rank_run_id=rank_run_id,
            qualification_run_id=qualification_run_id,
        )
        assert isinstance(resolved, InterventionSpec)
        outcome = service.execute(
            resolved,
            listener=listener,
            diagnostic_stream=diagnostic_stream,
        )
        intervention_run_ids.append(outcome.manifest.run_id)
        stages.append(_stage(outcome, resolved.name))

    direction_run_ids: list[str] = []
    for child in spec.directions:
        resolved = _resolve_causal_stage(
            child,
            rank_run_id=rank_run_id,
            qualification_run_id=qualification_run_id,
        )
        assert isinstance(resolved, DirectionInjectionSpec)
        outcome = service.execute(
            resolved,
            listener=listener,
            diagnostic_stream=diagnostic_stream,
        )
        direction_run_ids.append(outcome.manifest.run_id)
        stages.append(_stage(outcome, resolved.name))

    attention_rank_run_id = None
    attention_intervention_run_ids: list[str] = []
    attention_trace_run_ids: list[str] = []
    if spec.attention_rank is not None:
        resolved_attention_rank = _resolve_attention_rank_stage(
            spec.attention_rank,
            rank_run_id=rank_run_id,
            qualification_run_id=qualification_run_id,
        )
        attention_rank_outcome = service.execute(
            resolved_attention_rank,
            listener=listener,
            diagnostic_stream=diagnostic_stream,
        )
        attention_rank_run_id = attention_rank_outcome.manifest.run_id
        stages.append(_stage(attention_rank_outcome, resolved_attention_rank.name))

        for child in spec.attention_interventions:
            resolved = _resolve_attention_intervention_stage(
                child,
                attention_rank_run_id=attention_rank_run_id,
                qualification_run_id=qualification_run_id,
            )
            outcome = service.execute(
                resolved,
                listener=listener,
                diagnostic_stream=diagnostic_stream,
            )
            attention_intervention_run_ids.append(outcome.manifest.run_id)
            stages.append(_stage(outcome, resolved.name))

        for child in spec.attention_traces:
            updates: dict[str, str | None] = {
                "parent_run_id": attention_rank_run_id
            }
            if child.parent_intervention_run_id == "$attention_intervention":
                if len(attention_intervention_run_ids) != 1:
                    raise ValueError(
                        "attention trace symbolic intervention requires exactly one completed intervention"
                    )
                updates["parent_intervention_run_id"] = (
                    attention_intervention_run_ids[0]
                )
            resolved_trace = child.model_copy(update=updates)
            outcome = service.execute(
                resolved_trace,
                listener=listener,
                diagnostic_stream=diagnostic_stream,
            )
            attention_trace_run_ids.append(outcome.manifest.run_id)
            stages.append(_stage(outcome, resolved_trace.name))

    claims = []
    seen_claims: set[tuple[str, str]] = set()
    for stage in stages:
        for claim in stage.claims:
            key = (stage.run_id, claim.claim_id)
            if key not in seen_claims:
                claims.append(claim)
                seen_claims.add(key)
    warnings = tuple(
        dict.fromkeys(warning for stage in stages for warning in stage.warnings)
    )
    workflow_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
        f"{hash_value(spec)[:12]}"
    )
    result = ResearchWorkflowOutcome(
        workflow_id=workflow_id,
        name=spec.name,
        stages=tuple(stages),
        rank_run_id=rank_run_id,
        qualification_run_id=qualification_run_id,
        intervention_run_ids=tuple(intervention_run_ids),
        direction_run_ids=tuple(direction_run_ids),
        attention_rank_run_id=attention_rank_run_id,
        attention_intervention_run_ids=tuple(attention_intervention_run_ids),
        attention_trace_run_ids=tuple(attention_trace_run_ids),
        logical_forward_passes=sum(item.logical_forward_passes for item in stages),
        claims=tuple(claims),
        warnings=warnings,
    )
    persist_workflow(service.workspace, spec, result)
    return result


def persist_workflow(
    workspace: Path,
    spec: ResearchWorkflowSpec,
    result: ResearchWorkflowOutcome,
) -> Path:
    root = workspace.resolve() / "workflows"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / result.workflow_id
    if destination.exists():
        raise FileExistsError(f"workflow {result.workflow_id!r} already exists")
    staging = root / f".{result.workflow_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name, value in (("driver.json", spec), ("outcome.json", result)):
        path = staging / name
        path.write_text(
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    os.replace(staging, destination)
    return destination


__all__ = [
    "persist_workflow",
    "resolve_workflow_stage",
    "run_workflow",
    "workflow_stage_specs",
]

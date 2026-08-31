from __future__ import annotations

from pathlib import Path

from probing.contracts import FFNCouplingSpec, ResearchWorkflowSpec, TrajectorySpec
from probing.reporting import build_research_report
from probing.specs import load_document, parse_spec_data
from probing.workflow import run_workflow
from test_service import fake_rank_spec, make_service


def coupling_spec(
    parent_run_id: str,
    *,
    max_forward_passes: int = 2,
    max_backward_passes: int = 2,
) -> FFNCouplingSpec:
    return FFNCouplingSpec.model_validate(
        {
            "kind": "ffn_coupling",
            "name": "fixture-layer-aware-coupling",
            "parent_run_id": parent_run_id,
            "top_k": 2,
            "max_backward_passes": max_backward_passes,
            "execution": {
                "max_forward_passes": max_forward_passes,
                "max_artifact_bytes": 2_000_000,
                "seed": 19,
            },
        }
    )


def test_ffn_coupling_spec_round_trips() -> None:
    parsed = parse_spec_data(
        coupling_spec("rank-fixture").model_dump(mode="json")
    )

    assert isinstance(parsed, FFNCouplingSpec)
    assert parsed.methods == (
        "native_local_readout",
        "downstream_endpoint_gradient",
    )


def test_service_executes_verified_layer_aware_coupling(tmp_path) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    spec = coupling_spec(rank.manifest.run_id)

    plan = service.plan(spec)
    outcome = service.execute(spec)

    assert plan.forward_passes == 2
    assert plan.backward_passes == 2
    assert outcome.manifest.run_kind == "ffn_coupling"
    assert outcome.summary.logical_backward_passes == 2
    assert outcome.summary.total_neuron_count == 2
    assert len(outcome.summary.neurons) == 2
    leading = outcome.summary.neurons[0]
    assert leading.downstream_importance_rms > 0
    assert leading.direct_importance_rms > 0
    assert leading.native_importance_rms is not None
    assert service.repository.verify(outcome.manifest.run_id) == ()
    assert (outcome.run_directory / "ffn-coupling-tensors.safetensors").is_file()
    report = build_research_report(
        run_id=outcome.manifest.run_id,
        manifest=outcome.manifest,
        summary=outcome.summary.model_dump(mode="json"),
    )
    assert report.headline == "Observational layer-aware FFN coupling"


def test_ffn_coupling_plan_accounts_for_backward_budget(tmp_path) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    spec = coupling_spec(rank.manifest.run_id, max_backward_passes=1)

    plan = service.plan(spec)

    assert plan.forward_passes == 2
    assert plan.backward_passes == 2
    assert plan.within_budget is False
    assert "backward passes" in plan.warnings[0]


def test_workflow_resolves_trajectory_and_coupling_lineage(tmp_path) -> None:
    service = make_service(tmp_path)
    trajectory_spec = TrajectorySpec.model_validate(
        {
            "kind": "trajectory",
            "name": "workflow-trajectory",
            "parent_run_id": "$rank",
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 2_000_000,
            },
        }
    )
    coupling_specification = coupling_spec("$rank").model_copy(
        update={"trajectory_run_id": "$trajectory"}
    )
    workflow = ResearchWorkflowSpec(
        name="fixture-trajectory-coupling",
        rank=fake_rank_spec(),
        trajectory=trajectory_spec,
        ffn_coupling=coupling_specification,
    )

    outcome = run_workflow(service=service, spec=workflow)

    assert trajectory_spec.parent_run_id == "$rank"
    assert coupling_specification.trajectory_run_id == "$trajectory"
    assert outcome.trajectory_run_id is not None
    assert outcome.ffn_coupling_run_id is not None
    coupling_manifest = service.repository.load_manifest(outcome.ffn_coupling_run_id)
    assert coupling_manifest.parent_run_ids == (
        outcome.rank_run_id,
        outcome.trajectory_run_id,
    )


def test_checked_in_capital_workflow_preserves_boolean_like_tokens() -> None:
    driver = Path(__file__).parents[1] / "examples/workflows/capital-trajectory-coupling.yaml"

    workflow = ResearchWorkflowSpec.model_validate(load_document(driver))

    assert workflow.rank.observable.target_tokens == ("No",)
    assert workflow.rank.observable.control_tokens == ("Yes",)
    assert workflow.ffn_coupling is not None
    assert workflow.ffn_coupling.trajectory_run_id == "$trajectory"

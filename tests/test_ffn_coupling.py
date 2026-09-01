from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file

from helpers import FakeAdapter
from probing.engine import ProbeEngine
from probing.ffn_coupling import run_ffn_coupling
from probing.contracts import (
    FFNCouplingSpec,
    InterventionSpec,
    ResearchWorkflowSpec,
    TrajectorySpec,
)
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


def coupling_intervention(
    parent_run_id: str, *, trajectory_run_id: str | None = None
) -> InterventionSpec:
    return InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "name": "downstream-candidate-ablation",
            "parent_run_id": parent_run_id,
            "trajectory_run_id": trajectory_run_id,
            "selection": {"strategy": "ranked_top_k", "top_k": 1},
            "operation": {"mode": "ablate", "condition": "perturbed"},
            "sweep": {"neuron_counts": [1], "strengths": [0.0]},
            "controls": {"samples": 0},
            "execution": {
                "max_forward_passes": 1,
                "max_artifact_bytes": 2_000_000,
                "seed": 23,
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


def test_ffn_coupling_candidates_exclude_heldout_pairs(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = fake_rank_spec(pairs=2).model_dump(mode="json")
    payload["pairs"][1]["split"] = "heldout"
    rank_specification = type(fake_rank_spec()).model_validate(payload)
    rank = service.execute(rank_specification)
    tensors = load_file(rank.run_directory / "tensors.safetensors")
    tensors["activation_delta.pair_1.layer_0"] = tensors[
        "activation_delta.pair_1.layer_0"
    ] * 10_000
    engine = ProbeEngine(FakeAdapter())
    all_pairs = run_ffn_coupling(
        engine=engine,
        parent_spec=rank_specification,
        parent_summary=rank.summary,
        parent_tensors=tensors,
        spec=coupling_spec(rank.manifest.run_id, max_forward_passes=4, max_backward_passes=4),
        science_hash="all-pairs",
    ).summary
    discovery_only = run_ffn_coupling(
        engine=engine,
        parent_spec=rank_specification,
        parent_summary=rank.summary,
        parent_tensors=tensors,
        spec=coupling_spec(rank.manifest.run_id).model_copy(
            update={"pair_ids": ("capital",)}
        ),
        science_hash="discovery-only",
    ).summary

    assert all_pairs.pair_count == 2
    assert all_pairs.candidate_pair_ids == ("capital",)
    assert [
        (item.layer, item.neuron, item.downstream_importance_rms)
        for item in all_pairs.neurons
    ] == [
        (item.layer, item.neuron, item.downstream_importance_rms)
        for item in discovery_only.neurons
    ]


def test_intervention_selects_downstream_candidates_from_coupling_parent(
    tmp_path,
) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    coupling = service.execute(coupling_spec(rank.manifest.run_id))
    spec = coupling_intervention(coupling.manifest.run_id)

    assert service.plan(spec).forward_passes == 1
    outcome = service.execute(spec)

    expected = coupling.summary.neurons[0]
    selected = outcome.summary.selected_neurons[0]
    assert (selected.layer, selected.neuron) == (expected.layer, expected.neuron)
    assert selected.importance_mean == expected.downstream_importance_mean
    assert selected.score_method == "downstream_endpoint_gradient"
    assert outcome.summary.candidate_score_method == "downstream_endpoint_gradient"
    assert outcome.summary.parent_run_id == coupling.manifest.run_id
    assert outcome.summary.rank_run_id == rank.manifest.run_id
    assert outcome.manifest.parent_run_ids == (coupling.manifest.run_id,)
    assert service.repository.verify(outcome.manifest.run_id) == ()


def test_intervention_selects_preregistered_direct_downstream_overlap(tmp_path) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    coupling = service.execute(coupling_spec(rank.manifest.run_id))
    payload = coupling_intervention(coupling.manifest.run_id).model_dump(mode="json")
    payload["selection"] = {
        "strategy": "ranked_top_k",
        "top_k": 1,
        "candidate_method": "direct_downstream_overlap",
        "overlap_pool_size": 2,
    }
    spec = InterventionSpec.model_validate(payload)

    outcome = service.execute(spec)

    selected = outcome.summary.selected_neurons[0]
    direct_pool = {
        (item.layer, item.neuron)
        for item in sorted(
            coupling.summary.neurons,
            key=lambda item: (-item.direct_importance_rms, item.layer, item.neuron),
        )[:2]
    }
    downstream_pool = {
        (item.layer, item.neuron) for item in coupling.summary.neurons[:2]
    }
    assert (selected.layer, selected.neuron) in direct_pool & downstream_pool
    assert selected.score_method == "direct_downstream_overlap"
    assert outcome.summary.candidate_score_method == "direct_downstream_overlap"


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
    intervention = coupling_intervention(
        "$ffn_coupling", trajectory_run_id="$trajectory"
    )
    workflow = ResearchWorkflowSpec(
        name="fixture-trajectory-coupling",
        rank=fake_rank_spec(),
        trajectory=trajectory_spec,
        ffn_coupling=coupling_specification,
        interventions=(intervention,),
    )

    outcome = run_workflow(service=service, spec=workflow)

    assert trajectory_spec.parent_run_id == "$rank"
    assert coupling_specification.trajectory_run_id == "$trajectory"
    assert outcome.trajectory_run_id is not None
    assert outcome.ffn_coupling_run_id is not None
    assert len(outcome.intervention_run_ids) == 1
    coupling_manifest = service.repository.load_manifest(outcome.ffn_coupling_run_id)
    assert coupling_manifest.parent_run_ids == (
        outcome.rank_run_id,
        outcome.trajectory_run_id,
    )
    intervention_manifest = service.repository.load_manifest(
        outcome.intervention_run_ids[0]
    )
    intervention_summary = service.repository.load_summary(
        outcome.intervention_run_ids[0]
    )
    assert intervention_summary["trajectory_run_id"] == outcome.trajectory_run_id
    assert len(intervention_summary["trajectory_overlays"]) == 3
    assert {
        item["checkpoint"] for item in intervention_summary["trajectory_overlays"]
    } == {"block_input", "post_attention", "post_ffn"}
    assert intervention_manifest.parent_run_ids == (
        outcome.ffn_coupling_run_id,
        outcome.trajectory_run_id,
    )


def test_checked_in_capital_workflow_preserves_boolean_like_tokens() -> None:
    driver = Path(__file__).parents[1] / "examples/workflows/capital-trajectory-coupling.yaml"

    workflow = ResearchWorkflowSpec.model_validate(load_document(driver))

    assert workflow.rank.observable.target_tokens == ("No",)
    assert workflow.rank.observable.control_tokens == ("Yes",)
    assert workflow.ffn_coupling is not None
    assert workflow.ffn_coupling.trajectory_run_id == "$trajectory"


def test_checked_in_causal_followup_freezes_split_and_overlap_controls() -> None:
    driver = (
        Path(__file__).parents[1]
        / "examples/workflows/language-trajectory-causal-followup.yaml"
    )

    workflow = ResearchWorkflowSpec.model_validate(load_document(driver))

    assert [pair.split for pair in workflow.rank.pairs] == [
        "discovery",
        "discovery",
        "heldout",
    ]
    assert workflow.ffn_coupling is not None
    assert workflow.ffn_coupling.trajectory_run_id == "$trajectory"
    assert len(workflow.interventions) == 3
    overlap = workflow.interventions[2]
    assert overlap.parent_run_id == "$ffn_coupling"
    assert overlap.trajectory_run_id == "$trajectory"
    assert overlap.selection.candidate_method == "direct_downstream_overlap"
    assert overlap.selection.overlap_pool_size == 24

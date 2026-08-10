import pytest
from safetensors.torch import load_file

from probing.comparison import compare_rank_runs, rank_stability
from probing.sensitivity import perturbation_sensitivity
from probing.contracts import (
    DirectionInjectionSpec,
    InterventionSpec,
    QualificationSpec,
    RankRunSummary,
)
from test_service import fake_rank_spec, make_service


def _qualification(parent_run_id: str, *, passes: int = 2) -> QualificationSpec:
    return QualificationSpec.model_validate(
        {
            "kind": "qualify",
            "parent_run_id": parent_run_id,
            "generation": {"max_new_tokens": 1, "seed": 17},
            "evaluator": {"kind": "token_set"},
            "execution": {
                "max_forward_passes": passes,
                "max_artifact_bytes": 1_000_000,
                "seed": 17,
            },
        }
    )


def _ablation(parent_run_id: str, *, pairs: tuple[str, ...] = ()) -> InterventionSpec:
    return InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "parent_run_id": parent_run_id,
            "pair_ids": pairs,
            "selection": {"strategy": "ranked_top_k", "top_k": 1},
            "operation": {
                "mode": "ablate",
                "condition": "perturbed",
            },
            "sweep": {"neuron_counts": [1], "strengths": [0.0]},
            "controls": {"samples": 3},
            "execution": {
                "max_forward_passes": 8 if not pairs else 4,
                "max_artifact_bytes": 1_000_000,
                "seed": 23,
            },
        }
    )


def test_rank_records_first_token_qualification_and_claim_gate(tmp_path) -> None:
    service = make_service(tmp_path)

    outcome = service.execute(fake_rank_spec())

    pair = outcome.summary.pairs[0]
    assert pair.qualification is not None
    assert pair.qualification.status == "informative"
    assert outcome.summary.qualification is not None
    assert outcome.summary.qualification.claim_eligible is True
    assert outcome.summary.claims[0].claim_type == "observable_validity"
    assert "signal concentration" in pair.circuit_regime


def test_generated_qualification_is_immutable_child_run(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec())
    spec = _qualification(parent.manifest.run_id)

    assert service.plan(spec).forward_passes == 2
    outcome = service.execute(spec)

    assert outcome.manifest.run_kind == "qualify"
    assert outcome.manifest.parent_run_ids == (parent.manifest.run_id,)
    assert outcome.summary.aggregate.informative_pairs == 1
    assert all(
        item.agrees_with_observable
        for item in outcome.summary.pairs[0].generated
    )
    assert service.repository.verify(outcome.manifest.run_id) == ()


def test_ablation_sweep_uses_matched_random_controls(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec(pairs=2))
    spec = _ablation(parent.manifest.run_id)

    plan = service.plan(spec)
    assert plan.forward_passes == 8
    outcome = service.execute(spec)

    assert outcome.manifest.run_kind == "intervention"
    assert outcome.summary.logical_forward_passes == 8
    assert len(outcome.summary.observations) == 8
    dose = outcome.summary.doses[0]
    assert dose.pair_count == 2
    assert dose.random_observation_count == 6
    assert dose.controlled_absolute_effect is not None
    assert dose.controlled_absolute_effect > 0
    assert outcome.summary.claims[0].claim_type == "necessity"
    assert outcome.summary.claims[0].status == "supported"
    assert service.repository.verify(outcome.manifest.run_id) == ()


def test_patch_moves_original_toward_perturbed_source(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec())
    spec = InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "parent_run_id": parent.manifest.run_id,
            "selection": {"strategy": "ranked_top_k", "top_k": 1},
            "operation": {"mode": "patch"},
            "sweep": {"neuron_counts": [1], "strengths": [1.0]},
            "controls": {"samples": 0},
            "execution": {
                "max_forward_passes": 1,
                "max_artifact_bytes": 1_000_000,
            },
        }
    )

    outcome = service.execute(spec)
    observation = outcome.summary.observations[0]

    assert observation.condition == "original"
    assert observation.baseline_gap == pytest.approx(-2.0)
    assert observation.source_gap == pytest.approx(3.0)
    assert observation.intervention_gap == pytest.approx(2.0)
    assert observation.normalized_source_progress == pytest.approx(0.8)


def test_pairwise_additivity_reports_linear_fixture_closure(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec())
    spec = InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "parent_run_id": parent.manifest.run_id,
            "selection": {"strategy": "ranked_top_k", "top_k": 2},
            "operation": {"mode": "ablate", "condition": "perturbed"},
            "sweep": {"neuron_counts": [2], "strengths": [0.0]},
            "controls": {"samples": 0},
            "additivity": {"top_n": 2},
            "execution": {
                "max_forward_passes": 4,
                "max_artifact_bytes": 1_000_000,
            },
        }
    )

    outcome = service.execute(spec)

    assert outcome.summary.logical_forward_passes == 4
    assert len(outcome.summary.additivity) == 1
    assert outcome.summary.additivity[0].epsilon == pytest.approx(0.0)


def test_dose_response_estimates_causal_width_without_equating_it_to_top_k(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec())
    spec = InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "parent_run_id": parent.manifest.run_id,
            "selection": {"strategy": "ranked_top_k", "top_k": 2},
            "operation": {"mode": "ablate", "condition": "perturbed"},
            "sweep": {"neuron_counts": [1, 2], "strengths": [0.0]},
            "controls": {"samples": 0},
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 1_000_000,
            },
        }
    )

    outcome = service.execute(spec)

    assert outcome.summary.causal_width[0].width_at_90_percent == 1
    assert outcome.summary.causal_width[0].monotonic is False


def test_scientific_comparison_and_split_half_stability(tmp_path) -> None:
    service = make_service(tmp_path)
    first = service.execute(fake_rank_spec(pairs=2))
    second = service.execute(fake_rank_spec(pairs=2))
    report = compare_rank_runs(
        reference_run_id=first.manifest.run_id,
        reference=first.summary,
        candidates=((second.manifest.run_id, second.summary),),
        top_n=2,
    )

    assert report.scientific_replication is False
    assert report.comparisons[0].overlap_fraction == 1
    changed_spec = fake_rank_spec(pairs=2).model_copy(
        update={
            "observable": fake_rank_spec(pairs=2).observable.model_copy(
                update={"target_tokens": ("Maybe",)}
            )
        }
    )
    classified = compare_rank_runs(
        reference_run_id=first.manifest.run_id,
        reference=first.summary,
        candidates=((second.manifest.run_id, second.summary),),
        top_n=2,
        reference_spec=fake_rank_spec(pairs=2),
        candidate_specs=(changed_spec,),
    )
    assert classified.comparisons[0].changed_factors == ("observable_token_set",)
    tensors = load_file(first.run_directory / "tensors.safetensors")
    stability = rank_stability(
        run_id=first.manifest.run_id,
        summary=RankRunSummary.model_validate(first.summary),
        tensors=tensors,
        top_n=2,
        splits=10,
        bootstrap_iterations=20,
        seed=5,
    )
    assert stability.split_count >= 1
    assert stability.mean_top_n_overlap == 1
    assert len(stability.neuron_intervals) == 2


def test_direction_injection_uses_orthogonal_random_controls(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec(pairs=2))
    spec = DirectionInjectionSpec.model_validate(
        {
            "kind": "direction",
            "parent_run_id": parent.manifest.run_id,
            "layers": [0],
            "betas": [0.5],
            "condition": "perturbed",
            "normalization": "residual_norm",
            "random_direction_samples": 3,
            "execution": {
                "max_forward_passes": 8,
                "max_artifact_bytes": 1_000_000,
                "seed": 31,
            },
        }
    )

    assert service.plan(spec).forward_passes == 8
    outcome = service.execute(spec)

    assert outcome.manifest.run_kind == "direction"
    assert outcome.summary.logical_forward_passes == 8
    assert outcome.summary.doses[0].controlled_absolute_effect == pytest.approx(1.0)
    assert outcome.summary.claims[0].status == "supported"
    assert service.repository.verify(outcome.manifest.run_id) == ()


def test_intervention_records_collateral_observable_effects(tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec())
    spec = InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "parent_run_id": parent.manifest.run_id,
            "selection": {"strategy": "ranked_top_k", "top_k": 1},
            "operation": {"mode": "ablate", "condition": "perturbed"},
            "sweep": {"neuron_counts": [1], "strengths": [0.0]},
            "controls": {"samples": 0},
            "collateral_observables": [
                {
                    "name": "maybe-vs-yes",
                    "target_tokens": ["Maybe"],
                    "control_tokens": ["Yes"],
                }
            ],
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 1_000_000,
            },
        }
    )

    assert service.plan(spec).forward_passes == 2
    outcome = service.execute(spec)

    assert outcome.summary.observations[0].collateral_gap_effects == {
        "maybe-vs-yes": pytest.approx(0.0)
    }


def test_perturbation_family_sensitivity_uses_saved_per_pair_tensors(tmp_path) -> None:
    service = make_service(tmp_path)
    source = fake_rank_spec(pairs=2).model_dump(mode="json")
    source["pairs"][0]["metadata"] = {"perturbation_family": "lexical"}
    source["pairs"][1]["metadata"] = {"perturbation_family": "semantic"}
    spec = type(fake_rank_spec()).model_validate(source)
    outcome = service.execute(spec)
    tensors = load_file(outcome.run_directory / "tensors.safetensors")

    report = perturbation_sensitivity(
        run_id=outcome.manifest.run_id,
        spec=spec,
        summary=outcome.summary,
        tensors=tensors,
        metadata_key="perturbation_family",
        top_n=2,
    )

    assert set(report.groups) == {"lexical", "semantic"}
    assert report.comparisons[0].top_n_overlap == 1

from __future__ import annotations

from probing.contracts import TrajectorySpec
from probing.reporting import build_research_report
from probing.specs import parse_spec_data
from test_service import fake_rank_spec, make_service


def trajectory_spec(parent_run_id: str, *, max_forward_passes: int = 2) -> TrajectorySpec:
    return TrajectorySpec.model_validate(
        {
            "kind": "trajectory",
            "name": "fixture-native-trajectory",
            "parent_run_id": parent_run_id,
            "execution": {
                "max_forward_passes": max_forward_passes,
                "max_artifact_bytes": 2_000_000,
                "seed": 17,
            },
        }
    )


def test_trajectory_spec_round_trips_through_discriminated_parser() -> None:
    source = trajectory_spec("rank-fixture").model_dump(mode="json")

    parsed = parse_spec_data(source)

    assert isinstance(parsed, TrajectorySpec)
    assert parsed.checkpoints == ("block_input", "post_attention", "post_ffn")


def test_service_executes_verified_native_trajectory(tmp_path) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    spec = trajectory_spec(rank.manifest.run_id)

    plan = service.plan(spec)
    outcome = service.execute(spec)

    assert plan.kind == "trajectory"
    assert plan.forward_passes == 2
    assert outcome.manifest.run_kind == "trajectory"
    assert outcome.manifest.parent_run_ids == (rank.manifest.run_id,)
    assert outcome.summary.evidence_stage == "observational_trajectory"
    assert outcome.summary.logical_forward_passes == 2
    assert outcome.summary.pairs[0].final_pair_delta == rank.summary.measured_delta_mean
    assert len(outcome.summary.pairs[0].checkpoints) == 3
    assert outcome.summary.pairs[0].transitions[0].absolute_change > 0
    assert service.repository.verify(outcome.manifest.run_id) == ()
    assert (outcome.run_directory / "trajectory-checkpoints.jsonl").is_file()
    report = build_research_report(
        run_id=outcome.manifest.run_id,
        manifest=outcome.manifest,
        summary=outcome.summary.model_dump(mode="json"),
    )
    assert report.headline == "Observational native paired trajectory"


def test_trajectory_plan_rejects_insufficient_budget(tmp_path) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    spec = trajectory_spec(rank.manifest.run_id, max_forward_passes=1)

    plan = service.plan(spec)

    assert plan.forward_passes == 2
    assert plan.within_budget is False

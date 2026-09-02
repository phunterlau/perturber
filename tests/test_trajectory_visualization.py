from __future__ import annotations

import json

from typer.testing import CliRunner

from probing import cli
from probing.contracts import InterventionSpec
from probing.trajectory_visualization import render_trajectory_visualization
from test_service import fake_rank_spec, make_service
from test_trajectory import trajectory_spec


runner = CliRunner()


def _fixture_runs(tmp_path):
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    trajectory = service.execute(trajectory_spec(rank.manifest.run_id))
    intervention = service.execute(
        InterventionSpec.model_validate(
            {
                "kind": "intervention",
                "name": "trajectory-overlay-fixture",
                "parent_run_id": rank.manifest.run_id,
                "trajectory_run_id": trajectory.manifest.run_id,
                "selection": {"strategy": "ranked_top_k", "top_k": 1},
                "operation": {"mode": "patch", "condition": "auto"},
                "sweep": {"neuron_counts": [1], "strengths": [1.0]},
                "controls": {"samples": 3, "same_layer": True},
                "execution": {
                    "max_forward_passes": 4,
                    "max_artifact_bytes": 2_000_000,
                    "seed": 29,
                },
            }
        )
    )
    return service, trajectory, intervention


def test_renderer_separates_observational_and_controlled_evidence(tmp_path) -> None:
    _service, trajectory, intervention = _fixture_runs(tmp_path)

    html = render_trajectory_visualization(
        trajectory_run_id=trajectory.manifest.run_id,
        trajectory=trajectory.summary,
        intervention_runs=((intervention.manifest.run_id, intervention.summary),),
        pair_id="capital",
    )

    assert "Where the paired prediction separates" in html
    assert "These curves locate decodable change; they are observational" in html
    assert "Where controlled patch effects become decodable" in html
    assert "matched-random means" in html
    assert "not conserved flow" in html
    assert 'role="img"' in html
    assert trajectory.manifest.run_id in html
    assert intervention.manifest.run_id in html


def test_renderer_rejects_unrelated_intervention_lineage(tmp_path) -> None:
    _service, trajectory, intervention = _fixture_runs(tmp_path)
    unrelated = intervention.summary.model_copy(
        update={"trajectory_run_id": "another-trajectory"}
    )

    try:
        render_trajectory_visualization(
            trajectory_run_id=trajectory.manifest.run_id,
            trajectory=trajectory.summary,
            intervention_runs=((intervention.manifest.run_id, unrelated),),
        )
    except ValueError as exc:
        assert "does not descend" in str(exc)
    else:
        raise AssertionError("unrelated intervention lineage was accepted")


def test_cli_verifies_runs_and_writes_self_contained_html(monkeypatch, tmp_path) -> None:
    service, trajectory, intervention = _fixture_runs(tmp_path)
    monkeypatch.setattr(cli, "_service", lambda _context: service)
    output = tmp_path / "figure" / "trajectory.html"

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "runs",
            "trajectory-visualize",
            trajectory.manifest.run_id,
            intervention.manifest.run_id,
            "--pair",
            "capital",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert receipt["schema_version"] == "probe.trajectory-visualization-receipt/v1"
    assert receipt["verified_runs"] == {
        trajectory.manifest.run_id: True,
        intervention.manifest.run_id: True,
    }
    assert len(receipt["sha256"]) == 64
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")

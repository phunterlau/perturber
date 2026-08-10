import json

import pytest

from probing.contracts import (
    InterventionSpec,
    QualificationSpec,
    ResearchWorkflowSpec,
)
from probing.reporting import build_research_report, write_research_report
from probing.workflow import run_workflow
from test_service import fake_rank_spec, make_service


def _qualification(parent: str, *, evaluator: dict | None = None) -> QualificationSpec:
    return QualificationSpec.model_validate(
        {
            "kind": "qualify",
            "name": "generated-gate",
            "parent_run_id": parent,
            "generation": {"max_new_tokens": 1, "seed": 19},
            "evaluator": evaluator or {"kind": "token_set"},
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 1_000_000,
                "seed": 19,
            },
        }
    )


def _intervention(parent: str, qualification: str | None = None) -> InterventionSpec:
    return InterventionSpec.model_validate(
        {
            "kind": "intervention",
            "name": "top-unit-ablation",
            "parent_run_id": parent,
            "qualification_run_id": qualification,
            "selection": {"strategy": "ranked_top_k", "top_k": 1},
            "operation": {"mode": "ablate", "condition": "perturbed"},
            "sweep": {"neuron_counts": [1], "strengths": [0.0]},
            "controls": {"samples": 1},
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 1_000_000,
                "seed": 23,
            },
        }
    )


def test_generated_qualification_is_an_enforced_causal_gate(tmp_path) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    qualification = service.execute(
        _qualification(
            rank.manifest.run_id,
            evaluator={
                "kind": "exact",
                "target_values": ["red"],
                "control_values": ["blue"],
            },
        )
    )
    assert qualification.summary.aggregate.invalid_pairs == 1

    spec = _intervention(
        rank.manifest.run_id,
        qualification.manifest.run_id,
    )
    with pytest.raises(ValueError, match="no requested pairs are eligible"):
        service.plan(spec)


def test_seeded_workflow_resolves_and_persists_lineage(tmp_path) -> None:
    service = make_service(tmp_path)
    workflow = ResearchWorkflowSpec(
        name="fixture-causal-loop",
        rank=fake_rank_spec(),
        qualification=_qualification("$rank"),
        interventions=(_intervention("$rank", "$qualification"),),
    )

    result = run_workflow(service=service, spec=workflow)

    assert [stage.kind for stage in result.stages] == [
        "rank",
        "qualify",
        "intervention",
    ]
    assert result.logical_forward_passes == 6
    assert result.stages[-1].claims[0].status == "exploratory"
    assert any(
        "Fewer than three matched-random" in warning
        for warning in result.stages[-1].warnings
    )
    intervention = service.repository.load_manifest(result.intervention_run_ids[0])
    assert intervention.parent_run_ids == (
        result.rank_run_id,
        result.qualification_run_id,
    )
    stored = service.workspace / "workflows" / result.workflow_id
    assert json.loads((stored / "driver.json").read_text())["qualification"][
        "parent_run_id"
    ] == "$rank"
    assert json.loads((stored / "outcome.json").read_text())["rank_run_id"] == (
        result.rank_run_id
    )


def test_research_report_is_conservative_and_hashes_written_files(tmp_path) -> None:
    service = make_service(tmp_path)
    outcome = service.execute(fake_rank_spec())
    report = build_research_report(
        run_id=outcome.manifest.run_id,
        manifest=outcome.manifest,
        summary=outcome.summary.model_dump(mode="json"),
    )

    assert report.headline.startswith("Exploratory observational")
    assert "causal" not in report.headline.lower()
    receipt = write_research_report(
        report=report,
        output_directory=tmp_path / "report",
    )

    assert len(receipt.json_sha256) == 64
    assert len(receipt.markdown_sha256) == 64
    markdown = (tmp_path / "report" / "report.md").read_text()
    assert "## Claims" in markdown
    assert "Leading ranked unit" in markdown

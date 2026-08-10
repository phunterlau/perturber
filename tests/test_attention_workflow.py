from dataclasses import replace
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from safetensors.torch import load_file
from typer.testing import CliRunner

from probing import cli
from probing.adapters.base import GeneratedSequence
from probing.contracts import (
    AttentionHeadInterventionSpec,
    AttentionHeadRankSpec,
    AttentionTraceSpec,
    QualificationSpec,
    RankSpec,
    ResearchWorkflowSpec,
)
from probing.engine import ProbeEngine
from probing.reporting import build_research_report
from probing.service import ResearchService
from probing.workflow import run_workflow
from helpers import FakeAttentionAdapter
from test_service import fake_rank_spec


runner = CliRunner()


class SparseObservableAttentionAdapter(FakeAttentionAdapter):
    """Produces a gap crossing while argmax lies outside the sparse token sets."""

    def forward_capture(self, input_ids, tokenized, capture_position):
        capture = super().forward_capture(input_ids, tokenized, capture_position)
        logits = capture.logits.clone()
        logits[2] = 4.0
        return replace(capture, logits=logits)

    def generate(
        self,
        input_ids,
        *,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        seed,
        edits=(),
    ) -> GeneratedSequence:
        perturbed = bool(int(input_ids[0, 0]))
        token_id = 0 if perturbed else 1
        return GeneratedSequence(
            text=self.tokenizer.decode([token_id]),
            token_ids=(token_id,),
        )


def _service(tmp_path: Path) -> tuple[ResearchService, FakeAttentionAdapter]:
    adapter = FakeAttentionAdapter()
    service = ResearchService(
        workspace=tmp_path / "workspace",
        cache_dir=tmp_path / "cache",
        engine_factory=lambda _spec: ProbeEngine(adapter),
    )
    service.models.ensure_available = lambda *_args, **_kwargs: None
    service.models.is_cached = lambda *_args, **_kwargs: True
    service.models.inspect_cached = lambda *_args, **_kwargs: {
        "model_types": ["qwen3"]
    }
    return service, adapter


def _attention_rank(parent_run_id: str) -> AttentionHeadRankSpec:
    return AttentionHeadRankSpec.model_validate(
        {
            "kind": "attention_rank",
            "parent_run_id": parent_run_id,
            "ranking": {"top_k": 4, "pair_aggregation": "single_pair"},
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 2_000_000,
                "seed": 11,
            },
        }
    )


def _attention_intervention(
    parent_run_id: str,
) -> AttentionHeadInterventionSpec:
    return AttentionHeadInterventionSpec.model_validate(
        {
            "kind": "attention_intervention",
            "parent_run_id": parent_run_id,
            "selection": {"strategy": "ranked_top_k", "top_k": 2},
            "operation": {"mode": "patch"},
            "sweep": {"head_counts": [2], "strengths": [1.0]},
            "controls": {"samples": 0},
            "execution": {
                "max_forward_passes": 1,
                "max_artifact_bytes": 2_000_000,
                "seed": 13,
            },
        }
    )


def test_attention_rank_and_head_patch_are_parent_linked_and_replayable(tmp_path) -> None:
    service, _adapter = _service(tmp_path)
    rank = service.execute(fake_rank_spec())
    rank_spec = _attention_rank(rank.manifest.run_id)

    assert service.plan(rank_spec).forward_passes == 2
    attention_rank = service.execute(rank_spec)

    assert attention_rank.manifest.run_kind == "attention_rank"
    assert attention_rank.manifest.parent_run_ids == (rank.manifest.run_id,)
    assert attention_rank.summary.total_head_count == 4
    assert (attention_rank.summary.heads[0].layer, attention_rank.summary.heads[0].head) == (
        0,
        0,
    )
    assert attention_rank.summary.heads[0].direct_effect_rms == pytest.approx(4.0)
    tensors = load_file(
        attention_rank.run_directory / "attention-tensors.safetensors"
    )
    assert tensors["head_output_original.pair_0.layer_0"].shape == (2, 1)
    assert service.repository.verify(attention_rank.manifest.run_id) == ()

    intervention_spec = _attention_intervention(attention_rank.manifest.run_id)
    assert service.plan(intervention_spec).forward_passes == 1
    intervention = service.execute(intervention_spec)

    observation = intervention.summary.observations[0]
    assert intervention.manifest.run_kind == "attention_intervention"
    assert observation.condition == "original"
    assert observation.baseline_gap == pytest.approx(-2.0)
    assert observation.source_gap == pytest.approx(3.0)
    assert observation.intervention_gap != observation.baseline_gap
    assert intervention.summary.logical_forward_passes == 1
    assert service.repository.verify(intervention.manifest.run_id) == ()


def test_attention_rank_replay_has_identical_scientific_artifacts(tmp_path) -> None:
    service, _adapter = _service(tmp_path)
    rank = service.execute(fake_rank_spec())
    spec = _attention_rank(rank.manifest.run_id)

    first = service.execute(spec)
    second = service.execute(spec)

    assert first.manifest.run_id != second.manifest.run_id
    assert first.summary == second.summary
    stable_paths = {
        "summary.json",
        "attention-pairs.jsonl",
        "attention-layers.csv",
        "attention-heads.csv",
        "attention-tensors.safetensors",
    }
    first_hashes = {
        item.path: item.sha256
        for item in first.manifest.artifacts
        if item.path in stable_paths
    }
    second_hashes = {
        item.path: item.sha256
        for item in second.manifest.artifacts
        if item.path in stable_paths
    }
    assert first_hashes == second_hashes


def test_attention_cli_returns_agent_handoff_receipt(monkeypatch, tmp_path) -> None:
    service, _adapter = _service(tmp_path)
    rank = service.execute(fake_rank_spec())
    spec = _attention_rank(rank.manifest.run_id)
    path = tmp_path / "attention-rank.json"
    path.write_text(json.dumps(spec.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setattr(cli, "_service", lambda _context: service)

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "attention",
            "rank",
            "--spec",
            str(path),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert receipt["schema_version"] == "probe.execution-receipt/v1"
    assert receipt["run_kind"] == "attention_rank"
    assert receipt["logical_forward_passes"] == 2
    assert receipt["result"]["schema_version"] == "probe.attention-rank-result/v1"
    assert service.repository.load_manifest(receipt["run_id"]).run_kind == (
        "attention_rank"
    )


def test_token_edges_reconstruct_and_two_stage_path_patch_has_exact_budget(tmp_path) -> None:
    service, _adapter = _service(tmp_path)
    rank = service.execute(fake_rank_spec())
    attention_rank = service.execute(_attention_rank(rank.manifest.run_id))
    intervention = service.execute(
        _attention_intervention(attention_rank.manifest.run_id)
    )

    token_spec = AttentionTraceSpec.model_validate(
        {
            "kind": "attention_trace",
            "trace_kind": "token_edges",
            "parent_run_id": attention_rank.manifest.run_id,
            "heads": [{"layer": 0, "head": 0}, {"layer": 1, "head": 0}],
            "max_token_edges": 10,
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 2_000_000,
            },
        }
    )
    assert service.plan(token_spec).forward_passes == 2
    token_trace = service.execute(token_spec)
    assert token_trace.summary.logical_forward_passes == 2
    assert len(token_trace.summary.token_edges) == 4
    assert token_trace.summary.evidence_stage == "attention_hypothesis"

    path_spec = AttentionTraceSpec.model_validate(
        {
            "kind": "attention_trace",
            "trace_kind": "head_paths",
            "parent_run_id": attention_rank.manifest.run_id,
            "parent_intervention_run_id": intervention.manifest.run_id,
            "senders": [{"layer": 0, "head": 0}],
            "receivers": [{"layer": 1, "head": 0}],
            "alignments": [{"pair_id": "capital", "mode": "identity"}],
            "controls": {"samples": 1},
            "execution": {
                "max_forward_passes": 5,
                "max_artifact_bytes": 2_000_000,
                "seed": 17,
            },
        }
    )
    plan = service.plan(path_spec)
    assert plan.forward_passes == 5
    path_trace = service.execute(path_spec)

    assert path_trace.summary.logical_forward_passes == 5
    assert len(path_trace.summary.paths) == 2
    selected = next(
        item for item in path_trace.summary.paths if item.arm == "selected_path"
    )
    control = next(
        item
        for item in path_trace.summary.paths
        if item.arm == "matched_random_path"
    )
    selected_population = {
        (item.layer, item.head) for item in intervention.summary.selected_heads
    }
    assert (control.sender.layer, control.sender.head) not in selected_population
    assert (control.receiver.layer, control.receiver.head) not in selected_population
    assert selected.path_specific_effect == pytest.approx(1.5)
    assert abs(selected.path_specific_effect) > abs(control.path_specific_effect)
    assert path_trace.summary.claims[0].status == "supported"
    assert service.repository.verify(path_trace.manifest.run_id) == ()

    heads_query = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "attention",
            "heads",
            attention_rank.manifest.run_id,
            "--top",
            "1",
        ],
    )
    assert heads_query.exit_code == 0, heads_query.output
    heads_payload = json.loads(heads_query.stdout)
    assert heads_payload["schema_version"] == "probe.attention-head-query/v1"
    assert heads_payload["heads"][0]["layer"] == 0

    paths_query = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "attention",
            "paths",
            path_trace.manifest.run_id,
            "--limit",
            "1",
        ],
    )
    assert paths_query.exit_code == 0, paths_query.output
    paths_payload = json.loads(paths_query.stdout)
    assert paths_payload["schema_version"] == "probe.attention-path-query/v1"
    assert paths_payload["paths"][0]["arm"] == "selected_path"


def test_symbolic_workflow_runs_attention_rank_heads_tokens_and_paths(tmp_path) -> None:
    service, _adapter = _service(tmp_path)
    workflow = ResearchWorkflowSpec.model_validate(
        {
            "name": "fixture-attention-causal-loop",
            "rank": fake_rank_spec().model_dump(mode="json"),
            "attention_rank": {
                **_attention_rank("$rank").model_dump(mode="json"),
                "parent_run_id": "$rank",
            },
            "attention_interventions": [
                {
                    **_attention_intervention("$attention_rank").model_dump(
                        mode="json"
                    ),
                    "parent_run_id": "$attention_rank",
                }
            ],
            "attention_traces": [
                {
                    "kind": "attention_trace",
                    "name": "token-sources",
                    "trace_kind": "token_edges",
                    "parent_run_id": "$attention_rank",
                    "heads": [{"layer": 0, "head": 0}],
                    "execution": {
                        "max_forward_passes": 2,
                        "max_artifact_bytes": 2_000_000,
                        "seed": 17,
                    },
                },
                {
                    "kind": "attention_trace",
                    "name": "two-stage-path",
                    "trace_kind": "head_paths",
                    "parent_run_id": "$attention_rank",
                    "parent_intervention_run_id": "$attention_intervention",
                    "senders": [{"layer": 0, "head": 0}],
                    "receivers": [{"layer": 1, "head": 0}],
                    "alignments": [{"pair_id": "capital", "mode": "identity"}],
                    "controls": {"samples": 1},
                    "execution": {
                        "max_forward_passes": 5,
                        "max_artifact_bytes": 2_000_000,
                        "seed": 19,
                    },
                },
            ],
        }
    )

    outcome = run_workflow(service=service, spec=workflow)

    assert [stage.kind for stage in outcome.stages] == [
        "rank",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
        "attention_trace",
    ]
    assert outcome.logical_forward_passes == 12
    assert outcome.attention_rank_run_id is not None
    assert len(outcome.attention_intervention_run_ids) == 1
    assert len(outcome.attention_trace_run_ids) == 2
    path_manifest = service.repository.load_manifest(
        outcome.attention_trace_run_ids[-1]
    )
    assert path_manifest.parent_run_ids == (
        outcome.attention_rank_run_id,
        outcome.attention_intervention_run_ids[0],
    )
    reports = [
        build_research_report(
            run_id=run_id,
            manifest=service.repository.load_manifest(run_id),
            summary=service.repository.load_summary(run_id),
        )
        for run_id in (
            outcome.attention_rank_run_id,
            outcome.attention_intervention_run_ids[0],
            outcome.attention_trace_run_ids[-1],
        )
    ]
    assert reports[0].headline == "Observational attention-head routing hypotheses"
    assert reports[1].headline == "Controlled attention-head intervention evidence"
    assert reports[2].headline == "Supported local attention path-patching evidence"


def test_included_weak_pair_cannot_produce_supported_attention_claim(tmp_path) -> None:
    service, _adapter = _service(tmp_path)
    source = fake_rank_spec().model_dump(mode="json")
    source["observable"] = {
        "name": "weak-no-vs-maybe",
        "target_tokens": ["No"],
        "control_tokens": ["Maybe"],
    }
    parent = service.execute(RankSpec.model_validate(source))
    assert parent.summary.pairs[0].qualification is not None
    assert parent.summary.pairs[0].qualification.status == "weak"
    attention_rank = service.execute(
        _attention_rank(parent.manifest.run_id).model_copy(
            update={"include_weak_pairs": True}
        )
    )
    intervention = AttentionHeadInterventionSpec.model_validate(
        {
            "kind": "attention_intervention",
            "parent_run_id": attention_rank.manifest.run_id,
            "include_weak_pairs": True,
            "selection": {"strategy": "ranked_top_k", "top_k": 1},
            "operation": {"mode": "patch"},
            "sweep": {"head_counts": [1], "strengths": [1.0]},
            "controls": {"samples": 3},
            "execution": {
                "max_forward_passes": 4,
                "max_artifact_bytes": 2_000_000,
                "seed": 29,
            },
        }
    )

    outcome = service.execute(intervention)

    assert outcome.summary.claims[0].status != "supported"
    assert outcome.summary.claims[0].status == "exploratory"
    assert any("informative-observable gate" in item for item in outcome.summary.warnings)


def test_trace_inherits_generated_qualification_from_attention_lineage(tmp_path) -> None:
    adapter = SparseObservableAttentionAdapter()
    service = ResearchService(
        workspace=tmp_path / "workspace",
        cache_dir=tmp_path / "cache",
        engine_factory=lambda _spec: ProbeEngine(adapter),
    )
    service.models.ensure_available = lambda *_args, **_kwargs: None
    service.models.is_cached = lambda *_args, **_kwargs: True
    service.models.inspect_cached = lambda *_args, **_kwargs: {
        "model_types": ["qwen3"]
    }
    parent = service.execute(fake_rank_spec())
    assert parent.summary.pairs[0].qualification is not None
    assert parent.summary.pairs[0].qualification.status == "weak"
    qualification = service.execute(
        QualificationSpec.model_validate(
            {
                "kind": "qualify",
                "parent_run_id": parent.manifest.run_id,
                "generation": {"max_new_tokens": 1, "seed": 31},
                "evaluator": {
                    "kind": "exact",
                    "target_values": ["No"],
                    "control_values": ["Yes"],
                },
                "execution": {
                    "max_forward_passes": 2,
                    "max_artifact_bytes": 2_000_000,
                    "seed": 31,
                },
            }
        )
    )
    assert qualification.summary.aggregate.informative_pairs == 1
    attention_rank_spec = _attention_rank(parent.manifest.run_id).model_copy(
        update={"qualification_run_id": qualification.manifest.run_id}
    )
    attention_rank = service.execute(attention_rank_spec)
    intervention_spec = _attention_intervention(
        attention_rank.manifest.run_id
    ).model_copy(update={"qualification_run_id": qualification.manifest.run_id})
    intervention = service.execute(intervention_spec)
    trace = AttentionTraceSpec.model_validate(
        {
            "kind": "attention_trace",
            "trace_kind": "head_paths",
            "parent_run_id": attention_rank.manifest.run_id,
            "parent_intervention_run_id": intervention.manifest.run_id,
            "pair_ids": ["capital"],
            "senders": [{"layer": 0, "head": 0}],
            "receivers": [{"layer": 1, "head": 0}],
            "alignments": [{"pair_id": "capital", "mode": "identity"}],
            "controls": {"samples": 1},
            "execution": {
                "max_forward_passes": 5,
                "max_artifact_bytes": 2_000_000,
                "seed": 37,
            },
        }
    )

    assert service.plan(trace).pair_count == 1
    outcome = service.execute(trace)

    assert outcome.summary.claims[0].status == "supported"
    assert not any("informative-observable gate" in item for item in outcome.summary.warnings)


def test_path_contract_and_runtime_reject_inexact_or_untested_routes(tmp_path) -> None:
    base = {
        "kind": "attention_trace",
        "trace_kind": "head_paths",
        "parent_run_id": "attention-rank",
        "parent_intervention_run_id": "attention-intervention",
        "senders": [{"layer": 1, "head": 0}],
        "receivers": [{"layer": 0, "head": 0}],
        "alignments": [{"pair_id": "capital", "mode": "identity"}],
        "execution": {
            "max_forward_passes": 5,
            "max_artifact_bytes": 2_000_000,
        },
    }
    with pytest.raises(ValidationError, match="sender layer must precede"):
        AttentionTraceSpec.model_validate(base)

    service, _adapter = _service(tmp_path)
    rank = service.execute(fake_rank_spec())
    attention_rank = service.execute(_attention_rank(rank.manifest.run_id))
    intervention = service.execute(
        _attention_intervention(attention_rank.manifest.run_id)
    )
    invalid = AttentionTraceSpec.model_validate(
        {
            **base,
            "parent_run_id": attention_rank.manifest.run_id,
            "parent_intervention_run_id": intervention.manifest.run_id,
            "senders": [{"layer": 0, "head": 1}],
            "receivers": [{"layer": 1, "head": 0}],
            "controls": {"samples": 0},
            "execution": {
                "max_forward_passes": 3,
                "max_artifact_bytes": 2_000_000,
            },
        }
    )
    with pytest.raises(ValueError, match="untested"):
        service.execute(invalid)

    insufficient_unique_controls = AttentionTraceSpec.model_validate(
        {
            **base,
            "parent_run_id": attention_rank.manifest.run_id,
            "parent_intervention_run_id": intervention.manifest.run_id,
            "senders": [{"layer": 0, "head": 0}],
            "receivers": [{"layer": 1, "head": 0}],
            "controls": {"samples": 2},
            "execution": {
                "max_forward_passes": 7,
                "max_artifact_bytes": 2_000_000,
            },
        }
    )
    with pytest.raises(ValueError, match="too few unique same-layer paths"):
        service.plan(insufficient_unique_controls)

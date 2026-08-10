from __future__ import annotations

import io
import json
import time
import zipfile

from fastapi.testclient import TestClient
import pytest

from probing.cases import ResearchCaseRepository, build_research_packet
from probing.contracts import (
    ResearchCaseCreate,
    ResearchCaseUpdate,
    ResearchIntent,
    ResearchWorkflowSpec,
)
from probing.server import create_app
from probing.workflow import resolve_workflow_stage
from test_service import fake_rank_spec, make_service


def _intent() -> ResearchIntent:
    return ResearchIntent(
        hypothesis="A controlled prompt change moves the declared observable.",
        intended_perturbation="Replace only the factual premise.",
        invariants=("answer protocol",),
        falsifying_outcome="The observable does not move.",
    )


def _rank_workflow() -> ResearchWorkflowSpec:
    return ResearchWorkflowSpec(name="fixture-case", rank=fake_rank_spec())


def test_case_roundtrip_and_executed_stage_is_immutable(tmp_path) -> None:
    repository = ResearchCaseRepository(tmp_path / "workspace")
    created = repository.create(
        ResearchCaseCreate(intent=_intent(), workflow=_rank_workflow())
    )

    assert created.schema_version == "probe.research-case/v1"
    assert created.stages[0].key == "rank"
    assert repository.load(created.case_id) == created

    repository.update_stage(
        created.case_id,
        "rank",
        status="verified",
        run_id="immutable-rank",
    )
    changed = _rank_workflow().model_copy(
        update={"rank": fake_rank_spec().model_copy(update={"name": "changed"})}
    )
    with pytest.raises(Exception, match="cannot be changed"):
        repository.update(
            created.case_id,
            ResearchCaseUpdate(
                revision=2,
                intent=_intent(),
                workflow=changed,
            ),
        )


def test_case_api_runs_checkpoint_verifies_and_packages_handoff(tmp_path) -> None:
    service = make_service(tmp_path)
    app = create_app(
        workspace=service.workspace,
        cache_dir=tmp_path / "cache",
        token="secret",
        research_service=service,
    )
    headers = {"Authorization": "Bearer secret"}

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/cases",
            headers=headers,
            json=ResearchCaseCreate(
                intent=_intent(), workflow=_rank_workflow()
            ).model_dump(mode="json"),
        )
        assert created.status_code == 200, created.text
        case_id = created.json()["case_id"]

        planned = client.get(f"/api/v1/cases/{case_id}/plan", headers=headers)
        assert planned.status_code == 200
        assert planned.json()["stages"][0]["plan"]["forward_passes"] == 2

        preflight = client.post(
            f"/api/v1/cases/{case_id}/stages/rank/preflight", headers=headers
        )
        assert preflight.status_code == 200
        assert preflight.json()["executable"] is True

        started = client.post(
            f"/api/v1/cases/{case_id}/stages/rank/start", headers=headers
        )
        assert started.status_code == 200, started.text
        job_id = started.json()["job_id"]
        for _ in range(100):
            status = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if status["state"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert status["state"] == "completed"

        case = client.get(f"/api/v1/cases/{case_id}", headers=headers).json()
        assert case["stages"][0]["status"] == "verified"
        assert case["stages"][0]["run_id"] == status["run_id"]
        assert case["evidence_label"] == "observational"

        handoff = client.get(f"/api/v1/cases/{case_id}/handoff", headers=headers)
        assert handoff.status_code == 200
        assert status["run_id"] in handoff.json()["prompt"]

        packet = client.get(f"/api/v1/cases/{case_id}/packet", headers=headers)
        assert packet.status_code == 200, packet.text
        with zipfile.ZipFile(io.BytesIO(packet.content)) as archive:
            assert {
                "workflow.yaml",
                "case.json",
                "runs.json",
                "claims.json",
                "verification.json",
                "agent-context.json",
                "reports.json",
                "COMMANDS.md",
            } <= set(archive.namelist())
            context = json.loads(archive.read("agent-context.json"))
            assert context["run_ids"]["rank"] == status["run_id"]


def test_failed_qualification_gate_blocks_symbolic_causal_stage(tmp_path) -> None:
    service = make_service(tmp_path)
    workflow = ResearchWorkflowSpec.model_validate(
        {
            "name": "blocked-gate",
            "rank": fake_rank_spec().model_dump(mode="json"),
            "qualification": {
                "kind": "qualify",
                "name": "invalid-language-gate",
                "parent_run_id": "$rank",
                "generation": {"max_new_tokens": 1, "seed": 7},
                "evaluator": {
                    "kind": "unicode_script",
                    "target_values": ["han"],
                    "control_values": ["latin"],
                },
                "execution": {
                    "max_forward_passes": 2,
                    "max_artifact_bytes": 1_000_000,
                    "seed": 7,
                },
            },
            "attention_rank": {
                "kind": "attention_rank",
                "name": "blocked-head-rank",
                "parent_run_id": "$rank",
                "qualification_run_id": "$qualification",
                "ranking": {"top_k": 2},
                "execution": {
                    "max_forward_passes": 2,
                    "max_artifact_bytes": 1_000_000,
                    "seed": 7,
                },
            },
        }
    )
    repository = ResearchCaseRepository(service.workspace)
    case = repository.create(ResearchCaseCreate(intent=_intent(), workflow=workflow))

    rank = service.execute(resolve_workflow_stage(workflow, "rank", {}))
    repository.update_stage(case.case_id, "rank", status="verified", run_id=rank.manifest.run_id)
    qualification_spec = resolve_workflow_stage(
        workflow, "qualification", {"rank": rank.manifest.run_id}
    )
    qualification = service.execute(qualification_spec)
    repository.update_stage(
        case.case_id,
        "qualification",
        status="verified",
        run_id=qualification.manifest.run_id,
    )

    refreshed = repository.refresh(case.case_id, service)
    attention = next(item for item in refreshed.stages if item.key == "attention-rank")
    assert qualification.summary.aggregate.claim_eligible is False
    assert attention.status == "gate_failed"

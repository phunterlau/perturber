import json

from typer.testing import CliRunner

from probing import cli
from probing.client import ProbeClient
from probing.contracts import ErrorDetail
from probing.errors import RemoteProbeError
from test_service import make_service


runner = CliRunner()


def test_jsonl_run_keeps_machine_events_on_stdout(monkeypatch, tmp_path) -> None:
    service = make_service(tmp_path)
    monkeypatch.setattr(cli, "_service", lambda _context: service)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                **cli.example_rank_spec().model_dump(mode="json"),
                "model": {
                    **cli.example_rank_spec().model_dump(mode="json")["model"],
                    "id": "fake/qwen3",
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "run",
            "--spec",
            str(spec),
            "--events",
            "jsonl",
        ],
    )

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0]["schema_version"] == "probe.event/v1"
    assert events[-1]["event"] == "job.completed"
    assert events[-1]["payload"]["evidence_stage"] == "exploratory_pair"
    assert "summary" not in events[-1]["payload"]


def test_invalid_spec_is_a_versioned_machine_error(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("kind: future\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["validate", "--spec", str(path)])

    assert result.exit_code == 2
    error = json.loads(result.stdout)
    assert error["schema_version"] == "probe.error/v1"
    assert error["error"]["code"] == "invalid_spec"


def test_remote_run_failure_keeps_semantic_exit_code(monkeypatch, tmp_path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(cli.example_rank_spec().model_dump(mode="json")))

    class FailingClient:
        def run(self, *_args, **_kwargs):
            raise RemoteProbeError(
                ErrorDetail(code="model_policy_error", message="snapshot unavailable")
            )

    monkeypatch.setattr(
        ProbeClient,
        "from_context",
        classmethod(lambda _cls, _context: FailingClient()),
    )

    result = runner.invoke(
        cli.app,
        [
            "--endpoint",
            "http://127.0.0.1:9999",
            "run",
            "--spec",
            str(spec),
            "--events",
            "jsonl",
        ],
    )

    assert result.exit_code == 4
    error = json.loads(result.stdout)
    assert error["schema_version"] == "probe.error/v1"
    assert error["error"]["code"] == "model_policy_error"


def test_runs_files_serializes_typed_artifact_container(tmp_path) -> None:
    service = make_service(tmp_path)
    outcome = service.execute(
        cli.RankSpec.model_validate(
            {
                **cli.example_rank_spec().model_dump(mode="json"),
                "model": {
                    **cli.example_rank_spec().model_dump(mode="json")["model"],
                    "id": "fake/qwen3",
                },
            }
        )
    )

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "runs",
            "files",
            outcome.manifest.run_id,
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["schema_version"] == "probe.query/v1"
    assert response["query"] == "files"
    assert response["returned_count"] == 7
    assert {item["path"] for item in response["items"]} == {
        "events.jsonl",
        "layers.csv",
        "neurons.csv",
        "pairs.jsonl",
        "spec.json",
        "summary.json",
        "tensors.safetensors",
    }


def test_agent_workflow_and_report_commands_return_bounded_receipts(
    monkeypatch, tmp_path
) -> None:
    service = make_service(tmp_path)
    monkeypatch.setattr(cli, "_service", lambda _context: service)
    rank = {
        **cli.example_rank_spec().model_dump(mode="json"),
        "model": {
            **cli.example_rank_spec().model_dump(mode="json")["model"],
            "id": "fake/qwen3",
        },
    }
    driver = tmp_path / "workflow.json"
    driver.write_text(
        json.dumps(
            {
                "schema_version": "probe.workflow/v1",
                "name": "cli-fixture",
                "rank": rank,
                "qualification": {
                    "kind": "qualify",
                    "parent_run_id": "$rank",
                    "generation": {"max_new_tokens": 1, "seed": 7},
                    "evaluator": {"kind": "token_set"},
                    "execution": {
                        "max_forward_passes": 2,
                        "max_artifact_bytes": 1_000_000,
                        "seed": 7,
                    },
                },
                "interventions": [
                    {
                        "kind": "intervention",
                        "parent_run_id": "$rank",
                        "qualification_run_id": "$qualification",
                        "selection": {"strategy": "ranked_top_k", "top_k": 1},
                        "operation": {"mode": "ablate", "condition": "perturbed"},
                        "sweep": {"neuron_counts": [1], "strengths": [0.0]},
                        "controls": {"samples": 1},
                        "execution": {
                            "max_forward_passes": 2,
                            "max_artifact_bytes": 1_000_000,
                            "seed": 8,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "workflow",
            "--driver",
            str(driver),
            "--events",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    outcome = json.loads(result.stdout)
    assert outcome["schema_version"] == "probe.workflow-outcome/v1"
    assert [stage["kind"] for stage in outcome["stages"]] == [
        "rank",
        "qualify",
        "intervention",
    ]
    report_directory = tmp_path / "report"
    reported = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "report",
            outcome["intervention_run_ids"][0],
            "--output",
            str(report_directory),
        ],
    )
    assert reported.exit_code == 0, reported.output
    receipt = json.loads(reported.stdout)
    assert receipt["schema_version"] == "probe.report-receipt/v1"
    assert (report_directory / "report.json").is_file()
    assert (report_directory / "report.md").is_file()

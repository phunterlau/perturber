import json

from typer.testing import CliRunner

from probing import cli
from probing.contracts import FFNCouplingSpec, QualificationSpec, TrajectorySpec
from test_service import fake_rank_spec, make_service


runner = CliRunner()


def test_agent_compact_run_and_versioned_evidence_queries(monkeypatch, tmp_path) -> None:
    service = make_service(tmp_path)
    monkeypatch.setattr(cli, "_service", lambda _context: service)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(fake_rank_spec().model_dump(mode="json")), encoding="utf-8"
    )

    preflight_result = runner.invoke(
        cli.app,
        ["preflight", "--spec", str(spec_path)],
    )
    assert preflight_result.exit_code == 0, preflight_result.output
    preflight = json.loads(preflight_result.stdout)
    assert preflight["schema_version"] == "probe.preflight/v1"
    assert preflight["executable"] is True
    assert preflight["capabilities"]["dtype"] == "float32"

    run_result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "run",
            "--spec",
            str(spec_path),
            "--events",
            "none",
            "--result",
            "compact-json",
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    overview = json.loads(run_result.stdout)
    assert overview["schema_version"] == "probe.run-overview/v1"
    assert overview["logical_forward_passes"] == 2
    assert overview["top_neurons"][0]["observable_effect"] == "toward_target"
    run_id = overview["run_id"]

    commands = {
        "overview": ["runs", "overview", run_id],
        "manifest": ["runs", "manifest", run_id],
        "layers": ["runs", "layers", run_id, "--top", "1"],
        "neurons": [
            "runs",
            "neurons",
            run_id,
            "--top",
            "1",
            "--sign",
            "positive",
        ],
        "neurons_shared": [
            "runs",
            "neurons",
            run_id,
            "--top",
            "1",
            "--ranking-objective",
            "shared_direction",
        ],
        "neurons_magnitude": [
            "runs",
            "neurons",
            run_id,
            "--top",
            "1",
            "--ranking-objective",
            "effect_magnitude",
        ],
        "files": ["runs", "files", run_id],
        "verify": ["runs", "verify", run_id],
    }
    responses = {}
    for name, command in commands.items():
        result = runner.invoke(
            cli.app, ["--workspace", str(service.workspace), *command]
        )
        assert result.exit_code == 0, f"{name}: {result.output}"
        responses[name] = json.loads(result.stdout)

    assert responses["overview"]["run_id"] == run_id
    assert responses["manifest"]["schema_version"] == "probe.run/v1"
    assert responses["layers"]["schema_version"] == "probe.query/v1"
    assert responses["layers"]["sort"] == "rms_mass:desc,layer:asc"
    assert responses["neurons"]["parameters"]["sign"] == "positive"
    assert responses["neurons"]["items"][0]["observable_effect"] == "toward_target"
    assert (
        responses["neurons_shared"]["sort"]
        == "absolute_importance_mean:desc,rank:asc"
    )
    assert (
        responses["neurons_magnitude"]["sort"]
        == "importance_rms:desc,rank:asc"
    )
    assert responses["files"]["returned_count"] == 7
    assert responses["verify"]["schema_version"] == "probe.verification/v1"
    assert responses["verify"]["valid"] is True

    job_id = next(service.repository.jobs.iterdir()).name
    job_spec = runner.invoke(
        cli.app,
        ["--workspace", str(service.workspace), "jobs", "spec", job_id],
    )
    assert job_spec.exit_code == 0
    assert json.loads(job_spec.stdout)["pairs"][0]["id"] == "capital"


def test_agent_queries_trajectory_transitions_and_layer_aware_couplings(
    monkeypatch, tmp_path
) -> None:
    service = make_service(tmp_path)
    rank = service.execute(fake_rank_spec())
    trajectory = service.execute(
        TrajectorySpec.model_validate(
            {
                "kind": "trajectory",
                "parent_run_id": rank.manifest.run_id,
                "transition_limit": 3,
                "execution": {
                    "max_forward_passes": 2,
                    "max_artifact_bytes": 1_000_000,
                    "seed": 7,
                },
            }
        )
    )
    coupling = service.execute(
        FFNCouplingSpec.model_validate(
            {
                "kind": "ffn_coupling",
                "parent_run_id": rank.manifest.run_id,
                "trajectory_run_id": trajectory.manifest.run_id,
                "top_k": 10,
                "max_backward_passes": 2,
                "execution": {
                    "max_forward_passes": 2,
                    "max_artifact_bytes": 1_000_000,
                    "seed": 7,
                },
            }
        )
    )
    monkeypatch.setattr(cli, "_service", lambda _context: service)

    commands = {
        "trajectory": [
            "runs",
            "trajectory",
            trajectory.manifest.run_id,
            "--pair",
            "capital",
            "--metric",
            "target_rank",
            "--checkpoint",
            "post_ffn",
            "--limit",
            "2",
        ],
        "transitions": [
            "runs",
            "transitions",
            trajectory.manifest.run_id,
            "--split",
            "discovery",
            "--limit",
            "2",
        ],
        "couplings": [
            "runs",
            "ffn-couplings",
            coupling.manifest.run_id,
            "--method",
            "downstream",
            "--top",
            "2",
        ],
        "compare": [
            "runs",
            "coupling-compare",
            coupling.manifest.run_id,
            "--top",
            "2",
        ],
    }
    responses = {}
    for name, command in commands.items():
        result = runner.invoke(
            cli.app, ["--workspace", str(service.workspace), *command]
        )
        assert result.exit_code == 0, f"{name}: {result.output}"
        responses[name] = json.loads(result.stdout)

    assert responses["trajectory"]["query"] == "trajectory"
    assert responses["trajectory"]["parameters"]["lower_is_better"] is True
    assert responses["trajectory"]["returned_count"] == 1
    assert all(
        item["checkpoint"] == "post_ffn"
        for item in responses["trajectory"]["items"]
    )
    assert responses["transitions"]["query"] == "transitions"
    assert responses["transitions"]["matched_count"] >= 1
    assert responses["couplings"]["query"] == "ffn_couplings"
    assert responses["couplings"]["items"][0]["method"] == "downstream"
    assert responses["couplings"]["parameters"]["candidate_pair_ids"] == [
        "capital"
    ]
    assert responses["compare"]["query"] == "coupling_compare"
    assert "log10_downstream_to_direct" in responses["compare"]["items"][0]


def test_compact_result_requires_silent_events(tmp_path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(fake_rank_spec().model_dump(mode="json")), encoding="utf-8"
    )

    result = runner.invoke(
        cli.app,
        ["run", "--spec", str(spec_path), "--result", "compact-json"],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "invalid_spec"


def test_request_id_requires_daemon_endpoint(tmp_path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(fake_rank_spec().model_dump(mode="json")), encoding="utf-8"
    )

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--spec",
            str(spec_path),
            "--events",
            "none",
            "--result",
            "compact-json",
            "--request-id",
            "retry-key",
        ],
    )

    assert result.exit_code == 2
    assert "requires an explicit daemon endpoint" in result.stdout


def test_request_id_rejects_blank_remote_key(tmp_path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(fake_rank_spec().model_dump(mode="json")), encoding="utf-8"
    )

    result = runner.invoke(
        cli.app,
        [
            "--endpoint",
            "http://127.0.0.1:1",
            "run",
            "--spec",
            str(spec_path),
            "--events",
            "none",
            "--result",
            "compact-json",
            "--request-id",
            "   ",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "invalid_spec"


def test_agent_qualification_command_returns_execution_receipt(monkeypatch, tmp_path) -> None:
    service = make_service(tmp_path)
    parent = service.execute(fake_rank_spec())
    monkeypatch.setattr(cli, "_service", lambda _context: service)
    spec = QualificationSpec.model_validate(
        {
            "kind": "qualify",
            "parent_run_id": parent.manifest.run_id,
            "generation": {"max_new_tokens": 1},
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 1_000_000,
            },
        }
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(spec.model_dump(mode="json")), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "qualify",
            "--spec",
            str(path),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert receipt["schema_version"] == "probe.execution-receipt/v1"
    assert receipt["run_kind"] == "qualify"
    assert receipt["run_id"]
    assert receipt["logical_forward_passes"] == 2
    assert receipt["result"]["schema_version"] == "probe.qualification-result/v1"
    assert receipt["result"]["parent_run_id"] == parent.manifest.run_id
    assert receipt["result"]["aggregate"]["informative_pairs"] == 1

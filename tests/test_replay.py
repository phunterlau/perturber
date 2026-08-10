import json

import pytest
from typer.testing import CliRunner
import yaml

from probing import cli
from probing.contracts import RankSpec, ReplayDriver
from probing.errors import SpecError
from probing.replay import (
    compare_replay,
    load_replay_bundle,
    record_baseline,
    write_replay_report,
)
from test_service import fake_rank_spec, make_service


runner = CliRunner()


def _pinned_fake_spec() -> RankSpec:
    spec = fake_rank_spec(pairs=2)
    return spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={
                    "revision": "fixture",
                    "device": "cpu",
                    "dtype": "float32",
                }
            )
        }
    )


def _write_bundle(tmp_path, spec: RankSpec, *, artifact_hashes: str = "report"):
    root = tmp_path / "example"
    root.mkdir()
    (root / "spec.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    driver = {
        "schema_version": "probe.replay-driver/v1",
        "name": "deterministic-fixture",
        "spec": "spec.yaml",
        "baseline": "baseline.json",
        "report_directory": "reports",
        "reproducibility": {
            "torch_seed": 0,
            "resolved_model_revision": "fixture",
            "adapter": "fake",
            "device": "cpu",
            "model_dtype": "float32",
            "expected_environment": {},
        },
        "comparison": {
            "scalar_absolute_tolerance": 1e-7,
            "scalar_relative_tolerance": 1e-7,
            "ranking_top_n": 2,
            "minimum_top_n_overlap": 1,
            "minimum_sign_agreement": 1,
            "maximum_mean_rank_displacement": 0,
            "artifact_hashes": artifact_hashes,
        },
    }
    driver_path = root / "driver.yaml"
    driver_path.write_text(yaml.safe_dump(driver, sort_keys=False), encoding="utf-8")
    return load_replay_bundle(driver_path)


def test_replay_records_and_reproduces_fixture_run(tmp_path) -> None:
    service = make_service(tmp_path)
    bundle = _write_bundle(tmp_path, _pinned_fake_spec(), artifact_hashes="require")
    original = service.execute(bundle.spec)

    baseline = record_baseline(
        bundle,
        run_spec=service.repository.load_run_spec(original.manifest.run_id),
        manifest=original.manifest,
        summary=original.summary,
        integrity_failures=service.repository.verify(original.manifest.run_id),
    )
    replay = service.execute(bundle.spec)
    report = compare_replay(
        bundle,
        baseline,
        run_spec=service.repository.load_run_spec(replay.manifest.run_id),
        manifest=replay.manifest,
        summary=replay.summary,
        integrity_failures=service.repository.verify(replay.manifest.run_id),
    )
    report = write_replay_report(bundle, report)

    assert report.verdict == "passed"
    assert report.numeric["maximum_absolute_difference"] == 0
    assert report.ranking["overlap_fraction"] == 1
    assert report.artifact_hashes["all_matched"] is True
    assert bundle.baseline_path.is_file()
    for relative in report.report_files:
        assert (bundle.root / relative).is_file()


def test_replay_fails_when_numeric_observable_drifts(tmp_path) -> None:
    service = make_service(tmp_path)
    bundle = _write_bundle(tmp_path, _pinned_fake_spec())
    original = service.execute(bundle.spec)
    baseline = record_baseline(
        bundle,
        run_spec=bundle.spec,
        manifest=original.manifest,
        summary=original.summary,
    )
    changed = original.summary.model_dump(mode="json")
    changed["measured_delta_mean"] += 0.25

    report = compare_replay(
        bundle,
        baseline,
        run_spec=bundle.spec,
        manifest=original.manifest,
        summary=changed,
    )

    assert report.verdict == "failed"
    numeric = next(item for item in report.checks if item.name == "numeric_tolerances")
    assert numeric.required and not numeric.passed


def test_replay_rejects_driver_spec_disagreement(tmp_path) -> None:
    spec = _pinned_fake_spec()
    bundle = _write_bundle(tmp_path, spec)
    raw = yaml.safe_load(bundle.driver_path.read_text(encoding="utf-8"))
    raw["reproducibility"]["torch_seed"] = 99
    bundle.driver_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SpecError, match="execution.seed"):
        load_replay_bundle(bundle.driver_path)


def test_replay_paths_are_portable_and_cannot_escape_bundle() -> None:
    value = {
        "schema_version": "probe.replay-driver/v1",
        "name": "unsafe",
        "spec": "../spec.yaml",
        "reproducibility": {
            "torch_seed": 0,
            "resolved_model_revision": "fixture",
            "adapter": "fake",
            "device": "cpu",
            "model_dtype": "float32",
        },
    }

    with pytest.raises(ValueError, match="normalized relative POSIX"):
        ReplayDriver.model_validate(value)


def test_replay_baseline_is_immutable_without_explicit_overwrite(tmp_path) -> None:
    service = make_service(tmp_path)
    bundle = _write_bundle(tmp_path, _pinned_fake_spec())
    outcome = service.execute(bundle.spec)
    arguments = {
        "run_spec": bundle.spec,
        "manifest": outcome.manifest,
        "summary": outcome.summary,
    }
    record_baseline(bundle, **arguments)

    with pytest.raises(SpecError, match="--overwrite"):
        record_baseline(bundle, **arguments)

    replaced = record_baseline(bundle, **arguments, overwrite=True)
    stored = json.loads(bundle.baseline_path.read_text(encoding="utf-8"))
    assert stored["source_run_id"] == replaced.source_run_id


def test_agent_facing_replay_run_returns_one_report(monkeypatch, tmp_path) -> None:
    service = make_service(tmp_path)
    bundle = _write_bundle(tmp_path, _pinned_fake_spec())
    original = service.execute(bundle.spec)
    record_baseline(
        bundle,
        run_spec=bundle.spec,
        manifest=original.manifest,
        summary=original.summary,
    )
    monkeypatch.setattr(cli, "_service", lambda _context: service)

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "replay",
            "run",
            str(bundle.driver_path),
            "--events",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["schema_version"] == "probe.replay-outcome/v1"
    assert report["verdict"] == "passed"
    assert len(report["report_files"]) == 2


def test_agent_facing_replay_check_uses_exit_nine_on_mismatch(tmp_path) -> None:
    service = make_service(tmp_path)
    bundle = _write_bundle(tmp_path, _pinned_fake_spec())
    outcome = service.execute(bundle.spec)
    record_baseline(
        bundle,
        run_spec=bundle.spec,
        manifest=outcome.manifest,
        summary=outcome.summary,
    )
    summary_path = outcome.run_directory / "summary.json"
    changed = json.loads(summary_path.read_text(encoding="utf-8"))
    changed["measured_delta_mean"] += 0.25
    summary_path.write_text(json.dumps(changed), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--workspace",
            str(service.workspace),
            "replay",
            "check",
            str(bundle.driver_path),
            "--run-id",
            outcome.manifest.run_id,
        ],
    )

    assert result.exit_code == 9
    report = json.loads(result.stdout)
    assert report["schema_version"] == "probe.replay-outcome/v1"
    assert report["verdict"] == "failed"
    assert report["required_checks_passed"] < report["required_checks_total"]

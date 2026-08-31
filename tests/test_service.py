import threading
import time

import pytest
from safetensors.torch import load_file

from probing.contracts import RankSpec
from probing.engine import ProbeEngine
from probing.errors import ArtifactError, BudgetError, CapabilityError, JobCancelled
from probing.service import ResearchService
from probing.specs import example_rank_spec, example_replication_spec
from helpers import FakeAdapter


def fake_rank_spec(*, pairs: int = 1) -> RankSpec:
    source = example_rank_spec().model_dump(mode="json")
    source["model"]["id"] = "fake/qwen3"
    # Keep the fake service fixture host-independent. Production specs may use
    # auto selection, but this fixture's adapter explicitly reports CPU/float32
    # and its preflight assertions should not change on an MPS or CUDA host.
    source["model"]["device"] = "cpu"
    source["model"]["dtype"] = "float32"
    if pairs == 2:
        source["pairs"].append(
            {
                "id": "capital-replication",
                "original": "Paris",
                "perturbed": "London",
            }
        )
        source["ranking"]["pair_aggregation"] = "rms"
        source["execution"]["max_forward_passes"] = 4
    return RankSpec.model_validate(source)


def make_service(tmp_path) -> ResearchService:
    service = ResearchService(
        workspace=tmp_path / "workspace",
        cache_dir=tmp_path / "cache",
        engine_factory=lambda _spec: ProbeEngine(FakeAdapter()),
    )
    service.models.ensure_available = lambda *_args, **_kwargs: None
    service.models.is_cached = lambda *_args, **_kwargs: True
    service.models.inspect_cached = lambda *_args, **_kwargs: {
        "model_types": ["qwen3"]
    }
    return service


def test_service_runs_multi_pair_and_commits_verifiable_artifacts(tmp_path) -> None:
    service = make_service(tmp_path)
    spec = fake_rank_spec(pairs=2)
    events = []

    outcome = service.execute(spec, listener=events.append)

    assert outcome.summary.pair_count == 2
    assert outcome.summary.logical_forward_passes == 4
    assert outcome.summary.evidence_stage == "replicated_ranking"
    assert outcome.summary.neurons[0].importance_rms > 0
    assert outcome.summary.neurons[0].sign_consistency == 1
    assert service.repository.verify(outcome.manifest.run_id) == ()
    assert events[-1].event == "job.completed"
    assert service.repository.load_job(outcome.manifest.job_id).state == "completed"
    assert outcome.manifest.environment["python"]
    assert outcome.manifest.environment["torch"]
    assert outcome.manifest.environment["transformers"]
    assert outcome.manifest.environment["machine"]
    tensors = load_file(outcome.run_directory / "tensors.safetensors")
    assert "importance.pair_0.layer_0" in tensors
    assert "activation_original.pair_1.layer_0" in tensors
    assert tensors["importance.pair_0.layer_0"].shape == (2,)


def test_plan_rejects_insufficient_forward_budget(tmp_path) -> None:
    service = make_service(tmp_path)
    original = fake_rank_spec(pairs=2)
    spec = original.model_copy(
        update={
            "execution": original.execution.model_copy(
                update={"max_forward_passes": 2}
            )
        }
    )

    plan = service.plan(spec)

    assert plan.forward_passes == 4
    assert not plan.within_budget

    with pytest.raises(BudgetError):
        service.execute(spec)
    job_id = next(service.repository.jobs.iterdir()).name
    status = service.repository.load_job(job_id)
    assert status.state == "failed"
    assert status.error is not None and status.error.code == "budget_exceeded"


def test_preflight_combines_readiness_budget_and_resolved_dtype(tmp_path) -> None:
    service = make_service(tmp_path)

    report = service.preflight(fake_rank_spec())

    assert report.schema_version == "probe.preflight/v1"
    assert report.executable is True
    assert report.model_ready is True
    assert report.acquisition_required is False
    assert report.plan.forward_passes == 2
    assert report.capabilities.supported is True
    assert report.capabilities.device == "cpu"
    assert report.capabilities.dtype == "float32"


def test_preflight_distinguishes_cached_model_from_permitted_acquisition(tmp_path) -> None:
    service = make_service(tmp_path)
    service.models.is_cached = lambda *_args, **_kwargs: False
    value = fake_rank_spec().model_dump(mode="json")
    value["execution"]["allow_download"] = True
    value["execution"]["max_download_bytes"] = 2_000_000_000

    report = service.preflight(RankSpec.model_validate(value))

    assert report.executable is True
    assert report.model_ready is False
    assert report.acquisition_required is True


def test_artifact_budget_fails_without_committing_a_run(tmp_path) -> None:
    service = make_service(tmp_path)
    original = fake_rank_spec()
    spec = original.model_copy(
        update={
            "execution": original.execution.model_copy(
                update={"max_artifact_bytes": 1}
            )
        }
    )

    with pytest.raises(ArtifactError, match="exceeds max_artifact_bytes"):
        service.execute(spec)

    assert service.repository.list_runs() == ()
    jobs = list(service.repository.jobs.iterdir())
    assert len(jobs) == 1
    assert service.repository.load_job(jobs[0].name).state == "failed"
    assert not (jobs[0] / "staging").exists()


def test_capability_gate_rejects_an_unrecognized_auto_adapter(tmp_path) -> None:
    service = make_service(tmp_path)
    service.models.inspect_cached = lambda *_args, **_kwargs: {
        "model_types": ["llama"]
    }

    with pytest.raises(CapabilityError):
        service.execute(fake_rank_spec())

    assert service.repository.list_runs() == ()
    jobs = list(service.repository.jobs.iterdir())
    assert len(jobs) == 1
    status = service.repository.load_job(jobs[0].name)
    assert status.state == "failed"
    assert status.error is not None
    assert status.error.code == "capability_error"
    assert service.repository.read_events(status.job_id)[-1].event == "job.failed"


def test_paper_derived_examples_use_exactly_two_forwards_per_pair(tmp_path) -> None:
    adapter = FakeAdapter()
    service = ResearchService(
        workspace=tmp_path / "workspace",
        cache_dir=tmp_path / "cache",
        engine_factory=lambda _spec: ProbeEngine(adapter),
    )
    service.models.ensure_available = lambda *_args, **_kwargs: None
    service.models.inspect_cached = lambda *_args, **_kwargs: {"model_types": ["qwen3"]}
    service.models.resolve_cached_snapshot = lambda *_args, **_kwargs: tmp_path / "snapshot"
    value = example_replication_spec().model_dump(mode="json")
    value["model"]["id"] = "fake/qwen3"
    spec = RankSpec.model_validate(value)

    outcome = service.execute(spec)

    assert adapter.forward_calls == 6
    assert [item.pair_id for item in outcome.summary.pairs] == [
        "capital",
        "arithmetic",
        "science",
    ]
    assert all(item.measured_delta == pytest.approx(5.0) for item in outcome.summary.pairs)
    assert all(item.original_prediction == "Yes" for item in outcome.summary.pairs)
    assert all(item.perturbed_prediction == "No" for item in outcome.summary.pairs)
    assert outcome.summary.warnings[0].startswith(
        "Replicated ranking from 3 discovery prompt pairs remains observational"
    )
    assert not any(
        warning.startswith("Exploratory result from one prompt pair")
        for warning in outcome.summary.warnings
    )


def test_pre_cancelled_execution_is_durably_cancelled_without_a_forward(tmp_path) -> None:
    service = make_service(tmp_path)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(JobCancelled):
        service.execute(fake_rank_spec(), cancel=cancel)

    job_id = next(service.repository.jobs.iterdir()).name
    status = service.repository.load_job(job_id)
    assert status.state == "cancelled"
    assert status.error is not None and status.error.code == "job_cancelled"
    assert [event.event for event in service.repository.read_events(job_id)] == [
        "job.accepted",
        "job.cancelled",
    ]


def test_wall_deadline_is_a_failed_budget_not_a_user_cancellation(tmp_path) -> None:
    adapter = FakeAdapter()

    def slow_factory(_spec):
        time.sleep(0.02)
        return ProbeEngine(adapter)

    service = ResearchService(
        workspace=tmp_path / "workspace",
        cache_dir=tmp_path / "cache",
        engine_factory=slow_factory,
    )
    service.models.ensure_available = lambda *_args, **_kwargs: None
    service.models.inspect_cached = lambda *_args, **_kwargs: {"model_types": ["qwen3"]}
    original = fake_rank_spec()
    spec = original.model_copy(
        update={
            "execution": original.execution.model_copy(
                update={"max_wall_seconds": 0.001}
            )
        }
    )

    with pytest.raises(BudgetError, match="max_wall_seconds"):
        service.execute(spec)

    job_id = next(service.repository.jobs.iterdir()).name
    status = service.repository.load_job(job_id)
    assert status.state == "failed"
    assert status.error is not None and status.error.code == "budget_exceeded"
    assert adapter.forward_calls == 0


def test_unexpected_engine_error_is_durable_and_machine_readable(tmp_path) -> None:
    service = make_service(tmp_path)
    service.engine_factory = lambda _spec: (_ for _ in ()).throw(
        RuntimeError("synthetic model failure")
    )

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        service.execute(fake_rank_spec())

    job_id = next(service.repository.jobs.iterdir()).name
    status = service.repository.load_job(job_id)
    assert status.state == "failed"
    assert status.error is not None
    assert status.error.code == "runtime_error"
    assert status.error.details == {"exception_type": "RuntimeError"}
    assert service.repository.read_events(job_id)[-1].payload["error"]["code"] == "runtime_error"


def test_managed_engine_reports_load_then_reuse(monkeypatch, tmp_path) -> None:
    engine = ProbeEngine(FakeAdapter())
    monkeypatch.setattr(
        "probing.service.ProbeEngine.from_pretrained",
        lambda *_args, **_kwargs: engine,
    )
    service = ResearchService(
        workspace=tmp_path / "workspace", cache_dir=tmp_path / "cache"
    )
    service.models.ensure_available = lambda *_args, **_kwargs: None
    service.models.inspect_cached = lambda *_args, **_kwargs: {"model_types": ["qwen3"]}
    service.models.resolve_cached_snapshot = lambda *_args, **_kwargs: tmp_path / "snapshot"

    first_events = []
    second_events = []
    service.execute(fake_rank_spec(), listener=first_events.append)
    service.execute(fake_rank_spec(), listener=second_events.append)

    assert "model.loading" in [event.event for event in first_events]
    assert "model.reused" not in [event.event for event in first_events]
    assert "model.reused" in [event.event for event in second_events]
    assert "model.loading" not in [event.event for event in second_events]

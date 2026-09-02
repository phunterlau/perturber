import asyncio
import json
import threading
import time

import httpx

from probing.contracts import AttentionHeadRankSpec, QualificationSpec
from probing.engine import ProbeEngine
from probing.server import JobManager, create_app
from probing.service import ResearchService
from helpers import FakeAdapter, FakeAttentionAdapter
from test_service import fake_rank_spec, make_service


def test_authenticated_daemon_job_stream_matches_service(tmp_path) -> None:
    async def exercise() -> None:
        service = make_service(tmp_path)
        app = create_app(
            workspace=service.workspace,
            cache_dir=tmp_path / "cache",
            token="test-token",
            research_service=service,
        )
        headers = {"Authorization": "Bearer test-token"}
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                assert (await client.get("/api/v1/health")).status_code == 200
                unauthorized = await client.get("/api/v1/runs")
                assert unauthorized.status_code == 401
                assert unauthorized.json()["schema_version"] == "probe.error/v1"
                assert unauthorized.json()["error"]["code"] == "authentication_error"

                invalid = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json={"kind": "rank"},
                )
                assert invalid.status_code == 422
                assert invalid.json()["error"]["code"] == "invalid_spec"

                preflight = await client.post(
                    "/api/v1/preflight",
                    headers=headers,
                    json=fake_rank_spec().model_dump(mode="json"),
                )
                assert preflight.status_code == 200
                assert preflight.json()["schema_version"] == "probe.preflight/v1"
                assert preflight.json()["executable"] is True

                response = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json=fake_rank_spec().model_dump(mode="json"),
                )
                assert response.status_code == 200
                job_id = response.json()["job_id"]

                stream = await client.get(
                    f"/api/v1/jobs/{job_id}/events",
                    headers={**headers, "Accept": "application/x-ndjson"},
                )
                assert stream.status_code == 200
                events = [json.loads(line) for line in stream.text.splitlines()]
                assert events[-1]["event"] == "job.completed"

                status = (
                    await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
                ).json()
                assert status["state"] == "completed"
                summary = (
                    await client.get(
                        f"/api/v1/runs/{status['run_id']}/summary", headers=headers
                    )
                ).json()
                assert summary["measured_delta_mean"] == 5.0
                overview = (
                    await client.get(
                        f"/api/v1/runs/{status['run_id']}/overview", headers=headers
                    )
                ).json()
                assert overview["schema_version"] == "probe.run-overview/v1"
                assert overview["top_neurons"][0]["observable_effect"] == "toward_target"
                saved_spec = (
                    await client.get(f"/api/v1/jobs/{job_id}/spec", headers=headers)
                ).json()
                assert saved_spec["pairs"][0]["id"] == "capital"
                run_spec = (
                    await client.get(
                        f"/api/v1/runs/{status['run_id']}/spec", headers=headers
                    )
                ).json()
                assert run_spec == saved_spec
                verification = await client.get(
                    f"/api/v1/runs/{status['run_id']}/verify", headers=headers
                )
                assert verification.json() == {
                    "schema_version": "probe.verification/v1",
                    "run_id": status["run_id"],
                    "valid": True,
                    "failures": [],
                }

                qualification_spec = QualificationSpec.model_validate(
                    {
                        "kind": "qualify",
                        "parent_run_id": status["run_id"],
                        "generation": {"max_new_tokens": 1},
                        "execution": {
                            "max_forward_passes": 2,
                            "max_artifact_bytes": 1_000_000,
                        },
                    }
                )
                qualification_job = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json=qualification_spec.model_dump(mode="json"),
                )
                assert qualification_job.status_code == 200, qualification_job.text
                qualification_stream = await client.get(
                    f"/api/v1/jobs/{qualification_job.json()['job_id']}/events",
                    headers=headers,
                )
                qualification_events = [
                    json.loads(line) for line in qualification_stream.text.splitlines()
                ]
                qualification_run_id = qualification_events[-1]["payload"]["run_id"]
                qualification_summary = (
                    await client.get(
                        f"/api/v1/runs/{qualification_run_id}/summary",
                        headers=headers,
                    )
                ).json()
                assert qualification_summary["evidence_stage"] == "qualified_observable"
                assert qualification_summary["aggregate"]["informative_pairs"] == 1

                for path in (
                    "/api/v1/jobs/missing",
                    "/api/v1/jobs/missing/events",
                    "/api/v1/runs/missing",
                    "/api/v1/runs/missing/spec",
                    "/api/v1/runs/missing/summary",
                    "/api/v1/runs/missing/verify",
                ):
                    missing = await client.get(path, headers=headers)
                    assert missing.status_code == 404
                    assert missing.json()["schema_version"] == "probe.error/v1"
                    assert missing.json()["error"]["code"] == "artifact_error"

    asyncio.run(exercise())


def test_daemon_executes_attention_stage_through_shared_service(tmp_path) -> None:
    async def exercise() -> None:
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
        parent = service.execute(fake_rank_spec())
        spec = AttentionHeadRankSpec.model_validate(
            {
                "kind": "attention_rank",
                "parent_run_id": parent.manifest.run_id,
                "ranking": {"top_k": 4, "pair_aggregation": "single_pair"},
                "execution": {
                    "max_forward_passes": 2,
                    "max_artifact_bytes": 2_000_000,
                    "seed": 41,
                },
            }
        )
        app = create_app(
            workspace=service.workspace,
            cache_dir=tmp_path / "cache",
            token="test-token",
            research_service=service,
        )
        headers = {"Authorization": "Bearer test-token"}
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json=spec.model_dump(mode="json"),
                )
                assert response.status_code == 200, response.text
                stream = await client.get(
                    f"/api/v1/jobs/{response.json()['job_id']}/events",
                    headers=headers,
                )
                terminal = json.loads(stream.text.splitlines()[-1])
                run_id = terminal["payload"]["run_id"]
                summary = (
                    await client.get(
                        f"/api/v1/runs/{run_id}/summary", headers=headers
                    )
                ).json()
                assert summary["schema_version"] == "probe.attention-rank-result/v1"
                assert summary["parent_run_id"] == parent.manifest.run_id
                assert summary["heads"][0]["layer"] == 0

    asyncio.run(exercise())


def test_request_id_is_idempotent_and_conflicting_reuse_is_rejected(tmp_path) -> None:
    async def exercise() -> None:
        service = make_service(tmp_path)
        app = create_app(
            workspace=service.workspace,
            cache_dir=tmp_path / "cache",
            token="test-token",
            research_service=service,
        )
        headers = {
            "Authorization": "Bearer test-token",
            "X-Request-ID": "agent-attempt-1",
        }
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                blank = await client.post(
                    "/api/v1/jobs",
                    headers={
                        "Authorization": "Bearer test-token",
                        "X-Request-ID": "   ",
                    },
                    json=fake_rank_spec().model_dump(mode="json"),
                )
                assert blank.status_code == 422
                assert blank.json()["error"]["code"] == "invalid_spec"
                assert list(service.repository.jobs.iterdir()) == []

                first = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json=fake_rank_spec().model_dump(mode="json"),
                )
                assert first.status_code == 200
                stream = await client.get(
                    f"/api/v1/jobs/{first.json()['job_id']}/events", headers=headers
                )
                assert "job.completed" in stream.text

                # A retry must resolve to the durable job before current model
                # readiness is consulted. This matters after daemon restarts or
                # temporary cache/device changes.
                service.models.inspect_cached = lambda *_args, **_kwargs: {
                    "model_types": ["llama"]
                }
                second = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json=fake_rank_spec().model_dump(mode="json"),
                )
                assert second.status_code == 200
                assert first.json()["job_id"] == second.json()["job_id"]
                assert len(list(service.repository.jobs.iterdir())) == 1

                changed = fake_rank_spec().model_copy(update={"name": "different"})
                conflict = await client.post(
                    "/api/v1/jobs",
                    headers=headers,
                    json=changed.model_dump(mode="json"),
                )
                assert conflict.status_code == 409
                assert conflict.json()["error"]["code"] == "request_conflict"

    asyncio.run(exercise())


def test_daemon_rejects_unsupported_work_before_creating_a_job(tmp_path) -> None:
    async def exercise() -> None:
        service = make_service(tmp_path)
        service.models.inspect_cached = lambda *_args, **_kwargs: {
            "model_types": ["llama"]
        }
        app = create_app(
            workspace=service.workspace,
            cache_dir=tmp_path / "cache",
            token="test-token",
            research_service=service,
        )
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/jobs",
                    headers={"Authorization": "Bearer test-token"},
                    json=fake_rank_spec().model_dump(mode="json"),
                )
                assert response.status_code == 422
                assert response.json()["error"]["code"] == "capability_error"
                assert list(service.repository.jobs.iterdir()) == []

    asyncio.run(exercise())


def test_single_worker_queue_honors_cancellation_before_model_use(tmp_path) -> None:
    first_factory_started = threading.Event()
    release_first = threading.Event()
    factory_calls = 0
    factory_lock = threading.Lock()

    def blocking_factory(_spec):
        nonlocal factory_calls
        with factory_lock:
            call = factory_calls
            factory_calls += 1
        if call == 0:
            first_factory_started.set()
            assert release_first.wait(timeout=3)
        return ProbeEngine(FakeAdapter())

    service = make_service(tmp_path)
    service.engine_factory = blocking_factory
    manager = JobManager(service)
    try:
        first = manager.submit(fake_rank_spec())
        assert first_factory_started.wait(timeout=2)
        second = manager.submit(fake_rank_spec())
        manager.cancel(second.job_id)
        release_first.set()

        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            first_status = service.repository.load_job(first.job_id)
            second_status = service.repository.load_job(second.job_id)
            if first_status.state == "completed" and second_status.state == "cancelled":
                break
            time.sleep(0.01)

        assert first_status.state == "completed"
        assert second_status.state == "cancelled"
        assert second_status.error is not None
        assert second_status.error.code == "job_cancelled"
        assert factory_calls == 1
    finally:
        release_first.set()
        manager.shutdown()

from probing.contracts import JobEvent
from test_service import fake_rank_spec, make_service


def test_interrupted_job_is_failed_with_saved_spec_and_resumable_diagnostic(
    tmp_path,
) -> None:
    service = make_service(tmp_path)
    job = service.initialize_job(
        fake_rank_spec(), request_id="recover-me", state="running"
    )

    recovered = service.repository.recover_interrupted_jobs()

    assert len(recovered) == 1
    status = service.repository.load_job(job.job_id)
    assert status.state == "failed"
    assert status.error is not None
    assert status.error.code == "job_interrupted"
    assert status.error.retryable is True
    assert service.repository.load_job_spec(job.job_id) == fake_rank_spec()
    event = service.repository.read_events(job.job_id)[-1]
    assert event.event == "job.failed"
    assert event.sequence == 0


def test_recovery_reconciles_terminal_event_before_status_update(tmp_path) -> None:
    service = make_service(tmp_path)
    job = service.initialize_job(fake_rank_spec(), state="running")
    service.repository.append_event(
        JobEvent(
            event="job.completed",
            sequence=0,
            timestamp=job.created_at,
            job_id=job.job_id,
            request_id=job.request_id,
            science_hash=job.science_hash,
            payload={"run_id": "run-from-terminal-event"},
        )
    )

    service.repository.recover_interrupted_jobs()

    status = service.repository.load_job(job.job_id)
    assert status.state == "completed"
    assert status.run_id == "run-from-terminal-event"

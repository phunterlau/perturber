from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .contracts import JobEvent


EventListener = Callable[[JobEvent], None]


class EventEmitter:
    def __init__(
        self,
        *,
        job_id: str,
        request_id: str,
        science_hash: str,
        listeners: tuple[EventListener, ...] = (),
    ) -> None:
        self.job_id = job_id
        self.request_id = request_id
        self.science_hash = science_hash
        self.listeners = listeners
        self.sequence = 0

    def emit(self, event: str, **payload: Any) -> JobEvent:
        item = JobEvent(
            event=event,
            sequence=self.sequence,
            timestamp=datetime.now(timezone.utc),
            job_id=self.job_id,
            request_id=self.request_id,
            science_hash=self.science_hash,
            payload=payload,
        )
        self.sequence += 1
        for listener in self.listeners:
            listener(item)
        return item

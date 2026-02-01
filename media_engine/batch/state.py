"""Global state for batch processing."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from media_engine.batch.models import BatchJobStatus, BatchRequest

# In-memory batch store
batch_jobs: dict[str, BatchJobStatus] = {}
batch_jobs_lock = threading.Lock()

# Batch queue - only one batch runs at a time, others wait in queue
batch_queue: list[tuple[str, BatchRequest]] = []  # (batch_id, request) tuples
batch_queue_lock = threading.Lock()

# Use mutable container for batch_running state
_batch_state = {"running": False}


def is_batch_running() -> bool:
    """Check if a batch is currently running."""
    return _batch_state["running"]


def set_batch_running(value: bool) -> None:
    """Set the batch running state."""
    _batch_state["running"] = value

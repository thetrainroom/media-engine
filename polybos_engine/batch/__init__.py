"""Batch processing system for Polybos Media Engine."""

from polybos_engine.batch.models import (
    BatchFileStatus,
    BatchJobStatus,
    BatchRequest,
    ExtractorTiming,
    JobProgress,
)
from polybos_engine.batch.processor import run_batch_job
from polybos_engine.batch.queue import (
    cleanup_expired_batch_jobs,
    start_next_batch,
    update_queue_positions,
)
from polybos_engine.batch.state import (
    batch_jobs,
    batch_jobs_lock,
    batch_queue,
    batch_queue_lock,
    is_batch_running,
    set_batch_running,
)
from polybos_engine.batch.timing import (
    calculate_queue_eta,
    get_enabled_extractors_from_request,
    load_timing_history,
    save_timing_history,
)

__all__ = [
    # Models
    "BatchFileStatus",
    "BatchJobStatus",
    "BatchRequest",
    "ExtractorTiming",
    "JobProgress",
    # State
    "batch_jobs",
    "batch_jobs_lock",
    "batch_queue",
    "batch_queue_lock",
    "is_batch_running",
    "set_batch_running",
    # Queue
    "cleanup_expired_batch_jobs",
    "start_next_batch",
    "update_queue_positions",
    # Timing
    "calculate_queue_eta",
    "get_enabled_extractors_from_request",
    "load_timing_history",
    "save_timing_history",
    # Processor
    "run_batch_job",
]

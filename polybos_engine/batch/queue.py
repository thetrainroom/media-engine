"""Queue management for batch processing."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polybos_engine.batch.models import BatchRequest

from polybos_engine.batch.models import JOB_TTL_SECONDS
from polybos_engine.batch.state import (
    batch_jobs,
    batch_jobs_lock,
    batch_queue,
    batch_queue_lock,
    set_batch_running,
)

logger = logging.getLogger(__name__)


def cleanup_expired_batch_jobs() -> int:
    """Remove completed/failed batch jobs older than TTL.

    Returns:
        Number of batch jobs removed
    """
    now = datetime.now(timezone.utc)
    removed = 0

    with batch_jobs_lock:
        expired = [
            bid
            for bid, batch in batch_jobs.items()
            if batch.status in ("completed", "failed")
            and batch.completed_at is not None
            and (now - batch.completed_at).total_seconds() > JOB_TTL_SECONDS
        ]
        for bid in expired:
            del batch_jobs[bid]
            removed += 1

    if removed > 0:
        logger.info(f"Cleaned up {removed} expired batch jobs")

    return removed


def update_queue_positions() -> None:
    """Update queue_position for all queued batches."""
    with batch_queue_lock:
        with batch_jobs_lock:
            for i, (bid, _) in enumerate(batch_queue):
                if bid in batch_jobs:
                    batch_jobs[bid].queue_position = i + 1  # 1-indexed


def start_next_batch() -> None:
    """Start the next batch from the queue if one exists.

    Called when a batch completes or fails. Sets batch_running = False
    if no more batches in queue.
    """
    from polybos_engine.batch.processor import run_batch_job

    with batch_queue_lock:
        if not batch_queue:
            set_batch_running(False)
            logger.info("Batch queue empty, no more batches to run")
            return

        # Pop the next batch from queue
        next_batch_id, next_request = batch_queue.pop(0)
        logger.info(f"Starting next batch from queue: {next_batch_id}")

    # Update queue positions for remaining batches
    update_queue_positions()

    # Update batch status from queued to pending
    with batch_jobs_lock:
        if next_batch_id in batch_jobs:
            batch_jobs[next_batch_id].status = "pending"
            batch_jobs[next_batch_id].queue_position = None

    # Start the batch in a new thread
    thread = threading.Thread(target=run_batch_job, args=(next_batch_id, next_request))
    thread.start()


def create_batch_sync(
    batch_id: str, request: BatchRequest
) -> tuple[bool, int | None, str]:
    """Synchronous helper to create batch (runs in thread pool).

    Returns:
        (should_start, queue_position, status)
    """
    from pathlib import Path

    from polybos_engine.batch.models import BatchFileStatus, BatchJobStatus
    from polybos_engine.batch.state import is_batch_running, set_batch_running

    # Cleanup expired batch jobs
    cleanup_expired_batch_jobs()

    # Check if we should start immediately or queue
    with batch_queue_lock:
        should_start = not is_batch_running()
        if should_start:
            set_batch_running(True)
            queue_position = None
            status = "pending"
            logger.info(f"Starting batch {batch_id} immediately (no batch running)")
        else:
            # Add to queue
            batch_queue.append((batch_id, request))
            queue_position = len(batch_queue)
            status = "queued"
            logger.info(f"Queued batch {batch_id} at position {queue_position}")

    # Build initial extractor status for each file
    # Order matches processing order in run_batch_job
    # frame_decode is enabled if any visual extractor needs it
    frame_decode_needed = any([
        request.enable_objects,
        request.enable_faces,
        request.enable_ocr,
        request.enable_clip,
    ])
    extractor_flags = [
        ("metadata", request.enable_metadata),
        ("telemetry", True),  # Always runs
        ("vad", request.enable_vad),
        ("motion", request.enable_motion),
        ("scenes", request.enable_scenes),
        ("frame_decode", frame_decode_needed),
        ("objects", request.enable_objects),
        ("faces", request.enable_faces),
        ("ocr", request.enable_ocr),
        ("clip", request.enable_clip),
        ("visual", request.enable_visual),
        ("transcript", request.enable_transcript),
    ]
    initial_extractor_status = {
        name: "pending" if enabled else "skipped" for name, enabled in extractor_flags
    }

    batch = BatchJobStatus(
        batch_id=batch_id,
        status=status,
        queue_position=queue_position,
        files=[
            BatchFileStatus(
                file=f,
                filename=Path(f).name,
                status="pending",
                extractor_status=initial_extractor_status.copy(),
            )
            for f in request.files
        ],
        created_at=datetime.now(timezone.utc),
    )

    with batch_jobs_lock:
        batch_jobs[batch_id] = batch

    return should_start, queue_position, status


def get_batch_sync(batch_id: str, status_only: bool = False):
    """Synchronous helper to get batch status (runs in thread pool).

    Args:
        batch_id: The batch ID to look up
        status_only: If True, return status/progress without large result data
    """
    from polybos_engine.batch.models import BatchFileStatus, BatchJobStatus

    with batch_jobs_lock:
        batch = batch_jobs.get(batch_id)
        if batch is None:
            return None

        if status_only:
            # Return a copy with results stripped out (keep status, progress, timings)
            return BatchJobStatus(
                batch_id=batch.batch_id,
                status=batch.status,
                queue_position=batch.queue_position,
                current_extractor=batch.current_extractor,
                progress=batch.progress,
                files=[
                    BatchFileStatus(
                        file=f.file,
                        filename=f.filename,
                        status=f.status,
                        results={},  # Empty - no large data
                        error=f.error,
                        timings=f.timings,
                        extractor_status=f.extractor_status,
                    )
                    for f in batch.files
                ],
                created_at=batch.created_at,
                completed_at=batch.completed_at,
                extractor_timings=batch.extractor_timings,
                elapsed_seconds=batch.elapsed_seconds,
                memory_mb=batch.memory_mb,
                peak_memory_mb=batch.peak_memory_mb,
            )

        return batch


def delete_batch_sync(batch_id: str) -> tuple[bool, bool]:
    """Synchronous helper to delete batch (runs in thread pool).

    Returns:
        (found, was_queued) - whether batch was found and if it was queued
    """
    with batch_jobs_lock:
        if batch_id not in batch_jobs:
            return False, False
        was_queued = batch_jobs[batch_id].status == "queued"
        del batch_jobs[batch_id]

    # If it was queued, remove from queue and update positions
    if was_queued:
        with batch_queue_lock:
            batch_queue[:] = [(bid, req) for bid, req in batch_queue if bid != batch_id]
        update_queue_positions()

    return True, was_queued

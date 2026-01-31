"""Batch processing endpoints."""

import asyncio
import logging
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from polybos_engine.batch.models import BatchJobStatus, BatchRequest
from polybos_engine.batch.processor import run_batch_job
from polybos_engine.batch.queue import (
    create_batch_sync,
    delete_batch_sync,
    get_batch_sync,
)

router = APIRouter(tags=["batch"])
logger = logging.getLogger(__name__)


@router.post("/batch")
async def create_batch(request: BatchRequest) -> dict[str, str]:
    """Create a new batch extraction job (memory-efficient extractor-first processing).

    Only one batch runs at a time. If a batch is already running, new batches
    are queued and will start automatically when the current batch finishes.
    """
    # Validate all files exist
    for file_path in request.files:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    batch_id = str(uuid.uuid4())[:8]

    # Run lock operations in thread pool to avoid blocking event loop
    should_start, _, _ = await asyncio.to_thread(create_batch_sync, batch_id, request)

    # Start immediately if no batch running
    if should_start:
        thread = threading.Thread(target=run_batch_job, args=(batch_id, request))
        thread.start()

    return {"batch_id": batch_id}


@router.get("/batch/{batch_id}")
async def get_batch(batch_id: str, status_only: bool = False) -> BatchJobStatus:
    """Get batch job status and results.

    Args:
        batch_id: The batch ID to look up
        status_only: If True, return only status/progress without large result data.
            Use this for polling progress to avoid transferring large embeddings/transcripts.
    """
    # Run lock acquisition in thread pool to avoid blocking event loop
    result = await asyncio.to_thread(get_batch_sync, batch_id, status_only)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
    return result


@router.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str) -> dict[str, str]:
    """Delete a batch job and free its memory.

    Jobs can be deleted at any time. If the batch is queued, it will be
    removed from the queue. If running, deletion will not stop processing
    - it will just remove the status tracking.
    """
    # Run lock acquisition in thread pool to avoid blocking event loop
    found, _ = await asyncio.to_thread(delete_batch_sync, batch_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    logger.info(f"Deleted batch job {batch_id}")
    return {"status": "deleted", "batch_id": batch_id}

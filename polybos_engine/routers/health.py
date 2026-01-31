"""Health and monitoring endpoints."""

import logging
import os
import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException

from polybos_engine import __version__
from polybos_engine.config import get_settings, get_vram_summary
from polybos_engine.schemas import HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

LOG_FILE = "/tmp/polybos_engine.log"


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        api_version=settings.api_version,
    )


@router.get("/logs")
async def get_logs(
    lines: int = 100,
    level: str | None = None,
) -> dict[str, Any]:
    """Get recent log entries for debugging.

    Args:
        lines: Number of lines to return (default 100, max 1000)
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Dict with log lines and metadata
    """
    lines = min(lines, 1000)  # Cap at 1000 lines

    if not os.path.exists(LOG_FILE):
        return {"lines": [], "total": 0, "returned": 0, "file": LOG_FILE}

    try:
        # Use tail to efficiently read last N lines without loading entire file
        # Read more lines if filtering by level (we'll filter down after)
        read_lines = lines * 10 if level else lines

        result = subprocess.run(
            ["tail", "-n", str(read_lines), LOG_FILE],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"tail failed: {result.stderr}")

        all_lines = result.stdout.splitlines()

        # Filter by level if specified
        if level:
            level_upper = level.upper()
            all_lines = [line for line in all_lines if f" {level_upper} " in line]
            # Take only requested number after filtering
            all_lines = all_lines[-lines:]

        return {
            "lines": all_lines,
            "total": len(all_lines),  # Note: this is approximate when using tail
            "returned": len(all_lines),
            "file": LOG_FILE,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout reading logs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")


@router.get("/hardware")
async def hardware():
    """Get hardware capabilities and auto-selected models.

    Returns information about available GPU/VRAM and which models
    will be used with the current "auto" settings.
    """
    return get_vram_summary()

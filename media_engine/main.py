"""FastAPI application for Media Engine."""

# Prevent fork crashes on macOS with Hugging Face tokenizers library.
# The tokenizers library registers atfork handlers that panic when the process forks
# (e.g., to run ffmpeg via subprocess). This must be set BEFORE any imports.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# On macOS, use 'spawn' instead of 'fork' for multiprocessing to avoid crashes
# with libraries that aren't fork-safe (tokenizers, PyTorch, etc.)
import multiprocessing
import sys

if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# Setup logging before any other imports
# ruff: noqa: E402 (imports after environment setup is intentional)
from media_engine.utils.logging import setup_logging

setup_logging()

# Create the FastAPI application
from media_engine.app import create_app

app = create_app()

# Re-export batch state for backward compatibility with tests
# These were previously defined directly in main.py
# ruff: noqa: F401 (re-exports are intentional)
from media_engine.batch import state as _batch_state
from media_engine.batch.models import JOB_TTL_SECONDS  # noqa: F401
from media_engine.batch.queue import (  # noqa: F401
    cleanup_expired_batch_jobs as _cleanup_expired_batch_jobs,
)
from media_engine.batch.state import (  # noqa: F401
    batch_jobs,
    batch_jobs_lock,
    batch_queue,
    batch_queue_lock,
)

batch_running = _batch_state._batch_state["running"]


# Make batch_running assignable at module level for tests
# ruff: noqa: N807 (module-level __getattr__ and __setattr__ are valid Python)
def __getattr__(name: str):  # noqa: N807
    if name == "batch_running":
        return _batch_state._batch_state["running"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __setattr__(name: str, value):  # noqa: N807
    if name == "batch_running":
        _batch_state._batch_state["running"] = value
    else:
        globals()[name] = value


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

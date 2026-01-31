"""Non-blocking queue-based logging setup."""

import atexit
import logging
import logging.handlers
import os
import queue
import sys


def setup_logging() -> logging.handlers.QueueListener:
    """Configure non-blocking logging using a queue.

    Returns the QueueListener so it can be stopped on shutdown.
    """
    # When running under a parent process that doesn't read our stdout/stderr,
    # writes block when the pipe buffer fills up. Redirect to /dev/null to prevent this.
    is_interactive = sys.stdout.isatty() and sys.stderr.isatty()
    if not is_interactive:
        # Redirect stdout/stderr to /dev/null to prevent blocking writes
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

    # Configure non-blocking logging using a queue
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)  # Unlimited size
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # Always log to file (this is the only output when running non-interactively)
    file_handler = logging.FileHandler("/tmp/polybos_engine.log")
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(log_formatter)

    # Build handler list - only include stderr if running interactively
    handlers: list[logging.Handler] = [file_handler]
    if is_interactive:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        handlers.append(stream_handler)

    # QueueListener handles the actual I/O in a separate thread
    queue_listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    queue_listener.start()

    # Register cleanup on exit
    atexit.register(queue_listener.stop)

    # Configure root logger to use queue handler (non-blocking)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[queue_handler],
    )

    return queue_listener

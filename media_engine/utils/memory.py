"""Memory management utilities."""

import gc


def clear_memory() -> None:
    """Force garbage collection and clear GPU/MPS caches.

    Call before loading heavy AI models to free up memory.
    """
    # Multiple gc passes to handle circular references
    for _ in range(3):
        gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if hasattr(torch, "mps"):
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except ImportError:
        pass

    # Also try mlx cleanup
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    # Final gc pass after GPU cleanup
    gc.collect()


def get_memory_mb() -> int:
    """Get current process memory usage in MB."""
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process()
        return process.memory_info().rss // (1024 * 1024)
    except ImportError:
        return 0

"""CLI entry point for meng-server."""

import uvicorn


def run_server() -> None:
    """Run the Media Engine API server."""
    uvicorn.run(
        "media_engine.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )


if __name__ == "__main__":
    run_server()

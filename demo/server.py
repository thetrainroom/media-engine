"""Demo server with file browsing and video streaming utilities.

This is a lightweight development server that provides:
- Directory browsing for video files
- Video streaming with range request support

Run: python demo/server.py
Default port: 8002
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Polybos Demo Server")

# CORS for cross-origin requests from demo frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MIME types for video files
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mxf": "video/mxf",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
    ".webm": "video/webm",
}


@app.get("/browse")
async def browse_directory(
    path: str = Query("/Volumes/Backup", description="Directory to browse"),
):
    """Browse a directory for video files."""
    dir_path = Path(path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    video_extensions = {".mp4", ".mov", ".mxf", ".avi", ".mkv", ".m4v", ".webm"}
    items = []

    try:
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                items.append({"name": item.name, "path": str(item), "type": "directory"})
            elif item.suffix.lower() in video_extensions:
                size_mb = item.stat().st_size / (1024 * 1024)
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "video",
                    "size_mb": round(size_mb, 1),
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    return {
        "path": str(dir_path),
        "parent": str(dir_path.parent) if dir_path.parent != dir_path else None,
        "items": items,
    }


@app.get("/video")
async def stream_video(
    request: Request,
    file: str = Query(..., description="Path to video file"),
):
    """Stream a video file with range request support for seeking."""
    file_path = Path(file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file}")

    file_size = file_path.stat().st_size
    content_type = VIDEO_MIME_TYPES.get(file_path.suffix.lower(), "video/mp4")

    # Parse range header for seeking support
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end" format
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1

        # Clamp end to file size
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(remaining, 1024 * 1024)  # 1MB chunks
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,  # Partial Content
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )
    else:
        # No range header - stream entire file
        def iter_file():
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            iter_file(),
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )


# Serve demo static files at root
app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="demo")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)

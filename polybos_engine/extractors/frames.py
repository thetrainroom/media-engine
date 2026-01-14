"""Fast frame extraction using OpenCV.

OpenCV's VideoCapture is significantly faster than spawning ffmpeg processes
for each frame, especially when extracting multiple frames from the same video.
"""

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video using OpenCV.

    Uses cv2.VideoCapture for fast seeking and frame extraction.
    Falls back to ffmpeg for exotic codecs that OpenCV can't handle.
    """

    # Default max dimension - scale down 4K to ~HD for faster processing
    DEFAULT_MAX_DIMENSION = 1920

    def __init__(self, video_path: str, max_dimension: int | None = DEFAULT_MAX_DIMENSION):
        """Initialize frame extractor.

        Args:
            video_path: Path to video file
            max_dimension: Maximum width/height. Frames larger than this are scaled down.
                          Set to None to disable scaling. Default: 1920 (HD)
        """
        self.video_path = video_path
        self.max_dimension = max_dimension
        self.cap: cv2.VideoCapture | None = None
        self._duration: float | None = None
        self._fps: float | None = None
        self._frame_count: int | None = None
        self._use_ffmpeg_fallback = False

    def __enter__(self) -> "FrameExtractor":
        """Open video file."""
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            logger.warning(f"OpenCV failed to open {self.video_path}, using ffmpeg fallback")
            self._use_ffmpeg_fallback = True
            self.cap = None
        else:
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self._fps > 0 and self._frame_count > 0:
                self._duration = self._frame_count / self._fps
            else:
                # Seek to estimate duration
                self._duration = self._get_duration_ffprobe()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        if self._duration is None:
            self._duration = self._get_duration_ffprobe()
        return self._duration

    @property
    def fps(self) -> float:
        """Get video frame rate."""
        if self._fps is None:
            self._fps = 30.0  # Default fallback
        return self._fps

    def _get_duration_ffprobe(self) -> float:
        """Get duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0.0

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Scale down frame if larger than max_dimension.

        Maintains aspect ratio. Only scales down, never up.
        """
        if self.max_dimension is None:
            return frame

        h, w = frame.shape[:2]
        max_dim = max(h, w)

        if max_dim <= self.max_dimension:
            return frame

        # Calculate scale factor
        scale = self.max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Use INTER_AREA for downscaling (best quality)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def get_frame_at(self, timestamp: float) -> np.ndarray | None:
        """Extract a single frame at the given timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as BGR numpy array (scaled to max_dimension), or None if extraction failed
        """
        if self._use_ffmpeg_fallback:
            frame = self._get_frame_ffmpeg(timestamp)
            return self._scale_frame(frame) if frame is not None else None

        if self.cap is None:
            return None

        # Seek to timestamp
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        ret, frame = self.cap.read()
        if not ret:
            # Try ffmpeg fallback for this frame
            frame = self._get_frame_ffmpeg(timestamp)
            return self._scale_frame(frame) if frame is not None else None

        return self._scale_frame(frame)

    def get_frames_at(self, timestamps: list[float]) -> list[tuple[float, np.ndarray | None]]:
        """Extract multiple frames at given timestamps.

        More efficient than calling get_frame_at repeatedly as it
        processes timestamps in order to minimize seeking.

        Args:
            timestamps: List of times in seconds

        Returns:
            List of (timestamp, frame) tuples
        """
        # Sort timestamps for efficient sequential access
        sorted_ts = sorted(set(timestamps))
        results: dict[float, np.ndarray | None] = {}

        for ts in sorted_ts:
            results[ts] = self.get_frame_at(ts)

        # Return in original order
        return [(ts, results.get(ts)) for ts in timestamps]

    def _get_frame_ffmpeg(self, timestamp: float) -> np.ndarray | None:
        """Extract frame using ffmpeg (fallback)."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", self.video_path,
                "-frames:v", "1",
                "-q:v", "2",
                tmp_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            if os.path.exists(tmp_path):
                frame = cv2.imread(tmp_path)
                return frame
        except subprocess.CalledProcessError:
            pass
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return None

    def save_frame(self, frame: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """Save frame to file.

        Args:
            frame: BGR numpy array
            output_path: Output file path
            quality: JPEG quality (0-100)

        Returns:
            True if saved successfully
        """
        try:
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return os.path.exists(output_path)
        except Exception as e:
            logger.warning(f"Failed to save frame to {output_path}: {e}")
            return False


def extract_frames_batch(
    video_path: str,
    timestamps: list[float],
    output_dir: str | None = None,
) -> list[tuple[float, np.ndarray | None]]:
    """Extract multiple frames from a video.

    Convenience function that handles the context manager.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps in seconds
        output_dir: Optional directory to save frames as JPEG files

    Returns:
        List of (timestamp, frame) tuples
    """
    with FrameExtractor(video_path) as extractor:
        results = extractor.get_frames_at(timestamps)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for ts, frame in results:
                if frame is not None:
                    output_path = os.path.join(output_dir, f"frame_{ts:.3f}.jpg")
                    extractor.save_frame(frame, output_path)

        return results


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or 0 if unable to determine
    """
    # Try OpenCV first (faster)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0 and frame_count > 0:
            return frame_count / fps

    # Fallback to ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0

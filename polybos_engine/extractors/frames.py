"""Fast frame extraction using OpenCV or FFmpeg.

OpenCV's VideoCapture is fast for normal videos but decodes at full resolution.
For high-resolution videos (4K+), FFmpeg decoding at target resolution is faster.

Also supports direct image loading - when given an image file, it loads it
directly instead of trying to use VideoCapture.
"""

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

from polybos_engine.schemas import MediaType, get_media_type

logger = logging.getLogger(__name__)

# Resolution threshold for using FFmpeg decode (4K+)
HIGH_RES_THRESHOLD = 3840 * 2160  # ~8.3M pixels


class FrameExtractor:
    """Extract frames from video or image using OpenCV.

    Uses cv2.VideoCapture for fast seeking and frame extraction from videos.
    Falls back to ffmpeg for exotic codecs that OpenCV can't handle.
    For images, loads directly with cv2.imread (no frame extraction needed).
    """

    # Default max dimension - scale down 4K to ~HD for faster processing
    DEFAULT_MAX_DIMENSION = 1920

    def __init__(
        self, file_path: str, max_dimension: int | None = DEFAULT_MAX_DIMENSION
    ):
        """Initialize frame extractor.

        Args:
            file_path: Path to video or image file
            max_dimension: Maximum width/height. Frames larger than this are scaled down.
                          Set to None to disable scaling. Default: 1920 (HD)
        """
        self.video_path = file_path  # Keep name for compatibility
        self.max_dimension = max_dimension
        self.cap: cv2.VideoCapture | None = None
        self._duration: float | None = None
        self._fps: float | None = None
        self._frame_count: int | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._use_ffmpeg_fallback = False
        self._use_ffmpeg_decode = False  # For high-res, decode at lower res with FFmpeg
        # Image handling
        self._is_image = False
        self._image_frame: np.ndarray | None = None

    def __enter__(self) -> "FrameExtractor":
        """Open video or image file."""
        # Check if this is an image file
        media_type = get_media_type(self.video_path)
        if media_type == MediaType.IMAGE:
            self._is_image = True
            self._duration = 0.0
            self._fps = 1.0
            self._frame_count = 1
            # Load the image directly
            self._image_frame = cv2.imread(self.video_path)
            if self._image_frame is None:
                logger.warning(f"Failed to load image: {self.video_path}")
            else:
                # Apply scaling
                self._image_frame = self._scale_frame(self._image_frame)
                logger.debug(f"Loaded image directly: {self.video_path}")
            return self

        # Video file - use VideoCapture
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            logger.warning(
                f"OpenCV failed to open {self.video_path}, using ffmpeg fallback"
            )
            self._use_ffmpeg_fallback = True
            self.cap = None
        else:
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self._fps > 0 and self._frame_count > 0:
                self._duration = self._frame_count / self._fps
            else:
                # Seek to estimate duration
                self._duration = self._get_duration_ffprobe()

            # Check if this is high-res video that needs FFmpeg decode
            if self._width and self._height and self.max_dimension:
                pixels = self._width * self._height
                max_dim = max(self._width, self._height)
                if pixels > HIGH_RES_THRESHOLD and max_dim > self.max_dimension:
                    logger.info(
                        f"High-res video ({self._width}x{self._height}), "
                        f"using FFmpeg decode at {self.max_dimension}px"
                    )
                    self._use_ffmpeg_decode = True
                    # Release opencv capture - we'll use FFmpeg instead
                    self.cap.release()
                    self.cap = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # Clear image reference
        self._image_frame = None

    @property
    def is_image(self) -> bool:
        """Check if this extractor is handling an image (not a video)."""
        return self._is_image

    @property
    def duration(self) -> float:
        """Get video duration in seconds (0 for images)."""
        if self._duration is None:
            self._duration = self._get_duration_ffprobe()
        return self._duration

    @property
    def fps(self) -> float:
        """Get video frame rate (1 for images)."""
        if self._fps is None:
            self._fps = 30.0  # Default fallback
        return self._fps

    def _get_duration_ffprobe(self) -> float:
        """Get duration using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
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
            timestamp: Time in seconds (ignored for images)

        Returns:
            Frame as BGR numpy array (scaled to max_dimension), or None if extraction failed
        """
        # For images, always return the loaded image (timestamp is ignored)
        if self._is_image:
            return self._image_frame

        # High-res video: use FFmpeg with scale filter (decodes at target res)
        if self._use_ffmpeg_decode:
            return self._get_frame_ffmpeg_scaled(timestamp)

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

    def get_frames_at(
        self, timestamps: list[float]
    ) -> list[tuple[float, np.ndarray | None]]:
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
        """Extract frame using ffmpeg (fallback, no scaling)."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                self.video_path,
                "-frames:v",
                "1",
                "-update",
                "1",  # Required for ffmpeg 8.x single-image output
                "-q:v",
                "2",
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

    def _get_frame_ffmpeg_scaled(self, timestamp: float) -> np.ndarray | None:
        """Extract frame using FFmpeg with scale filter (for high-res videos).

        This is faster than decoding at full resolution and then scaling with cv2.
        """
        if self.max_dimension is None:
            return self._get_frame_ffmpeg(timestamp)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Scale filter that maintains aspect ratio
            # scale=W:H:force_original_aspect_ratio=decrease
            scale_filter = (
                f"scale={self.max_dimension}:{self.max_dimension}"
                f":force_original_aspect_ratio=decrease"
            )

            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                self.video_path,
                "-vf",
                scale_filter,
                "-frames:v",
                "1",
                "-update",
                "1",
                "-q:v",
                "2",
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

    def save_frame(
        self, frame: np.ndarray, output_path: str, quality: int = 95
    ) -> bool:
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


def get_video_duration(file_path: str) -> float:
    """Get video duration in seconds.

    Args:
        file_path: Path to video or image file

    Returns:
        Duration in seconds, or 0 for images/unknown files
    """
    # Check if this is an image - images have 0 duration
    media_type = get_media_type(file_path)
    if media_type == MediaType.IMAGE:
        return 0.0

    # Try OpenCV first (faster)
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0 and frame_count > 0:
            return frame_count / fps

    # Fallback to ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0

"""Shared frame buffer for efficient video processing.

Decodes video frames once and shares them across multiple extractors,
eliminating redundant video decoding which is the performance bottleneck.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from polybos_engine.extractors.metadata.base import get_video_info

logger = logging.getLogger(__name__)

# Cache for hardware acceleration detection
_hwaccel_cache: str | None = None


def _detect_hwaccel() -> str | None:
    """Detect available hardware acceleration for video decoding.

    Returns:
        Hardware acceleration method name, or None if not available.
        - "videotoolbox" for macOS
        - "cuda" for NVIDIA GPUs
        - None for software decoding
    """
    global _hwaccel_cache

    if _hwaccel_cache is not None:
        return _hwaccel_cache if _hwaccel_cache != "" else None

    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "videotoolbox" in result.stdout:
                logger.info("Hardware acceleration: VideoToolbox (macOS)")
                _hwaccel_cache = "videotoolbox"
                return "videotoolbox"
        except Exception:
            pass

    elif system == "Linux":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "cuda" in result.stdout:
                nvidia_check = subprocess.run(
                    ["nvidia-smi", "-L"],
                    capture_output=True,
                    timeout=5,
                )
                if nvidia_check.returncode == 0:
                    logger.info("Hardware acceleration: CUDA (NVIDIA)")
                    _hwaccel_cache = "cuda"
                    return "cuda"
        except Exception:
            pass

    logger.info("Hardware acceleration: None (software decoding)")
    _hwaccel_cache = ""
    return None


@dataclass
class SharedFrame:
    """A decoded frame with lazy format conversions.

    Stores the original BGR frame and provides lazy conversion to other formats
    (RGB, grayscale, PIL) to minimize memory usage and conversion overhead.
    """

    timestamp: float
    bgr: np.ndarray
    _rgb: np.ndarray | None = field(default=None, repr=False)
    _gray: np.ndarray | None = field(default=None, repr=False)
    _pil: Image.Image | None = field(default=None, repr=False)

    @property
    def rgb(self) -> np.ndarray:
        """Get RGB format (lazy conversion from BGR)."""
        if self._rgb is None:
            self._rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        return self._rgb

    @property
    def gray(self) -> np.ndarray:
        """Get grayscale format (lazy conversion from BGR)."""
        if self._gray is None:
            self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def pil(self) -> Image.Image:
        """Get PIL Image format (lazy conversion from RGB)."""
        if self._pil is None:
            self._pil = Image.fromarray(self.rgb)
        return self._pil


@dataclass
class SharedFrameBuffer:
    """Buffer of decoded frames for a video file.

    Holds pre-decoded frames that can be shared across multiple extractors,
    eliminating redundant video decoding.
    """

    file_path: str
    frames: dict[float, SharedFrame]  # timestamp -> frame
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.frames)

    def get_frame(self, timestamp: float) -> SharedFrame | None:
        """Get frame at exact timestamp."""
        return self.frames.get(timestamp)

    def get_nearest_frame(self, timestamp: float) -> SharedFrame | None:
        """Get frame nearest to timestamp."""
        if not self.frames:
            return None
        nearest_ts = min(self.frames.keys(), key=lambda t: abs(t - timestamp))
        return self.frames[nearest_ts]

    def timestamps(self) -> list[float]:
        """Get sorted list of all timestamps."""
        return sorted(self.frames.keys())


def _extract_single_frame(
    file_path: str,
    timestamp: float,
    out_width: int,
    out_height: int,
    hwaccel: str | None,
    src_width: int,
    src_height: int,
) -> np.ndarray | None:
    """Extract a single frame at the given timestamp.

    Returns BGR numpy array or None on failure.
    """
    cmd = ["ffmpeg", "-hide_banner"]

    # Calculate output height for hardware scaling (maintains aspect ratio)
    if hwaccel and src_width > 0 and src_height > 0:
        actual_out_height = int(out_width * src_height / src_width)
        actual_out_height = actual_out_height - (actual_out_height % 2)
    else:
        actual_out_height = out_height

    # Build filter chain based on hardware acceleration
    if hwaccel == "videotoolbox":
        cmd.extend(
            ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
        )
        vf_filter = (
            f"scale_vt=w={out_width}:h={actual_out_height},hwdownload,format=p010le"
        )
    elif hwaccel == "cuda":
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        vf_filter = f"scale_cuda={out_width}:{actual_out_height},hwdownload,format=nv12"
    else:
        actual_out_height = out_height
        vf_filter = (
            f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease"
        )

    cmd.extend(
        [
            "-ss",
            str(timestamp),
            "-i",
            file_path,
            "-vf",
            vf_filter,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    )

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_size = out_width * actual_out_height * 3  # BGR = 3 channels
        raw_frame, _ = process.communicate(timeout=30)

        if len(raw_frame) != frame_size:
            # Try without hardware acceleration
            if hwaccel:
                logger.debug(
                    f"Hardware decode failed for frame at {timestamp}s, trying software"
                )
                return _extract_single_frame(
                    file_path,
                    timestamp,
                    out_width,
                    out_height,
                    None,
                    src_width,
                    src_height,
                )
            return None

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            (actual_out_height, out_width, 3)
        )
        return frame.copy()

    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
        logger.warning(f"Timeout extracting frame at {timestamp}s")
        return None
    except Exception as e:
        logger.warning(f"Error extracting frame at {timestamp}s: {e}")
        return None


def decode_frames(
    file_path: str,
    timestamps: list[float],
    max_dimension: int = 1920,
    hwaccel: str | None = None,
) -> SharedFrameBuffer:
    """Decode specific frames from video with hardware acceleration.

    Uses VideoToolbox on macOS, CUDA on Linux, with automatic fallback
    to software decoding if hardware fails.

    Args:
        file_path: Path to video file
        timestamps: List of timestamps (in seconds) to extract
        max_dimension: Maximum width or height (maintains aspect ratio)
        hwaccel: Hardware acceleration method (auto-detect if None)

    Returns:
        SharedFrameBuffer with decoded frames
    """
    if hwaccel is None:
        hwaccel = _detect_hwaccel()

    # Get video info
    _, duration, src_width, src_height = get_video_info(file_path)

    # Calculate output dimensions (maintain aspect ratio, cap at max_dimension)
    if src_width > src_height:
        out_width = min(max_dimension, src_width)
        out_height = int(out_width * src_height / src_width)
    else:
        out_height = min(max_dimension, src_height)
        out_width = int(out_height * src_width / src_height)

    # Ensure even dimensions
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)

    logger.info(
        f"Decoding {len(timestamps)} frames from {file_path} "
        f"at {out_width}x{out_height}" + (f" (hwaccel={hwaccel})" if hwaccel else "")
    )

    frames: dict[float, SharedFrame] = {}

    for ts in timestamps:
        if ts < 0 or ts > duration:
            logger.debug(f"Skipping out-of-range timestamp: {ts}")
            continue

        frame_bgr = _extract_single_frame(
            file_path, ts, out_width, out_height, hwaccel, src_width, src_height
        )
        if frame_bgr is not None:
            frames[ts] = SharedFrame(timestamp=ts, bgr=frame_bgr)
        else:
            logger.debug(f"Failed to decode frame at {ts}s")

    logger.info(f"Decoded {len(frames)}/{len(timestamps)} frames successfully")

    return SharedFrameBuffer(
        file_path=file_path,
        frames=frames,
        width=out_width,
        height=out_height,
    )


def get_extractor_timestamps(
    is_stable: bool,
    avg_intensity: float,
    base_timestamps: list[float],
) -> list[float]:
    """Filter timestamps based on motion analysis for efficient sampling.

    For stable footage, reduces sampling significantly since content
    doesn't change much between frames.

    Args:
        is_stable: Whether motion analysis determined footage is stable
        avg_intensity: Average motion intensity from motion analysis
        base_timestamps: Full list of timestamps from adaptive sampling

    Returns:
        Filtered list of timestamps appropriate for the motion level
    """
    if not base_timestamps:
        return []

    if is_stable and avg_intensity < 1.0:
        # Very stable (tripod, hover) - just 3 frames
        if len(base_timestamps) <= 3:
            return base_timestamps
        return [
            base_timestamps[0],
            base_timestamps[len(base_timestamps) // 2],
            base_timestamps[-1],
        ]

    if is_stable:
        # Mostly stable - reduce to every 4th timestamp
        result = base_timestamps[::4]
        return result if result else base_timestamps[:1]

    # Dynamic footage - use all timestamps
    return base_timestamps

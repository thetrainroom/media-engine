"""Camera motion analysis using optical flow."""

import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from media_engine.extractors.metadata.base import get_video_info

logger = logging.getLogger(__name__)

# Cache for hardware acceleration detection
_hwaccel_cache: str | None = None

# Analysis resolution (scale down for speed)
ANALYSIS_HEIGHT = 720
ANALYSIS_WIDTH = 1280  # 720p aspect ratio

# Motion detection thresholds
MOTION_THRESHOLD = 2.0  # Minimum average flow magnitude for motion
PAN_TILT_THRESHOLD = 0.7  # Ratio of directional vs total flow for pan/tilt
ZOOM_THRESHOLD = 0.3  # Divergence threshold for zoom detection
STATIC_THRESHOLD = 0.5  # Below this = static

# Extended motion feature extraction (api_version 1.1)
# Bump MOTION_FEATURES_VERSION when any feature formula, the frequency split,
# or the smoothing parameters change - consumers cache derived scores keyed on it.
MOTION_FEATURES_VERSION = "v1"
FREQ_SPLIT_HZ = 2.0  # HF/LF band split for frequency energy (analysis runs at 5 fps -> Nyquist 2.5 Hz)
SMOOTH_WINDOW = 5  # Moving-average window on the magnitude series before differentiating
MIN_FEATURE_SAMPLES = 5  # Below this, features degenerate to documented defaults


class MotionType(StrEnum):
    """Types of camera motion.

    Note: PUSH_IN/PULL_OUT describe the optical flow pattern (radial expansion/contraction).
    This could be optical zoom OR physical camera movement (dolly/travel).
    Frontend can interpret based on metadata (lens type, GPS movement, device type).
    """

    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    PUSH_IN = "push_in"  # Radial expansion (zoom in or dolly forward)
    PULL_OUT = "pull_out"  # Radial contraction (zoom out or dolly backward)
    HANDHELD = "handheld"  # Random/shaky movement
    COMPLEX = "complex"  # Multiple motions combined


@dataclass(slots=True, frozen=True)
class MotionFeatures:
    """Extended motion statistics for a time span (segment or whole clip).

    All values are derived from the per-frame mean optical flow series that
    classification already computes; no additional model inference.
    """

    magnitude_mean: float  # Mean flow magnitude (same as MotionSegment.intensity)
    magnitude_std: float  # Std dev of magnitude
    magnitude_p90: float  # 90th percentile of magnitude (robust near-peak)
    magnitude_max: float  # Max magnitude (spikes / near-yanks)
    direction_consistency: float  # [0, 1]: 1.0 = monotonic direction, 0.0 = random/reversing
    direction_reversals_per_sec: float  # Direction changes >90 degrees per second
    acceleration_mean: float  # Mean |d(magnitude)/dt| (magnitude per second)
    jerk_max: float  # Peak |d2(magnitude)/dt2| (the signature of "yanking")
    hf_energy: float  # [0, 1]: normalized energy above FREQ_SPLIT_HZ (shake, tremor)
    lf_energy: float  # [0, 1]: normalized energy below FREQ_SPLIT_HZ (intentional movement)
    hf_lf_ratio: float  # hf / (hf + lf), in [0, 1]. High = shaky.


@dataclass(slots=True)
class MotionSegment:
    """A segment of video with consistent motion.

    Uses slots=True to reduce memory overhead per instance.
    """

    start: float
    end: float
    motion_type: MotionType
    intensity: float  # Average flow magnitude
    features: MotionFeatures | None = None  # Extended stats (api_version 1.1)


@dataclass(slots=True)
class MotionAnalysis:
    """Complete motion analysis for a video.

    Uses slots=True to reduce memory overhead per instance.
    """

    duration: float
    fps: float
    primary_motion: MotionType
    segments: list[MotionSegment]
    avg_intensity: float
    is_stable: bool  # True if mostly static/tripod
    # Clip-level feature summaries (api_version 1.1)
    magnitude_p90_overall: float = 0.0
    jerk_max_overall: float = 0.0
    hf_lf_ratio_overall: float = 0.0
    features_version: str = MOTION_FEATURES_VERSION


# Chunk duration in seconds (2 minutes)
CHUNK_DURATION = 120.0


def _detect_hwaccel() -> str | None:
    """Detect available hardware acceleration for video decoding.

    Returns:
        Hardware acceleration method name, or None if not available.
        - "videotoolbox" for macOS (Apple Silicon or Intel with VideoToolbox)
        - "cuda" for NVIDIA GPUs
        - None for software decoding
    """
    global _hwaccel_cache

    if _hwaccel_cache is not None:
        return _hwaccel_cache if _hwaccel_cache != "" else None

    # Check platform
    system = platform.system()

    if system == "Darwin":
        # macOS - check for VideoToolbox support
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
        # Linux - check for CUDA/NVDEC
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "cuda" in result.stdout:
                # Verify NVIDIA GPU is present
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


def _load_frames_chunk(
    file_path: str,
    start_time: float,
    chunk_duration: float,
    sample_fps: float,
    out_width: int,
    out_height: int,
    hwaccel: str | None = None,
    src_width: int = 0,
    src_height: int = 0,
) -> np.ndarray:
    """Load a chunk of frames into memory using FFmpeg.

    Args:
        file_path: Path to video file
        start_time: Start time in seconds
        chunk_duration: Duration of chunk to load in seconds
        sample_fps: Frames per second to sample
        out_width: Output frame width
        out_height: Output frame height
        hwaccel: Hardware acceleration method (videotoolbox, cuda, or None)
        src_width: Source video width (for aspect ratio calculation with hwaccel)
        src_height: Source video height (for aspect ratio calculation with hwaccel)

    Returns:
        numpy array of shape (num_frames, height, width) with grayscale frames
    """
    # Build command with optional hardware acceleration
    cmd = ["ffmpeg", "-hide_banner"]

    # For hardware acceleration, calculate output height based on source aspect ratio
    actual_out_height = out_height
    if hwaccel and src_width > 0 and src_height > 0:
        # Calculate height maintaining aspect ratio, rounded to even number
        actual_out_height = int(out_width * src_height / src_width)
        actual_out_height = actual_out_height - (actual_out_height % 2)  # Ensure even

    # Use hardware-accelerated decode and scaling if available
    if hwaccel == "videotoolbox":
        # Decode on hardware, scale on GPU, then transfer to CPU
        # p010le is required for VideoToolbox hwdownload (10-bit format)
        cmd.extend(["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"])
        vf_filter = f"scale_vt=w={out_width}:h={actual_out_height},hwdownload,format=p010le,fps={sample_fps}"
    elif hwaccel == "cuda":
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        vf_filter = f"scale_cuda={out_width}:{actual_out_height},hwdownload,format=nv12,fps={sample_fps}"
    else:
        actual_out_height = out_height  # Use the provided value for software
        vf_filter = f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease,fps={sample_fps}"

    cmd.extend(
        [
            "-ss",
            str(start_time),
            "-t",
            str(chunk_duration),
            "-i",
            file_path,
            "-vf",
            vf_filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )

    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frame_size = out_width * actual_out_height
    frames: list[np.ndarray] = []

    while True:
        raw_frame = process.stdout.read(frame_size)  # type: ignore[union-attr]
        if len(raw_frame) != frame_size:
            break
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((actual_out_height, out_width))
        frames.append(frame.copy())

    _, stderr = process.communicate()
    if stderr and logger.isEnabledFor(logging.DEBUG):
        # Log first few lines of stderr for debugging
        stderr_lines = stderr.decode(errors="ignore").strip().split("\n")[:5]
        for line in stderr_lines:
            if line:
                logger.debug(f"FFmpeg: {line}")

    # If hardware acceleration failed (no frames), retry without it
    if not frames and hwaccel:
        # Log the stderr to understand why it failed
        if stderr:
            logger.warning(f"Hardware acceleration ({hwaccel}) failed: {stderr.decode(errors='ignore')[:500]}")
        else:
            logger.warning(f"Hardware acceleration ({hwaccel}) failed, no frames produced")
        return _load_frames_chunk(
            file_path,
            start_time,
            chunk_duration,
            sample_fps,
            out_width,
            out_height,
            hwaccel=None,
            src_width=src_width,
            src_height=src_height,
        )

    if not frames:
        return np.array([], dtype=np.uint8)

    return np.stack(frames)


def analyze_motion(
    file_path: str,
    sample_fps: float = 5.0,  # Analyze every N frames per second
    chunk_duration: float = CHUNK_DURATION,  # Process 2 minutes at a time
) -> MotionAnalysis:
    """Analyze camera motion in a video using optical flow.

    Uses FFmpeg to decode frames at low resolution for efficiency with high-res video.
    Processes in 2-minute chunks to balance memory usage and I/O efficiency.

    Args:
        file_path: Path to video file
        sample_fps: How many frames per second to analyze (default 5)
        chunk_duration: Duration of each processing chunk in seconds (default 120)

    Returns:
        MotionAnalysis with motion type segments
    """
    # Get video info
    fps, duration, width, height = get_video_info(file_path)
    if duration == 0:
        # Fallback to opencv for duration
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

    total_samples = int(duration * sample_fps)

    # Detect hardware acceleration
    hwaccel = _detect_hwaccel()

    logger.info(f"Analyzing motion: {duration:.1f}s @ {fps:.1f}fps, ~{total_samples} samples" + (f" (hwaccel={hwaccel})" if hwaccel else ""))

    # Calculate actual frame dimensions after scaling
    if width > height:
        out_width = ANALYSIS_WIDTH
        out_height = int(height * ANALYSIS_WIDTH / width)
    else:
        out_height = ANALYSIS_HEIGHT
        out_width = int(width * ANALYSIS_HEIGHT / height)

    # Ensure even dimensions for video
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)

    # Per-frame series: (timestamp, motion_type, intensity, mean_x, mean_y)
    # The mean flow vector is retained for extended feature computation.
    frame_motions: list[tuple[float, MotionType, float, float, float]] = []
    prev_gray: np.ndarray | None = None
    global_frame_idx = 0

    # Timing stats
    total_load_time = 0.0
    total_flow_time = 0.0

    # Process video in chunks
    num_chunks = max(1, int(np.ceil(duration / chunk_duration)))
    logger.debug(f"Processing in {num_chunks} chunk(s) of {chunk_duration}s each")

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_duration
        actual_chunk_duration = min(chunk_duration, duration - chunk_start)

        if actual_chunk_duration <= 0:
            break

        logger.debug(f"Loading chunk {chunk_idx + 1}/{num_chunks}: {chunk_start:.1f}s - {chunk_start + actual_chunk_duration:.1f}s")

        # Load all frames for this chunk into memory
        load_start = time.perf_counter()
        frames = _load_frames_chunk(
            file_path,
            chunk_start,
            actual_chunk_duration,
            sample_fps,
            out_width,
            out_height,
            hwaccel=hwaccel,
            src_width=width,
            src_height=height,
        )
        total_load_time += time.perf_counter() - load_start

        if frames.size == 0:
            continue

        logger.debug(f"Loaded {len(frames)} frames into memory")

        # Process optical flow for this chunk
        flow_start = time.perf_counter()
        for i in range(len(frames)):
            gray = frames[i]
            timestamp = global_frame_idx / sample_fps

            if prev_gray is not None and prev_gray.shape == gray.shape:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,  # type: ignore[arg-type]
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                # Classify motion
                motion_type, intensity, mean_x, mean_y = _classify_flow(flow)
                frame_motions.append((timestamp, motion_type, intensity, mean_x, mean_y))

            prev_gray = gray.copy()
            global_frame_idx += 1
        total_flow_time += time.perf_counter() - flow_start

    # Log timing breakdown
    logger.info(f"Motion analysis timing: decode={total_load_time:.2f}s, optical_flow={total_flow_time:.2f}s, frames={global_frame_idx}")

    if not frame_motions:
        return MotionAnalysis(
            duration=duration,
            fps=fps,
            primary_motion=MotionType.STATIC,
            segments=[],
            avg_intensity=0.0,
            is_stable=True,
        )

    # Build segments from frame motions
    segments = _build_segments(frame_motions, sample_fps=sample_fps)

    # Determine primary motion (most common or longest)
    primary_motion = _get_primary_motion(segments, duration)

    # Calculate average intensity
    avg_intensity = np.mean([m[2] for m in frame_motions])

    # Determine if video is stable (mostly static or low intensity)
    static_time = sum(s.end - s.start for s in segments if s.motion_type == MotionType.STATIC)
    is_stable = static_time > duration * 0.7 or avg_intensity < MOTION_THRESHOLD

    # Clip-level feature summaries from the full per-frame series
    overall = compute_motion_features(
        magnitudes=np.array([m[2] for m in frame_motions]),
        mean_xs=np.array([m[3] for m in frame_motions]),
        mean_ys=np.array([m[4] for m in frame_motions]),
        sample_fps=sample_fps,
    )

    logger.info(f"Motion analysis: primary={primary_motion}, segments={len(segments)}, stable={is_stable}")

    return MotionAnalysis(
        duration=duration,
        fps=fps,
        primary_motion=primary_motion,
        segments=segments,
        avg_intensity=float(avg_intensity),
        is_stable=bool(is_stable),
        magnitude_p90_overall=overall.magnitude_p90,
        jerk_max_overall=overall.jerk_max,
        hf_lf_ratio_overall=overall.hf_lf_ratio,
    )


def _classify_flow(flow: np.ndarray) -> tuple[MotionType, float, float, float]:
    """Classify motion type from optical flow field.

    Args:
        flow: Optical flow array (H, W, 2) with x and y components

    Returns:
        (motion_type, intensity, mean_x, mean_y) - the mean flow vector is
        retained for extended feature computation (direction stats).
    """
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    # Calculate flow statistics
    mean_x = float(np.mean(flow_x))
    mean_y = float(np.mean(flow_y))
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mean_magnitude = np.mean(magnitude)

    # Check if motion is significant
    if mean_magnitude < STATIC_THRESHOLD:
        return MotionType.STATIC, float(mean_magnitude), mean_x, mean_y

    # Check for zoom (divergence from center)
    h, w = flow.shape[:2]
    center_y, center_x = h // 2, w // 2

    # Create coordinate grids relative to center
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_rel = x_coords - center_x
    y_rel = y_coords - center_y

    # Normalize relative positions
    dist_from_center = np.sqrt(x_rel**2 + y_rel**2) + 1e-7
    x_norm = x_rel / dist_from_center
    y_norm = y_rel / dist_from_center

    # Compute divergence (dot product of flow with radial direction)
    divergence = np.mean(flow_x * x_norm + flow_y * y_norm)

    if abs(divergence) > ZOOM_THRESHOLD * mean_magnitude:
        if divergence > 0:
            return MotionType.PUSH_IN, float(mean_magnitude), mean_x, mean_y
        else:
            return MotionType.PULL_OUT, float(mean_magnitude), mean_x, mean_y

    # Check for pan/tilt (consistent directional flow)
    abs_mean_x = abs(mean_x)
    abs_mean_y = abs(mean_y)

    # Strong horizontal motion = pan
    if abs_mean_x > abs_mean_y and abs_mean_x > MOTION_THRESHOLD:
        ratio = abs_mean_x / (mean_magnitude + 1e-7)
        if ratio > PAN_TILT_THRESHOLD:
            if mean_x > 0:
                return (
                    MotionType.PAN_LEFT,
                    float(mean_magnitude),
                    mean_x,
                    mean_y,
                )  # Flow right = camera pans left
            else:
                return MotionType.PAN_RIGHT, float(mean_magnitude), mean_x, mean_y

    # Strong vertical motion = tilt
    if abs_mean_y > abs_mean_x and abs_mean_y > MOTION_THRESHOLD:
        ratio = abs_mean_y / (mean_magnitude + 1e-7)
        if ratio > PAN_TILT_THRESHOLD:
            if mean_y > 0:
                return MotionType.TILT_UP, float(mean_magnitude), mean_x, mean_y  # Flow down = camera tilts up
            else:
                return MotionType.TILT_DOWN, float(mean_magnitude), mean_x, mean_y

    # Significant motion but not consistent direction = handheld/complex
    if mean_magnitude > MOTION_THRESHOLD:
        return MotionType.HANDHELD, float(mean_magnitude), mean_x, mean_y

    return MotionType.STATIC, float(mean_magnitude), mean_x, mean_y


def compute_motion_features(
    magnitudes: np.ndarray,
    mean_xs: np.ndarray,
    mean_ys: np.ndarray,
    sample_fps: float,
    magnitude_mean: float | None = None,
) -> MotionFeatures:
    """Compute extended motion statistics from a per-frame mean-flow series.

    Pure post-processing on data classification already produced - deterministic
    and cheap (no model inference). See MOTION_FEATURES_VERSION for the cache key.

    Args:
        magnitudes: Per-frame mean flow magnitudes
        mean_xs: Per-frame mean flow x components
        mean_ys: Per-frame mean flow y components
        sample_fps: Sample rate of the series in Hz (needed for time derivatives and FFT)
        magnitude_mean: Override for magnitude_mean so it can stay byte-identical to the
            segment's existing intensity value (which averages differently after merges)

    Returns:
        MotionFeatures with all statistics populated
    """
    n = len(magnitudes)
    mean_mag = float(magnitude_mean) if magnitude_mean is not None else (float(np.mean(magnitudes)) if n else 0.0)

    if n < MIN_FEATURE_SAMPLES:
        # Too few samples for meaningful statistics - documented degenerate defaults
        return MotionFeatures(
            magnitude_mean=mean_mag,
            magnitude_std=0.0,
            magnitude_p90=mean_mag,
            magnitude_max=mean_mag,
            direction_consistency=1.0,
            direction_reversals_per_sec=0.0,
            acceleration_mean=0.0,
            jerk_max=0.0,
            hf_energy=0.0,
            lf_energy=1.0,
            hf_lf_ratio=0.0,
        )

    # Magnitude distribution
    magnitude_std = float(np.std(magnitudes))
    magnitude_p90 = float(np.percentile(magnitudes, 90))
    magnitude_max = float(np.max(magnitudes))

    # Direction statistics from mean-flow angles (circular statistics).
    # Consistency = resultant vector length: 1.0 for monotonic direction, ~0 for random.
    angles = np.arctan2(mean_ys, mean_xs)
    direction_consistency = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))

    # Reversals: consecutive direction changes >90 degrees (negative dot product of unit vectors)
    dx, dy = np.cos(angles), np.sin(angles)
    reversals = int(np.sum(dx[1:] * dx[:-1] + dy[1:] * dy[:-1] < 0))
    series_duration = n / sample_fps
    direction_reversals_per_sec = reversals / series_duration if series_duration > 0 else 0.0

    # Acceleration and jerk from finite differences of the smoothed magnitude series.
    # Smoothing before differentiating avoids amplifying single-frame noise.
    window = min(SMOOTH_WINDOW, n)
    smoothed = np.convolve(magnitudes, np.ones(window) / window, mode="valid")
    accel = np.diff(smoothed) * sample_fps
    acceleration_mean = float(np.mean(np.abs(accel))) if len(accel) else 0.0
    jerk = np.diff(accel) * sample_fps
    jerk_max = float(np.max(np.abs(jerk))) if len(jerk) else 0.0

    # Frequency band energy: FFT of the raw (unsmoothed) magnitude series, DC excluded.
    # HF (>FREQ_SPLIT_HZ) = shake/tremor; LF = intentional pan/tilt/dolly.
    spectrum = np.abs(np.fft.rfft(magnitudes - np.mean(magnitudes))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_fps)
    lf = float(np.sum(spectrum[(freqs > 0) & (freqs < FREQ_SPLIT_HZ)]))
    hf = float(np.sum(spectrum[freqs >= FREQ_SPLIT_HZ]))
    total = lf + hf
    if total <= 1e-12:
        hf_energy, lf_energy = 0.0, 1.0
    else:
        hf_energy, lf_energy = hf / total, lf / total
    hf_lf_ratio = hf_energy / (hf_energy + lf_energy)

    return MotionFeatures(
        magnitude_mean=mean_mag,
        magnitude_std=magnitude_std,
        magnitude_p90=magnitude_p90,
        magnitude_max=magnitude_max,
        direction_consistency=direction_consistency,
        direction_reversals_per_sec=float(direction_reversals_per_sec),
        acceleration_mean=acceleration_mean,
        jerk_max=jerk_max,
        hf_energy=hf_energy,
        lf_energy=lf_energy,
        hf_lf_ratio=hf_lf_ratio,
    )


def _build_segments(
    frame_motions: list[tuple[float, MotionType, float, float, float]],
    min_segment_duration: float = 0.5,
    sample_fps: float = 5.0,
) -> list[MotionSegment]:
    """Build motion segments from frame-by-frame analysis.

    Merges consecutive frames with same motion type into segments. Tracks index
    ranges into frame_motions so extended features can be computed over the full
    (possibly merged) per-frame series of each segment.
    """
    if not frame_motions:
        return []

    # Runs as (start_ts, end_ts, motion_type, intensity, i0, i1) with i0/i1 an
    # index range into frame_motions (i1 exclusive)
    runs: list[tuple[float, float, MotionType, float, int, int]] = []
    current_type = frame_motions[0][1]
    current_start = frame_motions[0][0]
    current_i0 = 0
    current_intensities: list[float] = [frame_motions[0][2]]

    for idx, (timestamp, motion_type, intensity, _mx, _my) in enumerate(frame_motions[1:], start=1):
        if motion_type == current_type:
            current_intensities.append(intensity)
        else:
            # End current run
            runs.append((current_start, timestamp, current_type, float(np.mean(current_intensities)), current_i0, idx))
            current_type = motion_type
            current_start = timestamp
            current_i0 = idx
            current_intensities = [intensity]

    # Add final run (extend slightly past last frame)
    runs.append((current_start, frame_motions[-1][0] + 0.2, current_type, float(np.mean(current_intensities)), current_i0, len(frame_motions)))

    # Merge short runs into their predecessor (same rule as before: type from the
    # longer side, intensity as the plain average of the two - preserved for
    # backward compatibility of the intensity field)
    merged: list[tuple[float, float, MotionType, float, int, int]] = []
    for run in runs:
        start, end, motion_type, intensity, i0, i1 = run
        if end - start < min_segment_duration and merged:
            p_start, p_end, p_type, p_intensity, p_i0, _p_i1 = merged[-1]
            merged[-1] = (
                p_start,
                end,
                p_type if p_end - p_start > end - start else motion_type,
                (p_intensity + intensity) / 2,
                p_i0,
                i1,
            )
        else:
            merged.append(run)

    # Materialize segments with extended features computed over each index range
    segments: list[MotionSegment] = []
    for start, end, motion_type, intensity, i0, i1 in merged:
        series = frame_motions[i0:i1]
        features = compute_motion_features(
            magnitudes=np.array([m[2] for m in series]),
            mean_xs=np.array([m[3] for m in series]),
            mean_ys=np.array([m[4] for m in series]),
            sample_fps=sample_fps,
            magnitude_mean=intensity,
        )
        segments.append(
            MotionSegment(
                start=start,
                end=end,
                motion_type=motion_type,
                intensity=intensity,
                features=features,
            )
        )

    return segments


def _get_primary_motion(segments: list[MotionSegment], duration: float) -> MotionType:
    """Determine the primary motion type based on segment durations."""
    if not segments:
        return MotionType.STATIC

    # Sum duration per motion type
    type_durations: dict[MotionType, float] = {}
    for seg in segments:
        seg_duration = seg.end - seg.start
        type_durations[seg.motion_type] = type_durations.get(seg.motion_type, 0) + seg_duration

    # Return type with longest total duration
    return max(type_durations, key=type_durations.get)  # type: ignore


def motion_result_to_dict(analysis: MotionAnalysis, include_features: bool = True) -> dict:
    """Convert a MotionAnalysis to the batch-response dict.

    Args:
        analysis: Motion analysis result
        include_features: When False, the extended api-1.1 feature fields are
            omitted (settings escape hatch for strict-validating consumers)

    Returns:
        Dict matching the MotionResult response schema
    """
    segments = []
    for seg in analysis.segments:
        seg_dict: dict = {
            "start": seg.start,
            "end": seg.end,
            "motion_type": seg.motion_type.value,
            "intensity": float(seg.intensity),
        }
        if include_features and seg.features is not None:
            f = seg.features
            seg_dict.update(
                {
                    "magnitude_mean": f.magnitude_mean,
                    "magnitude_std": f.magnitude_std,
                    "magnitude_p90": f.magnitude_p90,
                    "magnitude_max": f.magnitude_max,
                    "direction_consistency": f.direction_consistency,
                    "direction_reversals_per_sec": f.direction_reversals_per_sec,
                    "acceleration_mean": f.acceleration_mean,
                    "jerk_max": f.jerk_max,
                    "hf_energy": f.hf_energy,
                    "lf_energy": f.lf_energy,
                    "hf_lf_ratio": f.hf_lf_ratio,
                    "features_version": analysis.features_version,
                }
            )
        segments.append(seg_dict)

    result: dict = {
        "duration": analysis.duration,
        "fps": analysis.fps,
        "primary_motion": analysis.primary_motion.value,
        "avg_intensity": float(analysis.avg_intensity),
        "is_stable": bool(analysis.is_stable),
        "segments": segments,
    }
    if include_features:
        result.update(
            {
                "magnitude_p90_overall": analysis.magnitude_p90_overall,
                "jerk_max_overall": analysis.jerk_max_overall,
                "hf_lf_ratio_overall": analysis.hf_lf_ratio_overall,
                "features_version": analysis.features_version,
            }
        )
    return result


def get_sample_timestamps(
    motion: MotionAnalysis,
    max_samples: int = 5,
) -> list[float]:
    """Get optimal timestamps for frame sampling based on motion analysis.

    Static segments need fewer samples, moving segments need more.

    Args:
        motion: Motion analysis result
        max_samples: Maximum number of samples to return

    Returns:
        List of timestamps to sample
    """
    if not motion.segments:
        if motion.is_stable or motion.primary_motion == MotionType.STATIC:
            # Static video - just sample middle
            return [motion.duration / 2]
        else:
            # No segments - sample evenly
            return [motion.duration * i / (max_samples + 1) for i in range(1, max_samples + 1)]

    # Check if there are any non-static segments
    has_motion_segments = any(seg.motion_type not in (MotionType.STATIC, MotionType.HANDHELD) for seg in motion.segments)

    if not has_motion_segments and motion.is_stable:
        # All static/handheld - just sample middle
        return [motion.duration / 2]

    timestamps: list[float] = []

    for seg in motion.segments:
        seg_duration = seg.end - seg.start

        if seg.motion_type == MotionType.STATIC:
            # Static segment - one sample from middle
            timestamps.append((seg.start + seg.end) / 2)

        elif seg.motion_type in (
            MotionType.PAN_LEFT,
            MotionType.PAN_RIGHT,
            MotionType.TILT_UP,
            MotionType.TILT_DOWN,
        ):
            # Pan/tilt - sample start and end (different content)
            timestamps.append(seg.start + 0.2)
            if seg_duration > 1.0:
                timestamps.append(seg.end - 0.2)

        elif seg.motion_type in (MotionType.PUSH_IN, MotionType.PULL_OUT):
            # Zoom - sample at different zoom levels
            timestamps.append(seg.start + 0.2)
            if seg_duration > 1.0:
                timestamps.append(seg.end - 0.2)

        elif seg.motion_type == MotionType.HANDHELD:
            # Handheld - content is same, just one sample
            timestamps.append((seg.start + seg.end) / 2)

        else:
            # Complex/unknown - sample middle
            timestamps.append((seg.start + seg.end) / 2)

    # Remove duplicates and sort
    timestamps = sorted(set(timestamps))

    # Limit to max_samples, keeping evenly distributed
    if len(timestamps) > max_samples:
        indices = np.linspace(0, len(timestamps) - 1, max_samples, dtype=int)
        timestamps = [timestamps[i] for i in indices]

    # Ensure timestamps are within bounds
    timestamps = [max(0.1, min(t, motion.duration - 0.1)) for t in timestamps]

    logger.info(f"Smart sampling: {len(timestamps)} frames from {len(motion.segments)} motion segments")

    return timestamps


def get_adaptive_timestamps(
    motion: MotionAnalysis,
    min_fps: float = 0.1,
    max_fps: float = 2.0,
    max_samples: int = 100,
) -> list[float]:
    """Get timestamps with adaptive sampling based on motion intensity.

    SMART OPTIMIZATION: For stable/static footage, returns very few samples
    since the content doesn't change. This dramatically speeds up processing
    for tripod shots, interviews, static drone hovers, etc.

    Stability-based limits:
    - Fully stable (is_stable=True, avg_intensity < 1.0): max 3 samples
    - Mostly stable (is_stable=True): max 5 samples
    - Some motion: uses intensity-based adaptive sampling

    Intensity to FPS mapping (for non-stable segments):
    - 0-0.5:   min_fps (static, nothing changing)
    - 0.5-2.0: 0.25 fps (stable, minimal change)
    - 2.0-4.0: 0.5 fps (moderate motion)
    - 4.0-6.0: 1.0 fps (active motion)
    - 6.0+:    max_fps (high motion, rapid changes)

    Args:
        motion: Motion analysis result
        min_fps: Minimum sample rate for static content (default 0.1 = 1 per 10s)
        max_fps: Maximum sample rate for high motion (default 2.0)
        max_samples: Maximum total samples to return

    Returns:
        List of timestamps to sample
    """
    # OPTIMIZATION: Very stable footage needs minimal sampling
    if motion.is_stable and motion.avg_intensity < 1.0:
        # Extremely stable (tripod, static drone) - just 3 samples
        if motion.duration < 10:
            timestamps = [motion.duration / 2]
        else:
            # Start, middle, end
            timestamps = [
                motion.duration * 0.15,
                motion.duration * 0.5,
                motion.duration * 0.85,
            ]
        logger.info(f"Stable video optimization: {len(timestamps)} frames only (avg_intensity={motion.avg_intensity:.1f})")
        return timestamps

    if motion.is_stable:
        # Mostly stable - cap at 5 samples spread across duration
        num_samples = min(5, max(1, int(motion.duration / 10)))
        if num_samples == 1:
            timestamps = [motion.duration / 2]
        else:
            step = motion.duration / (num_samples + 1)
            timestamps = [step * (i + 1) for i in range(num_samples)]
        logger.info(f"Stable video: {len(timestamps)} frames (avg_intensity={motion.avg_intensity:.1f})")
        return timestamps

    if not motion.segments:
        # No segments but not stable - sample at moderate rate
        interval = 2.0  # 0.5 fps
        timestamps = [t for t in _frange(0.1, motion.duration - 0.1, interval)]
        timestamps = timestamps[:max_samples]
        return timestamps

    timestamps: list[float] = []

    for seg in motion.segments:
        intensity = seg.intensity

        # Map intensity to fps
        if intensity < 0.5:
            fps = min_fps
        elif intensity < 2.0:
            fps = 0.25
        elif intensity < 4.0:
            fps = 0.5
        elif intensity < 6.0:
            fps = 1.0
        else:
            fps = max_fps

        # Generate timestamps for this segment
        interval = 1.0 / fps
        t = seg.start + 0.1  # Start slightly after segment boundary
        while t < seg.end - 0.1:
            timestamps.append(t)
            t += interval

        # Always include at least one sample per segment
        if not any(seg.start <= ts <= seg.end for ts in timestamps):
            timestamps.append((seg.start + seg.end) / 2)

    # Remove duplicates and sort
    timestamps = sorted(set(timestamps))

    # Cap at max_samples, keeping even distribution
    if len(timestamps) > max_samples:
        step = len(timestamps) / max_samples
        timestamps = [timestamps[int(i * step)] for i in range(max_samples)]

    # Ensure timestamps are within video bounds
    timestamps = [max(0.1, min(t, motion.duration - 0.1)) for t in timestamps]

    logger.info(f"Adaptive sampling: {len(timestamps)} frames (avg_intensity={motion.avg_intensity:.1f}, stable={motion.is_stable})")

    return timestamps


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    while start < stop:
        yield start
        start += step

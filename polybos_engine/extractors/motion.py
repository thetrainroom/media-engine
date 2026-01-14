"""Camera motion analysis using optical flow."""

import logging
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Analysis resolution (scale down 4K for speed)
ANALYSIS_HEIGHT = 720

# Motion detection thresholds
MOTION_THRESHOLD = 2.0  # Minimum average flow magnitude for motion
PAN_TILT_THRESHOLD = 0.7  # Ratio of directional vs total flow for pan/tilt
ZOOM_THRESHOLD = 0.3  # Divergence threshold for zoom detection
STATIC_THRESHOLD = 0.5  # Below this = static


class MotionType(StrEnum):
    """Types of camera motion."""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    HANDHELD = "handheld"  # Random/shaky movement
    COMPLEX = "complex"  # Multiple motions combined


@dataclass
class MotionSegment:
    """A segment of video with consistent motion."""
    start: float
    end: float
    motion_type: MotionType
    intensity: float  # Average flow magnitude


@dataclass
class MotionAnalysis:
    """Complete motion analysis for a video."""
    duration: float
    fps: float
    primary_motion: MotionType
    segments: list[MotionSegment]
    avg_intensity: float
    is_stable: bool  # True if mostly static/tripod


def analyze_motion(
    file_path: str,
    sample_fps: float = 5.0,  # Analyze every N frames per second
) -> MotionAnalysis:
    """Analyze camera motion in a video using optical flow.

    Args:
        file_path: Path to video file
        sample_fps: How many frames per second to analyze (default 5)

    Returns:
        MotionAnalysis with motion type segments
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Calculate frame skip for desired sample rate
    frame_skip = max(1, int(fps / sample_fps))
    total_samples = frame_count // frame_skip

    logger.info(f"Analyzing motion: {duration:.1f}s @ {fps:.1f}fps, ~{total_samples} samples")

    prev_gray = None
    frame_motions: list[tuple[float, MotionType, float]] = []  # (timestamp, type, intensity)

    for sample_idx in range(total_samples):
        frame_idx = sample_idx * frame_skip
        timestamp = frame_idx / fps

        # Seek to frame (faster than reading every frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Scale down if needed
        if frame.shape[0] > ANALYSIS_HEIGHT:
            scale = ANALYSIS_HEIGHT / frame.shape[0]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Classify motion
            motion_type, intensity = _classify_flow(flow)
            frame_motions.append((timestamp, motion_type, intensity))

        prev_gray = gray

    cap.release()

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
    segments = _build_segments(frame_motions)

    # Determine primary motion (most common or longest)
    primary_motion = _get_primary_motion(segments, duration)

    # Calculate average intensity
    avg_intensity = np.mean([m[2] for m in frame_motions])

    # Determine if video is stable (mostly static or low intensity)
    static_time = sum(s.end - s.start for s in segments if s.motion_type == MotionType.STATIC)
    is_stable = static_time > duration * 0.7 or avg_intensity < MOTION_THRESHOLD

    logger.info(f"Motion analysis: primary={primary_motion}, segments={len(segments)}, stable={is_stable}")

    return MotionAnalysis(
        duration=duration,
        fps=fps,
        primary_motion=primary_motion,
        segments=segments,
        avg_intensity=avg_intensity,
        is_stable=is_stable,
    )


def _classify_flow(flow: np.ndarray) -> tuple[MotionType, float]:
    """Classify motion type from optical flow field.

    Args:
        flow: Optical flow array (H, W, 2) with x and y components

    Returns:
        (motion_type, intensity)
    """
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    # Calculate flow statistics
    mean_x = np.mean(flow_x)
    mean_y = np.mean(flow_y)
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mean_magnitude = np.mean(magnitude)

    # Check if motion is significant
    if mean_magnitude < STATIC_THRESHOLD:
        return MotionType.STATIC, mean_magnitude

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
            return MotionType.ZOOM_IN, mean_magnitude
        else:
            return MotionType.ZOOM_OUT, mean_magnitude

    # Check for pan/tilt (consistent directional flow)
    abs_mean_x = abs(mean_x)
    abs_mean_y = abs(mean_y)

    # Strong horizontal motion = pan
    if abs_mean_x > abs_mean_y and abs_mean_x > MOTION_THRESHOLD:
        ratio = abs_mean_x / (mean_magnitude + 1e-7)
        if ratio > PAN_TILT_THRESHOLD:
            if mean_x > 0:
                return MotionType.PAN_LEFT, mean_magnitude  # Flow right = camera pans left
            else:
                return MotionType.PAN_RIGHT, mean_magnitude

    # Strong vertical motion = tilt
    if abs_mean_y > abs_mean_x and abs_mean_y > MOTION_THRESHOLD:
        ratio = abs_mean_y / (mean_magnitude + 1e-7)
        if ratio > PAN_TILT_THRESHOLD:
            if mean_y > 0:
                return MotionType.TILT_UP, mean_magnitude  # Flow down = camera tilts up
            else:
                return MotionType.TILT_DOWN, mean_magnitude

    # Significant motion but not consistent direction = handheld/complex
    if mean_magnitude > MOTION_THRESHOLD:
        return MotionType.HANDHELD, mean_magnitude

    return MotionType.STATIC, mean_magnitude


def _build_segments(
    frame_motions: list[tuple[float, MotionType, float]],
    min_segment_duration: float = 0.5,
) -> list[MotionSegment]:
    """Build motion segments from frame-by-frame analysis.

    Merges consecutive frames with same motion type into segments.
    """
    if not frame_motions:
        return []

    segments: list[MotionSegment] = []
    current_type = frame_motions[0][1]
    current_start = frame_motions[0][0]
    current_intensities: list[float] = [frame_motions[0][2]]

    for timestamp, motion_type, intensity in frame_motions[1:]:
        if motion_type == current_type:
            current_intensities.append(intensity)
        else:
            # End current segment
            segments.append(MotionSegment(
                start=current_start,
                end=timestamp,
                motion_type=current_type,
                intensity=np.mean(current_intensities),
            ))
            current_type = motion_type
            current_start = timestamp
            current_intensities = [intensity]

    # Add final segment
    if frame_motions:
        segments.append(MotionSegment(
            start=current_start,
            end=frame_motions[-1][0] + 0.2,  # Extend slightly past last frame
            motion_type=current_type,
            intensity=np.mean(current_intensities),
        ))

    # Merge short segments
    merged: list[MotionSegment] = []
    for seg in segments:
        if seg.end - seg.start < min_segment_duration and merged:
            # Merge with previous segment
            prev = merged[-1]
            merged[-1] = MotionSegment(
                start=prev.start,
                end=seg.end,
                motion_type=prev.motion_type if prev.end - prev.start > seg.end - seg.start else seg.motion_type,
                intensity=(prev.intensity + seg.intensity) / 2,
            )
        else:
            merged.append(seg)

    return merged


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
    has_motion_segments = any(
        seg.motion_type not in (MotionType.STATIC, MotionType.HANDHELD)
        for seg in motion.segments
    )

    if not has_motion_segments and motion.is_stable:
        # All static/handheld - just sample middle
        return [motion.duration / 2]

    timestamps: list[float] = []

    for seg in motion.segments:
        seg_duration = seg.end - seg.start

        if seg.motion_type == MotionType.STATIC:
            # Static segment - one sample from middle
            timestamps.append((seg.start + seg.end) / 2)

        elif seg.motion_type in (MotionType.PAN_LEFT, MotionType.PAN_RIGHT,
                                  MotionType.TILT_UP, MotionType.TILT_DOWN):
            # Pan/tilt - sample start and end (different content)
            timestamps.append(seg.start + 0.2)
            if seg_duration > 1.0:
                timestamps.append(seg.end - 0.2)

        elif seg.motion_type in (MotionType.ZOOM_IN, MotionType.ZOOM_OUT):
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

    Higher motion intensity = more frequent sampling.
    Lower motion intensity = sparser sampling.

    Intensity to FPS mapping:
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
    if not motion.segments:
        # No segments - sample based on overall stability
        if motion.is_stable:
            return [motion.duration / 2]  # Single sample for stable video
        else:
            # Sample at default rate
            interval = 1.0 / 0.5
            return [t for t in _frange(0.1, motion.duration - 0.1, interval)]

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

    logger.info(
        f"Adaptive sampling: {len(timestamps)} frames "
        f"(avg_intensity={motion.avg_intensity:.1f}, stable={motion.is_stable})"
    )

    return timestamps


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    while start < stop:
        yield start
        start += step

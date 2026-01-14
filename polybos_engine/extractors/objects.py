"""Object detection using YOLO."""

import logging
from pathlib import Path

from polybos_engine.config import get_device, DeviceType
from polybos_engine.extractors.frames import FrameExtractor, get_video_duration
from polybos_engine.schemas import BoundingBox, ObjectDetection, ObjectsResult, SceneDetection

logger = logging.getLogger(__name__)


def extract_objects(
    file_path: str,
    scenes: list[SceneDetection] | None = None,
    sample_fps: float = 0.5,
    timestamps: list[float] | None = None,  # Direct timestamp list (e.g., from motion analysis)
    min_confidence: float = 0.6,
    min_size: int = 50,
    model_name: str = "yolov8m.pt",
) -> ObjectsResult:
    """Extract objects from video file.

    Sampling strategy (in priority order):
    1. If timestamps provided: use those directly (e.g., from motion-adaptive sampling)
    2. If scenes provided: sample at scene boundaries + intervals within scenes
    3. Otherwise: sample at fixed fps

    Args:
        file_path: Path to video file
        scenes: Optional scene boundaries for smarter sampling
        sample_fps: Frame sampling rate (0.5 = every 2 seconds)
        timestamps: Optional list of specific timestamps to sample (overrides scenes/fps)
        min_confidence: Minimum detection confidence (0.6 recommended)
        min_size: Minimum object size in pixels (filters noise)
        model_name: YOLO model (yolov8m.pt recommended for accuracy)

    Returns:
        ObjectsResult with unique objects and summary
    """
    from ultralytics import YOLO

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Determine device for GPU acceleration
    device = get_device()
    device_str = "mps" if device == DeviceType.MPS else "cuda" if device == DeviceType.CUDA else "cpu"
    logger.info(f"Loading object detection model: {model_name} on {device_str}")

    # Load model
    model = YOLO(model_name)

    # Determine timestamps to sample (priority: explicit > scenes > fixed fps)
    if timestamps is not None:
        sample_timestamps = sorted(set(timestamps))
        logger.info(f"Using {len(sample_timestamps)} provided timestamps for object detection")
    elif scenes:
        sample_timestamps = _get_timestamps_from_scenes(scenes, sample_fps)
        logger.info(f"Sampling {len(sample_timestamps)} frames from {len(scenes)} scenes")
    else:
        duration = get_video_duration(file_path)
        interval = 1.0 / sample_fps
        sample_timestamps = list(_frange(0, duration, interval))
        logger.info(f"Sampling {len(sample_timestamps)} frames at {sample_fps} fps")

    # Extract frames using OpenCV (much faster than ffmpeg per-frame)
    raw_detections: list[ObjectDetection] = []

    with FrameExtractor(file_path) as extractor:
        for timestamp in sample_timestamps:
            frame = extractor.get_frame_at(timestamp)
            if frame is None:
                continue

            # Run YOLO on frame (with GPU if available)
            try:
                results = model(frame, verbose=False, device=device_str)

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for i in range(len(boxes)):
                        confidence = float(boxes.conf[i])
                        if confidence < min_confidence:
                            continue

                        # Get bounding box
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        # Filter small detections
                        if width < min_size or height < min_size:
                            continue

                        # Get class label
                        class_id = int(boxes.cls[i])
                        label = model.names[class_id] if model.names else str(class_id)

                        raw_detections.append(ObjectDetection(
                            timestamp=round(timestamp, 2),
                            label=label,
                            confidence=round(confidence, 3),
                            bbox=BoundingBox(
                                x=int(x1), y=int(y1),
                                width=width, height=height,
                            ),
                        ))

            except Exception as e:
                logger.warning(f"Failed to process frame at {timestamp}s: {e}")

    # Deduplicate - track unique objects
    unique_detections, summary = _deduplicate_objects(raw_detections)

    logger.info(
        f"Detected {len(raw_detections)} objects, "
        f"{len(unique_detections)} unique across {len(summary)} types"
    )

    return ObjectsResult(
        summary=summary,
        detections=unique_detections,
    )


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    while start < stop:
        yield start
        start += step


def _get_timestamps_from_scenes(
    scenes: list[SceneDetection], sample_fps: float
) -> list[float]:
    """Generate sample timestamps from scenes."""
    timestamps: list[float] = []
    interval = 1.0 / sample_fps

    for scene in scenes:
        timestamps.append(scene.start)
        t = scene.start + interval
        while t < scene.end - 0.5:
            timestamps.append(t)
            t += interval

    return sorted(set(timestamps))


def _bbox_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate IoU of two bounding boxes."""
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _deduplicate_objects(
    detections: list[ObjectDetection],
    max_time_gap: float = 5.0,
    min_iou: float = 0.3,
) -> tuple[list[ObjectDetection], dict[str, int]]:
    """Deduplicate objects using position tracking.

    Groups detections of same object type that overlap across frames.
    Returns unique objects (best detection per tracked object).
    """
    if not detections:
        return [], {}

    # Group by label first
    by_label: dict[str, list[ObjectDetection]] = {}
    for det in detections:
        if det.label not in by_label:
            by_label[det.label] = []
        by_label[det.label].append(det)

    unique_objects: list[ObjectDetection] = []
    summary: dict[str, int] = {}

    for label, label_dets in by_label.items():
        # Sort by timestamp
        sorted_dets = sorted(label_dets, key=lambda d: d.timestamp)

        # Track unique instances of this object type
        tracked: list[list[ObjectDetection]] = []

        for det in sorted_dets:
            matched_track = None

            # Find matching track (same position in recent frames)
            for track_idx, track in enumerate(tracked):
                last_det = track[-1]
                time_diff = det.timestamp - last_det.timestamp

                if time_diff <= max_time_gap:
                    iou = _bbox_iou(det.bbox, last_det.bbox)
                    if iou >= min_iou:
                        matched_track = track_idx
                        break

            if matched_track is not None:
                tracked[matched_track].append(det)
            else:
                tracked.append([det])

        # Keep best detection per track
        for track in tracked:
            best = max(track, key=lambda d: d.confidence)
            unique_objects.append(best)

        summary[label] = len(tracked)

    # Sort by timestamp
    unique_objects.sort(key=lambda d: d.timestamp)

    return unique_objects, summary

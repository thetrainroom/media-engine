"""Object detection using YOLO."""

import gc
import logging
from pathlib import Path
from typing import Any

from media_engine.config import DeviceType, get_device
from media_engine.extractors.frame_buffer import SharedFrameBuffer
from media_engine.schemas import (
    BoundingBox,
    ObjectDetection,
    ObjectsResult,
)

logger = logging.getLogger(__name__)

# Singleton YOLO model (lazy loaded)
_yolo_model: Any = None
_yolo_model_name: str | None = None


def unload_yolo_model() -> None:
    """Unload the YOLO model to free memory."""
    global _yolo_model, _yolo_model_name

    if _yolo_model is None:
        return

    logger.info("Unloading YOLO model to free memory")

    try:
        # Clear CUDA/MPS cache
        import torch

        del _yolo_model
        _yolo_model = None
        _yolo_model_name = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        gc.collect()
        logger.info("YOLO model unloaded")
    except Exception as e:
        logger.warning(f"Error unloading YOLO model: {e}")
        _yolo_model = None
        _yolo_model_name = None


def _get_yolo_model(model_name: str) -> Any:
    """Get or create the YOLO model (singleton with model switching)."""
    global _yolo_model, _yolo_model_name

    # If model name changed, unload old model
    if _yolo_model is not None and _yolo_model_name != model_name:
        logger.info(f"Switching YOLO model from {_yolo_model_name} to {model_name}")
        unload_yolo_model()

    if _yolo_model is None:
        from ultralytics import YOLO  # type: ignore[import-not-found]

        logger.info(f"Loading YOLO model: {model_name}")
        _yolo_model = YOLO(model_name)
        _yolo_model_name = model_name

    return _yolo_model


def extract_objects(
    file_path: str,
    frame_buffer: SharedFrameBuffer,
    min_confidence: float = 0.6,
    min_size: int = 50,
    model_name: str = "yolov8m.pt",
) -> ObjectsResult:
    """Extract objects from video frames using YOLO.

    Args:
        file_path: Path to video file (used for logging)
        frame_buffer: Pre-decoded frames from SharedFrameBuffer
        min_confidence: Minimum detection confidence (0.6 recommended)
        min_size: Minimum object size in pixels (filters noise)
        model_name: YOLO model (yolov8m.pt recommended for accuracy)

    Returns:
        ObjectsResult with unique objects and summary
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Determine device for GPU acceleration
    device = get_device()
    device_str = "mps" if device == DeviceType.MPS else "cuda" if device == DeviceType.CUDA else "cpu"

    # Load model (singleton)
    model = _get_yolo_model(model_name)

    # Process frames from shared buffer
    raw_detections: list[ObjectDetection] = []

    logger.info(f"Processing {len(frame_buffer.frames)} frames for object detection")
    for ts in sorted(frame_buffer.frames.keys()):
        shared_frame = frame_buffer.frames[ts]
        try:
            results = model(shared_frame.bgr, verbose=False, device=device_str)

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

                    raw_detections.append(
                        ObjectDetection(
                            timestamp=round(ts, 2),
                            label=label,
                            confidence=round(confidence, 3),
                            bbox=BoundingBox(
                                x=int(x1),
                                y=int(y1),
                                width=width,
                                height=height,
                            ),
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to process frame at {ts}s: {e}")

    # Deduplicate - track unique objects
    unique_detections, summary = _deduplicate_objects(raw_detections)

    logger.info(f"Detected {len(raw_detections)} objects, {len(unique_detections)} unique across {len(summary)} types")

    return ObjectsResult(
        summary=summary,
        detections=unique_detections,
    )


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

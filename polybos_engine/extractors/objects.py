"""Object detection using YOLO/RT-DETR."""

import logging
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from polybos_engine.schemas import BoundingBox, ObjectDetection, ObjectsResult

logger = logging.getLogger(__name__)


def extract_objects(
    file_path: str,
    sample_fps: float = 2.0,
    min_confidence: float = 0.5,
    model_name: str = "yolov8n.pt",
) -> ObjectsResult:
    """Extract objects from video file.

    Args:
        file_path: Path to video file
        sample_fps: Frame sampling rate
        min_confidence: Minimum detection confidence
        model_name: YOLO model name

    Returns:
        ObjectsResult with detected objects and summary
    """
    from ultralytics import YOLO

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Load model
    logger.info(f"Loading object detection model: {model_name}")
    model = YOLO(model_name)

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_objects_")

    try:
        # Extract frames at specified fps
        logger.info(f"Extracting frames at {sample_fps} fps")
        frame_paths = _extract_frames(file_path, temp_dir, sample_fps)

        detections: list[ObjectDetection] = []
        label_counter: Counter[str] = Counter()

        for frame_path in frame_paths:
            # Get timestamp from frame filename
            timestamp = _get_timestamp_from_frame(frame_path, sample_fps)

            try:
                # Run detection
                results = model(frame_path, verbose=False)

                for result in results:
                    boxes = result.boxes

                    if boxes is None:
                        continue

                    for i in range(len(boxes)):
                        confidence = float(boxes.conf[i])

                        if confidence < min_confidence:
                            continue

                        # Get class label
                        class_id = int(boxes.cls[i])
                        label = model.names[class_id] if model.names else str(class_id)

                        # Get bounding box (xyxy format)
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                        detection = ObjectDetection(
                            timestamp=timestamp,
                            label=label,
                            confidence=round(confidence, 3),
                            bbox=BoundingBox(
                                x=int(x1),
                                y=int(y1),
                                width=int(x2 - x1),
                                height=int(y2 - y1),
                            ),
                        )
                        detections.append(detection)
                        label_counter[label] += 1

            except Exception as e:
                logger.warning(f"Failed to process frame {frame_path}: {e}")
                continue

        logger.info(f"Detected {len(detections)} objects across {len(frame_paths)} frames")

        return ObjectsResult(
            summary=dict(label_counter),
            detections=detections,
        )

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_frames(video_path: str, output_dir: str, fps: float) -> list[str]:
    """Extract frames from video at specified fps."""
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        output_pattern,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract frames: {e.stderr}")
        raise RuntimeError(f"Failed to extract frames: {e.stderr}")

    # Get list of extracted frames
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def _get_timestamp_from_frame(frame_path: str, fps: float) -> float:
    """Get timestamp in seconds from frame filename."""
    filename = Path(frame_path).stem
    frame_num = int(filename.split("_")[1])
    return (frame_num - 1) / fps

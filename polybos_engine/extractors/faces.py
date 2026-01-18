"""Face detection using DeepFace with Facenet."""

import base64
import gc
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

from polybos_engine.extractors.frames import FrameExtractor
from polybos_engine.schemas import (
    BoundingBox,
    FaceDetection,
    FacesResult,
    MediaType,
    SceneDetection,
    get_media_type,
)

Embedding: TypeAlias = list[float]

logger = logging.getLogger(__name__)


def unload_face_model() -> None:
    """Unload DeepFace models to free memory.

    DeepFace caches models internally. This function clears those caches.
    """
    import sys

    # Only unload if deepface was actually imported (avoid importing during shutdown)
    if "deepface" not in sys.modules:
        return

    logger.info("Unloading face detection models to free memory")

    try:
        import torch

        # DeepFace caches models in deepface.modules.modeling
        try:
            from deepface.modules import modeling  # type: ignore[import-not-found]

            # Clear the model store if it exists
            if hasattr(modeling, "model_obj"):
                modeling.model_obj = {}
        except (ImportError, AttributeError):
            pass

        # Also try the older DeepFace.commons.functions cache
        try:
            from deepface.commons import functions  # type: ignore[import-not-found]

            if hasattr(functions, "model_obj"):
                functions.model_obj = {}
        except (ImportError, AttributeError):
            pass

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
        logger.info("Face detection models unloaded")
    except Exception as e:
        logger.warning(f"Error unloading face models: {e}")


def extract_faces(
    file_path: str,
    scenes: list[SceneDetection] | None = None,
    sample_fps: float = 0.5,  # Default: every 2 seconds
    timestamps: (
        list[float] | None
    ) = None,  # Direct timestamp list (e.g., from YOLO persons)
    min_face_size: int = 80,
    min_confidence: float = 0.5,  # Lowered from 0.9 - user can discard false positives
    extract_images: bool = True,
    face_image_size: int = 160,  # Output face thumbnail size
) -> FacesResult:
    """Extract faces from video file.

    Sampling strategy (in priority order):
    1. If timestamps provided: use those directly (e.g., from YOLO person detections)
    2. If scenes provided: sample at scene boundaries + every sample_interval within scenes
    3. Otherwise: sample at fixed fps

    Args:
        file_path: Path to video file
        scenes: Optional scene boundaries for smarter sampling
        sample_fps: Frame sampling rate (e.g., 0.5 = every 2 seconds)
        timestamps: Optional list of specific timestamps to sample (overrides scenes/fps)
        min_face_size: Minimum face size in pixels
        min_confidence: Minimum detection confidence
        extract_images: Whether to extract face thumbnail images
        face_image_size: Size of output face thumbnails (square)

    Returns:
        FacesResult with detected faces, embeddings, and optional images
    """
    from deepface import DeepFace  # type: ignore[import-not-found]

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_faces_")

    try:
        # Determine sample timestamps (priority: explicit > scenes > fixed fps)
        if timestamps is not None:
            sample_timestamps = sorted(set(timestamps))
            logger.info(
                f"Using {len(sample_timestamps)} provided timestamps for face detection"
            )
        elif scenes:
            sample_timestamps = _get_sample_timestamps_from_scenes(scenes, sample_fps)
            logger.info(
                f"Sampling {len(sample_timestamps)} frames from {len(scenes)} scenes"
            )
        else:
            # Get video duration and sample at fixed intervals
            duration = _get_video_duration(file_path)
            if duration == 0:
                # Image or zero-duration file: use single timestamp at 0
                sample_timestamps = [0.0]
                logger.info("Using single timestamp for image/zero-duration file")
            else:
                interval = 1.0 / sample_fps
                sample_timestamps = [float(t) for t in np.arange(0, duration, interval)]
                logger.info(f"Sampling {len(sample_timestamps)} frames at {sample_fps} fps")

        # Extract frames at specific timestamps
        frame_paths = _extract_frames_at_timestamps(
            file_path, temp_dir, sample_timestamps
        )

        detections: list[FaceDetection] = []
        all_embeddings: list[Embedding] = []

        for frame_path, timestamp in zip(frame_paths, sample_timestamps):
            if not os.path.exists(frame_path):
                continue

            try:
                # Load frame for cropping
                frame_img = Image.open(frame_path)

                # Detect faces in frame
                faces = DeepFace.extract_faces(
                    img_path=frame_path,
                    detector_backend="retinaface",
                    enforce_detection=False,
                    align=True,
                )

                for face in faces:
                    # Skip low confidence detections
                    confidence = face.get("confidence", 0)
                    if confidence < min_confidence:
                        continue

                    # Get bounding box
                    region: dict[str, int] = face.get("facial_area", {})
                    x, y = region.get("x", 0), region.get("y", 0)
                    w, h = region.get("w", 0), region.get("h", 0)

                    # Skip small faces
                    if w < min_face_size or h < min_face_size:
                        continue

                    # Crop face with padding for better embedding
                    face_crop = _crop_face_with_padding(
                        frame_img, x, y, w, h, padding=0.3
                    )

                    # Generate embedding from cropped face
                    embedding: Embedding = []
                    try:
                        # Save crop temporarily for DeepFace
                        crop_path = os.path.join(temp_dir, "temp_crop.jpg")
                        face_crop.save(crop_path, "JPEG", quality=95)

                        embedding_result = DeepFace.represent(
                            img_path=crop_path,
                            model_name="Facenet512",
                            detector_backend="skip",  # Already cropped
                            enforce_detection=False,
                        )

                        if embedding_result and len(embedding_result) > 0:
                            first_result = embedding_result[0]
                            if isinstance(first_result, dict):
                                embedding = first_result.get("embedding", [])
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")

                    # Create face thumbnail
                    image_base64: str | None = None
                    if extract_images:
                        image_base64 = _encode_face_image(face_crop, face_image_size)

                    detection = FaceDetection(
                        timestamp=round(float(timestamp), 2),
                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        confidence=round(float(confidence), 3),
                        embedding=embedding,
                        image_base64=image_base64,
                    )
                    detections.append(detection)

                    if embedding:
                        all_embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to process frame {frame_path}: {e}")
                continue

        # Get frame size for edge detection
        frame_size: tuple[int, int] | None = None
        if frame_paths and os.path.exists(frame_paths[0]):
            with Image.open(frame_paths[0]) as img:
                frame_size = img.size  # (width, height)

        # Cluster faces and keep best per person
        unique_faces, unique_estimate = _deduplicate_faces(
            detections, all_embeddings, frame_size=frame_size
        )

        needs_review = sum(1 for f in unique_faces if f.needs_review)
        logger.info(
            f"Detected {len(detections)} faces, {unique_estimate} unique, "
            f"{needs_review} need review"
        )

        return FacesResult(
            count=len(detections),
            unique_estimate=unique_estimate,
            detections=unique_faces,  # Only return deduplicated faces
        )

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_video_duration(file_path: str) -> float:
    """Get video/image duration in seconds (0 for images)."""
    from polybos_engine.extractors.frames import get_video_duration

    return get_video_duration(file_path)


def _get_sample_timestamps_from_scenes(
    scenes: list[SceneDetection], sample_fps: float
) -> list[float]:
    """Generate sample timestamps from scenes.

    Samples at:
    - Start of each scene (scene boundary)
    - Every 1/sample_fps seconds within each scene
    """
    timestamps: list[float] = []
    interval = 1.0 / sample_fps  # e.g., 0.5 fps = every 2 seconds

    for scene in scenes:
        # Always sample at scene start
        timestamps.append(scene.start)

        # Sample within scene at regular intervals
        t = scene.start + interval
        while t < scene.end - 0.5:  # Stop 0.5s before scene end
            timestamps.append(t)
            t += interval

    # Remove duplicates and sort
    timestamps = sorted(set(timestamps))
    return timestamps


def _extract_frames_at_timestamps(
    file_path: str, output_dir: str, timestamps: list[float]
) -> list[str]:
    """Extract frames at specific timestamps.

    Uses FrameExtractor which handles both videos (via OpenCV/ffmpeg)
    and images (via direct loading).
    """
    import cv2

    frame_paths: list[str] = []

    with FrameExtractor(file_path) as extractor:
        for i, ts in enumerate(timestamps):
            output_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
            frame = extractor.get_frame_at(ts)

            if frame is not None:
                # Save frame as JPEG
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    frame_paths.append(output_path)
                else:
                    logger.warning(f"Frame at {ts}s: could not save to {output_path}")
                    frame_paths.append("")
            else:
                logger.warning(f"Frame at {ts}s: extraction failed")
                frame_paths.append("")

    return frame_paths


def _crop_face_with_padding(
    img: Image.Image, x: int, y: int, w: int, h: int, padding: float = 0.3
) -> Image.Image:
    """Crop face region with padding for better context.

    Args:
        img: Source image
        x, y, w, h: Face bounding box
        padding: Padding as fraction of face size (0.3 = 30%)

    Returns:
        Cropped face image
    """
    img_w, img_h = img.size

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return img.crop((x1, y1, x2, y2))


def _encode_face_image(face_img: Image.Image, size: int) -> str:
    """Resize and encode face image as base64 JPEG."""
    # Resize to square thumbnail
    face_img = face_img.resize((size, size), Image.Resampling.LANCZOS)

    # Encode as JPEG base64
    buffer = io.BytesIO()
    face_img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def _bbox_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
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


def _embedding_distance(emb1: Embedding, emb2: Embedding) -> float:
    """Compute cosine distance between two embeddings."""
    a = np.array(emb1)
    b = np.array(emb2)
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _find_position_match(
    det: FaceDetection,
    persons: list[list[tuple[int, Embedding]]],
    detections: list[FaceDetection],
    max_time_gap: float,
    min_iou: float,
) -> int | None:
    """Find matching person by bbox position in recent frames."""
    for person_idx, person_dets in enumerate(persons):
        for prev_det_idx, _ in reversed(person_dets):
            prev_det = detections[prev_det_idx]
            time_diff = det.timestamp - prev_det.timestamp

            if time_diff <= max_time_gap:
                iou = _bbox_iou(det.bbox, prev_det.bbox)
                if iou >= min_iou:
                    return person_idx
            else:
                break  # Too old
    return None


def _find_embedding_match(
    emb: Embedding,
    persons: list[list[tuple[int, Embedding]]],
    threshold: float,
) -> int | None:
    """Find matching person by embedding similarity."""
    if not emb:
        return None

    best_dist = float("inf")
    best_person = None

    for person_idx, person_dets in enumerate(persons):
        for _, prev_emb in person_dets:
            if prev_emb:
                dist = _embedding_distance(emb, prev_emb)
                if dist < best_dist and dist < threshold:
                    best_dist = dist
                    best_person = person_idx

    return best_person


def _is_near_edge(
    bbox: BoundingBox, frame_width: int, frame_height: int, margin: float = 0.05
) -> bool:
    """Check if bbox is near frame edge (partially out of frame)."""
    margin_x = int(frame_width * margin)
    margin_y = int(frame_height * margin)

    return (
        bbox.x < margin_x
        or bbox.y < margin_y
        or bbox.x + bbox.width > frame_width - margin_x
        or bbox.y + bbox.height > frame_height - margin_y
    )


def _select_best_faces(
    persons: list[list[tuple[int, Embedding]]],
    detections: list[FaceDetection],
    frame_size: tuple[int, int] | None = None,
) -> list[FaceDetection]:
    """Select best face (highest confidence) per person and flag uncertain ones."""
    result: list[FaceDetection] = []

    for person_dets in persons:
        det_indices = [idx for idx, _ in person_dets]
        best_idx = max(det_indices, key=lambda i: detections[i].confidence)
        face = detections[best_idx].model_copy()

        # Flag for review if uncertain
        reasons: list[str] = []

        # Check if near frame edge
        if frame_size:
            if _is_near_edge(face.bbox, frame_size[0], frame_size[1]):
                reasons.append("near_edge")

        # Check if low confidence
        if face.confidence < 0.95:
            reasons.append("low_confidence")

        # Check if only one detection for this person (no tracking confirmation)
        if len(person_dets) == 1:
            reasons.append("single_detection")

        if reasons:
            face.needs_review = True
            face.review_reason = ", ".join(reasons)

        result.append(face)

    result.sort(key=lambda d: d.timestamp)
    return result


def _deduplicate_faces(
    detections: list[FaceDetection],
    embeddings: list[Embedding],
    frame_size: tuple[int, int] | None = None,
    max_time_gap: float = 5.0,
    min_iou: float = 0.2,
    embedding_threshold: float = 0.5,
) -> tuple[list[FaceDetection], int]:
    """Deduplicate faces using position tracking + embedding fallback.

    Strategy:
    1. Try position-based matching (bbox overlap within time window)
    2. Fall back to embedding similarity for non-adjacent detections
    """
    if not detections:
        return [], 0

    if len(detections) == 1:
        # Single detection - flag for review
        face = detections[0].model_copy()
        face.needs_review = True
        face.review_reason = "single_detection"
        return [face], 1

    # Sort by timestamp
    sorted_dets = sorted(enumerate(detections), key=lambda x: x[1].timestamp)

    # Track persons: list of (detection_idx, embedding) per person
    persons: list[list[tuple[int, Embedding]]] = []

    for det_idx, det in sorted_dets:
        emb = embeddings[det_idx] if det_idx < len(embeddings) else []

        # Try position match first, then embedding match
        match = _find_position_match(det, persons, detections, max_time_gap, min_iou)
        if match is None:
            match = _find_embedding_match(emb, persons, embedding_threshold)

        if match is not None:
            persons[match].append((det_idx, emb))
        else:
            persons.append([(det_idx, emb)])

    result = _select_best_faces(persons, detections, frame_size)
    return result, len(persons)

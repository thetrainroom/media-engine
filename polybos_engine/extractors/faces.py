"""Face detection using DeepFace with Facenet."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from polybos_engine.config import get_settings
from polybos_engine.schemas import BoundingBox, FaceDetection, FacesResult

logger = logging.getLogger(__name__)


def extract_faces(
    file_path: str,
    sample_fps: float = 1.0,
    min_face_size: int = 80,
    min_confidence: float = 0.9,
) -> FacesResult:
    """Extract faces from video file.

    Args:
        file_path: Path to video file
        sample_fps: Frame sampling rate
        min_face_size: Minimum face size in pixels
        min_confidence: Minimum detection confidence

    Returns:
        FacesResult with detected faces and embeddings
    """
    from deepface import DeepFace

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    settings = get_settings()

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_faces_")

    try:
        # Extract frames at specified fps
        logger.info(f"Extracting frames at {sample_fps} fps")
        frame_paths = _extract_frames(file_path, temp_dir, sample_fps)

        detections = []
        all_embeddings = []

        for frame_path in frame_paths:
            # Get timestamp from frame filename
            timestamp = _get_timestamp_from_frame(frame_path, sample_fps)

            try:
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
                    region = face.get("facial_area", {})
                    x, y = region.get("x", 0), region.get("y", 0)
                    w, h = region.get("w", 0), region.get("h", 0)

                    # Skip small faces
                    if w < min_face_size or h < min_face_size:
                        continue

                    # Generate embedding
                    try:
                        embedding_result = DeepFace.represent(
                            img_path=frame_path,
                            model_name="Facenet512",
                            detector_backend="skip",  # Already have face region
                            enforce_detection=False,
                        )

                        if embedding_result:
                            embedding = embedding_result[0].get("embedding", [])
                        else:
                            embedding = []
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")
                        embedding = []

                    detection = FaceDetection(
                        timestamp=timestamp,
                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        confidence=confidence,
                        embedding=embedding,
                    )
                    detections.append(detection)

                    if embedding:
                        all_embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to process frame {frame_path}: {e}")
                continue

        # Estimate unique faces by clustering embeddings
        unique_estimate = _estimate_unique_faces(all_embeddings)

        logger.info(f"Detected {len(detections)} faces, ~{unique_estimate} unique")

        return FacesResult(
            count=len(detections),
            unique_estimate=unique_estimate,
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
    # Frame filename format: frame_000001.jpg (1-indexed)
    filename = Path(frame_path).stem
    frame_num = int(filename.split("_")[1])
    return (frame_num - 1) / fps


def _estimate_unique_faces(embeddings: list[list[float]], threshold: float = 0.6) -> int:
    """Estimate number of unique faces by clustering embeddings.

    Uses simple distance-based clustering.
    """
    if not embeddings:
        return 0

    embeddings_array = np.array(embeddings)

    # Simple clustering: group faces within threshold distance
    clusters = []

    for emb in embeddings_array:
        found_cluster = False
        for cluster in clusters:
            # Calculate distance to cluster centroid
            centroid = np.mean(cluster, axis=0)
            distance = np.linalg.norm(emb - centroid)

            if distance < threshold:
                cluster.append(emb)
                found_cluster = True
                break

        if not found_cluster:
            clusters.append([emb])

    return len(clusters)

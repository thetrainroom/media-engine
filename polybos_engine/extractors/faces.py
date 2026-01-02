"""Face detection using DeepFace with Facenet."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from polybos_engine.schemas import BoundingBox, FaceDetection, FacesResult, SceneDetection

Embedding: TypeAlias = list[float]

logger = logging.getLogger(__name__)


def extract_faces(
    file_path: str,
    scenes: list[SceneDetection] | None = None,
    sample_fps: float = 1.0,
    min_face_size: int = 80,
    min_confidence: float = 0.9,
) -> FacesResult:
    """Extract faces from video file.

    Args:
        file_path: Path to video file
        scenes: Optional scene boundaries. If provided, samples from each scene
            instead of fixed fps. Recommended for better performance and accuracy.
        sample_fps: Frame sampling rate (used only if scenes not provided)
        min_face_size: Minimum face size in pixels
        min_confidence: Minimum detection confidence

    Returns:
        FacesResult with detected faces and embeddings
    """
    from deepface import DeepFace

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_faces_")

    try:
        # Extract frames - scene-based or fps-based
        if scenes:
            logger.info(f"Extracting frames from {len(scenes)} scenes")
            frame_paths, timestamps = _extract_scene_frames(file_path, temp_dir, scenes)
        else:
            logger.info(f"Extracting frames at {sample_fps} fps")
            frame_paths = _extract_frames(file_path, temp_dir, sample_fps)
            timestamps = None  # Will be calculated from frame filename

        detections: list[FaceDetection] = []
        all_embeddings: list[Embedding] = []

        for i, frame_path in enumerate(frame_paths):
            # Get timestamp - from scene extraction or calculate from filename
            if timestamps is not None:
                timestamp = timestamps[i]
            else:
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
                    region: dict[str, int] = face.get("facial_area", {})
                    x, y = region.get("x", 0), region.get("y", 0)
                    w, h = region.get("w", 0), region.get("h", 0)

                    # Skip small faces
                    if w < min_face_size or h < min_face_size:
                        continue

                    # Generate embedding
                    embedding: Embedding = []
                    try:
                        embedding_result = DeepFace.represent(
                            img_path=frame_path,
                            model_name="Facenet512",
                            detector_backend="skip",  # Already have face region
                            enforce_detection=False,
                        )

                        if embedding_result and len(embedding_result) > 0:
                            first_result = embedding_result[0]
                            if isinstance(first_result, dict):
                                embedding = first_result.get("embedding", [])
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")

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


def _extract_scene_frames(
    video_path: str, output_dir: str, scenes: list[SceneDetection]
) -> tuple[list[str], list[float]]:
    """Extract one frame from the midpoint of each scene.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        scenes: List of scene detections with start/end times

    Returns:
        Tuple of (frame_paths, timestamps)
    """
    frame_paths: list[str] = []
    timestamps: list[float] = []

    for i, scene in enumerate(scenes):
        # Calculate midpoint of scene
        midpoint = (scene.start + scene.end) / 2
        output_path = os.path.join(output_dir, f"scene_{i:04d}.jpg")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(midpoint),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if os.path.exists(output_path):
                frame_paths.append(output_path)
                timestamps.append(midpoint)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract frame for scene {i}: {e.stderr}")

    logger.info(f"Extracted {len(frame_paths)} scene frames")
    return frame_paths, timestamps


def _get_timestamp_from_frame(frame_path: str, fps: float) -> float:
    """Get timestamp in seconds from frame filename."""
    # Frame filename format: frame_000001.jpg (1-indexed)
    filename = Path(frame_path).stem
    frame_num = int(filename.split("_")[1])
    return (frame_num - 1) / fps


def _estimate_unique_faces(
    embeddings: list[Embedding], distance_threshold: float = 0.25
) -> int:
    """Estimate number of unique faces by clustering embeddings.

    Uses Agglomerative Clustering with cosine distance, which is more robust
    for face embeddings than Euclidean distance. FaceNet512 embeddings are
    normalized, so cosine distance works well for comparing face similarity.

    Args:
        embeddings: List of face embeddings (512-dim vectors from FaceNet512)
        distance_threshold: Maximum cosine distance to consider same person.
            Lower = stricter matching. Typical range: 0.3-0.5 for FaceNet.

    Returns:
        Estimated number of unique faces
    """
    if not embeddings:
        return 0

    if len(embeddings) == 1:
        return 1

    from sklearn.cluster import AgglomerativeClustering

    embeddings_array: NDArray[np.float64] = np.array(embeddings)

    # Agglomerative clustering with cosine distance
    # distance_threshold determines when to stop merging clusters
    clustering = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]  # sklearn accepts None with distance_threshold
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )

    labels = clustering.fit_predict(embeddings_array)

    return len(set(labels))

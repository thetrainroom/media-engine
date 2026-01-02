"""Scene detection using PySceneDetect."""

import logging
from pathlib import Path

from polybos_engine.schemas import SceneDetection, ScenesResult

logger = logging.getLogger(__name__)


def extract_scenes(file_path: str, threshold: float = 27.0) -> ScenesResult:
    """Detect scene boundaries in video file.

    Args:
        file_path: Path to video file
        threshold: Content detector threshold (lower = more sensitive)

    Returns:
        ScenesResult with detected scene boundaries
    """
    from scenedetect import ContentDetector, detect

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    logger.info(f"Detecting scenes in {file_path}")

    # Detect scenes using ContentDetector
    scenes = detect(file_path, ContentDetector(threshold=threshold))

    detections = []
    for i, (start_time, end_time) in enumerate(scenes):
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        detections.append(
            SceneDetection(
                index=i,
                start=start_sec,
                end=end_sec,
                duration=round(end_sec - start_sec, 3),
            )
        )

    logger.info(f"Detected {len(detections)} scenes")

    return ScenesResult(count=len(detections), detections=detections)

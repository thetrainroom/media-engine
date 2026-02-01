"""Scene detection using PySceneDetect."""

import logging
from pathlib import Path

from media_engine.schemas import SceneDetection, ScenesResult

logger = logging.getLogger(__name__)

# Resolution thresholds for frame skipping (pixel count)
RES_1080P = 1920 * 1080  # ~2M pixels
RES_4K = 3840 * 2160  # ~8.3M pixels
RES_5K = 5120 * 2880  # ~14.7M pixels


def extract_scenes(file_path: str, threshold: float = 27.0) -> ScenesResult:
    """Detect scene boundaries in video file.

    For high-resolution videos (4K+), uses frame skipping to improve performance.

    Args:
        file_path: Path to video file
        threshold: Content detector threshold (lower = more sensitive)

    Returns:
        ScenesResult with detected scene boundaries
    """
    from scenedetect import (  # type: ignore[import-not-found]
        ContentDetector,
        SceneManager,
        open_video,
    )

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    logger.info(f"Detecting scenes in {file_path}")

    # Open video to get resolution
    video = open_video(file_path)
    width, height = video.frame_size
    pixels = width * height

    # Determine frame skip based on resolution
    # Higher resolution = more frame skip for speed
    if pixels > RES_5K:
        frame_skip = 4  # Process every 5th frame for 5K+
        logger.info(f"High-res video ({width}x{height}), using frame_skip=4")
    elif pixels > RES_4K:
        frame_skip = 2  # Process every 3rd frame for 4K+
        logger.info(f"4K video ({width}x{height}), using frame_skip=2")
    elif pixels > RES_1080P:
        frame_skip = 1  # Process every 2nd frame for >1080p
        logger.info(f"High-res video ({width}x{height}), using frame_skip=1")
    else:
        frame_skip = 0  # Process every frame for 1080p and below

    # Use SceneManager API for frame_skip support
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Detect scenes with frame skipping
    scene_manager.detect_scenes(video, frame_skip=frame_skip)
    scenes = scene_manager.get_scene_list()

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

"""Object detection using Qwen2-VL vision-language model."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from polybos_engine.config import get_device, get_settings, DeviceType
from polybos_engine.schemas import BoundingBox, ObjectDetection, ObjectsResult, SceneDetection

logger = logging.getLogger(__name__)


def _build_analysis_prompt(context: dict[str, str] | None = None) -> str:
    """Build the analysis prompt, optionally including context."""
    base_prompt = """Analyze this image and return JSON with two fields:
{
  "objects": ["object1", "object2", ...],
  "description": "A brief 1-2 sentence description of the scene."
}

For objects:
- Be specific (e.g., "scissors" not "tool", "remote control" not "device")
- Include people as "person" or "man"/"woman" if clear
- Only list clearly visible objects
- For miniatures/models, prefix with "model " (e.g., "model train")

For description:
- Describe what's happening in the scene
- Mention the setting/environment
- Note any activity or action taking place"""

    if not context:
        return base_prompt

    # Build context section
    context_lines = ["Known context about this video:"]

    # Map context keys to human-readable labels
    labels = {
        "person": "Person identified",
        "location": "Location",
        "activity": "Activity",
        "language": "Language spoken",
        "device": "Filmed with",
        "topic": "Topic/Subject",
        "organization": "Organization",
        "event": "Event",
    }

    for key, value in context.items():
        if value:
            label = labels.get(key, key.replace("_", " ").title())
            context_lines.append(f"- {label}: {value}")

    context_section = "\n".join(context_lines)

    # Get person name for explicit instruction
    person_name = context.get("person", "")
    person_instruction = ""
    if person_name:
        person_instruction = f"""
IMPORTANT: The person in this video is "{person_name}".
- In objects list: use "{person_name}" instead of "person", "man", or "woman"
- In description: refer to them as "{person_name}", not "a person" or "someone"
"""

    # Enhanced prompt with context
    return f"""{context_section}
{person_instruction}
Analyze this image and return JSON with two fields:
{{
  "objects": ["object1", "object2", ...],
  "description": "A brief 1-2 sentence description."
}}

For objects:
- Be specific (e.g., "scissors" not "tool", "remote control" not "device")
- IMPORTANT: If a person is visible and identified above, use their name (e.g., "{person_name}" not "person")
- Only list clearly visible objects

For description:
- Use the person's name "{person_name}" if they are visible
- Reference the known location/activity if relevant
- Describe what's happening in the scene"""


def extract_objects_qwen(
    file_path: str,
    scenes: list[SceneDetection] | None = None,
    frames_per_scene: int | None = None,
    model_name: str | None = None,
    context: dict[str, str] | None = None,
) -> ObjectsResult:
    """Extract objects using Qwen2-VL vision-language model.

    Much more accurate than YOLO for contextual understanding.
    Samples one frame per scene by default.

    Args:
        file_path: Path to video file
        scenes: Scene boundaries (required for scene-based sampling)
        frames_per_scene: Frames to analyze per scene (default from config)
        model_name: Qwen model name (default from config)
        context: Optional context from earlier extraction steps, e.g.:
            - "person": Name of identified person
            - "location": Where this was filmed
            - "activity": What's happening (e.g., "tutorial", "interview")
            - "language": Language spoken in the video
            - "device": Camera/device used
            - "topic": Subject matter of the video

    Returns:
        ObjectsResult with detected objects and contextual descriptions
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    settings = get_settings()
    model_name = model_name or settings.qwen_model
    frames_per_scene = frames_per_scene or settings.qwen_frames_per_scene

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Determine device
    device = get_device()
    if device == DeviceType.MPS:
        torch_device = "mps"
        torch_dtype = torch.float16
    elif device == DeviceType.CUDA:
        torch_device = "cuda"
        torch_dtype = torch.float16
    else:
        torch_device = "cpu"
        torch_dtype = torch.float32

    logger.info(f"Loading Qwen2-VL model: {model_name} on {torch_device}")

    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=torch_device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_qwen_")

    try:
        # Get timestamps to sample
        if scenes:
            timestamps = _get_timestamps_from_scenes(scenes, frames_per_scene)
            logger.info(f"Sampling {len(timestamps)} frames from {len(scenes)} scenes")
        else:
            # No scenes - sample evenly across video
            duration = _get_video_duration(file_path)
            sample_count = max(5, int(duration / 30))  # ~1 frame per 30 seconds
            timestamps = [duration * i / (sample_count + 1) for i in range(1, sample_count + 1)]
            logger.info(f"No scenes provided, sampling {len(timestamps)} frames evenly")

        # Extract frames
        frame_paths = _extract_frames_at_timestamps(file_path, temp_dir, timestamps)

        all_objects: dict[str, int] = {}
        detections: list[ObjectDetection] = []
        descriptions: list[str] = []

        for frame_path, timestamp in zip(frame_paths, timestamps):
            if not frame_path or not os.path.exists(frame_path):
                continue

            try:
                # Build the prompt with optional context
                prompt = _build_analysis_prompt(context)

                # Prepare message for Qwen - ask for both objects and description
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{frame_path}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Process inputs
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(torch_device)

                # Generate response
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Parse response
                objects, description = _parse_objects_and_description(output_text)
                for obj in objects:
                    obj_lower = obj.lower().strip()
                    all_objects[obj_lower] = all_objects.get(obj_lower, 0) + 1

                    detections.append(ObjectDetection(
                        timestamp=round(timestamp, 2),
                        label=obj_lower,
                        confidence=0.95,  # VLM confidence is generally high
                        bbox=BoundingBox(x=0, y=0, width=0, height=0),  # No bbox from VLM
                    ))

                if description:
                    descriptions.append(description)

                logger.debug(f"Frame {timestamp:.1f}s: {objects}")

                # Clear memory after each frame
                del inputs, generated_ids
                if torch_device == "mps":
                    torch.mps.empty_cache()
                elif torch_device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Failed to process frame {frame_path}: {e}")
                # Try to recover memory
                if torch_device == "mps":
                    torch.mps.empty_cache()
                continue

        # Deduplicate - count unique objects per type
        unique_objects = _deduplicate_objects(all_objects)

        logger.info(f"Qwen detected {len(unique_objects)} unique object types, {len(descriptions)} descriptions")

        return ObjectsResult(
            summary=unique_objects,
            detections=detections,
            descriptions=descriptions if descriptions else None,
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _get_timestamps_from_scenes(
    scenes: list[SceneDetection], frames_per_scene: int
) -> list[float]:
    """Generate sample timestamps from scenes."""
    timestamps: list[float] = []

    for scene in scenes:
        scene_duration = scene.end - scene.start
        if frames_per_scene == 1:
            # Sample from middle of scene
            timestamps.append(scene.start + scene_duration / 2)
        else:
            # Sample evenly across scene
            for i in range(frames_per_scene):
                offset = scene_duration * (i + 1) / (frames_per_scene + 1)
                timestamps.append(scene.start + offset)

    return sorted(timestamps)


def _extract_frames_at_timestamps(
    video_path: str, output_dir: str, timestamps: list[float], max_width: int = 1280
) -> list[str]:
    """Extract frames at specific timestamps, resized for VLM inference."""
    frame_paths: list[str] = []

    for i, ts in enumerate(timestamps):
        output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        # Scale to max width while preserving aspect ratio, reduce quality for memory
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale='min({max_width},iw)':-2",
            "-q:v", "3",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            frame_paths.append(output_path)
        except subprocess.CalledProcessError:
            frame_paths.append("")

    return frame_paths


def _parse_objects_and_description(response: str) -> tuple[list[str], str | None]:
    """Parse objects and description from Qwen response."""
    objects: list[str] = []
    description: str | None = None

    # Try to find and parse JSON
    try:
        # Remove markdown code block markers
        clean_response = response.replace("```json", "").replace("```", "").strip()

        if "{" in clean_response:
            json_str = clean_response[clean_response.index("{"):clean_response.rindex("}") + 1]
            data = json.loads(json_str)

            # Extract objects
            raw_objects = data.get("objects", [])
            objects = [
                obj for obj in raw_objects
                if isinstance(obj, str) and len(obj) < 100 and not obj.startswith('"')
            ]

            # Extract description
            desc = data.get("description", "")
            if isinstance(desc, str) and len(desc) > 10:
                description = desc.strip()

            return objects, description
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to extract objects from plain text
    for line in response.split("\n"):
        line = line.strip().strip("-").strip("*").strip()
        # Skip JSON artifacts and code block markers
        if not line or line.startswith("{") or line.startswith("}"):
            continue
        if line.startswith("```") or line.startswith('"objects"'):
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].rstrip(",")

        if len(line) > 50 or "[" in line or ":" in line:
            continue

        parts = [p.strip().strip('"').strip("'") for p in line.split(",")]
        objects.extend([p for p in parts if p and len(p) < 50])

    return objects, description


def _deduplicate_objects(objects: dict[str, int]) -> dict[str, int]:
    """Deduplicate object counts.

    If an object appears in multiple frames, it's likely the same instance.
    Returns count of 1 for each unique object type.
    """
    return {obj: 1 for obj in objects.keys()}

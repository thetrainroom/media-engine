"""Object detection using Qwen2-VL vision-language model."""

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from media_engine.config import (
    DeviceType,
    get_device,
    get_free_memory_gb,
    get_settings,
)
from media_engine.extractors.frames import FrameExtractor
from media_engine.schemas import BoundingBox, ObjectDetection, ObjectsResult

logger = logging.getLogger(__name__)

# Progress callback type: (message, current, total) -> None
ProgressCallback = Callable[[str, int | None, int | None], None]

# Singleton model instances (lazy loaded, stays in memory between calls)
_qwen_model: Any = None
_qwen_processor: Any = None
_qwen_model_name: str | None = None
_qwen_device: str | None = None


def unload_qwen_model() -> None:
    """Unload Qwen model from memory to free GPU/MPS memory."""
    global _qwen_model, _qwen_processor, _qwen_model_name, _qwen_device

    if _qwen_model is not None:
        logger.info("Unloading Qwen model from memory")

        # Move model to CPU first to release MPS memory
        try:
            _qwen_model.to("cpu")
        except Exception:
            pass

        del _qwen_model
        del _qwen_processor
        _qwen_model = None
        _qwen_processor = None
        _qwen_model_name = None
        _qwen_device = None

        import gc

        gc.collect()

        # Free GPU memory with sync
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if hasattr(torch, "mps"):
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        gc.collect()


# Known LOG/HDR color transfer characteristics
# These indicate footage that needs color correction to look "normal"
LOG_COLOR_TRANSFERS = {
    # HDR transfer functions
    "arib-std-b67",  # HLG (Hybrid Log-Gamma)
    "smpte2084",  # PQ (Perceptual Quantizer) / HDR10
    "smpte428",  # DCI-P3
    # Manufacturer LOG profiles (as they appear in ffmpeg metadata)
    "log",  # Generic log
    "slog",  # Sony S-Log
    "slog2",  # Sony S-Log2
    "slog3",  # Sony S-Log3
    "vlog",  # Panasonic V-Log
    "clog",  # Canon C-Log
    "clog2",  # Canon C-Log2
    "clog3",  # Canon C-Log3
    "dlog",  # DJI D-Log
    "dlog-m",  # DJI D-Log M
    "hlg",  # HLG
    "n-log",  # Nikon N-Log
    "f-log",  # Fujifilm F-Log
    "f-log2",  # Fujifilm F-Log2
    "blackmagic",  # Blackmagic Film
    "arri",  # ARRI Log C
    "logc",  # ARRI Log C
    "redlogfilm",  # RED Log Film
}


def _is_log_color_space(color_transfer: str | None) -> bool:
    """Check if the color transfer characteristic indicates LOG/HDR footage.

    Args:
        color_transfer: The color transfer characteristic from video metadata
                       (e.g., "arib-std-b67", "smpte2084", "bt709")

    Returns:
        True if the footage appears to be in a LOG/flat/HDR color space
        that would benefit from color correction before viewing.
    """
    if not color_transfer:
        return False

    # Normalize to lowercase for comparison
    ct_lower = color_transfer.lower().replace("_", "-").replace(" ", "")

    # Check for exact matches first
    if ct_lower in LOG_COLOR_TRANSFERS:
        return True

    # Check for partial matches (e.g., "s-log3" contains "log")
    log_keywords = ["log", "hlg", "pq", "hdr", "dci-p3"]
    for keyword in log_keywords:
        if keyword in ct_lower:
            return True

    return False


def _get_qwen_model(
    model_name: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Any, Any, str]:
    """Get or create the Qwen model and processor (singleton).

    Returns (model, processor, device_str).
    Raises RuntimeError/MemoryError if model cannot be loaded (e.g., OOM).
    Model stays loaded in memory for subsequent calls.
    """
    global _qwen_model, _qwen_processor, _qwen_model_name, _qwen_device

    # Return cached model if same model requested
    if _qwen_model is not None and _qwen_model_name == model_name:
        logger.info(f"Reusing cached Qwen model: {model_name}")
        return _qwen_model, _qwen_processor, _qwen_device  # type: ignore

    # Log memory status (informational only - let PyTorch handle OOM)
    free_memory = get_free_memory_gb()
    model_memory_gb = 15.0 if "7B" in model_name else 5.0
    logger.info(f"Free memory: {free_memory:.1f}GB, model needs: ~{model_memory_gb:.0f}GB")

    # Clear existing GPU memory before loading
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")
    gc.collect()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore[import-not-found]

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
    if progress_callback:
        progress_callback("Loading Qwen model...", None, None)

    # Disable tqdm progress bars and warnings to avoid BrokenPipeError when running as daemon
    import transformers  # type: ignore[import-not-found]

    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()  # Suppress info/warning output
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Load model and processor with detailed error handling
    try:
        logger.info("Loading Qwen2VLForConditionalGeneration...")

        # For MPS (Apple Silicon), don't use device_map at all
        # device_map triggers accelerate's meta tensor handling which fails on MPS
        if torch_device == "mps":
            _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                # No device_map - load directly to CPU without accelerate dispatch
            )
            logger.info("Moving model to MPS...")
            _qwen_model = _qwen_model.to("mps")
        elif torch_device == "cuda":
            # CUDA works fine with device_map
            _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="cuda",
            )
        else:
            # CPU - no device_map needed
            _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )

        logger.info("Qwen model loaded, loading processor...")
        _qwen_processor = AutoProcessor.from_pretrained(model_name)
        logger.info("Qwen processor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Qwen model: {e}", exc_info=True)
        raise

    _qwen_model_name = model_name
    _qwen_device = torch_device

    return _qwen_model, _qwen_processor, torch_device


def _build_analysis_prompt(context: dict[str, str] | None = None) -> str:
    """Build the analysis prompt, optionally including context."""
    base_prompt = """Look at this image carefully and describe what you see.

List all visible objects and write a brief description of the scene.

You MUST respond with ONLY this exact JSON format:
{"objects": ["item1", "item2"], "description": "One or two sentences describing the scene."}

Rules for objects:
- Be specific: "scissors" not "tool", "laptop" not "device"
- Include people as "person" or "man"/"woman"
- Only list clearly visible objects

Rules for description:
- Describe what's happening
- Mention the setting/environment
- Keep it to 1-2 sentences

Respond with JSON only, no other text."""

    if not context:
        return base_prompt

    # Build context section
    context_lines = ["Known context about this video:"]

    # Map context keys to human-readable labels
    labels = {
        "person": "Person identified",
        "location": "Location",
        "nearby_landmarks": "Nearby landmarks/POIs",
        "activity": "Activity",
        "language": "Language spoken",
        "device": "Filmed with",
        "topic": "Topic/Subject",
        "organization": "Organization",
        "event": "Event",
    }

    # Handle log footage note separately (not as a bullet point)
    log_footage_note = context.get("log_footage_note", "")

    for key, value in context.items():
        if value and key != "log_footage_note":
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

    # Get nearby landmarks for naming instruction
    nearby_landmarks = context.get("nearby_landmarks", "")
    landmark_instruction = ""
    if nearby_landmarks:
        landmark_instruction = f"""
IMPORTANT: This location has these nearby landmarks: {nearby_landmarks}
- If you see any of these landmarks, use their PROPER NAME in the description
- Example: say "Alnes fyr lighthouse" not just "a lighthouse"
- Example: say "Eiffel Tower" not just "a tower"
"""

    # Add log footage instruction if applicable
    log_instruction = ""
    if log_footage_note:
        log_instruction = f"""
NOTE: {log_footage_note}
- Focus on describing the content and action, not the color grading
"""

    # Enhanced prompt with context
    return f"""{context_section}
{person_instruction}{landmark_instruction}{log_instruction}
Look at this image carefully and describe what you see.

You MUST respond with ONLY this exact JSON format:
{{"objects": ["item1", "item2"], "description": "One or two sentences describing the scene."}}

Rules for objects:
- Be specific: "scissors" not "tool", "laptop" not "device"
- If a person is visible and identified above, use their name ("{person_name}") not "person"
- If a known landmark is visible, use its proper name from the context
- Only list clearly visible objects

Rules for description:
- Use "{person_name}" if they are visible
- Use proper landmark names if visible
- Describe what's happening in the scene
- Keep it to 1-2 sentences

Respond with JSON only, no other text."""


def extract_objects_qwen(
    file_path: str,
    timestamps: list[float] | None = None,
    model_name: str | None = None,
    context: dict[str, str] | None = None,
    progress_callback: ProgressCallback | None = None,
    lut_path: str | None = None,
) -> ObjectsResult:
    """Extract objects using Qwen2-VL vision-language model.

    Much more accurate than YOLO for contextual understanding.

    Args:
        file_path: Path to video file
        timestamps: Specific timestamps to analyze. If None, samples from middle.
        model_name: Qwen model name (default from config)
        context: Optional context from earlier extraction steps, e.g.:
            - "person": Name of identified person
            - "location": Where this was filmed
            - "activity": What's happening (e.g., "tutorial", "interview")
            - "language": Language spoken in the video
            - "device": Camera/device used
            - "topic": Subject matter of the video
        progress_callback: Optional callback for progress updates (message, current, total)
        lut_path: Optional path to a LUT file (.cube) to apply for log footage color correction

    Returns:
        ObjectsResult with detected objects and contextual descriptions
    """
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

    logger.info(f"extract_objects_qwen called: file={file_path}, timestamps={timestamps}, context={context}")

    settings = get_settings()
    # Resolve model name (handles "auto")
    model_name = model_name or settings.get_qwen_model()
    logger.info(f"Using Qwen model: {model_name}")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_qwen_")

    try:
        # Use provided timestamps, or default to middle of video
        if timestamps is None:
            duration = _get_video_duration(file_path)
            timestamps = [duration / 2]
            logger.info(f"No timestamps provided, sampling from middle ({duration/2:.1f}s)")
        else:
            logger.info(f"Analyzing {len(timestamps)} provided timestamps")

        # Check for LOG/HDR color space from metadata
        color_transfer = context.get("color_transfer") if context else None
        is_log_footage = _is_log_color_space(color_transfer)

        # Add context hint for log footage
        if context is None:
            context = {}
        else:
            context = context.copy()  # Don't modify the original

        if lut_path and os.path.exists(lut_path):
            # LUT applied - colors are corrected but may still be slightly off
            context["log_footage_note"] = (
                "This footage was recorded in LOG profile and color-corrected with a LUT. " "Colors shown are the corrected version but may still appear slightly desaturated."
            )
            logger.info("Added log footage context hint (with LUT)")
        elif is_log_footage:
            # LOG detected but no LUT - colors are definitely off
            context["log_footage_note"] = (
                f"This footage appears to be in LOG/flat color profile ({color_transfer}). "
                "Colors are desaturated and not representative of the actual scene. "
                "Focus on describing content and action, not colors."
            )
            logger.info(f"Added log footage context hint (no LUT, color_transfer={color_transfer})")

        # IMPORTANT: Extract frames BEFORE loading the model!
        # ffmpeg can crash (SIGABRT) when forked from a process with MPS/Metal loaded.
        if progress_callback:
            progress_callback("Extracting frames...", None, None)
        frame_paths = _extract_frames_at_timestamps(file_path, temp_dir, timestamps, lut_path=lut_path)
        total_frames = len([p for p in frame_paths if p])

        if total_frames == 0:
            logger.warning(f"No frames could be extracted from {file_path} at timestamps {timestamps}")
            return ObjectsResult(summary={}, detections=[], descriptions=None)

        # Now load the model (after ffmpeg has finished)
        # If this fails due to OOM, the exception propagates up
        try:
            model, processor, torch_device = _get_qwen_model(model_name, progress_callback)
        except (RuntimeError, MemoryError, OSError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cannot allocate" in error_msg:
                logger.error(f"Out of memory loading Qwen model. " f"Close other apps or use a cloud vision API. Error: {e}")
                # Return empty result - frontend can fall back to cloud API if configured
                return ObjectsResult(
                    summary={},
                    detections=[],
                    descriptions=None,
                    error="out_of_memory",
                )
            raise  # Re-raise other errors

        logger.info(f"Processing {total_frames} frames for Qwen analysis")

        all_objects: dict[str, int] = {}
        detections: list[ObjectDetection] = []
        descriptions: list[str] = []
        frame_count = 0

        for frame_path, timestamp in zip(frame_paths, timestamps):
            if not frame_path or not os.path.exists(frame_path):
                logger.warning(f"Skipping missing frame at {timestamp}s: {frame_path}")
                continue

            frame_count += 1
            if progress_callback:
                progress_callback(
                    f"Analyzing frame {frame_count}/{total_frames}...",
                    frame_count,
                    total_frames,
                )

            try:
                # Build the prompt with optional context
                prompt = _build_analysis_prompt(context)

                # Log prompt on first frame for debugging
                if frame_count == 1:
                    logger.info(f"Qwen prompt: {prompt[:500]}")

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
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(torch_device)

                # Generate response with repetition penalty to prevent loops
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,  # Greedy decoding for consistent JSON
                        repetition_penalty=1.2,  # Penalize repetition
                        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    )
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Parse response
                logger.info(f"Qwen raw output for {timestamp:.1f}s: {output_text[:500]}")
                objects, description = _parse_objects_and_description(output_text)
                if not description:
                    logger.warning(f"No description parsed from Qwen output at {timestamp:.1f}s")
                for obj in objects:
                    obj_lower = obj.lower().strip()
                    all_objects[obj_lower] = all_objects.get(obj_lower, 0) + 1

                    detections.append(
                        ObjectDetection(
                            timestamp=round(timestamp, 2),
                            label=obj_lower,
                            confidence=0.95,  # VLM confidence is generally high
                            bbox=BoundingBox(x=0, y=0, width=0, height=0),  # No bbox from VLM
                        )
                    )

                if description:
                    descriptions.append(description)
                    logger.info(f"Frame {timestamp:.1f}s description: {description}")

                logger.info(f"Frame {timestamp:.1f}s objects: {objects}")

                # Clear memory after each frame
                del inputs, generated_ids
                if torch_device == "mps":
                    torch.mps.empty_cache()
                elif torch_device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to process frame {frame_path}: {e}", exc_info=True)
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


def _get_video_duration(file_path: str) -> float:
    """Get video/image duration in seconds (0 for images)."""
    from media_engine.extractors.frames import get_video_duration

    return get_video_duration(file_path)


def _extract_frames_at_timestamps(
    file_path: str,
    output_dir: str,
    timestamps: list[float],
    max_width: int = 1280,
    lut_path: str | None = None,
) -> list[str]:
    """Extract frames at specific timestamps, resized for VLM inference.

    Uses FrameExtractor which handles both videos (via OpenCV/ffmpeg)
    and images (via direct loading). When a LUT path is provided, uses
    ffmpeg directly to apply the LUT during extraction.

    Args:
        file_path: Path to video/image file
        output_dir: Directory to save extracted frames
        timestamps: List of timestamps to extract (in seconds)
        max_width: Maximum width for scaling (default 1280)
        lut_path: Optional path to a .cube LUT file for color correction
    """
    import subprocess

    import cv2

    frame_paths: list[str] = []

    logger.info(f"Extracting {len(timestamps)} frames from {file_path} at timestamps {timestamps}")

    # If LUT is provided, use ffmpeg directly for extraction with LUT applied
    if lut_path and os.path.exists(lut_path):
        logger.info(f"Applying LUT: {lut_path}")
        for i, ts in enumerate(timestamps):
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            try:
                # Build filter chain: LUT + scale
                scale_filter = f"scale={max_width}:{max_width}:force_original_aspect_ratio=decrease"
                lut_filter = f"lut3d='{lut_path}'"
                vf = f"{lut_filter},{scale_filter}"

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(ts),
                    "-i",
                    file_path,
                    "-vf",
                    vf,
                    "-frames:v",
                    "1",
                    "-update",
                    "1",
                    "-q:v",
                    "2",
                    output_path,
                ]
                subprocess.run(cmd, capture_output=True, check=True)

                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    frame_paths.append(output_path)
                    logger.info(f"Extracted frame {i} at {ts:.2f}s with LUT: {output_path}")
                else:
                    logger.warning(f"Frame at {ts:.2f}s: could not extract with LUT")
                    frame_paths.append("")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Frame at {ts:.2f}s: ffmpeg failed: {e}")
                frame_paths.append("")
    else:
        # Standard extraction without LUT
        with FrameExtractor(file_path, max_dimension=max_width) as extractor:
            for i, ts in enumerate(timestamps):
                output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                frame = extractor.get_frame_at(ts)

                if frame is not None:
                    # Save frame as JPEG with moderate quality for VLM
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        frame_paths.append(output_path)
                        logger.info(f"Extracted frame {i} at {ts:.2f}s: {output_path}")
                    else:
                        logger.warning(f"Frame at {ts:.2f}s: could not save to {output_path}")
                        frame_paths.append("")
                else:
                    logger.warning(f"Frame at {ts:.2f}s: extraction failed")
                    frame_paths.append("")

    successful = sum(1 for p in frame_paths if p)
    logger.info(f"Frame extraction complete: {successful}/{len(timestamps)} frames extracted")
    return frame_paths


def _parse_objects_and_description(response: str) -> tuple[list[str], str | None]:
    """Parse objects and description from Qwen response."""
    objects: list[str] = []
    description: str | None = None

    # Try to find and parse JSON
    try:
        # Remove markdown code block markers
        clean_response = response.replace("```json", "").replace("```", "").strip()

        # Try to parse as JSON (could be object or array)
        if "[" in clean_response or "{" in clean_response:
            # Find the JSON portion
            start_bracket = clean_response.find("[")
            start_brace = clean_response.find("{")

            if start_bracket >= 0 and (start_brace < 0 or start_bracket < start_brace):
                # Array format - find matching ]
                json_str = clean_response[start_bracket : clean_response.rindex("]") + 1]
                data = json.loads(json_str)

                # Array of objects - take the first non-empty one
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            raw_objects = item.get("objects", [])
                            for obj in raw_objects:
                                if isinstance(obj, str) and len(obj) < 100 and obj.strip():
                                    objects.append(obj)
                                elif isinstance(obj, dict):
                                    # Handle nested format: {"name": "person"}
                                    name = obj.get("name", "") or obj.get("label", "")
                                    if isinstance(name, str) and len(name) < 100 and name.strip():
                                        objects.append(name)
                            desc = item.get("description", "")
                            if isinstance(desc, str) and len(desc) > 10 and not description:
                                description = desc.strip()
                    return objects, description

            # Single object format
            if start_brace >= 0:
                json_str = clean_response[start_brace : clean_response.rindex("}") + 1]
                data = json.loads(json_str)

                # Extract objects - handle both string and dict formats
                raw_objects = data.get("objects", [])
                for obj in raw_objects:
                    if isinstance(obj, str) and len(obj) < 100 and obj.strip():
                        objects.append(obj)
                    elif isinstance(obj, dict):
                        # Handle nested format: {"name": "person", "position": "..."}
                        name = obj.get("name", "") or obj.get("label", "")
                        if isinstance(name, str) and len(name) < 100 and name.strip():
                            objects.append(name)

                # Extract description
                desc = data.get("description", "")
                if isinstance(desc, str) and len(desc) > 10:
                    description = desc.strip()

                return objects, description
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse JSON from Qwen response: {e}")
        logger.debug(f"Response was: {response[:500]}")

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

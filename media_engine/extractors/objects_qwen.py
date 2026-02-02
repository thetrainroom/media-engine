"""Object detection using Qwen2-VL vision-language model."""

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from media_engine.config import (
    DeviceType,
    QwenStrategy,
    get_auto_qwen_batch_size,
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


def _build_context_prompt(
    context: dict[str, str] | None = None,
    previous_description: str | None = None,
) -> str:
    """Build prompt for CONTEXT strategy - includes previous frame description."""
    base_prompt = _build_analysis_prompt(context)

    if not previous_description:
        return base_prompt

    # Insert previous frame context before the analysis request
    context_insert = f"""
Previous frame showed: {previous_description}

Describe what's happening NOW and how it relates to the previous frame.
Focus on: objects visible, actions occurring, any changes from before.

"""
    # Modify the JSON format to include "change" field
    modified_prompt = base_prompt.replace(
        '{"objects": ["item1", "item2"], "description": "One or two sentences describing the scene."}',
        '{"objects": ["item1", "item2"], "description": "What\'s happening now.", "change": "How this differs from the previous frame."}',
    )

    # Insert context after any existing context section but before "Look at this image"
    if "Look at this image" in modified_prompt:
        parts = modified_prompt.split("Look at this image")
        return parts[0] + context_insert + "Look at this image" + parts[1]

    return context_insert + modified_prompt


def _build_batch_prompt(
    context: dict[str, str] | None = None,
    num_frames: int = 3,
) -> str:
    """Build prompt for BATCH strategy - analyzes multiple frames together."""
    # Get person name from context for instructions
    person_name = context.get("person", "") if context else ""

    # Build context section if available
    context_section = ""
    if context:
        context_lines = ["Known context about this video:"]
        labels = {
            "person": "Person identified",
            "location": "Location",
            "nearby_landmarks": "Nearby landmarks/POIs",
            "activity": "Activity",
            "language": "Language spoken",
            "device": "Filmed with",
        }
        for key, value in context.items():
            if value and key not in ("log_footage_note", "color_transfer"):
                label = labels.get(key, key.replace("_", " ").title())
                context_lines.append(f"- {label}: {value}")
        context_section = "\n".join(context_lines) + "\n\n"

    person_instruction = ""
    if person_name:
        person_instruction = f'Use "{person_name}" instead of "person" in objects and description.\n'

    return f"""{context_section}These {num_frames} frames are from the same video in sequence.
{person_instruction}
Analyze what happens ACROSS these frames:
1. What objects/people are visible throughout?
2. What ACTION or movement occurs across the frames?
3. How does the scene change from first to last frame?

You MUST respond with ONLY this exact JSON format:
{{"objects": ["item1", "item2"], "action": "The action happening across frames", "description": "Overall scene description"}}

Rules:
- List objects visible in ANY of the frames
- Describe the ACTION that unfolds across frames (e.g., "person walks toward camera", "car turns left")
- Keep description to 1-2 sentences summarizing the sequence

Respond with JSON only, no other text."""


def _build_batch_context_prompt(
    context: dict[str, str] | None = None,
    num_frames: int = 3,
    group_context: str | None = None,
) -> str:
    """Build prompt for BATCH_CONTEXT strategy - batch with previous group context."""
    base_prompt = _build_batch_prompt(context, num_frames)

    if not group_context:
        return base_prompt

    context_insert = f"""Previous scene: {group_context}

What happens next in these frames? How does it continue from before?

"""
    # Modify JSON format to include "continues" field
    modified_prompt = base_prompt.replace(
        '{"objects": ["item1", "item2"], "action": "The action happening across frames", "description": "Overall scene description"}',
        '{"objects": ["item1", "item2"], "action": "The action in these frames", "description": "Scene description", "continues": "How this continues from the previous scene"}',
    )

    # Insert after context section but before "These X frames"
    if "These " in modified_prompt and " frames are" in modified_prompt:
        idx = modified_prompt.find("These ")
        return modified_prompt[:idx] + context_insert + modified_prompt[idx:]

    return context_insert + modified_prompt


def _analyze_frames_single(
    model: Any,
    processor: Any,
    torch_device: str,
    frame_paths: list[str],
    timestamps: list[float],
    context: dict[str, str] | None,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, int], list[ObjectDetection], list[str]]:
    """Analyze frames one at a time without temporal context (original behavior)."""
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

    all_objects: dict[str, int] = {}
    detections: list[ObjectDetection] = []
    descriptions: list[str] = []

    total_frames = len([p for p in frame_paths if p])
    frame_count = 0

    for frame_path, timestamp in zip(frame_paths, timestamps):
        if not frame_path or not os.path.exists(frame_path):
            continue

        frame_count += 1
        if progress_callback:
            progress_callback(
                f"Analyzing frame {frame_count}/{total_frames}...",
                frame_count,
                total_frames,
            )

        try:
            prompt = _build_analysis_prompt(context)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{frame_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

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

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(f"Qwen raw output for {timestamp:.1f}s: {output_text[:500]}")
            objects, description = _parse_objects_and_description(output_text)

            for obj in objects:
                obj_lower = obj.lower().strip()
                all_objects[obj_lower] = all_objects.get(obj_lower, 0) + 1
                detections.append(
                    ObjectDetection(
                        timestamp=round(timestamp, 2),
                        label=obj_lower,
                        confidence=0.95,
                        bbox=BoundingBox(x=0, y=0, width=0, height=0),
                    )
                )

            if description:
                descriptions.append(description)
                logger.info(f"Frame {timestamp:.1f}s description: {description}")

            logger.info(f"Frame {timestamp:.1f}s objects: {objects}")

            del inputs, generated_ids
            if torch_device == "mps":
                torch.mps.empty_cache()
            elif torch_device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to process frame {frame_path}: {e}", exc_info=True)
            if torch_device == "mps":
                torch.mps.empty_cache()

    return all_objects, detections, descriptions


def _analyze_frames_with_context(
    model: Any,
    processor: Any,
    torch_device: str,
    frame_paths: list[str],
    timestamps: list[float],
    context: dict[str, str] | None,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, int], list[ObjectDetection], list[str]]:
    """Analyze frames sequentially, passing previous description as context."""
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

    all_objects: dict[str, int] = {}
    detections: list[ObjectDetection] = []
    descriptions: list[str] = []

    total_frames = len([p for p in frame_paths if p])
    frame_count = 0
    previous_description: str | None = None

    for frame_path, timestamp in zip(frame_paths, timestamps):
        if not frame_path or not os.path.exists(frame_path):
            continue

        frame_count += 1
        if progress_callback:
            progress_callback(
                f"Analyzing frame {frame_count}/{total_frames} (with context)...",
                frame_count,
                total_frames,
            )

        try:
            # Build prompt with previous frame's description as context
            prompt = _build_context_prompt(context, previous_description)

            if frame_count == 1:
                logger.info(f"Qwen context prompt (first frame): {prompt[:500]}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{frame_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

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

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(f"Qwen raw output for {timestamp:.1f}s: {output_text[:500]}")
            objects, description = _parse_objects_and_description(output_text)

            for obj in objects:
                obj_lower = obj.lower().strip()
                all_objects[obj_lower] = all_objects.get(obj_lower, 0) + 1
                detections.append(
                    ObjectDetection(
                        timestamp=round(timestamp, 2),
                        label=obj_lower,
                        confidence=0.95,
                        bbox=BoundingBox(x=0, y=0, width=0, height=0),
                    )
                )

            if description:
                descriptions.append(description)
                previous_description = description  # Pass to next frame
                logger.info(f"Frame {timestamp:.1f}s description: {description}")

            logger.info(f"Frame {timestamp:.1f}s objects: {objects}")

            del inputs, generated_ids
            if torch_device == "mps":
                torch.mps.empty_cache()
            elif torch_device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to process frame {frame_path}: {e}", exc_info=True)
            if torch_device == "mps":
                torch.mps.empty_cache()

    return all_objects, detections, descriptions


def _analyze_frames_batch(
    model: Any,
    processor: Any,
    torch_device: str,
    frame_paths: list[str],
    timestamps: list[float],
    context: dict[str, str] | None,
    progress_callback: ProgressCallback | None,
    batch_size: int | None = None,
    overlap: bool = False,
) -> tuple[dict[str, int], list[ObjectDetection], list[str]]:
    """Analyze frames in batches for temporal understanding."""
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

    all_objects: dict[str, int] = {}
    detections: list[ObjectDetection] = []
    descriptions: list[str] = []

    # Auto-select batch size based on available memory
    if batch_size is None:
        batch_size = get_auto_qwen_batch_size()

    # Filter to valid frames
    valid_frames = [(p, t) for p, t in zip(frame_paths, timestamps) if p and os.path.exists(p)]
    if not valid_frames:
        return all_objects, detections, descriptions

    # Group frames into batches
    # With overlap: last frame of batch N = first frame of batch N+1 (visual continuity)
    # Without overlap: sequential non-overlapping batches (faster)
    batches: list[list[tuple[str, float]]] = []
    step = max(1, batch_size - 1) if overlap else batch_size
    for i in range(0, len(valid_frames), step):
        batch = valid_frames[i : i + batch_size]
        if overlap:
            if len(batch) >= 2:  # Need at least 2 frames for temporal analysis
                batches.append(batch)
            elif not batches:  # Edge case: very few frames
                batches.append(batch)
        else:
            batches.append(batch)

    total_batches = len(batches)
    overlap_str = "overlapping " if overlap else ""
    logger.info(f"Processing {len(valid_frames)} frames in {total_batches} {overlap_str}batches (size={batch_size}, step={step})")

    for batch_idx, batch in enumerate(batches):
        if progress_callback:
            progress_callback(
                f"Analyzing batch {batch_idx + 1}/{total_batches}...",
                batch_idx + 1,
                total_batches,
            )

        try:
            # Build multi-image message
            prompt = _build_batch_prompt(context, len(batch))

            if batch_idx == 0:
                logger.info(f"Qwen batch prompt: {prompt[:500]}")

            # Build content with all images in the batch
            content: list[dict[str, str]] = []
            for frame_path, _ in batch:
                content.append({"type": "image", "image": f"file://{frame_path}"})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

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

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(f"Qwen batch {batch_idx + 1} raw output: {output_text[:500]}")
            objects, description = _parse_batch_response(output_text)

            # Associate objects with the middle timestamp of the batch
            batch_timestamps = [t for _, t in batch]
            middle_timestamp = batch_timestamps[len(batch_timestamps) // 2]

            for obj in objects:
                obj_lower = obj.lower().strip()
                all_objects[obj_lower] = all_objects.get(obj_lower, 0) + 1
                detections.append(
                    ObjectDetection(
                        timestamp=round(middle_timestamp, 2),
                        label=obj_lower,
                        confidence=0.95,
                        bbox=BoundingBox(x=0, y=0, width=0, height=0),
                    )
                )

            if description:
                descriptions.append(description)
                logger.info(f"Batch {batch_idx + 1} description: {description}")

            logger.info(f"Batch {batch_idx + 1} objects: {objects}")

            del inputs, generated_ids
            if torch_device == "mps":
                torch.mps.empty_cache()
            elif torch_device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx + 1}: {e}", exc_info=True)
            if torch_device == "mps":
                torch.mps.empty_cache()

    return all_objects, detections, descriptions


def _analyze_frames_batch_context(
    model: Any,
    processor: Any,
    torch_device: str,
    frame_paths: list[str],
    timestamps: list[float],
    context: dict[str, str] | None,
    progress_callback: ProgressCallback | None,
    batch_size: int | None = None,
    overlap: bool = False,
) -> tuple[dict[str, int], list[ObjectDetection], list[str]]:
    """Analyze frames in batches with context passed between batches."""
    from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

    all_objects: dict[str, int] = {}
    detections: list[ObjectDetection] = []
    descriptions: list[str] = []

    # Auto-select batch size based on available memory
    if batch_size is None:
        batch_size = get_auto_qwen_batch_size()

    # Filter to valid frames
    valid_frames = [(p, t) for p, t in zip(frame_paths, timestamps) if p and os.path.exists(p)]
    if not valid_frames:
        return all_objects, detections, descriptions

    # Group frames into batches
    # With overlap: last frame of batch N = first frame of batch N+1 (visual continuity)
    # Without overlap: sequential non-overlapping batches (faster)
    batches: list[list[tuple[str, float]]] = []
    step = max(1, batch_size - 1) if overlap else batch_size
    for i in range(0, len(valid_frames), step):
        batch = valid_frames[i : i + batch_size]
        if overlap:
            if len(batch) >= 2:  # Need at least 2 frames for temporal analysis
                batches.append(batch)
            elif not batches:  # Edge case: very few frames
                batches.append(batch)
        else:
            batches.append(batch)

    total_batches = len(batches)
    overlap_str = "overlapping " if overlap else ""
    logger.info(f"Processing {len(valid_frames)} frames in {total_batches} {overlap_str}batches with context (size={batch_size}, step={step})")

    group_context: str | None = None

    for batch_idx, batch in enumerate(batches):
        if progress_callback:
            progress_callback(
                f"Analyzing batch {batch_idx + 1}/{total_batches} (with context)...",
                batch_idx + 1,
                total_batches,
            )

        try:
            # Build multi-image message with previous batch context
            prompt = _build_batch_context_prompt(context, len(batch), group_context)

            if batch_idx == 0:
                logger.info(f"Qwen batch-context prompt: {prompt[:500]}")

            # Build content with all images in the batch
            content: list[dict[str, str]] = []
            for frame_path, _ in batch:
                content.append({"type": "image", "image": f"file://{frame_path}"})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

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

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(f"Qwen batch {batch_idx + 1} raw output: {output_text[:500]}")
            objects, description = _parse_batch_response(output_text)

            # Use description as context for next batch
            if description:
                group_context = description

            # Associate objects with the middle timestamp of the batch
            batch_timestamps = [t for _, t in batch]
            middle_timestamp = batch_timestamps[len(batch_timestamps) // 2]

            for obj in objects:
                obj_lower = obj.lower().strip()
                all_objects[obj_lower] = all_objects.get(obj_lower, 0) + 1
                detections.append(
                    ObjectDetection(
                        timestamp=round(middle_timestamp, 2),
                        label=obj_lower,
                        confidence=0.95,
                        bbox=BoundingBox(x=0, y=0, width=0, height=0),
                    )
                )

            if description:
                descriptions.append(description)
                logger.info(f"Batch {batch_idx + 1} description: {description}")

            logger.info(f"Batch {batch_idx + 1} objects: {objects}")

            del inputs, generated_ids
            if torch_device == "mps":
                torch.mps.empty_cache()
            elif torch_device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx + 1}: {e}", exc_info=True)
            if torch_device == "mps":
                torch.mps.empty_cache()

    return all_objects, detections, descriptions


def _fix_malformed_json(text: str) -> str:
    """Fix common JSON malformations from VLM output."""
    # Remove markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()

    # Fix escaped quotes before colons: "action\": -> "action":
    text = text.replace('\\":', '":')

    # Replace single quotes with double quotes for keys and string values
    # But be careful not to replace apostrophes within words
    # First, handle keys: 'key': -> "key":
    text = re.sub(r"'(\w+)'(\s*):", r'"\1"\2:', text)

    # Handle string values: : 'value' -> : "value"
    # This regex looks for : followed by optional whitespace and a single-quoted string
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)

    # Remove trailing commas before ] or }
    text = re.sub(r",(\s*[\]\}])", r"\1", text)

    return text


def _parse_batch_response(response: str) -> tuple[list[str], str | None]:
    """Parse objects and description from batch analysis response.

    Handles both standard format and batch-specific format with action field.
    """
    objects: list[str] = []
    description: str | None = None

    try:
        clean_response = _fix_malformed_json(response)

        if "{" in clean_response:
            start_brace = clean_response.find("{")
            json_str = clean_response[start_brace : clean_response.rindex("}") + 1]
            data = json.loads(json_str)

            # Extract objects
            raw_objects = data.get("objects", [])
            for obj in raw_objects:
                if isinstance(obj, str) and len(obj) < 100 and obj.strip():
                    objects.append(obj)
                elif isinstance(obj, dict):
                    name = obj.get("name", "") or obj.get("label", "")
                    if isinstance(name, str) and len(name) < 100 and name.strip():
                        objects.append(name)

            # Build description from available fields
            desc_parts = []

            # Action field (batch-specific)
            action = data.get("action", "")
            if isinstance(action, str) and action.strip():
                desc_parts.append(action.strip())

            # Standard description
            desc = data.get("description", "")
            if isinstance(desc, str) and desc.strip():
                desc_parts.append(desc.strip())

            # Continues field (batch-context specific)
            continues = data.get("continues", "")
            if isinstance(continues, str) and continues.strip():
                desc_parts.append(continues.strip())

            # Change field (context-specific)
            change = data.get("change", "")
            if isinstance(change, str) and change.strip():
                desc_parts.append(f"Change: {change.strip()}")

            if desc_parts:
                description = " ".join(desc_parts)

            return objects, description

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse batch JSON from Qwen response: {e}")

    # Fallback to standard parser
    return _parse_objects_and_description(response)


def extract_objects_qwen(
    file_path: str,
    timestamps: list[float] | None = None,
    model_name: str | None = None,
    context: dict[str, str] | None = None,
    progress_callback: ProgressCallback | None = None,
    lut_path: str | None = None,
    batch_overlap: bool = False,
    strategy: str | None = None,
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
        batch_overlap: If True, batches overlap by 1 frame for visual continuity.
            Useful for unstable camera or videos with rapid scene changes.
            Default False for faster processing.
        strategy: Override Qwen strategy for this file. One of:
            - "single": No temporal context (fastest)
            - "context": Pass previous description as text
            - "batch": Multi-frame batches
            - "batch_context": Batches with text context between (richest)
            If None, uses global setting from config.

    Returns:
        ObjectsResult with detected objects and contextual descriptions
    """
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
            logger.info(f"No timestamps provided, sampling from middle ({duration / 2:.1f}s)")
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
                "This footage was recorded in LOG profile and color-corrected with a LUT. Colors shown are the corrected version but may still appear slightly desaturated."
            )
            logger.info("Added log footage context hint (with LUT)")
        elif is_log_footage:
            # LOG detected but no LUT - colors are definitely off
            context["log_footage_note"] = (
                f"This footage appears to be in LOG/flat color profile ({color_transfer}). Colors are desaturated and not representative of the actual scene. Focus on describing content and action, not colors."
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
                logger.error(f"Out of memory loading Qwen model. Close other apps or use a cloud vision API. Error: {e}")
                # Return empty result - frontend can fall back to cloud API if configured
                return ObjectsResult(
                    summary={},
                    detections=[],
                    descriptions=None,
                    error="out_of_memory",
                )
            raise  # Re-raise other errors

        logger.info(f"Processing {total_frames} frames for Qwen analysis")

        # Get strategy for multi-frame analysis (use override if provided)
        if strategy is not None:
            resolved_strategy = QwenStrategy(strategy)
            logger.info(f"Using Qwen strategy override: {resolved_strategy}")
        else:
            resolved_strategy = settings.get_qwen_strategy()
            logger.info(f"Using Qwen strategy from config: {resolved_strategy}")

        # Dispatch to appropriate strategy implementation
        if resolved_strategy == QwenStrategy.SINGLE:
            all_objects, detections, descriptions = _analyze_frames_single(model, processor, torch_device, frame_paths, timestamps, context, progress_callback)
        elif resolved_strategy == QwenStrategy.CONTEXT:
            all_objects, detections, descriptions = _analyze_frames_with_context(model, processor, torch_device, frame_paths, timestamps, context, progress_callback)
        elif resolved_strategy == QwenStrategy.BATCH:
            all_objects, detections, descriptions = _analyze_frames_batch(
                model,
                processor,
                torch_device,
                frame_paths,
                timestamps,
                context,
                progress_callback,
                overlap=batch_overlap,
            )
        else:  # BATCH_CONTEXT
            all_objects, detections, descriptions = _analyze_frames_batch_context(
                model,
                processor,
                torch_device,
                frame_paths,
                timestamps,
                context,
                progress_callback,
                overlap=batch_overlap,
            )

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
        clean_response = _fix_malformed_json(response)

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

"""Batch job processor - main extraction logic."""

from __future__ import annotations

import gc
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from media_engine.batch.models import (
    BatchRequest,
    ExtractorTiming,
    JobProgress,
)
from media_engine.batch.queue import cleanup_expired_batch_jobs, start_next_batch
from media_engine.batch.state import batch_jobs, batch_jobs_lock
from media_engine.batch.timing import (
    EXTRACTOR_ORDER,
    calculate_queue_eta,
    get_enabled_extractors_from_request,
    get_predicted_rate,
    get_resolution_bucket,
    predict_extractor_time,
    record_timing,
)
from media_engine.utils.memory import clear_memory, get_memory_mb

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def run_batch_job(batch_id: str, request: BatchRequest) -> None:
    """Run batch extraction - processes all files per extractor stage.

    This is more memory efficient as each model is loaded once,
    processes all files, then is unloaded before the next model.
    """
    from media_engine.config import get_settings
    from media_engine.extractors import (
        FFPROBE_WORKERS,
        SharedFrameBuffer,
        analyze_motion,
        check_faces_are_known,
        decode_frames,
        detect_voice_activity,
        extract_clip,
        extract_faces,
        extract_metadata,
        extract_objects,
        extract_objects_qwen,
        extract_ocr,
        extract_scenes,
        extract_telemetry,
        extract_transcript,
        get_adaptive_timestamps,
        get_extractor_timestamps,
        get_sample_timestamps,
        run_ffprobe_batch,
        unload_clip_model,
        unload_face_model,
        unload_ocr_model,
        unload_qwen_model,
        unload_vad_model,
        unload_whisper_model,
        unload_yolo_model,
    )
    from media_engine.extractors.vad import AudioContent
    from media_engine.schemas import (
        BoundingBox,
        FaceDetection,
        FacesResult,
        MediaType,
        get_media_type,
    )

    settings = get_settings()

    # Resolve models from settings (handles "auto" -> actual model name)
    whisper_model = settings.get_whisper_model()
    qwen_model = settings.get_qwen_model()
    yolo_model = settings.get_yolo_model()
    clip_model = settings.get_clip_model()

    logger.info(f"Batch {batch_id} models: whisper={whisper_model}, qwen={qwen_model}, yolo={yolo_model}, clip={clip_model}")

    batch_start_time = time.time()
    peak_memory = get_memory_mb()
    stage_start_times: dict[str, float] = {}  # extractor -> start time
    file_resolutions: dict[int, str] = {}  # file_idx -> resolution bucket (for timing predictions)
    file_durations: dict[int, float] = {}  # file_idx -> duration in seconds

    # Get enabled extractors for this batch
    enabled_extractors, enabled_sub_extractors = get_enabled_extractors_from_request(request)

    def calculate_total_eta(current_extractor: str, stage_eta: float) -> float:
        """Calculate total remaining time for the entire batch.

        Args:
            current_extractor: Currently running extractor
            stage_eta: Remaining time for current stage

        Returns:
            Total estimated remaining seconds for the batch
        """
        total_eta = stage_eta if stage_eta else 0.0

        # Get the current extractor's position in the order
        if current_extractor not in EXTRACTOR_ORDER:
            return total_eta

        current_ext_idx = EXTRACTOR_ORDER.index(current_extractor)
        num_files = len(request.files)

        # Add time for remaining extractors (after current one)
        remaining_extractors = EXTRACTOR_ORDER[current_ext_idx + 1 :]
        logger.info(f"ETA calc: current={current_extractor}, remaining={remaining_extractors}, enabled={enabled_extractors}")

        for ext in remaining_extractors:
            if ext not in enabled_extractors:
                logger.info(f"ETA calc: skipping {ext} (not enabled)")
                continue

            # Sum predicted time across all files
            for file_idx in range(num_files):
                resolution = file_resolutions.get(file_idx, "1080p")
                duration = file_durations.get(file_idx, 60.0)  # Default 1 min
                predicted = predict_extractor_time(
                    ext,
                    resolution,
                    duration,
                    enabled_sub_extractors=enabled_sub_extractors if ext == "visual_processing" else None,
                )
                total_eta += predicted
                logger.info(f"ETA calc: {ext} file={file_idx} res={resolution} dur={duration}s -> +{predicted:.1f}s")

        return round(total_eta, 1)

    def update_batch_progress(
        extractor: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        nonlocal peak_memory

        # Track stage start time
        if extractor not in stage_start_times:
            stage_start_times[extractor] = time.time()

        # Calculate ETA
        stage_elapsed: float | None = None
        eta: float | None = None
        if extractor in stage_start_times:
            stage_elapsed = round(time.time() - stage_start_times[extractor], 1)
            # Calculate ETA if we have progress info
            if current is not None and total is not None and current > 0:
                # Always use prediction-based ETA for remaining files + current file
                # This is more accurate than elapsed-based calculation
                eta = 0.0

                # Add predicted time for current file (estimate 50% remaining)
                current_file_idx = current - 1
                current_res = file_resolutions.get(current_file_idx, "1080p")
                current_dur = file_durations.get(current_file_idx, 60.0)
                current_predicted = predict_extractor_time(
                    extractor,
                    current_res,
                    current_dur,
                    enabled_sub_extractors=enabled_sub_extractors if extractor == "visual_processing" else None,
                )
                # Estimate we're ~halfway through current file if we have some elapsed time
                if stage_elapsed > 0 and current > 1:
                    # For files after the first, estimate based on avg time per completed file
                    avg_per_file = stage_elapsed / (current - 1)
                    eta += max(0, avg_per_file * 0.5)  # ~50% of avg remaining for current
                else:
                    eta += current_predicted * 0.5  # ~50% of predicted remaining for current

                # Add predicted time for remaining files
                for file_idx in range(current, total):
                    resolution = file_resolutions.get(file_idx, "1080p")
                    duration = file_durations.get(file_idx, 60.0)
                    eta += predict_extractor_time(
                        extractor,
                        resolution,
                        duration,
                        enabled_sub_extractors=enabled_sub_extractors if extractor == "visual_processing" else None,
                    )

                eta = round(eta, 1)
            elif current == 0 and total is not None and total > 0:
                # No progress yet - try to use historical timing for prediction
                # Use the most common resolution in the batch, or "unknown" if none set
                common_res = "unknown"
                if file_resolutions:
                    res_counts: dict[str, int] = {}
                    for res in file_resolutions.values():
                        res_counts[res] = res_counts.get(res, 0) + 1
                    common_res = max(res_counts, key=lambda r: res_counts[r])
                predicted = get_predicted_rate(extractor, common_res)
                if predicted is not None:
                    eta = round(predicted * total, 1)

        # Calculate total ETA for entire batch (current stage + remaining stages)
        total_eta = calculate_total_eta(extractor, eta or 0.0)

        # Debug logging for ETA calculation (use INFO level to see it)
        if total_eta and total_eta > 0:
            logger.info(f"ETA: {extractor} stage={eta}s, total={total_eta}s, subs={enabled_sub_extractors}, files={len(file_durations)}")

        # Calculate queue ETA (for all queued batches)
        queue_eta, queued_count = calculate_queue_eta()

        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].current_extractor = extractor
                batch_jobs[batch_id].progress = JobProgress(
                    message=message,
                    current=current,
                    total=total,
                    stage_elapsed_seconds=stage_elapsed,
                    eta_seconds=eta,
                    # Always send total_eta - even if 0, it's valid info
                    total_eta_seconds=total_eta,
                    queue_eta_seconds=queue_eta if queue_eta > 0 else None,
                    queued_batches=queued_count if queued_count > 0 else None,
                )
                # Update memory and elapsed time
                current_mem = get_memory_mb()
                peak_memory = max(peak_memory, current_mem)
                batch_jobs[batch_id].memory_mb = current_mem
                batch_jobs[batch_id].peak_memory_mb = peak_memory
                batch_jobs[batch_id].elapsed_seconds = round(time.time() - batch_start_time, 1)

    def update_file_status(
        file_idx: int,
        status: str,
        result_key: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].status = status
                if result_key and result is not None:
                    batch_jobs[batch_id].files[file_idx].results[result_key] = result
                if error:
                    batch_jobs[batch_id].files[file_idx].error = error

    def update_extractor_status(file_idx: int, extractor: str, status: str) -> None:
        """Update extractor status for a file.

        Args:
            file_idx: Index of the file in the batch
            extractor: Name of the extractor
            status: One of 'pending', 'active', 'completed', 'failed', 'skipped'
        """
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].extractor_status[extractor] = status

    def start_extractor_timing(extractor: str) -> datetime:
        """Start timing for an extractor stage."""
        started = datetime.now(timezone.utc)
        # Reset stage start time for ETA calculation
        stage_start_times[extractor] = time.time()
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].extractor_timings.append(ExtractorTiming(extractor=extractor, started_at=started))
        return started

    def end_extractor_timing(extractor: str, files_processed: int) -> None:
        """End timing for an extractor stage."""
        completed = datetime.now(timezone.utc)
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                for timing in batch_jobs[batch_id].extractor_timings:
                    if timing.extractor == extractor and timing.completed_at is None:
                        timing.completed_at = completed
                        timing.duration_seconds = round((completed - timing.started_at).total_seconds(), 2)
                        timing.files_processed = files_processed
                        break

    def update_file_timing(file_idx: int, extractor: str, duration: float, units: float | None = None) -> None:
        """Record per-file timing for an extractor.

        Args:
            file_idx: Index of the file in the batch
            extractor: Name of the extractor
            duration: Wall clock seconds to process
            units: Normalization units for rate calculation:
                - transcript: duration in minutes
                - visual: number of timestamps
                - objects/faces/ocr/clip: number of frames
                - None: store raw seconds (metadata, telemetry, etc.)
        """
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].timings[extractor] = round(duration, 2)
        # Record to historical timing for future ETA predictions
        resolution = file_resolutions.get(file_idx, "unknown")
        record_timing(extractor, resolution, duration, units)

    try:
        with batch_jobs_lock:
            batch_jobs[batch_id].status = "running"

        files = request.files
        total_files = len(files)

        # Track files that failed metadata extraction - skip them in all subsequent stages
        # If we can't read the file with ffprobe, there's no point trying other extractors
        failed_files: set[int] = set()

        # Always get file durations and resolutions for ETA predictions (lightweight ffprobe)
        # This runs even if metadata isn't enabled
        if not request.enable_metadata:
            from media_engine.extractors.metadata.base import get_video_info

            for i, file_path in enumerate(files):
                try:
                    _fps, duration, width, height = get_video_info(file_path)
                    if duration:
                        file_durations[i] = duration
                    # Determine resolution bucket from dimensions
                    file_resolutions[i] = get_resolution_bucket(width, height)
                    logger.info(f"ETA: file {i} duration={duration}s, res={file_resolutions[i]}")
                except Exception as e:
                    logger.warning(f"Could not get video info for {file_path}: {e}")

        # Stage 1: Metadata (parallel ffprobe for speed)
        if request.enable_metadata:
            start_extractor_timing("metadata")
            update_batch_progress(
                "metadata",
                f"Running ffprobe ({FFPROBE_WORKERS} parallel workers)...",
                0,
                total_files,
            )

            # Run all ffprobe calls in parallel
            probe_results = run_ffprobe_batch(files)

            # Extract metadata from each probe result
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("metadata", f"Processing {Path(file_path).name}", i + 1, total_files)
                update_extractor_status(i, "metadata", "active")
                probe_data = probe_results.get(file_path)

                if isinstance(probe_data, Exception):
                    logger.warning(f"Metadata failed for {file_path}: {probe_data}")
                    logger.warning(f"Skipping all extractors for {file_path} - file unreadable")
                    update_file_status(i, "failed", "metadata", None, str(probe_data))
                    update_extractor_status(i, "metadata", "failed")
                    update_file_timing(i, "metadata", time.time() - file_start)
                    failed_files.add(i)
                    continue

                try:
                    metadata = extract_metadata(file_path, probe_data)
                    update_file_status(i, "running", "metadata", metadata.model_dump())
                    update_extractor_status(i, "metadata", "completed")
                    # Store resolution bucket for timing predictions
                    file_resolutions[i] = get_resolution_bucket(
                        metadata.resolution.width,
                        metadata.resolution.height,
                    )
                    # Store duration for total ETA predictions
                    if metadata.duration is not None:
                        file_durations[i] = metadata.duration
                        logger.info(f"ETA: stored duration {metadata.duration}s for file {i}")
                except Exception as e:
                    logger.warning(f"Metadata failed for {file_path}: {e}")
                    logger.warning(f"Skipping all extractors for {file_path} - file unreadable")
                    update_file_status(i, "failed", "metadata", None, str(e))
                    update_extractor_status(i, "metadata", "failed")
                    failed_files.add(i)
                update_file_timing(i, "metadata", time.time() - file_start)
            end_extractor_timing("metadata", total_files)

        # Stage 2: Telemetry (always runs - lightweight, no models)
        start_extractor_timing("telemetry")
        update_batch_progress("telemetry", "Extracting telemetry...", 0, total_files)
        for i, file_path in enumerate(files):
            file_start = time.time()
            update_batch_progress("telemetry", f"Processing {Path(file_path).name}", i + 1, total_files)
            update_extractor_status(i, "telemetry", "active")
            try:
                telemetry = extract_telemetry(file_path)
                update_file_status(
                    i,
                    "running",
                    "telemetry",
                    telemetry.model_dump() if telemetry else None,
                )
                update_extractor_status(i, "telemetry", "completed")
            except Exception as e:
                logger.warning(f"Telemetry failed for {file_path}: {e}")
                update_extractor_status(i, "telemetry", "failed")
            update_file_timing(i, "telemetry", time.time() - file_start)
        end_extractor_timing("telemetry", total_files)

        # Stage 3: Voice Activity Detection (WebRTC VAD - lightweight)
        # Skip for images and files without audio tracks
        if request.enable_vad:
            start_extractor_timing("vad")
            update_batch_progress("vad", "Analyzing audio...", 0, total_files)
            vad_ran = False  # Track if we actually ran VAD on any file
            for i, file_path in enumerate(files):
                if i in failed_files:
                    update_extractor_status(i, "vad", "skipped")
                    continue
                file_start = time.time()
                update_extractor_status(i, "vad", "active")

                # Check media type - skip VAD for images
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
                    logger.info(f"Skipping VAD for {file_path} - image file")
                    no_audio_result = {
                        "audio_content": str(AudioContent.NO_AUDIO),
                        "speech_ratio": 0.0,
                        "speech_segments": [],
                        "total_duration": 0.0,
                    }
                    update_file_status(i, "running", "vad", no_audio_result)
                    update_extractor_status(i, "vad", "completed")
                    update_file_timing(i, "vad", time.time() - file_start)
                    continue

                # Check if metadata shows no audio track
                has_audio_track = True
                with batch_jobs_lock:
                    file_results = batch_jobs[batch_id].files[i].results
                    if file_results and file_results.get("metadata"):
                        metadata = file_results["metadata"]
                        if metadata.get("audio") is None:
                            has_audio_track = False

                if not has_audio_track:
                    logger.info(f"Skipping VAD for {file_path} - no audio track")
                    no_audio_result = {
                        "audio_content": str(AudioContent.NO_AUDIO),
                        "speech_ratio": 0.0,
                        "speech_segments": [],
                        "total_duration": 0.0,
                    }
                    update_file_status(i, "running", "vad", no_audio_result)
                    update_extractor_status(i, "vad", "completed")
                    update_file_timing(i, "vad", time.time() - file_start)
                    continue

                # Run VAD for files with audio
                update_batch_progress("vad", f"Analyzing {Path(file_path).name}", i + 1, total_files)
                try:
                    vad_result = detect_voice_activity(file_path)
                    update_file_status(i, "running", "vad", vad_result)
                    update_extractor_status(i, "vad", "completed")
                    vad_ran = True
                except Exception as e:
                    logger.warning(f"VAD failed for {file_path}: {e}")
                    update_extractor_status(i, "vad", "failed")
                update_file_timing(i, "vad", time.time() - file_start)

            # Only unload if we actually loaded the model
            if vad_ran:
                update_batch_progress("vad", "Unloading VAD model...", None, None)
                unload_vad_model()
            end_extractor_timing("vad", total_files)

        # Stage 4: Per-file visual processing
        # Process each file completely before moving to next (memory efficient)
        # Order: Motion → Scenes → Decode frames → Objects → Faces → OCR → CLIP → Release buffer
        #
        # This approach:
        # - Decodes frames once per file
        # - Runs all visual extractors on those frames
        # - Releases buffer before processing next file
        # - Keeps only one file's frames in memory at a time

        needs_visual_processing = any(
            [
                request.enable_motion,
                request.enable_scenes,
                request.enable_objects,
                request.enable_faces,
                request.enable_ocr,
                request.enable_clip,
            ]
        )

        # Track motion data for adaptive timestamps
        motion_data: dict[int, Any] = {}
        adaptive_timestamps: dict[int, list[float]] = {}

        # Track person timestamps for smart face detection
        person_timestamps: dict[int, list[float]] = {}

        # Skip motion analysis if timestamps are already provided
        has_precomputed_timestamps = bool(request.visual_timestamps)

        if needs_visual_processing:
            start_extractor_timing("visual_processing")
            update_batch_progress(
                "visual_processing",
                "Processing video frames...",
                0,
                total_files,
            )

            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue

                fname = Path(file_path).name
                media_type = get_media_type(file_path)
                file_start = time.time()

                update_batch_progress(
                    "visual_processing",
                    f"Processing {fname}",
                    i + 1,
                    total_files,
                )

                # --- Motion Analysis ---
                if request.enable_motion or (
                    (request.enable_objects or request.enable_faces or request.enable_clip or request.enable_ocr)
                    and not has_precomputed_timestamps
                    and media_type != MediaType.IMAGE
                ):
                    motion_start = time.time()
                    update_extractor_status(i, "motion", "active")
                    try:
                        if media_type == MediaType.IMAGE:
                            motion_data[i] = None
                            adaptive_timestamps[i] = [0.0]
                            update_extractor_status(i, "motion", "completed")
                        else:
                            motion = analyze_motion(file_path)
                            motion_data[i] = motion
                            adaptive_timestamps[i] = get_adaptive_timestamps(motion)

                            # Always store motion data when computed (needed for Pass 2 timestamps)
                            motion_result = {
                                "duration": motion.duration,
                                "fps": motion.fps,
                                "primary_motion": motion.primary_motion.value,
                                "avg_intensity": float(motion.avg_intensity),
                                "is_stable": bool(motion.is_stable),
                                "segments": [
                                    {
                                        "start": seg.start,
                                        "end": seg.end,
                                        "motion_type": seg.motion_type.value,
                                        "intensity": float(seg.intensity),
                                    }
                                    for seg in motion.segments
                                ],
                            }
                            update_file_status(i, "running", "motion", motion_result)
                            update_extractor_status(i, "motion", "completed")
                            logger.info(f"Motion for {fname}: stable={motion.is_stable}, timestamps={len(adaptive_timestamps[i])}")
                    except Exception as e:
                        logger.warning(f"Motion analysis failed for {file_path}: {e}")
                        update_extractor_status(i, "motion", "failed")
                        motion_data[i] = None
                        # Fallback: generate uniform timestamps from duration
                        # This ensures visual extractors still run even if motion fails
                        file_result = batch_jobs[batch_id].files[i]
                        meta = file_result.results.get("metadata")
                        if meta and meta.get("duration"):
                            duration = meta["duration"]
                            # Generate ~10 uniform timestamps
                            num_samples = min(10, max(3, int(duration / 10)))
                            step = duration / (num_samples + 1)
                            fallback_ts = [step * (j + 1) for j in range(num_samples)]
                            adaptive_timestamps[i] = fallback_ts
                            logger.info(f"Using fallback timestamps for {fname}: {num_samples} uniform samples")
                        else:
                            adaptive_timestamps[i] = []
                    update_file_timing(i, "motion", time.time() - motion_start)

                # --- Scene Detection ---
                if request.enable_scenes and media_type != MediaType.IMAGE:
                    scenes_start = time.time()
                    update_extractor_status(i, "scenes", "active")
                    try:
                        scenes = extract_scenes(file_path)
                        update_file_status(
                            i,
                            "running",
                            "scenes",
                            scenes.model_dump() if scenes else None,
                        )
                        update_extractor_status(i, "scenes", "completed")
                    except Exception as e:
                        logger.warning(f"Scenes failed for {file_path}: {e}")
                        update_extractor_status(i, "scenes", "failed")
                    update_file_timing(i, "scenes", time.time() - scenes_start)

                # --- Decode Frames (for Objects, Faces, OCR, CLIP) ---
                buffer: SharedFrameBuffer | None = None
                visual_extractors_needed = any(
                    [
                        request.enable_objects,
                        request.enable_faces,
                        request.enable_ocr,
                        request.enable_clip,
                    ]
                )

                if visual_extractors_needed:
                    decode_start = time.time()
                    update_extractor_status(i, "frame_decode", "active")
                    motion = motion_data.get(i)
                    timestamps = adaptive_timestamps.get(i, [])

                    # Use precomputed timestamps if provided for this file
                    if has_precomputed_timestamps and request.visual_timestamps:
                        file_timestamps = request.visual_timestamps.get(file_path)
                        if file_timestamps:
                            timestamps = file_timestamps

                    # Apply motion-based filtering for stable footage
                    if motion and motion.is_stable and timestamps:
                        timestamps = get_extractor_timestamps(motion.is_stable, motion.avg_intensity, timestamps)

                    # For images, use timestamp 0
                    if media_type == MediaType.IMAGE:
                        timestamps = [0.0]

                    if timestamps:
                        try:
                            buffer = decode_frames(
                                file_path,
                                timestamps=timestamps,
                                max_dimension=1920,
                            )
                            logger.info(f"Decoded {len(buffer.frames)}/{len(timestamps)} frames for {fname}")
                            update_extractor_status(i, "frame_decode", "completed")
                        except Exception as e:
                            logger.warning(f"Frame decode failed for {file_path}: {e}")
                            update_extractor_status(i, "frame_decode", "failed")
                    else:
                        update_extractor_status(i, "frame_decode", "skipped")
                    # Pass frame count as units for per-frame rate calculation
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(i, "frame_decode", time.time() - decode_start, num_frames)

                # --- Objects (YOLO) ---
                if request.enable_objects and buffer is not None:
                    objects_start = time.time()
                    update_extractor_status(i, "objects", "active")
                    try:
                        objects = extract_objects(
                            file_path,
                            frame_buffer=buffer,
                            model_name=yolo_model,
                        )
                        if objects:
                            update_file_status(i, "running", "objects", {"summary": objects.summary})
                            # Collect person timestamps for smart face sampling
                            person_ts = list(set(d.timestamp for d in objects.detections if d.label == "person"))
                            person_timestamps[i] = sorted(person_ts)
                            if person_ts:
                                logger.info(f"Found {len(person_ts)} person frames in {fname}")
                        else:
                            person_timestamps[i] = []
                        update_extractor_status(i, "objects", "completed")
                    except Exception as e:
                        logger.warning(f"Objects failed for {file_path}: {e}")
                        person_timestamps[i] = []
                        update_extractor_status(i, "objects", "failed")
                    # Use number of frames as units for rate calculation
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(i, "objects", time.time() - objects_start, num_frames)

                # --- Faces ---
                if request.enable_faces:
                    faces_start = time.time()
                    face_frame_count: int | None = None
                    update_extractor_status(i, "faces", "active")
                    try:
                        person_ts = person_timestamps.get(i, [])
                        motion = motion_data.get(i)

                        # Get video duration from motion data or metadata results
                        duration = 0.0
                        if motion is not None:
                            duration = motion.duration
                        else:
                            file_result = batch_jobs[batch_id].files[i]
                            if file_result.results.get("metadata"):
                                duration = file_result.results["metadata"].get("duration", 0.0)

                        # Calculate FPS based on motion intensity
                        intensity = motion.avg_intensity if motion else 0.0
                        if intensity >= 10.0:
                            face_fps = 3.0
                        elif intensity >= 6.0:
                            face_fps = 2.0
                        elif intensity >= 2.0:
                            face_fps = 1.5
                        else:
                            face_fps = 1.0

                        # Adaptive face detection for long videos
                        # Short videos (<60s): process all at once
                        # Long videos: use batched approach with early exit when faces stabilize
                        batch_duration = 30.0  # Process 30s at a time
                        verification_interval = 10.0  # Check every 10s once stable
                        min_consistent_batches = 2  # Need 2 batches of same faces to go sparse

                        faces = None
                        face_frame_count = 0
                        all_detections: list[dict[str, Any]] = []
                        known_embeddings: list[list[float]] = []
                        consistent_batches = 0
                        in_verification_mode = False

                        if duration <= 60.0:
                            # Short video - process all at once
                            num_samples = max(1, int(duration * face_fps))
                            step = duration / (num_samples + 1)
                            face_timestamps = [step * (j + 1) for j in range(num_samples)]

                            # Merge with YOLO person timestamps
                            if person_ts:
                                all_ts = sorted(set(person_ts + face_timestamps))
                                merged_ts: list[float] = []
                                for ts in all_ts:
                                    if not merged_ts or ts - merged_ts[-1] >= 0.3:
                                        merged_ts.append(ts)
                                face_timestamps = merged_ts

                            if face_timestamps:
                                face_buffer = decode_frames(file_path, timestamps=face_timestamps)
                                faces = extract_faces(file_path, frame_buffer=face_buffer)
                                face_frame_count = len(face_buffer.frames)
                                logger.info(f"Face detection on {face_frame_count} frames for {fname} (short video, {face_fps} FPS)")
                        else:
                            # Long video - use adaptive batching
                            current_time = 0.0
                            total_frames = 0

                            while current_time < duration:
                                # Determine batch parameters
                                if in_verification_mode:
                                    # Sparse verification: just check every 10s
                                    batch_end = min(current_time + verification_interval, duration)
                                    batch_timestamps = [current_time + verification_interval / 2]
                                else:
                                    # Normal dense sampling
                                    batch_end = min(current_time + batch_duration, duration)
                                    batch_dur = batch_end - current_time
                                    num_batch_samples = max(1, int(batch_dur * face_fps))
                                    step = batch_dur / (num_batch_samples + 1)
                                    batch_timestamps = [current_time + step * (j + 1) for j in range(num_batch_samples)]

                                    # Add YOLO person timestamps in this range
                                    batch_person_ts = [ts for ts in person_ts if current_time <= ts < batch_end]
                                    if batch_person_ts:
                                        all_ts = sorted(set(batch_person_ts + batch_timestamps))
                                        merged_ts = []
                                        for ts in all_ts:
                                            if not merged_ts or ts - merged_ts[-1] >= 0.3:
                                                merged_ts.append(ts)
                                        batch_timestamps = merged_ts

                                # Process this batch
                                if batch_timestamps:
                                    batch_buffer = decode_frames(file_path, timestamps=batch_timestamps)
                                    batch_faces = extract_faces(file_path, frame_buffer=batch_buffer)
                                    total_frames += len(batch_buffer.frames)

                                    if batch_faces and batch_faces.detections:
                                        # Add detections to our collection
                                        for d in batch_faces.detections:
                                            all_detections.append(
                                                {
                                                    "timestamp": d.timestamp,
                                                    "bbox": d.bbox.model_dump(),
                                                    "confidence": d.confidence,
                                                    "embedding": d.embedding,
                                                    "image_base64": d.image_base64,
                                                    "needs_review": d.needs_review,
                                                    "review_reason": d.review_reason,
                                                }
                                            )

                                        # Check if faces are all known
                                        all_known, new_embs = check_faces_are_known(batch_faces, known_embeddings)

                                        if new_embs:
                                            # New faces found - add to known and reset consistency
                                            known_embeddings.extend(new_embs)
                                            consistent_batches = 0
                                            if in_verification_mode:
                                                logger.info(f"New face detected at {current_time:.1f}s, exiting verification mode")
                                                in_verification_mode = False
                                        elif all_known and known_embeddings:
                                            # All faces are known
                                            consistent_batches += 1
                                            if consistent_batches >= min_consistent_batches and not in_verification_mode:
                                                in_verification_mode = True
                                                logger.info(f"Faces stable after {current_time:.1f}s, switching to verification mode (every 10s)")
                                    elif not known_embeddings:
                                        # No faces in this batch and no known faces yet
                                        consistent_batches += 1
                                        if consistent_batches >= min_consistent_batches:
                                            in_verification_mode = True

                                current_time = batch_end

                            face_frame_count = total_frames

                            # Create result from collected detections
                            if all_detections:
                                # Reconstruct FacesResult from batched detections
                                faces = FacesResult(
                                    count=len(all_detections),
                                    unique_estimate=len(known_embeddings),
                                    detections=[
                                        FaceDetection(
                                            timestamp=d["timestamp"],
                                            bbox=BoundingBox(**d["bbox"]),
                                            confidence=d["confidence"],
                                            embedding=d["embedding"],
                                            image_base64=d["image_base64"],
                                            needs_review=d.get("needs_review", False),
                                            review_reason=d.get("review_reason"),
                                        )
                                        for d in all_detections
                                    ],
                                )

                            mode_info = "verification" if in_verification_mode else "normal"
                            logger.info(f"Face detection on {total_frames} frames for {fname} (adaptive batching, {len(known_embeddings)} unique, ended in {mode_info} mode)")

                        # Fallback if no duration info
                        if faces is None and buffer is not None:
                            faces = extract_faces(file_path, frame_buffer=buffer)
                            face_frame_count = len(buffer.frames)
                            logger.info(f"Face detection on {len(buffer.frames)} frames for {fname} (using shared buffer)")

                        if faces:
                            faces_data = {
                                "count": faces.count,
                                "unique_estimate": faces.unique_estimate,
                                "detections": [
                                    {
                                        "timestamp": d.timestamp,
                                        "bbox": d.bbox.model_dump(),
                                        "confidence": d.confidence,
                                        "embedding": d.embedding,
                                        "image_base64": d.image_base64,
                                        "needs_review": d.needs_review,
                                        "review_reason": d.review_reason,
                                    }
                                    for d in faces.detections
                                ],
                            }
                            update_file_status(i, "running", "faces", faces_data)
                        else:
                            update_file_status(
                                i,
                                "running",
                                "faces",
                                {"count": 0, "unique_estimate": 0, "detections": []},
                            )
                        update_extractor_status(i, "faces", "completed")
                    except Exception as e:
                        logger.warning(f"Faces failed for {file_path}: {e}")
                        update_extractor_status(i, "faces", "failed")
                    update_file_timing(i, "faces", time.time() - faces_start, face_frame_count)

                # --- OCR ---
                if request.enable_ocr and buffer is not None:
                    ocr_start = time.time()
                    update_extractor_status(i, "ocr", "active")
                    try:
                        ocr = extract_ocr(file_path, frame_buffer=buffer)
                        update_file_status(i, "running", "ocr", ocr.model_dump() if ocr else None)
                        update_extractor_status(i, "ocr", "completed")
                    except Exception as e:
                        logger.warning(f"OCR failed for {file_path}: {e}")
                        update_extractor_status(i, "ocr", "failed")
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(i, "ocr", time.time() - ocr_start, num_frames)

                # --- CLIP ---
                if request.enable_clip and buffer is not None:
                    clip_start = time.time()
                    update_extractor_status(i, "clip", "active")
                    try:
                        clip = extract_clip(
                            file_path,
                            frame_buffer=buffer,
                            model_name=clip_model,
                        )
                        if clip:
                            update_file_status(i, "running", "clip", clip.model_dump())
                        else:
                            update_file_status(i, "running", "clip", None)
                        update_extractor_status(i, "clip", "completed")
                    except Exception as e:
                        logger.warning(f"CLIP failed for {file_path}: {e}")
                        update_extractor_status(i, "clip", "failed")
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(i, "clip", time.time() - clip_start, num_frames)

                # --- Release buffer for this file ---
                if buffer is not None:
                    logger.info(f"Releasing frame buffer for {fname}")
                    del buffer
                    gc.collect()

                # Update peak memory after each file
                peak_memory = max(peak_memory, get_memory_mb())

            # Unload all visual models after processing all files
            update_batch_progress("visual_processing", "Unloading models...", None, None)
            if request.enable_objects:
                unload_yolo_model()
            if request.enable_faces:
                unload_face_model()
            if request.enable_ocr:
                unload_ocr_model()
            if request.enable_clip:
                unload_clip_model()

            end_extractor_timing("visual_processing", total_files)

        # Stage 5: Visual (Qwen VLM - scene descriptions)
        # Separate stage because Qwen is very heavy and has its own frame handling
        if request.enable_visual:
            start_extractor_timing("visual")
            logger.info("Visual enabled (Qwen VLM)")
            clear_memory()
            update_batch_progress("visual", "Loading Qwen model...", 0, total_files)
            logger.info(f"Qwen batch contexts: {request.contexts}")

            for i, file_path in enumerate(files):
                if i in failed_files:
                    update_extractor_status(i, "visual", "skipped")
                    continue
                file_start = time.time()
                fname = Path(file_path).name
                update_batch_progress("visual", f"Analyzing: {fname}", i + 1, total_files)
                update_extractor_status(i, "visual", "active")
                # Get per-file timestamps if provided (declared before try so it's visible after)
                timestamps: list[float] | None = None
                try:
                    motion = motion_data.get(i)
                    if request.visual_timestamps:
                        timestamps = request.visual_timestamps.get(file_path)
                    if timestamps is None and motion:
                        timestamps = get_sample_timestamps(motion, max_samples=5)

                    file_context = request.contexts.get(file_path) if request.contexts else None
                    file_batch_overlap = request.visual_batch_overlap.get(file_path, False) if request.visual_batch_overlap else False
                    file_strategy = request.visual_strategy.get(file_path) if request.visual_strategy else None
                    logger.info(f"Calling Qwen for {fname}: context={file_context}, lut_path={request.lut_path}, batch_overlap={file_batch_overlap}, strategy={file_strategy}")
                    visual_result = extract_objects_qwen(
                        file_path,
                        timestamps=timestamps,
                        model_name=qwen_model,
                        context=file_context,
                        lut_path=request.lut_path,
                        batch_overlap=file_batch_overlap,
                        strategy=file_strategy,
                    )
                    visual_data: dict[str, Any] = {"summary": visual_result.summary}
                    if visual_result.descriptions:
                        visual_data["descriptions"] = visual_result.descriptions
                    update_file_status(i, "running", "visual", visual_data)
                    update_extractor_status(i, "visual", "completed")
                except Exception as e:
                    logger.warning(f"Visual failed for {file_path}: {e}", exc_info=True)
                    update_extractor_status(i, "visual", "failed")
                    update_file_status(i, "failed", error=str(e))
                    failed_files.add(i)
                # Use number of timestamps as units for rate calculation
                num_timestamps = len(timestamps) if timestamps else None
                update_file_timing(i, "visual", time.time() - file_start, num_timestamps)

            update_batch_progress("visual", "Unloading Qwen model...", None, None)
            unload_qwen_model()
            end_extractor_timing("visual", total_files)

        # Stage 6: Transcript (Whisper - heavy model)
        # Skip for images and files without audio tracks
        if request.enable_transcript:
            start_extractor_timing("transcript")
            whisper_ran = False  # Track if we actually ran Whisper

            # Check if any files need transcription before loading model
            files_to_transcribe: list[int] = []
            for i, file_path in enumerate(files):
                if i in failed_files:
                    update_extractor_status(i, "transcript", "skipped")
                    continue
                # Skip images
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
                    update_extractor_status(i, "transcript", "skipped")
                    continue
                # Check for audio track
                has_audio = True
                with batch_jobs_lock:
                    file_results = batch_jobs[batch_id].files[i].results
                    if file_results and file_results.get("metadata"):
                        if file_results["metadata"].get("audio") is None:
                            has_audio = False
                if has_audio:
                    files_to_transcribe.append(i)
                else:
                    update_extractor_status(i, "transcript", "skipped")

            if files_to_transcribe:
                # Clear memory before loading heavy model
                logger.info("Clearing memory before Whisper...")
                clear_memory()
                update_batch_progress(
                    "transcript",
                    "Loading Whisper model...",
                    0,
                    len(files_to_transcribe),
                )

                for idx, i in enumerate(files_to_transcribe):
                    file_path = files[i]
                    file_start = time.time()
                    update_batch_progress(
                        "transcript",
                        f"Transcribing {Path(file_path).name}",
                        idx + 1,
                        len(files_to_transcribe),
                    )
                    update_extractor_status(i, "transcript", "active")
                    try:
                        transcript = extract_transcript(
                            file_path,
                            model=whisper_model,
                            language=request.language,
                            fallback_language=settings.fallback_language,
                            language_hints=request.language_hints,
                            context_hint=request.context_hint,
                        )
                        update_file_status(
                            i,
                            "running",
                            "transcript",
                            transcript.model_dump() if transcript else None,
                        )
                        update_extractor_status(i, "transcript", "completed")
                        whisper_ran = True
                    except Exception as e:
                        logger.warning(f"Transcript failed for {file_path}: {e}")
                        update_extractor_status(i, "transcript", "failed")
                        update_file_status(i, "failed", error=str(e))
                        failed_files.add(i)
                    # Get duration in minutes for rate calculation
                    duration_minutes: float | None = None
                    with batch_jobs_lock:
                        file_results = batch_jobs[batch_id].files[i].results
                        if file_results and file_results.get("metadata"):
                            duration_sec = file_results["metadata"].get("duration")
                            if duration_sec:
                                duration_minutes = duration_sec / 60.0
                    update_file_timing(i, "transcript", time.time() - file_start, duration_minutes)

                # Unload Whisper to free memory
                if whisper_ran:
                    update_batch_progress("transcript", "Unloading Whisper model...", None, None)
                    unload_whisper_model()
            else:
                logger.info("Skipping Whisper - no files with audio tracks")

            end_extractor_timing("transcript", total_files)

        # Mark files as completed (skip failed files - they stay "failed")
        with batch_jobs_lock:
            for i in range(len(files)):
                if i in failed_files:
                    # File already marked as failed - don't overwrite
                    error_msg = batch_jobs[batch_id].files[i].error or "unknown error"
                    logger.info(f"Batch {batch_id} file {i} marked failed: {error_msg}")
                    continue
                # Log results before marking complete
                result_keys = list(batch_jobs[batch_id].files[i].results.keys())
                logger.info(f"Batch {batch_id} file {i} results before completion: keys={result_keys}")
                batch_jobs[batch_id].files[i].status = "completed"
            batch_jobs[batch_id].status = "completed"
            batch_jobs[batch_id].current_extractor = None
            batch_jobs[batch_id].progress = None
            batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)
            # Final metrics
            batch_jobs[batch_id].elapsed_seconds = round(time.time() - batch_start_time, 2)
            batch_jobs[batch_id].memory_mb = get_memory_mb()
            batch_jobs[batch_id].peak_memory_mb = max(peak_memory, get_memory_mb())

        # Log timing summary
        logger.info(f"Batch {batch_id} completed in {batch_jobs[batch_id].elapsed_seconds}s, peak memory: {batch_jobs[batch_id].peak_memory_mb}MB")
        for timing in batch_jobs[batch_id].extractor_timings:
            logger.info(f"  {timing.extractor}: {timing.duration_seconds}s ({timing.files_processed} files)")

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {e}")
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].status = "failed"
                batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)
                batch_jobs[batch_id].elapsed_seconds = round(time.time() - batch_start_time, 2)
                batch_jobs[batch_id].memory_mb = get_memory_mb()
                batch_jobs[batch_id].peak_memory_mb = peak_memory

    finally:
        # Cleanup old batch jobs to free memory
        cleanup_expired_batch_jobs()

        # Clear memory before starting next batch
        logger.info("Clearing memory after batch completion...")
        clear_memory()

        # Always start the next batch from queue (or set batch_running = False)
        start_next_batch()

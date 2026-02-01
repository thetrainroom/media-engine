#!/usr/bin/env python3
"""Stress/soak test for Polybos Media Engine.

Runs the engine repeatedly with various extractor combinations to verify
stability under sustained load. Monitors memory usage and checks for errors.

Usage:
    # Run with defaults (10 iterations, all small test videos)
    python tests/stress_test.py

    # Run for 50 iterations
    python tests/stress_test.py --iterations 50

    # Run for 1 hour
    python tests/stress_test.py --duration 3600

    # Use specific video
    python tests/stress_test.py --video /path/to/video.mp4

    # Heavy mode (all extractors including transcript)
    python tests/stress_test.py --heavy
"""

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

# Default engine URL (can be overridden via --url flag)
DEFAULT_ENGINE_URL = "http://localhost:8001"

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Test video search paths
VIDEO_SEARCH_PATHS = [
    PROJECT_ROOT / "test_data" / "video",
]


@dataclass
class TestResult:
    """Result of a single test iteration."""

    iteration: int
    batch_id: str
    files: int
    extractors: list[str]
    duration_seconds: float
    memory_mb: int
    peak_memory_mb: int
    success: bool
    error: str | None = None
    warnings: list[str] | None = None


def validate_batch_results(
    batch_result: dict,
    enabled_extractors: list[str],
) -> list[str]:
    """Validate batch results and return list of warnings.

    Checks:
    - All files have results
    - Enabled extractors produced output
    - Metadata has required fields
    - No unexpected errors in file results
    """
    warnings: list[str] = []

    files = batch_result.get("files", [])
    if not files:
        warnings.append("No files in result")
        return warnings

    for file_info in files:
        filename = file_info.get("filename", "unknown")
        status = file_info.get("status")
        results = file_info.get("results", {})

        # Check file status
        if status == "failed":
            error = file_info.get("error", "unknown error")
            warnings.append(f"{filename}: file failed - {error}")
            continue

        if status != "completed":
            warnings.append(f"{filename}: unexpected status '{status}'")
            continue

        # Check if file has audio (for transcript validation)
        has_audio = True
        if "metadata" in results and results["metadata"]:
            if results["metadata"].get("audio") is None:
                has_audio = False

        # Check each enabled extractor has results
        for extractor in enabled_extractors:
            # Skip transcript check for files without audio
            if extractor == "transcript" and not has_audio:
                continue

            if extractor not in results:
                warnings.append(f"{filename}: missing '{extractor}' results")
            elif results[extractor] is None:
                warnings.append(f"{filename}: '{extractor}' returned None")

        # Validate metadata if present
        if "metadata" in results and results["metadata"]:
            meta = results["metadata"]
            required_fields = ["duration", "resolution", "fps"]
            for field in required_fields:
                if field not in meta or meta[field] is None:
                    warnings.append(f"{filename}: metadata missing '{field}'")

            # Check duration is reasonable
            duration = meta.get("duration", 0)
            if duration <= 0:
                warnings.append(f"{filename}: invalid duration {duration}")

        # Validate scenes if present
        if "scenes" in results and results["scenes"]:
            scenes = results["scenes"]
            if "scenes" in scenes and len(scenes["scenes"]) == 0:
                # Zero scenes might be valid for very short clips
                pass

        # Validate objects if present
        # Note: objects only stores summary during processing, not full detections
        if "objects" in results and results["objects"]:
            obj = results["objects"]
            if "summary" not in obj:
                warnings.append(f"{filename}: objects missing 'summary'")

        # Validate faces if present
        if "faces" in results and results["faces"]:
            faces = results["faces"]
            if "detections" not in faces:
                warnings.append(f"{filename}: faces missing 'detections'")
            elif "count" in faces and faces["count"] > 0 and len(faces["detections"]) == 0:
                warnings.append(f"{filename}: faces count={faces['count']} but no detections")

        # Validate OCR if present
        if "ocr" in results and results["ocr"]:
            ocr = results["ocr"]
            if "detections" not in ocr:
                warnings.append(f"{filename}: ocr missing 'detections'")

        # Validate CLIP if present
        if "clip" in results and results["clip"]:
            clip = results["clip"]
            if "segments" not in clip:
                warnings.append(f"{filename}: clip missing 'segments'")
            elif len(clip["segments"]) == 0:
                warnings.append(f"{filename}: clip has no segments (all frames failed?)")

        # Validate visual/Qwen if present
        # Note: visual has summary (always) and descriptions (optional, may be empty for short clips)
        if "visual" in results and results["visual"]:
            visual = results["visual"]
            if "summary" not in visual:
                warnings.append(f"{filename}: visual missing 'summary'")

        # Validate transcript if present
        if "transcript" in results and results["transcript"]:
            transcript = results["transcript"]
            if "segments" not in transcript:
                warnings.append(f"{filename}: transcript missing 'segments'")

    return warnings


def find_test_videos(max_size_mb: int = 100) -> list[str]:
    """Find test videos under the size limit."""
    videos = []
    for search_path in VIDEO_SEARCH_PATHS:
        if not search_path.exists():
            continue
        for pattern in ["**/*.mp4", "**/*.MP4", "**/*.mov", "**/*.MOV"]:
            for path in search_path.glob(pattern):
                size_mb = path.stat().st_size / 1024 / 1024
                if size_mb <= max_size_mb:
                    videos.append(str(path))

    # Sort by size (smallest first)
    videos.sort(key=lambda p: os.path.getsize(p))
    return videos


def get_extractor_configs(heavy: bool = False) -> list[dict]:
    """Get list of extractor configurations to test."""
    configs = [
        # Lightweight
        {"enable_metadata": True},
        {"enable_metadata": True, "enable_vad": True},
        {"enable_metadata": True, "enable_scenes": True},
        {"enable_metadata": True, "enable_motion": True},
        # Medium
        {"enable_metadata": True, "enable_objects": True},
        {"enable_metadata": True, "enable_ocr": True},
        {"enable_metadata": True, "enable_clip": True},
        # Combined visual
        {
            "enable_metadata": True,
            "enable_objects": True,
            "enable_faces": True,
        },
        {
            "enable_metadata": True,
            "enable_objects": True,
            "enable_clip": True,
            "enable_ocr": True,
        },
        # Full visual pipeline
        {
            "enable_metadata": True,
            "enable_scenes": True,
            "enable_objects": True,
            "enable_faces": True,
            "enable_ocr": True,
            "enable_clip": True,
        },
    ]

    if heavy:
        # Add transcript (slow, memory-heavy)
        configs.extend(
            [
                {"enable_metadata": True, "enable_transcript": True},
                {
                    "enable_metadata": True,
                    "enable_transcript": True,
                    "enable_scenes": True,
                    "enable_objects": True,
                },
            ]
        )

        # Add Qwen VLM (very memory-heavy) - run multiple times to catch leaks
        for _ in range(3):
            configs.append({"enable_metadata": True, "enable_visual": True})

        # Combined heavy: Qwen + transcript
        configs.append(
            {
                "enable_metadata": True,
                "enable_visual": True,
                "enable_transcript": True,
            }
        )

        # Full pipeline with everything (ultimate stress test)
        configs.append(
            {
                "enable_metadata": True,
                "enable_scenes": True,
                "enable_transcript": True,
                "enable_objects": True,
                "enable_faces": True,
                "enable_ocr": True,
                "enable_clip": True,
                "enable_visual": True,
            }
        )

    return configs


def run_batch(
    files: list[str],
    extractors: dict,
    engine_url: str,
    timeout: int = 600,
) -> tuple[dict | None, str | None]:
    """Run a batch and wait for completion.

    Returns:
        (result_dict, error_string) - one will be None
    """
    batch_id = None
    try:
        # Create batch
        payload = {"files": files, **extractors}
        response = requests.post(f"{engine_url}/batch", json=payload, timeout=10)
        response.raise_for_status()
        batch_id = response.json()["batch_id"]

        # Poll for completion
        start = time.time()
        while time.time() - start < timeout:
            response = requests.get(f"{engine_url}/batch/{batch_id}", timeout=10)
            response.raise_for_status()
            status = response.json()

            if status["status"] == "completed":
                # Delete batch to free memory on engine
                try:
                    requests.delete(f"{engine_url}/batch/{batch_id}", timeout=5)
                except Exception:
                    pass  # Ignore delete failures
                return status, None
            elif status["status"] == "failed":
                # Delete batch to free memory on engine
                try:
                    requests.delete(f"{engine_url}/batch/{batch_id}", timeout=5)
                except Exception:
                    pass
                return None, f"Batch failed: {status}"

            time.sleep(1)

        return None, f"Timeout after {timeout}s"

    except requests.RequestException as e:
        return None, f"Request error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def check_health(engine_url: str) -> bool:
    """Check if engine is running."""
    try:
        response = requests.get(f"{engine_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_enabled_extractors(config: dict) -> list[str]:
    """Get list of enabled extractor names from config."""
    return [k.replace("enable_", "") for k, v in config.items() if k.startswith("enable_") and v]


def run_stress_test(
    videos: list[str],
    engine_url: str,
    iterations: int | None = None,
    duration_seconds: int | None = None,
    heavy: bool = False,
    files_per_batch: int = 1,
    randomize: bool = True,
    thorough: bool = False,
) -> list[TestResult]:
    """Run the stress test.

    Args:
        videos: List of video paths to use
        engine_url: Engine URL to connect to
        iterations: Number of iterations (None for unlimited)
        duration_seconds: Max duration in seconds (None for unlimited)
        heavy: Include heavy extractors (transcript)
        files_per_batch: Number of files per batch
        randomize: Randomize extractor selection
        thorough: Test every file with every config (overrides iterations)

    Returns:
        List of test results
    """
    results: list[TestResult] = []
    configs = get_extractor_configs(heavy)
    start_time = time.time()
    iteration = 0

    print(f"\n{'=' * 60}")
    print("Polybos Media Engine Stress Test")
    print(f"{'=' * 60}")
    print(f"Videos: {len(videos)}")
    print(f"Extractor configs: {len(configs)}")
    print(f"Files per batch: {files_per_batch}")
    print(f"Heavy mode: {heavy}")
    print(f"Thorough mode: {thorough}")
    if thorough:
        total_tests = len(videos) * len(configs)
        print(f"Total tests (files × configs): {total_tests}")
    elif iterations:
        print(f"Max iterations: {iterations}")
    if duration_seconds:
        print(f"Max duration: {duration_seconds}s")
    print(f"{'=' * 60}\n")

    # Build test queue for thorough mode
    if thorough:
        test_queue: list[tuple[list[str], dict]] = []
        for video in videos:
            for config in configs:
                test_queue.append(([video], config))
        random.shuffle(test_queue)  # Randomize order to spread load
    else:
        test_queue = []

    try:
        while True:
            # Check termination conditions
            if thorough:
                if iteration >= len(test_queue):
                    print(f"\nCompleted all {len(test_queue)} file×config combinations.")
                    break
            else:
                if iterations and iteration >= iterations:
                    print(f"\nReached {iterations} iterations, stopping.")
                    break
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                print(f"\nReached {duration_seconds}s duration, stopping.")
                break

            iteration += 1

            # Select files and config
            if thorough:
                batch_files, config = test_queue[iteration - 1]
            elif randomize:
                batch_files = random.sample(videos, min(files_per_batch, len(videos)))
                config = random.choice(configs)
            else:
                batch_files = videos[:files_per_batch]
                config = configs[iteration % len(configs)]

            extractors = get_enabled_extractors(config)
            file_names = [Path(f).name for f in batch_files]

            print(
                f"[{iteration:4d}] {', '.join(extractors):40s} | " f"files: {', '.join(file_names)[:30]:30s}",
                end=" | ",
                flush=True,
            )

            # Run batch
            batch_start = time.time()
            result, error = run_batch(batch_files, config, engine_url)
            batch_duration = time.time() - batch_start

            if result:
                # Validate results
                warnings = validate_batch_results(result, extractors)

                test_result = TestResult(
                    iteration=iteration,
                    batch_id=result["batch_id"],
                    files=len(batch_files),
                    extractors=extractors,
                    duration_seconds=round(batch_duration, 2),
                    memory_mb=result.get("memory_mb", 0),
                    peak_memory_mb=result.get("peak_memory_mb", 0),
                    success=True,
                    warnings=warnings if warnings else None,
                )

                # Build status indicator
                if warnings:
                    status = f"⚠ {len(warnings)} warnings"
                else:
                    status = "✓"

                print(f"{batch_duration:5.1f}s | " f"mem: {test_result.memory_mb:4d}MB | " f"peak: {test_result.peak_memory_mb:4d}MB | " f"{status}")
            else:
                test_result = TestResult(
                    iteration=iteration,
                    batch_id="",
                    files=len(batch_files),
                    extractors=extractors,
                    duration_seconds=round(batch_duration, 2),
                    memory_mb=0,
                    peak_memory_mb=0,
                    success=False,
                    error=error,
                )
                print(f"{batch_duration:5.1f}s | FAILED: {(error or 'Unknown error')[:50]}")

            results.append(test_result)

            # Brief pause between batches
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    return results


def print_summary(results: list[TestResult], total_duration: float):
    """Print test summary."""
    if not results:
        print("No results to summarize.")
        return

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    with_warnings = [r for r in successful if r.warnings]

    print(f"\n{'=' * 60}")
    print("STRESS TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total iterations: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"With warnings: {len(with_warnings)}")
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Success rate: {len(successful) / len(results) * 100:.1f}%")

    if successful:
        durations = [r.duration_seconds for r in successful]
        memories = [r.memory_mb for r in successful if r.memory_mb > 0]
        peak_memories = [r.peak_memory_mb for r in successful if r.peak_memory_mb > 0]

        print("\nTiming (successful runs):")
        print(f"  Min: {min(durations):.1f}s")
        print(f"  Max: {max(durations):.1f}s")
        print(f"  Avg: {sum(durations) / len(durations):.1f}s")

        if memories:
            print("\nMemory (successful runs):")
            print(f"  Final - Min: {min(memories)}MB, Max: {max(memories)}MB")
        if peak_memories:
            print(f"  Peak  - Min: {min(peak_memories)}MB, Max: {max(peak_memories)}MB")

            # Check for memory leak (growing trend)
            if len(peak_memories) >= 5:
                first_half = peak_memories[: len(peak_memories) // 2]
                second_half = peak_memories[len(peak_memories) // 2 :]
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                growth = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0

                if growth > 20:
                    print(f"\n⚠️  POSSIBLE MEMORY LEAK: Peak memory grew {growth:.1f}% over test run")
                    print(f"    First half avg: {first_avg:.0f}MB, Second half avg: {second_avg:.0f}MB")

    # Show validation warnings
    if with_warnings:
        print("\nValidation warnings:")
        # Collect all unique warnings
        all_warnings: dict[str, int] = {}
        for r in with_warnings:
            if r.warnings:
                for w in r.warnings:
                    all_warnings[w] = all_warnings.get(w, 0) + 1

        # Show top warnings by frequency
        sorted_warnings = sorted(all_warnings.items(), key=lambda x: -x[1])
        for warning, count in sorted_warnings[:10]:
            print(f"  [{count}x] {warning}")
        if len(sorted_warnings) > 10:
            print(f"  ... and {len(sorted_warnings) - 10} more unique warnings")

    if failed:
        print("\nFailed iterations:")
        for r in failed[:10]:  # Show first 10 failures
            print(f"  [{r.iteration}] {r.error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Overall health assessment
    print(f"\n{'=' * 60}")
    issues = []
    if failed:
        issues.append(f"{len(failed)} failures")
    if with_warnings:
        issues.append(f"{len(with_warnings)} with warnings")

    if not issues:
        print("✓ ENGINE HEALTHY - All tests passed with no warnings")
    else:
        print(f"⚠ ISSUES DETECTED: {', '.join(issues)}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Stress test for Polybos Media Engine")
    parser.add_argument("--iterations", "-n", type=int, help="Number of iterations to run")
    parser.add_argument("--duration", "-d", type=int, help="Max duration in seconds")
    parser.add_argument("--video", "-v", type=str, help="Specific video file to use")
    parser.add_argument("--heavy", action="store_true", help="Include heavy extractors (transcript)")
    parser.add_argument("--files", "-f", type=int, default=1, help="Files per batch (default: 1)")
    parser.add_argument("--url", type=str, default=DEFAULT_ENGINE_URL, help="Engine URL")
    parser.add_argument("--sequential", action="store_true", help="Run configs sequentially (not random)")
    parser.add_argument(
        "--thorough",
        "-t",
        action="store_true",
        help="Test every file with every config (comprehensive coverage)",
    )

    args = parser.parse_args()
    engine_url = args.url

    # Check engine is running
    if not check_health(engine_url):
        print(f"Error: Engine not running at {engine_url}")
        print("Start the engine with: uvicorn media_engine.main:app --port 8001")
        sys.exit(1)

    # Find videos
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        videos = [args.video]
    else:
        # Heavy mode: include larger videos (up to 2GB) for realistic stress testing
        max_size = 2000 if args.heavy else 100
        videos = find_test_videos(max_size_mb=max_size)
        if not videos:
            print("Error: No test videos found in test_data/video/")
            print("Add videos or use --video flag")
            sys.exit(1)
        if args.heavy:
            print(f"Heavy mode: including videos up to {max_size}MB ({len(videos)} found)")

    # Default to 10 iterations if neither specified (unless thorough mode)
    iterations = args.iterations
    duration = args.duration
    if iterations is None and duration is None and not args.thorough:
        iterations = 10

    # Run test
    start_time = time.time()
    results = run_stress_test(
        videos=videos,
        engine_url=engine_url,
        iterations=iterations,
        duration_seconds=duration,
        heavy=args.heavy,
        files_per_batch=args.files,
        randomize=not args.sequential,
        thorough=args.thorough,
    )
    total_duration = time.time() - start_time

    # Print summary
    print_summary(results, total_duration)

    # Exit with error if any failures
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

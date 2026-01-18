#!/usr/bin/env python3
"""Analyze camera motion in video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import analyze_motion


def main():
    parser = argparse.ArgumentParser(description="Analyze camera motion in video")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Sample rate for motion analysis (default: 2.0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        start_time = time.perf_counter()
        result = analyze_motion(args.file, sample_fps=args.sample_fps)
        elapsed = time.perf_counter() - start_time

        if args.json:
            data = {
                "duration": result.duration,
                "fps": result.fps,
                "primary_motion": result.primary_motion.value,
                "avg_intensity": float(result.avg_intensity),
                "is_stable": result.is_stable,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "motion_type": s.motion_type.value,
                        "intensity": float(s.intensity),
                    }
                    for s in result.segments
                ],
                "elapsed_seconds": round(elapsed, 2),
            }
            print(json.dumps(data, indent=2))
        else:
            print(f"File: {args.file}")
            print(f"Duration: {result.duration:.2f}s")
            print(f"Primary motion: {result.primary_motion.value}")
            print(f"Avg intensity: {result.avg_intensity:.2f}")
            print(f"Stable: {result.is_stable}")
            print(f"Segments: {len(result.segments)}")
            print()
            for i, seg in enumerate(result.segments[:10], 1):  # Show first 10
                print(
                    f"  {i}: {seg.start:.2f}s-{seg.end:.2f}s "
                    f"{seg.motion_type.value} (intensity: {seg.intensity:.2f})"
                )
            if len(result.segments) > 10:
                print(f"  ... and {len(result.segments) - 10} more")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

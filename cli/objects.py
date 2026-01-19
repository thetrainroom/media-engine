#!/usr/bin/env python3
"""Detect objects in video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import (
    analyze_motion,
    decode_frames,
    extract_objects,
    extract_objects_qwen,
    get_adaptive_timestamps,
)


def main():
    parser = argparse.ArgumentParser(description="Detect objects in video file")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=["yolo", "qwen"],
        help="Object detector to use (default: yolo)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Sample rate for YOLO detection (default: 2.0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
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
        if args.detector == "qwen":
            result = extract_objects_qwen(args.file)
        else:
            # Run motion analysis to get adaptive timestamps
            motion = analyze_motion(args.file)
            timestamps = get_adaptive_timestamps(motion)

            # Decode frames once using shared buffer
            frame_buffer = decode_frames(args.file, timestamps=timestamps)

            # Extract objects using shared frame buffer
            result = extract_objects(
                args.file,
                frame_buffer=frame_buffer,
                min_confidence=args.min_confidence,
            )
        elapsed = time.perf_counter() - start_time

        if args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Detector: {args.detector}")
            print(f"Detections: {len(result.detections)}")
            print()
            print("Summary:")
            for label, count in sorted(
                result.summary.items(), key=lambda x: x[1], reverse=True
            )[:15]:
                print(f"  {label}: {count}")
            if len(result.summary) > 15:
                print(f"  ... and {len(result.summary) - 15} more types")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detect faces in video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import (
    analyze_motion,
    decode_frames,
    extract_faces,
    get_adaptive_timestamps,
)


def main():
    parser = argparse.ArgumentParser(description="Detect faces in video file")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Sample rate for face detection (default: 1.0)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum face size in pixels (default: 80)",
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

        # Run motion analysis to get adaptive timestamps
        motion = analyze_motion(args.file)
        timestamps = get_adaptive_timestamps(motion)

        # Decode frames once using shared buffer
        frame_buffer = decode_frames(args.file, timestamps=timestamps)

        # Extract faces using shared frame buffer
        result = extract_faces(
            args.file,
            frame_buffer=frame_buffer,
            min_face_size=args.min_face_size,
            min_confidence=args.min_confidence,
        )
        elapsed = time.perf_counter() - start_time

        if args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Faces detected: {result.count}")
            print(f"Unique estimate: {result.unique_estimate}")
            print()
            for i, face in enumerate(result.detections[:20], 1):  # Show first 20
                bbox = face.bbox
                print(
                    f"  {i}: t={face.timestamp:.2f}s "
                    f"box=({bbox.x},{bbox.y},{bbox.width}x{bbox.height}) "
                    f"conf={face.confidence:.2f}"
                )
            if result.count > 20:
                print(f"  ... and {result.count - 20} more")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detect objects in video file."""

import argparse
import json
import logging
import sys

from polybos_engine.extractors import extract_objects, extract_objects_qwen


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
        if args.detector == "qwen":
            result = extract_objects_qwen(args.file)
        else:
            result = extract_objects(
                args.file,
                sample_fps=args.sample_fps,
                min_confidence=args.min_confidence,
            )

        if args.json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
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

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

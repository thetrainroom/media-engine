#!/usr/bin/env python3
"""Detect scene boundaries in video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import extract_scenes


def main():
    parser = argparse.ArgumentParser(description="Detect scene boundaries in video")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Content detection threshold (lower=more sensitive, default: 27.0)",
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
        result = extract_scenes(args.file, threshold=args.threshold)
        elapsed = time.perf_counter() - start_time

        if args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Scenes detected: {result.count}")
            print()
            for i, scene in enumerate(result.detections, 1):
                duration = scene.end - scene.start
                print(f"  Scene {i}: {scene.start:.2f}s - {scene.end:.2f}s ({duration:.2f}s)")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

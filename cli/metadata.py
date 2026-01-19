#!/usr/bin/env python3
"""Extract metadata from video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import extract_metadata


def main():
    parser = argparse.ArgumentParser(description="Extract metadata from video file")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON (default: human-readable)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        start_time = time.perf_counter()
        result = extract_metadata(args.file)
        elapsed = time.perf_counter() - start_time

        if args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Duration: {result.duration}s")
            print(f"Resolution: {result.resolution.width}x{result.resolution.height}")
            print(f"FPS: {result.fps}")
            if result.video_codec:
                print(f"Codec: {result.video_codec.name}")
            if result.device:
                print(f"Device: {result.device.make} {result.device.model}")
            if result.gps:
                print(f"GPS: {result.gps.latitude}, {result.gps.longitude}")
            if result.shot_type:
                print(f"Shot type: {result.shot_type.primary} ({result.shot_type.confidence:.2f})")
            if result.keyframes:
                kf = result.keyframes
                interval_type = "fixed GOP" if kf.is_fixed_interval else "irregular (likely cuts)"
                print(f"Keyframes: {kf.count} ({interval_type}, avg {kf.avg_interval}s)")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

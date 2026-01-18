#!/usr/bin/env python3
"""Extract telemetry (GPS/flight data) from video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import extract_telemetry


def main():
    parser = argparse.ArgumentParser(description="Extract telemetry from video file")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--gpx",
        action="store_true",
        help="Output as GPX format instead of JSON",
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
        result = extract_telemetry(args.file)
        elapsed = time.perf_counter() - start_time

        if result is None:
            print("No telemetry data found", file=sys.stderr)
            sys.exit(0)

        if args.gpx:
            print(result.to_gpx())
        elif args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Source: {result.source}")
            print(f"Points: {len(result.points)}")
            print()
            for i, pt in enumerate(result.points[:10], 1):  # Show first 10
                alt = f" alt={pt.altitude:.1f}m" if pt.altitude else ""
                print(f"  {i}: ({pt.latitude:.6f}, {pt.longitude:.6f}){alt}")
            if len(result.points) > 10:
                print(f"  ... and {len(result.points) - 10} more")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

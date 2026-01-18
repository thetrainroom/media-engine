#!/usr/bin/env python3
"""Extract CLIP embeddings from video file."""

import argparse
import json
import logging
import sys
import time

from polybos_engine.extractors import extract_clip


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings from video")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Fallback sample interval in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="CLIP model name (e.g., ViT-B-32, ViT-L-14)",
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
        result = extract_clip(
            args.file,
            fallback_interval=args.interval,
            model_name=args.model,
        )
        elapsed = time.perf_counter() - start_time

        if args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Model: {result.model}")
            print(f"Segments: {len(result.segments)}")
            if result.segments:
                print(f"Embedding dimensions: {len(result.segments[0].embedding)}")
            print()
            for i, seg in enumerate(result.segments[:10], 1):  # Show first 10
                print(f"  {i}: {seg.start:.2f}s-{seg.end:.2f}s embedding[{len(seg.embedding)}]")
            if len(result.segments) > 10:
                print(f"  ... and {len(result.segments) - 10} more")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

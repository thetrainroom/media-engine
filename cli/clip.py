#!/usr/bin/env python3
"""Extract CLIP embeddings from video file."""

import argparse
import json
import logging
import sys
import time

from media_engine.extractors import (
    analyze_motion,
    decode_frames,
    extract_clip,
    get_adaptive_timestamps,
)


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings from video")
    parser.add_argument("file", help="Path to video file")
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

        # Run motion analysis to get adaptive timestamps
        motion = analyze_motion(args.file)
        timestamps = get_adaptive_timestamps(motion)

        # Decode frames once using shared buffer
        frame_buffer = decode_frames(args.file, timestamps=timestamps)

        # Extract CLIP embeddings using shared frame buffer
        result = extract_clip(
            args.file,
            frame_buffer=frame_buffer,
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

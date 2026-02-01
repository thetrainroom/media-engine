#!/usr/bin/env python3
"""Extract text (OCR) from video file."""

import argparse
import json
import logging
import sys
import time

from media_engine.extractors import (
    analyze_motion,
    decode_frames,
    extract_ocr,
    get_adaptive_timestamps,
)


def main():
    parser = argparse.ArgumentParser(description="Extract text (OCR) from video file")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--skip-prefilter",
        action="store_true",
        help="Skip MSER pre-filter (run OCR on all frames)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="OCR languages, comma-separated (e.g., 'en,no,de')",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    languages = None
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",")]

    try:
        start_time = time.perf_counter()

        # Run motion analysis to get adaptive timestamps
        motion = analyze_motion(args.file)
        timestamps = get_adaptive_timestamps(motion)

        # Decode frames once using shared buffer
        frame_buffer = decode_frames(args.file, timestamps=timestamps)

        # Extract OCR using shared frame buffer
        result = extract_ocr(
            args.file,
            frame_buffer=frame_buffer,
            min_confidence=args.min_confidence,
            skip_prefilter=args.skip_prefilter,
            languages=languages,
        )
        elapsed = time.perf_counter() - start_time

        if args.json:
            output = result.model_dump()
            output["elapsed_seconds"] = round(elapsed, 2)
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Text regions detected: {len(result.detections)}")
            print()
            for i, det in enumerate(result.detections[:20], 1):  # Show first 20
                print(f'  {i}: t={det.timestamp:.2f}s "{det.text}" (conf={det.confidence:.2f})')
            if len(result.detections) > 20:
                print(f"  ... and {len(result.detections) - 20} more")
            print()
            print(f"Elapsed: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

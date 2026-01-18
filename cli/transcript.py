#!/usr/bin/env python3
"""Transcribe audio from video file."""

import argparse
import json
import logging
import sys

from polybos_engine.extractors import extract_transcript


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from video file")
    parser.add_argument("file", help="Path to video file")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "tiny", "small", "medium", "large-v3"],
        help="Whisper model size (default: auto)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language code (e.g., 'en', 'no')",
    )
    parser.add_argument(
        "--fallback-language",
        type=str,
        default="en",
        help="Fallback language for short clips (default: en)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        result = extract_transcript(
            args.file,
            model=args.model,
            language=args.language,
            fallback_language=args.fallback_language,
        )

        if args.json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            print(f"File: {args.file}")
            print(f"Language: {result.language} (confidence: {result.confidence:.2f})")
            print(f"Segments: {len(result.segments)}")
            print()
            for seg in result.segments:
                speaker = f"[{seg.speaker}] " if seg.speaker else ""
                print(f"  [{seg.start:.2f}s - {seg.end:.2f}s] {speaker}{seg.text}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""AVCHD structure parsing for spanned recordings.

AVCHD cameras split long recordings at ~2GB boundaries (FAT32 limit).
This module detects which MTS files belong to the same recording by
analyzing timestamps - spanned clips have matching end/start times.

Structure:
    AVCHD/
    └── BDMV/
        ├── CLIPINF/     # Clip info files (.CPI)
        ├── PLAYLIST/    # Playlist files (.MPL)
        ├── STREAM/      # Video files (.MTS)
        └── INDEX.BDM    # Index file
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AVCHDClip:
    """Information about a single AVCHD clip."""

    file_path: str
    clip_number: int
    start_time: float  # PTS start time in seconds
    duration: float
    file_size: int


@dataclass
class AVCHDRecording:
    """A recording that may span multiple clips."""

    clips: list[AVCHDClip]
    total_duration: float

    @property
    def is_spanned(self) -> bool:
        return len(self.clips) > 1

    @property
    def primary_file(self) -> str:
        """The first file of the recording."""
        return self.clips[0].file_path

    @property
    def all_files(self) -> list[str]:
        """All files in this recording."""
        return [c.file_path for c in self.clips]


def _get_clip_timing(file_path: str) -> tuple[float, float] | None:
    """Get start time and duration from MTS file.

    Returns:
        Tuple of (start_time, duration) in seconds, or None if failed.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=start_time,duration",
        "-of",
        "csv=p=0",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            start_time = float(parts[0]) if parts[0] else 0
            duration = float(parts[1]) if parts[1] else 0
            return start_time, duration
    except Exception as e:
        logger.warning(f"Failed to get timing for {file_path}: {e}")

    return None


def parse_avchd_structure(avchd_path: str) -> list[AVCHDRecording]:
    """Parse AVCHD folder structure and identify spanned recordings.

    Args:
        avchd_path: Path to AVCHD folder or any file within it.

    Returns:
        List of recordings, each containing one or more clips.
    """
    path = Path(avchd_path)

    # Find BDMV/STREAM folder
    if path.is_file():
        # Find AVCHD root from file path
        for parent in path.parents:
            stream_dir = parent / "BDMV" / "STREAM"
            if stream_dir.exists():
                break
            # Check if we're in STREAM folder
            if parent.name == "STREAM" and parent.parent.name == "BDMV":
                stream_dir = parent
                break
        else:
            return []
    else:
        stream_dir = path / "BDMV" / "STREAM"
        if not stream_dir.exists():
            return []

    # Get all MTS files sorted by number
    mts_files = sorted(stream_dir.glob("*.MTS"))
    if not mts_files:
        mts_files = sorted(stream_dir.glob("*.mts"))

    if not mts_files:
        return []

    # Get timing for each clip
    clips: list[AVCHDClip] = []
    for mts_path in mts_files:
        timing = _get_clip_timing(str(mts_path))
        if timing is None:
            continue

        start_time, duration = timing
        clip_num = int(mts_path.stem)

        clips.append(
            AVCHDClip(
                file_path=str(mts_path),
                clip_number=clip_num,
                start_time=start_time,
                duration=duration,
                file_size=mts_path.stat().st_size,
            )
        )

    if not clips:
        return []

    # Group clips into recordings based on timestamp continuity
    # Spanned clips have start_time matching previous clip's end time
    recordings: list[AVCHDRecording] = []
    current_group: list[AVCHDClip] = [clips[0]]

    for clip in clips[1:]:
        prev_clip = current_group[-1]
        prev_end_time = prev_clip.start_time + prev_clip.duration

        # Check if this clip continues from previous (within 1 second tolerance)
        if abs(clip.start_time - prev_end_time) < 1.0:
            # This is a continuation (spanned recording)
            current_group.append(clip)
        else:
            # New recording - save current group and start new one
            total_dur = sum(c.duration for c in current_group)
            recordings.append(AVCHDRecording(clips=current_group, total_duration=total_dur))
            current_group = [clip]

    # Don't forget the last group
    if current_group:
        total_dur = sum(c.duration for c in current_group)
        recordings.append(AVCHDRecording(clips=current_group, total_duration=total_dur))

    return recordings


def get_recording_for_file(file_path: str) -> AVCHDRecording | None:
    """Get the recording that contains the given file.

    Args:
        file_path: Path to an MTS file.

    Returns:
        The AVCHDRecording containing this file, or None.
    """
    recordings = parse_avchd_structure(file_path)
    file_path_resolved = str(Path(file_path).resolve())

    for recording in recordings:
        for clip in recording.clips:
            if str(Path(clip.file_path).resolve()) == file_path_resolved:
                return recording

    return None


def is_spanned_continuation(file_path: str) -> bool:
    """Check if this file is a continuation of a spanned recording.

    Returns True if this file is NOT the first file of its recording.
    """
    recording = get_recording_for_file(file_path)
    if recording is None or not recording.is_spanned:
        return False

    # Check if this is the first file
    file_path_resolved = str(Path(file_path).resolve())
    first_file = str(Path(recording.clips[0].file_path).resolve())
    return file_path_resolved != first_file

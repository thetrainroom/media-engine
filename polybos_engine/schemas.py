"""Pydantic schemas for request/response models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MediaDeviceType(StrEnum):
    """Type of media capture device."""

    DRONE = "drone"
    CAMERA = "camera"
    CINEMA_CAMERA = "cinema_camera"
    PHONE = "phone"
    ACTION_CAMERA = "action_camera"
    CAMERA_360 = "360_camera"
    UNKNOWN = "unknown"


class DetectionMethod(StrEnum):
    """Method used for detection."""

    METADATA = "metadata"
    XML_SIDECAR = "xml_sidecar"
    CLIP = "clip"


# === Request Models ===


# === Response Models ===


class Resolution(BaseModel):
    """Video resolution."""

    width: int
    height: int


class VideoCodec(BaseModel):
    """Video codec details."""

    name: str  # h264, hevc, prores, etc.
    profile: str | None = None  # Main 10, High, etc.
    bit_depth: int | None = None  # 8, 10, 12
    pixel_format: str | None = None  # yuv420p, yuv420p10le, etc.


class AudioInfo(BaseModel):
    """Audio stream information."""

    codec: str | None = None  # pcm_s16be, aac, etc.
    sample_rate: int | None = None  # 48000, 44100, etc.
    channels: int | None = None  # 1, 2, 6, etc.
    bit_depth: int | None = None  # 16, 24, 32
    bitrate: int | None = None  # Audio bitrate in bps


class Codec(BaseModel):
    """Video/audio codec info (simplified for backwards compat)."""

    video: str | None = None
    audio: str | None = None


class GPS(BaseModel):
    """GPS coordinates."""

    latitude: float
    longitude: float
    altitude: float | None = None


class ColorSpace(BaseModel):
    """Color space information for LOG/HDR footage."""

    transfer: str | None = (
        None  # Gamma/transfer function (e.g., "slog3", "bt709", "hlg")
    )
    primaries: str | None = None  # Color primaries (e.g., "sgamut3", "bt709", "bt2020")
    matrix: str | None = None  # Color matrix (e.g., "bt709", "bt2020nc")
    lut_file: str | None = None  # Reference to LUT file for conversion
    detection_method: DetectionMethod = DetectionMethod.METADATA


class LensInfo(BaseModel):
    """Lens and camera settings."""

    focal_length: float | None = None  # Focal length in mm
    focal_length_35mm: float | None = None  # 35mm equivalent focal length
    aperture: float | None = None  # f-number (e.g., 2.8)
    focus_distance: float | None = None  # Focus distance in meters
    iris: str | None = None  # Iris setting as string (e.g., "F2.8")
    detection_method: DetectionMethod = DetectionMethod.METADATA


class DeviceInfo(BaseModel):
    """Source device information."""

    make: str | None = None
    model: str | None = None
    software: str | None = None
    type: MediaDeviceType | None = None
    detection_method: DetectionMethod = DetectionMethod.METADATA
    confidence: float = 1.0


class ShotType(BaseModel):
    """Shot type classification."""

    primary: str  # aerial, interview, b-roll, studio, etc.
    confidence: float
    detection_method: str = "clip"


class Metadata(BaseModel):
    """Video metadata."""

    duration: float
    resolution: Resolution
    codec: Codec  # Simplified codec info for backwards compat
    video_codec: VideoCodec | None = None  # Detailed video codec info
    audio: AudioInfo | None = None  # Audio stream info
    fps: float | None = None
    bitrate: int | None = None  # Total bitrate in bps
    file_size: int  # File size in bytes
    timecode: str | None = None  # Start timecode (e.g., "01:15:07:17")
    created_at: datetime | None = None
    device: DeviceInfo | None = None
    gps: GPS | None = None
    color_space: ColorSpace | None = None
    lens: LensInfo | None = None
    shot_type: ShotType | None = None


class TranscriptSegment(BaseModel):
    """Single transcript segment."""

    start: float
    end: float
    text: str
    speaker: str | None = None  # Speaker ID from diarization (e.g., "SPEAKER_00")


class TranscriptHints(BaseModel):
    """Language hints used during transcription."""

    language_hints: list[str] = Field(default_factory=list)
    context_hint: str | None = None
    fallback_applied: bool = False


class Transcript(BaseModel):
    """Full transcript result."""

    language: str
    confidence: float
    duration: float
    speaker_count: int | None = (
        None  # Number of speakers detected (None if diarization disabled)
    )
    hints_used: TranscriptHints
    segments: list[TranscriptSegment]


class BoundingBox(BaseModel):
    """Bounding box for detected objects."""

    x: int
    y: int
    width: int
    height: int


class FaceDetection(BaseModel):
    """Single face detection."""

    timestamp: float
    bbox: BoundingBox
    confidence: float
    embedding: list[float]
    image_base64: str | None = None  # Base64-encoded JPEG of cropped face
    needs_review: bool = False  # Flag for uncertain detections
    review_reason: str | None = None  # Why review is needed


class FacesResult(BaseModel):
    """Face detection results."""

    count: int
    unique_estimate: int
    detections: list[FaceDetection]


class SceneDetection(BaseModel):
    """Single scene segment."""

    index: int
    start: float
    end: float
    duration: float


class ScenesResult(BaseModel):
    """Scene detection results."""

    count: int
    detections: list[SceneDetection]


class ObjectDetection(BaseModel):
    """Single object detection."""

    timestamp: float
    label: str
    confidence: float
    bbox: BoundingBox


class ObjectsResult(BaseModel):
    """Object detection results."""

    summary: dict[str, int]
    detections: list[ObjectDetection]
    descriptions: list[str] | None = None  # Scene descriptions from VLM
    error: str | None = None  # Error code if extraction failed (e.g., "out_of_memory")


class ClipSegment(BaseModel):
    """CLIP embedding for a segment."""

    start: float
    end: float
    scene_index: int | None = None
    embedding: list[float]


class ClipResult(BaseModel):
    """CLIP embedding results."""

    model: str
    segments: list[ClipSegment]


class OcrDetection(BaseModel):
    """Single OCR detection."""

    timestamp: float
    text: str
    confidence: float
    bbox: BoundingBox


class OcrResult(BaseModel):
    """OCR results."""

    detections: list[OcrDetection]


class MotionSegment(BaseModel):
    """A segment of video with consistent camera motion."""

    start: float
    end: float
    motion_type: str  # static, pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out, handheld
    intensity: float  # Average flow magnitude


class MotionResult(BaseModel):
    """Camera motion analysis results."""

    duration: float
    fps: float
    primary_motion: str  # Most common motion type
    segments: list[MotionSegment]
    avg_intensity: float
    is_stable: bool  # True if mostly static/tripod


class TelemetryPoint(BaseModel):
    """Single telemetry point from drone/camera."""

    timestamp: float  # Seconds from start of video
    recorded_at: datetime | None = None  # Actual datetime from telemetry
    latitude: float
    longitude: float
    altitude: float | None = None  # Absolute altitude in meters
    relative_altitude: float | None = None  # Altitude above takeoff
    # Camera settings
    iso: int | None = None
    shutter: float | None = None  # Shutter speed as fraction (1/100 = 0.01)
    aperture: float | None = None  # f-number
    focal_length: float | None = None
    color_mode: str | None = None  # d_log, d_cinelike, etc.


class TelemetryResult(BaseModel):
    """Telemetry/flight path results."""

    source: str  # "dji_srt", "gopro", etc.
    sample_rate: float  # Points per second
    duration: float  # Total duration in seconds
    points: list[TelemetryPoint]

    def to_gpx(self) -> str:
        """Export telemetry as GPX track."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="Polybos Media Engine">',
            "  <trk>",
            "    <name>Flight Path</name>",
            "    <trkseg>",
        ]
        for pt in self.points:
            ele = f"<ele>{pt.altitude}</ele>" if pt.altitude else ""
            time = ""
            if pt.recorded_at:
                time = f"<time>{pt.recorded_at.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z</time>"
            lines.append(
                f'      <trkpt lat="{pt.latitude}" lon="{pt.longitude}">{ele}{time}</trkpt>'
            )
        lines.extend(["    </trkseg>", "  </trk>", "</gpx>"])
        return "\n".join(lines)


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str
    version: str
    api_version: str


class SettingsResponse(BaseModel):
    """Response from GET /settings endpoint.

    All settings are returned, with sensitive values (hf_token) masked.
    """

    # API settings
    api_version: str
    log_level: str

    # Whisper settings
    whisper_model: str
    fallback_language: str

    # Speaker diarization
    hf_token_set: bool  # True if token is configured (actual value is masked)
    diarization_model: str

    # Processing settings
    face_sample_fps: float
    object_sample_fps: float
    min_face_size: int

    # Object detection
    object_detector: str
    qwen_model: str
    qwen_frames_per_scene: int
    yolo_model: str

    # CLIP
    clip_model: str

    # OCR
    ocr_languages: list[str]

    # Temp directory
    temp_dir: str


class SettingsUpdate(BaseModel):
    """Request body for PUT /settings endpoint.

    All fields are optional - only provided fields are updated.
    """

    # API settings
    log_level: str | None = None

    # Whisper settings
    whisper_model: str | None = None
    fallback_language: str | None = None

    # Speaker diarization
    hf_token: str | None = None  # Set to empty string to clear
    diarization_model: str | None = None

    # Processing settings
    face_sample_fps: float | None = None
    object_sample_fps: float | None = None
    min_face_size: int | None = None

    # Object detection
    object_detector: str | None = None
    qwen_model: str | None = None
    qwen_frames_per_scene: int | None = None
    yolo_model: str | None = None

    # CLIP
    clip_model: str | None = None

    # OCR
    ocr_languages: list[str] | None = None

    # Temp directory
    temp_dir: str | None = None

"""Pydantic schemas for request/response models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MediaDeviceType(StrEnum):
    """Type of media capture device."""

    DRONE = "drone"
    CAMERA = "camera"
    PHONE = "phone"
    ACTION_CAMERA = "action_camera"
    UNKNOWN = "unknown"


class DetectionMethod(StrEnum):
    """Method used for detection."""

    METADATA = "metadata"
    XML_SIDECAR = "xml_sidecar"
    CLIP = "clip"


# === Request Models ===


class ExtractRequest(BaseModel):
    """Request body for /extract endpoint."""

    file: str = Field(..., description="Path to video file")

    # Extractor toggles
    skip_transcript: bool = Field(default=False, description="Skip transcription")
    skip_faces: bool = Field(default=False, description="Skip face detection")
    skip_scenes: bool = Field(default=False, description="Skip scene detection")
    skip_objects: bool = Field(default=False, description="Skip object detection")
    skip_clip: bool = Field(default=False, description="Skip CLIP embeddings")
    skip_ocr: bool = Field(default=False, description="Skip OCR")

    # Whisper options
    whisper_model: str = Field(default="large-v3", description="Whisper model size")
    language: str | None = Field(default=None, description="Force language (skip detection)")
    fallback_language: str = Field(
        default="en", description="Fallback for short clips with low confidence"
    )
    language_hints: list[str] = Field(default_factory=list, description="Language hints")
    context_hint: str | None = Field(
        default=None, description="Context hint for transcription (e.g., dialect info)"
    )

    # Sampling options
    face_sample_fps: float = Field(default=1.0, description="Face detection sample rate")
    object_sample_fps: float = Field(default=2.0, description="Object detection sample rate")


# === Response Models ===


class Resolution(BaseModel):
    """Video resolution."""

    width: int
    height: int


class Codec(BaseModel):
    """Video/audio codec info."""

    video: str | None = None
    audio: str | None = None


class GPS(BaseModel):
    """GPS coordinates."""

    latitude: float
    longitude: float
    altitude: float | None = None


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
    codec: Codec
    fps: float | None = None
    bitrate: int | None = None
    file_size: int
    created_at: datetime | None = None
    device: DeviceInfo | None = None
    gps: GPS | None = None
    shot_type: ShotType | None = None


class TranscriptSegment(BaseModel):
    """Single transcript segment."""

    start: float
    end: float
    text: str


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


class ExtractResponse(BaseModel):
    """Response from /extract endpoint."""

    file: str
    filename: str
    extracted_at: datetime
    extraction_time_seconds: float
    api_version: str
    engine_version: str

    metadata: Metadata
    transcript: Transcript | None = None
    faces: FacesResult | None = None
    scenes: ScenesResult | None = None
    objects: ObjectsResult | None = None
    embeddings: ClipResult | None = None
    ocr: OcrResult | None = None


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str
    version: str
    api_version: str

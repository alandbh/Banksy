"""Presets and default configuration."""

from dataclasses import dataclass
from typing import Literal

PresetName = Literal["fast", "balanced", "strong"]
RedactMode = Literal["blur", "pixelate", "box"]
KeyboardMode = Literal["on", "off", "auto"]
FaceMode = Literal["on", "off"]
TextMode = Literal["on", "off"]


@dataclass
class Preset:
    """Preset configuration for processing."""

    sample_fps: float
    max_frames: int
    blur: int
    sample_frames_per_minute: int = 6  # For Gemini


PRESETS: dict[str, Preset] = {
    "fast": Preset(sample_fps=1.0, max_frames=300, blur=20, sample_frames_per_minute=4),
    "balanced": Preset(sample_fps=2.0, max_frames=600, blur=25, sample_frames_per_minute=6),
    "strong": Preset(sample_fps=4.0, max_frames=1200, blur=35, sample_frames_per_minute=12),
}

DEFAULT_PRESET = "balanced"
DEFAULT_BLUR = 25
DEFAULT_PIXEL_SIZE = 12
DEFAULT_KEYBOARD_BLUR = 35
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FACE_BBOX_EXPAND = 0.15  # +15% margin
KEYBOARD_TIME_WINDOW_SEC = 1.0  # ±1 second around typing events
KEYBOARD_REGION_BOTTOM_PCT = 0.40  # Lower 40% of frame (35-45% range)
INPUT_FIELD_REGION_HEIGHT_PCT = 0.10  # Around 10% of screen height above keyboard
INPUT_FIELD_REGION_WIDTH_PCT = 0.90  # Cover most of the input row width
INPUT_FIELD_REGION_GAP_PCT = 0.02  # Small gap between input row and keyboard
INPUT_FIELD_SCAN_HEIGHT_PCT = 0.20  # OCR hints near bottom for input-field text blur
IOU_THRESHOLD = 0.5
DETECTION_MAX_DIM = 960  # Downscale analysis frames to speed up OCR/face detection
GEMINI_FALLBACK_MIN_CONFIDENCE = 0.55  # Trigger Gemini fallback on low-confidence local detection
GEMINI_EVENT_MIN_CONFIDENCE = 0.45  # Ignore low-confidence full-video Gemini events
GEMINI_EVENT_TIME_MARGIN_SEC = 0.5  # Expand event windows to avoid leaking first/last characters
GEMINI_EVENT_BBOX_EXPAND_PCT = 0.10  # Expand Gemini bbox before redaction

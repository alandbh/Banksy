"""FFmpeg I/O: frame extraction and video re-encoding."""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CropBox = tuple[int, int, int, int]  # x, y, w, h


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg command with live logs in terminal."""
    subprocess.run(cmd, check=True)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    """Raise SystemExit with install instructions if ffmpeg is missing."""
    if not check_ffmpeg():
        print(
            "Error: ffmpeg is required but not found in PATH.\n"
            "Install it with:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg\n"
            "  Or download from https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


def get_video_info(input_path: Path) -> dict:
    """Get video metadata (fps, duration, width, height, has_audio)."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json

    data = json.loads(result.stdout)
    info: dict = {
        "fps": 30.0,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "frame_count": 0,
        "has_audio": False,
    }
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                info["fps"] = num / den if den else 30.0
            else:
                info["fps"] = float(fps_str)
            info["width"] = int(stream.get("width", 0))
            info["height"] = int(stream.get("height", 0))
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True
    fmt = data.get("format", {})
    info["duration"] = float(fmt.get("duration", 0))
    if info["fps"] > 0:
        info["frame_count"] = int(info["duration"] * info["fps"])
    return info


def _crop_box_to_filter(crop_box: Optional[CropBox]) -> Optional[str]:
    if crop_box is None:
        return None
    x, y, w, h = crop_box
    return f"crop={w}:{h}:{x}:{y}"


def _build_extract_vf(sample_fps: float, output_height: Optional[int] = None, crop_box: Optional[CropBox] = None) -> str:
    """Build ffmpeg -vf chain for extraction."""
    filters = []
    crop_filter = _crop_box_to_filter(crop_box)
    if crop_filter:
        filters.append(crop_filter)
    filters.append(f"fps={sample_fps}")
    if output_height is not None:
        filters.append(f"scale=-2:{output_height}")
    return ",".join(filters)


def _parse_cropdetect_output(stderr: str) -> list[CropBox]:
    matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", stderr)
    crops: list[CropBox] = []
    for w_s, h_s, x_s, y_s in matches:
        w = int(w_s)
        h = int(h_s)
        x = int(x_s)
        y = int(y_s)
        if w > 0 and h > 0:
            crops.append((x, y, w, h))
    return crops


def _select_stable_crop(crops: list[CropBox], src_w: int, src_h: int, padding_px: int = 0) -> Optional[CropBox]:
    if not crops or src_w <= 0 or src_h <= 0:
        return None

    xs = sorted(c[0] for c in crops)
    ys = sorted(c[1] for c in crops)
    ws = sorted(c[2] for c in crops)
    hs = sorted(c[3] for c in crops)
    mid = len(crops) // 2
    x = xs[mid]
    y = ys[mid]
    w = ws[mid]
    h = hs[mid]

    if padding_px > 0:
        x = max(0, x - padding_px)
        y = max(0, y - padding_px)
        w = min(src_w - x, w + (padding_px * 2))
        h = min(src_h - y, h + (padding_px * 2))

    if x % 2:
        x = max(0, x - 1)
    if y % 2:
        y = max(0, y - 1)
    if w % 2:
        w = max(2, w - 1)
    if h % 2:
        h = max(2, h - 1)

    if x + w > src_w:
        w = max(2, src_w - x)
        if w % 2:
            w = max(2, w - 1)
    if y + h > src_h:
        h = max(2, src_h - y)
        if h % 2:
            h = max(2, h - 1)

    # Ignore near-full-frame crops.
    if w >= int(src_w * 0.98) and h >= int(src_h * 0.98):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def detect_content_crop(
    input_path: Path,
    *,
    padding_px: int = 0,
    max_frames: int = 180,
) -> Optional[CropBox]:
    """Detect stable crop box for black-border videos using ffmpeg cropdetect."""
    info = get_video_info(input_path)
    src_w = int(info.get("width", 0))
    src_h = int(info.get("height", 0))
    if src_w <= 0 or src_h <= 0:
        return None

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vf",
        "cropdetect=24:16:0",
        "-frames:v",
        str(max_frames),
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr or ""
    crops = _parse_cropdetect_output(stderr)
    return _select_stable_crop(crops, src_w, src_h, padding_px=padding_px)


def extract_frames(
    input_path: Path,
    output_dir: Path,
    sample_fps: float,
    max_frames: Optional[int] = None,
    output_height: Optional[int] = None,
    crop_box: Optional[CropBox] = None,
) -> list[Path]:
    """Extract frames from video at given sample_fps. Returns list of frame paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.png"
    vf = _build_extract_vf(sample_fps, output_height, crop_box=crop_box)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-f",
        "image2",
        str(pattern),
    ]
    _run_ffmpeg(cmd)
    frames = sorted(output_dir.glob("frame_*.png"))
    if max_frames and len(frames) > max_frames:
        for f in frames[max_frames:]:
            f.unlink()
        frames = frames[:max_frames]
    return frames


def extract_all_frames(
    input_path: Path,
    output_dir: Path,
    fps: float,
    output_height: Optional[int] = None,
    crop_box: Optional[CropBox] = None,
) -> list[Path]:
    """Extract all frames at original fps for re-encoding."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.png"
    vf = _build_extract_vf(fps, output_height, crop_box=crop_box)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-f",
        "image2",
        str(pattern),
    ]
    _run_ffmpeg(cmd)
    return sorted(output_dir.glob("frame_*.png"))


def reencode_video(
    frames_dir: Path,
    original_video: Path,
    output_path: Path,
    fps: float,
    keep_audio: bool = True,
    encode_preset: str = "medium",
    crf: int = 18,
) -> None:
    """Re-encode video from frames, optionally with audio from original."""
    pattern = frames_dir / "frame_%06d.png"
    if keep_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(pattern),
            "-i",
            str(original_video),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            encode_preset,
            "-crf",
            str(crf),
            "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(pattern),
            "-c:v",
            "libx264",
            "-preset",
            encode_preset,
            "-crf",
            str(crf),
            str(output_path),
        ]
    _run_ffmpeg(cmd)


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    """Save BGR numpy array as image."""
    cv2.imwrite(str(path), img)

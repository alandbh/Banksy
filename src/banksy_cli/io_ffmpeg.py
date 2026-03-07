"""FFmpeg I/O: frame extraction and video re-encoding."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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


def _build_extract_vf(sample_fps: float, output_height: Optional[int] = None) -> str:
    """Build ffmpeg -vf chain for extraction."""
    filters = [f"fps={sample_fps}"]
    if output_height is not None:
        filters.append(f"scale=-2:{output_height}")
    return ",".join(filters)


def extract_frames(
    input_path: Path,
    output_dir: Path,
    sample_fps: float,
    max_frames: Optional[int] = None,
    output_height: Optional[int] = None,
) -> list[Path]:
    """Extract frames from video at given sample_fps. Returns list of frame paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.png"
    vf = _build_extract_vf(sample_fps, output_height)
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
) -> list[Path]:
    """Extract all frames at original fps for re-encoding."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.png"
    vf = _build_extract_vf(fps, output_height)
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

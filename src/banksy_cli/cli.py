"""CLI entry point for banksy."""

from pathlib import Path
from typing import Optional

import typer

from banksy_cli.config import DEFAULT_BLUR, DEFAULT_GEMINI_MODEL, DEFAULT_KEYBOARD_BLUR, DEFAULT_PRESET, PRESETS
from banksy_cli.pipeline import run_pipeline

app = typer.Typer(
    name="banksy",
    help="Auto-blur sensitive information (faces, PII, keyboard) in videos and images.",
    add_completion=False,
)

ENCODE_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


def _preset_callback(value: str) -> str:
    if value not in PRESETS:
        raise typer.BadParameter(f"Must be one of: {', '.join(PRESETS)}")
    return value


def _mode_callback(value: str) -> str:
    if value not in ("blur", "pixelate", "box"):
        raise typer.BadParameter("Must be 'blur', 'pixelate', or 'box'")
    return value


def _keyboard_callback(value: str) -> str:
    if value not in ("on", "off", "auto"):
        raise typer.BadParameter("Must be 'on', 'off', or 'auto'")
    return value


def _output_height_callback(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value < 2:
        raise typer.BadParameter("Must be >= 2")
    return value


def _output_fps_callback(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value <= 0:
        raise typer.BadParameter("Must be > 0")
    return value


def _crf_callback(value: int) -> int:
    if value < 0 or value > 51:
        raise typer.BadParameter("Must be between 0 and 51")
    return value


def _encode_preset_callback(value: str) -> str:
    if value not in ENCODE_PRESETS:
        raise typer.BadParameter(f"Must be one of: {', '.join(sorted(ENCODE_PRESETS))}")
    return value


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to input image or video",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        path_type=Path,
        help="Output path (default: {stem}.redacted.{ext})",
    ),
    mode: str = typer.Option(
        "blur",
        "--mode",
        callback=_mode_callback,
        help="Redaction mode: blur, pixelate, or box (debug)",
    ),
    blur: int = typer.Option(
        DEFAULT_BLUR,
        "--blur",
        help="Blur strength (Gaussian sigma or kernel size)",
    ),
    pixel_size: int = typer.Option(
        12,
        "--pixel-size",
        help="Pixel size for pixelate mode",
    ),
    preset: str = typer.Option(
        DEFAULT_PRESET,
        "--preset",
        callback=_preset_callback,
        help="Preset: fast, balanced, or strong",
    ),
    sample_fps: Optional[float] = typer.Option(
        None,
        "--sample-fps",
        help="Frames per second to analyze (overrides preset)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        help="Maximum frames to analyze (overrides preset)",
    ),
    output_height: Optional[int] = typer.Option(
        None,
        "--output-height",
        callback=_output_height_callback,
        help="Downscale video before processing (keeps aspect ratio, e.g. 480)",
    ),
    output_fps: Optional[float] = typer.Option(
        None,
        "--output-fps",
        callback=_output_fps_callback,
        help="Cap output FPS to speed up processing (e.g. 12 or 15)",
    ),
    use_gemini: bool = typer.Option(
        False,
        "--use-gemini",
        help="Use Gemini detection (full-video JSON events for videos, fallback for images)",
    ),
    gemini_model: str = typer.Option(
        DEFAULT_GEMINI_MODEL,
        "--gemini-model",
        help="Gemini model name",
    ),
    gemini_debug_json: Optional[Path] = typer.Option(
        None,
        "--gemini-debug-json",
        path_type=Path,
        help="Write Gemini raw/parsed JSON response to this file (video mode)",
    ),
    gemini_api_key: Optional[str] = typer.Option(
        None,
        "--gemini-api-key",
        envvar="GEMINI_API_KEY",
        help="Gemini API key (or set GEMINI_API_KEY)",
    ),
    keyboard: str = typer.Option(
        "auto",
        "--keyboard",
        callback=_keyboard_callback,
        help="Keyboard blur: on, off, or auto",
    ),
    keyboard_blur: int = typer.Option(
        DEFAULT_KEYBOARD_BLUR,
        "--keyboard-blur",
        help="Blur strength for keyboard region",
    ),
    face: str = typer.Option(
        "on",
        "--face",
        help="Blur faces: on or off",
    ),
    text: str = typer.Option(
        "on",
        "--text",
        help="Blur sensitive text: on or off",
    ),
    no_audio: bool = typer.Option(
        False,
        "--no-audio",
        help="Remove audio from video output",
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp",
        help="Keep intermediate frames for debugging",
    ),
    json_report: bool = typer.Option(
        False,
        "--json-report",
        help="Emit JSON report with timestamps and detected regions",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers for frame analysis and redaction",
    ),
    encode_preset: str = typer.Option(
        "medium",
        "--encode-preset",
        callback=_encode_preset_callback,
        help="FFmpeg x264 encode preset (ultrafast..veryslow)",
    ),
    crf: int = typer.Option(
        18,
        "--crf",
        callback=_crf_callback,
        help="Output quality/compression factor (0-51, higher=faster/smaller)",
    ),
) -> None:
    """Redact sensitive information from images and videos."""
    preset_config = PRESETS[preset]
    sample_fps_val = sample_fps if sample_fps is not None else preset_config.sample_fps
    max_frames_val = max_frames if max_frames is not None else preset_config.max_frames
    blur_val = blur if blur != DEFAULT_BLUR else preset_config.blur
    if blur == DEFAULT_BLUR and preset != DEFAULT_PRESET:
        blur_val = preset_config.blur

    if use_gemini:
        typer.echo(
            typer.style(
                "Warning: Frames will be sent to Gemini API. "
                "Ensure you are comfortable with this for your content.",
                fg=typer.colors.YELLOW,
            )
        )
        if not gemini_api_key:
            typer.echo(
                typer.style(
                    "Error: GEMINI_API_KEY or --gemini-api-key required when --use-gemini is set.",
                    fg=typer.colors.RED,
                )
            )
            raise typer.Exit(1)

    run_pipeline(
        input_path=input_path,
        output_path=out,
        mode=mode,
        blur=blur_val,
        pixel_size=pixel_size,
        preset=preset,
        sample_fps=sample_fps_val,
        max_frames=max_frames_val,
        use_gemini=use_gemini,
        gemini_model=gemini_model,
        gemini_debug_json=gemini_debug_json,
        gemini_api_key=gemini_api_key,
        keyboard=keyboard,
        keyboard_blur=keyboard_blur,
        face=face == "on",
        text=text == "on",
        keep_audio=not no_audio,
        keep_temp=keep_temp,
        json_report=json_report,
        verbose=verbose,
        workers=workers,
        output_height=output_height,
        output_fps=output_fps,
        encode_preset=encode_preset,
        crf=crf,
    )


if __name__ == "__main__":
    app()

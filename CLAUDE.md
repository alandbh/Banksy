# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**banksy** is a CLI tool that automatically detects and redacts sensitive information (faces, PII, keyboard input) in videos and images. The tool supports multiple redaction modes, detection presets, and optional enhanced detection via Google's Gemini Vision API.

**Key Languages**: Python 3.10+  
**Main Dependencies**: typer (CLI), opencv-python (face detection), easyocr (OCR), google-genai (Gemini API), numpy, pydantic, FFmpeg (external)

## Common Development Commands

### Setup & Installation
```bash
# Install in development mode (creates editable install)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Verify FFmpeg is available
which ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: choco install ffmpeg
```

### Running & Testing
```bash
# Run the CLI directly
python -m banksy_cli --help
# or after install:
banksy input.mp4 --preset balanced

# Run all tests
pytest

# Run a single test file
pytest tests/test_cli_options.py

# Run a specific test
pytest tests/test_cli_options.py::test_output_fps_callback_valid

# Run with coverage
pytest --cov=src/banksy_cli tests/

# Run tests in verbose mode
pytest -v

# Run tests for a specific module (e.g., pipeline performance)
pytest tests/test_pipeline_performance.py -v
```

### Code Quality
```bash
# Note: linting/formatting tools not currently configured in pyproject.toml
# Consider using ruff, black, or similar as needed
```

## Architecture & Key Modules

### High-Level Pipeline Flow

1. **CLI Entry Point** (`src/banksy_cli/cli.py`)
   - Uses typer to parse command-line arguments
   - Validates presets, modes, numeric ranges, and required API keys
   - Delegates to `run_pipeline()`

2. **Main Pipeline** (`src/banksy_cli/pipeline.py`)
   - Orchestrates the entire redaction workflow
   - Handles both image and video inputs
   - For **images**: Extract → Detect → Redact → Save
   - For **videos**: Extract frames → Detect per frame → Track regions → Redact → Re-encode
   - Supports optional Gemini API integration for full-video analysis
   - Applies heuristics for keyboard/input field regions
   - Generates optional JSON reports with region timestamps

3. **Core Detection Modules**
   - `detect_faces.py`: OpenCV Haar cascade-based face detection
   - `detect_text.py`: easyocr-based OCR + regex pattern matching for sensitive data (CPF, credit cards, phone numbers, bank account info)
   - `patterns.py`: Regex patterns and classification logic for sensitive text labels

4. **Gemini Integration** (Optional)
   - `gemini_client.py`: Image-level Gemini API calls (fallback for uncertain detections)
   - `gemini_video_client.py`: Full-video Gemini upload & analysis, returns structured JSON events with timestamps and bounding boxes

5. **Video/Image I/O** (`io_ffmpeg.py`)
   - FFmpeg wrapper for frame extraction, video metadata, re-encoding
   - Supports optional content crop detection (removes black borders)
   - Handles audio passthrough/strip, quality/FPS settings

6. **Redaction** (`redact.py`)
   - Applies blur, pixelate, or box (debug) redaction to detected regions

7. **Supporting Modules**
   - `tracking.py`: Temporal tracking and interpolation of regions across frames (IOU-based)
   - `report.py`: JSON report generation with timestamps and redacted regions
   - `config.py`: Presets (fast/balanced/strong) and configuration constants
   - `config.py` defines **Presets**, each with `sample_fps`, `max_frames`, `blur`, and `sample_frames_per_minute` for Gemini

### Key Design Patterns

- **Presets**: Trade-off speed vs. coverage. `fast` (1 FPS), `balanced` (2 FPS), `strong` (4 FPS). Frame rate can be overridden with `--sample-fps`.
- **Detection Fallback**: Local detection first (fast), optional Gemini API (slower, more accurate). If Gemini fails, local detection is used.
- **Heuristics for Context**: Keyboard region (lower 40%), input field region (just above keyboard) added when sensitive text is detected or when `--keyboard on` is set.
- **Multi-worker Processing**: Parallel frame analysis and redaction using ThreadPoolExecutor (configurable via `--workers`).
- **Region Tracking**: Uses Intersection over Union (IOU) to track regions across frames, avoiding flicker and leakage.

## Testing

### Test Organization
- `test_cli_options.py`: CLI argument validation and callbacks
- `test_detect_text.py`: Text detection and pattern recognition
- `test_gemini_client.py`: Gemini API integration (mocked)
- `test_gemini_video_client.py`: Video Gemini workflow
- `test_io_ffmpeg.py`: FFmpeg integration (frame extraction, video info)
- `test_redact.py`: Redaction mode and region application
- `test_regex_luhn.py`: Regex patterns and Luhn validation
- `test_tracking.py`: Region tracking and interpolation
- `test_pipeline_performance.py`: Full pipeline performance benchmarks

### Running Specific Test Categories
```bash
# Test CLI validation
pytest tests/test_cli_options.py -v

# Test detection logic
pytest tests/test_detect_text.py tests/test_gemini_client.py -v

# Test I/O and encoding
pytest tests/test_io_ffmpeg.py -v

# Test full pipeline (may be slow if using real video)
pytest tests/test_pipeline_performance.py -v
```

## Configuration & Constants

Key constants are in `src/banksy_cli/config.py`:
- **PRESETS**: fast (1 FPS, 300 max frames), balanced (2 FPS, 600), strong (4 FPS, 1200)
- **DETECTION_MAX_DIM**: 960 — frames larger than this are downscaled for faster OCR/face detection
- **IOU_THRESHOLD**: 0.5 — for region tracking across frames
- **Keyboard/Input Field Geometry**: KEYBOARD_REGION_BOTTOM_PCT, INPUT_FIELD_REGION_*
- **Gemini Config**: GEMINI_EVENT_MIN_CONFIDENCE, GEMINI_EVENT_TIME_MARGIN_SEC, etc.

## Important Implementation Notes

### FFmpeg Dependency
- FFmpeg is **external** and required for video processing. The tool does NOT bundle FFmpeg.
- The pipeline checks for FFmpeg availability at startup via `require_ffmpeg()`.

### Gemini API Integration
- **Optional**: Activated only with `--use-gemini` flag.
- **Privacy**: Frames/video files are sent to Google only when explicitly enabled.
- **Models**: Default is `gemini-2.5-flash`. Can be overridden with `--gemini-model`.
- **API Key**: Required via `GEMINI_API_KEY` env var or `--gemini-api-key` flag when `--use-gemini` is set.
- **Video Upload**: For video files, the tool uploads to Gemini Files API, requests full-video analysis, and returns structured JSON events.
- **Debug Output**: Use `--gemini-debug-json` to inspect raw/parsed Gemini responses.

### Region Labeling
- **Face**: From Haar cascade
- **CPF, CC, Phone, Bank**: From regex + Luhn validation on OCR'd text
- **OCR_Text**: General OCR text (only emitted if `include_non_sensitive=True`)
- **Input_Field, Input_Field_Text**: Heuristic regions added when typing sensitive data or `--keyboard on`
- **Keyboard**: Heuristic region added when sensitive text or explicit keyboard blur requested

### Performance Tuning
- Use `--preset fast` for initial testing, increase only if needed.
- `--output-height 480` downscales input, significantly faster.
- `--output-fps 12` or `15` caps FPS, reduces total frames.
- `--encode-preset veryfast --crf 28` for faster encoding (trade quality).
- Increase `--workers` on multi-core machines.
- Disable unneeded detectors: `--face off` and/or `--text off`.

## Debugging

### Verbose Output
```bash
banksy input.mp4 --verbose
```

### Debug Redaction Mode
Use `--mode box` to draw red boxes instead of blurring—helps visualize what would be redacted:
```bash
banksy input.mp4 --mode box -o debug.mp4
```

### Keep Intermediate Frames
```bash
banksy input.mp4 --keep-temp
# Extracted frames and intermediate redacted frames remain in the temp directory
```

### Inspect Gemini Responses
```bash
banksy input.mp4 --use-gemini --gemini-debug-json gemini-debug.json -o out.mp4
# gemini-debug.json contains raw response, parsed JSON, and normalized events
```

## Environment Variables

```bash
# Gemini API key (optional, only when --use-gemini)
export GEMINI_API_KEY="your_api_key_here"
```

See `.env.example` for reference.

## Git Workflow Notes

The repo contains:
- `src/banksy_cli/` — Source code
- `tests/` — Test suite
- `README.md` — User-facing documentation
- `pyproject.toml` — Build and dependency config
- `.env.example` — Example environment variables

Recent commits:
- `feat(video): Add automatic content cropping` — Auto-crop black borders/margins
- `first commit` — Initial commit

## Common Pitfalls & Tips

1. **FFmpeg Not Found**: Ensure FFmpeg is installed and in PATH. Error message will occur at runtime.
2. **Gemini API Key Missing**: If using `--use-gemini`, must provide `GEMINI_API_KEY` or `--gemini-api-key`.
3. **Large Videos with Strong Preset**: Can be slow. Start with `--preset balanced` or add `--output-fps 12` to reduce frame count.
4. **OCR Language**: Currently hardcoded to `["en", "pt"]` (English, Portuguese) in `detect_text.py`. Modify `_get_reader()` to add other languages if needed.
5. **Region Flickering in Output**: Indicates tracking issues. Check `--keep-temp` output and adjust tracking thresholds in `tracking.py` if needed.
6. **Sensitive Text Missed**: Use `--use-gemini` for better coverage or increase `--preset` from fast → balanced/strong.

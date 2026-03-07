# banksy

Auto-blur sensitive information (faces, PII, keyboard) in videos and images.

## Install

```bash
pip install .
# or
pipx install .
```

## Requirements

- **FFmpeg** (required for video processing)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: `choco install ffmpeg`
  - Or download from https://ffmpeg.org/download.html

## Usage

```bash
# Basic usage
banksy input.mp4
banksy input.png --out redacted.png

# Presets (fast, balanced, strong)
banksy input.mp4 --preset strong
banksy input.mp4 --preset balanced --use-gemini

# Redaction mode
banksy input.mp4 --mode pixelate --pixel-size 12
banksy input.mp4 --mode blur --blur 35
banksy input.mp4 --mode box  # debug: draw solid red boxes

# Sampling and workers
banksy input.mp4 --sample-fps 2 --workers 4
banksy input.mp4 --preset fast --workers 8
banksy input.mp4 --output-height 480 --preset fast --workers 8
banksy input.mp4 --output-height 480 --output-fps 12 --workers 8
banksy input.mp4 --encode-preset veryfast --crf 28

# Keyboard blur
banksy input.mp4 --keyboard auto --keyboard-blur 35  # blur only when typing looks sensitive
banksy input.mp4 --keyboard on   # always blur keyboard + input field row

# Detection toggles
banksy input.mp4 --face on --text on

# Output options
banksy input.mp4 --no-audio           # strip audio
banksy input.mp4 --output-height 480  # downscale before processing/output
banksy input.mp4 --auto-crop-content  # remove black borders/content margins
banksy input.mp4 --auto-crop-content --crop-padding 8
banksy input.mp4 --keep-temp          # keep intermediate frames
banksy input.mp4 --json-report        # emit demo.redacted.json

# Verbose
banksy input.mp4 --verbose
```

## Output

- **Default:** `{input_stem}.redacted.{ext}` next to the input
- **With `--json-report`:** `{input_stem}.redacted.json` with timestamps and detected regions

## Sensitive Data Detected

- **Faces** (OpenCV Haar cascade)
- **Phone numbers** (Brazilian + international formats)
- **CPF** (Brazilian tax ID)
- **Credit card numbers** (13–19 digits + Luhn validation)
- **Bank account numbers** (heuristics: agência, conta, account, etc.)
- **Keyboard + input field row** (when typing sensitive data, or always with `--keyboard on`)

## Gemini Vision (Optional)

Use `--use-gemini` for enhanced detection:

```bash
banksy input.mp4 --use-gemini --preset balanced
```

Requires `GEMINI_API_KEY` or `--gemini-api-key`.

- **Videos:** uploaded to Gemini Files API, analyzed as a whole, and converted to structured JSON events (`start/end + bbox + label`).
- **Images:** local detection remains default; Gemini is used as a fallback for uncertain cases.
- If Gemini is unavailable (timeout/network/API error), video processing falls back to local detection.
- If Gemini misses sensitive text in low-contrast UIs, a sparse local OCR fallback with contrast enhancement is applied.

To inspect what Gemini returned, write a debug JSON file:

```bash
banksy input.mp4 --use-gemini --gemini-debug-json gemini-debug.json
```

This file includes attempted models, raw response text, parsed JSON, and normalized events used by the pipeline.

## Performance Tips

- Start with `--preset fast` and increase only if needed.
- Downscale with `--output-height 480` for significantly faster runs.
- Cap FPS with `--output-fps 12` or `--output-fps 15` to reduce total frames.
- Use faster encoding: `--encode-preset veryfast --crf 28`.
- Increase `--workers` for multi-core machines.
- For best text coverage in videos, prefer `--use-gemini` with `--keyboard auto`.
- Disable detectors you do not need (`--face off` and/or `--text off`).
- Prefer `--keyboard auto`; `--keyboard on` will blur keyboard/input area for all frames.

## Privacy

- **Local by default:** All detection runs on your machine.
- **Gemini opt-in:** Images/frames/videos are only sent to Google's API when `--use-gemini` is set.

## License

MIT

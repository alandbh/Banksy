"""Gemini Vision API client for enhanced detection (optional)."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from banksy_cli.config import DEFAULT_GEMINI_MODEL

PROMPT = """Analyze this image for sensitive information. Return JSON with:
- faces: list of {bbox: [x,y,w,h], confidence} for each detected face
- sensitive_text: list of {type: "cpf"|"phone"|"bank"|"cc"|"other", text: "...", bbox: [x,y,w,h], confidence} for sensitive text
- keyboard: {present: bool, bbox: [x,y,w,h] or null, confidence} if on-screen keyboard visible
- input_field: {present: bool, bbox: [x,y,w,h] or null, confidence} if input field visible
- typing_sensitive: true if user appears to be typing sensitive data (e.g. near input field with keyboard)

Coordinates should be pixel values. Return only valid JSON."""


def _frame_hash(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def analyze_frame(
    img_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Analyze a single frame with Gemini Vision. Returns structured detection result."""
    cache_key = _frame_hash(img_bytes)
    if cache_dir:
        cache_file = cache_dir / f"gemini_{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _empty_result()

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT, types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower() or "resource_exhausted" in str(e).lower():
            time.sleep(2)
            return analyze_frame(img_bytes, api_key, model, cache_dir)
        return _empty_result()

    text = response.text if response.text else "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _empty_result()

    result = _parse_result(data)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"gemini_{cache_key}.json"
        try:
            cache_file.write_text(json.dumps(result))
        except Exception:
            pass
    return result


def _parse_result(data: dict) -> dict[str, Any]:
    """Parse and normalize Gemini response to our format."""
    faces = []
    for f in data.get("faces", []):
        if isinstance(f, dict):
            bbox = f.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                faces.append({"bbox": bbox[:4], "label": "face", "confidence": float(f.get("confidence", 0.8))})

    sensitive_text = []
    for s in data.get("sensitive_text", []):
        if isinstance(s, dict):
            bbox = s.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                sensitive_text.append({
                    "bbox": bbox[:4],
                    "label": s.get("type", "other"),
                    "text": str(s.get("text", ""))[:50],
                    "confidence": float(s.get("confidence", 0.8)),
                })

    keyboard = data.get("keyboard", {})
    keyboard_bbox = None
    if isinstance(keyboard, dict) and keyboard.get("present") and keyboard.get("bbox"):
        keyboard_bbox = {"bbox": keyboard["bbox"][:4], "label": "keyboard", "confidence": float(keyboard.get("confidence", 0.8))}

    input_field = data.get("input_field", {})
    input_field_bbox = None
    if isinstance(input_field, dict) and input_field.get("present") and input_field.get("bbox"):
        input_field_bbox = {
            "bbox": input_field["bbox"][:4],
            "label": "input_field",
            "confidence": float(input_field.get("confidence", 0.8)),
        }

    return {
        "faces": faces,
        "sensitive_text": sensitive_text,
        "keyboard": keyboard_bbox,
        "input_field": input_field_bbox,
        "typing_sensitive": bool(data.get("typing_sensitive", False)),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "faces": [],
        "sensitive_text": [],
        "keyboard": None,
        "input_field": None,
        "typing_sensitive": False,
    }

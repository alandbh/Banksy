"""Gemini Files API client for full-video structured detection."""

from __future__ import annotations

import json
import mimetypes
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from banksy_cli.config import DEFAULT_GEMINI_MODEL

GEMINI_API_BASE = "https://generativelanguage.googleapis.com"

ALLOWED_LABELS = {
    "face",
    "cpf",
    "phone",
    "cc",
    "bank",
    "keyboard",
    "input_field",
    "input_field_text",
}

VIDEO_DETECTION_PROMPT = """You are a video safety detector.
Analyze the full video and return ONLY valid JSON with this exact shape:
{
  "events": [
    {
      "start_sec": number,
      "end_sec": number,
      "label": "face|cpf|phone|cc|bank|keyboard|input_field|input_field_text",
      "bbox_norm": [x, y, w, h],
      "confidence": number
    }
  ]
}

Rules:
- Use normalized coordinates for bbox_norm, each value between 0 and 1.
- bbox_norm MUST be [x, y, w, h] (xywh), not [x1, y1, x2, y2].
- start_sec/end_sec are in seconds from video start, with decimal precision (at least centiseconds, e.g. 12.34).
- Use absolute seconds from video start (e.g. 55.20 means 55.20 seconds).
- Never encode time as MM.SS shorthand (for example, do not use 0.55 to mean 55 seconds).
- end_sec >= start_sec.
- Include an event whenever sensitive text is visible in an input field. Use label "input_field_text".
- Include keyboard/input_field events when visible.
- If the target moves due to scroll/animation, split into multiple shorter events with updated bbox_norm per segment.
- No markdown, no commentary, JSON only.
"""

T = TypeVar("T")
MODEL_FALLBACKS = [
    DEFAULT_GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _is_retriable_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in (408, 429, 500, 502, 503, 504)


def _is_retriable_exception(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return _is_retriable_http_error(exc)
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    return False


def _build_ssl_context() -> ssl.SSLContext:
    """Build SSL context; prefer certifi bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _with_retries(
    fn: Callable[[], T],
    *,
    description: str,
    attempts: int = 3,
    base_sleep_sec: float = 1.0,
) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not _is_retriable_exception(exc):
                break
            sleep_sec = base_sleep_sec * (2 ** (attempt - 1))
            time.sleep(sleep_sec)
    assert last_exc is not None
    raise RuntimeError(f"{description} failed after {attempts} attempt(s): {last_exc}") from last_exc


def _request_json(
    url: str,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    body: Optional[bytes] = None,
) -> dict[str, Any]:
    ssl_context = _build_ssl_context()

    def _do_request() -> str:
        req = urllib.request.Request(url, data=body, method=method)
        for key, val in (headers or {}).items():
            req.add_header(key, val)
        with urllib.request.urlopen(req, timeout=300, context=ssl_context) as response:
            return response.read().decode("utf-8")

    raw = _with_retries(_do_request, description=f"Gemini API {method} {url}")

    if not raw:
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini API returned non-JSON response: {raw[:200]}") from exc


def _upload_video_file(api_key: str, video_path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    file_size = video_path.stat().st_size

    init_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    init_body = json.dumps({"file": {"displayName": video_path.name}}).encode("utf-8")
    ssl_context = _build_ssl_context()

    def _start_upload() -> str:
        req = urllib.request.Request(
            f"{GEMINI_API_BASE}/upload/v1beta/files?key={api_key}",
            data=init_body,
            method="POST",
            headers=init_headers,
        )
        with urllib.request.urlopen(req, timeout=300, context=ssl_context) as response:
            upload_url_val = response.headers.get("X-Goog-Upload-URL")
            if not upload_url_val:
                raise RuntimeError("Gemini upload init did not return X-Goog-Upload-URL")
            return upload_url_val

    upload_url = _with_retries(_start_upload, description="Gemini upload init")

    data = video_path.read_bytes()

    upload_headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    def _finalize_upload() -> str:
        upload_req = urllib.request.Request(
            upload_url,
            data=data,
            method="POST",
            headers=upload_headers,
        )
        with urllib.request.urlopen(upload_req, timeout=600, context=ssl_context) as response:
            return response.read().decode("utf-8")

    payload = _with_retries(_finalize_upload, description="Gemini upload finalize")

    parsed = json.loads(payload) if payload else {}
    file_obj = parsed.get("file", parsed)
    if not isinstance(file_obj, dict) or not file_obj.get("name"):
        raise RuntimeError("Gemini upload response did not include file metadata")
    return file_obj


def _wait_for_file_active(api_key: str, file_name: str, timeout_sec: int = 300, poll_sec: float = 3.0) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        data = _request_json(f"{GEMINI_API_BASE}/v1beta/{file_name}?key={api_key}")
        file_obj = data.get("file", data)
        state = str(file_obj.get("state", "")).upper()
        if state == "ACTIVE":
            return file_obj
        if state == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {file_name}")
        time.sleep(poll_sec)
    raise RuntimeError(f"Timeout waiting Gemini file ACTIVE: {file_name}")


def _extract_text_from_generate_response(data: dict[str, Any]) -> str:
    # SDK-style convenience
    direct_text = data.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    candidates = data.get("candidates", [])
    for cand in candidates:
        content = cand.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return "{}"


def _normalize_model_name(name: str) -> str:
    model = name.strip()
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    return model


def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text and "not found" in text


def _list_generate_content_models(api_key: str) -> list[str]:
    models: list[str] = []
    page_token: Optional[str] = None

    for _ in range(10):
        url = f"{GEMINI_API_BASE}/v1beta/models?key={api_key}"
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token)}"
        data = _request_json(url)

        for model in data.get("models", []):
            if not isinstance(model, dict):
                continue
            methods = model.get("supportedGenerationMethods", [])
            methods_norm = {str(m).strip().lower() for m in methods}
            if "generatecontent" not in methods_norm:
                continue
            name = _normalize_model_name(str(model.get("name", "")))
            if name:
                models.append(name)

        page_token_val = data.get("nextPageToken")
        if not page_token_val:
            break
        page_token = str(page_token_val)

    # Keep order but de-duplicate.
    seen = set()
    unique = []
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        unique.append(m)
    return unique


def _build_model_candidates(api_key: str, preferred_model: str) -> list[str]:
    preferred = _normalize_model_name(preferred_model)
    candidates = [preferred] if preferred else []
    candidates.extend([m for m in MODEL_FALLBACKS if m != preferred])

    try:
        available = _list_generate_content_models(api_key)
    except Exception:
        available = []

    # Prioritize available Flash models first, then other available models.
    flash_models = [m for m in available if "flash" in m.lower()]
    non_flash_models = [m for m in available if "flash" not in m.lower()]
    candidates.extend(flash_models)
    candidates.extend(non_flash_models)

    seen = set()
    ordered = []
    for model in candidates:
        model_norm = _normalize_model_name(model)
        if not model_norm or model_norm in seen:
            continue
        seen.add(model_norm)
        ordered.append(model_norm)
    return ordered


def _coerce_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []

    out: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).strip().lower()
        if label not in ALLOWED_LABELS:
            continue

        try:
            start_sec = max(0.0, float(item.get("start_sec", 0.0)))
            end_sec = max(start_sec, float(item.get("end_sec", start_sec)))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        bbox = item.get("bbox_norm", item.get("bbox", [0, 0, 0, 0]))
        try:
            if isinstance(bbox, dict):
                x = float(bbox.get("x", 0))
                y = float(bbox.get("y", 0))
                w = float(bbox.get("w", 0))
                h = float(bbox.get("h", 0))
            else:
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                x = float(bbox[0])
                y = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
        except (TypeError, ValueError):
            continue

        out.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
                "bbox_norm": [x, y, w, h],
                "confidence": confidence,
            }
        )

    return out


def _delete_file_quietly(api_key: str, file_name: Optional[str]) -> None:
    if not file_name:
        return
    try:
        _request_json(f"{GEMINI_API_BASE}/v1beta/{file_name}?key={api_key}", method="DELETE")
    except Exception:
        pass


def analyze_video_file(
    video_path: Path,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_sec: int = 300,
    debug_output_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Upload a video to Gemini Files API and return structured detection events."""
    file_obj: dict[str, Any] = {}
    debug_info: dict[str, Any] = {"requested_model": model, "model_attempts": []}
    try:
        file_obj = _upload_video_file(api_key, Path(video_path))
        if str(file_obj.get("state", "")).upper() != "ACTIVE":
            file_obj = _wait_for_file_active(api_key, file_obj["name"], timeout_sec=timeout_sec)

        file_uri = file_obj.get("uri")
        mime_type = file_obj.get("mimeType") or file_obj.get("mime_type") or "video/mp4"
        if not file_uri:
            raise RuntimeError("Gemini file became ACTIVE but did not include URI")

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": VIDEO_DETECTION_PROMPT},
                        {"fileData": {"mimeType": mime_type, "fileUri": file_uri}},
                    ],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0,
            },
        }

        candidates = _build_model_candidates(api_key, model)
        debug_info["model_candidates"] = candidates
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                response = _request_json(
                    f"{GEMINI_API_BASE}/v1beta/models/{candidate}:generateContent?key={api_key}",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(body).encode("utf-8"),
                )
                text = _extract_text_from_generate_response(response)
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = {}
                events = _coerce_events(parsed)
                debug_info["used_model"] = candidate
                debug_info["raw_response_text"] = text
                debug_info["parsed_response"] = parsed
                debug_info["events"] = events
                return events
            except Exception as exc:
                last_exc = exc
                debug_info["model_attempts"].append({"model": candidate, "error": str(exc)})
                if _is_model_not_found_error(exc):
                    continue
                raise

        if last_exc is None:
            raise RuntimeError("No model candidates available for Gemini video analysis")
        raise RuntimeError(
            f"Gemini video analysis failed for all model candidates "
            f"({', '.join(candidates[:6])}{'...' if len(candidates) > 6 else ''}): {last_exc}"
        ) from last_exc
    finally:
        if debug_output_path:
            try:
                debug_output_path.parent.mkdir(parents=True, exist_ok=True)
                debug_output_path.write_text(json.dumps(debug_info, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        _delete_file_quietly(api_key, file_obj.get("name"))

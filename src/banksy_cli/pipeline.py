"""Main pipeline: orchestrate detection and redaction."""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from typing import Callable, Optional

from banksy_cli.config import (
    DETECTION_MAX_DIM,
    DEFAULT_GEMINI_MODEL,
    GEMINI_EVENT_BBOX_EXPAND_PCT,
    GEMINI_EVENT_MIN_CONFIDENCE,
    GEMINI_EVENT_TIME_MARGIN_SEC,
    GEMINI_FALLBACK_MIN_CONFIDENCE,
    INPUT_FIELD_REGION_GAP_PCT,
    INPUT_FIELD_REGION_HEIGHT_PCT,
    INPUT_FIELD_REGION_WIDTH_PCT,
    INPUT_FIELD_SCAN_HEIGHT_PCT,
    KEYBOARD_REGION_BOTTOM_PCT,
)
from banksy_cli.detect_faces import detect_faces
from banksy_cli.detect_text import detect_text
from banksy_cli.gemini_video_client import analyze_video_file
from banksy_cli.io_ffmpeg import (
    check_ffmpeg,
    detect_content_crop,
    extract_all_frames,
    extract_frames,
    get_video_info,
    load_image,
    reencode_video,
    require_ffmpeg,
    save_image,
)
from banksy_cli.redact import redact_regions
from banksy_cli.report import write_report
from banksy_cli.tracking import interpolate_frames, iou

# Media extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
TEMPLATE_TRACK_LABELS = {"input_field", "input_field_text", "cpf", "phone", "cc", "bank"}
TEMPLATE_MATCH_MIN_SCORE = 0.35
TEMPLATE_SEARCH_EXPAND_PCT = 0.80
TEXT_LIKE_LABELS = {"input_field", "input_field_text", "cpf", "phone", "cc", "bank"}
GEMINI_MIN_DURATION_COVERAGE_RATIO = 0.60
GEMINI_MIN_DURATION_COVERAGE_ABS_SEC = 3.0


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _default_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}.redacted{input_path.suffix}"


def _frame_idx_from_sample_position(position: int, fps: float, sample_fps: float) -> int:
    if sample_fps <= 0:
        return position
    return int(position * fps / sample_fps)


def _log_progress(message: str) -> None:
    """Print progress updates in a consistent format."""
    print(f"[banksy] {message}", flush=True)


def _maybe_log_counter(stage: str, done: int, total: int, last_pct: int, step_pct: int = 10) -> int:
    """Emit coarse-grained progress updates without flooding the terminal."""
    if total <= 0:
        return last_pct
    pct = int((done * 100) / total)
    if pct >= min(100, last_pct + step_pct) or done == total:
        _log_progress(f"{stage}: {done}/{total} ({pct}%)")
        return pct
    return last_pct


def _keyboard_region_heuristic(height: int, width: int) -> list[int]:
    """Lower portion of screen as keyboard region (x, y, w, h)."""
    y_start = int(height * (1 - KEYBOARD_REGION_BOTTOM_PCT))
    return [0, y_start, width, int(height * KEYBOARD_REGION_BOTTOM_PCT)]


def _input_field_region_heuristic(height: int, width: int) -> list[int]:
    """Heuristic for input field row just above the keyboard region."""
    keyboard_y_start = int(height * (1 - KEYBOARD_REGION_BOTTOM_PCT))
    field_h = max(1, int(height * INPUT_FIELD_REGION_HEIGHT_PCT))
    gap = max(0, int(height * INPUT_FIELD_REGION_GAP_PCT))
    y_start = max(0, keyboard_y_start - gap - field_h)
    field_w = max(1, int(width * INPUT_FIELD_REGION_WIDTH_PCT))
    x_start = max(0, (width - field_w) // 2)
    return [x_start, y_start, field_w, field_h]


def _typing_sensitive(regions: list[dict]) -> bool:
    return any(r.get("label") in ("cpf", "phone", "cc", "bank", "input_field_text") for r in regions)


def _is_lower_input_text(bbox: list[int], img_h: int) -> bool:
    """Check if OCR text box is in lower region where input fields usually appear."""
    if len(bbox) < 4 or img_h <= 0:
        return False
    center_y = bbox[1] + (bbox[3] / 2)
    y_start = int(img_h * (1 - KEYBOARD_REGION_BOTTOM_PCT - INPUT_FIELD_SCAN_HEIGHT_PCT))
    return center_y >= max(0, y_start)


def _maybe_add_keyboard_region(regions: list[dict], height: int, width: int, keyboard: str) -> list[dict]:
    if keyboard == "off":
        return regions
    if keyboard == "on" or (keyboard == "auto" and _typing_sensitive(regions)):
        if not any(r.get("label") == "keyboard" for r in regions):
            regions.append(
                {
                    "bbox": _keyboard_region_heuristic(height, width),
                    "label": "keyboard",
                    "confidence": 0.8,
                }
            )
        if not any(r.get("label") == "input_field" for r in regions):
            regions.append(
                {
                    "bbox": _input_field_region_heuristic(height, width),
                    "label": "input_field",
                    "confidence": 0.75,
                }
            )
    return regions


def _scale_regions(regions: list[dict], scale_x: float, scale_y: float) -> list[dict]:
    if scale_x == 1.0 and scale_y == 1.0:
        return regions

    scaled = []
    for region in regions:
        bbox = region.get("bbox", [0, 0, 0, 0])
        x = int(bbox[0] * scale_x)
        y = int(bbox[1] * scale_y)
        w = int(bbox[2] * scale_x)
        h = int(bbox[3] * scale_y)
        item = dict(region)
        item["bbox"] = [x, y, w, h]
        scaled.append(item)
    return scaled


def _detect_local_regions(img, *, face: bool, text: bool, keyboard: str) -> list[dict]:
    """Detect regions locally, downscaling large frames for faster OCR/face detection."""
    import cv2

    h, w = img.shape[:2]
    analysis_img = img
    scale_x = 1.0
    scale_y = 1.0

    longest_side = max(h, w)
    if longest_side > DETECTION_MAX_DIM:
        ratio = DETECTION_MAX_DIM / float(longest_side)
        resized_w = max(1, int(w * ratio))
        resized_h = max(1, int(h * ratio))
        analysis_img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        scale_x = w / float(resized_w)
        scale_y = h / float(resized_h)

    regions = []
    if face:
        regions.extend(detect_faces(analysis_img))
    if text:
        text_regions = _detect_text_regions(analysis_img, include_non_sensitive=keyboard != "off", low_contrast_retry=False)
        analysis_h = analysis_img.shape[0]
        for region in text_regions:
            label = region.get("label")
            if label != "ocr_text":
                regions.append(region)
                continue
            if _is_lower_input_text(region.get("bbox", [0, 0, 0, 0]), analysis_h):
                item = dict(region)
                item["label"] = "input_field_text"
                item["confidence"] = max(0.5, float(item.get("confidence", 0.5)))
                regions.append(item)

    regions = _scale_regions(regions, scale_x, scale_y)
    return _maybe_add_keyboard_region(regions, h, w, keyboard)


def _enhance_low_contrast_for_ocr(img):
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced_lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced


def _detect_text_regions(img, *, include_non_sensitive: bool, low_contrast_retry: bool) -> list[dict]:
    regions = detect_text(img, include_non_sensitive=include_non_sensitive)
    if not low_contrast_retry:
        return regions

    has_sensitive = any(str(r.get("label", "")).lower().strip() in TEXT_LIKE_LABELS for r in regions)
    if has_sensitive:
        return regions

    enhanced = _enhance_low_contrast_for_ocr(img)
    enhanced_regions = detect_text(enhanced, include_non_sensitive=include_non_sensitive)
    if not enhanced_regions:
        return regions
    return regions + enhanced_regions


def _events_have_sensitive_text(events: list[dict]) -> bool:
    for event in events:
        label = str(event.get("label", "")).lower().strip()
        if label in TEXT_LIKE_LABELS:
            return True
    return False


def _max_event_end_sec(events: list[dict]) -> float:
    max_end = 0.0
    for event in events:
        try:
            end_sec = float(event.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        if end_sec > max_end:
            max_end = end_sec
    return max_end


def _events_cover_video_duration(events: list[dict], duration_sec: float) -> bool:
    if duration_sec <= 0:
        return True
    max_end = _max_event_end_sec(events)
    if max_end <= 0:
        return False
    coverage_ratio = max_end / duration_sec
    trailing_gap = duration_sec - max_end
    if coverage_ratio < GEMINI_MIN_DURATION_COVERAGE_RATIO and trailing_gap > GEMINI_MIN_DURATION_COVERAGE_ABS_SEC:
        return False
    return True


def _normalize_mmss_event_timestamps_if_needed(events: list[dict], duration_sec: float) -> tuple[list[dict], bool]:
    """Convert MM.SS shorthand timestamps to absolute seconds when strongly indicated."""
    if duration_sec < 60 or not events:
        return events, False

    raw_max_end = _max_event_end_sec(events)
    # If Gemini already produced reasonable absolute seconds, keep as-is.
    if raw_max_end <= 0 or raw_max_end > 10:
        return events, False

    converted: list[dict] = []
    valid_count = 0
    for event in events:
        try:
            start_raw = float(event.get("start_sec", 0.0))
            end_raw = float(event.get("end_sec", start_raw))
        except (TypeError, ValueError):
            converted.append(dict(event))
            continue

        def _mmss_to_seconds(value: float) -> float:
            minutes = int(max(0.0, value))
            seconds_part = int(round((value - minutes) * 100))
            seconds_part = max(0, min(59, seconds_part))
            return float(minutes * 60 + seconds_part)

        start_conv = _mmss_to_seconds(start_raw)
        end_conv = _mmss_to_seconds(max(start_raw, end_raw))
        item = dict(event)
        item["start_sec"] = start_conv
        item["end_sec"] = max(start_conv, end_conv)
        converted.append(item)
        valid_count += 1

    if valid_count < max(1, int(len(events) * 0.8)):
        return events, False

    converted_max_end = _max_event_end_sec(converted)
    # Accept only if converted timeline plausibly matches video length.
    if converted_max_end < duration_sec * 0.60:
        return events, False
    if converted_max_end > duration_sec * 1.30:
        return events, False
    return converted, True


def _sparse_low_contrast_text_fallback(
    frame_paths: list[Path],
    *,
    fps: float,
    keyboard: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict[int, list[dict]]:
    """Run sparse OCR with contrast enhancement as fallback for low-contrast UIs."""
    if not frame_paths:
        return {}
    if fps <= 0:
        fps = 15.0

    sample_step = max(1, int(round(fps / 2.0)))  # about 2 FPS scan
    sampled_detections: dict[int, list[dict]] = {}
    scan_indices = list(range(0, len(frame_paths), sample_step))
    total_scans = len(scan_indices)
    for scan_pos, frame_idx in enumerate(scan_indices, start=1):
        img = load_image(frame_paths[frame_idx])
        regions = _detect_local_regions_with_retry(
            img,
            keyboard=keyboard,
            low_contrast_retry=True,
        )
        filtered = [r for r in regions if str(r.get("label", "")).lower().strip() in TEXT_LIKE_LABELS]
        if filtered:
            sampled_detections[frame_idx] = filtered
        if progress_cb is not None:
            progress_cb(scan_pos, total_scans)
    if not sampled_detections:
        return {}

    # Spread each sparse detection to a local temporal window to avoid
    # over-propagating one hit across long gaps without evidence.
    spread: dict[int, list[dict]] = {}
    window = max(1, sample_step // 2)
    total_frames = len(frame_paths)
    for center_idx, regions in sampled_detections.items():
        left = max(0, center_idx - window)
        right = min(total_frames - 1, center_idx + window)
        for frame_idx in range(left, right + 1):
            spread.setdefault(frame_idx, []).extend([dict(r) for r in regions])
    return spread


def _detect_local_regions_with_retry(img, *, keyboard: str, low_contrast_retry: bool) -> list[dict]:
    """Text-focused local detection that can retry with contrast enhancement."""
    import cv2

    h, w = img.shape[:2]
    analysis_img = img
    scale_x = 1.0
    scale_y = 1.0

    longest_side = max(h, w)
    if longest_side > DETECTION_MAX_DIM:
        ratio = DETECTION_MAX_DIM / float(longest_side)
        resized_w = max(1, int(w * ratio))
        resized_h = max(1, int(h * ratio))
        analysis_img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        scale_x = w / float(resized_w)
        scale_y = h / float(resized_h)

    regions = []
    text_regions = _detect_text_regions(
        analysis_img,
        include_non_sensitive=keyboard != "off",
        low_contrast_retry=low_contrast_retry,
    )
    analysis_h = analysis_img.shape[0]
    for region in text_regions:
        label = region.get("label")
        if label != "ocr_text":
            regions.append(region)
            continue
        if _is_lower_input_text(region.get("bbox", [0, 0, 0, 0]), analysis_h):
            item = dict(region)
            item["label"] = "input_field_text"
            item["confidence"] = max(0.5, float(item.get("confidence", 0.5)))
            regions.append(item)

    regions = _scale_regions(regions, scale_x, scale_y)
    return _maybe_add_keyboard_region(regions, h, w, keyboard)


def _needs_gemini_fallback(local_regions: list[dict], face: bool, text: bool) -> bool:
    """Only use Gemini when local detections look weak or absent."""
    if not (face or text):
        return False

    relevant_labels = {"face", "cpf", "phone", "cc", "bank"}
    relevant = [r for r in local_regions if r.get("label") in relevant_labels]
    if not relevant:
        return True

    return any(float(r.get("confidence", 0.0)) < GEMINI_FALLBACK_MIN_CONFIDENCE for r in relevant)


def _select_evenly_spaced(items: list[tuple[int, bytes]], limit: int) -> list[tuple[int, bytes]]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]

    selected = []
    used = set()
    step = len(items) / float(limit)

    for i in range(limit):
        idx = min(len(items) - 1, int(i * step))
        if idx in used:
            continue
        used.add(idx)
        selected.append(items[idx])

    if len(selected) < limit:
        for idx, item in enumerate(items):
            if idx in used:
                continue
            selected.append(item)
            if len(selected) == limit:
                break

    return selected


def _expand_bbox(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int, expand_pct: float) -> list[int]:
    if w <= 0 or h <= 0:
        return [x, y, w, h]
    margin_x = int(w * expand_pct)
    margin_y = int(h * expand_pct)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame_w, x + w + margin_x)
    y2 = min(frame_h, y + h + margin_y)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _expand_bbox_asymmetric(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_w: int,
    frame_h: int,
    *,
    expand_x_pct: float,
    expand_top_pct: float,
    expand_bottom_pct: float,
) -> list[int]:
    if w <= 0 or h <= 0:
        return [x, y, w, h]
    margin_x = int(w * expand_x_pct)
    margin_top = int(h * expand_top_pct)
    margin_bottom = int(h * expand_bottom_pct)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(frame_w, x + w + margin_x)
    y2 = min(frame_h, y + h + margin_bottom)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _tune_gemini_event_bbox(label: str, bbox: list[int], frame_w: int, frame_h: int) -> list[int]:
    """Apply label-aware bbox adjustments to reduce text leakage."""
    if len(bbox) < 4:
        return [0, 0, 0, 0]

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]

    text_like_labels = {"input_field", "input_field_text", "cpf", "phone", "cc", "bank"}
    if label not in text_like_labels:
        return _expand_bbox(x, y, w, h, frame_w, frame_h, GEMINI_EVENT_BBOX_EXPAND_PCT)

    # Input-field/text bboxes from VLMs are often too short in height.
    min_h = max(1, int(frame_h * 0.07))
    if h < min_h:
        missing = min_h - h
        y = max(0, y - int(missing * 0.20))
        h = min_h

    # In portrait mobile forms, Gemini sometimes places input-field text boxes too high.
    # Nudge these labels downward when they are suspiciously near the top area.
    if frame_h > frame_w and label in {"input_field", "input_field_text", "cpf"}:
        if y < int(frame_h * 0.20):
            y = min(max(0, frame_h - h), y + int(frame_h * 0.16))

    return _expand_bbox_asymmetric(
        x,
        y,
        w,
        h,
        frame_w,
        frame_h,
        expand_x_pct=max(0.03, GEMINI_EVENT_BBOX_EXPAND_PCT),
        expand_top_pct=max(0.12, GEMINI_EVENT_BBOX_EXPAND_PCT),
        expand_bottom_pct=max(0.45, GEMINI_EVENT_BBOX_EXPAND_PCT * 2.5),
    )


def _event_bbox_to_pixels(
    bbox: list[float],
    *,
    out_w: int,
    out_h: int,
    src_w: int,
    src_h: int,
    crop_box: Optional[tuple[int, int, int, int]] = None,
) -> list[int]:
    if len(bbox) < 4:
        return [0, 0, 0, 0]
    x, y, b3, b4 = bbox[0], bbox[1], bbox[2], bbox[3]

    def _project_from_source_pixels(px: float, py: float, pw: float, ph: float) -> list[int]:
        if crop_box is not None:
            crop_x, crop_y, crop_w, crop_h = crop_box
            x1 = max(0.0, px - crop_x)
            y1 = max(0.0, py - crop_y)
            x2 = min(float(crop_w), px + pw - crop_x)
            y2 = min(float(crop_h), py + ph - crop_y)
            if x2 <= x1 or y2 <= y1:
                return [0, 0, 0, 0]
            sx = out_w / float(max(1, crop_w))
            sy = out_h / float(max(1, crop_h))
            return [int(x1 * sx), int(y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy)]

        sx = out_w / float(max(1, src_w))
        sy = out_h / float(max(1, src_h))
        return [int(px * sx), int(py * sy), int(pw * sx), int(ph * sy)]

    # Preferred format from prompt: normalized 0..1.
    if max(abs(x), abs(y), abs(b3), abs(b4)) <= 1.5:
        # Heuristic: if it looks like x1,y1,x2,y2, convert to xywh.
        if b3 > x and b4 > y and (x + b3 > 1.01 or y + b4 > 1.01):
            w = b3 - x
            h = b4 - y
        else:
            w = b3
            h = b4
        src_px = max(0.0, x) * src_w
        src_py = max(0.0, y) * src_h
        src_pw = max(0.0, w) * src_w
        src_ph = max(0.0, h) * src_h
        return _project_from_source_pixels(src_px, src_py, src_pw, src_ph)

    # Fallback: treat as source-space pixels and rescale to output resolution.
    if b3 > x and b4 > y and (x + b3 > src_w * 1.01 or y + b4 > src_h * 1.01):
        w = b3 - x
        h = b4 - y
    else:
        w = b3
        h = b4
    return _project_from_source_pixels(x, y, w, h)


def _build_frame_detections_from_events(
    events: list[dict],
    *,
    total_frames: int,
    fps: float,
    out_w: int,
    out_h: int,
    src_w: int,
    src_h: int,
    crop_box: Optional[tuple[int, int, int, int]] = None,
) -> dict[int, list[dict]]:
    frame_detections: dict[int, list[dict]] = {}
    if total_frames <= 0 or fps <= 0:
        return frame_detections

    for event in events:
        try:
            confidence = float(event.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < GEMINI_EVENT_MIN_CONFIDENCE:
            continue

        label = str(event.get("label", "")).lower().strip()
        bbox = event.get("bbox_norm", event.get("bbox", [0, 0, 0, 0]))
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue

        pixel_bbox = _event_bbox_to_pixels(
            bbox,
            out_w=out_w,
            out_h=out_h,
            src_w=src_w,
            src_h=src_h,
            crop_box=crop_box,
        )
        expanded = _tune_gemini_event_bbox(label, pixel_bbox, out_w, out_h)
        if expanded[2] <= 0 or expanded[3] <= 0:
            continue

        try:
            start_sec = float(event.get("start_sec", 0.0))
            end_sec = float(event.get("end_sec", start_sec))
        except (TypeError, ValueError):
            continue
        start_sec = max(0.0, start_sec - GEMINI_EVENT_TIME_MARGIN_SEC)
        end_sec = max(start_sec, end_sec + GEMINI_EVENT_TIME_MARGIN_SEC)

        start_frame = max(0, int(math.floor(start_sec * fps)))
        end_frame = min(total_frames - 1, int(math.ceil(end_sec * fps)))
        if end_frame < start_frame:
            continue

        region = {
            "bbox": expanded,
            "label": label,
            "confidence": confidence,
        }
        for frame_idx in range(start_frame, end_frame + 1):
            frame_detections.setdefault(frame_idx, []).append(region)

    return frame_detections


def _extract_gray_patch(gray, bbox: list[int]):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_img, w_img = gray.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return gray[y1:y2, x1:x2]


def _template_track_bbox(gray, prev_bbox: list[int], template) -> list[int] | None:
    import cv2

    if template is None or template.size == 0:
        return None

    x, y, w, h = int(prev_bbox[0]), int(prev_bbox[1]), int(prev_bbox[2]), int(prev_bbox[3])
    if w <= 2 or h <= 2:
        return None
    h_img, w_img = gray.shape[:2]
    margin_x = int(w * TEMPLATE_SEARCH_EXPAND_PCT)
    margin_y = int(h * TEMPLATE_SEARCH_EXPAND_PCT)
    sx1 = max(0, x - margin_x)
    sy1 = max(0, y - margin_y)
    sx2 = min(w_img, x + w + margin_x)
    sy2 = min(h_img, y + h + margin_y)
    search = gray[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return None

    th, tw = template.shape[:2]
    sh, sw = search.shape[:2]
    if th > sh or tw > sw:
        return None

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < TEMPLATE_MATCH_MIN_SCORE:
        return None
    nx = sx1 + int(max_loc[0])
    ny = sy1 + int(max_loc[1])
    return [nx, ny, w, h]


def _refine_event_tracks_with_template(
    frame_paths: list[Path],
    frame_detections: dict[int, list[dict]],
) -> dict[int, list[dict]]:
    """Refine Gemini event boxes frame-by-frame to better follow scroll/motion."""
    import cv2

    if not frame_paths or not frame_detections:
        return frame_detections

    refined: dict[int, list[dict]] = {}
    active_tracks: list[dict] = []

    for frame_idx, frame_path in enumerate(frame_paths):
        regions = frame_detections.get(frame_idx)
        if regions is None:
            continue

        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            refined[frame_idx] = [dict(r) for r in regions]
            continue

        used_tracks: set[int] = set()
        out_regions: list[dict] = []
        for region in regions:
            item = dict(region)
            label = str(item.get("label", "")).lower().strip()
            bbox = item.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) < 4:
                out_regions.append(item)
                continue

            bbox_int = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            if label not in TEMPLATE_TRACK_LABELS:
                out_regions.append(item)
                continue

            best_track_idx = -1
            best_iou = 0.15
            for track_idx, track in enumerate(active_tracks):
                if track_idx in used_tracks:
                    continue
                if track.get("label") != label:
                    continue
                if frame_idx - int(track.get("last_frame", frame_idx)) > 2:
                    continue
                score = iou(track.get("bbox", [0, 0, 0, 0]), bbox_int)
                if score > best_iou:
                    best_iou = score
                    best_track_idx = track_idx

            tracked_bbox = bbox_int
            if best_track_idx >= 0:
                track = active_tracks[best_track_idx]
                candidate = _template_track_bbox(gray, track.get("bbox", bbox_int), track.get("template"))
                if candidate is not None:
                    tracked_bbox = candidate
                used_tracks.add(best_track_idx)
                track["bbox"] = tracked_bbox
                track["last_frame"] = frame_idx
                track["template"] = _extract_gray_patch(gray, tracked_bbox)
            else:
                active_tracks.append(
                    {
                        "label": label,
                        "bbox": tracked_bbox,
                        "last_frame": frame_idx,
                        "template": _extract_gray_patch(gray, tracked_bbox),
                    }
                )
                used_tracks.add(len(active_tracks) - 1)

            item["bbox"] = tracked_bbox
            out_regions.append(item)

        # Keep only recently used tracks to avoid stale drift.
        active_tracks = [t for t in active_tracks if frame_idx - int(t.get("last_frame", frame_idx)) <= 3]
        refined[frame_idx] = out_regions

    return refined


def _merge_gemini_results(local_regions: list[dict], gemini_result: dict, face: bool, text: bool, keyboard: str) -> list[dict]:
    """Merge local detections with Gemini results."""
    merged = list(local_regions)
    if not gemini_result:
        return merged

    gemini_faces = gemini_result.get("faces", [])
    if face and gemini_faces:
        for region in gemini_faces:
            merged.append(
                {
                    "bbox": region.get("bbox", [0, 0, 0, 0]),
                    "label": "face",
                    "confidence": region.get("confidence", 0.8),
                }
            )

    gemini_text = gemini_result.get("sensitive_text", [])
    if text and gemini_text:
        for region in gemini_text:
            merged.append(
                {
                    "bbox": region.get("bbox", [0, 0, 0, 0]),
                    "label": region.get("label") or region.get("type", "other"),
                    "text": region.get("text", ""),
                    "confidence": region.get("confidence", 0.8),
                }
            )

    keyboard_region = gemini_result.get("keyboard")
    input_field_region = gemini_result.get("input_field")
    typing = gemini_result.get("typing_sensitive", False)
    if keyboard != "off" and keyboard_region:
        merged.append(
            {
                "bbox": keyboard_region.get("bbox", [0, 0, 0, 0]),
                "label": "keyboard",
                "confidence": keyboard_region.get("confidence", 0.8),
            }
        )
    elif keyboard == "auto" and typing and keyboard_region:
        merged.append(
            {
                "bbox": keyboard_region.get("bbox", [0, 0, 0, 0]),
                "label": "keyboard",
                "confidence": keyboard_region.get("confidence", 0.8),
            }
        )

    if keyboard != "off" and input_field_region:
        merged.append(
            {
                "bbox": input_field_region.get("bbox", [0, 0, 0, 0]),
                "label": "input_field",
                "confidence": input_field_region.get("confidence", 0.8),
            }
        )
    elif keyboard == "auto" and typing and input_field_region:
        merged.append(
            {
                "bbox": input_field_region.get("bbox", [0, 0, 0, 0]),
                "label": "input_field",
                "confidence": input_field_region.get("confidence", 0.8),
            }
        )
    return merged


def _process_image(
    input_path: Path,
    output_path: Path,
    *,
    face: bool,
    text: bool,
    keyboard: str,
    keyboard_blur: int,
    mode: str,
    blur: int,
    pixel_size: int,
    json_report: bool,
    verbose: bool,
    use_gemini: bool = False,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_api_key: Optional[str] = None,
) -> None:
    """Process single image."""
    import cv2

    img = load_image(input_path)
    regions = _detect_local_regions(img, face=face, text=text, keyboard=keyboard)

    if use_gemini and gemini_api_key and _needs_gemini_fallback(regions, face, text):
        from banksy_cli.gemini_client import analyze_frame

        ok, buf = cv2.imencode(".png", img)
        if ok:
            gemini_result = analyze_frame(buf.tobytes(), gemini_api_key, gemini_model)
            regions = _merge_gemini_results(regions, gemini_result, face, text, keyboard)

    redacted = redact_regions(
        img,
        regions,
        mode=mode,
        blur_strength=blur,
        pixel_size=pixel_size,
        keyboard_blur=keyboard_blur,
    )
    save_image(output_path, redacted)

    if json_report:
        report_path = output_path.with_suffix(".json")
        write_report(
            report_path,
            [
                {
                    "timestamp": 0.0,
                    "frame_index": 0,
                    "regions": [
                        {
                            "label": r.get("label"),
                            "bbox": r.get("bbox"),
                            "confidence": r.get("confidence", 0),
                        }
                        for r in regions
                    ],
                }
            ],
        )
        if verbose:
            print(f"Report written to {report_path}")


def _process_video(
    input_path: Path,
    output_path: Path,
    *,
    face: bool,
    text: bool,
    keyboard: str,
    keyboard_blur: int,
    mode: str,
    blur: int,
    pixel_size: int,
    sample_fps: float,
    max_frames: int,
    keep_audio: bool,
    keep_temp: bool,
    json_report: bool,
    verbose: bool,
    use_gemini: bool = False,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_api_key: Optional[str] = None,
    preset: str = "balanced",
    workers: int = 1,
    output_height: Optional[int] = None,
    output_fps: Optional[float] = None,
    encode_preset: str = "medium",
    crf: int = 18,
    gemini_debug_json: Optional[Path] = None,
    auto_crop_content: bool = False,
    crop_padding: int = 0,
) -> None:
    """Process video: extract, detect, interpolate, redact, re-encode."""
    import tempfile

    require_ffmpeg()
    info = get_video_info(input_path)
    source_fps = info["fps"]
    fps = min(source_fps, output_fps) if output_fps is not None else source_fps
    sample_fps_eff = min(sample_fps, fps) if fps > 0 else sample_fps
    crop_box: Optional[tuple[int, int, int, int]] = None
    if auto_crop_content:
        crop_box = detect_content_crop(input_path, padding_px=max(0, int(crop_padding)))
        if crop_box is not None:
            crop_x, crop_y, crop_w, crop_h = crop_box
            _log_progress(f"Auto-crop enabled: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        else:
            _log_progress("Auto-crop enabled but no stable content crop was detected; using full frame.")

    with tempfile.TemporaryDirectory(prefix="banksy_") as tmpdir:
        tmp = Path(tmpdir)
        analyze_dir = tmp / "analyze"
        full_dir = tmp / "full"

        events: list[dict] = []
        frame_detections: dict[int, list[dict]] = {}
        analyzer_workers = max(1, workers)
        use_event_detections = False

        if use_gemini and gemini_api_key:
            _log_progress("1/6 Analyzing full video with Gemini")
            try:
                events = analyze_video_file(
                    input_path,
                    gemini_api_key,
                    gemini_model,
                    debug_output_path=gemini_debug_json,
                )
                _log_progress(f"Gemini returned {len(events)} event(s)")
                if events:
                    use_event_detections = True
                    _log_progress("2/6 Skipping local per-frame analysis (Gemini full-video mode)")
                    _log_progress("3/6 Gemini video analysis completed")
                else:
                    _log_progress("Gemini returned no events. Falling back to local detection.")
                    use_event_detections = False
            except Exception as exc:
                _log_progress(f"Gemini video analysis failed ({exc}). Falling back to local detection.")
                use_event_detections = False
        if not use_event_detections:
            # Extract frames for local analysis
            _log_progress(f"1/6 Extracting analysis frames (sample_fps={sample_fps_eff:.2f}, output_height={output_height or 'source'})")
            analyze_frames = extract_frames(
                input_path,
                analyze_dir,
                sample_fps_eff,
                max_frames,
                output_height=output_height,
                crop_box=crop_box,
            )
            if not analyze_frames:
                raise RuntimeError("No frames extracted from video")
            _log_progress(f"Analysis frames extracted: {len(analyze_frames)}")

            local_analysis_workers = analyzer_workers
            if text and analyzer_workers > 1:
                local_analysis_workers = 1
                _log_progress(
                    "2/6 Local OCR is unstable with multithreading on some environments; "
                    "using 1 worker for analysis."
                )
            else:
                _log_progress(f"2/6 Analyzing sampled frames with {local_analysis_workers} worker(s)")

            def analyze_sample(task: tuple[int, Path]) -> tuple[int, list[dict]]:
                position, frame_path = task
                img = load_image(frame_path)
                regions = _detect_local_regions(img, face=face, text=text, keyboard=keyboard)
                frame_idx = _frame_idx_from_sample_position(position, fps, sample_fps_eff)
                return frame_idx, regions

            analyzed = 0
            analysis_pct = -1
            if local_analysis_workers == 1:
                for task in enumerate(analyze_frames):
                    frame_idx, regions = analyze_sample(task)
                    frame_detections[frame_idx] = regions
                    analyzed += 1
                    analysis_pct = _maybe_log_counter("Analyze", analyzed, len(analyze_frames), analysis_pct)
            else:
                with ThreadPoolExecutor(max_workers=local_analysis_workers) as pool:
                    for frame_idx, regions in pool.map(analyze_sample, enumerate(analyze_frames)):
                        frame_detections[frame_idx] = regions
                        analyzed += 1
                        analysis_pct = _maybe_log_counter("Analyze", analyzed, len(analyze_frames), analysis_pct)

            _log_progress("3/6 Gemini video analysis skipped")

        # Extract ALL frames at original fps for redaction
        _log_progress(f"4/6 Extracting frames for redaction (output_fps={fps:.2f})")
        if crop_box is not None:
            _log_progress("Using cropped frame extraction for redaction timeline")
        all_frame_paths = extract_all_frames(
            input_path,
            full_dir,
            fps,
            output_height=output_height,
            crop_box=crop_box,
        )
        total_extracted = len(all_frame_paths)
        _log_progress(f"Frames to redact: {total_extracted}")

        # Build per-frame detections from Gemini events or interpolate local detections.
        if use_event_detections:
            duration_sec = float(info.get("duration", 0.0))
            events, mmss_converted = _normalize_mmss_event_timestamps_if_needed(events, duration_sec)
            if mmss_converted:
                _log_progress(
                    "Gemini timestamps looked like MM.SS shorthand; "
                    "converted to absolute seconds before timeline checks."
                )
            events_have_sensitive_text = _events_have_sensitive_text(events)
            events_cover_duration = _events_cover_video_duration(events, duration_sec)
            max_event_end = _max_event_end_sec(events)
            if not events_cover_duration:
                _log_progress(
                    "Gemini timeline appears incomplete "
                    f"(max_end={max_event_end:.2f}s, duration={duration_sec:.2f}s). "
                    "Enabling sparse OCR fallback."
                )
            if all_frame_paths:
                first_img = load_image(all_frame_paths[0])
                out_h, out_w = first_img.shape[:2]
                frame_detections = _build_frame_detections_from_events(
                    events,
                    total_frames=total_extracted,
                    fps=fps,
                    out_w=out_w,
                    out_h=out_h,
                    src_w=int(info.get("width", out_w)),
                    src_h=int(info.get("height", out_h)),
                    crop_box=crop_box,
                )
                if frame_detections:
                    _log_progress("4.5/6 Refining Gemini tracks across frames")
                    frame_detections = _refine_event_tracks_with_template(all_frame_paths, frame_detections)
                if text and (not events_have_sensitive_text or not events_cover_duration):
                    _log_progress(
                        "4.6/6 Running sparse low-contrast OCR fallback "
                        "(Gemini missed text or covered too little timeline)"
                    )
                    fallback_last_pct = -1

                    def _fallback_progress(done: int, total: int) -> None:
                        nonlocal fallback_last_pct
                        fallback_last_pct = _maybe_log_counter("Fallback OCR scan", done, total, fallback_last_pct, step_pct=20)

                    fallback = _sparse_low_contrast_text_fallback(
                        all_frame_paths,
                        fps=fps,
                        keyboard=keyboard,
                        progress_cb=_fallback_progress,
                    )
                    if fallback:
                        merged_frames = 0
                        merged_regions = 0
                        for frame_idx, regions in fallback.items():
                            if not regions:
                                continue
                            frame_detections.setdefault(frame_idx, []).extend(regions)
                            merged_frames += 1
                            merged_regions += len(regions)
                        _log_progress(
                            f"Low-contrast OCR fallback merged {merged_regions} region(s) across {merged_frames} frame(s)"
                        )
            interpolated = frame_detections
            sample_indices = sorted(frame_detections.keys())
        else:
            sample_indices = sorted(frame_detections.keys())
            interpolated = interpolate_frames(
                frame_detections,
                max(total_extracted, max(sample_indices) + 1) if sample_indices else total_extracted,
                sample_indices,
            )
        _log_progress("5/6 Redacting frames")

        def regions_for_frame(frame_number: int) -> list[dict]:
            regions = interpolated.get(frame_number)
            if regions is not None:
                return regions
            if use_event_detections:
                return []
            if not sample_indices:
                return []
            nearest = min(sample_indices, key=lambda sample: abs(sample - frame_number))
            return interpolated.get(nearest, [])

        # Apply redaction to each frame
        report_interval = max(1, int(fps))

        def redact_frame(task: tuple[int, Path]) -> dict | None:
            frame_number, frame_path = task
            img = load_image(frame_path)
            regions = regions_for_frame(frame_number)
            redacted = redact_regions(
                img,
                regions,
                mode=mode,
                blur_strength=blur,
                pixel_size=pixel_size,
                keyboard_blur=keyboard_blur,
            )
            save_image(frame_path, redacted)

            if json_report and (frame_number % report_interval == 0):
                return {
                    "timestamp": frame_number / fps if fps > 0 else 0,
                    "frame_index": frame_number,
                    "regions": [
                        {
                            "label": r.get("label"),
                            "bbox": r.get("bbox"),
                            "confidence": r.get("confidence", 0),
                        }
                        for r in regions
                    ],
                }
            return None

        report_frames = []
        redacted_count = 0
        redaction_pct = -1
        if analyzer_workers == 1:
            for task in enumerate(all_frame_paths):
                item = redact_frame(task)
                if item is not None:
                    report_frames.append(item)
                redacted_count += 1
                redaction_pct = _maybe_log_counter("Redact", redacted_count, total_extracted, redaction_pct)
        else:
            with ThreadPoolExecutor(max_workers=analyzer_workers) as pool:
                for item in pool.map(redact_frame, enumerate(all_frame_paths)):
                    if item is not None:
                        report_frames.append(item)
                    redacted_count += 1
                    redaction_pct = _maybe_log_counter("Redact", redacted_count, total_extracted, redaction_pct)

        _log_progress(f"6/6 Re-encoding output video (preset={encode_preset}, crf={crf})")
        reencode_video(full_dir, input_path, output_path, fps, keep_audio, encode_preset=encode_preset, crf=crf)
        _log_progress(f"Video written to {output_path}")

        if json_report:
            report_frames.sort(key=lambda item: item.get("frame_index", 0))
            report_path = output_path.with_suffix(".json")
            write_report(report_path, report_frames)
            if verbose:
                print(f"Report written to {report_path}")

        if keep_temp:
            # User wants to keep temp - copy to cwd or output dir
            import shutil

            keep_dir = output_path.parent / f"{output_path.stem}_temp"
            shutil.copytree(tmp, keep_dir)
            if verbose:
                print(f"Temp frames kept in {keep_dir}")


def run_pipeline(
    input_path: Path,
    output_path: Optional[Path] = None,
    mode: str = "blur",
    blur: int = 25,
    pixel_size: int = 12,
    preset: str = "balanced",
    sample_fps: float = 2.0,
    max_frames: int = 600,
    use_gemini: bool = False,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_api_key: Optional[str] = None,
    keyboard: str = "auto",
    keyboard_blur: int = 35,
    face: bool = True,
    text: bool = True,
    keep_audio: bool = True,
    keep_temp: bool = False,
    json_report: bool = False,
    verbose: bool = False,
    workers: int = 1,
    output_height: Optional[int] = None,
    output_fps: Optional[float] = None,
    encode_preset: str = "medium",
    crf: int = 18,
    gemini_debug_json: Optional[Path] = None,
    auto_crop_content: bool = False,
    crop_padding: int = 0,
) -> None:
    """Run the full redaction pipeline."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out = output_path or _default_output_path(input_path)
    out = Path(out)
    workers = max(1, int(workers))
    if output_height is not None and output_height < 2:
        raise ValueError("output_height must be >= 2")
    if output_fps is not None and output_fps <= 0:
        raise ValueError("output_fps must be > 0")
    if crf < 0 or crf > 51:
        raise ValueError("crf must be between 0 and 51")
    if crop_padding < 0:
        raise ValueError("crop_padding must be >= 0")

    if _is_image(input_path):
        if not check_ffmpeg():
            pass
        _process_image(
            input_path,
            out,
            face=face,
            text=text,
            keyboard=keyboard,
            keyboard_blur=keyboard_blur,
            mode=mode,
            blur=blur,
            pixel_size=pixel_size,
            json_report=json_report,
            verbose=verbose,
            use_gemini=use_gemini,
            gemini_model=gemini_model,
            gemini_api_key=gemini_api_key or None,
        )
    elif _is_video(input_path):
        _process_video(
            input_path,
            out,
            face=face,
            text=text,
            keyboard=keyboard,
            keyboard_blur=keyboard_blur,
            mode=mode,
            blur=blur,
            pixel_size=pixel_size,
            sample_fps=sample_fps,
            max_frames=max_frames,
            keep_audio=keep_audio,
            keep_temp=keep_temp,
            json_report=json_report,
            verbose=verbose,
            use_gemini=use_gemini,
            gemini_model=gemini_model,
            gemini_api_key=gemini_api_key or None,
            preset=preset,
            workers=workers,
            output_height=output_height,
            output_fps=output_fps,
            encode_preset=encode_preset,
            crf=crf,
            gemini_debug_json=gemini_debug_json,
            auto_crop_content=auto_crop_content,
            crop_padding=crop_padding,
        )
    else:
        raise ValueError(
            f"Unsupported format: {input_path.suffix}. "
            f"Supported: images {IMAGE_EXTENSIONS}, videos {VIDEO_EXTENSIONS}"
        )

    if verbose:
        print(f"Output written to {out}")

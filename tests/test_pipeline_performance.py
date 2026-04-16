"""Tests for pipeline performance-oriented helpers."""

from pathlib import Path

import cv2
import numpy as np

from banksy_cli.pipeline import (
    _build_frame_detections_from_events,
    _events_cover_video_duration,
    _events_have_sensitive_text,
    _frame_idx_from_sample_position,
    _max_event_end_sec,
    _is_lower_input_text,
    _maybe_add_keyboard_region,
    _needs_gemini_fallback,
    _normalize_mmss_event_timestamps_if_needed,
    _refine_event_tracks_with_template,
    _typing_sensitive,
    _select_evenly_spaced,
)


def test_frame_idx_from_sample_position():
    """Map sampled frame position to original timeline index."""
    assert _frame_idx_from_sample_position(0, fps=30.0, sample_fps=2.0) == 0
    assert _frame_idx_from_sample_position(3, fps=30.0, sample_fps=2.0) == 45


def test_needs_gemini_fallback_when_no_local_detection():
    """Fallback should run if local detection found nothing."""
    assert _needs_gemini_fallback([], face=True, text=True) is True


def test_needs_gemini_fallback_on_low_confidence():
    """Fallback should run when confidence is below threshold."""
    regions = [{"label": "cc", "confidence": 0.4, "bbox": [0, 0, 10, 10]}]
    assert _needs_gemini_fallback(regions, face=False, text=True) is True


def test_events_have_sensitive_text():
    events = [{"label": "keyboard"}, {"label": "input_field_text"}]
    assert _events_have_sensitive_text(events) is True
    assert _events_have_sensitive_text([{"label": "keyboard"}]) is False


def test_events_cover_video_duration_detects_incomplete_timeline():
    events = [{"end_sec": 0.56, "label": "input_field_text"}]
    assert _max_event_end_sec(events) == 0.56
    assert _events_cover_video_duration(events, duration_sec=72.0) is False
    assert _events_cover_video_duration(events, duration_sec=0.0) is True


def test_normalize_mmss_event_timestamps_if_needed_converts_shorthand():
    events = [
        {"start_sec": 0.09, "end_sec": 0.17, "label": "input_field"},
        {"start_sec": 1.26, "end_sec": 1.39, "label": "cpf"},
        {"start_sec": 3.30, "end_sec": 3.34, "label": "cpf"},
    ]
    normalized, converted = _normalize_mmss_event_timestamps_if_needed(events, duration_sec=214.1)
    assert converted is True
    assert normalized[0]["start_sec"] == 9.0
    assert normalized[0]["end_sec"] == 17.0
    assert normalized[1]["start_sec"] == 86.0
    assert normalized[1]["end_sec"] == 99.0
    assert normalized[2]["start_sec"] == 210.0
    assert normalized[2]["end_sec"] == 214.0


def test_normalize_mmss_event_timestamps_if_needed_keeps_regular_seconds():
    events = [
        {"start_sec": 5.2, "end_sec": 7.8, "label": "cpf"},
        {"start_sec": 21.0, "end_sec": 25.0, "label": "cpf"},
    ]
    normalized, converted = _normalize_mmss_event_timestamps_if_needed(events, duration_sec=214.1)
    assert converted is False
    assert normalized == events


def test_select_evenly_spaced_keeps_budget_and_order():
    """Candidate selection should be bounded and spread over timeline."""
    items = [(i, b"x") for i in range(10)]
    selected = _select_evenly_spaced(items, limit=4)

    assert len(selected) == 4
    assert [idx for idx, _ in selected] == sorted(idx for idx, _ in selected)
    assert selected[0][0] == 0


def test_keyboard_on_adds_keyboard_and_input_field_regions():
    """Keyboard forced mode should blur keyboard and text input row."""
    regions = _maybe_add_keyboard_region([], 1000, 500, keyboard="on")
    labels = {r["label"] for r in regions}
    assert "keyboard" in labels
    assert "input_field" in labels


def test_typing_sensitive_with_input_field_text_hint():
    """Lower OCR hints should trigger keyboard auto mode."""
    regions = [{"label": "input_field_text", "bbox": [10, 700, 80, 30], "confidence": 0.6}]
    assert _typing_sensitive(regions) is True


def test_is_lower_input_text():
    """Input text candidate must be in lower part of frame."""
    assert _is_lower_input_text([10, 700, 80, 30], 1000) is True
    assert _is_lower_input_text([10, 120, 80, 30], 1000) is False


def test_build_frame_detections_from_events_normalized_bbox():
    """Gemini normalized events should map into frame detections."""
    events = [
        {
            "start_sec": 0.0,
            "end_sec": 0.0,
            "label": "cpf",
            "bbox_norm": [0.1, 0.2, 0.3, 0.1],
            "confidence": 0.9,
        }
    ]
    detections = _build_frame_detections_from_events(
        events,
        total_frames=30,
        fps=30.0,
        out_w=1000,
        out_h=500,
        src_w=1000,
        src_h=500,
    )
    assert 0 in detections
    assert detections[0][0]["label"] == "cpf"


def test_build_frame_detections_from_events_expands_text_like_boxes():
    """Text-like Gemini labels should get extra vertical coverage."""
    events = [
        {
            "start_sec": 0.0,
            "end_sec": 0.0,
            "label": "input_field_text",
            "bbox_norm": [0.05, 0.10, 0.90, 0.02],
            "confidence": 0.9,
        }
    ]
    detections = _build_frame_detections_from_events(
        events,
        total_frames=30,
        fps=30.0,
        out_w=1000,
        out_h=500,
        src_w=1000,
        src_h=500,
    )
    assert 0 in detections
    bbox = detections[0][0]["bbox"]
    # Should be substantially taller than the raw 2% input height.
    assert bbox[3] >= 45


def test_build_frame_detections_from_events_shifts_high_text_box_down_in_portrait():
    """Portrait input-field text boxes near top should be nudged downward."""
    events = [
        {
            "start_sec": 0.0,
            "end_sec": 0.0,
            "label": "input_field_text",
            "bbox_norm": [0.046, 0.114, 0.907, 0.041],
            "confidence": 0.9,
        }
    ]
    detections = _build_frame_detections_from_events(
        events,
        total_frames=30,
        fps=30.0,
        out_w=300,
        out_h=600,
        src_w=300,
        src_h=600,
    )
    assert 0 in detections
    x, y, _w, _h = detections[0][0]["bbox"]
    assert x >= 0
    # Without correction y would be close to 68; with correction it should move down.
    assert y >= 120


def test_build_frame_detections_from_events_uses_ceil_for_end_frame():
    """End frame should include the last partial-frame instant."""
    events = [
        {
            "start_sec": 0.0,
            "end_sec": 1.01,
            "label": "cpf",
            "bbox_norm": [0.1, 0.2, 0.3, 0.1],
            "confidence": 0.9,
        }
    ]
    detections = _build_frame_detections_from_events(
        events,
        total_frames=40,
        fps=10.0,
        out_w=1000,
        out_h=500,
        src_w=1000,
        src_h=500,
    )
    # end_sec 1.01 + margin should include frame 16 with ceil at 10 fps.
    assert 16 in detections


def test_refine_event_tracks_with_template_follows_horizontal_motion(tmp_path):
    """Template refinement should move the bbox with the target."""
    frame_paths = []
    for idx in range(3):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        x = 20 + (idx * 12)
        cv2.rectangle(img, (x, 30), (x + 20, 45), (255, 255, 255), thickness=-1)
        frame_path = tmp_path / f"frame_{idx:06d}.png"
        cv2.imwrite(str(frame_path), img)
        frame_paths.append(Path(frame_path))

    frame_detections = {
        0: [{"bbox": [17, 27, 26, 21], "label": "input_field_text", "confidence": 0.9}],
        1: [{"bbox": [17, 27, 26, 21], "label": "input_field_text", "confidence": 0.9}],
        2: [{"bbox": [17, 27, 26, 21], "label": "input_field_text", "confidence": 0.9}],
    }
    refined = _refine_event_tracks_with_template(frame_paths, frame_detections)
    x0 = refined[0][0]["bbox"][0]
    x2 = refined[2][0]["bbox"][0]
    assert x2 > x0

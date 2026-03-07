"""Tests for tracking and interpolation."""

import pytest

from banksy_cli.tracking import iou, interpolate_box, interpolate_frames, match_boxes


def test_iou_same():
    """Same box has IoU 1."""
    b = [10, 10, 20, 20]
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    """Non-overlapping boxes have IoU 0."""
    a = [0, 0, 10, 10]
    b = [20, 20, 10, 10]
    assert iou(a, b) == 0.0


def test_iou_partial():
    """Partial overlap."""
    a = [0, 0, 10, 10]
    b = [5, 5, 10, 10]
    # Overlap: 5x5=25, union: 100+100-25=175
    assert iou(a, b) == pytest.approx(25 / 175)


def test_match_boxes():
    """Match boxes by label and IoU."""
    prev = [{"bbox": [0, 0, 10, 10], "label": "face"}]
    curr = [{"bbox": [2, 2, 10, 10], "label": "face"}]
    matches = match_boxes(prev, curr, iou_threshold=0.3)
    assert (0, 0) in matches


def test_interpolate_box():
    """Linear interpolation."""
    a = [0, 0, 10, 10]
    b = [10, 10, 20, 20]
    mid = interpolate_box(a, b, 0.5)
    assert mid == [5, 5, 15, 15]
    assert interpolate_box(a, b, 0) == a
    assert interpolate_box(a, b, 1) == b


def test_interpolate_frames():
    """Interpolate detections to all frames."""
    # Use overlapping boxes so IoU matching works (need IoU > 0.5)
    frame_detections = {
        0: [{"bbox": [0, 0, 10, 10], "label": "face", "confidence": 0.9}],
        2: [{"bbox": [2, 0, 10, 10], "label": "face", "confidence": 0.9}],
    }
    result = interpolate_frames(frame_detections, total_frames=3, sample_indices=[0, 2])
    assert 0 in result
    assert 1 in result
    assert 2 in result
    # Frame 1: interpolate between [0,0,10,10] and [2,0,10,10], t=0.5 -> x=1
    assert result[1][0]["bbox"][0] == 1

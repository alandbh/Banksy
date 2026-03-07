"""Tests for redaction strength by label."""

import numpy as np

import banksy_cli.redact as redact


def test_redact_regions_uses_stronger_blur_for_sensitive_text(monkeypatch):
    calls: list[int] = []

    def fake_blur(img, x, y, w, h, strength):
        calls.append(strength)
        return img

    monkeypatch.setattr(redact, "apply_blur", fake_blur)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    regions = [
        {"label": "face", "bbox": [0, 0, 10, 10]},
        {"label": "input_field_text", "bbox": [0, 0, 10, 10]},
        {"label": "cpf", "bbox": [0, 0, 10, 10]},
    ]
    redact.redact_regions(img, regions, mode="blur", blur_strength=25, keyboard_blur=35)

    assert calls == [25, 35, 55]


def test_redact_regions_box_mode_paints_red_region():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    regions = [{"label": "cpf", "bbox": [5, 5, 6, 4]}]
    out = redact.redact_regions(img, regions, mode="box")
    # BGR red fill in ROI.
    assert (out[6, 6] == np.array([0, 0, 255])).all()
    # Outside ROI should remain unchanged.
    assert (out[0, 0] == np.array([0, 0, 0])).all()

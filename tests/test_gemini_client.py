"""Tests for Gemini response parsing."""

from banksy_cli.gemini_client import _parse_result


def test_parse_result_includes_input_field_bbox():
    """Input field should be normalized when Gemini marks it as present."""
    parsed = _parse_result(
        {
            "faces": [],
            "sensitive_text": [],
            "keyboard": {"present": True, "bbox": [0, 500, 500, 300], "confidence": 0.9},
            "input_field": {"present": True, "bbox": [20, 430, 460, 60], "confidence": 0.8},
            "typing_sensitive": True,
        }
    )

    assert parsed["keyboard"] is not None
    assert parsed["input_field"] is not None
    assert parsed["input_field"]["label"] == "input_field"
    assert parsed["input_field"]["bbox"] == [20, 430, 460, 60]

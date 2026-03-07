"""Tests for CLI option callbacks."""

import pytest
from typer import BadParameter

from banksy_cli.cli import _crf_callback, _encode_preset_callback, _mode_callback, _output_fps_callback


def test_output_fps_callback_valid():
    """Output FPS callback should accept positive values."""
    assert _output_fps_callback(12.0) == 12.0


def test_output_fps_callback_invalid():
    """Output FPS callback should reject non-positive values."""
    with pytest.raises(BadParameter):
        _output_fps_callback(0)


def test_encode_preset_callback_valid():
    """Known ffmpeg presets should be accepted."""
    assert _encode_preset_callback("veryfast") == "veryfast"


def test_encode_preset_callback_invalid():
    """Unknown ffmpeg presets should be rejected."""
    with pytest.raises(BadParameter):
        _encode_preset_callback("turbo")


def test_crf_callback_bounds():
    """CRF should stay within x264 accepted range."""
    assert _crf_callback(0) == 0
    assert _crf_callback(51) == 51
    with pytest.raises(BadParameter):
        _crf_callback(52)


def test_mode_callback_accepts_box():
    """Debug box mode should be accepted."""
    assert _mode_callback("box") == "box"

"""Tests for ffmpeg extraction filter builder."""

from banksy_cli.io_ffmpeg import _build_extract_vf


def test_build_extract_vf_without_scale():
    """Should include only fps when output height is not set."""
    assert _build_extract_vf(2.0, None) == "fps=2.0"


def test_build_extract_vf_with_scale():
    """Should include scale keeping aspect ratio when output height is set."""
    assert _build_extract_vf(30.0, 480) == "fps=30.0,scale=-2:480"

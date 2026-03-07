"""Tests for ffmpeg extraction filter builder."""

from banksy_cli.io_ffmpeg import _build_extract_vf, _parse_cropdetect_output, _select_stable_crop


def test_build_extract_vf_without_scale():
    """Should include only fps when output height is not set."""
    assert _build_extract_vf(2.0, None) == "fps=2.0"


def test_build_extract_vf_with_scale():
    """Should include scale keeping aspect ratio when output height is set."""
    assert _build_extract_vf(30.0, 480) == "fps=30.0,scale=-2:480"


def test_build_extract_vf_with_crop_and_scale():
    """Crop should be applied before fps/scale filters."""
    assert _build_extract_vf(15.0, 540, crop_box=(10, 20, 300, 500)) == "crop=300:500:10:20,fps=15.0,scale=-2:540"


def test_parse_cropdetect_output_and_select_stable_crop():
    stderr = "foo crop=300:500:10:20 bar\ncrop=302:500:10:20\ncrop=300:500:10:20"
    crops = _parse_cropdetect_output(stderr)
    assert len(crops) == 3
    stable = _select_stable_crop(crops, src_w=1280, src_h=720, padding_px=4)
    assert stable is not None
    x, y, w, h = stable
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0

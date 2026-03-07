"""Tests for Gemini full-video JSON normalization."""

import urllib.error

import pytest

from banksy_cli.gemini_video_client import _build_ssl_context, _coerce_events, _with_retries
from banksy_cli.gemini_video_client import _is_model_not_found_error, _normalize_model_name


def test_coerce_events_filters_and_normalizes():
    data = {
        "events": [
            {
                "start_sec": 1.2,
                "end_sec": 2.4,
                "label": "cpf",
                "bbox_norm": [0.1, 0.2, 0.3, 0.1],
                "confidence": 0.9,
            },
            {
                "start_sec": 0,
                "end_sec": 1,
                "label": "unknown",
                "bbox_norm": [0, 0, 1, 1],
                "confidence": 1,
            },
        ]
    }

    events = _coerce_events(data)
    assert len(events) == 1
    assert events[0]["label"] == "cpf"
    assert events[0]["bbox_norm"] == [0.1, 0.2, 0.3, 0.1]


def test_coerce_events_handles_missing_list():
    assert _coerce_events({"events": "invalid"}) == []


def test_with_retries_recovers_from_transient_error():
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.URLError("timeout")
        return "ok"

    assert _with_retries(flaky, description="test", attempts=3, base_sleep_sec=0) == "ok"


def test_with_retries_fails_after_attempts():
    def always_fail() -> str:
        raise urllib.error.URLError("timeout")

    with pytest.raises(RuntimeError):
        _with_retries(always_fail, description="test", attempts=2, base_sleep_sec=0)


def test_build_ssl_context():
    ctx = _build_ssl_context()
    assert ctx is not None


def test_normalize_model_name():
    assert _normalize_model_name("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert _normalize_model_name("gemini-2.5-flash") == "gemini-2.5-flash"


def test_model_not_found_error_detection():
    assert _is_model_not_found_error(RuntimeError("HTTP Error 404: Not Found")) is True
    assert _is_model_not_found_error(RuntimeError("HTTP Error 500")) is False

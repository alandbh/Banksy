"""Tests for text detection classification."""

import pytest

from banksy_cli.patterns import classify_text


def test_classify_non_sensitive():
    """Non-sensitive text returns None."""
    assert classify_text("hello world") is None
    assert classify_text("123") is None  # Too short
    assert classify_text("") is None


def test_classify_cpf_formats():
    """Various CPF formats."""
    assert classify_text("CPF: 123.456.789-01")[0] == "cpf"
    assert classify_text("12345678901")[0] == "cpf"


def test_classify_phone_formats():
    """Various phone formats."""
    assert classify_text("Tel: (11) 99999-9999")[0] == "phone"
    assert classify_text("+55 11 98765-4321")[0] == "phone"

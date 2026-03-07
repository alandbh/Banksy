"""Tests for regex and Luhn validation in detect_text."""

import pytest

from banksy_cli.patterns import classify_text, luhn_check


def test_luhn_valid():
    """Valid Luhn numbers."""
    assert luhn_check("4532015112830366") is True  # Visa test
    assert luhn_check("5425233430109903") is True  # Mastercard test
    assert luhn_check("4532 0151 1283 0366") is True
    assert luhn_check("4532-0151-1283-0366") is True


def test_luhn_invalid():
    """Invalid Luhn numbers."""
    assert luhn_check("4532015112830367") is False  # Wrong last digit
    assert luhn_check("1234567890123456") is False
    assert luhn_check("0000000000000000") is True  # Edge: sum=0, 0%10=0


def test_classify_cpf():
    """CPF patterns."""
    r = classify_text("123.456.789-01")
    assert r is not None
    assert r[0] == "cpf"
    r = classify_text("12345678901")
    assert r is not None
    assert r[0] == "cpf"


def test_classify_phone():
    """Phone patterns."""
    r = classify_text("(11) 99999-9999")
    assert r is not None
    assert r[0] == "phone"
    r = classify_text("+55 11 99999-9999")
    assert r is not None
    assert r[0] == "phone"


def test_classify_credit_card():
    """Credit card + Luhn."""
    r = classify_text("4532015112830366")
    assert r is not None
    assert r[0] == "cc"
    r = classify_text("1234567890123456")  # Invalid Luhn
    assert r is None or r[0] != "cc"


def test_classify_bank():
    """Bank context."""
    r = classify_text("conta 12345")
    assert r is not None
    assert r[0] == "bank"
    r = classify_text("agência 1234")
    assert r is not None
    assert r[0] == "bank"

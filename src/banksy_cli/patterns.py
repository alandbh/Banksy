"""Regex patterns and Luhn validation for sensitive data (no heavy deps)."""

import re
from typing import Optional

# Regex patterns for sensitive data
PHONE_PATTERNS = [
    re.compile(r"(\+55)?\s*\(?\d{2}\)?\s*\d{4,5}[-\s]?\d{4}"),
    re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{4,5}[-.\s]?\d{4}"),
]

CPF_PATTERNS = [
    re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),
]

CC_PATTERN = re.compile(r"\b\d{13,19}\b")

BANK_KEYWORDS = [
    "agência", "agencia", "conta", "account", "routing", "iban",
    "swift", "banco", "bank", "número", "numero", "number",
]


def luhn_check(digits: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = digits.replace(" ", "").replace("-", "")
    if not digits.isdigit():
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def classify_text(text: str) -> Optional[tuple[str, float]]:
    """Classify text as sensitive type. Returns (label, confidence) or None."""
    norm = text.strip()
    if not norm or len(norm) < 6:
        return None

    # Check credit card first (13-19 digits + Luhn) - avoids CPF false match on 11-digit substring
    for m in CC_PATTERN.finditer(norm):
        if luhn_check(m.group()):
            return ("cc", 0.9)

    for pat in CPF_PATTERNS:
        m = pat.search(norm)
        if m:
            digits = re.sub(r"\D", "", m.group())
            if len(digits) == 11:
                return ("cpf", 0.95)

    for pat in PHONE_PATTERNS:
        if pat.search(norm):
            return ("phone", 0.85)

    lower = norm.lower()
    if any(kw in lower for kw in BANK_KEYWORDS) and re.search(r"\d{4,}", norm):
        return ("bank", 0.7)

    return None

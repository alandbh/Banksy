"""Text detection via OCR + regex + Luhn validation for sensitive data."""

from typing import Optional

import numpy as np

from banksy_cli.patterns import classify_text

# Lazy init
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr

            _reader = easyocr.Reader(["en", "pt"], gpu=False, verbose=False)
        except Exception:
            _reader = False  # Mark as failed
    return _reader if _reader else None


def detect_text(img: np.ndarray, include_non_sensitive: bool = False) -> list[dict]:
    """Detect text in BGR image via OCR.

    Returns list of {bbox, label, text, confidence}.
    - Sensitive text labels: cpf, phone, cc, bank
    - Non-sensitive OCR label (optional): ocr_text
    """
    reader = _get_reader()
    if reader is None:
        return []

    # EasyOCR expects RGB or BGR
    result = reader.readtext(img)

    regions = []
    for (bbox_points, text, conf) in result:
        if not text or conf < 0.3:
            continue
        classified = classify_text(text)
        if classified is None:
            if not include_non_sensitive:
                continue
            label = "ocr_text"
            label_conf = 0.6
        else:
            label, label_conf = classified
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))
        regions.append(
            {
                "bbox": [x, y, w, h],
                "label": label,
                "text": text[:50],  # Truncate for report
                "confidence": float(conf * label_conf),
            }
        )
    return regions

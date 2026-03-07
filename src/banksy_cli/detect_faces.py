"""Face detection using OpenCV Haar cascade (built-in, no model download)."""

import cv2
import numpy as np

from banksy_cli.config import FACE_BBOX_EXPAND

# Lazy init
_face_cascade = None


def _expand_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Expand bbox by FACE_BBOX_EXPAND margin, clamped to image bounds."""
    margin_w = int(w * FACE_BBOX_EXPAND)
    margin_h = int(h * FACE_BBOX_EXPAND)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)
    return x1, y1, x2 - x1, y2 - y1


def detect_faces(img: np.ndarray) -> list[dict]:
    """Detect faces in BGR image. Returns list of {bbox, label, confidence}."""
    global _face_cascade
    h, w = img.shape[:2]
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    regions = []
    for (x, y, bw, bh) in faces:
        x, y, bw, bh = _expand_bbox(x, y, bw, bh, w, h)
        regions.append({"bbox": [x, y, bw, bh], "label": "face", "confidence": 0.9})
    return regions

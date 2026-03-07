"""Apply blur, pixelate, or debug box overlays to regions in images."""

import cv2
import numpy as np


def _clamp_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp roi to image bounds."""
    h_img, w_img = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    return x1, y1, x2 - x1, y2 - y1


def apply_blur(img: np.ndarray, x: int, y: int, w: int, h: int, strength: int) -> np.ndarray:
    """Apply Gaussian blur to region. Modifies img in place, returns it."""
    x, y, w, h = _clamp_roi(img, x, y, w, h)
    if w <= 0 or h <= 0:
        return img
    # Ensure odd kernel
    k = max(3, strength if strength % 2 else strength + 1)
    roi = img[y : y + h, x : x + w]
    blurred = cv2.GaussianBlur(roi, (k, k), strength)
    img[y : y + h, x : x + w] = blurred
    return img


def apply_pixelate(img: np.ndarray, x: int, y: int, w: int, h: int, pixel_size: int) -> np.ndarray:
    """Apply pixelation to region. Modifies img in place, returns it."""
    x, y, w, h = _clamp_roi(img, x, y, w, h)
    if w <= 0 or h <= 0 or pixel_size < 2:
        return img
    roi = img[y : y + h, x : x + w]
    small_w = max(2, w // pixel_size)
    small_h = max(2, h // pixel_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y : y + h, x : x + w] = pixelated
    return img


def apply_box(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """Apply a solid debug box to region. Modifies img in place, returns it."""
    x, y, w, h = _clamp_roi(img, x, y, w, h)
    if w <= 0 or h <= 0:
        return img
    img[y : y + h, x : x + w] = color
    return img


def redact_regions(
    img: np.ndarray,
    regions: list[dict],
    mode: str = "blur",
    blur_strength: int = 25,
    pixel_size: int = 12,
    keyboard_blur: int = 35,
) -> np.ndarray:
    """Apply redaction to all regions. Keyboard/input labels use keyboard_blur."""
    img = img.copy()
    text_sensitive_labels = {"cpf", "phone", "cc", "bank", "input_field_text"}
    for r in regions:
        bbox = r.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        label = r.get("label", "")
        if label in ("keyboard", "input_field", "input_field_text"):
            strength = keyboard_blur
        elif label in text_sensitive_labels:
            # Make sensitive text less legible even with compact boxes.
            strength = max(keyboard_blur, blur_strength, 55)
        else:
            strength = blur_strength
        if mode == "pixelate":
            apply_pixelate(img, x, y, w, h, pixel_size)
        elif mode == "box":
            apply_box(img, x, y, w, h)
        else:
            apply_blur(img, x, y, w, h, strength)
    return img

"""Temporal tracking and interpolation of bounding boxes across frames."""

from typing import Any


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two boxes [x, y, w, h]."""
    x1_a, y1_a = box_a[0], box_a[1]
    x2_a, y2_a = box_a[0] + box_a[2], box_a[1] + box_a[3]
    x1_b, y1_b = box_b[0], box_b[1]
    x2_b, y2_b = box_b[0] + box_b[2], box_b[1] + box_b[3]

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_boxes(
    prev: list[dict],
    curr: list[dict],
    iou_threshold: float = 0.5,
) -> list[tuple[int | None, int | None]]:
    """Match boxes between prev and curr by IoU. Returns list of (prev_idx, curr_idx)."""
    matches: list[tuple[int | None, int | None]] = []
    used_curr = set()

    for pi, p in enumerate(prev):
        best_j = -1
        best_iou_val = iou_threshold
        pb = p.get("bbox", [0, 0, 0, 0])
        for j, c in enumerate(curr):
            if j in used_curr:
                continue
            cb = c.get("bbox", [0, 0, 0, 0])
            if p.get("label") != c.get("label"):
                continue
            iou_val = iou(pb, cb)
            if iou_val > best_iou_val:
                best_iou_val = iou_val
                best_j = j
        if best_j >= 0:
            matches.append((pi, best_j))
            used_curr.add(best_j)
        else:
            matches.append((pi, None))

    for j in range(len(curr)):
        if j not in used_curr:
            matches.append((None, j))

    return matches


def interpolate_box(
    box_a: list[float],
    box_b: list[float],
    t: float,
) -> list[float]:
    """Interpolate between two boxes. t in [0, 1]; 0=a, 1=b."""
    return [
        box_a[0] + t * (box_b[0] - box_a[0]),
        box_a[1] + t * (box_b[1] - box_a[1]),
        box_a[2] + t * (box_b[2] - box_a[2]),
        box_a[3] + t * (box_b[3] - box_a[3]),
    ]


def interpolate_frames(
    frame_detections: dict[int, list[dict]],
    total_frames: int,
    sample_indices: list[int],
) -> dict[int, list[dict]]:
    """Interpolate detections to all frames. frame_detections maps frame_idx -> regions."""
    result: dict[int, list[dict]] = {}
    if not sample_indices:
        return result

    sorted_indices = sorted(sample_indices)
    for fi in range(total_frames):
        if fi in frame_detections:
            result[fi] = [dict(r) for r in frame_detections[fi]]
            continue

        # Find surrounding analyzed frames
        prev_idx = None
        next_idx = None
        for i, si in enumerate(sorted_indices):
            if si <= fi:
                prev_idx = si
            if si >= fi and next_idx is None:
                next_idx = si
                break

        if prev_idx is None:
            prev_idx = sorted_indices[0]
        if next_idx is None:
            next_idx = sorted_indices[-1]

        prev_regions = frame_detections.get(prev_idx, [])
        next_regions = frame_detections.get(next_idx, [])

        if prev_idx == next_idx:
            result[fi] = [dict(r) for r in prev_regions]
            continue

        t = (fi - prev_idx) / (next_idx - prev_idx) if next_idx != prev_idx else 0
        matched = match_boxes(prev_regions, next_regions)

        interpolated = []
        for (pi, ni) in matched:
            if pi is not None and ni is not None:
                pa = prev_regions[pi]
                na = next_regions[ni]
                new_bbox = interpolate_box(
                    pa.get("bbox", [0, 0, 0, 0]),
                    na.get("bbox", [0, 0, 0, 0]),
                    t,
                )
                interpolated.append(
                    {
                        "bbox": [int(v) for v in new_bbox],
                        "label": pa.get("label", "unknown"),
                        "confidence": (pa.get("confidence", 0) + na.get("confidence", 0)) / 2,
                    }
                )
            elif pi is not None:
                r = dict(prev_regions[pi])
                interpolated.append(r)
            elif ni is not None:
                r = dict(next_regions[ni])
                interpolated.append(r)
        result[fi] = interpolated

    return result
